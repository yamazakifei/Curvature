"""Implement the TensorFlow 1.15 node-only actor and centralized critic PPO model."""

from pathlib import Path
from typing import Mapping

import numpy as np

from .features import GLOBAL_STATE_DIM, NODE_FEATURE_DIM, TARGET_TX_RATIO_INDEX


class CTDEPPO:
    """Share one local actor across nodes and use the critic only during training."""

    def __init__(
        self,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        entropy_coefficient: float = 0.01,
        seed: int = 0,
        session_config=None,
    ) -> None:
        import tensorflow as tensorflow

        self.tf = tensorflow.compat.v1
        self.tf.disable_v2_behavior()
        self.graph = self.tf.Graph()
        with self.graph.as_default():
            self.tf.set_random_seed(int(seed))
            self._build_graph(
                float(learning_rate),
                float(clip_ratio),
                float(entropy_coefficient),
            )
            self.saver = self.tf.train.Saver(max_to_keep=5)
            self.init_op = self.tf.global_variables_initializer()
        self.session = self.tf.Session(graph=self.graph, config=session_config)
        self.session.run(self.init_op)

    def _build_graph(self, learning_rate, clip_ratio, entropy_coefficient) -> None:
        tf = self.tf
        self.node_features = tf.placeholder(
            tf.float32, [None, NODE_FEATURE_DIM], name="node_features"
        )
        self.actions = tf.placeholder(tf.float32, [None], name="actions")
        self.old_log_probabilities = tf.placeholder(
            tf.float32, [None], name="old_log_probabilities"
        )
        self.advantages = tf.placeholder(tf.float32, [None], name="advantages")

        with tf.variable_scope("actor"):
            actor_hidden = tf.layers.dense(
                self.node_features, 64, activation=tf.nn.relu, name="node_dense_1"
            )
            actor_hidden = tf.layers.dense(
                actor_hidden, 64, activation=tf.nn.relu, name="node_dense_2"
            )

            # A zero-initialized residual starts every condition at its target budget.
            residual_logit = tf.squeeze(
                tf.layers.dense(
                    actor_hidden,
                    1,
                    kernel_initializer=tf.zeros_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    name="transmission_residual_logit",
                ),
                axis=1,
            )
            target_probability = tf.clip_by_value(
                self.node_features[:, TARGET_TX_RATIO_INDEX], 1e-6, 1.0 - 1e-6
            )
            target_logit = tf.log(target_probability) - tf.log(1.0 - target_probability)
            self.logits = target_logit + residual_logit
            self.probabilities = tf.nn.sigmoid(self.logits, name="transmission_probability")

        epsilon = 1e-7
        probability = tf.clip_by_value(self.probabilities, epsilon, 1.0 - epsilon)
        log_probability = (
            self.actions * tf.log(probability)
            + (1.0 - self.actions) * tf.log(1.0 - probability)
        )
        self.log_probabilities = log_probability
        ratio = tf.exp(log_probability - self.old_log_probabilities)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        surrogate = tf.minimum(ratio * self.advantages, clipped_ratio * self.advantages)
        entropy = -(
            probability * tf.log(probability)
            + (1.0 - probability) * tf.log(1.0 - probability)
        )
        self.mean_entropy = tf.reduce_mean(entropy)
        self.actor_loss = (
            -tf.reduce_mean(surrogate)
            - entropy_coefficient * self.mean_entropy
        )

        self.global_states = tf.placeholder(
            tf.float32, [None, GLOBAL_STATE_DIM], name="global_states"
        )
        self.returns = tf.placeholder(tf.float32, [None], name="returns")
        with tf.variable_scope("critic"):
            critic_hidden = tf.layers.dense(
                self.global_states, 64, activation=tf.nn.relu, name="dense_1"
            )
            critic_hidden = tf.layers.dense(
                critic_hidden, 64, activation=tf.nn.relu, name="dense_2"
            )
            self.values = tf.squeeze(
                tf.layers.dense(critic_hidden, 1, name="value"), axis=1
            )
        self.critic_loss = tf.reduce_mean(tf.square(self.returns - self.values))

        actor_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor"
        )
        critic_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic"
        )
        self.actor_train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            self.actor_loss, var_list=actor_variables
        )
        self.critic_train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            self.critic_loss, var_list=critic_variables
        )

    def predict_probabilities(self, encoded) -> np.ndarray:
        return self.session.run(
            self.probabilities,
            feed_dict={self.node_features: encoded.node_features},
        )

    def act(self, encoded, rng: np.random.Generator):
        probabilities = self.predict_probabilities(encoded)
        actions = (rng.random(probabilities.size) < probabilities).astype(np.float32)
        log_probabilities = (
            actions * np.log(np.maximum(probabilities, 1e-7))
            + (1.0 - actions) * np.log(np.maximum(1.0 - probabilities, 1e-7))
        )
        return actions, probabilities, log_probabilities.astype(np.float32)

    def value(self, global_states: np.ndarray) -> np.ndarray:
        states = np.asarray(global_states, dtype=np.float32).reshape(
            -1, GLOBAL_STATE_DIM
        )
        return self.session.run(
            self.values, feed_dict={self.global_states: states}
        )

    def update(self, actor_batch: Mapping, critic_batch: Mapping, epochs: int = 4):
        actor_feed = {
            self.node_features: actor_batch["node_features"],
            self.actions: actor_batch["actions"],
            self.old_log_probabilities: actor_batch["old_log_probabilities"],
            self.advantages: actor_batch["advantages"],
        }
        critic_feed = {
            self.global_states: critic_batch["global_states"],
            self.returns: critic_batch["returns"],
        }
        actor_loss = critic_loss = entropy = 0.0
        for _ in range(int(epochs)):
            actor_loss, entropy, _ = self.session.run(
                [self.actor_loss, self.mean_entropy, self.actor_train_op],
                feed_dict=actor_feed,
            )
            critic_loss, _ = self.session.run(
                [self.critic_loss, self.critic_train_op], feed_dict=critic_feed
            )
        return {
            "actor_loss": float(actor_loss),
            "critic_loss": float(critic_loss),
            "entropy": float(entropy),
        }

    def save(self, checkpoint_prefix: str) -> str:
        path = Path(checkpoint_prefix)
        path.parent.mkdir(parents=True, exist_ok=True)
        return self.saver.save(self.session, str(path))

    def restore(self, checkpoint_prefix: str) -> None:
        self.saver.restore(self.session, str(checkpoint_prefix))

    def close(self) -> None:
        self.session.close()
