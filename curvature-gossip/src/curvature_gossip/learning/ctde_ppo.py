"""实现 TensorFlow 1.15 兼容的共享 DeepSets actor 与集中 critic PPO。"""

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .features import EDGE_FEATURE_DIM, GLOBAL_STATE_DIM, NODE_FEATURE_DIM


class CTDEPPO:
    """所有节点共享 actor 参数；critic 只在集中训练阶段使用。"""

    def __init__(
        self,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        entropy_coefficient: float = 0.01,
        q_min: float = 0.001,
        q_max: float = 0.8,
        seed: int = 0,
        session_config=None,
    ) -> None:
        import tensorflow as tensorflow

        self.tf = tensorflow.compat.v1
        self.tf.disable_v2_behavior()
        self.q_min = float(q_min)
        self.q_max = float(q_max)
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
        self.edge_features = tf.placeholder(
            tf.float32, [None, None, EDGE_FEATURE_DIM], name="edge_features"
        )
        self.edge_mask = tf.placeholder(tf.float32, [None, None], name="edge_mask")
        self.actions = tf.placeholder(tf.float32, [None], name="actions")
        self.old_log_probabilities = tf.placeholder(
            tf.float32, [None], name="old_log_probabilities"
        )
        self.advantages = tf.placeholder(tf.float32, [None], name="advantages")

        with tf.variable_scope("actor"):
            edge_hidden = tf.layers.dense(
                self.edge_features, 32, activation=tf.nn.relu, name="edge_dense_1"
            )
            edge_hidden = tf.layers.dense(
                edge_hidden, 32, activation=tf.nn.relu, name="edge_dense_2"
            )
            mask = tf.expand_dims(self.edge_mask, axis=-1)
            edge_sum = tf.reduce_sum(edge_hidden * mask, axis=1)
            edge_count = tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)
            edge_mean = edge_sum / edge_count
            masked_hidden = edge_hidden + (1.0 - mask) * -1e9
            edge_max = tf.reduce_max(masked_hidden, axis=1)
            has_edge = tf.cast(tf.reduce_sum(self.edge_mask, axis=1) > 0.0, tf.float32)
            edge_max = edge_max * tf.expand_dims(has_edge, axis=-1)
            actor_input = tf.concat(
                [self.node_features, edge_mean, edge_max], axis=1
            )
            actor_hidden = tf.layers.dense(
                actor_input, 64, activation=tf.nn.relu, name="node_dense_1"
            )
            actor_hidden = tf.layers.dense(
                actor_hidden, 64, activation=tf.nn.relu, name="node_dense_2"
            )
            logits = tf.squeeze(
                tf.layers.dense(actor_hidden, 1, name="transmission_logit"), axis=1
            )
            raw_probability = tf.nn.sigmoid(logits)
            self.probabilities = (
                self.q_min + (self.q_max - self.q_min) * raw_probability
            )

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
        self.actor_loss = -tf.reduce_mean(surrogate) - entropy_coefficient * tf.reduce_mean(entropy)

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

        actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor")
        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")
        self.actor_train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            self.actor_loss, var_list=actor_variables
        )
        self.critic_train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            self.critic_loss, var_list=critic_variables
        )

    @staticmethod
    def _observation_feed(encoded) -> Mapping:
        return {
            "node_features": encoded.node_features,
            "edge_features": encoded.edge_features,
            "edge_mask": encoded.edge_mask,
        }

    def predict_probabilities(self, encoded) -> np.ndarray:
        feed = {
            self.node_features: encoded.node_features,
            self.edge_features: encoded.edge_features,
            self.edge_mask: encoded.edge_mask,
        }
        return self.session.run(self.probabilities, feed_dict=feed)

    def act(self, encoded, rng: np.random.Generator):
        probabilities = self.predict_probabilities(encoded)
        actions = (rng.random(probabilities.size) < probabilities).astype(np.float32)
        log_probabilities = (
            actions * np.log(np.maximum(probabilities, 1e-7))
            + (1.0 - actions) * np.log(np.maximum(1.0 - probabilities, 1e-7))
        )
        return actions, probabilities, log_probabilities.astype(np.float32)

    def value(self, global_states: np.ndarray) -> np.ndarray:
        states = np.asarray(global_states, dtype=np.float32).reshape(-1, GLOBAL_STATE_DIM)
        return self.session.run(self.values, feed_dict={self.global_states: states})

    def update(self, actor_batch: Mapping, critic_batch: Mapping, epochs: int = 4):
        actor_feed = {
            self.node_features: actor_batch["node_features"],
            self.edge_features: actor_batch["edge_features"],
            self.edge_mask: actor_batch["edge_mask"],
            self.actions: actor_batch["actions"],
            self.old_log_probabilities: actor_batch["old_log_probabilities"],
            self.advantages: actor_batch["advantages"],
        }
        critic_feed = {
            self.global_states: critic_batch["global_states"],
            self.returns: critic_batch["returns"],
        }
        actor_loss = critic_loss = 0.0
        for _ in range(int(epochs)):
            actor_loss, _ = self.session.run(
                [self.actor_loss, self.actor_train_op], feed_dict=actor_feed
            )
            critic_loss, _ = self.session.run(
                [self.critic_loss, self.critic_train_op], feed_dict=critic_feed
            )
        return {"actor_loss": float(actor_loss), "critic_loss": float(critic_loss)}

    def save(self, checkpoint_prefix: str) -> str:
        path = Path(checkpoint_prefix)
        path.parent.mkdir(parents=True, exist_ok=True)
        return self.saver.save(self.session, str(path))

    def restore(self, checkpoint_prefix: str) -> None:
        self.saver.restore(self.session, str(checkpoint_prefix))

    def close(self) -> None:
        self.session.close()
