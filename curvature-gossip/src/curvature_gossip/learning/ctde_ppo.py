"""Implement CTDE PPO with a legacy V3 actor or the Stage-1 curvature actor.

Stage 1 deliberately has one shared actor parameter, ``alpha_raw``.  The
centralized critic remains unchanged and is used only while training.
"""

from pathlib import Path
from typing import Mapping

import numpy as np

from .features import GLOBAL_STATE_DIM, NODE_FEATURE_DIM


class CTDEPPO:
    """PPO model whose actor architecture is selected explicitly by ``actor.stage``."""

    def __init__(self, learning_rate=3e-4, clip_ratio=0.2, entropy_coefficient=0.01,
                 seed=0, session_config=None, actor_config=None) -> None:
        import tensorflow as tensorflow

        self.actor_config = dict(actor_config or {})
        self.actor_stage = int(self.actor_config.get("stage", 0))
        if self.actor_stage not in (0, 1):
            raise ValueError("CTDEPPO supports legacy stage 0 or Stage-1 actor only")
        self.tf = tensorflow.compat.v1
        self.tf.disable_v2_behavior()
        self.graph = self.tf.Graph()
        with self.graph.as_default():
            self.tf.set_random_seed(int(seed))
            self._build_graph(float(learning_rate), float(clip_ratio), float(entropy_coefficient))
            self.saver = self.tf.train.Saver(max_to_keep=None)
            self.init_op = self.tf.global_variables_initializer()
        self.session = self.tf.Session(graph=self.graph, config=session_config)
        self.session.run(self.init_op)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("actor.curvature.alpha_init must be finite and positive")
        return float(np.log(np.expm1(value)))

    def _build_actor(self):
        tf = self.tf
        if self.actor_stage == 1:
            curvature = dict(self.actor_config.get("curvature", {}))
            center = float(curvature.get("center", 0.0))
            override = curvature.get("alpha_override")
            if not np.isfinite(center):
                raise ValueError("actor.curvature.center must be finite")
            if override is not None and (not np.isfinite(float(override)) or float(override) < 0.0):
                raise ValueError("actor.curvature.alpha_override must be null or nonnegative")
            self.curvature_scores = tf.placeholder(tf.float32, [None, 1], name="curvature_scores")
            self.target_tx_ratios = tf.placeholder(tf.float32, [None], name="target_tx_ratios")
            with tf.variable_scope("actor"):
                self.alpha_raw = tf.get_variable(
                    "alpha_raw", initializer=self._inverse_softplus(float(curvature.get("alpha_init", 0.1)))
                )
                self.alpha_kappa = tf.nn.softplus(self.alpha_raw, name="alpha_kappa")
                effective_alpha = (
                    tf.constant(float(override), dtype=tf.float32, name="alpha_override")
                    if override is not None else self.alpha_kappa
                )
                safe_b = tf.clip_by_value(self.target_tx_ratios, 1e-6, 1.0 - 1e-6)
                base_logit = tf.log(safe_b) - tf.log(1.0 - safe_b)
                self.base_logits = base_logit + effective_alpha * (self.curvature_scores[:, 0] - center)
                self.logits = tf.identity(self.base_logits, name="final_logits")
                self.probabilities = tf.nn.sigmoid(self.logits, name="transmission_probability")
            self.alpha_override_active = override is not None
            return

        self.node_features = tf.placeholder(tf.float32, [None, NODE_FEATURE_DIM], name="node_features")
        with tf.variable_scope("actor"):
            hidden = tf.layers.dense(self.node_features, 64, activation=tf.nn.relu, name="node_dense_1")
            hidden = tf.layers.dense(hidden, 64, activation=tf.nn.relu, name="node_dense_2")
            residual = tf.squeeze(tf.layers.dense(
                hidden, 1, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer(),
                name="transmission_residual_logit"), axis=1)
            target = tf.clip_by_value(self.node_features[:, 13], 1e-6, 1.0 - 1e-6)
            self.logits = tf.log(target) - tf.log(1.0 - target) + residual
            self.probabilities = tf.clip_by_value(tf.nn.sigmoid(self.logits), 1e-6, 1.0 - 1e-6)
        self.alpha_raw = self.alpha_kappa = None
        self.alpha_override_active = False

    def _build_graph(self, learning_rate, clip_ratio, entropy_coefficient):
        tf = self.tf
        self.actions = tf.placeholder(tf.float32, [None], name="actions")
        self.old_log_probabilities = tf.placeholder(tf.float32, [None], name="old_log_probabilities")
        self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
        self._build_actor()
        distribution = tf.distributions.Bernoulli(logits=self.logits)
        self.log_probabilities = distribution.log_prob(self.actions)
        ratio = tf.exp(self.log_probabilities - self.old_log_probabilities)
        clipped = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        self.surrogate_actor_loss = -tf.reduce_mean(tf.minimum(ratio * self.advantages, clipped * self.advantages))
        self.mean_entropy = tf.reduce_mean(distribution.entropy())
        self.actor_loss = self.surrogate_actor_loss - entropy_coefficient * self.mean_entropy
        self.approx_kl = tf.reduce_mean(self.old_log_probabilities - self.log_probabilities)
        self.clip_fraction = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > clip_ratio, tf.float32))

        self.global_states = tf.placeholder(tf.float32, [None, GLOBAL_STATE_DIM], name="global_states")
        self.returns = tf.placeholder(tf.float32, [None], name="returns")
        with tf.variable_scope("critic"):
            hidden = tf.layers.dense(self.global_states, 64, activation=tf.nn.relu, name="dense_1")
            hidden = tf.layers.dense(hidden, 64, activation=tf.nn.relu, name="dense_2")
            self.values = tf.squeeze(tf.layers.dense(hidden, 1, name="value"), axis=1)
        self.critic_loss = tf.reduce_mean(tf.square(self.returns - self.values))
        actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor")
        gradients = tf.gradients(self.surrogate_actor_loss, actor_variables)
        self.policy_gradient_norm_op = tf.global_norm([item for item in gradients if item is not None])
        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")
        actor_update_variables = [] if self.alpha_override_active else actor_variables
        self.actor_train_op = (tf.train.AdamOptimizer(learning_rate).minimize(self.actor_loss, var_list=actor_update_variables)
                               if actor_update_variables else tf.no_op())
        self.critic_train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.critic_loss, var_list=critic_variables)

    def _encoded_feed(self, encoded):
        if self.actor_stage == 1:
            return {self.curvature_scores: encoded.curvature_scores, self.target_tx_ratios: encoded.target_tx_ratios}
        return {self.node_features: encoded.node_features}

    def actor_batch_inputs(self, encoded_steps):
        if self.actor_stage == 1:
            return {"curvature_scores": np.concatenate([item.curvature_scores for item in encoded_steps]),
                    "target_tx_ratios": np.concatenate([item.target_tx_ratios for item in encoded_steps])}
        return {"node_features": np.concatenate([item.node_features for item in encoded_steps])}

    def _actor_batch_feed(self, batch):
        if self.actor_stage == 1:
            return {self.curvature_scores: batch["curvature_scores"], self.target_tx_ratios: batch["target_tx_ratios"]}
        return {self.node_features: batch["node_features"]}

    def predict_probabilities(self, encoded):
        return self.session.run(self.probabilities, feed_dict=self._encoded_feed(encoded))

    def act(self, encoded, rng):
        probabilities = self.predict_probabilities(encoded)
        actions = (rng.random(probabilities.size) < probabilities).astype(np.float32)
        feed = self._encoded_feed(encoded)
        feed[self.actions] = actions
        log_probabilities = self.session.run(self.log_probabilities, feed_dict=feed)
        return actions, probabilities, log_probabilities.astype(np.float32)

    def value(self, global_states):
        states = np.asarray(global_states, dtype=np.float32).reshape(-1, GLOBAL_STATE_DIM)
        return self.session.run(self.values, feed_dict={self.global_states: states})

    def update(self, actor_batch: Mapping, critic_batch: Mapping, epochs=4):
        actor_feed = self._actor_batch_feed(actor_batch)
        actor_feed.update({self.actions: actor_batch["actions"], self.old_log_probabilities: actor_batch["old_log_probabilities"], self.advantages: actor_batch["advantages"]})
        critic_feed = {self.global_states: critic_batch["global_states"], self.returns: critic_batch["returns"]}
        for _ in range(int(epochs)):
            actor_loss, entropy, approx_kl, clip_fraction, _ = self.session.run(
                [self.actor_loss, self.mean_entropy, self.approx_kl, self.clip_fraction, self.actor_train_op], feed_dict=actor_feed)
            critic_loss, _ = self.session.run([self.critic_loss, self.critic_train_op], feed_dict=critic_feed)
        return {"actor_loss": float(actor_loss), "critic_loss": float(critic_loss), "entropy": float(entropy), "approx_kl": float(approx_kl), "clip_fraction": float(clip_fraction)}

    def policy_gradient_norm(self, actor_batch: Mapping) -> float:
        feed = self._actor_batch_feed(actor_batch)
        feed.update({self.actions: actor_batch["actions"], self.old_log_probabilities: actor_batch["old_log_probabilities"], self.advantages: actor_batch["advantages"]})
        return float(self.session.run(self.policy_gradient_norm_op, feed_dict=feed))

    def stage1_diagnostics(self, encoded):
        """Return alpha and q-base values for Stage-1 training logs and tests."""
        if self.actor_stage != 1:
            return {}
        values = self.session.run({"alpha_raw": self.alpha_raw, "alpha_kappa": self.alpha_kappa,
                                   "q_base": self.probabilities}, feed_dict=self._encoded_feed(encoded))
        return {key: np.asarray(value) for key, value in values.items()}

    def variable_snapshot(self):
        with self.graph.as_default():
            variables = self.tf.global_variables()
        return tuple(self.session.run(variables))

    def save(self, checkpoint_prefix):
        path = Path(checkpoint_prefix); path.parent.mkdir(parents=True, exist_ok=True)
        return self.saver.save(self.session, str(path))

    def restore(self, checkpoint_prefix):
        self.saver.restore(self.session, str(checkpoint_prefix))

    def close(self):
        self.session.close()
