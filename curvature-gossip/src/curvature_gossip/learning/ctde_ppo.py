"""Implement legacy, Stage-1, and hierarchical Stage-2 CTDE PPO actors.

Stage 1 owns only a shared curvature alpha.  Stage 2 supports its original
local-context residual MLP and a strictly local edge-message MPNN residual.
The centralized critic remains training-only for every actor architecture.
"""

from pathlib import Path
from typing import Mapping

import numpy as np

from .features import GLOBAL_STATE_DIM, NODE_FEATURE_DIM


def resolve_learning_rates(training_config: Mapping, default: float = 3e-4):
    """Resolve backward-compatible Actor and Critic learning rates from YAML."""
    training = dict(training_config or {})
    shared_learning_rate = float(training.get("learning_rate", default))
    actor_learning_rate = float(training.get("actor_learning_rate", shared_learning_rate))
    critic_learning_rate = float(training.get("critic_learning_rate", shared_learning_rate))
    for name, value in (("learning_rate", shared_learning_rate),
                        ("actor_learning_rate", actor_learning_rate),
                        ("critic_learning_rate", critic_learning_rate)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("training.{} must be finite and positive".format(name))
    return actor_learning_rate, critic_learning_rate


class CTDEPPO:
    """PPO model whose actor architecture is selected explicitly by ``actor.stage``."""

    def __init__(self, learning_rate=3e-4, clip_ratio=0.2, entropy_coefficient=0.01,
                 seed=0, session_config=None, actor_config=None,
                 actor_learning_rate=None, critic_learning_rate=None) -> None:
        import tensorflow as tensorflow

        # Keep the original shared argument while allowing either optimizer to override it.
        shared_learning_rate = float(learning_rate)
        self.actor_learning_rate = (
            shared_learning_rate if actor_learning_rate is None else float(actor_learning_rate)
        )
        self.critic_learning_rate = (
            shared_learning_rate if critic_learning_rate is None else float(critic_learning_rate)
        )
        for name, value in (("actor_learning_rate", self.actor_learning_rate),
                            ("critic_learning_rate", self.critic_learning_rate)):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be finite and positive".format(name))
        self.actor_config = dict(actor_config or {})
        self.actor_stage = int(self.actor_config.get("stage", 0))
        if self.actor_stage not in (0, 1, 2):
            raise ValueError("CTDEPPO supports legacy stage 0, Stage 1, or Stage 2 actors")
        self.tf = tensorflow.compat.v1
        self.tf.disable_v2_behavior()
        self.graph = self.tf.Graph()
        with self.graph.as_default():
            self.tf.set_random_seed(int(seed))
            self._build_graph(
                self.actor_learning_rate, self.critic_learning_rate,
                float(clip_ratio), float(entropy_coefficient),
            )
            self.saver = self.tf.train.Saver(max_to_keep=None)
            self.init_op = self.tf.global_variables_initializer()
        self.session = self.tf.Session(graph=self.graph, config=session_config)
        self.session.run(self.init_op)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("actor.curvature.alpha_init must be finite and positive")
        return float(np.log(np.expm1(value)))

    def _build_curvature_base(self):
        """Build the shared curvature base logit used by both hierarchical stages."""
        tf = self.tf
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
            self.effective_alpha = (
                tf.constant(float(override), dtype=tf.float32, name="alpha_override")
                if override is not None else self.alpha_kappa
            )
            safe_b = tf.clip_by_value(self.target_tx_ratios, 1e-6, 1.0 - 1e-6)
            base_logit = tf.log(safe_b) - tf.log(1.0 - safe_b)
            self.base_logits = base_logit + self.effective_alpha * (self.curvature_scores[:, 0] - center)
            self.base_probabilities = tf.nn.sigmoid(self.base_logits, name="stage1_probability")
        self.alpha_override_active = override is not None

    def _build_actor(self):
        tf = self.tf
        if self.actor_stage in (1, 2):
            self._build_curvature_base()
            if self.actor_stage == 1:
                self.logits = tf.identity(self.base_logits, name="final_logits")
                self.probabilities = tf.identity(self.base_probabilities, name="transmission_probability")
                self.residual_delta = self.residual_input = None
                return

            residual = dict(self.actor_config.get("residual", {}))
            self.residual_architecture = str(residual.get("architecture", "mlp"))
            if self.residual_architecture not in ("mlp", "mpnn"):
                raise ValueError("Stage 2 residual.architecture must be 'mlp' or 'mpnn'")
            context_width = 11 if bool(residual.get("include_scenario_context", False)) else 8
            hidden_dims = list(residual.get("hidden_dims", [64, 64]))
            if hidden_dims != [64, 64]:
                raise ValueError("Stage 2 requires residual.hidden_dims=[64, 64]")
            delta_max = float(residual.get("delta_max", 1.0))
            if not np.isfinite(delta_max) or delta_max <= 0.0:
                raise ValueError("Stage 2 residual.delta_max must be finite and positive")
            if self.residual_architecture == "mpnn":
                self._build_mpnn_residual(residual)
                return

            self.residual_context = tf.placeholder(tf.float32, [None, context_width], name="residual_context")
            use_stage1_reference = bool(residual.get("use_stage1_reference", True))
            if use_stage1_reference:
                # The detached first-layer probability is the ninth (or twelfth) MLP input.
                q_base_reference = tf.stop_gradient(
                    tf.expand_dims(self.base_probabilities, axis=1), name="detached_stage1_reference"
                )
                self.residual_input = tf.concat(
                    [self.residual_context, q_base_reference], axis=1, name="residual_input"
                )
            else:
                # Strict ablation: no Stage-1 reference or curvature-derived tensor enters the MLP.
                self.residual_input = tf.identity(self.residual_context, name="residual_input")
            with tf.variable_scope("actor/residual_mlp"):
                hidden = tf.layers.dense(self.residual_input, 64, activation=tf.nn.relu, name="dense_1")
                hidden = tf.layers.dense(hidden, 64, activation=tf.nn.relu, name="dense_2")
                residual_raw = tf.squeeze(tf.layers.dense(
                    hidden, 1, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer(),
                    name="delta_logit",
                ), axis=1)
            self.residual_delta = tf.identity(delta_max * tf.tanh(residual_raw), name="residual_delta")
            self.logits = tf.identity(self.base_logits + self.residual_delta, name="final_logits")
            self.probabilities = tf.nn.sigmoid(self.logits, name="transmission_probability")
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
        self.base_logits = self.base_probabilities = self.residual_delta = self.residual_input = None
        self.effective_alpha = None
        self.residual_architecture = None

    def _build_mpnn_residual(self, residual):
        """Build the receiver-conditioned one-hop MPNN with vectorized segment aggregation."""
        tf = self.tf
        node_width = 8 if bool(residual.get("include_scenario_context", False)) else 5
        mpnn = dict(residual.get("mpnn", {}))
        if (list(mpnn.get("message_hidden_dims", [32])) != [32]
                or int(mpnn.get("message_dim", 16)) != 16
                or list(mpnn.get("update_hidden_dims", [64, 64])) != [64, 64]
                or mpnn.get("aggregation", "mean_max") != "mean_max"
                or not bool(mpnn.get("condition_message_on_receiver", True))
                or bool(mpnn.get("use_sender_node_features", False))
                or bool(mpnn.get("use_curvature_edge_feature", False))):
            raise ValueError("Stage 2 MPNN must use the fixed receiver-conditioned mean_max architecture")
        delta_max = float(residual.get("delta_max", 1.0))
        if not np.isfinite(delta_max) or delta_max <= 0.0:
            raise ValueError("Stage 2 residual.delta_max must be finite and positive")
        self.mpnn_node_context = tf.placeholder(tf.float32, [None, node_width], name="mpnn_node_context")
        self.mpnn_edge_features = tf.placeholder(tf.float32, [None, 3], name="mpnn_edge_features")
        self.mpnn_edge_receivers = tf.placeholder(tf.int32, [None], name="mpnn_edge_receivers")
        receiver_context = tf.gather(self.mpnn_node_context, self.mpnn_edge_receivers)
        message_input = tf.concat([receiver_context, self.mpnn_edge_features], axis=1, name="message_input")
        with tf.variable_scope("actor/residual_mpnn/message_mlp"):
            message_hidden = tf.layers.dense(message_input, 32, activation=tf.nn.relu, name="dense_1")
            edge_messages = tf.layers.dense(message_hidden, 16, activation=tf.nn.relu, name="dense_2")
        node_count = tf.shape(self.mpnn_node_context)[0]
        message_sum = tf.math.unsorted_segment_sum(edge_messages, self.mpnn_edge_receivers, node_count)
        edge_count = tf.math.unsorted_segment_sum(
            tf.ones_like(self.mpnn_edge_receivers, dtype=tf.float32), self.mpnn_edge_receivers, node_count
        )
        mean_message = message_sum / tf.maximum(tf.expand_dims(edge_count, 1), 1.0)
        max_message = tf.math.unsorted_segment_max(edge_messages, self.mpnn_edge_receivers, node_count)
        # TensorFlow 1.x Select does not broadcast the [N, 1] edge mask.
        has_edges = tf.tile(tf.expand_dims(edge_count > 0.0, 1), [1, 16])
        max_message = tf.where(has_edges, max_message, tf.zeros_like(max_message))
        self.mpnn_edge_messages = tf.identity(edge_messages, name="edge_messages")
        self.mpnn_aggregated_messages = tf.identity(
            tf.concat([mean_message, max_message], axis=1), name="aggregated_messages"
        )
        use_stage1_reference = bool(residual.get("use_stage1_reference", True))
        decoder_parts = [self.mpnn_node_context, self.mpnn_aggregated_messages]
        if use_stage1_reference:
            decoder_parts.append(tf.stop_gradient(
                tf.expand_dims(self.base_probabilities, axis=1), name="detached_stage1_reference"
            ))
        self.residual_input = tf.concat(decoder_parts, axis=1, name="residual_input")
        with tf.variable_scope("actor/residual_mpnn/update_mlp"):
            hidden = tf.layers.dense(self.residual_input, 64, activation=tf.nn.relu, name="dense_1")
            hidden = tf.layers.dense(hidden, 64, activation=tf.nn.relu, name="dense_2")
            residual_raw = tf.squeeze(tf.layers.dense(
                hidden, 1, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer(),
                name="delta_logit",
            ), axis=1)
        self.residual_delta = tf.identity(delta_max * tf.tanh(residual_raw), name="residual_delta")
        self.logits = tf.identity(self.base_logits + self.residual_delta, name="final_logits")
        self.probabilities = tf.nn.sigmoid(self.logits, name="transmission_probability")

    def _build_graph(self, actor_learning_rate, critic_learning_rate, clip_ratio, entropy_coefficient):
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
        if self.actor_stage == 1:
            actor_update_variables = [] if self.alpha_override_active else actor_variables
        elif self.actor_stage == 2:
            freeze_stage1 = bool(dict(self.actor_config.get("residual", {})).get("freeze_stage1", False))
            actor_update_variables = [
                variable for variable in actor_variables
                if not (variable.name.startswith("actor/alpha_raw") and (freeze_stage1 or self.alpha_override_active))
            ]
        else:
            actor_update_variables = actor_variables
        self.actor_train_op = (tf.train.AdamOptimizer(actor_learning_rate).minimize(self.actor_loss, var_list=actor_update_variables)
                               if actor_update_variables else tf.no_op())
        self.critic_train_op = tf.train.AdamOptimizer(critic_learning_rate).minimize(self.critic_loss, var_list=critic_variables)

    def _encoded_feed(self, encoded):
        if self.actor_stage == 2:
            if self.residual_architecture == "mpnn":
                return {self.curvature_scores: encoded.curvature_scores, self.target_tx_ratios: encoded.target_tx_ratios,
                        self.mpnn_node_context: encoded.node_context, self.mpnn_edge_features: encoded.edge_features,
                        self.mpnn_edge_receivers: encoded.edge_index[1]}
            return {self.curvature_scores: encoded.curvature_scores, self.target_tx_ratios: encoded.target_tx_ratios,
                    self.residual_context: encoded.residual_context}
        if self.actor_stage == 1:
            return {self.curvature_scores: encoded.curvature_scores, self.target_tx_ratios: encoded.target_tx_ratios}
        return {self.node_features: encoded.node_features}

    def actor_batch_inputs(self, encoded_steps):
        if self.actor_stage == 2:
            if self.residual_architecture == "mpnn":
                node_counts = [item.node_context.shape[0] for item in encoded_steps]
                offsets = np.cumsum([0] + node_counts[:-1]).astype(np.int32)
                edge_indices = [item.edge_index + offset for item, offset in zip(encoded_steps, offsets)]
                return {"curvature_scores": np.concatenate([item.curvature_scores for item in encoded_steps]),
                        "target_tx_ratios": np.concatenate([item.target_tx_ratios for item in encoded_steps]),
                        "node_context": np.concatenate([item.node_context for item in encoded_steps]),
                        "edge_index": np.concatenate(edge_indices, axis=1).astype(np.int32, copy=False),
                        "edge_features": np.concatenate([item.edge_features for item in encoded_steps]).astype(np.float32, copy=False)}
            return {"curvature_scores": np.concatenate([item.curvature_scores for item in encoded_steps]),
                    "target_tx_ratios": np.concatenate([item.target_tx_ratios for item in encoded_steps]),
                    "residual_context": np.concatenate([item.residual_context for item in encoded_steps])}
        if self.actor_stage == 1:
            return {"curvature_scores": np.concatenate([item.curvature_scores for item in encoded_steps]),
                    "target_tx_ratios": np.concatenate([item.target_tx_ratios for item in encoded_steps])}
        return {"node_features": np.concatenate([item.node_features for item in encoded_steps])}

    def _actor_batch_feed(self, batch):
        if self.actor_stage == 2:
            if self.residual_architecture == "mpnn":
                return {self.curvature_scores: batch["curvature_scores"], self.target_tx_ratios: batch["target_tx_ratios"],
                        self.mpnn_node_context: batch["node_context"], self.mpnn_edge_features: batch["edge_features"],
                        self.mpnn_edge_receivers: batch["edge_index"][1]}
            return {self.curvature_scores: batch["curvature_scores"], self.target_tx_ratios: batch["target_tx_ratios"],
                    self.residual_context: batch["residual_context"]}
        if self.actor_stage == 1:
            return {self.curvature_scores: batch["curvature_scores"], self.target_tx_ratios: batch["target_tx_ratios"]}
        return {self.node_features: batch["node_features"]}

    def predict_probabilities(self, encoded):
        return self.session.run(self.probabilities, feed_dict=self._encoded_feed(encoded))

    def predict_stage1_probabilities(self, encoded):
        """Return the frozen curvature-base probabilities for paired Stage-1-only validation."""
        if self.actor_stage not in (1, 2):
            raise ValueError("Stage-1-only probabilities require a Stage 1 or Stage 2 actor")
        probabilities = self.base_probabilities if self.actor_stage == 2 else self.probabilities
        return self.session.run(probabilities, feed_dict=self._encoded_feed(encoded))

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
        actor_feed.update({
            self.actions: actor_batch["actions"],
            self.old_log_probabilities: actor_batch["old_log_probabilities"],
            self.advantages: actor_batch["advantages"],
        })
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

    def stage2_diagnostics(self, encoded):
        """Return base, residual, and final probabilities for Stage-2 logs and tests."""
        if self.actor_stage != 2:
            return {}
        values = self.session.run({
            "alpha_raw": self.alpha_raw, "alpha_kappa": self.alpha_kappa,
            "effective_alpha": self.effective_alpha,
            "q_base": self.base_probabilities, "delta": self.residual_delta,
            "q_final": self.probabilities, "residual_input": self.residual_input,
        }, feed_dict=self._encoded_feed(encoded))
        if self.residual_architecture == "mpnn":
            values.update(self.session.run({
                "edge_messages": self.mpnn_edge_messages,
                "aggregated_messages": self.mpnn_aggregated_messages,
            }, feed_dict=self._encoded_feed(encoded)))
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

    def restore_compatible(self, checkpoint_prefix):
        """Restore matching old-stage variables while retaining newly initialized MLP variables."""
        reader = self.tf.train.NewCheckpointReader(str(checkpoint_prefix))
        available = reader.get_variable_to_shape_map()
        with self.graph.as_default():
            variables = self.tf.global_variables()
        matched = {
            variable.name.split(":")[0]: variable for variable in variables
            if variable.name.split(":")[0] in available
            and tuple(variable.shape.as_list()) == tuple(available[variable.name.split(":")[0]])
        }
        if not matched:
            raise ValueError("checkpoint has no variables compatible with the current model")
        self.tf.train.Saver(var_list=matched).restore(self.session, str(checkpoint_prefix))
        return tuple(sorted(matched))

    def close(self):
        self.session.close()
