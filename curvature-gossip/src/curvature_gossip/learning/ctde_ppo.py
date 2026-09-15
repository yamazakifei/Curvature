"""Implement legacy, Stage-1, and hierarchical Stage-2 CTDE PPO actors.

Stage 1 owns only a shared curvature alpha.  Stage 2 supports its original
local-context residual MLP and a strictly local edge-message MPNN residual.
The centralized critic remains training-only for every actor architecture.
"""

from pathlib import Path
from typing import Mapping

import numpy as np

from .features import (
    GLOBAL_STATE_DIM, NODE_FEATURE_DIM, node_critic_feature_names,
    stage2_mpnn_edge_feature_names,
)


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
                 critic_config=None, actor_learning_rate=None, critic_learning_rate=None,
                 common_mode_coefficient=0.0,
                 probability_budget_coefficient=0.0) -> None:
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
        self.common_mode_coefficient = float(common_mode_coefficient)
        if not np.isfinite(self.common_mode_coefficient) or self.common_mode_coefficient < 0.0:
            raise ValueError("common_mode_coefficient must be finite and nonnegative")
        self.probability_budget_coefficient = float(probability_budget_coefficient)
        if (not np.isfinite(self.probability_budget_coefficient)
                or self.probability_budget_coefficient < 0.0):
            raise ValueError("probability_budget_coefficient must be finite and nonnegative")
        self.actor_config = dict(actor_config or {})
        self.critic_config = dict(critic_config or {})
        self.critic_architecture = str(self.critic_config.get("architecture", "scalar_global"))
        if self.critic_architecture not in ("scalar_global", "node_conditioned"):
            raise ValueError("critic.architecture must be scalar_global or node_conditioned")
        if self.critic_architecture == "node_conditioned":
            self.critic_include_scenario_context = bool(
                self.critic_config.get("include_scenario_context", True)
            )
            self.critic_include_exact_node_features = bool(
                self.critic_config.get("include_exact_node_features", True)
            )
            self.critic_include_channel_features = bool(
                self.critic_config.get("include_channel_features", True)
            )
            self.critic_include_curvature_features = bool(
                self.critic_config.get("include_curvature_features", True)
            )
            self.critic_feature_names = node_critic_feature_names(
                self.critic_include_scenario_context,
                self.critic_include_exact_node_features,
                self.critic_include_channel_features,
                self.critic_include_curvature_features,
            )
            self.critic_input_dim = int(self.critic_config.get("input_dim", len(self.critic_feature_names)))
            if self.critic_input_dim != len(self.critic_feature_names):
                raise ValueError("critic.input_dim does not match the configured feature groups")
        else:
            self.critic_feature_names = tuple("legacy_global_{}".format(i) for i in range(GLOBAL_STATE_DIM))
            self.critic_input_dim = GLOBAL_STATE_DIM
        self.actor_stage = int(self.actor_config.get("stage", 0))
        if self.actor_stage not in (0, 1, 2):
            raise ValueError("CTDEPPO supports legacy stage 0, Stage 1, or Stage 2 actors")
        self.actor_curvature_enabled = bool(
            dict(self.actor_config.get("curvature", {})).get("enabled", True)
        )
        if self.actor_stage == 1 and not self.actor_curvature_enabled:
            raise ValueError("Stage 1 requires actor.curvature.enabled=true")
        self.tf = tensorflow.compat.v1
        self.tf.disable_v2_behavior()
        self.graph = self.tf.Graph()
        with self.graph.as_default():
            self.tf.set_random_seed(int(seed))
            self._build_graph(
                self.actor_learning_rate, self.critic_learning_rate,
                float(clip_ratio), float(entropy_coefficient), self.common_mode_coefficient,
                self.probability_budget_coefficient,
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
        """Build the shared Stage-1 base logit used by both hierarchical stages.

        ``max_tx_ratio`` remains a scenario/budget feature.  A calibrated
        intercept or explicit ``base_tx_ratio`` controls the Stage-1 center.
        """
        tf = self.tf
        curvature = dict(self.actor_config.get("curvature", {}))
        # No-curvature ablations do not need the auto-calibrated center; avoid
        # converting the literal ``auto`` into a float on this path.
        center_value = curvature.get("center", 0.0)
        center = 0.0 if not self.actor_curvature_enabled else float(center_value)
        override = curvature.get("alpha_override")
        if not np.isfinite(center):
            raise ValueError("actor.curvature.center must be finite")
        if override is not None and (not np.isfinite(float(override)) or float(override) < 0.0):
            raise ValueError("actor.curvature.alpha_override must be null or nonnegative")
        if self.actor_curvature_enabled:
            self.curvature_scores = tf.placeholder(tf.float32, [None, 1], name="curvature_scores")
        else:
            # Keep alpha variables for checkpoint diagnostics, but disconnect curvature from forward pass.
            self.curvature_scores = None
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
            configured_intercept = curvature.get("calibrated_intercept")
            configured_base = curvature.get("base_tx_ratio")
            if configured_intercept is not None:
                configured_intercept = float(configured_intercept)
                if not np.isfinite(configured_intercept):
                    raise ValueError("actor.curvature.calibrated_intercept must be finite")
                base_intercept = configured_intercept
            elif configured_base is not None:
                configured_base = float(configured_base)
                if not np.isfinite(configured_base) or not 0.0 < configured_base <= 1.0:
                    raise ValueError("actor.curvature.base_tx_ratio must be in (0, 1]")
                safe_base = float(np.clip(configured_base, 1e-6, 1.0 - 1e-6))
                base_intercept = float(np.log(safe_base) - np.log1p(-safe_base))
            else:
                # Legacy models use the per-scenario target as their base center.
                safe_b = tf.clip_by_value(self.target_tx_ratios, 1e-6, 1.0 - 1e-6)
                base_logit = tf.log(safe_b) - tf.log(1.0 - safe_b)
                base_intercept = None
            if base_intercept is not None:
                base_logit = tf.ones_like(self.target_tx_ratios) * float(base_intercept)
            self.base_logits = (
                base_logit + self.effective_alpha * (self.curvature_scores[:, 0] - center)
                if self.actor_curvature_enabled else base_logit
            )
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
                    hidden, 1, use_bias=False, kernel_initializer=tf.zeros_initializer(),
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
        """Build the receiver-conditioned one-hop MPNN and its pooling rule.

        V3.5 adds a fixed curvature-prior attention option.  The attention is
        deliberately parameter-free: normalized bottleneck curvature controls
        only the mean-pooling weights, while the element-wise max branch is
        retained unchanged for a clean V3.4 comparison.
        """
        tf = self.tf
        node_width = 8 if bool(residual.get("include_scenario_context", False)) else 5
        mpnn = dict(residual.get("mpnn", {}))
        aggregation = str(mpnn.get("aggregation", "mean_max"))
        if (list(mpnn.get("message_hidden_dims", [32])) != [32]
                or int(mpnn.get("message_dim", 16)) != 16
                or list(mpnn.get("update_hidden_dims", [64, 64])) != [64, 64]
                or aggregation not in ("mean_max", "curvature_attention_max")
                or not bool(mpnn.get("condition_message_on_receiver", True))
                or bool(mpnn.get("use_sender_node_features", False))):
            raise ValueError(
                "Stage 2 MPNN must use the fixed receiver-conditioned mean_max "
                "or curvature_attention_max architecture"
            )
        self.use_curvature_edge_feature = bool(mpnn.get("use_curvature_edge_feature", False))
        if aggregation == "curvature_attention_max" and not self.use_curvature_edge_feature:
            raise ValueError(
                "curvature_attention_max requires use_curvature_edge_feature=true"
            )
        self.mpnn_aggregation = aggregation
        self.mpnn_attention_temperature = float(mpnn.get("attention_temperature", 0.5))
        self.mpnn_attention_uniform_mix = float(mpnn.get("attention_uniform_mix", 0.0))
        if (not np.isfinite(self.mpnn_attention_temperature)
                or self.mpnn_attention_temperature <= 0.0):
            raise ValueError("Stage 2 MPNN attention_temperature must be finite and positive")
        if (not np.isfinite(self.mpnn_attention_uniform_mix)
                or not 0.0 <= self.mpnn_attention_uniform_mix <= 1.0):
            raise ValueError("Stage 2 MPNN attention_uniform_mix must be in [0, 1]")
        self.mpnn_bmax = float(mpnn.get("bmax", 1.0))
        if not np.isfinite(self.mpnn_bmax) or self.mpnn_bmax <= 0.0:
            raise ValueError("Stage 2 MPNN bmax must be finite and positive")
        delta_max = float(residual.get("delta_max", 1.0))
        if not np.isfinite(delta_max) or delta_max <= 0.0:
            raise ValueError("Stage 2 residual.delta_max must be finite and positive")
        self.mpnn_edge_input_dim = len(stage2_mpnn_edge_feature_names(
            self.use_curvature_edge_feature,
            str(mpnn.get("edge_freshness_feature", "binary_fraction")),
        ))
        self.mpnn_edge_curvature_index = self.mpnn_edge_input_dim - 1
        self.mpnn_node_context = tf.placeholder(tf.float32, [None, node_width], name="mpnn_node_context")
        self.mpnn_edge_features = tf.placeholder(
            tf.float32, [None, self.mpnn_edge_input_dim], name="mpnn_edge_features"
        )
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
        # V3.5 uses a stable per-receiver softmax over the normalized curvature
        # edge feature.  A uniform mix can limit attention collapse when enabled.
        if aggregation == "curvature_attention_max":
            curvature_scores = tf.clip_by_value(
                self.mpnn_edge_features[:, self.mpnn_edge_curvature_index], 0.0, 1.0
            )
            attention_logits = curvature_scores / self.mpnn_attention_temperature
            receiver_max_logits = tf.math.unsorted_segment_max(
                attention_logits, self.mpnn_edge_receivers, node_count
            )
            stable_logits = attention_logits - tf.gather(
                receiver_max_logits, self.mpnn_edge_receivers
            )
            unnormalized_weights = tf.exp(stable_logits)
            weight_sum = tf.math.unsorted_segment_sum(
                unnormalized_weights, self.mpnn_edge_receivers, node_count
            )
            curvature_weights = unnormalized_weights / tf.maximum(
                tf.gather(weight_sum, self.mpnn_edge_receivers), 1e-12
            )
            edge_uniform_weights = 1.0 / tf.maximum(
                tf.gather(edge_count, self.mpnn_edge_receivers), 1.0
            )
            attention_weights = (
                (1.0 - self.mpnn_attention_uniform_mix) * curvature_weights
                + self.mpnn_attention_uniform_mix * edge_uniform_weights
            )
            weighted_messages = edge_messages * tf.expand_dims(attention_weights, 1)
            mean_message = tf.math.unsorted_segment_sum(
                weighted_messages, self.mpnn_edge_receivers, node_count
            )
        else:
            # The legacy mean branch has the same normalized weights for
            # diagnostics, while its numerical aggregation stays unchanged.
            attention_weights = 1.0 / tf.maximum(
                tf.gather(edge_count, self.mpnn_edge_receivers), 1.0
            )
        attention_weight_sq_sum = tf.math.unsorted_segment_sum(
            tf.square(attention_weights), self.mpnn_edge_receivers, node_count
        )
        attention_effective_degree = tf.where(
            edge_count > 0.0,
            1.0 / tf.maximum(attention_weight_sq_sum, 1e-12),
            tf.zeros_like(attention_weight_sq_sum),
        )
        attention_entropy_terms = attention_weights * tf.log(
            tf.maximum(attention_weights, 1e-12)
        )
        attention_entropy = -tf.math.unsorted_segment_sum(
            attention_entropy_terms, self.mpnn_edge_receivers, node_count
        )
        attention_max = tf.math.unsorted_segment_max(
            attention_weights, self.mpnn_edge_receivers, node_count
        )
        attention_max = tf.where(
            edge_count > 0.0,
            attention_max,
            tf.zeros_like(attention_max),
        )
        self.mpnn_edge_messages = tf.identity(edge_messages, name="edge_messages")
        self.mpnn_attention_weights = tf.identity(attention_weights, name="attention_weights")
        self.mpnn_attention_entropy = tf.identity(attention_entropy, name="attention_entropy")
        self.mpnn_attention_effective_degree = tf.identity(
            attention_effective_degree, name="attention_effective_degree"
        )
        self.mpnn_attention_max_weight = tf.identity(
            attention_max, name="attention_max_weight"
        )
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
                hidden, 1, use_bias=False, kernel_initializer=tf.zeros_initializer(),
                name="delta_logit",
            ), axis=1)
        self.residual_delta = tf.identity(delta_max * tf.tanh(residual_raw), name="residual_delta")
        self.logits = tf.identity(self.base_logits + self.residual_delta, name="final_logits")
        self.probabilities = tf.nn.sigmoid(self.logits, name="transmission_probability")

    def _build_graph(
        self, actor_learning_rate, critic_learning_rate, clip_ratio,
        entropy_coefficient, common_mode_coefficient, probability_budget_coefficient,
    ):
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
        self.common_mode_loss = tf.constant(0.0, dtype=tf.float32, name="common_mode_loss")
        self.probability_budget_loss = tf.constant(
            0.0, dtype=tf.float32, name="probability_budget_loss"
        )
        self.probability_budget_excess = tf.constant(
            0.0, dtype=tf.float32, name="probability_budget_excess"
        )
        self.actor_step_ids = None
        self.number_of_steps = None
        if self.actor_stage == 2:
            # These placeholders exist only for the training update; inference never feeds them.
            self.actor_step_ids = tf.placeholder(tf.int32, [None], name="actor_step_ids")
            self.number_of_steps = tf.placeholder(tf.int32, shape=(), name="number_of_steps")
            residual_sum = tf.math.unsorted_segment_sum(
                self.residual_delta, self.actor_step_ids, self.number_of_steps
            )
            residual_count = tf.math.unsorted_segment_sum(
                tf.ones_like(self.residual_delta), self.actor_step_ids, self.number_of_steps
            )
            residual_mean = residual_sum / tf.maximum(residual_count, 1.0)
            self.common_mode_loss = tf.identity(
                tf.reduce_mean(tf.square(residual_mean)), name="common_mode_loss"
            )
            if probability_budget_coefficient > 0.0:
                # Penalize the mean probability of each rollout step when it
                # exceeds that step's Bmax; Stage-1 remains untouched.
                probability_sum = tf.math.unsorted_segment_sum(
                    self.probabilities, self.actor_step_ids, self.number_of_steps
                )
                probability_count = tf.math.unsorted_segment_sum(
                    tf.ones_like(self.probabilities), self.actor_step_ids, self.number_of_steps
                )
                step_mean_probability = probability_sum / tf.maximum(probability_count, 1.0)
                bmax_sum = tf.math.unsorted_segment_sum(
                    self.target_tx_ratios, self.actor_step_ids, self.number_of_steps
                )
                step_bmax = bmax_sum / tf.maximum(probability_count, 1.0)
                relative_excess = tf.nn.relu(
                    (step_mean_probability - step_bmax) / tf.maximum(step_bmax, 1e-6)
                )
                self.probability_budget_excess = tf.identity(
                    tf.reduce_mean(relative_excess), name="probability_budget_excess"
                )
                self.probability_budget_loss = tf.identity(
                    probability_budget_coefficient * tf.reduce_mean(tf.square(relative_excess)),
                    name="probability_budget_loss",
                )
        self.actor_loss = (
            self.surrogate_actor_loss
            - entropy_coefficient * self.mean_entropy
            + common_mode_coefficient * self.common_mode_loss
            + self.probability_budget_loss
        )
        self.approx_kl = tf.reduce_mean(self.old_log_probabilities - self.log_probabilities)
        self.clip_fraction = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > clip_ratio, tf.float32))

        if self.critic_architecture == "node_conditioned":
            # Node values are flattened time-major by the trainer, matching actor samples.
            self.critic_inputs = tf.placeholder(
                tf.float32, [None, self.critic_input_dim], name="critic_inputs"
            )
            critic_source = self.critic_inputs
        else:
            self.global_states = tf.placeholder(tf.float32, [None, GLOBAL_STATE_DIM], name="global_states")
            critic_source = self.global_states
        self.returns = tf.placeholder(tf.float32, [None], name="returns")
        with tf.variable_scope("critic"):
            hidden = tf.layers.dense(critic_source, 64, activation=tf.nn.relu, name="dense_1")
            hidden = tf.layers.dense(hidden, 64, activation=tf.nn.relu, name="dense_2")
            self.values = tf.squeeze(tf.layers.dense(hidden, 1, name="value"), axis=1)
        self.critic_loss = tf.reduce_mean(tf.square(self.returns - self.values))
        actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor")
        gradients = tf.gradients(self.surrogate_actor_loss, actor_variables)
        self.policy_gradient_norm_op = tf.global_norm([item for item in gradients if item is not None])
        common_gradients = tf.gradients(common_mode_coefficient * self.common_mode_loss, actor_variables)
        self.common_mode_gradient_norm_op = tf.global_norm([item for item in common_gradients if item is not None])
        total_gradients = tf.gradients(self.actor_loss, actor_variables)
        self.total_actor_gradient_norm_op = tf.global_norm([item for item in total_gradients if item is not None])
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
                edge_features = self._prepare_mpnn_edge_features(encoded.edge_features)
                feed = {self.target_tx_ratios: encoded.target_tx_ratios,
                        self.mpnn_node_context: encoded.node_context, self.mpnn_edge_features: edge_features,
                        self.mpnn_edge_receivers: encoded.edge_index[1]}
            else:
                feed = {self.target_tx_ratios: encoded.target_tx_ratios,
                        self.residual_context: encoded.residual_context}
            if self.actor_curvature_enabled:
                feed[self.curvature_scores] = encoded.curvature_scores
            return feed
        if self.actor_stage == 1:
            return {self.curvature_scores: encoded.curvature_scores, self.target_tx_ratios: encoded.target_tx_ratios}
        return {self.node_features: encoded.node_features}

    def _prepare_mpnn_edge_features(self, edge_features):
        """Validate the configured edge width and mask curvature in ablation mode."""
        edge_features = np.asarray(edge_features, dtype=np.float32)
        if edge_features.ndim != 2 or edge_features.shape[1] != self.mpnn_edge_input_dim:
            raise ValueError(
                "MPNN edge features must have shape [E, {}]".format(self.mpnn_edge_input_dim)
            )
        if not self.actor_curvature_enabled or not self.use_curvature_edge_feature:
            # Preserve every non-curvature attribute and hard-disable only the curvature column.
            edge_features = edge_features.copy()
            edge_features[:, self.mpnn_edge_curvature_index] = 0.0
        return edge_features

    def actor_batch_inputs(self, encoded_steps):
        def step_node_count(item):
            if self.actor_stage == 2 and self.residual_architecture == "mpnn":
                return item.node_context.shape[0]
            if self.actor_stage == 2:
                return item.residual_context.shape[0]
            if self.actor_stage == 1:
                return item.curvature_scores.shape[0]
            return item.node_features.shape[0]

        step_ids = np.concatenate([
            np.full(
                step_node_count(item),
                index, dtype=np.int32,
            )
            for index, item in enumerate(encoded_steps)
        ]) if encoded_steps else np.empty((0,), dtype=np.int32)
        number_of_steps = len(encoded_steps)
        if self.actor_stage == 2:
            if self.residual_architecture == "mpnn":
                node_counts = [item.node_context.shape[0] for item in encoded_steps]
                offsets = np.cumsum([0] + node_counts[:-1]).astype(np.int32)
                edge_indices = [item.edge_index + offset for item, offset in zip(encoded_steps, offsets)]
                edge_features = np.concatenate([item.edge_features for item in encoded_steps]).astype(np.float32, copy=False)
                edge_features = self._prepare_mpnn_edge_features(edge_features)
                batch = {"target_tx_ratios": np.concatenate([item.target_tx_ratios for item in encoded_steps]),
                        "node_context": np.concatenate([item.node_context for item in encoded_steps]),
                        "edge_index": np.concatenate(edge_indices, axis=1).astype(np.int32, copy=False),
                        "edge_features": edge_features,
                        "step_ids": step_ids, "number_of_steps": number_of_steps}
                if self.actor_curvature_enabled:
                    batch["curvature_scores"] = np.concatenate([item.curvature_scores for item in encoded_steps])
                return batch
            batch = {"target_tx_ratios": np.concatenate([item.target_tx_ratios for item in encoded_steps]),
                    "residual_context": np.concatenate([item.residual_context for item in encoded_steps]),
                    "step_ids": step_ids, "number_of_steps": number_of_steps}
            if self.actor_curvature_enabled:
                batch["curvature_scores"] = np.concatenate([item.curvature_scores for item in encoded_steps])
            return batch
        if self.actor_stage == 1:
            return {"curvature_scores": np.concatenate([item.curvature_scores for item in encoded_steps]),
                    "target_tx_ratios": np.concatenate([item.target_tx_ratios for item in encoded_steps])}
        return {"node_features": np.concatenate([item.node_features for item in encoded_steps])}

    def _actor_batch_feed(self, batch):
        if self.actor_stage == 2:
            if self.residual_architecture == "mpnn":
                edge_features = self._prepare_mpnn_edge_features(batch["edge_features"])
                feed = {self.target_tx_ratios: batch["target_tx_ratios"],
                        self.mpnn_node_context: batch["node_context"], self.mpnn_edge_features: edge_features,
                        self.mpnn_edge_receivers: batch["edge_index"][1]}
            else:
                feed = {self.target_tx_ratios: batch["target_tx_ratios"],
                        self.residual_context: batch["residual_context"]}
            if self.actor_curvature_enabled:
                feed[self.curvature_scores] = batch["curvature_scores"]
            return feed
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
        states = np.asarray(global_states, dtype=np.float32)
        if self.critic_architecture == "node_conditioned":
            states = states.reshape(-1, self.critic_input_dim)
            return self.session.run(self.values, feed_dict={self.critic_inputs: states})
        states = states.reshape(-1, GLOBAL_STATE_DIM)
        return self.session.run(self.values, feed_dict={self.global_states: states})

    def update(self, actor_batch: Mapping, critic_batch: Mapping, epochs=4):
        actor_feed = self._actor_batch_feed(actor_batch)
        actor_feed.update({
            self.actions: actor_batch["actions"],
            self.old_log_probabilities: actor_batch["old_log_probabilities"],
            self.advantages: actor_batch["advantages"],
        })
        if self.actor_stage == 2:
            actor_feed.update({
                self.actor_step_ids: actor_batch["step_ids"],
                self.number_of_steps: actor_batch["number_of_steps"],
            })
        if self.critic_architecture == "node_conditioned":
            critic_feed = {
                self.critic_inputs: critic_batch["critic_inputs"],
                self.returns: critic_batch["returns"],
            }
        else:
            critic_feed = {
                self.global_states: critic_batch["global_states"],
                self.returns: critic_batch["returns"],
            }
        for _ in range(int(epochs)):
            actor_loss, entropy, approx_kl, clip_fraction, common_loss, probability_loss, probability_excess, _ = self.session.run(
                [self.actor_loss, self.mean_entropy, self.approx_kl, self.clip_fraction,
                 self.common_mode_loss, self.probability_budget_loss,
                 self.probability_budget_excess, self.actor_train_op], feed_dict=actor_feed)
            critic_loss, _ = self.session.run([self.critic_loss, self.critic_train_op], feed_dict=critic_feed)
        return {"actor_loss": float(actor_loss), "critic_loss": float(critic_loss), "entropy": float(entropy),
                "approx_kl": float(approx_kl), "clip_fraction": float(clip_fraction),
                "common_mode_loss": float(common_loss),
                "probability_budget_loss": float(probability_loss),
                "probability_budget_excess": float(probability_excess)}

    def policy_gradient_norm(self, actor_batch: Mapping) -> float:
        feed = self._actor_batch_feed(actor_batch)
        feed.update({self.actions: actor_batch["actions"], self.old_log_probabilities: actor_batch["old_log_probabilities"], self.advantages: actor_batch["advantages"]})
        return float(self.session.run(self.policy_gradient_norm_op, feed_dict=feed))

    def common_mode_gradient_norm(self, actor_batch: Mapping) -> float:
        """Return the gradient norm contributed by the configured common-mode term."""
        if self.actor_stage != 2:
            return 0.0
        feed = self._actor_batch_feed(actor_batch)
        feed.update({self.actions: actor_batch["actions"], self.old_log_probabilities: actor_batch["old_log_probabilities"],
                     self.advantages: actor_batch["advantages"], self.actor_step_ids: actor_batch["step_ids"],
                     self.number_of_steps: actor_batch["number_of_steps"]})
        return float(self.session.run(self.common_mode_gradient_norm_op, feed_dict=feed))

    def total_actor_gradient_norm(self, actor_batch: Mapping) -> float:
        """Return the gradient norm of the complete Actor loss."""
        feed = self._actor_batch_feed(actor_batch)
        feed.update({self.actions: actor_batch["actions"], self.old_log_probabilities: actor_batch["old_log_probabilities"],
                     self.advantages: actor_batch["advantages"]})
        if self.actor_stage == 2:
            feed.update({self.actor_step_ids: actor_batch["step_ids"], self.number_of_steps: actor_batch["number_of_steps"]})
        return float(self.session.run(self.total_actor_gradient_norm_op, feed_dict=feed))

    def stage1_diagnostics(self, encoded):
        """Return alpha and q-base values for Stage-1 training logs and tests."""
        if self.actor_stage != 1:
            return {}
        values = self.session.run({"alpha_raw": self.alpha_raw, "alpha_kappa": self.alpha_kappa,
                                   "q_base": self.probabilities}, feed_dict=self._encoded_feed(encoded))
        return {key: np.asarray(value) for key, value in values.items()}

    def stage2_diagnostics(
        self, encoded, include_mpnn_messages=True, include_mpnn_attention=False
    ):
        """Return Stage-2 diagnostics, optionally excluding MPNN messages.

        The message tensors are useful for detailed analysis but require a
        second TensorFlow fetch.  Attention tensors are independently
        optional, so training can skip all detailed pooling diagnostics while
        retaining the base probability and residual statistics.
        """
        if self.actor_stage != 2:
            return {}
        values = self.session.run({
            "alpha_raw": self.alpha_raw, "alpha_kappa": self.alpha_kappa,
            "effective_alpha": self.effective_alpha,
            "q_base": self.base_probabilities, "delta": self.residual_delta,
            "q_final": self.probabilities, "residual_input": self.residual_input,
        }, feed_dict=self._encoded_feed(encoded))
        if self.residual_architecture == "mpnn":
            optional_fetches = {}
            if include_mpnn_messages:
                optional_fetches.update({
                    "edge_messages": self.mpnn_edge_messages,
                    "aggregated_messages": self.mpnn_aggregated_messages,
                })
            if include_mpnn_attention:
                optional_fetches.update({
                    "attention_weights": self.mpnn_attention_weights,
                    "attention_entropy": self.mpnn_attention_entropy,
                    "attention_effective_degree": self.mpnn_attention_effective_degree,
                    "attention_max_weight": self.mpnn_attention_max_weight,
                })
            if optional_fetches:
                values.update(self.session.run(
                    optional_fetches, feed_dict=self._encoded_feed(encoded)
                ))
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

    def restore_actor_only(self, checkpoint_prefix):
        """Restore only ``actor/*`` variables, leaving the new Critic random-initialized."""
        reader = self.tf.train.NewCheckpointReader(str(checkpoint_prefix))
        available = reader.get_variable_to_shape_map()
        with self.graph.as_default():
            variables = self.tf.get_collection(self.tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor")
        matched = {
            variable.name.split(":")[0]: variable for variable in variables
            if variable.name.split(":")[0] in available
            and tuple(variable.shape.as_list()) == tuple(available[variable.name.split(":")[0]])
        }
        if not matched:
            raise ValueError("checkpoint has no compatible actor variables")
        self.tf.train.Saver(var_list=matched).restore(self.session, str(checkpoint_prefix))
        return tuple(sorted(matched))

    def close(self):
        self.session.close()
