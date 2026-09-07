"""Test Stage-2 common-mode regularization and bias-free residual heads."""

import numpy as np

from curvature_gossip.learning.ctde_ppo import CTDEPPO
from curvature_gossip.learning.features import Stage2EncodedObservations, Stage2MPNNEncodedObservations


def _mlp_actor():
    return {
        "stage": 2,
        "curvature": {"center": 0.4, "alpha_init": 1.5, "alpha_override": 1.5},
        "residual": {"enabled": True, "hidden_dims": [64, 64], "delta_max": 1.0,
                      "include_scenario_context": False, "freeze_stage1": True},
    }


def _mpnn_actor():
    actor = _mlp_actor()
    actor["residual"].update({
        "architecture": "mpnn", "use_stage1_reference": True,
        "mpnn": {"message_hidden_dims": [32], "message_dim": 16,
                 "update_hidden_dims": [64, 64], "aggregation": "mean_max",
                 "condition_message_on_receiver": True, "use_sender_node_features": False,
                 "use_curvature_edge_feature": True, "bmax": 1.0},
    })
    return actor


def _mlp_encoded():
    return Stage2EncodedObservations(
        curvature_scores=np.asarray([[0.1], [0.2]], dtype=np.float32),
        target_tx_ratios=np.asarray([0.1, 0.1], dtype=np.float32),
        residual_context=np.zeros((2, 8), dtype=np.float32),
    )


def test_segment_common_mode_loss_respects_each_rollout_step():
    """Opposite residual means in different steps must not cancel globally."""
    model = CTDEPPO(seed=101, actor_config=_mlp_actor(), common_mode_coefficient=0.05)
    try:
        tf = model.tf
        with model.graph.as_default():
            residual = tf.placeholder(tf.float32, [None], name="test_residual")
            step_ids = tf.placeholder(tf.int32, [None], name="test_step_ids")
            steps = tf.placeholder(tf.int32, shape=(), name="test_steps")
            sums = tf.math.unsorted_segment_sum(residual, step_ids, steps)
            counts = tf.math.unsorted_segment_sum(tf.ones_like(residual), step_ids, steps)
            loss = tf.reduce_mean(tf.square(sums / tf.maximum(counts, 1.0)))
        value = model.session.run(
            loss,
            feed_dict={residual: [0.2, 0.2, -0.2, -0.2], step_ids: [0, 0, 1, 1], steps: 2},
        )
        assert np.isclose(value, 0.04)
        same_step = model.session.run(
            loss,
            feed_dict={residual: [0.2, -0.2], step_ids: [0, 0], steps: 1},
        )
        assert np.isclose(same_step, 0.0)
    finally:
        model.close()


def test_stage2_residual_heads_have_no_output_bias_and_batch_has_training_step_ids():
    for seed, actor, encoded in (
        (102, _mlp_actor(), _mlp_encoded()),
        (103, _mpnn_actor(), Stage2MPNNEncodedObservations(
            curvature_scores=np.asarray([[0.1], [0.2]], dtype=np.float32),
            target_tx_ratios=np.asarray([0.1, 0.1], dtype=np.float32),
            node_context=np.zeros((2, 5), dtype=np.float32),
            edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int32),
            edge_features=np.ones((2, 3), dtype=np.float32),
        )),
    ):
        model = CTDEPPO(seed=seed, actor_config=actor, common_mode_coefficient=0.05)
        try:
            with model.graph.as_default():
                names = [variable.name for variable in model.tf.trainable_variables()]
            assert not any(name.endswith("delta_logit/bias:0") for name in names)
            batch = model.actor_batch_inputs([encoded, encoded])
            assert np.array_equal(batch["step_ids"], [0, 0, 1, 1])
            assert batch["number_of_steps"] == 2
            assert np.allclose(model.stage2_diagnostics(encoded)["delta"], 0.0)
        finally:
            model.close()


def test_zero_coefficient_preserves_old_actor_loss_and_penalty_gradient_reaches_residual():
    encoded = _mlp_encoded()
    zero_model = CTDEPPO(seed=104, actor_config=_mlp_actor(), common_mode_coefficient=0.0)
    try:
        batch = zero_model.actor_batch_inputs([encoded])
        batch.update({
            "actions": np.asarray([0.0, 1.0], dtype=np.float32),
            "old_log_probabilities": np.log(np.asarray([0.1, 0.1], dtype=np.float32)),
            "advantages": np.asarray([1.0, -1.0], dtype=np.float32),
        })
        feed = zero_model._actor_batch_feed(batch)
        feed.update({
            zero_model.actions: batch["actions"],
            zero_model.old_log_probabilities: batch["old_log_probabilities"],
            zero_model.advantages: batch["advantages"],
            zero_model.actor_step_ids: batch["step_ids"],
            zero_model.number_of_steps: batch["number_of_steps"],
        })
        loss, surrogate, entropy = zero_model.session.run(
            [zero_model.actor_loss, zero_model.surrogate_actor_loss, zero_model.mean_entropy],
            feed_dict=feed,
        )
        assert np.isclose(loss, surrogate - 0.01 * entropy)
    finally:
        zero_model.close()

    model = CTDEPPO(seed=105, actor_config=_mlp_actor(), common_mode_coefficient=0.05)
    try:
        with model.graph.as_default():
            output_kernel = next(
                variable for variable in model.tf.trainable_variables()
                if variable.name == "actor/residual_mlp/delta_logit/kernel:0"
            )
            model.session.run(model.tf.assign(output_kernel, np.full((64, 1), 0.01, dtype=np.float32)))
        batch = model.actor_batch_inputs([encoded])
        batch.update({
            "actions": np.asarray([0.0, 1.0], dtype=np.float32),
            "old_log_probabilities": np.log(np.asarray([0.1, 0.1], dtype=np.float32)),
            "advantages": np.asarray([1.0, -1.0], dtype=np.float32),
        })
        assert model.common_mode_gradient_norm(batch) > 0.0
        losses = model.update(
            batch,
            {"global_states": np.zeros((2, 8), dtype=np.float32), "returns": np.zeros(2, dtype=np.float32)},
            epochs=1,
        )
        assert np.isfinite(losses["actor_loss"])
        assert np.isfinite(losses["common_mode_loss"])
    finally:
        model.close()


def test_mean_probability_budget_penalty_is_one_sided_and_normalized_by_bmax():
    """The Stage-2 loss penalizes a rollout mean only when it exceeds Bmax."""
    actor = _mlp_actor()
    actor["curvature"].update({"base_tx_ratio": 0.5, "alpha_override": 0.0})
    model = CTDEPPO(
        seed=106, actor_config=actor, probability_budget_coefficient=1.0
    )
    try:
        batch = model.actor_batch_inputs([_mlp_encoded()])
        feed = model._actor_batch_feed(batch)
        feed.update({
            model.actor_step_ids: batch["step_ids"],
            model.number_of_steps: batch["number_of_steps"],
        })
        loss, excess = model.session.run(
            [model.probability_budget_loss, model.probability_budget_excess],
            feed_dict=feed,
        )
        # (0.5 - 0.1) / 0.1 = 4, so the normalized squared penalty is 16.
        assert np.isclose(excess, 4.0)
        assert np.isclose(loss, 16.0)
    finally:
        model.close()
