"""Verify the Stage-2 residual actor preserves and explicitly reads its Stage-1 base."""

import numpy as np

from curvature_gossip.learning.ctde_ppo import CTDEPPO
from curvature_gossip.learning.features import Stage2EncodedObservations
from curvature_gossip.learning.trainer import _stage1_actor_config


def _encoded():
    return Stage2EncodedObservations(
        curvature_scores=np.asarray([[0.1], [0.5], [0.9]], dtype=np.float32),
        target_tx_ratios=np.full(3, 0.1, dtype=np.float32),
        residual_context=np.asarray([[0.1] * 8, [0.2] * 8, [0.3] * 8], dtype=np.float32),
    )


def _actor():
    return {
        "stage": 2,
        "curvature": {"center": 0.4, "alpha_init": 1.5, "alpha_override": 1.5},
        "residual": {
            "enabled": True, "hidden_dims": [64, 64], "delta_max": 1.0,
            "include_scenario_context": False, "freeze_stage1": True,
        },
    }


def _no_curvature_actor():
    """Build the strict ablation: a uniform logit anchor plus local residual context."""
    return {
        "stage": 2,
        "curvature": {"enabled": False, "center": 0.4, "alpha_init": 1.5, "alpha_override": 0.0},
        "residual": {
            "enabled": True, "hidden_dims": [64, 64], "delta_max": 1.0,
            "include_scenario_context": False, "freeze_stage1": True,
            "use_stage1_reference": False,
        },
    }


def test_stage2_zero_initialized_residual_equals_fixed_stage1_base_and_reads_q_base():
    model = CTDEPPO(seed=17, actor_config=_actor())
    try:
        diagnostics = model.stage2_diagnostics(_encoded())
        assert diagnostics["residual_input"].shape == (3, 9)
        assert np.allclose(diagnostics["residual_input"][:, -1], diagnostics["q_base"])
        assert np.allclose(diagnostics["delta"], 0.0, atol=1e-7)
        assert np.allclose(diagnostics["q_final"], diagnostics["q_base"], atol=1e-7)
        assert np.all(np.diff(diagnostics["q_base"]) > 0.0)
    finally:
        model.close()


def test_stage2_fixed_alpha_stays_fixed_while_residual_output_learns():
    model = CTDEPPO(seed=19, actor_config=_actor())
    encoded = _encoded()
    try:
        batch = model.actor_batch_inputs([encoded])
        batch.update({
            "actions": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
            "old_log_probabilities": np.log(np.asarray([0.9, 0.1, 0.1], dtype=np.float32)),
            "advantages": np.asarray([-1.0, 1.0, 1.0], dtype=np.float32),
        })
        critic = {"global_states": np.zeros((1, 8), dtype=np.float32), "returns": np.zeros(1, dtype=np.float32)}
        before = model.stage2_diagnostics(encoded)
        model.update(batch, critic, epochs=4)
        after = model.stage2_diagnostics(encoded)
        assert np.isclose(before["alpha_raw"], after["alpha_raw"])
        assert not np.allclose(after["delta"], 0.0)
    finally:
        model.close()


def test_bmax_feature_does_not_change_fixed_stage1_base_probability():
    actor = _actor()
    actor["curvature"]["base_tx_ratio"] = 0.1
    model = CTDEPPO(seed=21, actor_config=actor)
    try:
        first = _encoded()
        second = Stage2EncodedObservations(
            curvature_scores=first.curvature_scores,
            target_tx_ratios=np.full(3, 0.2, dtype=np.float32),
            residual_context=first.residual_context,
        )
        assert np.allclose(
            model.stage2_diagnostics(first)["q_base"],
            model.stage2_diagnostics(second)["q_base"],
        )
        actor["curvature"]["calibrated_intercept"] = float(np.log(0.2 / 0.8))
        other = CTDEPPO(seed=22, actor_config=actor)
        try:
            assert not np.allclose(
                model.stage2_diagnostics(first)["q_base"],
                other.stage2_diagnostics(first)["q_base"],
            )
        finally:
            other.close()
    finally:
        model.close()


def test_no_curvature_stage2_has_eight_context_inputs_and_ignores_curvature_scores():
    model = CTDEPPO(seed=31, actor_config=_no_curvature_actor())
    try:
        encoded = _encoded()
        changed_scores = Stage2EncodedObservations(
            curvature_scores=np.ones((3, 1), dtype=np.float32),
            target_tx_ratios=encoded.target_tx_ratios,
            residual_context=encoded.residual_context,
        )
        before = model.stage2_diagnostics(encoded)
        after = model.stage2_diagnostics(changed_scores)
        assert before["residual_input"].shape == (3, 8)
        assert np.allclose(before["q_base"], 0.1, atol=1e-6)
        assert np.allclose(before["q_final"], after["q_final"], atol=1e-6)
        assert np.isclose(before["effective_alpha"], 0.0)
    finally:
        model.close()


def test_no_curvature_config_records_a_distinct_architecture_and_rejects_leakage():
    raw = {"actor": _no_curvature_actor()}
    actor = _stage1_actor_config(raw)
    assert actor["architecture_version"] == "no_curvature_stage2_residual_v1"
    assert actor["curvature_enabled"] is False
    assert actor["input_feature_names"] == [
        "normalized_degree", "time_since_last_tx", "consecutive_tx_attempts",
        "congestion_ewma", "self_information_increment", "neighbor_freshness_mean",
        "neighbor_freshness_max", "neighbor_confidence_mean",
    ]

    invalid = _no_curvature_actor()
    invalid["curvature"]["alpha_override"] = 1.5
    try:
        _stage1_actor_config({"actor": invalid})
        assert False, "curvature must not remain active in no-reference mode"
    except ValueError as error:
        assert "alpha_override=0.0" in str(error)


def test_stage2_accepts_stage1_checkpoint_while_leaving_new_residual_initialized(tmp_path):
    stage1 = CTDEPPO(seed=23, actor_config={
        "stage": 1, "curvature": {"center": 0.4, "alpha_init": 1.5, "alpha_override": 1.5},
    })
    checkpoint = tmp_path / "stage1" / "model"
    try:
        stage1.save(str(checkpoint))
    finally:
        stage1.close()
    stage2 = CTDEPPO(seed=29, actor_config=_actor())
    try:
        restored = stage2.restore_compatible(str(checkpoint))
        diagnostics = stage2.stage2_diagnostics(_encoded())
        assert "critic/value/kernel" in restored
        assert np.allclose(diagnostics["delta"], 0.0, atol=1e-7)
    finally:
        stage2.close()
