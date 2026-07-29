"""Verify the Stage-2 residual actor preserves and explicitly reads its Stage-1 base."""

import numpy as np

from curvature_gossip.learning.ctde_ppo import CTDEPPO
from curvature_gossip.learning.features import Stage2EncodedObservations


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
