"""Test the single-parameter Stage-1 curvature Actor invariants."""

import numpy as np

from curvature_gossip.learning.ctde_ppo import CTDEPPO
from curvature_gossip.learning.features import Stage1EncodedObservations


def _encoded(scores=(0.0, 0.5, 1.0), target=0.1):
    return Stage1EncodedObservations(
        curvature_scores=np.asarray(scores, dtype=np.float32).reshape(-1, 1),
        target_tx_ratios=np.full(len(scores), target, dtype=np.float32),
    )


def _actor_config(override=None):
    return {"stage": 1, "curvature": {
        "score": "incident_bottleneck_max", "center": 0.0,
        "alpha_parameterization": "softplus", "alpha_init": 0.1,
        "alpha_override": override,
    }, "residual": {"enabled": False}}


def test_stage1_has_only_alpha_raw_and_monotone_probabilities():
    """The Stage-1 actor must be a shared one-parameter monotone policy."""
    model = CTDEPPO(seed=9, actor_config=_actor_config())
    try:
        with model.graph.as_default():
            names = [item.name for item in model.tf.trainable_variables() if item.name.startswith("actor/")]
        assert names == ["actor/alpha_raw:0"]
        probabilities = model.predict_probabilities(_encoded())
        assert np.all(np.diff(probabilities) >= 0.0)
    finally:
        model.close()


def test_alpha_override_zero_strictly_returns_target_rate():
    """The explicit override must bypass softplus(alpha_raw) exactly."""
    model = CTDEPPO(seed=9, actor_config=_actor_config(0.0))
    try:
        assert np.allclose(model.predict_probabilities(_encoded()), 0.1, atol=1e-7)
    finally:
        model.close()


def test_alpha_override_disables_alpha_raw_updates():
    """An override is an ablation switch, not a trainable near-zero alpha."""
    model = CTDEPPO(seed=9, actor_config=_actor_config(0.0))
    try:
        encoded = _encoded()
        batch = model.actor_batch_inputs([encoded])
        batch.update({
            "actions": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
            "old_log_probabilities": np.zeros(3, dtype=np.float32),
            "advantages": np.ones(3, dtype=np.float32),
        })
        before = model.stage1_diagnostics(encoded)["alpha_raw"]
        model.update(batch, {"global_states": np.zeros((1, 8), dtype=np.float32), "returns": np.zeros(1, dtype=np.float32)}, epochs=1)
        after = model.stage1_diagnostics(encoded)["alpha_raw"]
        assert before == after
    finally:
        model.close()
