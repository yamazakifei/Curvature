"""Verify fixed validation is reproducible, paired, and inference-only."""

import copy
from pathlib import Path

import numpy as np
import yaml

from curvature_gossip.learning.ctde_ppo import CTDEPPO
from curvature_gossip.learning.validation import evaluate_fixed_validation, probability_statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_validation_config():
    with (PROJECT_ROOT / "configs" / "nn_ctde_v3_fixed_validation_smoke.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        raw = yaml.safe_load(stream)
    raw = copy.deepcopy(raw)
    raw["topology"]["params"].update({
        "n_nodes": 12, "area_width_m": 100.0, "area_height_m": 100.0,
        "communication_radius_m": 75.0,
    })
    raw["validation"]["scenarios"] = [
        dict(raw["validation"]["scenarios"][0], slots=6, n_nodes=12)
    ]
    return raw


def test_fixed_validation_is_reproducible_and_uses_matched_rate_random():
    raw = _tiny_validation_config()
    model = CTDEPPO(seed=123)
    try:
        before = model.variable_snapshot()
        summary_a, rows_a = evaluate_fixed_validation(model, raw, 0, "initial")
        after = model.variable_snapshot()
        summary_b, rows_b = evaluate_fixed_validation(model, raw, 0, "initial")
    finally:
        model.close()
    assert all(np.array_equal(old, new) for old, new in zip(before, after))
    assert summary_a == summary_b
    assert rows_a == rows_b
    assert len(rows_a) == 3
    nn_row = next(row for row in rows_a if row["policy"] == "nn")
    matched_row = next(row for row in rows_a if row["policy"] == "matched_random")
    assert matched_row["policy_probability"] == nn_row["action_prob_mean"]
    assert summary_a["n_scenarios"] == 1
    for name in (
        "action_prob_node_std_mean", "action_prob_global_std",
        "action_prob_per_node_mean_std",
    ):
        assert name in summary_a and np.isfinite(summary_a[name])


def test_probability_statistics_separates_node_and_global_variation():
    statistics = probability_statistics(np.array([[0.1, 0.3], [0.2, 0.4]], dtype=np.float32))
    assert np.isclose(statistics["action_prob_mean"], 0.25)
    assert np.isclose(statistics["action_prob_node_std_mean"], 0.1)
    assert statistics["action_prob_global_std"] > statistics["action_prob_node_std_mean"]
    assert np.isclose(statistics["action_prob_per_node_mean_std"], 0.1)
