"""Test fixed-alpha grid summary calculations without running TensorFlow."""

import numpy as np

from curvature_gossip.learning.fixed_alpha_grid import _summary_row


def test_fixed_alpha_summary_includes_probability_variance_and_baseline_improvements():
    summary = {
        "n_scenarios": 2,
        "nn_mean_VAoI": 8.0,
        "matched_random_mean_VAoI": 10.0,
        "fixed_random_mean_VAoI": 12.0,
        "nn_mean_action_probability": 0.11,
        "nn_actual_tx_ratio": 0.12,
        "action_prob_min": 0.03,
        "action_prob_max": 0.25,
        "action_prob_global_std": 0.04,
        "action_prob_per_node_mean_std": 0.03,
        "action_prob_node_std_mean": 0.02,
        "delta_vaoi_matched_mean": -2.0,
        "delta_vaoi_matched_ci95_low": -3.0,
        "delta_vaoi_matched_ci95_high": -1.0,
        "nn_beats_matched_fraction": 1.0,
    }
    row = _summary_row(summary, 0.2, 0.5, 0.4)
    assert np.isclose(row["improvement_vs_matched_random_pct"], 20.0)
    assert np.isclose(row["improvement_vs_fixed_random_pct"], 100.0 / 3.0)
    assert np.isclose(row["probability_global_variance"], 0.04 ** 2)
    assert np.isclose(row["node_mean_probability_variance"], 0.03 ** 2)
