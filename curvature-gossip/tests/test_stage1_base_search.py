"""验证 Stage-1 b 候选生成、alpha 独立细化和 beta0 pooled 校准。"""

import numpy as np
from pathlib import Path

from curvature_gossip.learning.stage1_search import (
    calibrate_intercept,
    coarse_base_candidates,
    refine_base_candidates,
    _print_selected_summary,
)


def test_default_coarse_candidates_are_bounded_and_deterministic():
    values = coarse_base_candidates(0.2, {
        "min_fraction_of_max": 0.25,
        "max_fraction_of_max": 1.0,
        "coarse_count": 7,
    })
    assert np.allclose(values, [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    assert list(values) == sorted(set(values))
    assert all(0.0 < value <= 0.2 for value in values)


def test_refine_respects_left_and_right_boundaries():
    coarse = (0.05, 0.1, 0.15, 0.2)
    assert min(refine_base_candidates(coarse, 0, 0.2, 5)) >= 0.0
    assert max(refine_base_candidates(coarse, 3, 0.2, 5)) <= 0.2 + 1e-12
    assert min(refine_base_candidates(coarse, 1, 0.2, 5)) >= 0.05
    assert max(refine_base_candidates(coarse, 1, 0.2, 5)) <= 0.15


def test_beta0_is_one_pool_constant_and_hits_target_mean():
    scores = np.asarray([0.0, 0.2, 0.5, 0.9, 1.0] * 4)
    result = calibrate_intercept(scores, 0.1, 1.5, float(np.mean(scores)), tolerance=1e-8)
    assert abs(result["calibration_probability_mean"] - 0.1) <= 1e-8
    assert np.isfinite(result["calibrated_intercept"])
    assert result["converged"]


def test_search_completion_prints_selected_configuration(capsys):
    _print_selected_summary({
        "base_tx_ratio": 0.1,
        "alpha": 1.5,
        "calibrated_intercept": -2.0,
        "center": 0.4,
        "search_probability_mean": 0.1,
        "actual_tx_ratio": 0.098,
        "mean_VAoI": 12.3,
        "constraint_feasible": True,
    }, 0.2, Path("search-output"))
    output = capsys.readouterr().out
    assert "搜索完成" in output
    assert "b=0.100000" in output
    assert "alpha=1.500000" in output
    assert "beta0=-2.000000" in output
