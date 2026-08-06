"""验证 V3.2 Stage-1 的中心模式、粗细筛选、缓存与运行时审计输出。"""

import copy
import json
from pathlib import Path

import yaml

from curvature_gossip.learning.stage1_search import resolve_center_mode, run_stage1_search


def _small_search_config():
    config_path = Path(__file__).parents[1] / "configs" / "GNN" / "mpnnV3.2_ch1_search.yaml"
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw = copy.deepcopy(raw)
    raw["experiment"]["slots"] = 4
    raw["topology"] = {
        "type": "ring",
        "params": {"n_nodes": 12, "circle_radius_m": 1.0},
    }
    search = raw["actor"]["curvature"]["base_search"]
    search["calibration"].update({"topology_count": 1, "intercept_max_iterations": 30})
    search["base_candidates"].update({"coarse_count": 3, "refine": False})
    search["alpha_candidates"] = [1.0]
    search["evaluation"].update({
        "topology_count": 2,
        "dynamic_repeats": 1,
        "coarse_topology_count": 1,
        "coarse_dynamic_repeats": 1,
        "coarse_slots": 2,
        "full_slots": 4,
    })
    search["evaluation"]["candidate_selection"]["coarse_keep_top_k_per_alpha"] = 1
    search["evaluation"]["parallel"] = {
        "enabled": False, "backend": "process", "workers": 1,
        "chunksize": 1, "deterministic_order": True,
    }
    search["evaluation"]["runtime_profile"] = {
        "enabled": True, "output_file": "stage1_search_runtime.json",
    }
    return raw


def test_center_mode_keeps_legacy_resolution_and_v32_none():
    assert resolve_center_mode({"center_mode": "none", "center": 0.0}) == "none"
    assert resolve_center_mode({"center": 0.4}) == "fixed"
    assert resolve_center_mode({"center": "auto"}) == "calibration_mean"


def test_v32_search_prunes_coarse_candidates_and_writes_runtime(tmp_path):
    result = run_stage1_search(_small_search_config(), tmp_path)

    assert result["center"] == 0.0
    assert result["search_summary"]["selected_center_mode"] == "none"
    with (tmp_path / "stage1_search_runtime.json").open(encoding="utf-8") as stream:
        runtime = json.load(stream)
    assert runtime["coarse_candidate_count"] == 3
    assert runtime["full_candidate_count"] == 1
    assert runtime["coarse_slots"] == 2
    assert runtime["full_slots"] == 4

    rows = (tmp_path / "stage1_search_results.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 5  # header + three coarse rows + one promoted full row
    assert "coarse_rank_within_alpha" in rows[0]
    assert "evaluated_on_full" in rows[0]
    assert "pruned_after_coarse" in rows[0]
    assert "evaluation_phase" in rows[0]


def test_v32_process_parallel_path_preserves_candidate_results(tmp_path):
    raw = _small_search_config()
    raw["actor"]["curvature"]["base_search"]["evaluation"]["candidate_selection"][
        "coarse_keep_top_k_per_alpha"
    ] = 3
    raw["actor"]["curvature"]["base_search"]["evaluation"]["parallel"].update({
        "enabled": True, "workers": 2,
    })
    result = run_stage1_search(raw, tmp_path)
    assert result["search_summary"]["selected_center"] == 0.0
    with (tmp_path / "stage1_search_runtime.json").open(encoding="utf-8") as stream:
        runtime = json.load(stream)
    assert runtime["parallel_enabled"] is True
    assert runtime["full_candidate_count"] == 3
