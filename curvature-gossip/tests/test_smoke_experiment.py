"""运行三种启发式策略的快速端到端实验，并验证结果文件和六类图片。"""

import csv
from pathlib import Path

import yaml

from curvature_gossip.experiments import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_experiment_runs_all_policies_and_saves_outputs(tmp_path):
    with (PROJECT_ROOT / "configs" / "smoke_test.yaml").open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["experiment"]["slots"] = 30
    config["experiment"]["warmup_slots"] = 5
    config["output"]["root"] = str(tmp_path / "results")
    config["output"]["node_diagnostics_stride"] = 7
    config_path = tmp_path / "smoke.yaml"
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    result = run_experiment(str(config_path), progress=False)
    assert {row["policy"] for row in result.run_summaries} == {
        "random", "freshness_only", "orc_freshness"
    }
    for policy in ("random", "freshness_only", "orc_freshness"):
        topology_seed = config["experiment"]["topology_seeds"][0]
        run_directory = result.experiment_directory / str(topology_seed) / policy
        for filename in (
            "resolved_config.yaml", "summary.json", "per_slot.csv",
            "dissemination_delays.csv", "edge_curvature.csv",
            "topology_nodes.csv", "topology_edges.csv",
        ):
            assert (run_directory / filename).is_file()
        with (run_directory / "edge_curvature.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert "bottleneck_importance" in rows[0]
        assert all(0.0 <= float(row["bottleneck_importance"]) <= 1.0 for row in rows)

    diagnostic_path = (
        result.experiment_directory / str(topology_seed) / "node_policy_diagnostics.csv"
    )
    with diagnostic_path.open(newline="", encoding="utf-8") as stream:
        diagnostic_rows = list(csv.DictReader(stream))
    assert {row["policy"] for row in diagnostic_rows} == {
        "freshness_only", "orc_freshness"
    }
    assert {int(row["slot"]) for row in diagnostic_rows} == {5, 12, 19, 26}
    assert set(diagnostic_rows[0]) == {
        "topology_seed", "channel_seed", "update_seed", "policy", "slot", "node",
        "fresh_score", "orc_score", "base_probability", "backoff_factor",
        "final_probability",
    }
    expected_plots = {
        "topology_curvature.png", "curvature_histogram.png", "max_vaoi_trace.png",
        "max_vaoi_ecdf.png", "policy_vaoi_comparison.png", "vaoi_vs_transmissions.png",
    }
    actual_plots = {path.name for path in (result.experiment_directory / "plots").glob("*.png")}
    assert expected_plots <= actual_plots
