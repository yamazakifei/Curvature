"""验证历史曲率 CSV 可按原始配置补写瓶颈重要度。"""

import csv
from pathlib import Path

import yaml

from curvature_gossip.experiments import (
    annotate_bottleneck_importance, run_experiment,
)


def test_annotation_restores_bottleneck_importance_for_saved_run(tmp_path):
    """删除导出列后，补写结果应与全局归一化范围一致。"""
    source_path = Path(__file__).parents[1] / "configs" / "smoke_test.yaml"
    with source_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["experiment"]["slots"] = 30
    config["experiment"]["warmup_slots"] = 5
    config["output"]["root"] = str(tmp_path / "results")
    config_path = tmp_path / "smoke.yaml"
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    result = run_experiment(str(config_path), progress=False)
    seed = 1
    run_directory = result.experiment_directory / str(seed) / "orc_freshness"
    path = run_directory / "edge_curvature.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        del row["bottleneck_importance"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["node_i", "node_j", "curvature"])
        writer.writeheader()
        writer.writerows(rows)

    annotate_bottleneck_importance(str(run_directory))

    with path.open(newline="", encoding="utf-8") as stream:
        restored = list(csv.DictReader(stream))
    assert "bottleneck_importance" in restored[0]
    assert all(0.0 <= float(row["bottleneck_importance"]) <= 1.0 for row in restored)
