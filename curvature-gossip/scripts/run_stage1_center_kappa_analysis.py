"""分析 Stage-1 固定曲率中心与 N=50--150 测试拓扑之间的偏差。

功能：复用已有的 N=50--150、每个节点规模 5 个场景的测试配置，统计
节点级 c_kappa 与训练阶段固定中心的差异；随后在 N=150 的同一批场景上，
只把第一层的固定中心替换为 N=150 测试集中心，配对比较最终 VAoI。
脚本不会重新训练模型，也不会修改原始模型目录；所有统计、图表、复现实验
配置和元数据写入 result_GNN/0804Stage1_Center_kappa_analysis/。
"""

from __future__ import division

import argparse
import csv
import json
import math
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result_GNN"
SCALABILITY_ROOT = RESULT_ROOT / "0804Scalability_N"
OUTPUT_ROOT = RESULT_ROOT / "0804Stage1_Center_kappa_analysis"
CONFIG_ROOT = OUTPUT_ROOT / "configs"

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvature_gossip.curvature import bottleneck_importance
from curvature_gossip.experiments.runner import _curvature_provider
from curvature_gossip.learning.ctde_ppo import CTDEPPO, resolve_learning_rates
from curvature_gossip.learning.features import (
    encode_observations,
    encode_stage1_observations,
    encode_stage2_mpnn_observations,
    encode_stage2_observations,
)
from curvature_gossip.learning.trainer import _critic_config, _stage1_actor_config
from curvature_gossip.learning.validation import (
    _build_simulator,
    _run_neural_scenario,
    validation_scenarios,
)
from curvature_gossip.random_streams import make_rng
from curvature_gossip.topology import get_topology_generator


MODEL_DIRS = {
    "community": {
        "curvature": RESULT_ROOT / "mpnnV3_heuristic_channel_n100_u0.20_b0.10_a1.5_lowLR",
    },
    "random": {
        "curvature": RESULT_ROOT / "mpnnV3_random_n100_u0.20_b0.10_a1.5_lowLR",
    },
}

TOPOLOGY_LABELS = {
    "community": "Community topology",
    "random": "Random-geometric topology",
}

TOPOLOGY_NAMES = ("community", "random")
NODE_COUNTS = list(range(50, 151, 10))
REFERENCE_NODE_COUNT = 100
N150 = 150
SCENARIOS_PER_NODE_COUNT = 5
DEFAULT_SLOTS = 200

FONT_SIZES = {
    "title": 17,
    "axis_label": 15,
    "tick": 13,
    "legend": 12,
    "annotation": 11,
}


def read_yaml(path):
    """Read one YAML mapping and include the path in validation errors."""
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("expected a YAML mapping: {}".format(path))
    return value


def checkpoint_path(model_dir):
    """Return the persisted best-validation checkpoint prefix."""
    path = model_dir / "checkpoints" / "best_validation" / "model"
    if not path.with_suffix(".index").is_file():
        raise FileNotFoundError("missing best-validation checkpoint: {}".format(path))
    return path


def checkpoint_episode(model_dir):
    """Read checkpoint provenance without requiring the training process."""
    path = model_dir / "checkpoints" / "best_validation" / "checkpoint_info.json"
    if not path.is_file():
        return -1
    with path.open("r", encoding="utf-8") as stream:
        return int(json.load(stream).get("episode", -1))


def write_csv(path, rows):
    """Write a nonempty list of dictionaries as UTF-8 CSV."""
    if not rows:
        raise ValueError("cannot write empty CSV: {}".format(path))
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std_ci(values):
    """Return mean, sample standard deviation, and normal-approximation 95% CI."""
    values = np.asarray(tuple(values), dtype=float)
    if values.size == 0:
        raise ValueError("values must not be empty")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, std, ci95


def training_center(topology_name):
    """Read the frozen scalar center saved with the curvature checkpoint."""
    raw = read_yaml(MODEL_DIRS[topology_name]["curvature"] / "training_config.yaml")
    center = raw.get("actor", {}).get("curvature", {}).get("center")
    if center is None or not np.isfinite(float(center)):
        raise ValueError("missing finite actor.curvature.center for {}".format(topology_name))
    return float(center)


def config_path(topology_name, node_count):
    """Locate the exact configuration used by the earlier scalability sweep."""
    path = SCALABILITY_ROOT / "configs" / topology_name / "n{:03d}.yaml".format(node_count)
    if not path.is_file():
        raise FileNotFoundError("missing scalability config: {}".format(path))
    return path


def copy_existing_configs():
    """Copy the already-used N-sweep YAMLs into this analysis directory."""
    copied = {}
    for topology_name in TOPOLOGY_NAMES:
        target_dir = CONFIG_ROOT / topology_name
        target_dir.mkdir(parents=True, exist_ok=True)
        copied[topology_name] = []
        for node_count in NODE_COUNTS:
            source = config_path(topology_name, node_count)
            target = target_dir / source.name
            shutil.copy2(str(source), str(target))
            copied[topology_name].append(str(target))
    return copied


def scenario_topology(raw, scenario):
    """Generate exactly the topology used by one existing validation scenario."""
    generator = get_topology_generator(raw["topology"]["type"])
    params = dict(raw["topology"].get("params", {}))
    params["n_nodes"] = int(scenario.n_nodes)
    if "cluster_sizes" in params:
        params["cluster_sizes"] = [scenario.n_nodes // 2, scenario.n_nodes - scenario.n_nodes // 2]
    master_seed = int(raw.get("experiment", {}).get("master_seed", 0))
    return generator.generate(
        make_rng(master_seed, "fixed_validation", "topology", scenario.topology_seed),
        params,
    )


def scenario_curvature_scores(raw, scenario):
    """Compute node-level c_kappa=max incident bottleneck importance for one topology."""
    topology = scenario_topology(raw, scenario)
    curvature_config = raw.get("curvature", {})
    curvature = _curvature_provider(curvature_config).compute(topology)
    importance = bottleneck_importance(
        topology,
        curvature,
        curvature_config.get("normalization", "local_degree_bound"),
    )
    scores = np.asarray(
        [
            max(
                [importance[tuple(sorted((node, neighbor)))] for neighbor in topology.graph.neighbors(node)]
                or [0.0]
            )
            for node in topology.graph.nodes
        ],
        dtype=float,
    )
    if scores.shape != (scenario.n_nodes,) or not np.isfinite(scores).all():
        raise ValueError("invalid c_kappa scores for {}".format(scenario.scenario_id))
    return scores


def collect_center_statistics():
    """Collect per-scenario c_kappa distributions and aggregate by node count."""
    per_scenario = []
    for topology_name in TOPOLOGY_NAMES:
        fixed_center = training_center(topology_name)
        for node_count in NODE_COUNTS:
            raw = read_yaml(config_path(topology_name, node_count))
            for scenario in validation_scenarios(raw):
                scores = scenario_curvature_scores(raw, scenario)
                per_scenario.append(
                    {
                        "topology": topology_name,
                        "topology_label": TOPOLOGY_LABELS[topology_name],
                        "node_count": int(node_count),
                        "scenario_id": scenario.scenario_id,
                        "n_nodes": int(scores.size),
                        "fixed_training_center": fixed_center,
                        "mean_c_kappa": float(np.mean(scores)),
                        "std_c_kappa_nodes": float(np.std(scores)),
                        "median_c_kappa": float(np.median(scores)),
                        "p05_c_kappa": float(np.percentile(scores, 5)),
                        "p95_c_kappa": float(np.percentile(scores, 95)),
                        "mean_delta_c_kappa": float(np.mean(scores - fixed_center)),
                        "mean_abs_delta_c_kappa": float(np.mean(np.abs(scores - fixed_center))),
                        "rmse_delta_c_kappa": float(np.sqrt(np.mean((scores - fixed_center) ** 2))),
                    }
                )

    summary = []
    for topology_name in TOPOLOGY_NAMES:
        for node_count in NODE_COUNTS:
            rows = [
                row
                for row in per_scenario
                if row["topology"] == topology_name and row["node_count"] == node_count
            ]
            means = [row["mean_c_kappa"] for row in rows]
            deltas = [row["mean_delta_c_kappa"] for row in rows]
            mean_c, std_c, ci_c = mean_std_ci(means)
            delta, std_delta, ci_delta = mean_std_ci(deltas)
            summary.append(
                {
                    "topology": topology_name,
                    "topology_label": TOPOLOGY_LABELS[topology_name],
                    "node_count": int(node_count),
                    "n_scenarios": len(rows),
                    "fixed_training_center": training_center(topology_name),
                    "mean_test_c_kappa": mean_c,
                    "std_test_c_kappa": std_c,
                    "ci95_test_c_kappa": ci_c,
                    "mean_center_delta": delta,
                    "std_center_delta": std_delta,
                    "ci95_center_delta": ci_delta,
                    "mean_abs_delta_c_kappa": float(np.mean([row["mean_abs_delta_c_kappa"] for row in rows])),
                    "mean_rmse_delta_c_kappa": float(np.mean([row["rmse_delta_c_kappa"] for row in rows])),
                }
            )
    return per_scenario, summary


def plot_center_statistics(summary):
    """Plot test-set mean c_kappa and the frozen training center versus N."""
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=False)
    for axis, topology_name in zip(axes, TOPOLOGY_NAMES):
        rows = sorted(
            [row for row in summary if row["topology"] == topology_name],
            key=lambda row: row["node_count"],
        )
        x = np.asarray([row["node_count"] for row in rows], dtype=float)
        mean = np.asarray([row["mean_test_c_kappa"] for row in rows], dtype=float)
        ci95 = np.asarray([row["ci95_test_c_kappa"] for row in rows], dtype=float)
        center = float(rows[0]["fixed_training_center"])
        axis.plot(x, mean, marker="o", linewidth=2.4, color="#2F6B9A", label="Test mean c_kappa")
        axis.fill_between(x, mean - ci95, mean + ci95, color="#2F6B9A", alpha=0.16, label="95% CI")
        axis.axhline(center, color="#D97732", linestyle="--", linewidth=2.0, label="Fixed training center")
        axis.set_title(TOPOLOGY_LABELS[topology_name], fontsize=FONT_SIZES["title"], pad=12)
        axis.set_xlabel("Number of nodes N", fontsize=FONT_SIZES["axis_label"])
        axis.set_ylabel("c_kappa", fontsize=FONT_SIZES["axis_label"])
        axis.set_xticks(NODE_COUNTS)
        axis.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
        axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        axis.set_axisbelow(True)
        axis.legend(fontsize=FONT_SIZES["legend"], loc="best")
    figure.suptitle("Stage-1 c_kappa center shift across N", fontsize=FONT_SIZES["title"] + 1)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUTPUT_ROOT / "center_statistics.png"
    pdf_path = OUTPUT_ROOT / "center_statistics.pdf"
    figure.savefig(str(path), dpi=240, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    return path, pdf_path


def build_model_with_center(model_dir, center_override=None):
    """Restore a checkpoint while optionally replacing only actor.curvature.center."""
    raw = read_yaml(model_dir / "training_config.yaml")
    if center_override is not None:
        raw = deepcopy(raw)
        raw.setdefault("actor", {}).setdefault("curvature", {})["center"] = float(center_override)
    actor = _stage1_actor_config(raw)
    critic = _critic_config(raw)
    actor_lr, critic_lr = resolve_learning_rates(raw.get("training", {}))
    model = CTDEPPO(
        learning_rate=float(raw.get("training", {}).get("learning_rate", 3e-4)),
        actor_learning_rate=actor_lr,
        critic_learning_rate=critic_lr,
        clip_ratio=float(raw.get("training", {}).get("clip_ratio", 0.2)),
        entropy_coefficient=float(raw.get("training", {}).get("entropy_coefficient", 0.0)),
        seed=int(raw.get("experiment", {}).get("master_seed", 0)),
        actor_config=actor,
        critic_config=critic,
    )
    model.restore(str(checkpoint_path(model_dir)))
    return model, raw


def load_existing_n150_rows():
    """Load the original N=150 scalability rows as the paired baseline."""
    rows = {}
    for topology_name in TOPOLOGY_NAMES:
        path = SCALABILITY_ROOT / topology_name / "scalability_per_scenario.csv"
        if not path.is_file():
            raise FileNotFoundError("missing existing scalability rows: {}".format(path))
        with path.open("r", newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if int(row["node_count"]) != N150 or row["policy"] != "nn":
                    continue
                key = (topology_name, row["scenario_id"], row["model"])
                rows[key] = row
    return rows


def evaluate_recentered_n150(center_summary):
    """Evaluate only the recentered curvature model on the original N=150 scenarios."""
    baseline = load_existing_n150_rows()
    comparison_rows = []
    provenance = []
    for topology_name in TOPOLOGY_NAMES:
        eval_raw = read_yaml(config_path(topology_name, N150))
        test_center = float(
            [
                row["mean_test_c_kappa"]
                for row in center_summary
                if row["topology"] == topology_name and row["node_count"] == N150
            ][0]
        )
        fixed_center = training_center(topology_name)
        model_dir = MODEL_DIRS[topology_name]["curvature"]
        model, model_raw = build_model_with_center(model_dir, test_center)
        merged = deepcopy(model_raw)
        for key in ("topology", "source", "channel", "curvature", "constraints", "observation", "validation"):
            merged[key] = deepcopy(eval_raw[key])
        merged["experiment"] = dict(merged.get("experiment", {}))
        merged["experiment"]["master_seed"] = int(eval_raw["experiment"].get("master_seed", 20260804))
        provenance.append(
            {
                "topology": topology_name,
                "model": "curvature",
                "model_dir": str(model_dir),
                "checkpoint_episode": checkpoint_episode(model_dir),
                "fixed_training_center": fixed_center,
                "test_n150_center": test_center,
            }
        )
        try:
            before = model.variable_snapshot()
            for scenario in validation_scenarios(merged):
                summary, stats, _ = _run_neural_scenario(model, merged, scenario, policy_name="nn")
                key = (topology_name, scenario.scenario_id, "curvature")
                original = baseline.get(key)
                if original is None:
                    raise KeyError("missing original N=150 baseline row: {}".format(key))
                comparison_rows.append(
                    {
                        "topology": topology_name,
                        "topology_label": TOPOLOGY_LABELS[topology_name],
                        "scenario_id": scenario.scenario_id,
                        "node_count": N150,
                        "slots": scenario.slots,
                        "fixed_training_center": fixed_center,
                        "test_n150_center": test_center,
                        "original_fixed_mean_VAoI": float(original["mean_VAoI"]),
                        "recentered_mean_VAoI": float(summary["mean_VAoI"]),
                        "delta_recentered_minus_fixed_VAoI": float(summary["mean_VAoI"]) - float(original["mean_VAoI"]),
                        "original_fixed_actual_tx_ratio": float(original["actual_tx_ratio"]),
                        "recentered_actual_tx_ratio": float(summary["avg_tx_per_slot"] / scenario.n_nodes),
                        "delta_recentered_minus_fixed_tx_ratio": float(summary["avg_tx_per_slot"] / scenario.n_nodes) - float(original["actual_tx_ratio"]),
                        "original_fixed_action_prob_mean": float(original["action_prob_mean"]),
                        "recentered_action_prob_mean": float(stats["action_prob_mean"]),
                        "delta_recentered_minus_fixed_action_prob": float(stats["action_prob_mean"]) - float(original["action_prob_mean"]),
                        "no_curvature_mean_VAoI": float(baseline[(topology_name, scenario.scenario_id, "no_curvature")]["mean_VAoI"]),
                    }
                )
            after = model.variable_snapshot()
            if len(before) != len(after) or any(not np.array_equal(old, new) for old, new in zip(before, after)):
                raise RuntimeError("recentered validation unexpectedly changed model variables")
        finally:
            model.close()
    return comparison_rows, provenance


def summarize_center_comparison(rows):
    """Aggregate paired N=150 center comparisons with scenario-level CIs."""
    output = []
    for topology_name in TOPOLOGY_NAMES:
        group = [row for row in rows if row["topology"] == topology_name]
        fixed_mean, fixed_std, fixed_ci = mean_std_ci([row["original_fixed_mean_VAoI"] for row in group])
        recentered_mean, recentered_std, recentered_ci = mean_std_ci([row["recentered_mean_VAoI"] for row in group])
        delta_mean, delta_std, delta_ci = mean_std_ci([row["delta_recentered_minus_fixed_VAoI"] for row in group])
        tx_mean, tx_std, tx_ci = mean_std_ci([row["recentered_actual_tx_ratio"] for row in group])
        prob_mean, prob_std, prob_ci = mean_std_ci([row["recentered_action_prob_mean"] for row in group])
        output.append(
            {
                "topology": topology_name,
                "topology_label": TOPOLOGY_LABELS[topology_name],
                "node_count": N150,
                "n_scenarios": len(group),
                "fixed_training_center": group[0]["fixed_training_center"],
                "test_n150_center": group[0]["test_n150_center"],
                "original_fixed_mean_VAoI": fixed_mean,
                "original_fixed_std_VAoI": fixed_std,
                "original_fixed_ci95_VAoI": fixed_ci,
                "recentered_mean_VAoI": recentered_mean,
                "recentered_std_VAoI": recentered_std,
                "recentered_ci95_VAoI": recentered_ci,
                "delta_recentered_minus_fixed_mean_VAoI": delta_mean,
                "delta_recentered_minus_fixed_std_VAoI": delta_std,
                "delta_recentered_minus_fixed_ci95_VAoI": delta_ci,
                "recentered_mean_actual_tx_ratio": tx_mean,
                "recentered_ci95_actual_tx_ratio": tx_ci,
                "recentered_mean_action_probability": prob_mean,
                "recentered_ci95_action_probability": prob_ci,
                "no_curvature_mean_VAoI": float(np.mean([row["no_curvature_mean_VAoI"] for row in group])),
            }
        )
    return output


def plot_center_comparison(summary):
    """Plot original center, N=150 center, and no-curvature VAoI side by side."""
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharey=False)
    labels = ["Fixed\ntraining center", "N=150 test\ncenter", "No curvature"]
    colors = ["#2F6B9A", "#D97732", "#8C8C8C"]
    for axis, topology_name in zip(axes, TOPOLOGY_NAMES):
        row = [item for item in summary if item["topology"] == topology_name][0]
        values = [
            float(row["original_fixed_mean_VAoI"]),
            float(row["recentered_mean_VAoI"]),
            float(row["no_curvature_mean_VAoI"]),
        ]
        errors = [
            float(row["original_fixed_ci95_VAoI"]),
            float(row["recentered_ci95_VAoI"]),
            0.0,
        ]
        bars = axis.bar(np.arange(3), values, yerr=errors, capsize=5, color=colors, alpha=0.92)
        axis.set_title(TOPOLOGY_LABELS[topology_name], fontsize=FONT_SIZES["title"], pad=12)
        axis.set_xticks(np.arange(3))
        axis.set_xticklabels(labels, fontsize=FONT_SIZES["tick"])
        axis.set_ylabel("Mean VAoI (lower is better)", fontsize=FONT_SIZES["axis_label"])
        axis.tick_params(axis="y", labelsize=FONT_SIZES["tick"])
        axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(errors) * 0.08 + 0.02,
                "{:.3f}".format(value),
                ha="center",
                va="bottom",
                fontsize=FONT_SIZES["annotation"],
            )
    figure.suptitle("N=150 Stage-1 center replacement comparison", fontsize=FONT_SIZES["title"] + 1)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUTPUT_ROOT / "n150_center_comparison.png"
    pdf_path = OUTPUT_ROOT / "n150_center_comparison.pdf"
    figure.savefig(str(path), dpi=240, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    return path, pdf_path


def main():
    """Run both the c_kappa distribution analysis and the N=150 paired test."""
    parser = argparse.ArgumentParser(description="Analyze frozen Stage-1 curvature centers")
    parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    copied_configs = copy_existing_configs()
    per_scenario, center_summary = collect_center_statistics()
    write_csv(OUTPUT_ROOT / "center_statistics_per_scenario.csv", per_scenario)
    write_csv(OUTPUT_ROOT / "center_statistics_summary.csv", center_summary)
    center_plot_paths = plot_center_statistics(center_summary)

    comparison_rows, provenance = evaluate_recentered_n150(center_summary)
    comparison_summary = summarize_center_comparison(comparison_rows)
    write_csv(OUTPUT_ROOT / "n150_center_comparison_per_scenario.csv", comparison_rows)
    write_csv(OUTPUT_ROOT / "n150_center_comparison_summary.csv", comparison_summary)
    comparison_plot_paths = plot_center_comparison(comparison_summary)

    metadata = {
        "source_scalability_root": str(SCALABILITY_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "node_counts": NODE_COUNTS,
        "n150": N150,
        "scenarios_per_node_count": SCENARIOS_PER_NODE_COUNT,
        "fixed_centers": {topology: training_center(topology) for topology in TOPOLOGY_NAMES},
        "center_definition": "node c_kappa=max incident local_degree_bound bottleneck importance; test center is the mean over all N=150 test nodes and five scenarios",
        "configs": copied_configs,
        "plots": {
            "center_statistics": [str(path) for path in center_plot_paths],
            "n150_center_comparison": [str(path) for path in comparison_plot_paths],
        },
        "models": provenance,
    }
    with (OUTPUT_ROOT / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)

    print("Saved Stage-1 center analysis to {}".format(OUTPUT_ROOT))


if __name__ == "__main__":
    main()
