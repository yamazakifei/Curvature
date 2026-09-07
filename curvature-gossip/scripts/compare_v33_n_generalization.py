"""Compare the V3.3 single-N and cross-N MPNN checkpoints on shared N sweeps.

功能：复用 0806ScalabilityV3.2_N 的固定 community/random 场景，使用相同
的拓扑、随机种子、节点数和仿真时长，对比 V3.3 单 N 与 cross-N 模型的
VAoI、实际广播比例和平均 Actor 广播概率，并保存 CSV、图和 provenance。
"""

from __future__ import division

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for import_root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_gnn_scalability_generalization as scalability


DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "result_GNN" / "0806ScalabilityV3.2_N"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "result_cross"
    / "mpnnV3.3_Bpen_ch1_cross_n80_120"
    / "0811ScalabilityV3.3_N"
)
DEFAULT_MODEL_DIRS = {
    "single_n_v3_3": (
        PROJECT_ROOT
        / "result_GNN"
        / "mpnnV3.3_Bpen_ch1_n100_u0.20_Bmax0.10_stage1_reused"
    ),
    "cross_n_v3_3": (
        PROJECT_ROOT
        / "result_cross"
        / "mpnnV3.3_Bpen_ch1_cross_n80_120"
        / "mpnnV3.3_Bpen_ch1_cross_n80_120"
    ),
}
MODEL_LABELS = {
    "single_n_v3_3": "V3.3 single-N MPNN",
    "cross_n_v3_3": "V3.3 cross-N MPNN",
}
MODEL_COLORS = {
    "single_n_v3_3": "#2F6B9A",
    "cross_n_v3_3": "#C05640",
}
MATCHED_COLORS = {
    "single_n_v3_3": "#8FB6D1",
    "cross_n_v3_3": "#E5A28F",
}


def copy_config_tree(source_root, output_root):
    """Copy the exact 0806 scenario YAMLs into the new result directory."""
    config_paths = {}
    for topology_name in ("community", "random"):
        source_dir = source_root / "configs" / topology_name
        output_dir = output_root / "configs" / topology_name
        output_dir.mkdir(parents=True, exist_ok=True)
        config_paths[topology_name] = {}
        for source_path in sorted(source_dir.glob("n*.yaml")):
            target_path = output_dir / source_path.name
            shutil.copy2(str(source_path), str(target_path))
            config_paths[topology_name][int(source_path.stem[1:])] = target_path
    return config_paths


def evaluate_model(model_key, model_dir, config_paths, topology_names, include_matched_random):
    """Evaluate one checkpoint while overlaying only shared scenario settings."""
    model, model_raw = scalability.build_model(model_dir)
    checkpoint_episode = scalability.checkpoint_episode(model_dir)
    scenario_rows = []
    try:
        for topology_name in topology_names:
            for node_count in scalability.NODE_COUNTS:
                evaluation_raw = scalability.read_yaml(config_paths[topology_name][node_count])
                # Keep each model's actor/critic architecture and Stage-1 values;
                # replace only topology, channel, source, validation, and seeds.
                merged = deepcopy(model_raw)
                for key in (
                    "topology",
                    "source",
                    "channel",
                    "curvature",
                    "constraints",
                    "observation",
                    "validation",
                ):
                    merged[key] = deepcopy(evaluation_raw[key])
                merged["experiment"] = dict(merged.get("experiment", {}))
                merged["experiment"]["master_seed"] = scalability.EVALUATION_MASTER_SEED
                summary, rows = scalability.evaluate_required_policies(
                    model,
                    merged,
                    checkpoint_episode,
                    include_matched_random=include_matched_random,
                )
                for row in rows:
                    tagged = dict(row)
                    tagged.update(
                        {
                            "topology": topology_name,
                            "model": model_key,
                            "model_label": MODEL_LABELS[model_key],
                            "node_count": int(node_count),
                        }
                    )
                    scenario_rows.append(tagged)
                print(
                    "{} | n={} | {} | nn_VAoI={:.4f}".format(
                        topology_name,
                        node_count,
                        MODEL_LABELS[model_key],
                        float(summary["nn_mean_VAoI"]),
                    ),
                    flush=True,
                )
    finally:
        model.close()
    return scenario_rows, checkpoint_episode


def add_broadcast_probability_summary(summary_rows, scenario_rows):
    """Add mean Actor broadcast probability and its normal 95% CI."""
    grouped = defaultdict(list)
    for row in scenario_rows:
        key = (row["topology"], row["node_count"], row["model"], row["policy"])
        grouped[key].append(float(row["action_prob_mean"]))
    for row in summary_rows:
        key = (row["topology"], row["node_count"], row["model"], row["policy"])
        values = grouped[key]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["mean_action_probability"] = mean
        row["std_action_probability"] = std
        row["ci95_action_probability"] = float(1.96 * std / math.sqrt(len(values))) if values else 0.0
    return summary_rows


def build_model_comparison(summary_rows):
    """Build direct cross-N minus single-N rows for the neural policy."""
    indexed = {
        (row["topology"], int(row["node_count"]), row["model"], row["policy"]): row
        for row in summary_rows
    }
    rows = []
    topology_names = [
        topology_name
        for topology_name in ("community", "random")
        if any(row["topology"] == topology_name for row in summary_rows)
    ]
    for topology_name in topology_names:
        for node_count in scalability.NODE_COUNTS:
            single = indexed[(topology_name, node_count, "single_n_v3_3", "nn")]
            cross = indexed[(topology_name, node_count, "cross_n_v3_3", "nn")]
            rows.append(
                {
                    "topology": topology_name,
                    "node_count": node_count,
                    "single_n_mean_vaoi": single["mean_vaoi"],
                    "cross_n_mean_vaoi": cross["mean_vaoi"],
                    "cross_minus_single_vaoi": cross["mean_vaoi"] - single["mean_vaoi"],
                    "single_n_mean_action_probability": single["mean_action_probability"],
                    "cross_n_mean_action_probability": cross["mean_action_probability"],
                    "cross_minus_single_action_probability": (
                        cross["mean_action_probability"]
                        - single["mean_action_probability"]
                    ),
                }
            )
    return rows


def write_csv(path, rows):
    """Write dictionaries to UTF-8 CSV while preserving first-seen columns."""
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


def plot_topology(summary_rows, topology_name, output_root):
    """Plot both neural curves and matched-rate references with p annotations."""
    figure, axis = plt.subplots(figsize=(10.4, 6.6))
    x = np.asarray(scalability.NODE_COUNTS, dtype=float)
    for model_key in ("single_n_v3_3", "cross_n_v3_3"):
        policies = ["nn"]
        if any(
            row["topology"] == topology_name
            and row["model"] == model_key
            and row["policy"] == "matched_random"
            for row in summary_rows
        ):
            policies.append("matched_random")
        for policy in policies:
            rows = sorted(
                [
                    row
                    for row in summary_rows
                    if row["topology"] == topology_name
                    and row["model"] == model_key
                    and row["policy"] == policy
                ],
                key=lambda row: row["node_count"],
            )
            if len(rows) != len(scalability.NODE_COUNTS):
                raise ValueError("incomplete plot rows for {} / {} / {}".format(topology_name, model_key, policy))
            means = np.asarray([row["mean_vaoi"] for row in rows], dtype=float)
            ci95 = np.asarray([row["ci95_vaoi"] for row in rows], dtype=float)
            color = MODEL_COLORS[model_key] if policy == "nn" else MATCHED_COLORS[model_key]
            linestyle = "-" if policy == "nn" else "--"
            label = MODEL_LABELS[model_key] if policy == "nn" else MODEL_LABELS[model_key] + " matched random"
            axis.plot(x, means, marker="o", linewidth=2.0, markersize=4.8, color=color, linestyle=linestyle, label=label)
            axis.fill_between(x, means - ci95, means + ci95, color=color, alpha=0.09)
            if policy == "nn":
                for node_count, mean_vaoi, row in zip(x, means, rows):
                    axis.annotate(
                        "p={:.3f}".format(float(row["mean_action_probability"])),
                        xy=(node_count, mean_vaoi),
                        xytext=(0, 8 if model_key == "single_n_v3_3" else -14),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if model_key == "single_n_v3_3" else "top",
                        fontsize=7.8,
                        color=color,
                        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
                    )
    title = "Community" if topology_name == "community" else "Random-geometric"
    axis.set_title("V3.3 {} N generalization".format(title), fontsize=16, pad=14)
    axis.set_xlabel("Number of nodes", fontsize=14)
    axis.set_ylabel("Mean VAoI (lower is better)", fontsize=14)
    axis.set_xticks(scalability.NODE_COUNTS)
    axis.tick_params(axis="both", labelsize=12)
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(fontsize=10, frameon=True, loc="best")
    figure.tight_layout()
    png_path = output_root / "scalability_{}.png".format(topology_name)
    pdf_path = output_root / "scalability_{}.pdf".format(topology_name)
    figure.savefig(str(png_path), dpi=240, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main():
    """Copy scenarios, evaluate both checkpoints, aggregate, plot, and record provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--single-model-dir", type=Path, default=DEFAULT_MODEL_DIRS["single_n_v3_3"])
    parser.add_argument("--cross-model-dir", type=Path, default=DEFAULT_MODEL_DIRS["cross_n_v3_3"])
    parser.add_argument(
        "--include-matched-random",
        action="store_true",
        help="also evaluate matched-random references; omitted for the main two-model comparison",
    )
    parser.add_argument(
        "--topology",
        choices=("both", "community", "random"),
        default="both",
        help="evaluate one topology at a time when splitting a long sweep",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="retain rows for the other topology already present in the output directory",
    )
    args = parser.parse_args()

    for path in (args.source_root, args.single_model_dir, args.cross_model_dir):
        if not path.is_dir():
            raise FileNotFoundError("missing input directory: {}".format(path))
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_paths = copy_config_tree(args.source_root, args.output_root)

    topology_names = ("community", "random") if args.topology == "both" else (args.topology,)

    all_rows = []
    existing_rows_path = args.output_root / "scalability_per_scenario.csv"
    if args.append and existing_rows_path.is_file():
        with existing_rows_path.open("r", newline="", encoding="utf-8") as stream:
            all_rows = [
                row
                for row in csv.DictReader(stream)
                if row.get("topology") not in topology_names
            ]
        # CSV append reads numeric fields as strings; normalize grouping keys and
        # probability values before rebuilding the combined summaries.
        numeric_fields = (
            "checkpoint_episode",
            "n_nodes",
            "slots",
            "update_probability",
            "target_tx_ratio",
            "matched_rate_probability",
            "policy_probability",
            "mean_VAoI",
            "actual_tx_ratio",
            "action_prob_mean",
            "action_prob_std",
            "action_prob_min",
            "action_prob_p10",
            "action_prob_p50",
            "action_prob_p90",
            "action_prob_max",
            "action_prob_node_std_mean",
            "action_prob_global_std",
            "action_prob_per_node_mean_std",
            "node_count",
        )
        for row in all_rows:
            for field in numeric_fields:
                if field in row:
                    row[field] = float(row[field])
            row["checkpoint_episode"] = int(row["checkpoint_episode"])
            row["n_nodes"] = int(row["n_nodes"])
            row["slots"] = int(row["slots"])
            row["node_count"] = int(row["node_count"])
    checkpoints = {}
    model_dirs = {
        "single_n_v3_3": args.single_model_dir,
        "cross_n_v3_3": args.cross_model_dir,
    }
    for model_key, model_dir in model_dirs.items():
        rows, checkpoint = evaluate_model(
            model_key,
            model_dir,
            config_paths,
            topology_names,
            args.include_matched_random,
        )
        all_rows.extend(rows)
        checkpoints[model_key] = checkpoint

    summary_rows = add_broadcast_probability_summary(
        scalability.summarize_rows(all_rows), all_rows
    )
    comparison_rows = build_model_comparison(summary_rows)
    write_csv(args.output_root / "scalability_per_scenario.csv", all_rows)
    write_csv(args.output_root / "scalability_summary.csv", summary_rows)
    write_csv(args.output_root / "model_comparison_summary.csv", comparison_rows)
    available_topologies = [
        topology
        for topology in ("community", "random")
        if any(row["topology"] == topology for row in summary_rows)
    ]
    plot_paths = {
        topology: plot_topology(summary_rows, topology, args.output_root)
        for topology in available_topologies
    }

    metadata = {
        "evaluation": "V3.3 single-N versus cross-N MPNN on the existing 0806ScalabilityV3.2_N scenarios",
        "source_config_root": str(args.source_root),
        "output_root": str(args.output_root),
        "model_dirs": {key: str(value) for key, value in model_dirs.items()},
        "checkpoint_episodes": checkpoints,
        "node_counts": list(scalability.NODE_COUNTS),
        "scenarios_per_node_count": scalability.SCENARIOS_PER_NODE_COUNT,
        "slots_per_scenario": scalability.SLOTS_PER_SCENARIO,
        "evaluation_master_seed": scalability.EVALUATION_MASTER_SEED,
        "topologies": available_topologies,
        "density_scaling": "copied unchanged from 0806ScalabilityV3.2_N",
        "comparison": "cross_n_minus_single_n for the neural policy",
        "include_matched_random": bool(args.include_matched_random),
        "plots": {topology: [str(path) for path in paths] for topology, paths in plot_paths.items()},
        "summary_note": "summary CSVs include mean_action_probability for each model and policy",
    }
    with (args.output_root / "scalability_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    with (args.output_root / "README.md").open("w", encoding="utf-8") as stream:
        stream.write(
            "# 0811ScalabilityV3.3_N\n\n"
            "This directory compares the V3.3 single-N and cross-N MPNN checkpoints "
            "on the unchanged scenarios copied from `0806ScalabilityV3.2_N`. "
            "The summary includes mean VAoI, actual transmission ratio, and mean "
            "Actor broadcast probability for both neural policies.\n"
        )
    print("Saved V3.3 N-generalization comparison to {}".format(args.output_root))


if __name__ == "__main__":
    main()
