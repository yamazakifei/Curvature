"""Run the fixed-N scalability evaluation for one cross-N MPNN checkpoint.

功能：复用 0806ScalabilityV3.2_N 的固定 YAML、随机种子和场景设置，
仅评估指定的 cross-N 模型，并将 VAoI、matched-random 和平均广播概率
统计保存到模型目录下的独立结果文件夹中。
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


DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "result_cross"
    / "mpnnV3.2_ch1_cross_n80_120_resume"
)
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "result_GNN" / "0806ScalabilityV3.2_N"
DEFAULT_OUTPUT_ROOT = DEFAULT_MODEL_DIR / "0806ScalabilityV3.2_N_CrossModel"


def copy_config_tree(source_root, output_root):
    """Copy reference YAMLs and return paths keyed by topology and node count."""
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


def mean_std_ci(values):
    """Return mean, sample standard deviation, and normal 95% CI."""
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        std = math.sqrt(variance)
        ci95 = 1.96 * std / math.sqrt(len(values))
    else:
        std = 0.0
        ci95 = 0.0
    return mean, std, ci95


def add_broadcast_probability_summary(summary_rows, scenario_rows):
    """Add average actor broadcast probability and its 95% CI to each summary row."""
    grouped = defaultdict(list)
    for row in scenario_rows:
        key = (row["topology"], row["node_count"], row["model"], row["policy"])
        grouped[key].append(row["action_prob_mean"])
    for row in summary_rows:
        key = (row["topology"], row["node_count"], row["model"], row["policy"])
        mean, std, ci95 = mean_std_ci(grouped[key])
        row["mean_action_probability"] = mean
        row["std_action_probability"] = std
        row["ci95_action_probability"] = ci95
    return summary_rows


def evaluate_single_model(model_dir, config_paths):
    """Evaluate one checkpoint on both topology families and all N values."""
    model, model_raw = scalability.build_model(model_dir)
    checkpoint_episode = scalability.checkpoint_episode(model_dir)
    scenario_rows = []
    try:
        for topology_name in ("community", "random"):
            for node_count in scalability.NODE_COUNTS:
                evaluation_raw = scalability.read_yaml(
                    config_paths[topology_name][node_count]
                )
                # Preserve the model's actor/critic settings and overlay only
                # simulator, topology, and validation settings from the reference.
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
                    include_matched_random=True,
                )
                for row in rows:
                    tagged = dict(row)
                    tagged.update(
                        {
                            "topology": topology_name,
                            "model": "cross_model",
                            "node_count": int(node_count),
                        }
                    )
                    scenario_rows.append(tagged)
                print(
                    "{} | n={} | cross_model | nn_VAoI={:.4f}".format(
                        topology_name, node_count, float(summary["nn_mean_VAoI"])
                    )
                )
    finally:
        model.close()
    return scenario_rows, checkpoint_episode


def plot_topology(summary_rows, topology_name, output_root):
    """Plot cross-N VAoI curves for neural and matched-random policies."""
    figure, axis = plt.subplots(figsize=(9.8, 6.4))
    x = np.asarray(scalability.NODE_COUNTS, dtype=float)
    plot_order = [
        ("cross_model", "nn", "Cross-N curvature MPNN", "#2F6B9A"),
        ("cross_model", "matched_random", "Cross-N matched random", "#E6A23C"),
    ]
    for model_name, policy, label, color in plot_order:
        rows = sorted(
            [
                row
                for row in summary_rows
                if row["topology"] == topology_name
                and row["model"] == model_name
                and row["policy"] == policy
            ],
            key=lambda row: row["node_count"],
        )
        if len(rows) != len(scalability.NODE_COUNTS):
            raise ValueError("incomplete plot rows for {} / {}".format(topology_name, policy))
        means = np.asarray([row["mean_vaoi"] for row in rows], dtype=float)
        ci95 = np.asarray([row["ci95_vaoi"] for row in rows], dtype=float)
        axis.plot(x, means, marker="o", linewidth=2.2, markersize=5.5, color=color, label=label)
        axis.fill_between(x, means - ci95, means + ci95, color=color, alpha=0.14)
        # Put the average broadcast probability beside every VAoI point.
        label_offset = (0, 8) if policy == "nn" else (0, -14)
        vertical_alignment = "bottom" if policy == "nn" else "top"
        for node_count, mean_vaoi, row in zip(x, means, rows):
            axis.annotate(
                "p={:.3f}".format(float(row["mean_action_probability"])),
                xy=(node_count, mean_vaoi),
                xytext=label_offset,
                textcoords="offset points",
                ha="center",
                va=vertical_alignment,
                fontsize=8.5,
                color=color,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            )

    axis.set_title(
        "{} cross-N scalability generalization".format(
            scalability.TOPOLOGY_LABELS[topology_name]
        ),
        fontsize=16,
        pad=14,
    )
    axis.set_xlabel("Number of nodes", fontsize=14)
    axis.set_ylabel("Mean VAoI (lower is better)", fontsize=14)
    axis.set_xticks(scalability.NODE_COUNTS)
    axis.tick_params(axis="both", labelsize=12.5)
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(fontsize=12, frameon=True, loc="best")
    figure.tight_layout()
    png_path = output_root / "scalability_{}.png".format(topology_name)
    pdf_path = output_root / "scalability_{}.pdf".format(topology_name)
    figure.savefig(str(png_path), dpi=240, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def read_summary_csv(path):
    """Read summary CSV values needed to refresh annotated plots."""
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in (
            "node_count",
            "mean_vaoi",
            "ci95_vaoi",
            "mean_action_probability",
        ):
            row[field] = float(row[field])
        row["node_count"] = int(row["node_count"])
    return rows


def main():
    """Copy reference scenarios, evaluate the checkpoint, and save artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="refresh annotated plots from an existing summary CSV without simulation",
    )
    args = parser.parse_args()

    if args.plot_only:
        summary_path = args.output_root / "scalability_summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError("missing summary CSV: {}".format(summary_path))
        summary_rows = read_summary_csv(summary_path)
        for topology_name in ("community", "random"):
            plot_topology(summary_rows, topology_name, args.output_root)
        print("Refreshed annotated scalability plots in {}".format(args.output_root))
        return

    if not args.model_dir.is_dir():
        raise FileNotFoundError("missing model directory: {}".format(args.model_dir))
    if not args.source_root.is_dir():
        raise FileNotFoundError("missing source scalability directory: {}".format(args.source_root))

    args.output_root.mkdir(parents=True, exist_ok=True)
    config_paths = copy_config_tree(args.source_root, args.output_root)
    scenario_rows, checkpoint_episode = evaluate_single_model(
        args.model_dir, config_paths
    )
    summary_rows = add_broadcast_probability_summary(
        scalability.summarize_rows(scenario_rows), scenario_rows
    )
    write_csv(args.output_root / "scalability_per_scenario.csv", scenario_rows)
    write_csv(args.output_root / "scalability_summary.csv", summary_rows)
    plot_paths = {
        topology: plot_topology(summary_rows, topology, args.output_root)
        for topology in ("community", "random")
    }

    metadata = {
        "evaluation": "cross-N MPNN checkpoint on the existing 0806ScalabilityV3.2_N scenarios",
        "source_config_root": str(args.source_root),
        "model_dir": str(args.model_dir),
        "checkpoint_episode": checkpoint_episode,
        "node_counts": list(scalability.NODE_COUNTS),
        "scenarios_per_node_count": scalability.SCENARIOS_PER_NODE_COUNT,
        "slots_per_scenario": 200,
        "evaluation_master_seed": scalability.EVALUATION_MASTER_SEED,
        "density_scaling": "reused from 0806ScalabilityV3.2_N",
        "configs": {
            topology: [str(path) for path in paths.values()]
            for topology, paths in config_paths.items()
        },
        "plots": {
            topology: [str(path) for path in paths]
            for topology, paths in plot_paths.items()
        },
        "summary_note": "scalability_summary.csv includes mean_action_probability for each VAoI row",
    }
    with (args.output_root / "scalability_metadata.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    print("Saved cross-model scalability output to {}".format(args.output_root))


if __name__ == "__main__":
    main()
