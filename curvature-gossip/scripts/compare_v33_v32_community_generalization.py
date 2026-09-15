"""Compare five MPNN checkpoints on the shared community N-generalization sweep.

This script reuses the existing four-model comparison artifacts and the
community scenario YAMLs.  It evaluates only the supplied new V3.3 checkpoint
on exactly the same scenarios, writes combined CSV/provenance files, and plots
five mean-VAoI curves.  Every point is annotated with the
mean Actor broadcast probability using the same font size as the legend.
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
    / "result_GNN"
    / "0908ScalabilityV3.3_community_four_models"
)
DEFAULT_MODEL_DIRS = {
    "v33_curv_bmax20": (
        PROJECT_ROOT
        / "result_GNN"
        / "mpnnV3.3_Bpen_ch1_n100_u0.20_curvBmax20_stage1_reused"
    ),
    "v33_random_trained": (
        PROJECT_ROOT
        / "result_GNN"
        / "mpnnV3.3_Bpen_random_n100_u0.20_Bmax0.10_stage1_reused"
    ),
    "v33_no_curvature": (
        PROJECT_ROOT
        / "result_GNN"
        / "mpnnV3.3_Bpen_ch1_NoCurv_n100_u0.20"
    ),
}

MODEL_ORDER = (
    "v32_curvature",
    "v32_no_curvature",
    "v33_curv_bmax20",
    "v33_random_trained",
    "v33_no_curvature",
)
MODEL_LABELS = {
    "v32_curvature": "V3.2 curvature MPNN",
    "v32_no_curvature": "V3.2 no-curvature MPNN",
    "v33_curv_bmax20": "V3.3 single-N MPNN (curv bmax=20)",
    "v33_random_trained": "V3.3 single-N MPNN (curv bmax=1)",
    "v33_no_curvature": "V3.3 single-N MPNN (no curvature)",
}
MODEL_COLORS = {
    "v32_curvature": "#2F6B9A",
    "v32_no_curvature": "#8C8C8C",
    "v33_curv_bmax20": "#C05640",
    "v33_random_trained": "#7B4FA3",
    "v33_no_curvature": "#2A9D8F",
}
MODEL_LINESTYLES = {
    "v32_curvature": "--",
    "v32_no_curvature": ":",
    "v33_curv_bmax20": "-",
    "v33_random_trained": "-.",
    "v33_no_curvature": (0, (4, 1, 1, 1)),
}
MODEL_MARKERS = {
    "v32_curvature": "s",
    "v32_no_curvature": "^",
    "v33_curv_bmax20": "o",
    "v33_random_trained": "D",
    "v33_no_curvature": "X",
}
ANNOTATION_OFFSETS = {
    "v32_curvature": (-1, 15),
    "v32_no_curvature": (-1, -17),
    "v33_curv_bmax20": (1, 31),
    "v33_random_trained": (1, -33),
    "v33_no_curvature": (1, -49),
}
FONT_SIZES = {
    "title": 16,
    "axis_label": 14,
    "tick": 12,
    "legend": 10.5,
}


def copy_community_configs(source_root, output_root):
    """Copy the unchanged community scenario YAMLs into the output bundle."""
    source_dir = source_root / "configs" / "community"
    output_dir = output_root / "configs" / "community"
    if not source_dir.is_dir():
        raise FileNotFoundError("missing community scenario directory: {}".format(source_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = {}
    for source_path in sorted(source_dir.glob("n*.yaml")):
        node_count = int(source_path.stem[1:])
        target_path = output_dir / source_path.name
        shutil.copy2(str(source_path), str(target_path))
        config_paths[node_count] = target_path
    if set(config_paths) != set(scalability.NODE_COUNTS):
        raise ValueError("community scenario files do not cover the expected N sweep")
    return config_paths


def _as_int(row, key):
    return int(float(row[key]))


def _as_float(row, key):
    return float(row[key])


def load_v32_community_baselines(source_root):
    """Load the two existing V3.2 community neural curves and their p values."""
    summary_path = source_root / "scalability_summary.csv"
    per_scenario_path = source_root / "scalability_per_scenario.csv"
    for path in (summary_path, per_scenario_path):
        if not path.is_file():
            raise FileNotFoundError("missing V3.2 scalability artifact: {}".format(path))

    model_map = {"curvature": "v32_curvature", "no_curvature": "v32_no_curvature"}
    summary_rows = []
    with summary_path.open("r", newline="", encoding="utf-8") as stream:
        for raw_row in csv.DictReader(stream):
            if raw_row.get("topology") != "community" or raw_row.get("policy") != "nn":
                continue
            if raw_row.get("model") not in model_map:
                continue
            row = dict(raw_row)
            row["model"] = model_map[row["model"]]
            row["model_label"] = MODEL_LABELS[row["model"]]
            row["node_count"] = _as_int(row, "node_count")
            for key in (
                "n_scenarios", "mean_vaoi", "std_vaoi", "ci95_vaoi",
                "mean_actual_tx_ratio", "std_actual_tx_ratio", "ci95_actual_tx_ratio",
                "mean_action_probability", "std_action_probability", "ci95_action_probability",
            ):
                row[key] = _as_float(row, key)
            summary_rows.append(row)

    scenario_rows = []
    with per_scenario_path.open("r", newline="", encoding="utf-8") as stream:
        for raw_row in csv.DictReader(stream):
            if raw_row.get("topology") != "community" or raw_row.get("policy") != "nn":
                continue
            if raw_row.get("model") not in model_map:
                continue
            row = dict(raw_row)
            row["model"] = model_map[row["model"]]
            row["model_label"] = MODEL_LABELS[row["model"]]
            row["node_count"] = _as_int(row, "node_count")
            scenario_rows.append(row)

    expected_rows = len(scalability.NODE_COUNTS)
    if len(summary_rows) != 2 * expected_rows:
        raise ValueError("expected {} V3.2 baseline summary rows, got {}".format(2 * expected_rows, len(summary_rows)))
    if len(scenario_rows) != 2 * expected_rows * scalability.SCENARIOS_PER_NODE_COUNT:
        raise ValueError("unexpected V3.2 baseline scenario-row count: {}".format(len(scenario_rows)))
    return summary_rows, scenario_rows


def load_existing_four_model_results(output_root):
    """Reuse the existing four curves without rerunning their checkpoints."""
    summary_path = output_root / "scalability_summary.csv"
    scenario_path = output_root / "scalability_per_scenario.csv"
    for path in (summary_path, scenario_path):
        if not path.is_file():
            raise FileNotFoundError("missing existing comparison artifact: {}".format(path))

    existing_models = set(MODEL_ORDER[:-1])
    summary_rows = []
    with summary_path.open("r", newline="", encoding="utf-8") as stream:
        for raw_row in csv.DictReader(stream):
            if raw_row.get("model") not in existing_models:
                continue
            row = dict(raw_row)
            row["model_label"] = MODEL_LABELS[row["model"]]
            summary_rows.append(row)

    scenario_rows = []
    with scenario_path.open("r", newline="", encoding="utf-8") as stream:
        for raw_row in csv.DictReader(stream):
            if raw_row.get("model") not in existing_models:
                continue
            row = dict(raw_row)
            row["model_label"] = MODEL_LABELS[row["model"]]
            scenario_rows.append(row)

    expected_summary = len(MODEL_ORDER[:-1]) * len(scalability.NODE_COUNTS)
    expected_scenarios = expected_summary * scalability.SCENARIOS_PER_NODE_COUNT
    if len(summary_rows) != expected_summary:
        raise ValueError(
            "expected {} existing summary rows, got {}".format(
                expected_summary, len(summary_rows)
            )
        )
    if len(scenario_rows) != expected_scenarios:
        raise ValueError(
            "expected {} existing scenario rows, got {}".format(
                expected_scenarios, len(scenario_rows)
            )
        )
    return summary_rows, scenario_rows


def evaluate_v33_model(model_key, model_dir, config_paths):
    """Evaluate one V3.3 checkpoint on every shared community scenario."""
    model, model_raw = scalability.build_model(model_dir)
    checkpoint = scalability.checkpoint_episode(model_dir)
    scenario_rows = []
    try:
        for node_count in scalability.NODE_COUNTS:
            evaluation_raw = scalability.read_yaml(config_paths[node_count])
            merged = deepcopy(model_raw)
            # Overlay only the shared simulator/scenario settings; retain the
            # checkpoint's Actor, Critic, and Stage-1 parameters.
            for key in (
                "topology", "source", "channel", "curvature",
                "constraints", "observation", "validation",
            ):
                merged[key] = deepcopy(evaluation_raw[key])
            merged["experiment"] = dict(merged.get("experiment", {}))
            merged["experiment"]["master_seed"] = scalability.EVALUATION_MASTER_SEED
            summary, rows = scalability.evaluate_required_policies(
                model, merged, checkpoint, include_matched_random=False
            )
            for row in rows:
                tagged = dict(row)
                tagged.update({
                    "topology": "community",
                    "model": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "node_count": int(node_count),
                })
                scenario_rows.append(tagged)
            print(
                "community | n={} | {} | nn_VAoI={:.4f}".format(
                    node_count, MODEL_LABELS[model_key], float(summary["nn_mean_VAoI"])
                ),
                flush=True,
            )
    finally:
        model.close()
    return scenario_rows, checkpoint


def add_probability_summary(summary_rows, scenario_rows):
    """Add mean Actor broadcast probability and its 95% confidence interval."""
    grouped = defaultdict(list)
    for row in scenario_rows:
        grouped[(row["model"], int(row["node_count"]))].append(float(row["action_prob_mean"]))
    for row in summary_rows:
        values = grouped[(row["model"], int(row["node_count"]))]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["mean_action_probability"] = mean
        row["std_action_probability"] = std
        row["ci95_action_probability"] = float(1.96 * std / math.sqrt(len(values))) if values else 0.0
    return summary_rows


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


def plot_community(
    summary_rows,
    output_root,
    model_keys=None,
    filename_stem="scalability_community_four_models",
    title="Community N-generalization: V3.2 and V3.3 MPNN",
    annotation_offsets=None,
    legend_columns=2,
):
    """Plot selected community curves and annotate every point with mean p."""
    figure, axis = plt.subplots(figsize=(12.6, 7.4))
    model_keys = tuple(model_keys or MODEL_ORDER)
    annotation_offsets = annotation_offsets or ANNOTATION_OFFSETS
    x = np.asarray(scalability.NODE_COUNTS, dtype=float)
    for model_key in model_keys:
        rows = sorted(
            [
                row for row in summary_rows
                if row["topology"] == "community"
                and row["model"] == model_key
                and row["policy"] == "nn"
            ],
            key=lambda row: int(row["node_count"]),
        )
        if len(rows) != len(scalability.NODE_COUNTS):
            raise ValueError("incomplete community curve for {}".format(model_key))
        means = np.asarray([float(row["mean_vaoi"]) for row in rows], dtype=float)
        ci95 = np.asarray([float(row["ci95_vaoi"]) for row in rows], dtype=float)
        axis.plot(
            x, means,
            color=MODEL_COLORS[model_key],
            linestyle=MODEL_LINESTYLES[model_key],
            marker=MODEL_MARKERS[model_key],
            linewidth=2.1,
            markersize=5.5,
            label=MODEL_LABELS[model_key],
        )
        axis.fill_between(x, means - ci95, means + ci95, color=MODEL_COLORS[model_key], alpha=0.055)
        x_offset, y_offset = annotation_offsets[model_key]
        for node_count, mean_vaoi, row in zip(x, means, rows):
            # Keep annotation and legend typography identical as requested.
            axis.annotate(
                "p={:.3f}".format(float(row["mean_action_probability"])),
                xy=(node_count, mean_vaoi),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if y_offset > 0 else "top",
                fontsize=FONT_SIZES["legend"],
                color=MODEL_COLORS[model_key],
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "alpha": 0.72,
                    "edgecolor": "none",
                },
            )

    axis.set_title(title, fontsize=FONT_SIZES["title"], pad=14)
    axis.set_xlabel("Number of nodes", fontsize=FONT_SIZES["axis_label"])
    axis.set_ylabel("Mean VAoI (lower is better)", fontsize=FONT_SIZES["axis_label"])
    axis.text(
        0.01, 0.99, "p = mean Actor broadcast probability",
        transform=axis.transAxes, ha="left", va="top", fontsize=FONT_SIZES["legend"],
        color="#444444",
    )
    axis.set_xticks(scalability.NODE_COUNTS)
    axis.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(fontsize=FONT_SIZES["legend"], frameon=True, loc="best", ncol=legend_columns)
    figure.tight_layout()
    # Keep the historical full-comparison filename and use a separate stem for subsets.
    png_path = output_root / (filename_stem + ".png")
    pdf_path = output_root / (filename_stem + ".pdf")
    figure.savefig(str(png_path), dpi=240, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main():
    """Reuse four existing curves, evaluate NoCurv, and save five curves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--no-curvature-model-dir", type=Path,
        default=DEFAULT_MODEL_DIRS["v33_no_curvature"],
    )
    args = parser.parse_args()

    for path in (
        args.source_root,
        args.no_curvature_model_dir,
    ):
        if not path.is_dir():
            raise FileNotFoundError("missing input directory: {}".format(path))
    args.output_root.mkdir(parents=True, exist_ok=True)

    config_paths = copy_community_configs(args.source_root, args.output_root)
    existing_summary, existing_scenario_rows = load_existing_four_model_results(
        args.output_root
    )
    no_curvature_rows, no_curvature_checkpoint = evaluate_v33_model(
        "v33_no_curvature", args.no_curvature_model_dir, config_paths
    )
    no_curvature_summary = add_probability_summary(
        scalability.summarize_rows(no_curvature_rows), no_curvature_rows
    )
    for row in no_curvature_summary:
        row["model_label"] = MODEL_LABELS["v33_no_curvature"]
    summary_rows = existing_summary + no_curvature_summary
    summary_rows.sort(key=lambda row: (MODEL_ORDER.index(row["model"]), int(row["node_count"])))
    scenario_rows = existing_scenario_rows + no_curvature_rows
    write_csv(args.output_root / "scalability_per_scenario.csv", scenario_rows)
    write_csv(args.output_root / "scalability_summary.csv", summary_rows)
    plot_paths = plot_community(summary_rows, args.output_root)
    v33_subset_plot_paths = plot_community(
        summary_rows,
        args.output_root,
        model_keys=("v33_curv_bmax20", "v33_no_curvature"),
        filename_stem="scalability_v33_bmax20_nocurvature",
        title="V3.3 curvature comparison: community N-generalization",
        annotation_offsets={
            "v33_curv_bmax20": (1, 14),
            "v33_no_curvature": (1, -14),
        },
        legend_columns=1,
    )

    # Record checkpoint provenance for reused curves without running inference.
    checkpoint_episodes = {}
    for model_key in ("v33_curv_bmax20", "v33_random_trained"):
        episode = scalability.checkpoint_episode(DEFAULT_MODEL_DIRS[model_key])
        if episode >= 0:
            checkpoint_episodes[model_key] = int(episode)
    checkpoint_episodes["v33_no_curvature"] = int(no_curvature_checkpoint)

    metadata = {
        "evaluation": "Five-model community N-generalization using the unchanged 0806ScalabilityV3.2_N scenarios",
        "source_scenario_root": str(args.source_root),
        "reference_v32_summary": str(args.source_root / "scalability_summary.csv"),
        "output_root": str(args.output_root),
        "models": {
            "v32_curvature": "reused from the existing four-model summary",
            "v32_no_curvature": "reused from the existing four-model summary",
            "v33_curv_bmax20": "reused from the existing four-model summary",
            "v33_random_trained": "reused from the existing four-model summary; relabeled as requested",
            "v33_no_curvature": str(args.no_curvature_model_dir),
        },
        "checkpoint_episodes": checkpoint_episodes,
        "line_order": list(MODEL_ORDER),
        "node_counts": list(scalability.NODE_COUNTS),
        "scenarios_per_node_count": scalability.SCENARIOS_PER_NODE_COUNT,
        "slots_per_scenario": scalability.SLOTS_PER_SCENARIO,
        "evaluation_master_seed": scalability.EVALUATION_MASTER_SEED,
        "topology": "community only",
        "scenario_policy": "copied unchanged from 0806ScalabilityV3.2_N",
        "mean_broadcast_probability": "summary mean of per-scenario action_prob_mean",
        "annotation_fontsize": FONT_SIZES["legend"],
        "plots": [str(path) for path in plot_paths],
        "subset_plots": [str(path) for path in v33_subset_plot_paths],
    }
    with (args.output_root / "scalability_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    with (args.output_root / "README.md").open("w", encoding="utf-8") as stream:
        stream.write(
            "# V3.2/V3.3 community N-generalization\n\n"
            "This comparison reuses the community YAML scenarios from "
            "`0806ScalabilityV3.2_N` unchanged. It contains five neural curves: "
            "V3.2 curvature MPNN, V3.2 no-curvature MPNN, V3.3 single-N "
            "curvature-edge bmax=20, V3.3 single-N curvature-edge bmax=1, and "
            "the V3.3 single-N no-curvature model. The four existing curves "
            "are reused from the prior comparison artifacts; only the new "
            "no-curvature checkpoint is evaluated. A separate plot containing "
            "only the bmax=20 and no-curvature V3.3 curves is saved as "
            "`scalability_v33_bmax20_nocurvature.png` and `.pdf`. "
            "Every plotted point is annotated with its mean Actor broadcast "
            "probability using the same font size as the legend.\n"
        )
    print("Saved five-model community comparison to {}".format(args.output_root))


if __name__ == "__main__":
    main()
