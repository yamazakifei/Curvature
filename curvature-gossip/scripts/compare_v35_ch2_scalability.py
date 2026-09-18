"""Generate the V3.5 ch2 community scalability comparison plot.

功能：加载两个 V3.5 Stage-2 MPNN 的 ``best_validation_Bmax`` checkpoint，
在 50--150 节点的 community N-generalization 场景上评估曲率、无曲率和
“仅 Stage-1 概率（无 Stage-2 微调）”三条曲线，保存 VAoI/广播概率数据、
PNG/PDF 图、评估场景和复现实验元数据。

The evaluation keeps the V3.5 ch2 channel configuration from the curvature
model and scales the community geometry with sqrt(N/100), matching the
reference scalability figure's density policy.
"""

from __future__ import division

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import run_gnn_scalability_generalization as scalability
from curvature_gossip.learning.ctde_ppo import CTDEPPO, resolve_learning_rates
from curvature_gossip.learning.trainer import _critic_config, _stage1_actor_config
from curvature_gossip.learning.validation import _run_neural_scenario, validation_scenarios


RESULT_ROOT = PROJECT_ROOT / "result_GNN"
DEFAULT_OUTPUT_ROOT = RESULT_ROOT / "0917ScalabilityV35_ch2"
DEFAULT_CURVATURE_MODEL = RESULT_ROOT / "mpnnV3.5_CurvAttn_ch2_n100_u0.20_4EdgeFeature_rawBmax"
DEFAULT_NO_CURVATURE_MODEL = RESULT_ROOT / "mpnnV3.5_CurvAttn_ch2_n100_u0.20_noCurv"

MODEL_ORDER = (
    "v35_curvature_stage2",
    "v35_no_curvature_stage2",
    "v35_curvature_stage1_only",
)
MODEL_LABELS = {
    "v35_curvature_stage2": "V3.5 ch2 curvature MPNN (Stage-2)",
    "v35_no_curvature_stage2": "V3.5 ch2 no-curvature MPNN (Stage-2)",
    "v35_curvature_stage1_only": "V3.5 ch2 curvature base (Stage-1 only)",
}
MODEL_COLORS = {
    "v35_curvature_stage2": "#C05640",
    "v35_no_curvature_stage2": "#2A9D8F",
    "v35_curvature_stage1_only": "#2F6B9A",
}
MODEL_LINESTYLES = {
    "v35_curvature_stage2": "-",
    "v35_no_curvature_stage2": "--",
    "v35_curvature_stage1_only": "-.",
}
MODEL_MARKERS = {
    "v35_curvature_stage2": "o",
    "v35_no_curvature_stage2": "X",
    "v35_curvature_stage1_only": "s",
}
# Fixed offsets keep the three probability labels readable near each point.
ANNOTATION_OFFSETS = {
    "v35_curvature_stage2": (1, 18),
    "v35_no_curvature_stage2": (1, -18),
    "v35_curvature_stage1_only": (1, 2),
}
FONT_SIZES = {
    "title": 16,
    "axis_label": 14,
    "tick": 12,
    "legend": 10.5,
}


def read_yaml(path):
    """Read one YAML mapping and report the path on malformed input."""
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("expected a YAML mapping: {}".format(path))
    return value


def bmax_checkpoint_path(model_dir):
    """Return the explicitly requested probability-feasible checkpoint."""
    path = model_dir / "checkpoints" / "best_validation_Bmax" / "model"
    if not path.with_suffix(".index").is_file():
        raise FileNotFoundError("missing best_validation_Bmax checkpoint: {}".format(path))
    return path


def checkpoint_episode(model_dir):
    """Read the selected checkpoint episode for provenance."""
    info_path = model_dir / "checkpoints" / "best_validation_Bmax" / "checkpoint_info.json"
    if not info_path.is_file():
        return -1
    with info_path.open("r", encoding="utf-8") as stream:
        return int(json.load(stream).get("episode", -1))


def build_model(model_dir):
    """Restore a V3.5 model using its persisted training configuration."""
    raw = read_yaml(model_dir / "training_config.yaml")
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
    model.restore(str(bmax_checkpoint_path(model_dir)))
    return model, raw


def build_evaluation_config(reference_raw, node_count, topology_index=0):
    """Build one ch2 community scenario bundle while preserving ch2 physics."""
    topology, scale = scalability.scale_topology_params(
        reference_raw["topology"], "community", node_count
    )
    raw = {
        "experiment": {"master_seed": scalability.EVALUATION_MASTER_SEED},
        "topology": topology,
        "source": deepcopy(reference_raw["source"]),
        "channel": deepcopy(reference_raw["channel"]),
        "curvature": deepcopy(reference_raw["curvature"]),
        "constraints": deepcopy(reference_raw["constraints"]),
        "observation": deepcopy(reference_raw["observation"]),
        "validation": {"scenarios": scalability.build_scenarios(node_count, topology_index)},
        "metadata": {
            "topology_name": "community",
            "node_count": int(node_count),
            "reference_node_count": scalability.REFERENCE_NODE_COUNT,
            "linear_geometry_scale": scale,
            "density_policy": "area and community geometry scale as sqrt(N/100); communication radius is fixed",
            "physics_source": "V3.5 ch2 curvature model training_config.yaml",
        },
    }
    return raw


def write_evaluation_configs(reference_raw, output_root):
    """Write the fixed community YAML sweep under the result directory."""
    config_dir = output_root / "configs" / "community"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for node_count in scalability.NODE_COUNTS:
        path = config_dir / "n{:03d}.yaml".format(node_count)
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                build_evaluation_config(reference_raw, node_count),
                stream,
                sort_keys=False,
                allow_unicode=True,
            )
        paths[node_count] = path
    return paths


def evaluate_model(model, model_raw, model_key, checkpoint_value, config_paths, stage1_only=False):
    """Evaluate one policy mode on every scenario in the common ch2 sweep."""
    rows = []
    before = model.variable_snapshot()
    try:
        for node_count in scalability.NODE_COUNTS:
            evaluation_raw = read_yaml(config_paths[node_count])
            merged = deepcopy(model_raw)
            # Only simulator/scenario fields come from the scalable evaluation YAML;
            # Actor/Critic architecture and frozen Stage-1 values stay model-specific.
            for key in (
                "topology", "source", "channel", "curvature",
                "constraints", "observation", "validation",
            ):
                merged[key] = deepcopy(evaluation_raw[key])
            merged["experiment"] = dict(merged.get("experiment", {}))
            merged["experiment"]["master_seed"] = scalability.EVALUATION_MASTER_SEED

            for scenario in validation_scenarios(merged):
                summary, stats, probability_matrix = _run_neural_scenario(
                    model,
                    merged,
                    scenario,
                    policy_name="nn",
                    stage1_only=stage1_only,
                )
                mean_probability = float(np.mean(probability_matrix))
                rows.append(
                    {
                        "checkpoint_episode": int(checkpoint_value),
                        "scenario_id": scenario.scenario_id,
                        "policy": "stage1_only" if stage1_only else "nn",
                        "n_nodes": scenario.n_nodes,
                        "slots": scenario.slots,
                        "update_probability": scenario.update_probability,
                        "target_tx_ratio": scenario.target_tx_ratio,
                        "policy_probability": mean_probability,
                        "mean_VAoI": float(summary["mean_VAoI"]),
                        "actual_tx_ratio": float(summary["avg_tx_per_slot"] / scenario.n_nodes),
                        "topology": "community",
                        "model": model_key,
                        "node_count": int(node_count),
                        "model_label": MODEL_LABELS[model_key],
                        **stats
                    }
                )
            print(
                "community | n={} | {} | VAoI={:.4f}".format(
                    node_count,
                    MODEL_LABELS[model_key],
                    float(np.mean([
                        row["mean_VAoI"] for row in rows
                        if row["node_count"] == node_count and row["model"] == model_key
                    ])),
                ),
                flush=True,
            )
    finally:
        after = model.variable_snapshot()
        if len(before) != len(after) or any(
            not np.array_equal(old, new) for old, new in zip(before, after)
        ):
            raise RuntimeError("scalability validation unexpectedly changed model variables")
    return rows


def mean_ci(values):
    """Return mean, standard deviation, and normal-approximation 95% CI."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, std, ci95


def summarize_rows(scenario_rows):
    """Aggregate the five scenarios for each model and node count."""
    grouped = {}
    for row in scenario_rows:
        key = (row["topology"], int(row["node_count"]), row["model"], row["policy"])
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (topology, node_count, model, policy), rows in sorted(grouped.items()):
        vaoi_mean, vaoi_std, vaoi_ci = mean_ci([row["mean_VAoI"] for row in rows])
        tx_mean, tx_std, tx_ci = mean_ci([row["actual_tx_ratio"] for row in rows])
        p_mean, p_std, p_ci = mean_ci([row["action_prob_mean"] for row in rows])
        summaries.append(
            {
                "topology": topology,
                "topology_label": "Community topology",
                "node_count": node_count,
                "model": model,
                "policy": policy,
                "model_label": MODEL_LABELS[model],
                "n_scenarios": len(rows),
                "mean_vaoi": vaoi_mean,
                "std_vaoi": vaoi_std,
                "ci95_vaoi": vaoi_ci,
                "mean_actual_tx_ratio": tx_mean,
                "std_actual_tx_ratio": tx_std,
                "ci95_actual_tx_ratio": tx_ci,
                "mean_action_probability": p_mean,
                "std_action_probability": p_std,
                "ci95_action_probability": p_ci,
            }
        )
    return sorted(summaries, key=lambda row: (MODEL_ORDER.index(row["model"]), row["node_count"]))


def write_csv(path, rows):
    """Write dictionaries as UTF-8 CSV while preserving first-seen columns."""
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


def plot_scalability(summary_rows, output_root):
    """Plot the three VAoI curves, CI bands, and mean broadcast probabilities."""
    figure, axis = plt.subplots(figsize=(12.6, 7.4))
    x = np.asarray(scalability.NODE_COUNTS, dtype=float)
    for model_key in MODEL_ORDER:
        rows = sorted(
            [
                row for row in summary_rows
                if row["topology"] == "community" and row["model"] == model_key
            ],
            key=lambda row: int(row["node_count"]),
        )
        if len(rows) != len(scalability.NODE_COUNTS):
            raise ValueError("incomplete scalability curve for {}".format(model_key))
        means = np.asarray([float(row["mean_vaoi"]) for row in rows], dtype=float)
        ci95 = np.asarray([float(row["ci95_vaoi"]) for row in rows], dtype=float)
        color = MODEL_COLORS[model_key]
        axis.plot(
            x,
            means,
            color=color,
            linestyle=MODEL_LINESTYLES[model_key],
            marker=MODEL_MARKERS[model_key],
            linewidth=2.1,
            markersize=5.5,
            label=MODEL_LABELS[model_key],
        )
        axis.fill_between(x, means - ci95, means + ci95, color=color, alpha=0.055)
        x_offset, y_offset = ANNOTATION_OFFSETS[model_key]
        for node_count, mean_vaoi, row in zip(x, means, rows):
            # The label reports the same per-scenario mean probability used in CSV.
            point_x_offset, point_y_offset = x_offset, y_offset
            # Keep the tallest Stage-1 labels below the upper-right legend.
            if model_key == "v35_curvature_stage1_only" and node_count >= 140:
                point_x_offset, point_y_offset = -18, -18
            axis.annotate(
                "p={:.3f}".format(float(row["mean_action_probability"])),
                xy=(node_count, mean_vaoi),
                xytext=(point_x_offset, point_y_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if point_y_offset >= 0 else "top",
                fontsize=FONT_SIZES["legend"],
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "alpha": 0.72,
                    "edgecolor": "none",
                },
            )

    axis.set_title("V3.5 ch2 curvature comparison: community N-generalization", fontsize=FONT_SIZES["title"], pad=14)
    axis.set_xlabel("Number of nodes", fontsize=FONT_SIZES["axis_label"])
    axis.set_ylabel("Mean VAoI (lower is better)", fontsize=FONT_SIZES["axis_label"])
    axis.text(
        0.01,
        0.99,
        "p = mean Actor broadcast probability",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZES["legend"],
        color="#444444",
    )
    axis.set_xticks(scalability.NODE_COUNTS)
    axis.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)
    # Keep the explanatory probability note visible in the upper-left corner.
    axis.legend(fontsize=FONT_SIZES["legend"], frameon=True, loc="upper right", ncol=1)
    figure.tight_layout()
    png_path = output_root / "scalability_v35_ch2.png"
    pdf_path = output_root / "scalability_v35_ch2.pdf"
    figure.savefig(str(png_path), dpi=240, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def write_readme(output_root):
    """Write a compact provenance note next to the generated artifacts."""
    text = (
        "# V3.5 ch2 community scalability\n\n"
        "This result evaluates the `best_validation_Bmax` checkpoint from the "
        "V3.5 ch2 curvature-attention MPNN and the V3.5 ch2 no-curvature MPNN "
        "on a fixed community N-generalization sweep (`N=50..150`, step 10, "
        "five scenarios per N, 200 slots per scenario). The scenario physics "
        "are inherited from the V3.5 ch2 training configuration, while the "
        "community geometry is scaled with `sqrt(N/100)` to keep density stable.\n\n"
        "The third curve uses the curvature model checkpoint's Stage-1 base "
        "probability output directly; Stage-2 residual inference is disabled "
        "for that curve. Every point is annotated with the mean Actor broadcast "
        "probability, and the shaded regions are normal-approximation 95% CIs "
        "over the five scenarios.\n"
    )
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main():
    """Run evaluation, aggregate CSVs, plot results, and save provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curvature-model", type=Path, default=DEFAULT_CURVATURE_MODEL)
    parser.add_argument("--no-curvature-model", type=Path, default=DEFAULT_NO_CURVATURE_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    # Allow a shorter reproducible sweep when CPU-only evaluation is too slow.
    parser.add_argument(
        "--slots",
        type=int,
        default=None,
        help="slots per scenario; defaults to the reference 200-slot setting",
    )
    args = parser.parse_args()

    if args.slots is not None:
        if args.slots < 1:
            raise ValueError("--slots must be a positive integer")
        scalability.SLOTS_PER_SCENARIO = int(args.slots)

    for path in (args.curvature_model, args.no_curvature_model):
        if not path.is_dir():
            raise FileNotFoundError("missing input directory: {}".format(path))
        bmax_checkpoint_path(path)
    args.output_root.mkdir(parents=True, exist_ok=True)

    curvature_model, curvature_raw = build_model(args.curvature_model)
    no_curvature_model, no_curvature_raw = build_model(args.no_curvature_model)
    try:
        config_paths = write_evaluation_configs(curvature_raw, args.output_root)
        scenario_rows = []
        curvature_episode = checkpoint_episode(args.curvature_model)
        no_curvature_episode = checkpoint_episode(args.no_curvature_model)
        scenario_rows.extend(
            evaluate_model(
                curvature_model,
                curvature_raw,
                "v35_curvature_stage2",
                curvature_episode,
                config_paths,
                stage1_only=False,
            )
        )
        scenario_rows.extend(
            evaluate_model(
                no_curvature_model,
                no_curvature_raw,
                "v35_no_curvature_stage2",
                no_curvature_episode,
                config_paths,
                stage1_only=False,
            )
        )
        scenario_rows.extend(
            evaluate_model(
                curvature_model,
                curvature_raw,
                "v35_curvature_stage1_only",
                curvature_episode,
                config_paths,
                stage1_only=True,
            )
        )
    finally:
        curvature_model.close()
        no_curvature_model.close()

    summary_rows = summarize_rows(scenario_rows)
    write_csv(args.output_root / "scalability_per_scenario.csv", scenario_rows)
    write_csv(args.output_root / "scalability_summary.csv", summary_rows)
    plot_paths = plot_scalability(summary_rows, args.output_root)
    metadata = {
        "evaluation": "V3.5 ch2 community N-generalization with Stage-1-only overlay",
        "output_root": str(args.output_root),
        "models": {
            "v35_curvature_stage2": str(args.curvature_model),
            "v35_no_curvature_stage2": str(args.no_curvature_model),
            "v35_curvature_stage1_only": "same curvature checkpoint; predict_stage1_probabilities",
        },
        "checkpoint_paths": {
            "v35_curvature": str(bmax_checkpoint_path(args.curvature_model)),
            "v35_no_curvature": str(bmax_checkpoint_path(args.no_curvature_model)),
        },
        "checkpoint_episodes": {
            "v35_curvature": curvature_episode,
            "v35_no_curvature": no_curvature_episode,
        },
        "line_order": list(MODEL_ORDER),
        "node_counts": list(scalability.NODE_COUNTS),
        "scenarios_per_node_count": scalability.SCENARIOS_PER_NODE_COUNT,
        "slots_per_scenario": scalability.SLOTS_PER_SCENARIO,
        "evaluation_master_seed": scalability.EVALUATION_MASTER_SEED,
        "topology": "community only",
        "physics_source": str(args.curvature_model / "training_config.yaml"),
        "geometry_policy": "area and community geometry scale as sqrt(N/100); communication radius is fixed",
        "mean_broadcast_probability": "summary mean of per-scenario action_prob_mean",
        "annotation_fontsize": FONT_SIZES["legend"],
        "plots": [str(path) for path in plot_paths],
    }
    with (args.output_root / "scalability_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    write_readme(args.output_root)
    print("Saved V3.5 ch2 scalability comparison to {}".format(args.output_root))


if __name__ == "__main__":
    main()
