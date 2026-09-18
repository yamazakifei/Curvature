"""Plot the N=100 maximum-VAoI distributions for the three V3.5 ch2 modes.

功能：仅评估 N=100 的 5 个 community 场景，分别提取曲率 Stage-2、无曲率
Stage-2 和曲率 Stage-1-only 的逐 slot 最大 VAoI，绘制分布图，并保存每个
样本的最大 VAoI、分位数统计和平均 Actor 广播概率。不会重新运行 N=50--150
的完整 scalability sweep。

The maximum-VAoI distribution is the distribution of the per-slot maximum
version age over five scenarios and 200 slots per scenario.
"""

from __future__ import division

import argparse
import csv
import json
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
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for import_root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from curvature_gossip.learning.features import (
    encode_stage1_observations,
    encode_stage2_mpnn_observations,
    encode_stage2_observations,
)
from curvature_gossip.learning.validation import _build_simulator, validation_scenarios
import run_gnn_scalability_generalization as scalability
from compare_v35_ch2_scalability import (
    DEFAULT_CURVATURE_MODEL,
    DEFAULT_NO_CURVATURE_MODEL,
    DEFAULT_OUTPUT_ROOT,
    bmax_checkpoint_path,
    build_evaluation_config,
    build_model,
    checkpoint_episode,
    read_yaml,
)


MODEL_ORDER = (
    "v35_curvature_stage2",
    "v35_no_curvature_stage2",
    "v35_curvature_stage1_only",
)
MODEL_LABELS = {
    "v35_curvature_stage2": "Curvature Stage-2",
    "v35_no_curvature_stage2": "No-curvature Stage-2",
    "v35_curvature_stage1_only": "Curvature Stage-1-only",
}
MODEL_COLORS = {
    "v35_curvature_stage2": "#C05640",
    "v35_no_curvature_stage2": "#2A9D8F",
    "v35_curvature_stage1_only": "#2F6B9A",
}


def write_yaml(path, raw):
    """Write the exact N=100 evaluation configuration used by this plot."""
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(raw, stream, sort_keys=False, allow_unicode=True)


def overlay_evaluation_raw(model_raw, evaluation_raw):
    """Keep model architecture while applying the fixed N=100 environment."""
    merged = deepcopy(model_raw)
    for key in (
        "topology", "source", "channel", "curvature",
        "constraints", "observation", "validation",
    ):
        merged[key] = deepcopy(evaluation_raw[key])
    merged["experiment"] = dict(merged.get("experiment", {}))
    # Match the deterministic seed namespace used by the scalability sweep.
    merged["experiment"]["master_seed"] = scalability.EVALUATION_MASTER_SEED
    return merged


def encode_observations(model, model_raw, observations, scenario):
    """Encode one simulator observation using the model's persisted schema."""
    training = model_raw.get("training", {})
    if model.actor_stage == 1:
        return encode_stage1_observations(observations, scenario.target_tx_ratio)
    if model.actor_stage != 2:
        raise ValueError("unexpected actor stage: {}".format(model.actor_stage))

    residual = model_raw.get("actor", {}).get("residual", {})
    include_scenario_context = bool(residual.get("include_scenario_context", False))
    encoder_args = (
        observations,
        scenario.target_tx_ratio,
        scenario.update_probability,
        float(training.get("consecutive_tx_scale", 3.0)),
        float(training.get("neighbor_confidence_time_constant", 20.0)),
        float(model_raw.get("observation", {}).get("congestion_feature_scale", 5.0)),
        include_scenario_context,
    )
    if residual.get("architecture", "mlp") == "mpnn":
        mpnn = residual.get("mpnn", {})
        return encode_stage2_mpnn_observations(
            *encoder_args,
            use_curvature_edge_feature=bool(mpnn.get("use_curvature_edge_feature", False)),
            bmax=float(mpnn.get("bmax", 1.0)),
            edge_normalization=str(mpnn.get("edge_normalization", "raw_bmax")),
            edge_freshness_feature=str(mpnn.get("edge_freshness_feature", "binary_fraction")),
            version_gap_tau=mpnn.get("version_gap_tau"),
            use_curvature=model.actor_curvature_enabled,
        )
    return encode_stage2_observations(*encoder_args, use_curvature=model.actor_curvature_enabled)


def evaluate_mode(model, model_raw, model_key, scenarios, stage1_only):
    """Collect per-slot maximum VAoI and probabilities for one policy mode."""
    before = model.variable_snapshot()
    rows = []
    try:
        for scenario in scenarios:
            simulator = _build_simulator(model_raw, scenario, "nn", 0.0)
            probability_steps = []
            try:
                for slot in range(scenario.slots):
                    observations = simulator.begin_step(slot)
                    encoded = encode_observations(model, model_raw, observations, scenario)
                    probabilities = np.asarray(
                        model.predict_stage1_probabilities(encoded)
                        if stage1_only else model.predict_probabilities(encoded),
                        dtype=np.float32,
                    )
                    if probabilities.shape != (scenario.n_nodes,) or not np.isfinite(probabilities).all():
                        raise ValueError("Actor evaluation must return finite [N] probabilities")
                    actions = simulator.policy_rng.random(scenario.n_nodes) < probabilities
                    simulator.complete_step(actions)
                    probability_steps.append(probabilities)

                per_slot = simulator.metrics.per_slot
                if len(per_slot) != scenario.slots:
                    raise RuntimeError(
                        "expected {} per-slot metrics, got {}".format(scenario.slots, len(per_slot))
                    )
                probabilities = np.asarray(probability_steps, dtype=np.float32)
                for metric_row, probability_row in zip(per_slot, probabilities):
                    rows.append(
                        {
                            "model": model_key,
                            "model_label": MODEL_LABELS[model_key],
                            "scenario_id": scenario.scenario_id,
                            "slot": int(metric_row["slot"]),
                            "max_VAoI": float(metric_row["max_version_age"]),
                            "action_prob_mean": float(np.mean(probability_row)),
                        }
                    )
            finally:
                del simulator
    finally:
        after = model.variable_snapshot()
        if len(before) != len(after) or any(
            not np.array_equal(old, new) for old, new in zip(before, after)
        ):
            raise RuntimeError("maximum-VAoI validation changed model variables")
    return rows


def summarize(rows):
    """Summarize distribution quantiles and mean broadcast probability."""
    output = []
    for model_key in MODEL_ORDER:
        selected = [row for row in rows if row["model"] == model_key]
        values = np.asarray([row["max_VAoI"] for row in selected], dtype=float)
        probabilities = np.asarray([row["action_prob_mean"] for row in selected], dtype=float)
        output.append(
            {
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "n_scenarios": len(set(row["scenario_id"] for row in selected)),
                "n_slots": len(selected),
                "mean_max_VAoI": float(np.mean(values)),
                "std_max_VAoI": float(np.std(values, ddof=1)),
                "min_max_VAoI": float(np.min(values)),
                "p50_max_VAoI": float(np.percentile(values, 50)),
                "p90_max_VAoI": float(np.percentile(values, 90)),
                "p95_max_VAoI": float(np.percentile(values, 95)),
                "p99_max_VAoI": float(np.percentile(values, 99)),
                "max_of_max_VAoI": float(np.max(values)),
                "mean_action_probability": float(np.mean(probabilities)),
            }
        )
    return output


def write_csv(path, rows):
    """Write dictionaries as UTF-8 CSV."""
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_distribution(rows, summary_rows, output_root):
    """Plot violin distributions with boxplots and mean-p annotations."""
    distributions = [
        np.asarray([row["max_VAoI"] for row in rows if row["model"] == model_key], dtype=float)
        for model_key in MODEL_ORDER
    ]
    figure, axis = plt.subplots(figsize=(10.8, 7.0))
    positions = np.arange(1, len(MODEL_ORDER) + 1)
    violin = axis.violinplot(distributions, positions=positions, showmeans=False, showmedians=False, widths=0.78)
    for body, model_key in zip(violin["bodies"], MODEL_ORDER):
        body.set_facecolor(MODEL_COLORS[model_key])
        body.set_edgecolor(MODEL_COLORS[model_key])
        body.set_alpha(0.28)
    axis.boxplot(
        distributions,
        positions=positions,
        widths=0.18,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "#444444", "linewidth": 1.0},
        medianprops={"color": "#222222", "linewidth": 1.5},
        whiskerprops={"color": "#444444"},
        capprops={"color": "#444444"},
    )
    for position, model_key, summary_row in zip(positions, MODEL_ORDER, summary_rows):
        # Show the mean point and the mean broadcast probability above each violin.
        axis.scatter(
            [position],
            [summary_row["mean_max_VAoI"]],
            color=MODEL_COLORS[model_key],
            edgecolor="white",
            linewidth=0.8,
            s=42,
            zorder=4,
        )
        axis.annotate(
            "mean p={:.3f}".format(summary_row["mean_action_probability"]),
            xy=(position, summary_row["p95_max_VAoI"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=MODEL_COLORS[model_key],
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )

    labels = [
        "{}\n(p={:.3f})".format(summary_row["model_label"], summary_row["mean_action_probability"])
        for summary_row in summary_rows
    ]
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=11)
    axis.set_ylabel("Per-slot maximum VAoI", fontsize=14)
    axis.set_title("V3.5 ch2, N=100: maximum VAoI distribution", fontsize=16, pad=14)
    axis.text(
        0.01,
        0.99,
        "Violin: distribution; box: median and interquartile range; dot: mean",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#444444",
    )
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)
    figure.tight_layout()
    png_path = output_root / "max_vaoi_n100_distribution.png"
    pdf_path = output_root / "max_vaoi_n100_distribution.pdf"
    figure.savefig(str(png_path), dpi=240, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main():
    """Run only N=100, save maximum-VAoI samples, and plot distributions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curvature-model", type=Path, default=DEFAULT_CURVATURE_MODEL)
    parser.add_argument("--no-curvature-model", type=Path, default=DEFAULT_NO_CURVATURE_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    curvature_model, curvature_raw = build_model(args.curvature_model)
    no_curvature_model, no_curvature_raw = build_model(args.no_curvature_model)
    try:
        evaluation_raw = build_evaluation_config(curvature_raw, 100)
        config_path = args.output_root / "max_vaoi_n100_config.yaml"
        write_yaml(config_path, evaluation_raw)
        scenarios = validation_scenarios(evaluation_raw)
        rows = []
        curvature_episode = checkpoint_episode(args.curvature_model)
        no_curvature_episode = checkpoint_episode(args.no_curvature_model)
        curvature_eval_raw = overlay_evaluation_raw(curvature_raw, evaluation_raw)
        no_curvature_eval_raw = overlay_evaluation_raw(no_curvature_raw, evaluation_raw)
        rows.extend(evaluate_mode(
            curvature_model, curvature_eval_raw, "v35_curvature_stage2", scenarios, False
        ))
        rows.extend(evaluate_mode(
            no_curvature_model, no_curvature_eval_raw, "v35_no_curvature_stage2", scenarios, False
        ))
        rows.extend(evaluate_mode(
            curvature_model, curvature_eval_raw, "v35_curvature_stage1_only", scenarios, True
        ))
    finally:
        curvature_model.close()
        no_curvature_model.close()

    summary_rows = summarize(rows)
    write_csv(args.output_root / "max_vaoi_n100_per_slot.csv", rows)
    write_csv(args.output_root / "max_vaoi_n100_summary.csv", summary_rows)
    plot_paths = plot_distribution(rows, summary_rows, args.output_root)
    metadata = {
        "evaluation": "N=100 maximum-VAoI distribution for three V3.5 ch2 modes",
        "definition": "per-slot maximum_version_age over 5 scenarios and 200 slots per scenario",
        "node_count": 100,
        "scenarios_per_model": 5,
        "slots_per_scenario": 200,
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
        "plots": [str(path) for path in plot_paths],
        "data_files": [
            str(args.output_root / "max_vaoi_n100_per_slot.csv"),
            str(args.output_root / "max_vaoi_n100_summary.csv"),
        ],
    }
    with (args.output_root / "max_vaoi_n100_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    readme_path = args.output_root / "README.md"
    with readme_path.open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## N=100 maximum VAoI distribution\n\n"
            "`max_vaoi_n100_distribution.png` compares the per-slot maximum "
            "VAoI over five N=100 community scenarios and 200 slots per "
            "scenario for the three V3.5 ch2 modes. The x-axis labels and "
            "annotations include the mean Actor broadcast probability. Raw "
            "samples and percentile statistics are stored in the two CSV files.\n"
        )
    print("Saved N=100 maximum-VAoI distribution to {}".format(args.output_root))


if __name__ == "__main__":
    main()
