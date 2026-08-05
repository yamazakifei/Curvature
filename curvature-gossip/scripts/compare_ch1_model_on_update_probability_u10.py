"""比较 ch1 曲率模型在 0804Scalability_u 的 u=0.10 场景上的效果。

功能：恢复 ``mpnnV3_ch1_budget_n100_u0.10_b0.10_a1.5_lowLR``，复用
result_GNN/0804Scalability_u/configs 下 u=0.10 的社区和随机测试配置及
测试 seed，分别进行配对推理。结果同时读取同一批场景已有的曲率、无曲率
和 matched-random 基线，生成逐场景 CSV、汇总 CSV、置信区间图和元数据。
脚本只做推理，不训练或修改原模型目录；输出写入 0804Scalability_u 下的
新子目录。
"""

from __future__ import division

import csv
import argparse
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
SWEEP_ROOT = RESULT_ROOT / "0804Scalability_u"
OUTPUT_ROOT = SWEEP_ROOT / "ch1_u10_comparison"
BASELINE_ROWS = SWEEP_ROOT / "update_probability_per_scenario.csv"

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvature_gossip.learning.ctde_ppo import CTDEPPO, resolve_learning_rates
from curvature_gossip.learning.trainer import _critic_config, _stage1_actor_config
from curvature_gossip.learning.validation import _run_neural_scenario, validation_scenarios


CH1_MODEL_DIR = RESULT_ROOT / "mpnnV3_ch1_budget_n100_u0.10_b0.10_a1.5_lowLR"
TOPOLOGY_NAMES = ("community", "random")
TOPOLOGY_LABELS = {
    "community": "Community topology",
    "random": "Random-geometric topology",
}

# Existing rows were produced by the u=0.10 scalability sweep.  The new ch1
# model is evaluated separately on the identical scenarios and then merged.
COMPARISON_ORDER = (
    "ch1_curvature",
    "baseline_curvature",
    "no_curvature",
    "matched_random",
)
MODEL_LABELS = {
    "ch1_curvature": "Ch1 curvature MPNN",
    "baseline_curvature": "Existing curvature MPNN",
    "no_curvature": "No-curvature MPNN",
    "matched_random": "Matched random",
}
MODEL_COLORS = {
    "ch1_curvature": "#2F6B9A",
    "baseline_curvature": "#4C956C",
    "no_curvature": "#8C8C8C",
    "matched_random": "#E6A23C",
}
FONT_SIZES = {
    "title": 17,
    "axis_label": 15,
    "tick": 13,
    "legend": 11,
    "annotation": 10,
}


def read_yaml(path):
    """Read one YAML mapping and report the source path on failure."""
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("expected a YAML mapping: {}".format(path))
    return value


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
    """Return mean, sample standard deviation and normal-approximation 95% CI."""
    values = np.asarray(tuple(values), dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, std, ci95


def checkpoint_path(model_dir):
    """Return the best-validation checkpoint prefix."""
    path = model_dir / "checkpoints" / "best_validation" / "model"
    if not path.with_suffix(".index").is_file():
        raise FileNotFoundError("missing checkpoint: {}".format(path))
    return path


def checkpoint_episode(model_dir):
    """Read checkpoint provenance."""
    info_path = model_dir / "checkpoints" / "best_validation" / "checkpoint_info.json"
    if not info_path.is_file():
        return -1
    with info_path.open("r", encoding="utf-8") as stream:
        return int(json.load(stream).get("episode", -1))


def build_model():
    """Restore the ch1 model with its own architecture and checkpoint metadata."""
    raw = read_yaml(CH1_MODEL_DIR / "training_config.yaml")
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
    model.restore(str(checkpoint_path(CH1_MODEL_DIR)))
    return model, raw


def merge_evaluation_config(model_raw, evaluation_raw):
    """Keep ch1 architecture while replacing only test topology/scenario settings."""
    merged = deepcopy(model_raw)
    for key in ("topology", "source", "channel", "curvature", "constraints", "observation", "validation"):
        merged[key] = deepcopy(evaluation_raw[key])
    merged["experiment"] = dict(merged.get("experiment", {}))
    merged["experiment"]["master_seed"] = int(evaluation_raw["experiment"].get("master_seed", 20260804))
    return merged


def evaluate_ch1_on_configs(model, model_raw):
    """Run ch1 on both u=0.10 topology configurations with identical seeds."""
    rows = []
    before = model.variable_snapshot()
    for topology_name in TOPOLOGY_NAMES:
        config_path = SWEEP_ROOT / "configs" / topology_name / "u10.yaml"
        evaluation_raw = read_yaml(config_path)
        merged = merge_evaluation_config(model_raw, evaluation_raw)
        for scenario in validation_scenarios(merged):
            summary, stats, _ = _run_neural_scenario(model, merged, scenario, policy_name="nn")
            rows.append(
                {
                    "topology": topology_name,
                    "topology_label": TOPOLOGY_LABELS[topology_name],
                    "scenario_id": scenario.scenario_id,
                    "model_key": "ch1_curvature",
                    "model_label": MODEL_LABELS["ch1_curvature"],
                    "source": "new_ch1_model",
                    "policy": "nn",
                    "node_count": scenario.n_nodes,
                    "slots": scenario.slots,
                    "update_probability": scenario.update_probability,
                    "target_tx_ratio": scenario.target_tx_ratio,
                    "mean_VAoI": float(summary["mean_VAoI"]),
                    "actual_tx_ratio": float(summary["avg_tx_per_slot"] / scenario.n_nodes),
                    "action_prob_mean": float(stats["action_prob_mean"]),
                    "action_prob_std": float(stats["action_prob_std"]),
                }
            )
    after = model.variable_snapshot()
    if len(before) != len(after) or any(not np.array_equal(old, new) for old, new in zip(before, after)):
        raise RuntimeError("ch1 validation unexpectedly changed model variables")
    return rows


def load_baseline_rows():
    """Load existing u=0.10 rows and map them to comparison labels."""
    if not BASELINE_ROWS.is_file():
        raise FileNotFoundError("missing baseline rows: {}".format(BASELINE_ROWS))
    output = []
    with BASELINE_ROWS.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if abs(float(row["update_probability"]) - 0.10) > 1e-9:
                continue
            topology_name = row["topology"]
            if row["policy"] == "nn" and row["model"] == "curvature":
                model_key = "baseline_curvature"
                source = "existing_u10_sweep"
            elif row["policy"] == "nn" and row["model"] == "no_curvature":
                model_key = "no_curvature"
                source = "existing_u10_sweep"
            elif row["policy"] == "matched_random" and row["model"] == "curvature":
                model_key = "matched_random"
                source = "existing_u10_sweep"
            else:
                continue
            output.append(
                {
                    "topology": topology_name,
                    "topology_label": TOPOLOGY_LABELS[topology_name],
                    "scenario_id": row["scenario_id"],
                    "model_key": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "source": source,
                    "policy": row["policy"],
                    "node_count": int(row["n_nodes"]),
                    "slots": int(row["slots"]),
                    "update_probability": float(row["update_probability"]),
                    "target_tx_ratio": float(row["target_tx_ratio"]),
                    "mean_VAoI": float(row["mean_VAoI"]),
                    "actual_tx_ratio": float(row["actual_tx_ratio"]),
                    "action_prob_mean": float(row["action_prob_mean"]),
                    "action_prob_std": float(row["action_prob_std"]),
                }
            )
    return output


def summarize(rows):
    """Aggregate five scenarios per topology and comparison model."""
    output = []
    for topology_name in TOPOLOGY_NAMES:
        for model_key in COMPARISON_ORDER:
            group = [
                row for row in rows
                if row["topology"] == topology_name and row["model_key"] == model_key
            ]
            if len(group) != 5:
                raise ValueError("expected 5 rows for {} / {}, got {}".format(topology_name, model_key, len(group)))
            mean_vaoi, std_vaoi, ci_vaoi = mean_std_ci([row["mean_VAoI"] for row in group])
            mean_tx, std_tx, ci_tx = mean_std_ci([row["actual_tx_ratio"] for row in group])
            mean_prob, std_prob, ci_prob = mean_std_ci([row["action_prob_mean"] for row in group])
            output.append(
                {
                    "topology": topology_name,
                    "topology_label": TOPOLOGY_LABELS[topology_name],
                    "model_key": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "node_count": 100,
                    "update_probability": 0.10,
                    "n_scenarios": len(group),
                    "mean_VAoI": mean_vaoi,
                    "std_VAoI": std_vaoi,
                    "ci95_VAoI": ci_vaoi,
                    "mean_actual_tx_ratio": mean_tx,
                    "ci95_actual_tx_ratio": ci_tx,
                    "mean_action_probability": mean_prob,
                    "ci95_action_probability": ci_prob,
                }
            )
    for topology_name in TOPOLOGY_NAMES:
        baseline = next(
            row["mean_VAoI"] for row in output
            if row["topology"] == topology_name and row["model_key"] == "no_curvature"
        )
        for row in output:
            if row["topology"] == topology_name:
                row["relative_VAoI_reduction_vs_no_curvature"] = (baseline - row["mean_VAoI"]) / baseline
    return output


def plot_comparison(rows, summary):
    """Plot per-scenario curves and aggregate bars for both topologies."""
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 10.2), gridspec_kw={"height_ratios": [1.05, 1.0]})
    for column, topology_name in enumerate(TOPOLOGY_NAMES):
        topology_rows = [row for row in rows if row["topology"] == topology_name]
        scenario_ids = sorted({row["scenario_id"] for row in topology_rows})
        x = np.arange(len(scenario_ids))
        for model_key in COMPARISON_ORDER:
            values = [
                next(row["mean_VAoI"] for row in topology_rows
                     if row["model_key"] == model_key and row["scenario_id"] == scenario_id)
                for scenario_id in scenario_ids
            ]
            axes[0, column].plot(
                x, values, marker="o", linewidth=2.0, markersize=5.5,
                color=MODEL_COLORS[model_key], label=MODEL_LABELS[model_key],
            )
        axes[0, column].set_title(TOPOLOGY_LABELS[topology_name], fontsize=FONT_SIZES["title"], pad=10)
        axes[0, column].set_xlabel("Scenario", fontsize=FONT_SIZES["axis_label"])
        axes[0, column].set_ylabel("Mean VAoI (lower is better)", fontsize=FONT_SIZES["axis_label"])
        axes[0, column].set_xticks(x)
        axes[0, column].set_xticklabels(scenario_ids, rotation=30, ha="right", fontsize=FONT_SIZES["tick"])
        axes[0, column].tick_params(axis="y", labelsize=FONT_SIZES["tick"])
        axes[0, column].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        axes[0, column].set_axisbelow(True)
        axes[0, column].legend(fontsize=FONT_SIZES["legend"], loc="best")

        summary_by_model = {
            row["model_key"]: row for row in summary if row["topology"] == topology_name
        }
        values = [float(summary_by_model[key]["mean_VAoI"]) for key in COMPARISON_ORDER]
        errors = [float(summary_by_model[key]["ci95_VAoI"]) for key in COMPARISON_ORDER]
        mean_probabilities = [
            float(summary_by_model[key]["mean_action_probability"])
            for key in COMPARISON_ORDER
        ]
        bars = axes[1, column].bar(
            np.arange(len(COMPARISON_ORDER)), values, yerr=errors, capsize=4,
            color=[MODEL_COLORS[key] for key in COMPARISON_ORDER],
        )
        axes[1, column].set_title("Aggregate comparison", fontsize=FONT_SIZES["title"], pad=10)
        axes[1, column].set_xticks(np.arange(len(COMPARISON_ORDER)))
        axes[1, column].set_xticklabels(
            ["Ch1\ncurvature", "Existing\ncurvature", "No-\ncurvature", "Matched\nrandom"],
            fontsize=FONT_SIZES["tick"],
        )
        axes[1, column].set_ylabel("Mean VAoI (lower is better)", fontsize=FONT_SIZES["axis_label"])
        axes[1, column].tick_params(axis="y", labelsize=FONT_SIZES["tick"])
        axes[1, column].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        axes[1, column].set_axisbelow(True)
        # Include the average actor broadcast probability beside each VAoI bar.
        for bar, value, mean_probability in zip(bars, values, mean_probabilities):
            axes[1, column].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(errors) * 0.08 + 0.01,
                "VAoI={:.3f}\np={:.3f}".format(value, mean_probability),
                ha="center", va="bottom",
                fontsize=FONT_SIZES["annotation"],
            )
    figure.suptitle("Ch1 model comparison at u=0.10 on the existing test seeds", fontsize=FONT_SIZES["title"] + 1)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = OUTPUT_ROOT / "ch1_u10_comparison.png"
    pdf_path = OUTPUT_ROOT / "ch1_u10_comparison.pdf"
    figure.savefig(str(png_path), dpi=240, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main():
    """Run ch1 inference, merge baselines, summarize, plot, and save metadata."""
    global CH1_MODEL_DIR, OUTPUT_ROOT

    parser = argparse.ArgumentParser(
        description="Compare a Ch1 checkpoint with the existing u=0.10 baselines."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=CH1_MODEL_DIR,
        help="Ch1 model directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Output directory for this comparison (default: %(default)s)",
    )
    args = parser.parse_args()

    # Variant runs override only the model and output paths; all test seeds and
    # baseline rows remain shared with the original u=0.10 scalability sweep.
    CH1_MODEL_DIR = args.model_dir if args.model_dir.is_absolute() else PROJECT_ROOT / args.model_dir
    OUTPUT_ROOT = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    if not CH1_MODEL_DIR.is_dir():
        raise FileNotFoundError("missing ch1 model directory: {}".format(CH1_MODEL_DIR))
    if not BASELINE_ROWS.is_file():
        raise FileNotFoundError("missing u=0.10 baseline rows: {}".format(BASELINE_ROWS))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    copied_configs = {}
    for topology_name in TOPOLOGY_NAMES:
        source = SWEEP_ROOT / "configs" / topology_name / "u10.yaml"
        if not source.is_file():
            raise FileNotFoundError("missing u=0.10 config: {}".format(source))
        target = OUTPUT_ROOT / "u10_{}_config.yaml".format(topology_name)
        shutil.copy2(str(source), str(target))
        copied_configs[topology_name] = str(target)

    model, model_raw = build_model()
    try:
        ch1_rows = evaluate_ch1_on_configs(model, model_raw)
    finally:
        model.close()
    rows = ch1_rows + load_baseline_rows()
    summary = summarize(rows)
    write_csv(OUTPUT_ROOT / "ch1_u10_comparison_per_scenario.csv", rows)
    write_csv(OUTPUT_ROOT / "ch1_u10_comparison_summary.csv", summary)
    plot_paths = plot_comparison(rows, summary)
    metadata = {
        "ch1_model_dir": str(CH1_MODEL_DIR),
        "ch1_checkpoint_episode": checkpoint_episode(CH1_MODEL_DIR),
        "baseline_rows": str(BASELINE_ROWS),
        "configs": copied_configs,
        "topologies": list(TOPOLOGY_NAMES),
        "node_count": 100,
        "update_probability": 0.10,
        "scenarios_per_topology": 5,
        "slots_per_scenario": 500,
        "comparison_order": list(COMPARISON_ORDER),
        "plots": [str(path) for path in plot_paths],
    }
    with (OUTPUT_ROOT / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    print("Saved ch1 u=0.10 comparison to {}".format(OUTPUT_ROOT))


if __name__ == "__main__":
    main()
