"""比较社区结构训练的曲率模型与无曲率消融在随机 N=100 场景上的泛化。

功能：恢复指定的社区训练 checkpoint，把两个模型放到同一批 N=100
随机几何验证场景中进行配对推理，统计每个场景的 VAoI、发送率和动作概率，
并生成逐场景对比图与均值/95%置信区间图。脚本不训练、不修改模型目录，
结果写入曲率模型目录下的 random_n100_generalization 子目录。
"""

from __future__ import division

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
EVALUATION_CONFIG = RESULT_ROOT / "0804Scalability_N" / "configs" / "random" / "n100.yaml"
OUTPUT_ROOT = RESULT_ROOT / "mpnnV3_heuristic_channel_n100_u0.20_b0.10_a1.5" / "random_n100_generalization"

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvature_gossip.learning.ctde_ppo import CTDEPPO, resolve_learning_rates
from curvature_gossip.learning.trainer import _critic_config, _stage1_actor_config
from curvature_gossip.learning.validation import _run_neural_scenario, validation_scenarios


MODEL_DIRS = {
    "curvature": RESULT_ROOT / "mpnnV3_heuristic_channel_n100_u0.20_b0.10_a1.5",
    "no_curvature": RESULT_ROOT / "mpnnV3_no_curvature_heuristic_channel_n100_u0.20_b0.10",
    "random_curvature": RESULT_ROOT / "mpnnV3_random_n100_u0.20_b0.10_a1.5_lowLR",
}

MODEL_LABELS = {
    "curvature": "Community-trained curvature MPNN",
    "no_curvature": "Community-trained no-curvature MPNN",
    "random_curvature": "Random-trained curvature MPNN",
}

MODEL_COLORS = {
    "curvature": "#2F6B9A",
    "no_curvature": "#8C8C8C",
    "random_curvature": "#E6A23C",
}

MODEL_ORDER = ("curvature", "no_curvature", "random_curvature")

FONT_SIZES = {
    "title": 17,
    "axis_label": 15,
    "tick": 13,
    "legend": 12,
    "annotation": 11,
}


def read_yaml(path):
    """Read a YAML mapping and fail with the source path on invalid content."""
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("expected a YAML mapping: {}".format(path))
    return value


def checkpoint_path(model_dir):
    """Return the best-validation checkpoint prefix."""
    path = model_dir / "checkpoints" / "best_validation" / "model"
    if not path.with_suffix(".index").is_file():
        raise FileNotFoundError("missing checkpoint: {}".format(path))
    return path


def checkpoint_episode(model_dir):
    """Read the saved checkpoint episode for provenance."""
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
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, std, ci95


def build_model(model_dir):
    """Restore one model using its own actor/critic architecture metadata."""
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
    model.restore(str(checkpoint_path(model_dir)))
    return model, raw


def merge_evaluation_config(model_raw, evaluation_raw):
    """Keep model architecture and overlay only common random-test settings."""
    merged = deepcopy(model_raw)
    for key in ("topology", "source", "channel", "curvature", "constraints", "observation", "validation"):
        merged[key] = deepcopy(evaluation_raw[key])
    merged["experiment"] = dict(merged.get("experiment", {}))
    merged["experiment"]["master_seed"] = int(evaluation_raw["experiment"].get("master_seed", 20260804))
    return merged


def evaluate_model(model, raw, model_name):
    """Run the restored model on every paired random N=100 scenario."""
    before = model.variable_snapshot()
    rows = []
    for scenario in validation_scenarios(raw):
        summary, stats, _ = _run_neural_scenario(model, raw, scenario, policy_name="nn")
        rows.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "scenario_id": scenario.scenario_id,
                "topology": raw["topology"]["type"],
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
        raise RuntimeError("cross-topology validation unexpectedly changed model variables")
    return rows


def summarize(rows):
    """Aggregate the paired random scenarios for each model."""
    output = []
    for model_name in MODEL_ORDER:
        group = [row for row in rows if row["model"] == model_name]
        mean_vaoi, std_vaoi, ci_vaoi = mean_std_ci([row["mean_VAoI"] for row in group])
        mean_tx, std_tx, ci_tx = mean_std_ci([row["actual_tx_ratio"] for row in group])
        mean_prob, std_prob, ci_prob = mean_std_ci([row["action_prob_mean"] for row in group])
        output.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "topology": "random_geometric",
                "node_count": 100,
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
    no_curvature = [row for row in output if row["model"] == "no_curvature"][0]
    for row in output:
        row["relative_VAoI_reduction_vs_no_curvature"] = (
            (no_curvature["mean_VAoI"] - row["mean_VAoI"]) / no_curvature["mean_VAoI"]
        )
    return output


def plot_comparison(rows, summary):
    """Create paired per-scenario and aggregate VAoI comparison panels."""
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    scenario_ids = sorted({row["scenario_id"] for row in rows})
    x = np.arange(len(scenario_ids))
    for model_name in MODEL_ORDER:
        values = [
            next(row["mean_VAoI"] for row in rows if row["model"] == model_name and row["scenario_id"] == scenario_id)
            for scenario_id in scenario_ids
        ]
        axes[0].plot(
            x,
            values,
            marker="o",
            linewidth=2.3,
            markersize=6,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )
    axes[0].set_title("Paired random N=100 scenarios", fontsize=FONT_SIZES["title"], pad=12)
    axes[0].set_xlabel("Scenario", fontsize=FONT_SIZES["axis_label"])
    axes[0].set_ylabel("Mean VAoI (lower is better)", fontsize=FONT_SIZES["axis_label"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenario_ids, rotation=35, ha="right", fontsize=FONT_SIZES["tick"])
    axes[0].tick_params(axis="y", labelsize=FONT_SIZES["tick"])
    axes[0].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axes[0].set_axisbelow(True)
    axes[0].legend(fontsize=FONT_SIZES["legend"], loc="best")

    labels = ["Community\ncurvature", "Community\nno-curvature", "Random\ncurvature"]
    summary_by_model = {row["model"]: row for row in summary}
    values = [float(summary_by_model[model]["mean_VAoI"]) for model in MODEL_ORDER]
    errors = [float(summary_by_model[model]["ci95_VAoI"]) for model in MODEL_ORDER]
    bars = axes[1].bar(
        np.arange(len(MODEL_ORDER)),
        values,
        yerr=errors,
        capsize=5,
        color=[MODEL_COLORS[model] for model in MODEL_ORDER],
    )
    reduction = float(summary_by_model["curvature"]["relative_VAoI_reduction_vs_no_curvature"])
    random_reduction = float(summary_by_model["random_curvature"]["relative_VAoI_reduction_vs_no_curvature"])
    axes[1].set_title(
        "Aggregate comparison\nReduction vs no-curvature: community {:.1%}, random {:.1%}".format(
            reduction, random_reduction
        ),
        fontsize=FONT_SIZES["title"],
        pad=12,
    )
    axes[1].set_xticks(np.arange(len(MODEL_ORDER)))
    axes[1].set_xticklabels(labels, fontsize=FONT_SIZES["tick"])
    axes[1].set_ylabel("Mean VAoI (lower is better)", fontsize=FONT_SIZES["axis_label"])
    axes[1].tick_params(axis="y", labelsize=FONT_SIZES["tick"])
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axes[1].set_axisbelow(True)
    for bar, value in zip(bars, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(errors) * 0.08 + 0.02,
            "{:.3f}".format(value),
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["annotation"],
        )
    figure.suptitle("Community-trained models on random-geometric N=100", fontsize=FONT_SIZES["title"] + 1)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    png_path = OUTPUT_ROOT / "random_n100_comparison.png"
    pdf_path = OUTPUT_ROOT / "random_n100_comparison.pdf"
    figure.savefig(str(png_path), dpi=240, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main():
    """Restore all comparison models, run paired random tests, and save outputs."""
    if not EVALUATION_CONFIG.is_file():
        raise FileNotFoundError("missing random N=100 evaluation config: {}".format(EVALUATION_CONFIG))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    evaluation_raw = read_yaml(EVALUATION_CONFIG)
    shutil.copy2(str(EVALUATION_CONFIG), str(OUTPUT_ROOT / "evaluation_config.yaml"))

    all_rows = []
    provenance = []
    for model_name in MODEL_ORDER:
        model_dir = MODEL_DIRS[model_name]
        model, model_raw = build_model(model_dir)
        merged_raw = merge_evaluation_config(model_raw, evaluation_raw)
        provenance.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "model_dir": str(model_dir),
                "checkpoint_episode": checkpoint_episode(model_dir),
            }
        )
        try:
            all_rows.extend(evaluate_model(model, merged_raw, model_name))
        finally:
            model.close()

    summary = summarize(all_rows)
    write_csv(OUTPUT_ROOT / "random_n100_per_scenario.csv", all_rows)
    write_csv(OUTPUT_ROOT / "random_n100_summary.csv", summary)
    plot_paths = plot_comparison(all_rows, summary)
    metadata = {
        "evaluation_config": str(EVALUATION_CONFIG),
        "output_root": str(OUTPUT_ROOT),
        "topology": "random_geometric",
        "node_count": 100,
        "scenarios_per_model": len(validation_scenarios(evaluation_raw)),
        "models": provenance,
        "plots": [str(path) for path in plot_paths],
        "paired_evaluation": True,
    }
    with (OUTPUT_ROOT / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    print("Saved community-model random N=100 comparison to {}".format(OUTPUT_ROOT))


if __name__ == "__main__":
    main()
