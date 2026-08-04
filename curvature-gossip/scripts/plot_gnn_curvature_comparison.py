"""绘制 GNN 曲率消融的性能对比图。

功能：读取四个 MPNN 实验目录的固定验证结果，分别选取验证集平均 VAoI
最低的 checkpoint，生成一张包含绝对 VAoI 柱状图和曲率相对收益图的对比图。
图中误差线是 10 个固定验证场景的 95% 置信区间；同时导出汇总 CSV，便于
复核图中的数值和 checkpoint 选择。
"""

from __future__ import division

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result_GNN"
OUTPUT_ROOT = RESULT_ROOT / "curvature_comparison"

# The two topology pairs use the same validation seed layout within each pair.
EXPERIMENTS = {
    "community": {
        "curvature": RESULT_ROOT / "mpnnV3_heuristic_channel_n100_u0.20_b0.10_a1.5_lowLR",
        "no_curvature": RESULT_ROOT / "mpnnV3_no_curvature_heuristic_channel_n100_u0.20_b0.10",
    },
    "random": {
        "curvature": RESULT_ROOT / "mpnnV3_random_n100_u0.20_b0.10_a1.5_lowLR",
        "no_curvature": RESULT_ROOT / "mpnnV3_no_curvature_random_n100_u0.20_b0.10",
    },
}

TOPOLOGY_LABELS = {
    "community": "Community topology",
    "random": "Random-geometric topology",
}

MODEL_LABELS = {
    "curvature": "Curvature MPNN",
    "no_curvature": "No-curvature MPNN",
    "matched_random": "Matched random",
}

MODEL_COLORS = {
    "curvature": "#2F6B9A",
    "no_curvature": "#8C8C8C",
    "matched_random": "#E6A23C",
}

# Font sizes are presentation-oriented so the exported figure remains readable
# when inserted into a PowerPoint slide.
FONT_SIZES = {
    "suptitle": 16,
    "panel_title": 15,
    "axis_label": 13.5,
    "tick_label": 12.5,
    "legend": 11.5,
    "annotation": 11,
}


def mean_and_ci95(values):
    """Return mean, sample count, and a normal-approximation 95% CI."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarize an empty validation result")
    mean = float(np.mean(array))
    if array.size == 1:
        ci95 = 0.0
    else:
        ci95 = float(1.96 * np.std(array, ddof=1) / math.sqrt(array.size))
    return mean, int(array.size), ci95


def read_best_checkpoint(experiment_dir):
    """Select the checkpoint with the lowest fixed-validation NN mean VAoI."""
    history_path = experiment_dir / "validation_history.csv"
    history = pd.read_csv(history_path)
    history["nn_mean_VAoI"] = pd.to_numeric(history["nn_mean_VAoI"])
    best_row = history.loc[history["nn_mean_VAoI"].idxmin()]
    return int(best_row["checkpoint_episode"]), float(best_row["nn_mean_VAoI"])


def read_policy_values(experiment_dir, checkpoint_episode, policy):
    """Read one policy's per-scenario VAoI and transmission-rate values."""
    path = experiment_dir / "validation_per_scenario.csv"
    scenarios = pd.read_csv(path)
    scenarios["checkpoint_episode"] = pd.to_numeric(scenarios["checkpoint_episode"])
    selected = scenarios[
        (scenarios["checkpoint_episode"] == checkpoint_episode)
        & (scenarios["policy"] == policy)
    ].copy()
    if selected.empty:
        raise ValueError(
            "no rows for policy '{}' at checkpoint {} in {}".format(
                policy, checkpoint_episode, path
            )
        )
    return {
        "vaoi": selected["mean_VAoI"].astype(float).to_numpy(),
        "actual_tx_ratio": selected["actual_tx_ratio"].astype(float).to_numpy(),
        "scenario_id": selected["scenario_id"].astype(str).to_numpy(),
    }


def summarize_policy(topology, model, experiment_dir, checkpoint, policy, values):
    """Build one output row for the absolute VAoI comparison."""
    mean_vaoi, n_scenarios, ci95_vaoi = mean_and_ci95(values["vaoi"])
    mean_tx, _, ci95_tx = mean_and_ci95(values["actual_tx_ratio"])
    return {
        "topology": topology,
        "topology_label": TOPOLOGY_LABELS[topology],
        "model": model,
        "label": MODEL_LABELS[model],
        "policy": policy,
        "experiment_dir": str(experiment_dir),
        "checkpoint_episode": checkpoint,
        "n_scenarios": n_scenarios,
        "mean_vaoi": mean_vaoi,
        "ci95_vaoi": ci95_vaoi,
        "mean_actual_tx_ratio": mean_tx,
        "ci95_actual_tx_ratio": ci95_tx,
    }


def relative_gain(values, reference_values):
    """Compute per-scenario percentage reduction relative to a reference."""
    if len(values) != len(reference_values):
        raise ValueError("paired validation arrays must have equal length")
    reference = np.asarray(reference_values, dtype=float)
    if np.any(reference == 0.0):
        raise ValueError("reference VAoI contains zero, cannot compute relative gain")
    return 100.0 * (np.asarray(reference_values) - np.asarray(values)) / reference


def align_scenario_values(values, reference):
    """Align two policy arrays by scenario ID before computing paired gains."""
    reference_index = {
        scenario_id: index
        for index, scenario_id in enumerate(reference["scenario_id"])
    }
    if set(values["scenario_id"]) != set(reference["scenario_id"]):
        raise ValueError("paired validation policies use different scenario IDs")
    reference_order = [reference_index[scenario_id] for scenario_id in values["scenario_id"]]
    return values["vaoi"], reference["vaoi"][reference_order]


def collect_results():
    """Collect absolute comparison rows and paired relative-gain rows."""
    absolute_rows = []
    gain_rows = []

    for topology, pair in EXPERIMENTS.items():
        curvature_dir = pair["curvature"]
        no_curvature_dir = pair["no_curvature"]
        curvature_checkpoint, _ = read_best_checkpoint(curvature_dir)
        no_curvature_checkpoint, _ = read_best_checkpoint(no_curvature_dir)

        # Evaluate both policies at their own best checkpoint; matched random is
        # taken from the curvature run at the same checkpoint and rate.
        curvature_nn = read_policy_values(curvature_dir, curvature_checkpoint, "nn")
        curvature_matched = read_policy_values(
            curvature_dir, curvature_checkpoint, "matched_random"
        )
        no_curvature_nn = read_policy_values(
            no_curvature_dir, no_curvature_checkpoint, "nn"
        )

        absolute_rows.extend(
            [
                summarize_policy(
                    topology,
                    "curvature",
                    curvature_dir,
                    curvature_checkpoint,
                    "nn",
                    curvature_nn,
                ),
                summarize_policy(
                    topology,
                    "no_curvature",
                    no_curvature_dir,
                    no_curvature_checkpoint,
                    "nn",
                    no_curvature_nn,
                ),
                summarize_policy(
                    topology,
                    "matched_random",
                    curvature_dir,
                    curvature_checkpoint,
                    "matched_random",
                    curvature_matched,
                ),
            ]
        )

        # Paired differences use the same fixed scenario IDs, which makes the
        # right panel focus on the effect of curvature rather than topology noise.
        curvature_values, no_curvature_values = align_scenario_values(
            curvature_nn, no_curvature_nn
        )
        curvature_values_for_random, matched_values = align_scenario_values(
            curvature_nn, curvature_matched
        )
        curvature_vs_no_curvature = relative_gain(
            curvature_values, no_curvature_values
        )
        curvature_vs_matched = relative_gain(
            curvature_values_for_random, matched_values
        )
        for comparison, gains in [
            ("vs_no_curvature", curvature_vs_no_curvature),
            ("vs_matched_random", curvature_vs_matched),
        ]:
            mean_gain, n_scenarios, ci95_gain = mean_and_ci95(gains)
            gain_rows.append(
                {
                    "topology": topology,
                    "topology_label": TOPOLOGY_LABELS[topology],
                    "comparison": comparison,
                    "label": {
                        "vs_no_curvature": "Curvature vs no-curvature",
                        "vs_matched_random": "Curvature vs matched random",
                    }[comparison],
                    "checkpoint_curvature": curvature_checkpoint,
                    "checkpoint_no_curvature": no_curvature_checkpoint,
                    "n_scenarios": n_scenarios,
                    "mean_gain_percent": mean_gain,
                    "ci95_gain_percent": ci95_gain,
                }
            )

    return absolute_rows, gain_rows


def write_csv(path, rows):
    """Write dictionaries with a stable, human-readable column order."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def annotate_bars(axis, bars, values, errors, formatter):
    """Add compact value labels above bars, including their CI-aware offset."""
    for bar, value, error in zip(bars, values, errors):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + error + formatter["offset"],
            formatter["text"].format(value),
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["annotation"],
        )


def plot_comparison(absolute_rows, gain_rows):
    """Create the two-panel absolute-performance and relative-gain figure."""
    topology_order = ["community", "random"]
    model_order = ["curvature", "no_curvature", "matched_random"]
    gain_order = ["vs_no_curvature", "vs_matched_random"]
    x = np.arange(len(topology_order), dtype=float)

    figure, (absolute_axis, gain_axis) = plt.subplots(
        1, 2, figsize=(13.5, 6.4), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    # Panel (a): the user's proposed three-bar comparison for each topology.
    width = 0.22
    for index, model in enumerate(model_order):
        rows = [
            row
            for topology in topology_order
            for row in absolute_rows
            if row["topology"] == topology and row["model"] == model
        ]
        values = [row["mean_vaoi"] for row in rows]
        errors = [row["ci95_vaoi"] for row in rows]
        bars = absolute_axis.bar(
            x + (index - 1) * width,
            values,
            width,
            yerr=errors,
            capsize=4,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.7,
            label=MODEL_LABELS[model],
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        annotate_bars(
            absolute_axis,
            bars,
            values,
            errors,
            {"offset": 0.08, "text": "{:.2f}"},
        )

    absolute_axis.set_title(
        "(a) Absolute validation performance",
        fontsize=FONT_SIZES["panel_title"],
        pad=14,
    )
    absolute_axis.set_ylabel(
        "Mean VAoI (lower is better)", fontsize=FONT_SIZES["axis_label"]
    )
    absolute_axis.set_xticks(x)
    absolute_axis.set_xticklabels(
        [TOPOLOGY_LABELS[item] for item in topology_order],
        fontsize=FONT_SIZES["tick_label"],
    )
    absolute_axis.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"])
    absolute_axis.set_ylim(0.0, max(row["mean_vaoi"] + row["ci95_vaoi"] for row in absolute_rows) * 1.18)
    absolute_axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    absolute_axis.set_axisbelow(True)

    # Panel (b): paired percentage reductions make the curvature effect visible.
    gain_colors = {"vs_no_curvature": "#6A4C93", "vs_matched_random": "#D97732"}
    gain_width = 0.30
    for index, comparison in enumerate(gain_order):
        rows = [
            row
            for topology in topology_order
            for row in gain_rows
            if row["topology"] == topology and row["comparison"] == comparison
        ]
        values = [row["mean_gain_percent"] for row in rows]
        errors = [row["ci95_gain_percent"] for row in rows]
        bars = gain_axis.bar(
            x + (index - 0.5) * gain_width,
            values,
            gain_width,
            yerr=errors,
            capsize=4,
            color=gain_colors[comparison],
            edgecolor="white",
            linewidth=0.7,
            label=rows[0]["label"],
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        annotate_bars(
            gain_axis,
            bars,
            values,
            errors,
            {"offset": 0.6, "text": "{:.1f}%"},
        )

    gain_axis.axhline(0.0, color="#444444", linewidth=0.8)
    gain_axis.set_title(
        "(b) Curvature relative gain",
        fontsize=FONT_SIZES["panel_title"],
        pad=14,
    )
    gain_axis.set_ylabel(
        "VAoI reduction (%)", fontsize=FONT_SIZES["axis_label"]
    )
    gain_axis.set_xticks(x)
    gain_axis.set_xticklabels(
        [TOPOLOGY_LABELS[item] for item in topology_order],
        fontsize=FONT_SIZES["tick_label"],
    )
    gain_axis.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"])
    gain_lows = [row["mean_gain_percent"] - row["ci95_gain_percent"] for row in gain_rows]
    gain_highs = [row["mean_gain_percent"] + row["ci95_gain_percent"] for row in gain_rows]
    gain_axis.set_ylim(min(0.0, min(gain_lows) - 1.0), max(gain_highs) * 1.18)
    gain_axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    gain_axis.set_axisbelow(True)
    # Keep the legend away from the tallest community gain annotation.
    gain_axis.legend(
        loc="upper right", fontsize=FONT_SIZES["legend"], frameon=True
    )

    # Keep the shared figure readable when labels and legends are expanded.
    absolute_axis.legend(
        loc="upper right", fontsize=FONT_SIZES["legend"], frameon=True, ncol=1
    )
    figure.suptitle(
        "MPNN curvature ablation at n=100, u=0.20, b=0.10\n"
        "Best fixed-validation checkpoint; error bars show 95% CI over 10 scenarios",
        fontsize=FONT_SIZES["suptitle"],
    )
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.84])
    png_path = OUTPUT_ROOT / "gnn_curvature_performance_comparison.png"
    pdf_path = OUTPUT_ROOT / "gnn_curvature_performance_comparison.pdf"
    figure.savefig(str(png_path), dpi=220, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    absolute_rows, gain_rows = collect_results()
    write_csv(OUTPUT_ROOT / "curvature_comparison_data.csv", absolute_rows)
    write_csv(OUTPUT_ROOT / "curvature_relative_gains.csv", gain_rows)
    png_path, pdf_path = plot_comparison(absolute_rows, gain_rows)
    print("Saved absolute data: {}".format(OUTPUT_ROOT / "curvature_comparison_data.csv"))
    print("Saved relative gains: {}".format(OUTPUT_ROOT / "curvature_relative_gains.csv"))
    print("Saved PNG: {}".format(png_path))
    print("Saved PDF: {}".format(pdf_path))
    for row in absolute_rows:
        print(
            "{} | {} | checkpoint={} | VAoI={:.4f} +/- {:.4f}".format(
                row["topology"], row["label"], row["checkpoint_episode"],
                row["mean_vaoi"], row["ci95_vaoi"]
            )
        )


if __name__ == "__main__":
    main()
