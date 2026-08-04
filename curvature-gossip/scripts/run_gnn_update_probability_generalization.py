"""运行并绘制 GNN 对源更新概率 u 的泛化性能。

功能：在原始 N=100 社区拓扑和随机几何拓扑配置下，将更新概率设为
0.05、0.10、0.15、0.20、0.25、0.30；每个概率使用 5 个固定验证场景，
恢复当前的曲率/无曲率 best-validation MPNN，并输出两张 VAoI 折线图。
曲率模型的 matched-random 作为第三条曲线，阴影区域表示 5 个场景的
近似 95% 置信区间。验证 YAML 和结果保存在 result_GNN/0804Scalability_u。
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the tested checkpoint restoration and lean validation path from the
# node-scalability sweep; this script only changes the sweep axis and YAMLs.
import scripts.run_gnn_scalability_generalization as scalability


RESULT_ROOT = PROJECT_ROOT / "result_GNN"
OUTPUT_ROOT = RESULT_ROOT / "0804Scalability_u"
CONFIG_ROOT = OUTPUT_ROOT / "configs"

UPDATE_PROBABILITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
NODE_COUNT = 100
SCENARIOS_PER_UPDATE_PROBABILITY = 5
DEFAULT_SLOTS = scalability.SLOTS_PER_SCENARIO
EVALUATION_MASTER_SEED = 20260804

MODEL_DIRS = scalability.MODEL_DIRS
TOPOLOGY_LABELS = scalability.TOPOLOGY_LABELS
PLOT_LABELS = scalability.PLOT_LABELS
PLOT_COLORS = scalability.PLOT_COLORS
FONT_SIZES = scalability.FONT_SIZES


def read_yaml(path):
    """Read one YAML mapping and fail clearly if the file is malformed."""
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("expected a YAML mapping: {}".format(path))
    return value


def build_scenarios(update_probability, topology_index, slots):
    """Create five paired fixed scenarios for one update probability."""
    probability_index = UPDATE_PROBABILITIES.index(float(update_probability))
    scenarios = []
    for scenario_index in range(SCENARIOS_PER_UPDATE_PROBABILITY):
        offset = topology_index * 10000 + probability_index * 100 + scenario_index
        scenarios.append(
            {
                "id": "u{:.2f}_s{:02d}".format(update_probability, scenario_index),
                "topology_seed": 95000 + offset,
                "source_update_seed": 96000 + offset,
                "channel_shadowing_seed": 97000 + offset,
                "channel_fading_seed": 98000 + offset,
                "policy_action_seed": 99000 + offset,
                "slots": int(slots),
                "n_nodes": NODE_COUNT,
                "update_probability": float(update_probability),
                "target_tx_ratio": 0.10,
            }
        )
    return scenarios


def build_evaluation_config(reference_raw, topology_name, update_probability, topology_index, slots):
    """Build one N=100 evaluation YAML while changing only source update rate."""
    raw = {
        "experiment": {"master_seed": EVALUATION_MASTER_SEED},
        "topology": deepcopy(reference_raw["topology"]),
        "source": deepcopy(reference_raw["source"]),
        "channel": deepcopy(reference_raw["channel"]),
        "curvature": deepcopy(reference_raw["curvature"]),
        "constraints": deepcopy(reference_raw["constraints"]),
        "observation": deepcopy(reference_raw["observation"]),
        "validation": {
            "scenarios": build_scenarios(
                update_probability, topology_index, slots
            )
        },
    }
    # Keep the original N=100 geometry and update only the source rate.
    raw["topology"]["params"]["n_nodes"] = NODE_COUNT
    if "cluster_sizes" in raw["topology"]["params"]:
        raw["topology"]["params"]["cluster_sizes"] = [50, 50]
    raw["source"]["update_probability"] = float(update_probability)
    raw["metadata"] = {
        "topology_name": topology_name,
        "node_count": NODE_COUNT,
        "update_probability": float(update_probability),
        "scenarios_per_update_probability": SCENARIOS_PER_UPDATE_PROBABILITY,
        "slots_per_scenario": int(slots),
        "change_policy": "only source update probability u changes; N=100 geometry is unchanged",
    }
    return raw


def write_evaluation_configs(reference_raw_by_topology, slots):
    """Write 12 YAMLs, grouped by topology, under the requested result root."""
    config_paths = {}
    for topology_index, topology_name in enumerate(("community", "random")):
        topology_dir = CONFIG_ROOT / topology_name
        topology_dir.mkdir(parents=True, exist_ok=True)
        config_paths[topology_name] = {}
        reference_raw = reference_raw_by_topology[topology_name]
        for update_probability in UPDATE_PROBABILITIES:
            raw = build_evaluation_config(
                reference_raw,
                topology_name,
                update_probability,
                topology_index,
                slots,
            )
            path = topology_dir / "u{:02d}.yaml".format(int(round(update_probability * 100)))
            with path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(raw, stream, sort_keys=False, allow_unicode=True)
            config_paths[topology_name][update_probability] = path
    return config_paths


def evaluate_all(config_paths, topology_names, slots):
    """Evaluate only the three policies needed for the update-rate curves."""
    scenario_rows = []
    provenance = []
    for topology_name in topology_names:
        curvature_dir = MODEL_DIRS[topology_name]["curvature"]
        no_curvature_dir = MODEL_DIRS[topology_name]["no_curvature"]
        curvature_model, curvature_raw = scalability.build_model(curvature_dir)
        no_curvature_model, no_curvature_raw = scalability.build_model(no_curvature_dir)
        provenance.extend(
            [
                {
                    "topology": topology_name,
                    "model": "curvature",
                    "model_dir": str(curvature_dir),
                    "checkpoint_episode": scalability.checkpoint_episode(curvature_dir),
                },
                {
                    "topology": topology_name,
                    "model": "no_curvature",
                    "model_dir": str(no_curvature_dir),
                    "checkpoint_episode": scalability.checkpoint_episode(no_curvature_dir),
                },
            ]
        )
        try:
            for update_probability in UPDATE_PROBABILITIES:
                evaluation_raw = read_yaml(
                    config_paths[topology_name][update_probability]
                )
                for model_name, model, model_raw in (
                    ("curvature", curvature_model, curvature_raw),
                    ("no_curvature", no_curvature_model, no_curvature_raw),
                ):
                    merged = deepcopy(model_raw)
                    for key in (
                        "topology", "source", "channel", "curvature",
                        "constraints", "observation", "validation",
                    ):
                        merged[key] = deepcopy(evaluation_raw[key])
                    merged["experiment"] = dict(merged.get("experiment", {}))
                    merged["experiment"]["master_seed"] = EVALUATION_MASTER_SEED
                    summary, rows = scalability.evaluate_required_policies(
                        model,
                        merged,
                        scalability.checkpoint_episode(
                            MODEL_DIRS[topology_name][model_name]
                        ),
                        include_matched_random=(model_name == "curvature"),
                    )
                    for row in rows:
                        tagged = dict(row)
                        tagged.update(
                            {
                                "topology": topology_name,
                                "model": model_name,
                                "update_probability": float(update_probability),
                            }
                        )
                        scenario_rows.append(tagged)
                    print(
                        "{} | u={:.2f} | {} | nn_VAoI={:.4f}".format(
                            topology_name,
                            update_probability,
                            model_name,
                            float(summary["nn_mean_VAoI"]),
                        )
                    )
        finally:
            curvature_model.close()
            no_curvature_model.close()
    return scenario_rows, provenance


def mean_ci(values):
    """Return mean, standard deviation, and approximate 95% confidence width."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, std, ci95


def summarize_rows(scenario_rows):
    """Aggregate the five validation scenarios at each update probability."""
    groups = {}
    for row in scenario_rows:
        key = (
            row["topology"],
            float(row["update_probability"]),
            row["model"],
            row["policy"],
        )
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for (topology, update_probability, model, policy), rows in sorted(groups.items()):
        vaoi_mean, vaoi_std, vaoi_ci = mean_ci(
            [row["mean_VAoI"] for row in rows]
        )
        tx_mean, tx_std, tx_ci = mean_ci(
            [row["actual_tx_ratio"] for row in rows]
        )
        summary_rows.append(
            {
                "topology": topology,
                "topology_label": TOPOLOGY_LABELS[topology],
                "node_count": NODE_COUNT,
                "update_probability": update_probability,
                "model": model,
                "policy": policy,
                "n_scenarios": len(rows),
                "mean_vaoi": vaoi_mean,
                "std_vaoi": vaoi_std,
                "ci95_vaoi": vaoi_ci,
                "mean_actual_tx_ratio": tx_mean,
                "std_actual_tx_ratio": tx_std,
                "ci95_actual_tx_ratio": tx_ci,
            }
        )
    return summary_rows


def write_csv(path, rows):
    """Write dictionaries to a UTF-8 CSV with stable column order."""
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


def plot_topology(summary_rows, topology_name, output_dir):
    """Plot one topology's VAoI curves against source update probability."""
    figure, axis = plt.subplots(figsize=(9.8, 6.4))
    x = np.asarray(UPDATE_PROBABILITIES, dtype=float)
    plot_order = [
        ("curvature", "nn"),
        ("no_curvature", "nn"),
        ("curvature", "matched_random"),
    ]
    for model, policy in plot_order:
        rows = [
            row
            for row in summary_rows
            if row["topology"] == topology_name
            and row["model"] == model
            and row["policy"] == policy
        ]
        rows = sorted(rows, key=lambda row: row["update_probability"])
        if len(rows) != len(UPDATE_PROBABILITIES):
            raise ValueError(
                "expected {} points for {} / {}, got {}".format(
                    len(UPDATE_PROBABILITIES), model, policy, len(rows)
                )
            )
        means = np.asarray([row["mean_vaoi"] for row in rows], dtype=float)
        ci95 = np.asarray([row["ci95_vaoi"] for row in rows], dtype=float)
        color = PLOT_COLORS[(model, policy)]
        axis.plot(
            x,
            means,
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=color,
            label=PLOT_LABELS[(model, policy)],
        )
        axis.fill_between(x, means - ci95, means + ci95, color=color, alpha=0.14)

    axis.set_title(
        "{} update-probability generalization".format(TOPOLOGY_LABELS[topology_name]),
        fontsize=FONT_SIZES["title"],
        pad=14,
    )
    axis.set_xlabel("Source update probability u", fontsize=FONT_SIZES["axis_label"])
    axis.set_ylabel("Mean VAoI (lower is better)", fontsize=FONT_SIZES["axis_label"])
    axis.set_xticks(UPDATE_PROBABILITIES)
    axis.set_xticklabels(
        ["{:.2f}".format(value) for value in UPDATE_PROBABILITIES],
        fontsize=FONT_SIZES["tick"],
    )
    axis.tick_params(axis="y", labelsize=FONT_SIZES["tick"])
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(fontsize=FONT_SIZES["legend"], frameon=True, loc="best")
    figure.tight_layout()
    png_path = output_dir / "update_probability_{}.png".format(topology_name)
    pdf_path = output_dir / "update_probability_{}.pdf".format(topology_name)
    figure.savefig(str(png_path), dpi=240, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main():
    """Generate YAMLs, run the paired sweep, aggregate results, and plot."""
    parser = argparse.ArgumentParser(
        description="Evaluate GNN generalization across source update probabilities"
    )
    parser.add_argument(
        "--topology",
        choices=("community", "random", "both"),
        default="both",
        help="run one topology for resumable batches, or both (default)",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=DEFAULT_SLOTS,
        help="slots per validation scenario (default: {})".format(DEFAULT_SLOTS),
    )
    args = parser.parse_args()
    if args.slots < 1:
        raise ValueError("--slots must be positive")
    topology_names = (
        ("community", "random") if args.topology == "both" else (args.topology,)
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reference_raw_by_topology = {
        topology: read_yaml(MODEL_DIRS[topology]["curvature"] / "training_config.yaml")
        for topology in ("community", "random")
    }
    config_paths = write_evaluation_configs(reference_raw_by_topology, args.slots)
    scenario_rows, provenance = evaluate_all(config_paths, topology_names, args.slots)
    summary_rows = summarize_rows(scenario_rows)
    output_dir = OUTPUT_ROOT if args.topology == "both" else OUTPUT_ROOT / args.topology
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "update_probability_per_scenario.csv", scenario_rows)
    write_csv(output_dir / "update_probability_summary.csv", summary_rows)
    plot_paths = {
        topology: plot_topology(summary_rows, topology, output_dir)
        for topology in topology_names
    }
    with (output_dir / "update_probability_metadata.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "node_count": NODE_COUNT,
                "update_probabilities": UPDATE_PROBABILITIES,
                "scenarios_per_update_probability": SCENARIOS_PER_UPDATE_PROBABILITY,
                "slots_per_scenario": args.slots,
                "evaluation_master_seed": EVALUATION_MASTER_SEED,
                "models": provenance,
                "configs": {
                    topology: [str(path) for path in paths.values()]
                    for topology, paths in config_paths.items()
                },
                "plots": {
                    topology: [str(path) for path in paths]
                    for topology, paths in plot_paths.items()
                },
            },
            stream,
            indent=2,
            sort_keys=True,
        )
    print("Saved update-probability output to {}".format(output_dir))


if __name__ == "__main__":
    main()
