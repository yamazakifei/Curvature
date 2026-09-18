"""Plot V3.5 ch2 node probabilities against node minimum curvature.

功能：在同一个 N=100 community 场景上运行 500 slots，提取三个方案的逐
节点 Actor 广播概率和节点最小相邻边曲率。输出 2×3 六子图：第一行是可
复现随机 slot 的概率分布，第二行是 500 slots 的逐节点概率均值；每个
子图包含散点、线性趋势线和平均广播概率说明。

Node curvature is defined as
``kappa_node(v) = min_{u in N(v)} kappa_edge(v, u)``.
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

import run_gnn_scalability_generalization as scalability
from curvature_gossip.learning.validation import _build_simulator
from curvature_gossip.learning.validation import validation_scenarios
from compare_v35_ch2_scalability import (
    DEFAULT_CURVATURE_MODEL,
    DEFAULT_NO_CURVATURE_MODEL,
    DEFAULT_OUTPUT_ROOT,
    bmax_checkpoint_path,
    build_evaluation_config,
    build_model,
    checkpoint_episode,
)
from plot_v35_ch2_n100_max_vaoi import encode_observations, overlay_evaluation_raw


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
RANDOM_SLOT_SEED = 20260917
N_SLOTS = 500


def write_yaml(path, raw):
    """Write the exact 500-slot N=100 evaluation configuration."""
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(raw, stream, sort_keys=False, allow_unicode=True)


def node_min_curvature(simulator):
    """Compute min incident-edge curvature for every node in the topology."""
    values = np.empty(simulator.topology.graph.number_of_nodes(), dtype=float)
    for node in simulator.topology.graph.nodes:
        incident = [
            simulator.curvature.value(node, neighbor)
            for neighbor in simulator.topology.graph.neighbors(node)
        ]
        if not incident:
            raise ValueError("node {} has no incident edges".format(node))
        values[int(node)] = float(np.min(incident))
    return values


def evaluate_mode(model, model_raw, model_key, scenario, stage1_only):
    """Collect [slot, node] output probabilities and node curvatures."""
    before = model.variable_snapshot()
    simulator = _build_simulator(model_raw, scenario, "nn", 0.0)
    try:
        curvature_values = node_min_curvature(simulator)
        probability_steps = []
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
        probabilities = np.asarray(probability_steps, dtype=float)
    finally:
        del simulator
        after = model.variable_snapshot()
        if len(before) != len(after) or any(
            not np.array_equal(old, new) for old, new in zip(before, after)
        ):
            raise RuntimeError("probability-curvature validation changed model variables")
    return curvature_values, probabilities


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


def linear_fit(x_values, y_values):
    """Return a sorted two-point linear trend line, if x has variation."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if np.ptp(x_values) <= 1e-12:
        return None
    coefficients = np.polyfit(x_values, y_values, 1)
    x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 100)
    return x_line, np.polyval(coefficients, x_line)


def plot_six_panel(curvature_values, mode_data, random_slot, output_root):
    """Create the 2×3 random-slot and 500-slot-mean scatter panels."""
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 8.2), sharex=True, sharey=True)
    for column, model_key in enumerate(MODEL_ORDER):
        probabilities = mode_data[model_key]
        random_values = probabilities[random_slot]
        mean_values = np.mean(probabilities, axis=0)
        for row_index, (panel_values, panel_label) in enumerate(
            ((random_values, "Random slot {}".format(random_slot)),
             (mean_values, "500-slot mean"))
        ):
            axis = axes[row_index, column]
            color = MODEL_COLORS[model_key]
            axis.scatter(
                curvature_values,
                panel_values,
                s=27,
                alpha=0.78,
                color=color,
                edgecolors="white",
                linewidths=0.35,
                label=MODEL_LABELS[model_key],
            )
            trend = linear_fit(curvature_values, panel_values)
            if trend is not None:
                axis.plot(trend[0], trend[1], color="#444444", linewidth=1.4)
            axis.axhline(float(np.mean(panel_values)), color="#777777", linewidth=0.8, alpha=0.65)
            axis.set_title(
                "{}\n{} (mean p={:.3f})".format(
                    MODEL_LABELS[model_key], panel_label, float(np.mean(panel_values))
                ),
                fontsize=11.5,
            )
            axis.grid(True, linewidth=0.6, alpha=0.28)
            axis.set_axisbelow(True)
            axis.tick_params(labelsize=9.5)
    for axis in axes[1, :]:
        axis.set_xlabel("minimum incident-edge curvature", fontsize=12)
    for axis in axes[:, 0]:
        axis.set_ylabel("Actor broadcast probability", fontsize=12)
    figure.suptitle(
        "V3.5 ch2, N=100: broadcast probability versus node curvature",
        fontsize=16,
        y=0.99,
    )
    figure.text(
        0.5,
        0.012,
        "Node curvature = minimum curvature among incident edges; gray line = linear trend",
        ha="center",
        fontsize=10.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    png_path = output_root / "broadcast_probability_vs_node_curvature_n100.png"
    pdf_path = output_root / "broadcast_probability_vs_node_curvature_n100.pdf"
    figure.savefig(str(png_path), dpi=240, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main():
    """Run one 500-slot N=100 scene and save the six-panel comparison."""
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
        evaluation_raw["validation"]["scenarios"] = [
            deepcopy(evaluation_raw["validation"]["scenarios"][0])
        ]
        evaluation_raw["validation"]["scenarios"][0]["slots"] = N_SLOTS
        evaluation_raw["metadata"]["slots_per_scenario"] = N_SLOTS
        config_path = args.output_root / "probability_curvature_n100_config.yaml"
        write_yaml(config_path, evaluation_raw)
        scenario = validation_scenarios(evaluation_raw)[0]
        curvature_eval_raw = overlay_evaluation_raw(curvature_raw, evaluation_raw)
        no_curvature_eval_raw = overlay_evaluation_raw(no_curvature_raw, evaluation_raw)
        mode_data = {}
        curvature_values, mode_data["v35_curvature_stage2"] = evaluate_mode(
            curvature_model, curvature_eval_raw, "v35_curvature_stage2", scenario, False
        )
        check_values, mode_data["v35_no_curvature_stage2"] = evaluate_mode(
            no_curvature_model, no_curvature_eval_raw, "v35_no_curvature_stage2", scenario, False
        )
        if not np.allclose(curvature_values, check_values):
            raise RuntimeError("three schemes did not use the same N=100 topology curvature")
        check_values, mode_data["v35_curvature_stage1_only"] = evaluate_mode(
            curvature_model, curvature_eval_raw, "v35_curvature_stage1_only", scenario, True
        )
        if not np.allclose(curvature_values, check_values):
            raise RuntimeError("Stage-1-only topology curvature does not match Stage-2")
    finally:
        curvature_model.close()
        no_curvature_model.close()

    random_slot = int(np.random.default_rng(RANDOM_SLOT_SEED).integers(0, N_SLOTS))
    sample_rows = []
    summary_rows = []
    for model_key in MODEL_ORDER:
        probabilities = mode_data[model_key]
        mean_values = np.mean(probabilities, axis=0)
        for node, curvature in enumerate(curvature_values):
            sample_rows.extend([
                {
                    "model": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "panel": "random_slot",
                    "slot": random_slot,
                    "node": node,
                    "node_min_curvature": float(curvature),
                    "broadcast_probability": float(probabilities[random_slot, node]),
                },
                {
                    "model": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "panel": "mean_500_slots",
                    "slot": "all",
                    "node": node,
                    "node_min_curvature": float(curvature),
                    "broadcast_probability": float(mean_values[node]),
                },
            ])
        summary_rows.extend([
            {
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "panel": "random_slot",
                "slot": random_slot,
                "mean_action_probability": float(np.mean(probabilities[random_slot])),
                "min_action_probability": float(np.min(probabilities[random_slot])),
                "max_action_probability": float(np.max(probabilities[random_slot])),
            },
            {
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "panel": "mean_500_slots",
                "slot": "all",
                "mean_action_probability": float(np.mean(mean_values)),
                "min_action_probability": float(np.min(mean_values)),
                "max_action_probability": float(np.max(mean_values)),
            },
        ])

    write_csv(args.output_root / "broadcast_probability_vs_node_curvature_n100.csv", sample_rows)
    write_csv(args.output_root / "broadcast_probability_vs_node_curvature_n100_summary.csv", summary_rows)
    plot_paths = plot_six_panel(curvature_values, mode_data, random_slot, args.output_root)
    metadata = {
        "evaluation": "N=100 V3.5 ch2 node broadcast probability versus minimum incident-edge curvature",
        "node_count": 100,
        "slots": N_SLOTS,
        "scenario_id": scenario.scenario_id,
        "random_slot_seed": RANDOM_SLOT_SEED,
        "random_slot": random_slot,
        "node_curvature_definition": "min incident edge curvature",
        "node_curvature_formula": "kappa_node(v) = min_{u in N(v)} kappa_edge(v,u)",
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
            "v35_curvature": checkpoint_episode(args.curvature_model),
            "v35_no_curvature": checkpoint_episode(args.no_curvature_model),
        },
        "plots": [str(path) for path in plot_paths],
    }
    with (args.output_root / "broadcast_probability_vs_node_curvature_n100_metadata.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    with (args.output_root / "README.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## N=100 probability versus node curvature\n\n"
            "`broadcast_probability_vs_node_curvature_n100.png` contains six "
            "panels for the three V3.5 ch2 modes. The first row uses a fixed "
            "reproducibly random slot from a 500-slot N=100 scene; the second "
            "row uses each node's mean output probability over all 500 slots. "
            "The x-axis is the minimum incident-edge curvature, and the panel "
            "titles report the mean broadcast probability.\n"
        )
    print("Saved six-panel probability-curvature plot to {}".format(args.output_root))


if __name__ == "__main__":
    main()
