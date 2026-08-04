"""运行并绘制 GNN 在不同节点规模下的泛化性能。

功能：恢复社区拓扑和随机几何拓扑对应的当前 best-validation MPNN，生成
节点数 50--150（步长 10）的固定验证 YAML；每个节点数使用 5 个场景，
并通过同步缩放几何区域保持节点密度近似不变。脚本随后评估曲率 MPNN、
无曲率 MPNN 及曲率模型的 matched-random，并分别输出两张 VAoI 泛化曲线。
每条曲线的阴影区域是该节点数 5 个场景的近似 95% 置信区间。
"""

from __future__ import division

import csv
import argparse
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
RESULT_ROOT = PROJECT_ROOT / "result_GNN"
OUTPUT_ROOT = RESULT_ROOT / "0804Scalability"
CONFIG_ROOT = OUTPUT_ROOT / "configs"

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvature_gossip.learning.ctde_ppo import CTDEPPO, resolve_learning_rates
from curvature_gossip.learning.trainer import _critic_config, _stage1_actor_config
from curvature_gossip.learning.validation import (
    _run_neural_scenario,
    _run_random_scenario,
    validation_scenarios,
)


MODEL_DIRS = {
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

NODE_COUNTS = list(range(50, 151, 10))
SCENARIOS_PER_NODE_COUNT = 5
# 200 slots keeps the 22-point scalability sweep practical; use --slots 500
# when exact alignment with the original fixed-validation horizon is required.
SLOTS_PER_SCENARIO = 200
REFERENCE_NODE_COUNT = 100
EVALUATION_MASTER_SEED = 20260804
TOPOLOGY_SEED_BASE = 85000
SOURCE_SEED_BASE = 86000
SHADOWING_SEED_BASE = 87000
FADING_SEED_BASE = 88000
ACTION_SEED_BASE = 89000

PLOT_LABELS = {
    ("curvature", "nn"): "Curvature MPNN",
    ("no_curvature", "nn"): "No-curvature MPNN",
    ("curvature", "matched_random"): "Curvature matched random",
}

PLOT_COLORS = {
    ("curvature", "nn"): "#2F6B9A",
    ("no_curvature", "nn"): "#8C8C8C",
    ("curvature", "matched_random"): "#E6A23C",
}

FONT_SIZES = {
    "title": 16,
    "axis_label": 14,
    "tick": 12.5,
    "legend": 12,
}


def read_yaml(path):
    """Read one YAML mapping and fail with its path if the content is invalid."""
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("expected a YAML mapping: {}".format(path))
    return value


def checkpoint_path(model_dir):
    """Return the restored best-validation checkpoint prefix."""
    path = model_dir / "checkpoints" / "best_validation" / "model"
    if not path.with_suffix(".index").is_file():
        raise FileNotFoundError("missing best-validation checkpoint: {}".format(path))
    return path


def checkpoint_episode(model_dir):
    """Read the checkpoint episode for output provenance."""
    path = model_dir / "checkpoints" / "best_validation" / "checkpoint_info.json"
    if not path.is_file():
        return -1
    with path.open("r", encoding="utf-8") as stream:
        return int(json.load(stream).get("episode", -1))


def scale_topology_params(topology_config, topology_name, node_count):
    """Scale geometry lengths with sqrt(N) so areal node density stays stable."""
    topology = deepcopy(topology_config)
    params = dict(topology.get("params", {}))
    scale = math.sqrt(float(node_count) / float(REFERENCE_NODE_COUNT))
    params["n_nodes"] = int(node_count)
    if "cluster_sizes" in params:
        params["cluster_sizes"] = [node_count // 2, node_count - node_count // 2]

    # Communication radius remains fixed; all physical layout lengths scale.
    if topology_name == "community":
        for key in ("area_width_m", "area_height_m", "cluster_radius_m", "center_separation_m"):
            if key in params:
                params[key] = float(params[key]) * scale
    elif topology_name == "random":
        for key in ("area_width_m", "area_height_m"):
            if key in params:
                params[key] = float(params[key]) * scale
    else:
        raise ValueError("unknown topology name: {}".format(topology_name))
    topology["params"] = params
    return topology, scale


def build_scenarios(node_count, topology_index):
    """Create five paired scenarios for one node count."""
    scenarios = []
    for scenario_index in range(SCENARIOS_PER_NODE_COUNT):
        offset = topology_index * 10000 + ((node_count - 50) // 10) * 100 + scenario_index
        scenarios.append(
            {
                "id": "n{:03d}_s{:02d}".format(node_count, scenario_index),
                "topology_seed": TOPOLOGY_SEED_BASE + offset,
                "source_update_seed": SOURCE_SEED_BASE + offset,
                "channel_shadowing_seed": SHADOWING_SEED_BASE + offset,
                "channel_fading_seed": FADING_SEED_BASE + offset,
                "policy_action_seed": ACTION_SEED_BASE + offset,
                "slots": SLOTS_PER_SCENARIO,
                "n_nodes": int(node_count),
                "update_probability": 0.20,
                "target_tx_ratio": 0.10,
            }
        )
    return scenarios


def build_evaluation_config(reference_raw, topology_name, node_count, topology_index):
    """Build one common evaluation YAML for both paired models."""
    raw = {
        "experiment": {"master_seed": EVALUATION_MASTER_SEED},
        "topology": None,
        "source": deepcopy(reference_raw["source"]),
        "channel": deepcopy(reference_raw["channel"]),
        "curvature": deepcopy(reference_raw["curvature"]),
        "constraints": deepcopy(reference_raw["constraints"]),
        "observation": deepcopy(reference_raw["observation"]),
        "validation": {"scenarios": build_scenarios(node_count, topology_index)},
    }
    raw["topology"], scale = scale_topology_params(
        reference_raw["topology"], topology_name, node_count
    )
    raw["metadata"] = {
        "topology_name": topology_name,
        "node_count": int(node_count),
        "reference_node_count": REFERENCE_NODE_COUNT,
        "linear_geometry_scale": scale,
        "density_policy": "area and community geometry scale as sqrt(N/100); communication radius is fixed",
    }
    return raw


def write_evaluation_configs(reference_raw_by_topology):
    """Write the 22 node-scale YAMLs under result_GNN/0804Scalability."""
    config_paths = {}
    for topology_index, topology_name in enumerate(("community", "random")):
        reference_raw = reference_raw_by_topology[topology_name]
        topology_dir = CONFIG_ROOT / topology_name
        topology_dir.mkdir(parents=True, exist_ok=True)
        config_paths[topology_name] = {}
        for node_count in NODE_COUNTS:
            raw = build_evaluation_config(
                reference_raw, topology_name, node_count, topology_index
            )
            path = topology_dir / "n{:03d}.yaml".format(node_count)
            with path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(raw, stream, sort_keys=False, allow_unicode=True)
            config_paths[topology_name][node_count] = path
    return config_paths


def build_model(model_dir):
    """Restore one model exactly from its persisted training configuration."""
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


def evaluate_required_policies(model, raw, checkpoint_episode_value, include_matched_random):
    """Evaluate only the policies needed by the scalability figure.

    The project's full fixed-validation helper also evaluates Stage-1-only and
    fixed-random policies. Those extra policies are useful for diagnostics but
    are not needed for this three-curve scalability comparison, so this lean
    path avoids repeating most of the simulator work.
    """
    before = model.variable_snapshot()
    rows = []
    nn_values = []
    for scenario in validation_scenarios(raw):
        nn_summary, nn_stats, nn_matrix = _run_neural_scenario(
            model, raw, scenario, policy_name="nn"
        )
        matched_probability = float(np.mean(nn_matrix))
        nn_values.append(float(nn_summary["mean_VAoI"]))
        rows.append(
            {
                "checkpoint_episode": int(checkpoint_episode_value),
                "scenario_id": scenario.scenario_id,
                "policy": "nn",
                "n_nodes": scenario.n_nodes,
                "slots": scenario.slots,
                "update_probability": scenario.update_probability,
                "target_tx_ratio": scenario.target_tx_ratio,
                "matched_rate_probability": matched_probability,
                "policy_probability": matched_probability,
                "mean_VAoI": float(nn_summary["mean_VAoI"]),
                "actual_tx_ratio": float(
                    nn_summary["avg_tx_per_slot"] / scenario.n_nodes
                ),
                **nn_stats
            }
        )
        if include_matched_random:
            matched_summary, matched_stats = _run_random_scenario(
                raw, scenario, "matched_random", matched_probability
            )
            rows.append(
                {
                    "checkpoint_episode": int(checkpoint_episode_value),
                    "scenario_id": scenario.scenario_id,
                    "policy": "matched_random",
                    "n_nodes": scenario.n_nodes,
                    "slots": scenario.slots,
                    "update_probability": scenario.update_probability,
                    "target_tx_ratio": scenario.target_tx_ratio,
                    "matched_rate_probability": matched_probability,
                    "policy_probability": matched_probability,
                    "mean_VAoI": float(matched_summary["mean_VAoI"]),
                    "actual_tx_ratio": float(
                        matched_summary["avg_tx_per_slot"] / scenario.n_nodes
                    ),
                    **matched_stats
                }
            )
    after = model.variable_snapshot()
    if len(before) != len(after) or any(
        not np.array_equal(old, new) for old, new in zip(before, after)
    ):
        raise RuntimeError("scalability validation unexpectedly changed model variables")
    return {"nn_mean_VAoI": float(np.mean(nn_values))}, rows


def evaluate_all(config_paths, topology_names):
    """Evaluate paired checkpoints at every scale and retain all policy rows."""
    scenario_rows = []
    provenance = []
    for topology_name in topology_names:
        curvature_dir = MODEL_DIRS[topology_name]["curvature"]
        no_curvature_dir = MODEL_DIRS[topology_name]["no_curvature"]
        curvature_model, curvature_raw = build_model(curvature_dir)
        no_curvature_model, no_curvature_raw = build_model(no_curvature_dir)
        provenance.extend(
            [
                {
                    "topology": topology_name,
                    "model": "curvature",
                    "model_dir": str(curvature_dir),
                    "checkpoint_episode": checkpoint_episode(curvature_dir),
                },
                {
                    "topology": topology_name,
                    "model": "no_curvature",
                    "model_dir": str(no_curvature_dir),
                    "checkpoint_episode": checkpoint_episode(no_curvature_dir),
                },
            ]
        )
        try:
            for node_count in NODE_COUNTS:
                evaluation_raw = read_yaml(config_paths[topology_name][node_count])
                for model_name, model, model_raw in (
                    ("curvature", curvature_model, curvature_raw),
                    ("no_curvature", no_curvature_model, no_curvature_raw),
                ):
                    # Overlay only simulator/scenario settings; the model's actor,
                    # critic, and frozen Layer-1 center remain from its own YAML.
                    merged = deepcopy(model_raw)
                    for key in (
                        "topology", "source", "channel", "curvature",
                        "constraints", "observation", "validation",
                    ):
                        merged[key] = deepcopy(evaluation_raw[key])
                    merged["experiment"] = dict(merged.get("experiment", {}))
                    merged["experiment"]["master_seed"] = EVALUATION_MASTER_SEED
                    summary, rows = evaluate_required_policies(
                        model,
                        merged,
                        checkpoint_episode(MODEL_DIRS[topology_name][model_name]),
                        include_matched_random=(model_name == "curvature"),
                    )
                    for row in rows:
                        tagged = dict(row)
                        tagged.update(
                            {
                                "topology": topology_name,
                                "model": model_name,
                                "node_count": int(node_count),
                            }
                        )
                        scenario_rows.append(tagged)
                    print(
                        "{} | n={} | {} | nn_VAoI={:.4f}".format(
                            topology_name,
                            node_count,
                            model_name,
                            float(summary["nn_mean_VAoI"]),
                        )
                    )
        finally:
            curvature_model.close()
            no_curvature_model.close()
    return scenario_rows, provenance


def mean_ci(values):
    """Return mean, standard deviation, and normal-approximation 95% CI."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, std, ci95


def summarize_rows(scenario_rows):
    """Aggregate five scenarios per node count and policy."""
    groups = {}
    for row in scenario_rows:
        key = (row["topology"], int(row["node_count"]), row["model"], row["policy"])
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for (topology, node_count, model, policy), rows in sorted(groups.items()):
        vaoi_mean, vaoi_std, vaoi_ci = mean_ci([row["mean_VAoI"] for row in rows])
        tx_mean, tx_std, tx_ci = mean_ci([row["actual_tx_ratio"] for row in rows])
        summary_rows.append(
            {
                "topology": topology,
                "topology_label": TOPOLOGY_LABELS[topology],
                "node_count": node_count,
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
    """Write a list of dictionaries as UTF-8 CSV."""
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
    """Plot one topology's three performance curves with scenario CI bands."""
    figure, axis = plt.subplots(figsize=(9.8, 6.4))
    x = np.asarray(NODE_COUNTS, dtype=float)
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
        rows = sorted(rows, key=lambda row: row["node_count"])
        if len(rows) != len(NODE_COUNTS):
            raise ValueError(
                "expected {} points for {} / {}, got {}".format(
                    len(NODE_COUNTS), model, policy, len(rows)
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
        "{} scalability generalization".format(TOPOLOGY_LABELS[topology_name]),
        fontsize=FONT_SIZES["title"],
        pad=14,
    )
    axis.set_xlabel("Number of nodes", fontsize=FONT_SIZES["axis_label"])
    axis.set_ylabel("Mean VAoI (lower is better)", fontsize=FONT_SIZES["axis_label"])
    axis.set_xticks(NODE_COUNTS)
    axis.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(fontsize=FONT_SIZES["legend"], frameon=True, loc="best")
    figure.tight_layout()
    png_path = output_dir / "scalability_{}.png".format(topology_name)
    pdf_path = output_dir / "scalability_{}.pdf".format(topology_name)
    figure.savefig(str(png_path), dpi=240, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main():
    """Generate configs, run paired evaluations, aggregate results, and plot."""
    global SLOTS_PER_SCENARIO
    parser = argparse.ArgumentParser(
        description="Evaluate GNN scalability on community and random topologies"
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
        default=SLOTS_PER_SCENARIO,
        help="slots per validation scenario (default: {})".format(SLOTS_PER_SCENARIO),
    )
    args = parser.parse_args()
    if args.slots < 1:
        raise ValueError("--slots must be positive")
    SLOTS_PER_SCENARIO = args.slots
    topology_names = ("community", "random") if args.topology == "both" else (args.topology,)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reference_raw_by_topology = {
        topology: read_yaml(MODEL_DIRS[topology]["curvature"] / "training_config.yaml")
        for topology in ("community", "random")
    }
    config_paths = write_evaluation_configs(reference_raw_by_topology)
    scenario_rows, provenance = evaluate_all(config_paths, topology_names)
    summary_rows = summarize_rows(scenario_rows)
    output_dir = OUTPUT_ROOT if args.topology == "both" else OUTPUT_ROOT / args.topology
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "scalability_per_scenario.csv", scenario_rows)
    write_csv(output_dir / "scalability_summary.csv", summary_rows)
    plot_paths = {
        topology: plot_topology(summary_rows, topology, output_dir)
        for topology in topology_names
    }
    with (output_dir / "scalability_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "node_counts": NODE_COUNTS,
                "scenarios_per_node_count": SCENARIOS_PER_NODE_COUNT,
                "slots_per_scenario": SLOTS_PER_SCENARIO,
                "evaluation_master_seed": EVALUATION_MASTER_SEED,
                "density_scaling": "area and community geometry scale as sqrt(N/100); communication radius remains fixed",
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
    print("Saved scalability output to {}".format(output_dir))


if __name__ == "__main__":
    main()
