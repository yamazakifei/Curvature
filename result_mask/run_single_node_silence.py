"""独立执行单节点静默故障实验，并统计、绘制平均/最大/最小 AoI。

本脚本不修改 curvature-gossip 现有源码。它复用现有拓扑、曲率、信道和
GossipSimulator，在每个时隙用固定 Bernoulli 概率生成广播动作，再将指定
故障节点的动作强制设为 False，从而实现节点静默故障。基准组和每个故障组
使用完全相同的外生随机流，便于进行配对的 AoI 退化比较。实验结束后还会
自动生成节点平均曲率、节点最小相邻边曲率和边曲率分布图。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Allow direct execution from the workspace root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "curvature-gossip"
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr

from curvature_gossip.channel import ChannelParameters, PropagationModel
from curvature_gossip.config import load_config
from curvature_gossip.curvature import (
    CurvatureResult,
    DistributedAF3Curvature,
    GlobalAF3Curvature,
    GlobalORCCurvature,
    bottleneck_importance,
)
from curvature_gossip.policies import UniformRandomPolicy
from curvature_gossip.random_streams import make_rng
from curvature_gossip.simulator import GossipSimulator, SimulationParameters
from curvature_gossip.topology import get_topology_generator
from curvature_gossip.topology.base import Topology


AOI_METRICS = ("mean_aoi", "mean_max_aoi", "mean_min_aoi")
ALL_AOI_METRICS = AOI_METRICS + ("max_aoi", "min_aoi")


@dataclass
class ScenarioResult:
    """保存单个基准或故障场景的聚合指标和可选时隙轨迹。"""

    summary: Dict[str, Any]
    per_slot: List[Dict[str, Any]]
    dissemination_records: Sequence[Any]


def _json_safe(value: Any) -> Any:
    """将 numpy、Path 和嵌套容器转换为 JSON 可序列化对象。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, indent=2, ensure_ascii=False, sort_keys=True)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _resolve_project_path(path_value: str, base_directory: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_directory / path


def _curvature_provider(raw_config: Mapping[str, Any]):
    config = raw_config.get("curvature", {})
    method = config.get("method", "global_orc")
    if method == "global_orc":
        return GlobalORCCurvature(alpha=float(config.get("alpha", 0.5)))
    if method in ("af3", "global_af3"):
        return GlobalAF3Curvature()
    if method == "distributed_af3":
        return DistributedAF3Curvature()
    raise ValueError("unknown curvature method: {}".format(method))


def _simulation_parameters(raw_config: Mapping[str, Any], slots_override: Optional[int]) -> SimulationParameters:
    """从新实验配置构造现有仿真器所需参数。"""
    experiment = raw_config["experiment"]
    source = raw_config.get("source", {})
    output = raw_config.get("output", {})
    constraints = raw_config.get("constraints", {})
    observation = raw_config.get("observation", {})
    slots = int(slots_override if slots_override is not None else experiment.get("slots", 1000))
    return SimulationParameters(
        slots=slots,
        update_probability=float(source.get("update_probability", 0.1)),
        warmup_slots=int(experiment.get("warmup_slots", 0)),
        trace_stride=int(experiment.get("trace_stride", 1)),
        node_diagnostics_stride=0,
        # This is only the existing simulator's optional cap; random p is configured separately.
        target_tx_ratio=float(constraints.get("target_tx_ratio", 1.0)),
        per_node_cap_multiplier=float(constraints.get("per_node_cap_multiplier", 1.0)),
        congestion_ewma_beta=float(observation.get("congestion_ewma_beta", 0.8)),
        congestion_feature_scale=float(observation.get("congestion_feature_scale", 5.0)),
    )


def _build_node_table(topology: Topology, curvature: CurvatureResult) -> List[Dict[str, Any]]:
    """导出每个节点的曲率属性，并附带基础拓扑信息供后续排名分析。"""
    graph = topology.graph
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    values = np.asarray(curvature.node_values, dtype=float)
    if values.shape != (graph.number_of_nodes(),) or not np.all(np.isfinite(values)):
        raise ValueError("curvature.node_values must be finite and have shape [N]")
    order_raw = sorted(range(len(values)), key=lambda node: (-values[node], node))
    order_abs = sorted(range(len(values)), key=lambda node: (-abs(values[node]), node))
    raw_rank = {node: rank + 1 for rank, node in enumerate(order_raw)}
    abs_rank = {node: rank + 1 for rank, node in enumerate(order_abs)}
    rows = []
    for node, (x_position, y_position) in enumerate(topology.positions):
        rows.append({
            "node": int(node),
            "x": float(x_position),
            "y": float(y_position),
            "label": "" if topology.node_labels is None else int(topology.node_labels[node]),
            "node_curvature": float(values[node]),
            "absolute_node_curvature": float(abs(values[node])),
            "curvature_rank_desc": int(raw_rank[node]),
            "absolute_curvature_rank_desc": int(abs_rank[node]),
            "degree": int(graph.degree(node)),
            "betweenness": float(betweenness[node]),
        })
    return rows


def _build_edge_table(topology: Topology, curvature: CurvatureResult) -> List[Dict[str, Any]]:
    rows = []
    for i, j in sorted(topology.graph.edges):
        edge = (min(int(i), int(j)), max(int(i), int(j)))
        rows.append({
            "node_i": edge[0],
            "node_j": edge[1],
            "edge_curvature": float(curvature.edge_values[edge]),
            "is_inter_cluster": bool(
                topology.node_labels is not None
                and topology.node_labels[edge[0]] != topology.node_labels[edge[1]]
            ),
        })
    return rows


def _run_masked_simulation(
    topology: Topology,
    curvature: CurvatureResult,
    importance: Mapping,
    propagation: PropagationModel,
    parameters: SimulationParameters,
    tx_probability: float,
    failure_node: Optional[int],
    update_rng: np.random.Generator,
    fading_rng: np.random.Generator,
    policy_rng: np.random.Generator,
) -> ScenarioResult:
    """运行一个场景；故障节点仍可接收信息，但永远不会发起广播。"""
    policy = UniformRandomPolicy(tx_probability=tx_probability)
    simulator = GossipSimulator(
        topology,
        curvature,
        importance,
        propagation,
        policy,
        parameters,
        update_rng,
        fading_rng,
        policy_rng,
    )
    measured_mean = []
    measured_max = []
    measured_min = []
    measured_slot_values: Dict[int, Tuple[float, float, float]] = {}

    # Keep the original engine's update/channel order and only mask one action.
    for slot in range(parameters.slots):
        observations = simulator.begin_step(slot)
        probabilities = np.asarray(policy.transmission_probabilities(observations), dtype=float)
        actions = policy_rng.random(topology.graph.number_of_nodes()) < probabilities
        if failure_node is not None:
            actions[int(failure_node)] = False
        simulator.complete_step(actions)

        if slot < parameters.warmup_slots:
            continue
        ages = simulator.state.version_age()
        pair_ages = ages[~np.eye(simulator.state.n_nodes, dtype=bool)].astype(float)
        current = (float(np.mean(pair_ages)), float(np.max(pair_ages)), float(np.min(pair_ages)))
        measured_mean.append(current[0])
        measured_max.append(current[1])
        measured_min.append(current[2])
        if (slot - parameters.warmup_slots) % parameters.trace_stride == 0:
            measured_slot_values[int(slot)] = current

    metric_summary = simulator.metrics.summary()
    metric_summary.update(simulator.tracker.summary())
    if not measured_mean:
        raise RuntimeError("no post-warmup slots were measured")

    # Report both time-averaged extrema and all-sample extrema; min_aoi is usually less informative.
    metric_summary.update({
        "mean_aoi": float(np.mean(measured_mean)),
        "mean_max_aoi": float(np.mean(measured_max)),
        "mean_min_aoi": float(np.mean(measured_min)),
        "max_aoi": float(np.max(measured_max)),
        "min_aoi": float(np.min(measured_min)),
        "configured_broadcast_probability": float(tx_probability),
        "mean_policy_probability": float(tx_probability),
        "average_broadcast_probability": float(tx_probability),
        "actual_tx_ratio": float(metric_summary["avg_tx_per_slot"] / simulator.state.n_nodes),
        "active_node_actual_tx_ratio": None,
        "failure_node": -1 if failure_node is None else int(failure_node),
        "failure_node_activity_ratio": None,
        "measured_slots": int(len(measured_mean)),
    })
    activity = metric_summary.get("per_node_activity_ratio", [])
    active_node_count = simulator.state.n_nodes if failure_node is None else simulator.state.n_nodes - 1
    metric_summary["active_node_actual_tx_ratio"] = float(
        metric_summary["avg_tx_per_slot"] / active_node_count
    ) if active_node_count > 0 else 0.0
    if failure_node is not None and activity:
        metric_summary["failure_node_activity_ratio"] = float(activity[int(failure_node)])

    per_slot = []
    for row in simulator.metrics.per_slot:
        output_row = dict(row)
        values = measured_slot_values.get(int(row["slot"]))
        if values is not None:
            output_row.update({
                "mean_aoi": values[0],
                "max_aoi": values[1],
                "min_aoi": values[2],
            })
        per_slot.append(output_row)
    return ScenarioResult(metric_summary, per_slot, simulator.tracker.records)


def _scenario_rngs(
    master_seed: int,
    topology_seed: int,
    channel_seed: int,
    update_seed: int,
) -> Tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """返回配对实验使用的更新、衰落和广播随机流。"""
    return (
        make_rng(master_seed, "updates", topology_seed, update_seed),
        make_rng(master_seed, "fading", topology_seed, channel_seed, update_seed),
        # Same label for baseline and all failures gives common random numbers.
        make_rng(master_seed, "policy", topology_seed, channel_seed, update_seed, "random_broadcast"),
    )


def _save_scenario(
    directory: Path,
    raw_config: Mapping[str, Any],
    summary: Mapping[str, Any],
    result: ScenarioResult,
    save_per_slot: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "summary.json", summary)
    with (directory / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(_json_safe(raw_config), stream, sort_keys=False, allow_unicode=True)
    if save_per_slot:
        _write_csv(directory / "per_slot.csv", result.per_slot)
    delay_rows = [{
        "source": int(record.source),
        "version": int(record.version),
        "generation_slot": int(record.generation_slot),
        "completion_slot": int(record.completion_slot) if record.completed else "",
        "delay": int(record.delay) if record.completed else "",
        "completed": bool(record.completed),
    } for record in result.dissemination_records]
    _write_csv(
        directory / "dissemination_delays.csv",
        delay_rows,
        ["source", "version", "generation_slot", "completion_slot", "delay", "completed"],
    )


def _write_topology_files(directory: Path, topology: Topology, curvature: CurvatureResult) -> List[Dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    node_rows = _build_node_table(topology, curvature)
    _write_csv(directory / "topology_nodes.csv", node_rows)
    _write_csv(directory / "topology_edges.csv", _build_edge_table(topology, curvature))
    _write_json(directory / "topology_metadata.json", topology.metadata)
    return node_rows


def _scenario_row(
    topology_seed: int,
    channel_seed: int,
    update_seed: int,
    failure_node: Optional[int],
    node_info: Optional[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "topology_seed": int(topology_seed),
        "channel_seed": int(channel_seed),
        "update_seed": int(update_seed),
        "failure_node": -1 if failure_node is None else int(failure_node),
        "scenario": "baseline" if failure_node is None else "single_node_silence",
        "node_curvature": None if node_info is None else node_info["node_curvature"],
        "absolute_node_curvature": None if node_info is None else node_info["absolute_node_curvature"],
        "curvature_rank_desc": None if node_info is None else node_info["curvature_rank_desc"],
        "absolute_curvature_rank_desc": None if node_info is None else node_info["absolute_curvature_rank_desc"],
        "degree": None if node_info is None else node_info["degree"],
        "betweenness": None if node_info is None else node_info["betweenness"],
        "configured_broadcast_probability": failure["configured_broadcast_probability"],
        "mean_policy_probability": failure["mean_policy_probability"],
        "average_broadcast_probability": failure["average_broadcast_probability"],
        "actual_tx_ratio": failure["actual_tx_ratio"],
        "active_node_actual_tx_ratio": failure["active_node_actual_tx_ratio"],
        "failure_node_activity_ratio": failure["failure_node_activity_ratio"],
    }
    for metric in ALL_AOI_METRICS:
        row["baseline_{}".format(metric)] = baseline[metric]
        row["failure_{}".format(metric)] = failure[metric]
        if failure_node is None:
            row["delta_{}".format(metric)] = 0.0
            row["relative_delta_{}".format(metric)] = 0.0
        else:
            delta = float(failure[metric] - baseline[metric])
            row["delta_{}".format(metric)] = delta
            denominator = float(baseline[metric])
            row["relative_delta_{}".format(metric)] = (
                delta / denominator if abs(denominator) > 1e-12 else None
            )
    return row


def _plot_topology(path: Path, topology: Topology, node_rows: Sequence[Mapping[str, Any]]) -> None:
    """绘制节点曲率拓扑图，便于检查高/低曲率节点的空间位置。"""
    values = np.asarray([float(row["node_curvature"]) for row in node_rows])
    positions = {node: topology.positions[node] for node in topology.graph.nodes}
    figure, axis = plt.subplots(figsize=(8, 7))
    nx.draw_networkx_edges(topology.graph, positions, ax=axis, alpha=0.35, width=1.0)
    nodes = nx.draw_networkx_nodes(
        topology.graph,
        positions,
        node_color=values,
        cmap="coolwarm",
        node_size=90,
        ax=axis,
    )
    nx.draw_networkx_labels(topology.graph, positions, font_size=7, ax=axis)
    figure.colorbar(nodes, ax=axis, label="node curvature (incident-edge mean)")
    axis.set_title("Node curvature and topology")
    axis.set_aspect("equal")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(str(path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_impact_scatter(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fault_rows = [row for row in rows if row["failure_node"] != -1]
    if not fault_rows:
        return
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=False)
    metrics = (
        ("delta_mean_aoi", "Δ mean AoI"),
        ("delta_mean_max_aoi", "Δ time-averaged max AoI"),
        ("delta_mean_min_aoi", "Δ time-averaged min AoI"),
    )
    for axis, (metric, label) in zip(axes, metrics):
        x = np.asarray([float(row["node_curvature"]) for row in fault_rows])
        y = np.asarray([float(row[metric]) for row in fault_rows])
        seeds = np.asarray([int(row["topology_seed"]) for row in fault_rows])
        scatter = axis.scatter(x, y, c=seeds, cmap="viridis", s=42, alpha=0.85)
        if len(x) >= 2 and np.ptp(x) > 1e-12:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            axis.plot(grid, slope * grid + intercept, color="black", linewidth=1.2, alpha=0.75)
        axis.axhline(0.0, color="gray", linewidth=0.8)
        axis.set_xlabel("node curvature")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    if len(set(int(row["topology_seed"]) for row in fault_rows)) > 1:
        figure.colorbar(scatter, ax=axes, label="topology seed")
    figure.suptitle("Single-node silence impact versus node curvature")
    figure.tight_layout()
    figure.savefig(str(path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_node_impact(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """按曲率从低到高排列节点，比较三个 AoI 指标的故障退化。"""
    fault_rows = [row for row in rows if row["failure_node"] != -1]
    if not fault_rows:
        return
    # This plot is most readable for one topology/channel/update tuple.
    key = (fault_rows[0]["topology_seed"], fault_rows[0]["channel_seed"], fault_rows[0]["update_seed"])
    selected = [row for row in fault_rows if (row["topology_seed"], row["channel_seed"], row["update_seed"]) == key]
    selected.sort(key=lambda row: (float(row["node_curvature"]), int(row["failure_node"])))
    x = np.arange(len(selected))
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    metrics = (
        ("delta_mean_aoi", "Δ mean AoI"),
        ("delta_mean_max_aoi", "Δ time-averaged max AoI"),
        ("delta_mean_min_aoi", "Δ time-averaged min AoI"),
    )
    for axis, (metric, label) in zip(axes, metrics):
        y = [float(row[metric]) for row in selected]
        axis.axhline(0.0, color="gray", linewidth=0.8)
        axis.plot(x, y, marker="o", linewidth=1.0, markersize=3.5)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("nodes ordered by increasing node curvature")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(row["failure_node"]) for row in selected], rotation=90)
    figure.suptitle("Per-node AoI degradation under single-node silence")
    figure.tight_layout()
    figure.savefig(str(path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_baseline_failure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fault_rows = [row for row in rows if row["failure_node"] != -1]
    if not fault_rows:
        return
    key = (fault_rows[0]["topology_seed"], fault_rows[0]["channel_seed"], fault_rows[0]["update_seed"])
    selected = [row for row in fault_rows if (row["topology_seed"], row["channel_seed"], row["update_seed"]) == key]
    baseline = next(
        row for row in rows
        if row["failure_node"] == -1
        and (row["topology_seed"], row["channel_seed"], row["update_seed"]) == key
    )
    selected = sorted(selected, key=lambda row: int(row["failure_node"]))
    x = np.arange(len(selected))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    labels = {
        "mean_aoi": "mean AoI",
        "mean_max_aoi": "time-averaged max AoI",
        "mean_min_aoi": "time-averaged min AoI",
    }
    for axis, metric in zip(axes, AOI_METRICS):
        base_key = "baseline_{}".format(metric)
        fail_key = "failure_{}".format(metric)
        axis.axhline(float(baseline[base_key]), color="black", linestyle="--", label="baseline")
        axis.plot(x, [float(row[fail_key]) for row in selected], marker="o", label="node silent")
        axis.set_title(labels[metric])
        axis.set_xlabel("failed node")
        axis.set_xticks(x)
        axis.set_xticklabels([str(row["failure_node"]) for row in selected], rotation=90)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("AoI")
    axes[0].legend()
    figure.suptitle("Baseline versus each single-node silence scenario")
    figure.tight_layout()
    figure.savefig(str(path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _make_plots(
    plots_directory: Path,
    topologies: Mapping[int, Tuple[Topology, Sequence[Mapping[str, Any]]]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    plots_directory.mkdir(parents=True, exist_ok=True)
    for topology_seed, (topology, node_rows) in topologies.items():
        _plot_topology(plots_directory / "topology_{}_curvature.png".format(topology_seed), topology, node_rows)
    _plot_node_impact(plots_directory / "per_node_aoi_impact.png", rows)
    _plot_baseline_failure(plots_directory / "baseline_vs_failure_aoi.png", rows)


def _correlation_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """计算曲率与节点故障 AoI 退化的 Pearson/Spearman 统计量。"""
    fault_rows = [row for row in rows if row["failure_node"] != -1]
    output: Dict[str, Any] = {"sample_count": len(fault_rows), "metrics": {}}
    for curvature_name, curvature_key in (
        ("node_curvature", "node_curvature"),
        ("absolute_node_curvature", "absolute_node_curvature"),
    ):
        output["metrics"][curvature_name] = {}
        for metric in AOI_METRICS:
            delta_key = "delta_{}".format(metric)
            valid = [
                row for row in fault_rows
                if row.get(curvature_key) is not None and row.get(delta_key) is not None
            ]
            x = np.asarray([float(row[curvature_key]) for row in valid], dtype=float)
            y = np.asarray([float(row[delta_key]) for row in valid], dtype=float)
            record: Dict[str, Any] = {"n": int(len(valid))}
            if len(valid) >= 2 and np.ptp(x) > 1e-12 and np.ptp(y) > 1e-12:
                pearson = pearsonr(x, y)
                spearman = spearmanr(x, y)
                record.update({
                    # SciPy older than 1.9 returns a two-item tuple here.
                    "pearson_r": float(pearson[0]),
                    "pearson_p": float(pearson[1]),
                    "spearman_r": float(spearman[0]),
                    "spearman_p": float(spearman[1]),
                })
            else:
                record.update({
                    "pearson_r": None,
                    "pearson_p": None,
                    "spearman_r": None,
                    "spearman_p": None,
                })
            output["metrics"][curvature_name][metric] = record
    return output


def run_experiment(
    config_path: Path,
    output_override: Optional[Path] = None,
    slots_override: Optional[int] = None,
    limit_nodes: Optional[int] = None,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """运行所有配置拓扑，并为每个拓扑中的每个节点执行一次静默故障。"""
    config = load_config(str(config_path))
    raw = dict(config.raw)
    experiment = raw["experiment"]
    master_seed = int(experiment.get("master_seed", 0))
    topology_seeds = [int(seed) for seed in experiment.get("topology_seeds", [0])]
    channel_seeds = [int(seed) for seed in experiment.get("channel_seeds", [0])]
    update_seeds = [int(seed) for seed in experiment.get("update_seeds", [0])]
    broadcast = raw.get("broadcast", {})
    tx_probability = float(broadcast.get("tx_probability", 0.1))
    if not 0.0 <= tx_probability <= 1.0:
        raise ValueError("broadcast.tx_probability must be in [0, 1]")
    if limit_nodes is not None and int(limit_nodes) < 1:
        raise ValueError("--limit-nodes must be positive when provided")
    parameters = _simulation_parameters(raw, slots_override)
    output_config = raw.get("output", {})
    output_directory = output_override or _resolve_project_path(
        str(output_config.get("root", "result_mask/outputs/single_node_silence_p01"))
    )
    if output_directory.exists() and not overwrite:
        existing = next(output_directory.iterdir(), None)
        if existing is not None:
            raise FileExistsError(
                "output directory is not empty: {}; use --overwrite or another --output".format(output_directory)
            )
    output_directory.mkdir(parents=True, exist_ok=True)

    generator = get_topology_generator(config.topology.type)
    curvature_config = raw.get("curvature", {})
    channel_parameters = ChannelParameters.from_mapping(raw.get("channel", {}))
    rows: List[Dict[str, Any]] = []
    topologies: Dict[int, Tuple[Topology, Sequence[Mapping[str, Any]]]] = {}
    save_per_slot = bool(output_config.get("save_per_slot", True))
    all_scenario_count = 0

    for topology_seed in topology_seeds:
        topology = generator.generate(
            make_rng(master_seed, "topology", topology_seed), config.topology.params
        )
        curvature = _curvature_provider(curvature_config).compute(topology)
        importance = bottleneck_importance(
            topology,
            curvature,
            curvature_config.get("normalization", "global_negative_max"),
        )
        node_rows = _build_node_table(topology, curvature)
        topologies[topology_seed] = (topology, node_rows)
        topology_directory = output_directory / "topology_{}".format(topology_seed)
        _write_topology_files(topology_directory, topology, curvature)
        node_count = topology.graph.number_of_nodes()
        failure_nodes = list(range(node_count))
        if limit_nodes is not None:
            failure_nodes = failure_nodes[: int(limit_nodes)]

        for channel_seed in channel_seeds:
            # One propagation model fixes shadowing for the paired scenarios.
            propagation = PropagationModel(
                topology.positions,
                channel_parameters,
                make_rng(master_seed, "shadowing", topology_seed, channel_seed),
            )
            for update_seed in update_seeds:
                scenario_root = topology_directory / "channel_{}".format(channel_seed) / "update_{}".format(update_seed)
                base_update_rng, base_fading_rng, base_policy_rng = _scenario_rngs(
                    master_seed, topology_seed, channel_seed, update_seed
                )
                baseline_result = _run_masked_simulation(
                    topology, curvature, importance, propagation, parameters, tx_probability, None,
                    base_update_rng, base_fading_rng, base_policy_rng,
                )
                baseline_summary = dict(baseline_result.summary)
                baseline_summary.update({
                    "topology_seed": topology_seed,
                    "channel_seed": channel_seed,
                    "update_seed": update_seed,
                    "scenario": "baseline",
                })
                _save_scenario(scenario_root / "baseline", raw, baseline_summary, baseline_result, save_per_slot)
                rows.append(_scenario_row(
                    topology_seed, channel_seed, update_seed, None, None,
                    baseline_result.summary, baseline_result.summary,
                ))
                all_scenario_count += 1
                if progress:
                    print("topology={} channel={} update={} baseline".format(topology_seed, channel_seed, update_seed))

                for failure_node in failure_nodes:
                    update_rng, fading_rng, policy_rng = _scenario_rngs(
                        master_seed, topology_seed, channel_seed, update_seed
                    )
                    failure_result = _run_masked_simulation(
                        topology, curvature, importance, propagation, parameters, tx_probability, failure_node,
                        update_rng, fading_rng, policy_rng,
                    )
                    failure_summary = dict(failure_result.summary)
                    failure_summary.update({
                        "topology_seed": topology_seed,
                        "channel_seed": channel_seed,
                        "update_seed": update_seed,
                        "scenario": "single_node_silence",
                    })
                    scenario_directory = scenario_root / "node_{:04d}".format(failure_node)
                    _save_scenario(scenario_directory, raw, failure_summary, failure_result, save_per_slot)
                    row = _scenario_row(
                        topology_seed,
                        channel_seed,
                        update_seed,
                        failure_node,
                        node_rows[failure_node],
                        baseline_result.summary,
                        failure_result.summary,
                    )
                    rows.append(row)
                    all_scenario_count += 1
                    if progress:
                        print(
                            "topology={} channel={} update={} failed_node={} "
                            "mean_aoi={:.6g} max_aoi={:.6g} min_aoi={:.6g}".format(
                                topology_seed,
                                channel_seed,
                                update_seed,
                                failure_node,
                                failure_result.summary["mean_aoi"],
                                failure_result.summary["mean_max_aoi"],
                                failure_result.summary["mean_min_aoi"],
                            )
                        )

    result_rows = output_directory / "single_node_failure_results.csv"
    _write_csv(result_rows, rows)
    _write_json(output_directory / "single_node_failure_results.json", rows)
    _write_json(output_directory / "curvature_aoi_correlations.json", _correlation_summary(rows))
    _write_json(output_directory / "experiment_manifest.json", {
        "script": str(Path(__file__).resolve()),
        "config": str(config_path.resolve()),
        "output_directory": str(output_directory.resolve()),
        "experiment_id": experiment.get("id", "single_node_silence"),
        "failure_model": "silence: failed node cannot transmit, but remains as a receiver/source state",
        "configured_broadcast_probability": tx_probability,
        "mean_policy_probability": tx_probability,
        "topology_type": config.topology.type,
        "curvature_method": curvature_config.get("method", "global_orc"),
        "topology_seeds": topology_seeds,
        "channel_seeds": channel_seeds,
        "update_seeds": update_seeds,
        "slots": parameters.slots,
        "warmup_slots": parameters.warmup_slots,
        "trace_stride": parameters.trace_stride,
        "scenario_count": all_scenario_count,
        "paired_random_streams": True,
        "aoi_metric_definition": {
            "mean_aoi": "post-warmup mean over ordered source-receiver pairs and slots",
            "mean_max_aoi": "post-warmup mean of the per-slot maximum pair AoI",
            "mean_min_aoi": "post-warmup mean of the per-slot minimum pair AoI",
            "max_aoi": "maximum pair AoI observed over post-warmup slots",
            "min_aoi": "minimum pair AoI observed over post-warmup slots",
        },
    })
    if bool(output_config.get("make_plots", True)):
        _make_plots(output_directory / "plots", topologies, rows)
        # 复用独立后处理模块，自动补充节点最小曲率和边曲率分布图。
        try:
            from plot_node_min_curvature import postprocess_scenario
        except ImportError:
            # 支持从项目根目录以模块方式调用本脚本。
            from result_mask.plot_node_min_curvature import postprocess_scenario
        postprocess_scenario(output_directory)
    return output_directory


def _parse_args() -> argparse.Namespace:
    default_config = Path(__file__).with_name("config_single_node_silence.yaml")
    parser = argparse.ArgumentParser(description="Run single-node silence AoI impact experiments.")
    parser.add_argument("--config", type=Path, default=default_config, help="YAML configuration path")
    parser.add_argument("--output", type=Path, default=None, help="override output directory")
    parser.add_argument("--slots", type=int, default=None, help="override experiment.slots for a quick smoke run")
    parser.add_argument("--limit-nodes", type=int, default=None, help="only run the first N nodes; for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty output directory")
    parser.add_argument("--no-progress", action="store_true", help="disable per-scenario progress output")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    output_override = None
    if args.output is not None:
        output_override = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_directory = run_experiment(
        config_path=config_path,
        output_override=output_override,
        slots_override=args.slots,
        limit_nodes=args.limit_nodes,
        overwrite=args.overwrite,
        progress=not args.no_progress,
    )
    print("Results written to {}".format(output_directory))


if __name__ == "__main__":
    main()
