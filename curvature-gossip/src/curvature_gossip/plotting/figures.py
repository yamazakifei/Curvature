"""绘制拓扑、曲率及启发式算法实验结果，保持与仿真逻辑完全分离。"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

from ..topology.base import Topology


def plot_topology(
    topology: Topology,
    output_path: Optional[str] = None,
    show_labels: bool = True,
    edge_curvatures: Optional[Mapping[Tuple[int, int], float]] = None,
):
    """绘制物理图，可按曲率着色并突出实际跨簇边。"""
    figure, axis = plt.subplots(figsize=(7, 6))
    positions = {i: topology.positions[i] for i in topology.graph.nodes}
    bridge_edges = {tuple(sorted(edge)) for edge in topology.metadata.get("inter_cluster_edges", [])}
    regular_edges = [edge for edge in topology.graph.edges if tuple(sorted(edge)) not in bridge_edges]
    if edge_curvatures:
        edge_list = list(topology.graph.edges)
        edge_colors = [edge_curvatures[tuple(sorted(edge))] for edge in edge_list]
        drawn = nx.draw_networkx_edges(
            topology.graph, positions, edgelist=edge_list, edge_color=edge_colors,
            edge_cmap=plt.cm.coolwarm, width=1.8, ax=axis,
        )
        figure.colorbar(drawn, ax=axis, label="edge curvature")
    else:
        nx.draw_networkx_edges(
            topology.graph, positions, edgelist=regular_edges, ax=axis, alpha=0.5
        )
    if bridge_edges:
        nx.draw_networkx_edges(
            topology.graph, positions, edgelist=list(bridge_edges), ax=axis,
            edge_color="black" if edge_curvatures else "crimson", width=2.8,
        )
    colors = topology.node_labels if topology.node_labels is not None else "tab:blue"
    nx.draw_networkx_nodes(
        topology.graph, positions, node_color=colors, cmap="coolwarm", ax=axis
    )
    if show_labels:
        nx.draw_networkx_labels(topology.graph, positions, font_size=8, ax=axis)
    axis.set_aspect("equal")
    axis.set_title("{} (N={}, R={:.3g})".format(
        topology.metadata.get("type", "topology"), len(positions), topology.communication_radius
    ))
    axis.set_axis_off()
    figure.tight_layout()
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(str(destination), dpi=160, bbox_inches="tight")
    return figure


def plot_curvature_histogram(
    edge_curvatures: Mapping[Tuple[int, int], float], output_path: Optional[str] = None
):
    """绘制固定拓扑上的边曲率直方图。"""
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(list(edge_curvatures.values()), bins="auto", color="steelblue", edgecolor="white")
    axis.set_xlabel("edge curvature")
    axis.set_ylabel("count")
    axis.set_title("Edge-curvature distribution")
    figure.tight_layout()
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(str(destination), dpi=160, bbox_inches="tight")
    return figure


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _plot_saved_topology(run_directory: Path, plots_directory: Path) -> None:
    node_rows = _read_csv(run_directory / "topology_nodes.csv")
    edge_rows = _read_csv(run_directory / "topology_edges.csv")
    curvature_rows = _read_csv(run_directory / "edge_curvature.csv")
    positions = np.asarray([[float(row["x"]), float(row["y"])] for row in node_rows])
    labels = None
    if node_rows and node_rows[0]["label"] != "":
        labels = np.asarray([int(row["label"]) for row in node_rows])
    graph = nx.Graph()
    graph.add_nodes_from(range(len(node_rows)))
    graph.add_edges_from((int(row["node_i"]), int(row["node_j"])) for row in edge_rows)
    with (run_directory / "resolved_config.yaml").open("r", encoding="utf-8") as stream:
        resolved = yaml.safe_load(stream)
    radius = float(resolved["topology"]["params"].get("communication_radius_m", 1.0))
    bridge_edges = [
        (int(row["node_i"]), int(row["node_j"])) for row in edge_rows
        if row["is_inter_cluster"].lower() == "true"
    ]
    topology = Topology(positions, graph, radius, labels, {
        "type": resolved["topology"]["type"], "inter_cluster_edges": bridge_edges,
    })
    edge_curvatures = {
        (int(row["node_i"]), int(row["node_j"])): float(row["curvature"])
        for row in curvature_rows
    }
    figure = plot_topology(
        topology, str(plots_directory / "topology_curvature.png"),
        edge_curvatures=edge_curvatures,
    )
    plt.close(figure)
    figure = plot_curvature_histogram(
        edge_curvatures, str(plots_directory / "curvature_histogram.png")
    )
    plt.close(figure)


def plot_experiment_results(results_directory: str, smoothing_window: int = 10):
    """读取持久化标量结果并生成文档要求的六类图片。"""
    root = Path(results_directory)
    summary_paths = sorted(root.rglob("summary.json"))
    if not summary_paths:
        raise ValueError("no summary.json files found under {}".format(root))
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    runs = []
    for summary_path in summary_paths:
        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        trace_path = summary_path.parent / "per_slot.csv"
        trace = _read_csv(trace_path) if trace_path.exists() else []
        runs.append((summary, trace, summary_path.parent))

    # 曲率拓扑和直方图只需使用任一配对运行，因为同拓扑策略共享曲率。
    _plot_saved_topology(runs[0][2], plots)
    by_policy = defaultdict(list)
    for summary, trace, _ in runs:
        by_policy[summary["policy"]].append((summary, trace))

    # 原始降采样轨迹保持透明显示，平滑曲线明确标注窗口。
    figure, axis = plt.subplots(figsize=(8, 5))
    for policy, policy_runs in sorted(by_policy.items()):
        trace = policy_runs[0][1]
        if not trace:
            continue
        slots = np.asarray([int(row["slot"]) for row in trace])
        values = np.asarray([float(row["max_version_age"]) for row in trace])
        axis.plot(slots, values, alpha=0.18)
        window = min(max(1, smoothing_window), values.size)
        smooth = np.convolve(values, np.ones(window) / window, mode="valid")
        axis.plot(slots[window - 1 :], smooth, label=policy)
    axis.set_xlabel("slot")
    axis.set_ylabel("maximum VAoI")
    axis.set_title("Maximum-VAoI trace (moving-average window={})".format(smoothing_window))
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plots / "max_vaoi_trace.png"), dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    for policy, policy_runs in sorted(by_policy.items()):
        values = np.asarray([
            float(row["max_version_age"]) for _, trace in policy_runs for row in trace
        ])
        if not values.size:
            continue
        values.sort()
        axis.step(values, np.arange(1, values.size + 1) / values.size, where="post", label=policy)
    axis.set_xlabel("instantaneous maximum VAoI")
    axis.set_ylabel("empirical CDF")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plots / "max_vaoi_ecdf.png"), dpi=160)
    plt.close(figure)

    with (root / "policy_comparison.json").open("r", encoding="utf-8") as stream:
        aggregate = json.load(stream)
    policies = [row["policy"] for row in aggregate]
    x_positions = np.arange(len(policies))
    figure, axis = plt.subplots(figsize=(8, 5))
    width = 0.25
    for offset, (key, label) in enumerate([
        ("mean_VAoI", "mean"),
        ("mean_max_VAoI", "maximum"),
        ("mean_tail_VAoI", "top-5% mean"),
    ]):
        axis.bar(
            x_positions + (offset - 1) * width,
            [row[key] for row in aggregate], width, label=label,
        )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(policies)
    axis.set_ylabel("VAoI")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plots / "policy_vaoi_comparison.png"), dpi=160)
    plt.close(figure)

    # 广播量不做预算匹配，因此用散点图显式呈现性能-活动量关系。
    figure, axis = plt.subplots(figsize=(6, 4))
    for row in aggregate:
        axis.scatter(row["avg_tx_per_slot"], row["mean_VAoI"], s=55, label=row["policy"])
    axis.set_xlabel("average transmissions per slot")
    axis.set_ylabel("mean VAoI")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plots / "vaoi_vs_transmissions.png"), dpi=160)
    plt.close(figure)
    return plots
