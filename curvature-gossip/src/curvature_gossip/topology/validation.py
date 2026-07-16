"""提供所有拓扑生成器共享的距离一致性与图结构校验。"""

from typing import Any, Mapping, Optional

import networkx as nx
import numpy as np

from .base import Topology


class TopologyValidationError(ValueError):
    """表示生成拓扑不满足通用约束。"""


def graph_from_positions(positions: np.ndarray, radius: float, tolerance: float = 1e-12) -> nx.Graph:
    """严格依据单位圆盘规则建图。"""
    n_nodes = positions.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        distances = np.linalg.norm(positions[i + 1 :] - positions[i], axis=1)
        for offset in np.flatnonzero(distances <= radius + tolerance):
            graph.add_edge(i, i + 1 + int(offset))
    return graph


def validate_topology(
    topology: Topology,
    constraints: Optional[Mapping[str, Any]] = None,
    tolerance: float = 1e-9,
) -> None:
    constraints = constraints or {}
    positions = np.asarray(topology.positions)
    graph = topology.graph
    n_nodes = positions.shape[0] if positions.ndim == 2 else 0

    if positions.ndim != 2 or positions.shape[1] != 2 or n_nodes < 2:
        raise TopologyValidationError("positions 必须具有形状 [N, 2] 且 N >= 2")
    if not np.all(np.isfinite(positions)):
        raise TopologyValidationError("节点位置必须全部有限")
    if np.unique(positions, axis=0).shape[0] != n_nodes:
        raise TopologyValidationError("节点位置必须唯一")
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise TopologyValidationError("graph 必须是无向简单图")
    if set(graph.nodes) != set(range(n_nodes)):
        raise TopologyValidationError("节点 ID 必须恰好为 0..N-1")
    if nx.number_of_selfloops(graph):
        raise TopologyValidationError("拓扑不允许自环")
    if not nx.is_connected(graph):
        raise TopologyValidationError("拓扑必须连通")
    if not np.isfinite(topology.communication_radius) or topology.communication_radius <= 0:
        raise TopologyValidationError("通信半径必须为有限正数")

    # 对每个点对同时检查边和非边，保证物理图与距离阈值完全一致。
    radius = topology.communication_radius
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if graph.has_edge(i, j) and distance > radius + tolerance:
                raise TopologyValidationError("边 ({}, {}) 超出通信半径".format(i, j))
            if not graph.has_edge(i, j) and distance <= radius - tolerance:
                raise TopologyValidationError("非边 ({}, {}) 位于通信半径内".format(i, j))

    degrees = np.asarray([graph.degree(i) for i in range(n_nodes)], dtype=float)
    if "min_degree" in constraints and degrees.min() < int(constraints["min_degree"]):
        raise TopologyValidationError("实际最小度不满足 min_degree")
    if "max_degree" in constraints and degrees.max() > int(constraints["max_degree"]):
        raise TopologyValidationError("实际最大度不满足 max_degree")
    if "mean_degree_min" in constraints and degrees.mean() < float(constraints["mean_degree_min"]):
        raise TopologyValidationError("实际平均度低于 mean_degree_min")
    if "mean_degree_max" in constraints and degrees.mean() > float(constraints["mean_degree_max"]):
        raise TopologyValidationError("实际平均度高于 mean_degree_max")

