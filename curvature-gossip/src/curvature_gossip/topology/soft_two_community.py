"""生成具有多个跨社区瓶颈边、但不存在固定单一网关的软双社区随机几何图。
流程：
1. 在两个相邻的圆盘中均匀采样节点位置。
2. 通过通信半径生成随机几何图。
3. 仅接受跨社区边数量和端点覆盖满足要求的样本。
4. 可选地要求每个社区的诱导子图连通。
5. 可选地要求整个图的边连通度至少为指定值。
6. 可选地要求总边数在指定范围内。
7. 生成的拓扑可用于配对多策略实验。
参数：
- n_nodes: 节点总数 (至少 2)
- cluster_sizes: 两个社区的节点数 (必须为正且总和等于 n_nodes)
- communication_radius_m: 节点之间的通信半径 (必须为正)
- cluster_radius_m: 每个社区的采样圆盘半径 (必须为正)
- center_separation_m: 两个采样圆盘的中心间距 (必须非负且保证两个圆盘完整位于区域内)
- area_width_m: 区域宽度 (必须大于 center_separation + 2 * cluster_radius)
- area_height_m : 区域高度 (必须大于 2 * cluster_radius)
- cross_edge_count_range: 跨社区边数的闭区间 (必须在合法范围内)
- min_cross_endpoints_per_cluster: 每个社区至少有多少个节点参与跨社区边 (必须在合法社区规模内)
- require_induced_cluster_connected: 是否要求每个社区的诱导子图连通 (布尔值)
- min_edge_connectivity: 整个图的最小边连通度 (至少为 1)
- total_edge_count_range: 可选的总边数闭区间 (必须在合法范围内)
"""

from typing import Any, Mapping, Tuple

import networkx as nx
import numpy as np

from .base import Topology, TopologyGenerator
from .registry import register_topology
from .validation import TopologyValidationError, graph_from_positions, validate_topology


def _pair_of_positive_ints(
    params: Mapping[str, Any], name: str, default: Tuple[int, int]
) -> Tuple[int, int]:
    raw = params.get(name, default)
    if len(raw) != 2:
        raise ValueError("{} 必须包含两个整数".format(name))
    first, second = int(raw[0]), int(raw[1])
    if first < 0 or second < first:
        raise ValueError("{} 必须满足 0 <= lower <= upper".format(name))
    return first, second


def _cluster_sizes(params: Mapping[str, Any], n_nodes: int) -> Tuple[int, int]:
    raw = params.get("cluster_sizes", [n_nodes // 2, n_nodes - n_nodes // 2])
    if len(raw) != 2:
        raise ValueError("cluster_sizes 必须包含两个正整数")
    left, right = int(raw[0]), int(raw[1])
    if min(left, right) < 1 or left + right != n_nodes:
        raise ValueError("cluster_sizes 必须为正且总和等于 n_nodes")
    return left, right


def _sample_disk(
    rng: np.random.Generator, count: int, center: np.ndarray, disk_radius: float
) -> np.ndarray:
    """按面积均匀地在圆盘中采样，而不是让节点偏向圆心。"""
    angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
    radii = disk_radius * np.sqrt(rng.uniform(0.0, 1.0, size=count))
    offsets = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return center + offsets


def _scaled_geometry_value(
    params: Mapping[str, Any], name: str, value: float, n_nodes: int,
) -> float:
    """Keep node density and local distances stable while changing N.

    The communication radius stays in physical units.  The enclosing area and
    community geometry therefore grow with sqrt(N), which keeps both density
    and the local neighbor-distance distribution approximately unchanged.
    """
    if not bool(params.get("scale_geometry_with_n", False)):
        return float(value)
    reference_n = int(params.get("reference_n_nodes", params.get("n_nodes", n_nodes)))
    if reference_n < 2:
        raise ValueError("reference_n_nodes must be at least two")
    return float(value) * float(np.sqrt(float(n_nodes) / float(reference_n)))


def _n_dependent_range(
    params: Mapping[str, Any], name: str, n_nodes: int, fallback: Tuple[int, int],
) -> Tuple[int, int]:
    """Resolve an integer range, linearly interpolating configured N anchors."""
    mapping = params.get(name)
    if not isinstance(mapping, Mapping) or not mapping:
        return fallback
    anchors = []
    for key, raw_range in mapping.items():
        anchor_n = int(key)
        anchors.append((
            anchor_n,
            _pair_of_positive_ints({"range": raw_range}, "range", fallback),
        ))
    anchors.sort()
    if n_nodes <= anchors[0][0]:
        return anchors[0][1]
    if n_nodes >= anchors[-1][0]:
        return anchors[-1][1]
    for (left_n, left), (right_n, right) in zip(anchors, anchors[1:]):
        if left_n <= n_nodes <= right_n:
            fraction = float(n_nodes - left_n) / float(right_n - left_n)
            low = int(round(left[0] + fraction * (right[0] - left[0])))
            high = int(round(left[1] + fraction * (right[1] - left[1])))
            return max(0, low), max(low, high)
    return fallback


@register_topology("soft_two_community")
class SoftTwoCommunityGenerator(TopologyGenerator):
    """通过两个相邻采样圆盘生成边界柔和、具有多条跨社区边的拓扑。"""

    def generate(self, rng: np.random.Generator, params: Mapping[str, Any]) -> Topology:
        n_nodes = int(params.get("n_nodes", 40))
        left_size, right_size = _cluster_sizes(params, n_nodes)
        communication_radius = float(params.get("communication_radius_m", 40.0))
        cluster_radius = _scaled_geometry_value(
            params, "cluster_radius_m", float(params.get("cluster_radius_m", 70.0)), n_nodes
        )
        center_separation = _scaled_geometry_value(
            params, "center_separation_m", float(params.get("center_separation_m", 120.0)), n_nodes
        )
        width_default = center_separation + 2.0 * cluster_radius
        height_default = 2.0 * cluster_radius
        width = _scaled_geometry_value(
            params, "area_width_m", float(params.get("area_width_m", width_default)), n_nodes
        )
        height = _scaled_geometry_value(
            params, "area_height_m", float(params.get("area_height_m", height_default)), n_nodes
        )
        max_attempts = int(params.get("max_attempts", 2000))

        cross_low, cross_high = _pair_of_positive_ints(
            {"cross_edge_count_range": _n_dependent_range(
                params, "cross_edge_count_range_by_n", n_nodes,
                _pair_of_positive_ints(params, "cross_edge_count_range", (5, 15)),
            )},
            "cross_edge_count_range", (5, 15),
        )
        min_cross_endpoints = int(params.get("min_cross_endpoints_per_cluster", 3))
        require_cluster_connected = bool(params.get("require_induced_cluster_connected", True))
        min_edge_connectivity = int(params.get("min_edge_connectivity", 1))

        total_edge_range = None
        if "total_edge_count_range" in params:
            total_edge_range = _pair_of_positive_ints(
                params, "total_edge_count_range", (0, n_nodes * (n_nodes - 1) // 2)
            )

        if n_nodes < 2 or min(communication_radius, cluster_radius, width, height) <= 0:
            raise ValueError("节点数至少为 2，通信半径、簇半径和区域尺寸必须为正")
        if center_separation < 0 or max_attempts < 1:
            raise ValueError("center_separation_m 必须非负且 max_attempts 至少为 1")
        if center_separation + 2.0 * cluster_radius > width or 2.0 * cluster_radius > height:
            raise ValueError("两个采样圆盘必须完整位于 area_width_m x area_height_m 区域内")
        if cross_high > left_size * right_size:
            raise ValueError("cross_edge_count_range 超过两个社区间可能的最大边数")
        if not 1 <= min_cross_endpoints <= min(left_size, right_size):
            raise ValueError("min_cross_endpoints_per_cluster 必须在合法社区规模内")
        if min_edge_connectivity < 1:
            raise ValueError("min_edge_connectivity 必须至少为 1")

        labels = np.concatenate((np.zeros(left_size, dtype=int), np.ones(right_size, dtype=int)))
        center_y = height / 2.0
        left_center = np.asarray([(width - center_separation) / 2.0, center_y])
        right_center = np.asarray([(width + center_separation) / 2.0, center_y])

        for attempt in range(1, max_attempts + 1):
            positions = np.vstack((
                _sample_disk(rng, left_size, left_center, cluster_radius),
                _sample_disk(rng, right_size, right_center, cluster_radius),
            ))
            graph = graph_from_positions(positions, communication_radius)

            # 跨社区边由物理距离自然产生，只接受数量和端点覆盖满足要求的样本。
            cross_edges = sorted(
                (min(i, j), max(i, j))
                for i, j in graph.edges
                if labels[i] != labels[j]
            )
            left_endpoints = {i if labels[i] == 0 else j for i, j in cross_edges}
            right_endpoints = {i if labels[i] == 1 else j for i, j in cross_edges}
            if not cross_low <= len(cross_edges) <= cross_high:
                continue
            if min(len(left_endpoints), len(right_endpoints)) < min_cross_endpoints:
                continue
            if total_edge_range is not None:
                edge_low, edge_high = total_edge_range
                if not edge_low <= graph.number_of_edges() <= edge_high:
                    continue
            if require_cluster_connected:
                left_connected = nx.is_connected(graph.subgraph(range(left_size)))
                right_connected = nx.is_connected(graph.subgraph(range(left_size, n_nodes)))
                if not left_connected or not right_connected:
                    continue

            topology = Topology(
                positions=positions,
                graph=graph,
                communication_radius=communication_radius,
                node_labels=labels,
                metadata={
                    "type": "soft_two_community",
                    "attempts": attempt,
                    "inter_cluster_edges": cross_edges,
                    "cross_endpoints": {
                        "left": sorted(left_endpoints),
                        "right": sorted(right_endpoints),
                    },
                    "cluster_centers": [left_center.tolist(), right_center.tolist()],
                },
            )
            volumes = [
                float(sum(graph.degree(i) for i in range(left_size))),
                float(sum(graph.degree(i) for i in range(left_size, n_nodes))),
            ]
            cross_count = len(cross_edges)
            topology.metadata.update({
                "node_density_per_m2": float(n_nodes / (width * height)),
                "mean_degree": float(2.0 * graph.number_of_edges() / n_nodes),
                "cross_edge_count": int(cross_count),
                "community_conductance": float(cross_count / max(min(volumes), 1.0)),
                "normalized_cut_strength": float(
                    cross_count / max(volumes[0], 1.0) + cross_count / max(volumes[1], 1.0)
                ),
                "scaled_geometry": bool(params.get("scale_geometry_with_n", False)),
                "reference_n_nodes": int(params.get("reference_n_nodes", n_nodes)),
                "resolved_cross_edge_count_range": [int(cross_low), int(cross_high)],
            })
            try:
                validate_topology(topology, params)
                # 大于 1 时排除任何一条边单独成为全图割边的情况。
                if nx.edge_connectivity(graph) < min_edge_connectivity:
                    raise TopologyValidationError("实际边连通度低于 min_edge_connectivity")
                return topology
            except TopologyValidationError:
                continue

        raise TopologyValidationError(
            "soft_two_community 在 {} 次尝试后仍未满足约束; params={}".format(
                max_attempts, dict(params)
            )
        )
