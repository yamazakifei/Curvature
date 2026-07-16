"""生成具有精确跨簇边数的双簇网关桥接单位圆盘图。"""

from typing import Any, Mapping, Tuple

import numpy as np

from .base import Topology, TopologyGenerator
from .registry import register_topology
from .validation import TopologyValidationError, graph_from_positions, validate_topology


def _cluster_sizes(params: Mapping[str, Any], n_nodes: int) -> Tuple[int, int]:
    raw = params.get("cluster_sizes")
    if raw is None:
        left = n_nodes // 2
        return left, n_nodes - left
    if len(raw) != 2 or int(raw[0]) + int(raw[1]) != n_nodes:
        raise ValueError("cluster_sizes 必须包含两个正数且总和等于 n_nodes")
    return int(raw[0]), int(raw[1])


@register_topology("two_cluster_bridge")
class TwoClusterBridgeGenerator(TopologyGenerator):
    def generate(self, rng: np.random.Generator, params: Mapping[str, Any]) -> Topology:
        n_nodes = int(params.get("n_nodes", 20))
        radius = float(params.get("communication_radius_m", 1.0))
        bridge_count = int(params.get("bridge_edge_count", params.get("gateway_pairs", 1)))
        max_attempts = int(params.get("max_attempts", 500))
        left_size, right_size = _cluster_sizes(params, n_nodes)
        if radius <= 0 or bridge_count < 1 or min(left_size, right_size) < bridge_count:
            raise ValueError("每个簇必须至少包含一个节点/网关，且通信半径为正")

        # 相邻走廊相隔 0.6R；配对网关横距 0.9R，使非配对网关距离超过 R。
        center_separation = float(params.get("center_separation_m", 2.3 * radius))
        gateway_offset = 0.7 * radius
        gateway_gap = center_separation - 2.0 * gateway_offset
        if not (0.0 < gateway_gap <= radius):
            raise ValueError("center_separation_m 需使网关横距位于 (0, communication_radius] 内")
        corridor_spacing = float(params.get("corridor_spacing_m", 0.6 * radius))
        if np.hypot(gateway_gap, corridor_spacing) <= radius:
            raise ValueError("corridor_spacing_m 太小，会产生非配对网关跨簇边")
        spread = float(params.get("cluster_spread_m", 0.08 * radius))
        if spread <= 0 or gateway_offset + np.sqrt(2.0) * spread >= radius:
            raise ValueError("cluster_spread_m 必须为正且保证网关连接本簇")

        width = float(params.get("area_width_m", center_separation + 2.5 * radius))
        height = float(params.get("area_height_m", max(3.0 * radius, bridge_count * corridor_spacing + radius)))
        y_offsets = (np.arange(bridge_count) - (bridge_count - 1) / 2.0) * corridor_spacing
        left_center_x = (width - center_separation) / 2.0
        right_center_x = left_center_x + center_separation
        center_y = height / 2.0

        for attempt in range(1, max_attempts + 1):
            positions = np.empty((n_nodes, 2), dtype=float)
            labels = np.concatenate((np.zeros(left_size, dtype=int), np.ones(right_size, dtype=int)))

            # 网关固定在内边界；其余节点围绕对应走廊中心有界采样。
            for corridor in range(bridge_count):
                y = center_y + y_offsets[corridor]
                positions[corridor] = [left_center_x + gateway_offset, y]
                positions[left_size + corridor] = [right_center_x - gateway_offset, y]
            for local_id in range(bridge_count, left_size):
                corridor = (local_id - bridge_count) % bridge_count
                jitter = rng.uniform(-spread, spread, size=2)
                positions[local_id] = [left_center_x, center_y + y_offsets[corridor]] + jitter
            for local_id in range(bridge_count, right_size):
                corridor = (local_id - bridge_count) % bridge_count
                jitter = rng.uniform(-spread, spread, size=2)
                positions[left_size + local_id] = [right_center_x, center_y + y_offsets[corridor]] + jitter

            graph = graph_from_positions(positions, radius)
            cross_edges = sorted(
                (min(i, j), max(i, j)) for i, j in graph.edges if labels[i] != labels[j]
            )
            topology = Topology(
                positions, graph, radius, labels,
                {
                    "type": "two_cluster_bridge",
                    "attempts": attempt,
                    "inter_cluster_edges": cross_edges,
                    "gateway_pairs": [(i, left_size + i) for i in range(bridge_count)],
                },
            )
            try:
                validate_topology(topology, params)
                if "bridge_edge_count_range" in params:
                    low, high = params["bridge_edge_count_range"]
                    valid_count = int(low) <= len(cross_edges) <= int(high)
                else:
                    valid_count = len(cross_edges) == bridge_count
                if not valid_count:
                    raise TopologyValidationError("实际跨簇边数为 {}".format(len(cross_edges)))
                # 每个固定网关都必须连接至少一个同簇节点。
                gateways = list(range(bridge_count)) + list(range(left_size, left_size + bridge_count))
                if any(not any(labels[node] == labels[nbr] for nbr in graph.neighbors(node)) for node in gateways):
                    raise TopologyValidationError("至少一个网关未连接本簇")
                return topology
            except TopologyValidationError:
                continue
        raise TopologyValidationError(
            "two_cluster_bridge 在 {} 次尝试后失败; 约束=精确/范围跨簇边与连通性, params={}".format(
                max_attempts, dict(params)
            )
        )

