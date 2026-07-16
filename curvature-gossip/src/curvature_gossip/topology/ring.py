"""生成圆周等间距节点组成的单位圆盘环拓扑。"""

from typing import Any, Mapping

import numpy as np

from .base import Topology, TopologyGenerator
from .registry import register_topology
from .validation import graph_from_positions, validate_topology


@register_topology("ring")
class RingGenerator(TopologyGenerator):
    def generate(self, rng: np.random.Generator, params: Mapping[str, Any]) -> Topology:
        del rng  # 环拓扑完全确定。
        n_nodes = int(params.get("n_nodes", 8))
        circle_radius = float(params.get("circle_radius_m", 1.0))
        if n_nodes < 4 or circle_radius <= 0:
            raise ValueError("环拓扑要求 n_nodes>=4 且 circle_radius_m>0")
        nearest = 2.0 * circle_radius * np.sin(np.pi / n_nodes)
        next_nearest = 2.0 * circle_radius * np.sin(2.0 * np.pi / n_nodes)
        default_radius = (nearest + next_nearest) / 2.0
        radius = float(params.get("communication_radius_m", default_radius))
        if radius < nearest or radius >= next_nearest:
            raise ValueError("环通信半径必须覆盖最近邻且排除次近邻")

        angles = 2.0 * np.pi * np.arange(n_nodes) / n_nodes
        positions = circle_radius * np.column_stack((np.cos(angles), np.sin(angles)))
        topology = Topology(
            positions, graph_from_positions(positions, radius), radius, None,
            {"type": "ring", "circle_radius_m": circle_radius},
        )
        validate_topology(topology, params)
        return topology

