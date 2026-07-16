"""生成矩形区域内均匀采样的连通随机几何图。"""

from typing import Any, Mapping

import numpy as np

from .base import Topology, TopologyGenerator
from .registry import register_topology
from .validation import TopologyValidationError, graph_from_positions, validate_topology


@register_topology("random_geometric")
class RandomGeometricGenerator(TopologyGenerator):
    def generate(self, rng: np.random.Generator, params: Mapping[str, Any]) -> Topology:
        n_nodes = int(params.get("n_nodes", 20))
        width = float(params.get("area_width_m", 1.0))
        height = float(params.get("area_height_m", 1.0))
        radius = float(params.get("communication_radius_m", 0.3))
        max_attempts = int(params.get("max_attempts", 500))
        if n_nodes < 2 or min(width, height, radius) <= 0 or max_attempts < 1:
            raise ValueError("随机几何图参数要求 N>=2，尺寸/半径/尝试次数为正")

        for attempt in range(1, max_attempts + 1):
            positions = rng.uniform([0.0, 0.0], [width, height], size=(n_nodes, 2))
            graph = graph_from_positions(positions, radius)
            topology = Topology(
                positions=positions,
                graph=graph,
                communication_radius=radius,
                node_labels=None,
                metadata={"type": "random_geometric", "attempts": attempt},
            )
            try:
                validate_topology(topology, params)
                return topology
            except TopologyValidationError:
                continue
        raise TopologyValidationError(
            "random_geometric 在 {} 次尝试后仍不满足约束; params={}".format(max_attempts, dict(params))
        )

