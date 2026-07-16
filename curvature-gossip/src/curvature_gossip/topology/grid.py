"""生成默认仅含水平和垂直最近邻边的二维规则网格。"""

from typing import Any, Mapping

import numpy as np

from .base import Topology, TopologyGenerator
from .registry import register_topology
from .validation import graph_from_positions, validate_topology


@register_topology("grid_2d")
class Grid2DGenerator(TopologyGenerator):
    def generate(self, rng: np.random.Generator, params: Mapping[str, Any]) -> Topology:
        del rng  # 规则网格不消耗随机数。
        rows = int(params.get("rows", 3))
        columns = int(params.get("columns", 4))
        spacing = float(params.get("spacing_m", 1.0))
        radius = float(params.get("communication_radius_m", spacing * (1.0 + 1e-8)))
        if rows < 1 or columns < 1 or rows * columns < 2 or spacing <= 0:
            raise ValueError("网格要求正行列数、至少两个节点及正间距")
        if radius < spacing:
            raise ValueError("通信半径必须至少覆盖水平/垂直最近邻")
        if not bool(params.get("include_diagonals", False)) and radius >= spacing * np.sqrt(2.0):
            raise ValueError("默认网格半径必须小于对角距离；如需对角边请设置 include_diagonals")

        positions = np.asarray(
            [(column * spacing, row * spacing) for row in range(rows) for column in range(columns)],
            dtype=float,
        )
        topology = Topology(
            positions, graph_from_positions(positions, radius), radius, None,
            {"type": "grid_2d", "rows": rows, "columns": columns},
        )
        validate_topology(topology, params)
        return topology

