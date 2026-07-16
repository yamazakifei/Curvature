"""实现无需最优运输求解的全局离线 AF3 边曲率后端。"""

from typing import Dict, Tuple

from ..topology.base import Topology
from .base import CurvatureProvider, CurvatureResult, canonical_edge, incident_node_means


class GlobalAF3Curvature(CurvatureProvider):
    def compute(self, topology: Topology) -> CurvatureResult:
        graph = topology.graph
        edge_values = {}  # type: Dict[Tuple[int, int], float]
        for i, j in graph.edges:
            common_neighbors = len(set(graph.neighbors(i)).intersection(graph.neighbors(j)))
            value = 4 - graph.degree(i) - graph.degree(j) + 3 * common_neighbors
            edge_values[canonical_edge(i, j)] = float(value)
        return CurvatureResult(
            method="global_af3",
            edge_values=edge_values,
            node_values=incident_node_means(topology, edge_values),
            metadata={"formula": "4-deg(i)-deg(j)+3*common_neighbors"},
        )

