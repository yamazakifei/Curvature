"""通过线性规划精确计算固定无权图每条边的全局 Ollivier-Ricci 曲率。"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linprog

from ..topology.base import Topology
from .base import CurvatureProvider, CurvatureResult, canonical_edge, incident_node_means


class GlobalORCCurvature(CurvatureProvider):
    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("ORC alpha must be in [0, 1]")
        self.alpha = float(alpha)

    def _measure(self, graph: nx.Graph, node: int):
        neighbors = sorted(graph.neighbors(node))
        support = [node] + neighbors
        mass = np.zeros(len(support), dtype=float)
        mass[0] = self.alpha
        if neighbors:
            mass[1:] = (1.0 - self.alpha) / len(neighbors)
        return support, mass

    @staticmethod
    def _transport_distance(
        source_support: List[int],
        source_mass: np.ndarray,
        target_support: List[int],
        target_mass: np.ndarray,
        shortest_paths: np.ndarray,
    ) -> float:
        n_source = len(source_support)
        n_target = len(target_support)
        costs = shortest_paths[np.ix_(source_support, target_support)].reshape(-1)
        constraints = []
        bounds = []

        # 每个源支持点的流出量等于其质量。
        for source_index in range(n_source):
            row = np.zeros((n_source, n_target), dtype=float)
            row[source_index, :] = 1.0
            constraints.append(row.reshape(-1))
            bounds.append(source_mass[source_index])
        # 每个目标支持点的流入量等于其质量。
        for target_index in range(n_target):
            row = np.zeros((n_source, n_target), dtype=float)
            row[:, target_index] = 1.0
            constraints.append(row.reshape(-1))
            bounds.append(target_mass[target_index])

        solution = linprog(
            costs,
            A_eq=np.asarray(constraints),
            b_eq=np.asarray(bounds),
            bounds=(0.0, None),
            method="highs",
        )
        if not solution.success or solution.fun is None:
            raise RuntimeError("ORC transport LP failed: {}".format(solution.message))
        return float(solution.fun)

    def compute(self, topology: Topology) -> CurvatureResult:
        graph = topology.graph
        if not nx.is_connected(graph):
            raise ValueError("ORC requires a connected topology")
        n_nodes = graph.number_of_nodes()
        shortest_paths = np.full((n_nodes, n_nodes), np.inf, dtype=float)
        # 全源最短路只计算一次并供所有边的运输代价复用。
        for source, lengths in nx.all_pairs_shortest_path_length(graph):
            for target, distance in lengths.items():
                shortest_paths[source, target] = distance

        edge_values = {}  # type: Dict[Tuple[int, int], float]
        for i, j in sorted(graph.edges):
            source_support, source_mass = self._measure(graph, i)
            target_support, target_mass = self._measure(graph, j)
            wasserstein = self._transport_distance(
                source_support, source_mass, target_support, target_mass, shortest_paths
            )
            graph_distance = shortest_paths[i, j]
            curvature = 1.0 - wasserstein / graph_distance
            if not np.isfinite(curvature):
                raise RuntimeError("non-finite ORC on edge ({}, {})".format(i, j))
            edge_values[canonical_edge(i, j)] = float(curvature)

        return CurvatureResult(
            method="global_orc",
            edge_values=edge_values,
            node_values=incident_node_means(topology, edge_values),
            metadata={"alpha": self.alpha, "distance": "unweighted_shortest_path"},
        )

