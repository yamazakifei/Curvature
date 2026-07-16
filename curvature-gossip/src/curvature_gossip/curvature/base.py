"""定义所有曲率后端共享的结果对象、接口和负曲率重要度变换。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from ..topology.base import Topology


def canonical_edge(i: int, j: int) -> Tuple[int, int]:
    return (int(i), int(j)) if i < j else (int(j), int(i))


@dataclass(frozen=True)
class CurvatureResult:
    method: str
    edge_values: Mapping[Tuple[int, int], float]
    node_values: np.ndarray
    metadata: Mapping[str, Any]

    def value(self, i: int, j: int) -> float:
        return float(self.edge_values[canonical_edge(i, j)])


class CurvatureProvider(ABC):
    @abstractmethod
    def compute(self, topology: Topology) -> CurvatureResult:
        raise NotImplementedError


def incident_node_means(topology: Topology, edge_values: Mapping[Tuple[int, int], float]) -> np.ndarray:
    values = np.zeros(topology.graph.number_of_nodes(), dtype=float)
    for node in topology.graph.nodes:
        incident = [edge_values[canonical_edge(node, neighbor)] for neighbor in topology.graph.neighbors(node)]
        values[node] = float(np.mean(incident))
    return values


def bottleneck_importance(
    topology: Topology,
    result: CurvatureResult,
    normalization: str = "global_negative_max",
) -> Dict[Tuple[int, int], float]:
    """将负曲率转换到非负瓶颈重要度；默认全图最大负部归一化。"""
    raw = {edge: max(-float(value), 0.0) for edge, value in result.edge_values.items()}
    if normalization == "none":
        return raw
    if normalization == "global_negative_max":
        maximum = max(raw.values()) if raw else 0.0
        return {edge: value / maximum if maximum > 0 else 0.0 for edge, value in raw.items()}
    if normalization == "incident_negative_max":
        output = {}
        for edge, value in raw.items():
            i, j = edge
            incident = [
                raw[canonical_edge(i, neighbor)] for neighbor in topology.graph.neighbors(i)
            ] + [
                raw[canonical_edge(j, neighbor)] for neighbor in topology.graph.neighbors(j)
            ]
            maximum = max(incident) if incident else 0.0
            output[edge] = value / maximum if maximum > 0 else 0.0
        return output
    raise ValueError("unknown curvature normalization: {}".format(normalization))

