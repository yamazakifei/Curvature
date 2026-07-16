"""构建字段白名单严格受限、与全局可变状态隔离的节点观测。"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

import numpy as np

from ..curvature import CurvatureResult, canonical_edge
from ..state import LocalKnowledge, VersionState
from ..topology.base import Topology


@dataclass(frozen=True)
class NodeObservation:
    node_id: int
    own_cache_versions: np.ndarray
    neighbor_ids: Tuple[int, ...]
    incident_curvatures: Mapping[int, float]
    incident_bottleneck_importance: Mapping[int, float]
    neighbor_cache_estimates: np.ndarray
    neighbor_estimate_valid: np.ndarray
    time_since_last_tx: int
    transmitted_previous_slot: bool
    previous_interference_power: float
    previous_interference_valid: bool
    congestion_ewma: float
    consecutive_tx_attempts: int


class ObservationBuilder:
    def __init__(
        self,
        topology: Topology,
        curvature: CurvatureResult,
        bottleneck_importance: Mapping[Tuple[int, int], float],
    ) -> None:
        self.topology = topology
        self.curvature = curvature
        self.bottleneck_importance = bottleneck_importance

    @staticmethod
    def _readonly_copy(values: np.ndarray) -> np.ndarray:
        copied = np.asarray(values).copy()
        copied.setflags(write=False)
        return copied

    def build(self, node_id: int, slot: int, state: VersionState, knowledge: LocalKnowledge) -> NodeObservation:
        neighbors = tuple(sorted(self.topology.graph.neighbors(node_id)))
        incident_curvatures = {
            neighbor: self.curvature.edge_values[canonical_edge(node_id, neighbor)]
            for neighbor in neighbors
        }
        incident_importance = {
            neighbor: self.bottleneck_importance[canonical_edge(node_id, neighbor)]
            for neighbor in neighbors
        }
        estimates = knowledge.neighbor_cache_estimate[node_id, list(neighbors), :]
        validity = knowledge.neighbor_estimate_valid[node_id, list(neighbors)]
        last_tx = int(knowledge.last_tx_slot[node_id])
        time_since = slot + 1 if last_tx < 0 else max(0, slot - last_tx)
        # 所有数组均为深拷贝，策略无法通过观测修改仿真器内部状态。
        return NodeObservation(
            node_id=int(node_id),
            own_cache_versions=self._readonly_copy(state.cache_versions[node_id]),
            neighbor_ids=neighbors,
            incident_curvatures=MappingProxyType(incident_curvatures),
            incident_bottleneck_importance=MappingProxyType(incident_importance),
            neighbor_cache_estimates=self._readonly_copy(estimates),
            neighbor_estimate_valid=self._readonly_copy(validity),
            time_since_last_tx=time_since,
            transmitted_previous_slot=bool(knowledge.previous_transmitted[node_id]),
            previous_interference_power=float(knowledge.previous_interference_power[node_id]),
            previous_interference_valid=bool(knowledge.previous_interference_valid[node_id]),
            congestion_ewma=float(knowledge.congestion_ewma[node_id]),
            consecutive_tx_attempts=int(knowledge.consecutive_tx_attempts[node_id]),
        )
