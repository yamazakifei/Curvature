"""用负 ORC 瓶颈重要度放大 freshness 增益，并导出双分数诊断。"""

import numpy as np

from ..simulator.observations import NodeObservation
from .freshness_policy import FreshnessPolicy


class CurvatureFreshnessPolicy(FreshnessPolicy):
    def __init__(self, curvature_weight: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.curvature_weight = float(curvature_weight)

    def node_score(self, observation: NodeObservation) -> float:
        gains = self.edge_gains(observation)  # 基础边增益已完成压缩和归一化。
        if gains.size:
            # incident_bottleneck_importance 是由负 ORC 转换的 0-1 瓶颈重要度。
            importance = np.asarray([
                observation.incident_bottleneck_importance[neighbor]
                for neighbor in observation.neighbor_ids
            ])
            gated_gains = (1.0 + self.curvature_weight * importance) * gains
            innovation_score = float(np.max(gated_gains))
        else:
            innovation_score = 0.0
        return innovation_score + self.age_weight * observation.time_since_last_tx

    def probability_diagnostics(self, observation: NodeObservation):
        """在同一局部观测上同时记录未加 ORC 和加 ORC 的节点分数。"""
        fresh_score = super().node_score(observation)
        orc_score = self.node_score(observation)
        base_probability, backoff_factor, final_probability = (
            self._probability_components(orc_score, observation)
        )
        return {
            "fresh_score": float(fresh_score),
            "orc_score": float(orc_score),
            "base_probability": base_probability,
            "backoff_factor": backoff_factor,
            "final_probability": final_probability,
        }
