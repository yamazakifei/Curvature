"""定义节点独立发送概率接口、公共参数解析与轻量策略工厂。"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from ..simulator.observations import NodeObservation


class DistributedBroadcastPolicy(ABC):
    @abstractmethod
    def transmission_probability(self, observation: "NodeObservation") -> float:
        raise NotImplementedError

    def transmission_probabilities(self, observations) -> np.ndarray:
        """默认逐节点计算；神经策略可覆盖该方法以执行一次批量推理。"""
        return np.asarray([
            self.transmission_probability(observation)
            for observation in observations
        ], dtype=float)


def create_policy(policy_type: str, params: Mapping[str, Any]) -> DistributedBroadcastPolicy:
    # 延迟导入避免策略模块之间形成循环依赖。
    from .curvature_freshness_policy import CurvatureFreshnessPolicy
    from .freshness_policy import FreshnessPolicy
    from .neural_policy import NeuralCTDEPolicy
    from .random_policy import UniformRandomPolicy

    registry = {
        "random": UniformRandomPolicy,
        "freshness": FreshnessPolicy,
        "curvature_freshness": CurvatureFreshnessPolicy,
        "neural_ctde": NeuralCTDEPolicy,
    }
    try:
        return registry[policy_type](**dict(params))
    except KeyError:
        raise ValueError("unknown policy type: {}".format(policy_type))
