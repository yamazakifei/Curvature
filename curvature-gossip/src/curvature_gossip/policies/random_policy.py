"""实现所有节点使用相同固定广播概率的均匀随机基线。"""

from ..simulator.observations import NodeObservation
from .base import DistributedBroadcastPolicy


class UniformRandomPolicy(DistributedBroadcastPolicy):
    def __init__(self, tx_probability: float = 0.1) -> None:
        if not 0.0 <= tx_probability <= 1.0:
            raise ValueError("tx_probability must be in [0, 1]")
        self.tx_probability = float(tx_probability)

    def transmission_probability(self, observation: NodeObservation) -> float:
        del observation
        return self.tx_probability

