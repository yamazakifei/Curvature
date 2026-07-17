"""加载训练后的共享 actor，在每个节点仅凭本地观测输出广播概率。"""

from ..learning.ctde_ppo import CTDEPPO
from ..learning.features import encode_observations
from ..simulator.observations import NodeObservation
from .base import DistributedBroadcastPolicy


class NeuralCTDEPolicy(DistributedBroadcastPolicy):
    def __init__(
        self,
        checkpoint_path: str,
        target_tx_ratio: float = 0.1,
        q_min: float = 0.001,
        q_max: float = 0.8,
    ) -> None:
        self.target_tx_ratio = float(target_tx_ratio)
        self.model = CTDEPPO(q_min=q_min, q_max=q_max)
        self.model.restore(checkpoint_path)

    def transmission_probability(self, observation: NodeObservation) -> float:
        encoded = encode_observations((observation,), self.target_tx_ratio)
        return float(self.model.predict_probabilities(encoded)[0])

    def transmission_probabilities(self, observations):
        encoded = encode_observations(tuple(observations), self.target_tx_ratio)
        return self.model.predict_probabilities(encoded)

    def close(self) -> None:
        self.model.close()
