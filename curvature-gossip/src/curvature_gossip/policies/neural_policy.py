"""Load the node-only CTDE actor and produce decentralized broadcast probabilities."""

from ..learning.ctde_ppo import CTDEPPO
from ..learning.features import encode_observations
from ..simulator.observations import NodeObservation
from .base import DistributedBroadcastPolicy


class NeuralCTDEPolicy(DistributedBroadcastPolicy):
    """Run the V3 shared actor with local state and fixed environment context."""

    def __init__(
        self,
        checkpoint_path: str,
        target_tx_ratio: float = 0.1,
        update_probability: float = 0.05,
        consecutive_tx_scale: float = 3.0,
        neighbor_confidence_time_constant: float = 20.0,
    ) -> None:
        self.target_tx_ratio = float(target_tx_ratio)
        self.update_probability = float(update_probability)
        self.consecutive_tx_scale = float(consecutive_tx_scale)
        self.neighbor_confidence_time_constant = float(neighbor_confidence_time_constant)
        self.model = CTDEPPO()
        self.model.restore(checkpoint_path)

    def transmission_probability(self, observation: NodeObservation) -> float:
        encoded = encode_observations(
            (observation,), self.target_tx_ratio, self.update_probability,
            self.consecutive_tx_scale, self.neighbor_confidence_time_constant,
        )
        return float(self.model.predict_probabilities(encoded)[0])

    def transmission_probabilities(self, observations):
        encoded = encode_observations(
            tuple(observations), self.target_tx_ratio, self.update_probability,
            self.consecutive_tx_scale, self.neighbor_confidence_time_constant,
        )
        return self.model.predict_probabilities(encoded)

    def close(self) -> None:
        self.model.close()
