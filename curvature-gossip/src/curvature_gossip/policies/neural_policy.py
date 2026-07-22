"""Load the node-only CTDE actor and produce decentralized broadcast probabilities."""

from ..learning.ctde_ppo import CTDEPPO
from ..learning.features import encode_observations
from ..simulator.observations import NodeObservation
from .base import DistributedBroadcastPolicy


class NeuralCTDEPolicy(DistributedBroadcastPolicy):
    """Run the shared V2 actor with locally available state and known environment context."""

    def __init__(
        self,
        checkpoint_path: str,
        target_tx_ratio: float = 0.1,
        update_probability: float = 0.05,
    ) -> None:
        self.target_tx_ratio = float(target_tx_ratio)
        self.update_probability = float(update_probability)
        self.model = CTDEPPO()
        self.model.restore(checkpoint_path)

    def transmission_probability(self, observation: NodeObservation) -> float:
        encoded = encode_observations(
            (observation,), self.target_tx_ratio, self.update_probability
        )
        return float(self.model.predict_probabilities(encoded)[0])

    def transmission_probabilities(self, observations):
        encoded = encode_observations(
            tuple(observations), self.target_tx_ratio, self.update_probability
        )
        return self.model.predict_probabilities(encoded)

    def close(self) -> None:
        self.model.close()
