"""Load the node-only CTDE actor and produce decentralized broadcast probabilities."""

from ..learning.ctde_ppo import CTDEPPO
from ..learning.features import encode_observations, encode_stage1_observations
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
        congestion_feature_scale: float = 5.0,
        actor=None,
    ) -> None:
        self.target_tx_ratio = float(target_tx_ratio)
        self.update_probability = float(update_probability)
        self.consecutive_tx_scale = float(consecutive_tx_scale)
        self.neighbor_confidence_time_constant = float(neighbor_confidence_time_constant)
        self.congestion_feature_scale = float(congestion_feature_scale)
        self.model = CTDEPPO(actor_config=actor)
        self.model.restore(checkpoint_path)

    def transmission_probability(self, observation: NodeObservation) -> float:
        encoded = self._encode((observation,))
        return float(self.model.predict_probabilities(encoded)[0])

    def transmission_probabilities(self, observations):
        encoded = self._encode(tuple(observations))
        return self.model.predict_probabilities(encoded)

    def _encode(self, observations):
        """Select the checkpoint-compatible local encoder without global state access."""
        if self.model.actor_stage == 1:
            return encode_stage1_observations(observations, self.target_tx_ratio)
        return encode_observations(
            observations, self.target_tx_ratio, self.update_probability,
            self.consecutive_tx_scale, self.neighbor_confidence_time_constant,
            self.congestion_feature_scale,
        )

    def close(self) -> None:
        self.model.close()
