"""Load the node-only CTDE actor and produce decentralized broadcast probabilities."""

from ..learning.ctde_ppo import CTDEPPO
from ..learning.features import (
    encode_observations, encode_stage1_observations, encode_stage2_observations,
    encode_stage2_mpnn_observations,
)
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
        self.actor = dict(actor or {})
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
        if self.model.actor_stage == 2:
            residual = self.actor.get("residual", {})
            include_scenario_context = bool(residual.get("include_scenario_context", False))
            use_curvature = bool(self.actor.get("curvature", {}).get("enabled", True))
            if residual.get("architecture", "mlp") == "mpnn":
                return encode_stage2_mpnn_observations(
                    observations, self.target_tx_ratio, self.update_probability,
                    self.consecutive_tx_scale, self.neighbor_confidence_time_constant,
                    self.congestion_feature_scale, include_scenario_context,
                    use_curvature_edge_feature=bool(residual.get("mpnn", {}).get("use_curvature_edge_feature", False)),
                    bmax=float(residual.get("mpnn", {}).get("bmax", 1.0)),
                    edge_normalization=str(residual.get("mpnn", {}).get("edge_normalization", "raw_bmax")),
                    edge_freshness_feature=str(residual.get("mpnn", {}).get("edge_freshness_feature", "binary_fraction")),
                    version_gap_tau=residual.get("mpnn", {}).get("version_gap_tau"),
                    use_curvature=use_curvature,
                    use_node_curvature_score=bool(residual.get("mpnn", {}).get("use_node_curvature_score", False)),
                    use_raw_af3_min_edge_curvature=bool(residual.get("mpnn", {}).get("use_raw_af3_min_edge_curvature", False)),
                    node_curvature_normalization=str(residual.get("mpnn", {}).get("node_curvature_normalization", "raw_bmax")),
                )
            return encode_stage2_observations(
                observations, self.target_tx_ratio, self.update_probability,
                self.consecutive_tx_scale, self.neighbor_confidence_time_constant,
                self.congestion_feature_scale, include_scenario_context,
                use_curvature=use_curvature,
            )
        return encode_observations(
            observations, self.target_tx_ratio, self.update_probability,
            self.consecutive_tx_scale, self.neighbor_confidence_time_constant,
            self.congestion_feature_scale,
        )

    def close(self) -> None:
        self.model.close()
