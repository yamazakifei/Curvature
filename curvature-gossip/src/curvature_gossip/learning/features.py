"""Encode local node observations and training context for the node-only CTDE policy."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..simulator.observations import NodeObservation


NODE_FEATURE_DIM = 14
BASE_GLOBAL_STATE_DIM = 6
GLOBAL_STATE_DIM = 9
TARGET_TX_RATIO_INDEX = 8
UPDATE_PROBABILITY_INDEX = 13


@dataclass(frozen=True)
class EncodedObservations:
    """Fixed-width local inputs consumed by the shared node actor."""

    node_features: np.ndarray


def _safe_scale(value: float) -> float:
    return float(np.tanh(max(float(value), 0.0)))


def encode_observations(
    observations: Sequence[NodeObservation],
    target_tx_ratio: float,
    update_probability: float,
) -> EncodedObservations:
    """Build node-only features; neighbor-specific edge tensors are intentionally omitted."""
    observations = tuple(observations)
    if not observations:
        raise ValueError("at least one node observation is required")
    if not 0.0 < float(target_tx_ratio) <= 1.0:
        raise ValueError("target_tx_ratio must be in (0, 1]")
    if not 0.0 <= float(update_probability) <= 1.0:
        raise ValueError("update_probability must be in [0, 1]")

    node_features = np.zeros((len(observations), NODE_FEATURE_DIM), dtype=np.float32)
    for row, observation in enumerate(observations):
        curvatures = np.asarray(
            list(observation.incident_curvatures.values()), dtype=float
        )
        negative_fraction = (
            float(np.mean(curvatures < 0.0)) if curvatures.size else 0.0
        )
        valid_fraction = (
            float(np.mean(observation.neighbor_estimate_valid))
            if observation.neighbor_ids
            else 0.0
        )

        # Keep aggregate local topology information while removing the per-edge branch.
        node_features[row] = np.asarray([
            np.log1p(len(observation.neighbor_ids)) / 5.0,
            np.log1p(observation.time_since_last_tx) / 5.0,
            float(observation.transmitted_previous_slot),
            _safe_scale(observation.congestion_ewma / 5.0),
            np.log1p(observation.consecutive_tx_attempts) / 5.0,
            float(observation.previous_interference_valid),
            _safe_scale(observation.broadcast_debt / 5.0),
            float(observation.broadcast_limit),
            float(target_tx_ratio),
            float(np.tanh(np.min(curvatures) / 8.0)) if curvatures.size else 0.0,
            float(np.tanh(np.mean(curvatures) / 8.0)) if curvatures.size else 0.0,
            negative_fraction,
            valid_fraction,
            float(update_probability),
        ], dtype=np.float32)
    return EncodedObservations(node_features=node_features)


def encode_global_state(
    base_state: np.ndarray,
    target_tx_ratio: float,
    update_probability: float,
    n_nodes: int,
) -> np.ndarray:
    """Append budget, offered load, and network scale to the centralized critic state."""
    state = np.asarray(base_state, dtype=np.float32).reshape(-1)
    if state.size != BASE_GLOBAL_STATE_DIM:
        raise ValueError("base global state must contain six simulator statistics")
    if int(n_nodes) <= 0:
        raise ValueError("n_nodes must be positive")
    context = np.asarray([
        float(target_tx_ratio),
        float(update_probability),
        np.log1p(int(n_nodes)) / 5.0,
    ], dtype=np.float32)
    return np.concatenate([state, context]).astype(np.float32, copy=False)
