"""把可变节点数和邻居数的本地观测编码为排列不变网络所需张量。"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..simulator.observations import NodeObservation


NODE_FEATURE_DIM = 13
EDGE_FEATURE_DIM = 7
GLOBAL_STATE_DIM = 6


@dataclass(frozen=True)
class EncodedObservations:
    node_features: np.ndarray
    edge_features: np.ndarray
    edge_mask: np.ndarray


def _safe_scale(value: float) -> float:
    return float(np.tanh(max(float(value), 0.0)))


def _edge_features(observation: NodeObservation, neighbor_index: int) -> np.ndarray:
    neighbor = observation.neighbor_ids[neighbor_index]
    valid = bool(observation.neighbor_estimate_valid[neighbor_index])
    if valid:
        differences = np.maximum(
            observation.own_cache_versions
            - observation.neighbor_cache_estimates[neighbor_index],
            0,
        ).astype(float)
        differences = np.log1p(differences) / np.log1p(
            max(2, observation.own_cache_versions.size)
        )
    else:
        differences = np.zeros(observation.own_cache_versions.size, dtype=float)
    positive_fraction = float(np.mean(differences > 0.0))
    curvature = float(observation.incident_curvatures[neighbor])
    importance = float(observation.incident_bottleneck_importance[neighbor])
    return np.asarray([
        float(np.max(differences)),
        float(np.mean(differences)),
        float(np.percentile(differences, 90)),
        positive_fraction,
        float(valid),
        float(np.tanh(curvature / 8.0)),
        importance,
    ], dtype=np.float32)


def encode_observations(
    observations: Sequence[NodeObservation],
    target_tx_ratio: float,
) -> EncodedObservations:
    """编码节点集合；邻居顺序只影响 padding 顺序，不影响后续池化结果。"""
    observations = tuple(observations)
    if not observations:
        raise ValueError("at least one node observation is required")
    maximum_degree = max(1, max(len(item.neighbor_ids) for item in observations))
    node_features = np.zeros((len(observations), NODE_FEATURE_DIM), dtype=np.float32)
    edge_features = np.zeros(
        (len(observations), maximum_degree, EDGE_FEATURE_DIM), dtype=np.float32
    )
    edge_mask = np.zeros((len(observations), maximum_degree), dtype=np.float32)

    for row, observation in enumerate(observations):
        curvatures = np.asarray(list(observation.incident_curvatures.values()), dtype=float)
        negative_fraction = float(np.mean(curvatures < 0.0)) if curvatures.size else 0.0
        valid_fraction = float(np.mean(observation.neighbor_estimate_valid)) if observation.neighbor_ids else 0.0
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
        ], dtype=np.float32)
        for column in range(len(observation.neighbor_ids)):
            edge_features[row, column] = _edge_features(observation, column)
            edge_mask[row, column] = 1.0
    return EncodedObservations(node_features, edge_features, edge_mask)
