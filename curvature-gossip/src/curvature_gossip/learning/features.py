"""Encode strictly local V3 actor inputs and centralized critic inputs.

The shared actor receives only data exposed by :class:`NodeObservation`; the
centralized critic is used only during training.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from ..simulator.observations import NodeObservation


NODE_FEATURE_DIM = 14
BASE_GLOBAL_STATE_DIM = 5
GLOBAL_STATE_DIM = 8
NETWORK_SIZE_INDEX = 11
UPDATE_PROBABILITY_INDEX = 12
TARGET_TX_RATIO_INDEX = 13


@dataclass(frozen=True)
class EncodedObservations:
    """Fixed-width local inputs consumed by the shared node actor."""

    node_features: np.ndarray


@dataclass(frozen=True)
class Stage1EncodedObservations:
    """Local Stage-1 inputs: one curvature score and the scenario anchor b."""

    curvature_scores: np.ndarray
    target_tx_ratios: np.ndarray


@dataclass(frozen=True)
class Stage2EncodedObservations:
    """Stage-2 local inputs: curvature base inputs plus non-curvature context."""

    curvature_scores: np.ndarray
    target_tx_ratios: np.ndarray
    residual_context: np.ndarray


STAGE1_FEATURE_NAMES = ("incident_bottleneck_max",)
STAGE2_CONTEXT_FEATURE_NAMES = (
    "normalized_degree", "time_since_last_tx", "consecutive_tx_attempts",
    "congestion_ewma", "self_information_increment", "neighbor_freshness_mean",
    "neighbor_freshness_max", "neighbor_confidence_mean",
)
STAGE2_SCENARIO_CONTEXT_FEATURE_NAMES = ("log_network_size", "update_probability", "target_tx_ratio")
STAGE2_CONTEXT_INDICES = (0, 1, 2, 3, 6, 7, 8, 10)


def _validate_probability(name: str, value: float, strict_lower: bool = False) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0 or (strict_lower and value == 0.0):
        bound = "(0, 1]" if strict_lower else "[0, 1]"
        raise ValueError(f"{name} must be in {bound}")
    return value


def _local_freshness_features(
    observation: NodeObservation,
    bottlenecks: np.ndarray,
    confidence_time_constant: float,
) -> Tuple[float, float, float, float]:
    """Aggregate valid local neighbor estimates without consulting global cache state."""
    degree = len(observation.neighbor_ids)
    estimates = np.asarray(observation.neighbor_cache_estimates, dtype=np.int64)
    valid = np.asarray(observation.neighbor_estimate_valid, dtype=bool)
    ages = np.asarray(observation.neighbor_estimate_age, dtype=np.int64)
    own = np.asarray(observation.own_cache_versions, dtype=np.int64)
    if estimates.shape != (degree, own.size) or valid.shape != (degree,) or ages.shape != (degree,):
        raise ValueError("neighbor estimate arrays must align with neighbor_ids and cache size")
    if bottlenecks.shape != (degree,):
        raise ValueError("incident bottleneck importance must align with neighbor_ids")
    if np.any(valid & (ages < 0)) or np.any(~valid & (ages != -1)):
        raise ValueError("valid estimates require nonnegative ages and invalid ones age -1")

    gains = np.zeros(degree, dtype=float)
    if degree:
        gains[valid] = np.mean(own[None, :] > estimates[valid], axis=1)
    confidence = np.zeros(degree, dtype=float)
    confidence[valid] = np.exp(-ages[valid] / confidence_time_constant)
    confidence_sum = float(np.sum(confidence))
    weighted_gain = float(np.dot(confidence, gains) / confidence_sum) if confidence_sum > 0.0 else 0.0
    max_gain = float(np.max(gains[valid])) if np.any(valid) else 0.0
    bottle_weights = confidence * bottlenecks
    bottle_sum = float(np.sum(bottle_weights))
    bottle_gain = float(np.dot(bottle_weights, gains) / bottle_sum) if bottle_sum > 0.0 else 0.0
    confidence_mean = float(np.mean(confidence)) if degree else 0.0
    return weighted_gain, max_gain, bottle_gain, confidence_mean


def encode_curvature_score(observations: Sequence[NodeObservation]) -> np.ndarray:
    """Encode the Stage-1 local AF3 score ``max_j b_ij`` as shape ``[N, 1]``.

    This encoder intentionally accesses only incident bottleneck values.  It
    neither reads dynamic state nor computes a graph-wide normalization.
    """
    observations = tuple(observations)
    if not observations:
        raise ValueError("at least one node observation is required")
    scores = np.zeros((len(observations), 1), dtype=np.float32)
    for row, observation in enumerate(observations):
        bottlenecks = np.asarray([
            observation.incident_bottleneck_importance[neighbor]
            for neighbor in observation.neighbor_ids
        ], dtype=float)
        if np.any(~np.isfinite(bottlenecks)) or np.any((bottlenecks < 0.0) | (bottlenecks > 1.0)):
            raise ValueError("bottleneck importance must be finite and in [0, 1]")
        scores[row, 0] = float(np.max(bottlenecks)) if bottlenecks.size else 0.0
    return scores


def encode_stage1_observations(
    observations: Sequence[NodeObservation], target_tx_ratio: float
) -> Stage1EncodedObservations:
    """Combine the Stage-1 curvature score with a per-node target-rate anchor."""
    observations = tuple(observations)
    target_tx_ratio = _validate_probability("target_tx_ratio", target_tx_ratio, True)
    return Stage1EncodedObservations(
        curvature_scores=encode_curvature_score(observations),
        target_tx_ratios=np.full(len(observations), target_tx_ratio, dtype=np.float32),
    )


def stage2_context_feature_names(include_scenario_context: bool = False) -> Tuple[str, ...]:
    """Return the checkpointed Stage-2 context order, excluding the q-base reference."""
    return STAGE2_CONTEXT_FEATURE_NAMES + (
        STAGE2_SCENARIO_CONTEXT_FEATURE_NAMES if include_scenario_context else ()
    )


def encode_stage2_context(
    observations: Sequence[NodeObservation],
    target_tx_ratio: float,
    update_probability: float,
    consecutive_tx_scale: float = 3.0,
    neighbor_confidence_time_constant: float = 20.0,
    congestion_feature_scale: float = 5.0,
    include_scenario_context: bool = False,
) -> np.ndarray:
    """Encode the clean Stage-2 context without duplicating curvature-max input."""
    encoded = encode_observations(
        observations, target_tx_ratio, update_probability, consecutive_tx_scale,
        neighbor_confidence_time_constant, congestion_feature_scale,
    ).node_features
    context = encoded[:, STAGE2_CONTEXT_INDICES]
    if include_scenario_context:
        context = np.concatenate([context, encoded[:, (11, 12, 13)]], axis=1)
    expected_width = 11 if include_scenario_context else 8
    if context.shape != (encoded.shape[0], expected_width) or not np.isfinite(context).all():
        raise ValueError("Stage-2 context must be finite with the configured fixed width")
    return context.astype(np.float32, copy=False)


def encode_stage2_observations(
    observations: Sequence[NodeObservation],
    target_tx_ratio: float,
    update_probability: float,
    consecutive_tx_scale: float = 3.0,
    neighbor_confidence_time_constant: float = 20.0,
    congestion_feature_scale: float = 5.0,
    include_scenario_context: bool = False,
) -> Stage2EncodedObservations:
    """Combine the Stage-1 score with clean local context for the residual MLP."""
    observations = tuple(observations)
    target_tx_ratio = _validate_probability("target_tx_ratio", target_tx_ratio, True)
    return Stage2EncodedObservations(
        curvature_scores=encode_curvature_score(observations),
        target_tx_ratios=np.full(len(observations), target_tx_ratio, dtype=np.float32),
        residual_context=encode_stage2_context(
            observations, target_tx_ratio, update_probability, consecutive_tx_scale,
            neighbor_confidence_time_constant, congestion_feature_scale,
            include_scenario_context,
        ),
    )


def encode_observations(
    observations: Sequence[NodeObservation],
    target_tx_ratio: float,
    update_probability: float,
    consecutive_tx_scale: float = 3.0,
    neighbor_confidence_time_constant: float = 20.0,
    congestion_feature_scale: float = 5.0,
) -> EncodedObservations:
    """Build the 14 V3 node features in their fixed checkpoint-compatible order."""
    observations = tuple(observations)
    if not observations:
        raise ValueError("at least one node observation is required")
    target_tx_ratio = _validate_probability("target_tx_ratio", target_tx_ratio, True)
    update_probability = _validate_probability("update_probability", update_probability)
    if not np.isfinite(consecutive_tx_scale) or consecutive_tx_scale <= 0.0:
        raise ValueError("consecutive_tx_scale must be positive")
    if not np.isfinite(neighbor_confidence_time_constant) or neighbor_confidence_time_constant <= 0.0:
        raise ValueError("neighbor_confidence_time_constant must be positive")
    if not np.isfinite(congestion_feature_scale) or congestion_feature_scale <= 0.0:
        raise ValueError("congestion_feature_scale must be positive")

    n_nodes = int(np.asarray(observations[0].own_cache_versions).size)
    if n_nodes <= 0:
        raise ValueError("own_cache_versions must not be empty")
    node_features = np.zeros((len(observations), NODE_FEATURE_DIM), dtype=np.float32)
    for row, observation in enumerate(observations):
        own = np.asarray(observation.own_cache_versions, dtype=np.int64)
        last_tx = np.asarray(observation.last_tx_cache_versions, dtype=np.int64)
        if own.shape != (n_nodes,) or last_tx.shape != (n_nodes,):
            raise ValueError("own and last transmitted cache versions must have common shape [N]")
        degree = len(observation.neighbor_ids)
        bottlenecks = np.asarray(
            [observation.incident_bottleneck_importance[neighbor] for neighbor in observation.neighbor_ids],
            dtype=float,
        )
        if np.any(~np.isfinite(bottlenecks)) or np.any((bottlenecks < 0.0) | (bottlenecks > 1.0)):
            raise ValueError("bottleneck importance must be finite and in [0, 1]")
        fresh_mean, fresh_max, fresh_bottle, confidence_mean = _local_freshness_features(
            observation, bottlenecks, float(neighbor_confidence_time_constant)
        )
        self_increment = float(np.mean(own > last_tx))
        node_features[row] = np.asarray([
            np.log1p(degree) / np.log1p(n_nodes),
            np.tanh(target_tx_ratio * max(0, int(observation.time_since_last_tx))),
            np.tanh(max(0, int(observation.consecutive_tx_attempts)) / consecutive_tx_scale),
            np.tanh(max(0.0, float(observation.congestion_ewma)) / congestion_feature_scale),
            float(np.max(bottlenecks)) if degree else 0.0,
            float(np.mean(bottlenecks)) if degree else 0.0,
            self_increment,
            fresh_mean,
            fresh_max,
            fresh_bottle,
            confidence_mean,
            np.log1p(n_nodes),
            update_probability,
            target_tx_ratio,
        ], dtype=np.float32)

    if not np.isfinite(node_features).all():
        raise ValueError("encoded node features must be finite")
    bounded_indices = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13)
    if np.any(node_features[:, bounded_indices] < -1e-6) or np.any(node_features[:, bounded_indices] > 1.0 + 1e-6):
        raise ValueError("bounded node features must lie in [0, 1]")
    return EncodedObservations(node_features=node_features)


def encode_global_state(
    base_state: np.ndarray,
    target_tx_ratio: float,
    update_probability: float,
    n_nodes: int,
) -> np.ndarray:
    """Encode five raw simulator statistics into the eight-dimensional V3 critic state."""
    state = np.asarray(base_state, dtype=np.float32).reshape(-1)
    if state.size != BASE_GLOBAL_STATE_DIM:
        raise ValueError("base global state must contain five simulator statistics")
    if np.any(~np.isfinite(state[:4])) or np.any(state[:4] < 0.0):
        raise ValueError("VAoI statistics must be finite and nonnegative")
    last_tx_ratio = _validate_probability("last_tx_ratio", float(state[4]))
    if int(n_nodes) <= 0:
        raise ValueError("n_nodes must be positive")
    update_probability = _validate_probability("update_probability", update_probability)
    target_tx_ratio = _validate_probability("target_tx_ratio", target_tx_ratio, True)
    return np.asarray([
        np.log1p(state[0]), np.log1p(state[1]), np.log1p(state[2]), np.log1p(state[3]),
        last_tx_ratio, np.log1p(int(n_nodes)), update_probability, target_tx_ratio,
    ], dtype=np.float32)
