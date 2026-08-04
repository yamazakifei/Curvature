"""Encode strictly local V3 actor inputs and centralized critic inputs.

The shared actor receives only data exposed by :class:`NodeObservation`; the
centralized critic is used only during training.  The node-conditioned critic
encoder added here deliberately keeps its exact-state and channel features
out of the actor observation path.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

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


@dataclass(frozen=True)
class Stage2MPNNEncodedObservations:
    """Strictly local Stage-2 MPNN inputs, including directed cache-estimate edges."""

    curvature_scores: np.ndarray
    target_tx_ratios: np.ndarray
    node_context: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray


STAGE1_FEATURE_NAMES = ("incident_bottleneck_max",)
STAGE2_CONTEXT_FEATURE_NAMES = (
    "normalized_degree", "time_since_last_tx", "consecutive_tx_attempts",
    "congestion_ewma", "self_information_increment", "neighbor_freshness_mean",
    "neighbor_freshness_max", "neighbor_confidence_mean",
)
STAGE2_SCENARIO_CONTEXT_FEATURE_NAMES = ("log_network_size", "update_probability", "target_tx_ratio")
STAGE2_CONTEXT_INDICES = (0, 1, 2, 3, 6, 7, 8, 10)
STAGE2_MPNN_NODE_FEATURE_NAMES = (
    "normalized_degree", "time_since_last_tx", "consecutive_tx_attempts",
    "congestion_ewma", "self_information_increment",
)
STAGE2_MPNN_EDGE_FEATURE_NAMES = (
    "neighbor_freshness_gain", "neighbor_estimate_confidence", "clipped_bottleneck_score",
)
STAGE2_MPNN_NO_CURVATURE_EDGE_FEATURE_NAMES = (
    "neighbor_freshness_gain", "neighbor_estimate_confidence", "disabled_zero_placeholder",
)
STAGE2_MPNN_NODE_CONTEXT_INDICES = (0, 1, 2, 3, 6)

# Feature groups are explicit so checkpoint metadata can explain every critic
# input and permutation tests can verify that no node identity is encoded.
NODE_CRITIC_DYNAMIC_GLOBAL_FEATURE_NAMES = (
    "global_mean_vaoi", "global_std_vaoi", "global_tail5_mean_vaoi", "last_tx_ratio",
)
NODE_CRITIC_SCENARIO_FEATURE_NAMES = (
    "log_network_size", "update_probability", "target_tx_ratio",
)
NODE_CRITIC_LOCAL_FEATURE_NAMES = (
    "normalized_degree", "time_since_last_tx", "consecutive_tx_attempts",
    "congestion_ewma", "incident_bottleneck_max", "incident_bottleneck_mean",
    "self_information_increment_fraction", "neighbor_confidence_mean",
)
NODE_CRITIC_NO_CURVATURE_LOCAL_FEATURE_NAMES = (
    "normalized_degree", "time_since_last_tx", "consecutive_tx_attempts",
    "congestion_ewma", "self_information_increment_fraction", "neighbor_confidence_mean",
)
NODE_CRITIC_EXACT_FEATURE_NAMES = (
    "receiver_vaoi_mean", "receiver_vaoi_tail", "source_vaoi_mean",
    "source_vaoi_tail", "exact_innovation_fraction_mean",
    "exact_innovation_fraction_max", "exact_innovation_magnitude_mean",
    "exact_bottleneck_weighted_innovation",
)
NODE_CRITIC_NO_CURVATURE_EXACT_FEATURE_NAMES = (
    "receiver_vaoi_mean", "receiver_vaoi_tail", "source_vaoi_mean",
    "source_vaoi_tail", "exact_innovation_fraction_mean",
    "exact_innovation_fraction_max", "exact_innovation_magnitude_mean",
)
NODE_CRITIC_CHANNEL_FEATURE_NAMES = ("mean_link_margin", "weak_link_margin")


@dataclass(frozen=True)
class EncodedNodeCriticInputs:
    """Training-only node critic inputs with shape ``[N, D]``."""

    critic_inputs: np.ndarray
    feature_names: Tuple[str, ...]


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


def _local_non_curvature_features(
    observation: NodeObservation,
    confidence_time_constant: float,
) -> Tuple[float, float, float]:
    """Compute freshness gains and confidence without reading curvature fields."""
    degree = len(observation.neighbor_ids)
    estimates = np.asarray(observation.neighbor_cache_estimates, dtype=np.int64)
    valid = np.asarray(observation.neighbor_estimate_valid, dtype=bool)
    ages = np.asarray(observation.neighbor_estimate_age, dtype=np.int64)
    own = np.asarray(observation.own_cache_versions, dtype=np.int64)
    if estimates.shape != (degree, own.size) or valid.shape != (degree,) or ages.shape != (degree,):
        raise ValueError("neighbor estimate arrays must align with neighbor_ids and cache size")
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
    confidence_mean = float(np.mean(confidence)) if degree else 0.0
    return weighted_gain, max_gain, confidence_mean


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
    use_curvature: bool = True,
) -> np.ndarray:
    """Encode Stage-2 context, optionally using only non-curvature local state."""
    observations = tuple(observations)
    target_tx_ratio = _validate_probability("target_tx_ratio", target_tx_ratio, True)
    update_probability = _validate_probability("update_probability", update_probability)
    if use_curvature:
        encoded = encode_observations(
            observations, target_tx_ratio, update_probability, consecutive_tx_scale,
            neighbor_confidence_time_constant, congestion_feature_scale,
        ).node_features
        context = encoded[:, STAGE2_CONTEXT_INDICES]
        if include_scenario_context:
            context = np.concatenate([context, encoded[:, (11, 12, 13)]], axis=1)
    else:
        if not observations:
            raise ValueError("at least one node observation is required")
        n_nodes = int(np.asarray(observations[0].own_cache_versions).size)
        if n_nodes <= 0:
            raise ValueError("own_cache_versions must not be empty")
        if not np.isfinite(consecutive_tx_scale) or consecutive_tx_scale <= 0.0:
            raise ValueError("consecutive_tx_scale must be positive")
        if not np.isfinite(neighbor_confidence_time_constant) or neighbor_confidence_time_constant <= 0.0:
            raise ValueError("neighbor_confidence_time_constant must be positive")
        if not np.isfinite(congestion_feature_scale) or congestion_feature_scale <= 0.0:
            raise ValueError("congestion_feature_scale must be positive")
        context = np.zeros((len(observations), 8), dtype=np.float32)
        for row, observation in enumerate(observations):
            own = np.asarray(observation.own_cache_versions, dtype=np.int64)
            last_tx = np.asarray(observation.last_tx_cache_versions, dtype=np.int64)
            if own.shape != (n_nodes,) or last_tx.shape != (n_nodes,):
                raise ValueError("own and last transmitted cache versions must have common shape [N]")
            fresh_mean, fresh_max, confidence_mean = _local_non_curvature_features(
                observation, float(neighbor_confidence_time_constant)
            )
            context[row] = np.asarray([
                np.log1p(len(observation.neighbor_ids)) / np.log1p(n_nodes),
                np.tanh(target_tx_ratio * max(0, int(observation.time_since_last_tx))),
                np.tanh(max(0, int(observation.consecutive_tx_attempts)) / consecutive_tx_scale),
                np.tanh(max(0.0, float(observation.congestion_ewma)) / congestion_feature_scale),
                float(np.mean(own > last_tx)), fresh_mean, fresh_max, confidence_mean,
            ], dtype=np.float32)
        if include_scenario_context:
            context = np.concatenate([
                context,
                np.tile(np.asarray([np.log1p(n_nodes), update_probability, target_tx_ratio], dtype=np.float32),
                         (len(observations), 1)),
            ], axis=1)
    expected_width = 11 if include_scenario_context else 8
    if context.shape != (len(observations), expected_width) or not np.isfinite(context).all():
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
    use_curvature: bool = True,
) -> Stage2EncodedObservations:
    """Combine Stage-2 inputs, with an explicit curvature-free encoding mode."""
    observations = tuple(observations)
    target_tx_ratio = _validate_probability("target_tx_ratio", target_tx_ratio, True)
    return Stage2EncodedObservations(
        curvature_scores=(encode_curvature_score(observations) if use_curvature
                          else np.zeros((len(observations), 1), dtype=np.float32)),
        target_tx_ratios=np.full(len(observations), target_tx_ratio, dtype=np.float32),
        residual_context=encode_stage2_context(
            observations, target_tx_ratio, update_probability, consecutive_tx_scale,
            neighbor_confidence_time_constant, congestion_feature_scale,
            include_scenario_context,
            use_curvature,
        ),
    )


def encode_stage2_mpnn_observations(
    observations: Sequence[NodeObservation],
    target_tx_ratio: float,
    update_probability: float,
    consecutive_tx_scale: float = 3.0,
    neighbor_confidence_time_constant: float = 20.0,
    congestion_feature_scale: float = 5.0,
    include_scenario_context: bool = False,
    use_curvature_edge_feature: bool = False,
    bmax: float = 1.0,
    use_curvature: bool = True,
) -> Stage2MPNNEncodedObservations:
    """Encode local node state, cache estimates, and an optional clipped curvature edge feature.

    Each directed ``j -> i`` edge uses only state cached by receiver ``i``;
    sender current state is intentionally never read.  When enabled, the third
    edge feature is ``min(max(-kappa_ij, 0), bmax) / bmax``.  In no-curvature
    mode the third edge column remains a fixed zero placeholder.
    """
    observations = tuple(observations)
    if not observations:
        raise ValueError("at least one node observation is required")
    target_tx_ratio = _validate_probability("target_tx_ratio", target_tx_ratio, True)
    if not np.isfinite(neighbor_confidence_time_constant) or neighbor_confidence_time_constant <= 0.0:
        raise ValueError("neighbor_confidence_time_constant must be positive")
    bmax = float(bmax)
    if not np.isfinite(bmax) or bmax <= 0.0:
        raise ValueError("bmax must be finite and positive")

    # Keep the fixed node width while avoiding all curvature reads in ablation mode.
    if use_curvature:
        legacy = encode_observations(
            observations, target_tx_ratio, update_probability, consecutive_tx_scale,
            neighbor_confidence_time_constant, congestion_feature_scale,
        ).node_features
        node_context = legacy[:, STAGE2_MPNN_NODE_CONTEXT_INDICES]
        scenario_context = legacy[:, (11, 12, 13)]
    else:
        context = encode_stage2_context(
            observations, target_tx_ratio, update_probability, consecutive_tx_scale,
            neighbor_confidence_time_constant, congestion_feature_scale,
            include_scenario_context=False, use_curvature=False,
        )
        node_context = context[:, :5]
        n_nodes = int(np.asarray(observations[0].own_cache_versions).size)
        scenario_context = np.tile(
            np.asarray([np.log1p(n_nodes), update_probability, target_tx_ratio], dtype=np.float32),
            (len(observations), 1),
        )
    if include_scenario_context:
        node_context = np.concatenate([node_context, scenario_context], axis=1)
    expected_width = 8 if include_scenario_context else 5
    if node_context.shape != (len(observations), expected_width) or not np.isfinite(node_context).all():
        raise ValueError("MPNN node context must be finite with the configured fixed width")

    node_ids = [int(item.node_id) for item in observations]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("each node observation must have a unique node_id")
    node_rows = {node_id: row for row, node_id in enumerate(node_ids)}
    senders, receivers, features = [], [], []
    for receiver_row, observation in enumerate(observations):
        degree = len(observation.neighbor_ids)
        estimates = np.asarray(observation.neighbor_cache_estimates, dtype=np.int64)
        valid = np.asarray(observation.neighbor_estimate_valid, dtype=bool)
        ages = np.asarray(observation.neighbor_estimate_age, dtype=np.int64)
        own = np.asarray(observation.own_cache_versions, dtype=np.int64)
        if estimates.shape != (degree, own.size) or valid.shape != (degree,) or ages.shape != (degree,):
            raise ValueError("neighbor estimate arrays must align with neighbor_ids and cache size")
        if np.any(valid & (ages < 0)) or np.any(~valid & (ages != -1)):
            raise ValueError("valid estimates require nonnegative ages and invalid ones age -1")
        for neighbor_index, sender_id in enumerate(observation.neighbor_ids):
            if int(sender_id) not in node_rows:
                raise ValueError("every observed neighbor must be present in the observation batch")
            gain = (float(np.mean(own > estimates[neighbor_index])) if valid[neighbor_index] else 0.0)
            confidence = (float(np.exp(-ages[neighbor_index] / neighbor_confidence_time_constant))
                          if valid[neighbor_index] else 0.0)
            if use_curvature and use_curvature_edge_feature:
                curvature = float(observation.incident_curvatures[int(sender_id)])
                raw_bottleneck = max(-curvature, 0.0)
                clipped_bottleneck = min(raw_bottleneck, bmax) / bmax
            else:
                clipped_bottleneck = 0.0
            # Keep all edge inputs on the same [0, 1] scale for the message MLP.
            feature = np.asarray([gain, confidence, clipped_bottleneck], dtype=np.float32)
            if not np.isfinite(feature).all() or np.any(feature < 0.0) or np.any(feature > 1.0):
                raise ValueError("MPNN edge features must be finite and in [0, 1]")
            senders.append(node_rows[int(sender_id)])
            receivers.append(receiver_row)
            features.append(feature)
    edge_index = (np.asarray([senders, receivers], dtype=np.int32)
                  if senders else np.empty((2, 0), dtype=np.int32))
    edge_features = (np.asarray(features, dtype=np.float32).reshape(-1, 3)
                     if features else np.empty((0, 3), dtype=np.float32))
    if edge_index.shape[1] != edge_features.shape[0]:
        raise ValueError("MPNN edge index and feature counts must agree")
    return Stage2MPNNEncodedObservations(
        curvature_scores=(encode_curvature_score(observations) if use_curvature
                          else np.zeros((len(observations), 1), dtype=np.float32)),
        target_tx_ratios=np.full(len(observations), target_tx_ratio, dtype=np.float32),
        node_context=node_context.astype(np.float32, copy=False),
        edge_index=edge_index,
        edge_features=edge_features,
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


def node_critic_feature_names(
    include_scenario_context: bool = True,
    include_exact_node_features: bool = True,
    include_channel_features: bool = True,
    include_curvature_features: bool = True,
) -> Tuple[str, ...]:
    """Return the stable concatenation order for node-conditioned Critic inputs."""
    names = list(NODE_CRITIC_DYNAMIC_GLOBAL_FEATURE_NAMES)
    if include_scenario_context:
        names.extend(NODE_CRITIC_SCENARIO_FEATURE_NAMES)
    names.extend(
        NODE_CRITIC_LOCAL_FEATURE_NAMES if include_curvature_features
        else NODE_CRITIC_NO_CURVATURE_LOCAL_FEATURE_NAMES
    )
    if include_exact_node_features:
        names.extend(
            NODE_CRITIC_EXACT_FEATURE_NAMES if include_curvature_features
            else NODE_CRITIC_NO_CURVATURE_EXACT_FEATURE_NAMES
        )
    if include_channel_features:
        names.extend(NODE_CRITIC_CHANNEL_FEATURE_NAMES)
    return tuple(names)


def _tail5mean(values: np.ndarray) -> float:
    """Return the mean of the largest five percent, retaining at least one item."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    count = max(1, int(np.ceil(0.05 * values.size)))
    return float(np.mean(np.partition(values, values.size - count)[-count:]))


def _log_age_stats(version_age: np.ndarray) -> Tuple[float, float, float]:
    """Summarize off-diagonal version age without max/p95 redundancy."""
    ages = np.asarray(version_age, dtype=np.float32)
    if ages.ndim != 2 or ages.shape[0] != ages.shape[1] or ages.shape[0] < 2:
        raise ValueError("version_age must be a square matrix with at least two nodes")
    off_diagonal = ages[~np.eye(ages.shape[0], dtype=bool)]
    if np.any(~np.isfinite(off_diagonal)) or np.any(off_diagonal < 0.0):
        raise ValueError("version_age must be finite and nonnegative")
    return (
        float(np.log1p(np.mean(off_diagonal))),
        float(np.log1p(np.std(off_diagonal))),
        float(np.log1p(_tail5mean(off_diagonal))),
    )


def encode_node_critic_inputs(
    observations: Sequence[NodeObservation],
    version_age: np.ndarray,
    last_tx_ratio: float,
    target_tx_ratio: float,
    update_probability: float,
    n_nodes: int,
    mean_rx_power_mw: np.ndarray = None,
    noise_power_mw: float = None,
    sinr_threshold_db: float = None,
    include_scenario_context: bool = True,
    include_exact_node_features: bool = True,
    include_channel_features: bool = True,
    include_curvature_features: bool = True,
    consecutive_tx_scale: float = 3.0,
    neighbor_confidence_time_constant: float = 20.0,
    congestion_feature_scale: float = 5.0,
) -> EncodedNodeCriticInputs:
    """Build permutation-equivariant, training-only ``[N, D]`` Critic inputs.

    ``version_age`` and ``mean_rx_power_mw`` are centralized training data.
    The local feature block is recomputed from immutable observations and is
    never passed to the distributed actor through ``NodeObservation``.
    """
    observations = tuple(observations)
    if not observations:
        raise ValueError("at least one node observation is required")
    n_nodes = int(n_nodes)
    if n_nodes != len(observations) or n_nodes < 2:
        raise ValueError("n_nodes must match observations and be at least two")
    target_tx_ratio = _validate_probability("target_tx_ratio", target_tx_ratio, True)
    update_probability = _validate_probability("update_probability", update_probability)
    last_tx_ratio = _validate_probability("last_tx_ratio", last_tx_ratio)
    ages = np.asarray(version_age, dtype=np.float32)
    if ages.shape != (n_nodes, n_nodes):
        raise ValueError("version_age must have shape [N, N]")
    node_ids = [int(item.node_id) for item in observations]
    if sorted(node_ids) != list(range(n_nodes)):
        raise ValueError("node critic observations must contain node ids 0..N-1")
    observations_by_id = {int(item.node_id): item for item in observations}

    # Dynamic global context is replicated for every node; this is not a node ID.
    mean_age, std_age, tail_age = _log_age_stats(ages)
    dynamic = np.asarray([mean_age, std_age, tail_age, last_tx_ratio], dtype=np.float32)
    scenario = np.asarray(
        [np.log1p(n_nodes), update_probability, target_tx_ratio], dtype=np.float32
    )
    if include_curvature_features:
        local_actor_features = encode_observations(
            observations, target_tx_ratio, update_probability, consecutive_tx_scale,
            neighbor_confidence_time_constant, congestion_feature_scale,
        ).node_features.astype(np.float32, copy=False)
        # Keep only Actor-visible state that is not a duplicate of exact innovation.
        local = local_actor_features[:, (0, 1, 2, 3, 4, 5, 6, 10)]
    else:
        # Use the independent no-curvature path; do not read bottleneck fields.
        local_context = encode_stage2_context(
            observations, target_tx_ratio, update_probability, consecutive_tx_scale,
            neighbor_confidence_time_constant, congestion_feature_scale,
            include_scenario_context=False, use_curvature=False,
        )
        local = local_context[:, (0, 1, 2, 3, 4, 7)]

    exact_width = 8 if include_curvature_features else 7
    exact = np.zeros((n_nodes, exact_width), dtype=np.float32)
    for row, observation in enumerate(observations):
        node = int(observation.node_id)
        row_values = np.delete(ages[node], node)
        column_values = np.delete(ages[:, node], node)
        own = np.asarray(observation.own_cache_versions, dtype=np.float32)
        innovations = []
        for neighbor in observation.neighbor_ids:
            neighbor = int(neighbor)
            neighbor_own = np.asarray(observations_by_id[neighbor].own_cache_versions, dtype=np.float32)
            difference = np.maximum(own - neighbor_own, 0.0)
            innovation = [float(np.mean(difference > 0.0)), float(np.mean(difference))]
            if include_curvature_features:
                innovation.append(float(observation.incident_bottleneck_importance[neighbor]))
            innovations.append(tuple(innovation))
        degree = len(innovations)
        innovation_fractions = [item[0] for item in innovations]
        innovation_magnitudes = [item[1] for item in innovations]
        innovation_fraction = float(np.mean(innovation_fractions)) if degree else 0.0
        innovation_fraction_max = float(np.max(innovation_fractions)) if degree else 0.0
        innovation_magnitude = float(np.mean(innovation_magnitudes)) if degree else 0.0
        exact_values = [
            np.log1p(np.mean(row_values)), np.log1p(_tail5mean(row_values)),
            np.log1p(np.mean(column_values)), np.log1p(_tail5mean(column_values)),
            innovation_fraction, innovation_fraction_max,
            np.log1p(innovation_magnitude),
        ]
        if include_curvature_features:
            bottleneck_weights = np.asarray([item[2] for item in innovations], dtype=np.float32)
            exact_values.append(
                float(np.dot(bottleneck_weights, innovation_fractions)
                      / (float(np.sum(bottleneck_weights)) + 1e-8))
                if degree else 0.0
            )
        exact[row] = np.asarray(exact_values, dtype=np.float32)

    channel = np.zeros((n_nodes, 2), dtype=np.float32)
    if include_channel_features:
        power = np.asarray(mean_rx_power_mw, dtype=np.float64)
        if power.shape != (n_nodes, n_nodes) or noise_power_mw is None or sinr_threshold_db is None:
            raise ValueError("channel features require mean_rx_power_mw and channel parameters")
        if not np.isfinite(power).all() or np.any(power < 0.0) or noise_power_mw <= 0.0:
            raise ValueError("mean receive power and noise must be finite and positive")
        for row, observation in enumerate(observations):
            node = int(observation.node_id)
            margins = []
            for neighbor in observation.neighbor_ids:
                received = max(float(power[node, int(neighbor)]), np.finfo(float).tiny)
                margin = 10.0 * np.log10(received / float(noise_power_mw)) - float(sinr_threshold_db)
                margins.append(margin)
            if margins:
                channel[row] = np.asarray([
                    np.tanh(np.mean(margins) / 10.0), np.tanh(np.percentile(margins, 10) / 10.0)
                ], dtype=np.float32)

    blocks = [np.repeat(dynamic[None, :], n_nodes, axis=0)]
    if include_scenario_context:
        blocks.append(np.repeat(scenario[None, :], n_nodes, axis=0))
    blocks.append(local)
    if include_exact_node_features:
        blocks.append(exact)
    if include_channel_features:
        blocks.append(channel)
    encoded = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    names = node_critic_feature_names(
        include_scenario_context, include_exact_node_features, include_channel_features,
        include_curvature_features,
    )
    if encoded.shape != (n_nodes, len(names)) or not np.isfinite(encoded).all():
        raise ValueError("node critic inputs must be finite and match configured feature names")
    return EncodedNodeCriticInputs(critic_inputs=encoded, feature_names=names)
