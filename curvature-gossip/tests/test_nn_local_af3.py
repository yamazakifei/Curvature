"""Test V3 CTDE local features, critic encoding, and bounded budget learning terms."""

import dataclasses

import networkx as nx
import numpy as np

from curvature_gossip.curvature import DistributedAF3Curvature, GlobalAF3Curvature, bottleneck_importance
from curvature_gossip.learning.ctde_ppo import CTDEPPO
from curvature_gossip.learning.features import (
    TARGET_TX_RATIO_INDEX, UPDATE_PROBABILITY_INDEX, encode_global_state,
    encode_observations,
)
from curvature_gossip.learning.trainer import _budget_advantages, _update_multiplier, _update_tx_ratio_ema
from curvature_gossip.simulator import ObservationBuilder
from curvature_gossip.state import LocalKnowledge, VersionState
from curvature_gossip.topology.base import Topology


def _topology():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
    positions = np.column_stack((np.arange(4), np.zeros(4)))
    return Topology(positions, graph, 1.0, None, {})


def _observation():
    topology = _topology()
    curvature = DistributedAF3Curvature().compute(topology)
    importance = bottleneck_importance(topology, curvature, "local_degree_bound")
    state = VersionState(4)
    state.apply_source_updates(0, np.array([True, False, True, False]))
    knowledge = LocalKnowledge(topology.graph)
    return ObservationBuilder(topology, curvature, importance).build(2, 1, state, knowledge)


def test_distributed_af3_matches_formula_without_global_normalization():
    topology = _topology()
    distributed = DistributedAF3Curvature().compute(topology)
    global_result = GlobalAF3Curvature().compute(topology)
    assert distributed.edge_values == global_result.edge_values
    assert distributed.metadata["control_message_count"] == 2 * topology.graph.number_of_edges()
    assert all(0.0 <= value <= 1.0 for value in bottleneck_importance(topology, distributed, "local_degree_bound").values())


def test_v3_actor_features_use_local_frozen_snapshots_and_confidence():
    observation = _observation()
    # One valid neighbor has two newer entries; the other remains explicitly unknown.
    estimates = observation.neighbor_cache_estimates.copy()
    valid = np.array([True, False, False])
    ages = np.array([1, -1, -1])
    estimates[0] = np.array([0, 0, 0, 0])
    last_tx = np.array([0, 0, 0, 0])
    updated = dataclasses.replace(
        observation, neighbor_cache_estimates=estimates, neighbor_estimate_valid=valid,
        neighbor_estimate_age=ages, last_tx_cache_versions=last_tx,
    )
    encoded = encode_observations((updated,), 0.1, 0.05, 3.0, 20.0).node_features[0]
    assert encoded.shape == (14,)
    assert encoded[0] == np.float32(np.log1p(3) / np.log1p(4))
    assert encoded[1] == np.float32(np.tanh(0.1 * 2))
    assert encoded[6] == np.float32(0.25)
    assert encoded[7] == np.float32(0.25)
    assert encoded[8] == np.float32(0.25)
    assert encoded[10] == np.float32(np.exp(-1.0 / 20.0) / 3.0)
    assert encoded[11] == np.float32(np.log1p(4))
    assert encoded[UPDATE_PROBABILITY_INDEX] == np.float32(0.05)
    assert encoded[TARGET_TX_RATIO_INDEX] == np.float32(0.1)
    assert np.isfinite(encoded).all()


def test_neural_actor_is_invariant_to_neighbor_order():
    observation = _observation()
    order = np.arange(len(observation.neighbor_ids))[::-1]
    reordered = dataclasses.replace(
        observation,
        neighbor_ids=tuple(observation.neighbor_ids[index] for index in order),
        neighbor_cache_estimates=observation.neighbor_cache_estimates[order],
        neighbor_estimate_valid=observation.neighbor_estimate_valid[order],
        neighbor_estimate_age=observation.neighbor_estimate_age[order],
    )
    assert np.array_equal(encode_observations((observation,), 0.1, 0.05).node_features, encode_observations((reordered,), 0.1, 0.05).node_features)


def test_node_actor_starts_at_budget_and_has_no_fixed_qmax():
    encoded = encode_observations((_observation(),), 0.1, 0.08)
    model = CTDEPPO(seed=123)
    try:
        assert np.allclose(model.predict_probabilities(encoded), [0.1], atol=1e-6)
        with model.graph.as_default():
            residual_bias = next(variable for variable in model.tf.trainable_variables() if variable.name == "actor/transmission_residual_logit/bias:0")
            model.session.run(model.tf.assign(residual_bias, [4.0]))
        assert model.predict_probabilities(encoded)[0] > 0.2
    finally:
        model.close()


def test_v3_critic_encoding_order_and_budget_advantage():
    critic = encode_global_state(np.array([1.0, 3.0, 2.0, 2.5, 0.25]), 0.1, 0.05, 4)
    assert critic.dtype == np.float32 and critic.shape == (8,)
    assert np.allclose(critic, [np.log1p(1), np.log1p(3), np.log1p(2), np.log1p(2.5), 0.25, np.log1p(4), 0.05, 0.1])
    budget = _budget_advantages(np.array([1.0, 0.0]), np.array([0.1, 0.1]), 0.3, 0.15)
    assert np.allclose(budget, [-0.135, 0.015])
    assert np.max(np.abs(budget)) <= 0.15


def test_v3_multiplier_ema_deadzone_clip_and_case_independence_primitives():
    ema = _update_tx_ratio_ema(0.1, 0.3, 0.9)
    assert np.isclose(ema, 0.12)
    multiplier, relative_error, deadzone = _update_multiplier(0.1, 0.101, 0.1, 0.01, 0.15, 0.05, 1.0)
    assert multiplier == 0.1 and deadzone == 0.0 and relative_error > 0.0
    multiplier, _, deadzone = _update_multiplier(0.14, 1.0, 0.1, 1.0, 0.15, 0.05, 1.0)
    assert multiplier == 0.15 and deadzone == 1.0
