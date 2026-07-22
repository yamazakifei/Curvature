"""验证局部 AF3、节点广播债务和神经集合编码的关键分布式约束。"""

import dataclasses

import networkx as nx
import numpy as np

from curvature_gossip.curvature import (
    DistributedAF3Curvature, GlobalAF3Curvature, bottleneck_importance,
)
from curvature_gossip.learning.ctde_ppo import CTDEPPO
from curvature_gossip.learning.features import (
    UPDATE_PROBABILITY_INDEX,
    encode_observations,
)
from curvature_gossip.learning.trainer import _node_constraint_advantages
from curvature_gossip.simulator import ObservationBuilder
from curvature_gossip.state import LocalKnowledge, VersionState
from curvature_gossip.topology.base import Topology


def _topology():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
    positions = np.column_stack((np.arange(4), np.zeros(4)))
    return Topology(positions, graph, 1.0, None, {})


def test_distributed_af3_matches_formula_without_global_normalization():
    topology = _topology()
    distributed = DistributedAF3Curvature().compute(topology)
    global_result = GlobalAF3Curvature().compute(topology)
    assert distributed.edge_values == global_result.edge_values
    assert distributed.metadata["control_message_count"] == 2 * topology.graph.number_of_edges()
    importance = bottleneck_importance(topology, distributed, "local_degree_bound")
    assert all(0.0 <= value <= 1.0 for value in importance.values())


def test_node_broadcast_debt_is_updated_from_own_actions_only():
    knowledge = LocalKnowledge(nx.path_graph(2), broadcast_limit=0.25)
    knowledge.update_slot_history(
        slot=0,
        transmitted=np.array([True, False]),
        interference_power=np.array([1.0, 1.0]),
        noise_power=1.0,
        congestion_ewma_alpha=0.8,
    )
    assert np.allclose(knowledge.broadcast_debt, [0.75, 0.0])
    knowledge.update_slot_history(
        slot=1,
        transmitted=np.array([False, False]),
        interference_power=np.array([1.0, 1.0]),
        noise_power=1.0,
        congestion_ewma_alpha=0.8,
    )
    assert np.allclose(knowledge.broadcast_debt, [0.5, 0.0])


def test_neural_actor_is_invariant_to_neighbor_order():
    topology = _topology()
    curvature = DistributedAF3Curvature().compute(topology)
    importance = bottleneck_importance(topology, curvature, "local_degree_bound")
    state = VersionState(4)
    state.apply_source_updates(0, np.array([True, False, True, False]))
    knowledge = LocalKnowledge(topology.graph, broadcast_limit=0.15)
    observation = ObservationBuilder(topology, curvature, importance).build(
        2, 1, state, knowledge
    )
    order = np.arange(len(observation.neighbor_ids))[::-1]
    reordered = dataclasses.replace(
        observation,
        neighbor_ids=tuple(observation.neighbor_ids[index] for index in order),
        neighbor_cache_estimates=observation.neighbor_cache_estimates[order],
        neighbor_estimate_valid=observation.neighbor_estimate_valid[order],
    )
    first = encode_observations((observation,), 0.1, 0.05)
    second = encode_observations((reordered,), 0.1, 0.05)
    model = CTDEPPO(seed=123)
    try:
        probability_a = model.predict_probabilities(first)
        probability_b = model.predict_probabilities(second)
    finally:
        model.close()
    assert np.allclose(probability_a, probability_b, atol=1e-7)


def test_node_actor_starts_at_budget_and_has_no_fixed_qmax():
    topology = _topology()
    curvature = DistributedAF3Curvature().compute(topology)
    importance = bottleneck_importance(topology, curvature, "local_degree_bound")
    observation = ObservationBuilder(topology, curvature, importance).build(
        0, 1, VersionState(4), LocalKnowledge(topology.graph, broadcast_limit=0.15)
    )
    encoded = encode_observations((observation,), 0.1, 0.08)
    assert encoded.node_features[0, UPDATE_PROBABILITY_INDEX] == np.float32(0.08)

    model = CTDEPPO(seed=123)
    try:
        assert np.allclose(model.predict_probabilities(encoded), [0.1], atol=1e-6)
        with model.graph.as_default():
            residual_bias = next(
                variable
                for variable in model.tf.trainable_variables()
                if variable.name == "actor/transmission_residual_logit/bias:0"
            )
            model.session.run(model.tf.assign(residual_bias, [4.0]))
        assert model.predict_probabilities(encoded)[0] > 0.2
    finally:
        model.close()


def test_node_constraint_advantages_are_bounded_and_action_specific():
    budget, debt = _node_constraint_advantages(
        actions=np.array([1.0, 0.0]),
        old_probabilities=np.array([0.1, 0.1]),
        broadcast_debts=np.array([2.0, 2.0]),
        lagrange=30.0,
        local_debt_penalty=0.2,
        budget_weight=0.15,
        debt_weight=0.05,
        signal_scale=1.0,
    )
    assert budget[0] < 0.0 < budget[1]
    assert debt[0] < 0.0 < debt[1]
    assert np.max(np.abs(budget)) <= 0.15 + 1e-7
    assert np.max(np.abs(debt)) <= 0.05 + 1e-7
