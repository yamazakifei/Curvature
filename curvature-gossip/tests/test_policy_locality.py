"""验证策略字段白名单、观测隔离、曲率零权重等价性和零增益基线。"""

import dataclasses

import networkx as nx
import numpy as np
import pytest

from curvature_gossip.curvature import CurvatureResult
from curvature_gossip.policies import CurvatureFreshnessPolicy, FreshnessPolicy
from curvature_gossip.simulator import NodeObservation, ObservationBuilder
from curvature_gossip.state import LocalKnowledge, VersionState
from curvature_gossip.topology.base import Topology


def _observation():
    graph = nx.path_graph(3)
    topology = Topology(np.column_stack((np.arange(3), np.zeros(3))), graph, 1.0, None, {})
    curvature = CurvatureResult("test", {(0, 1): -1.0, (1, 2): 0.0}, np.zeros(3), {})
    state = VersionState(3)
    state.apply_source_updates(0, np.array([False, True, False]))
    knowledge = LocalKnowledge(graph)
    builder = ObservationBuilder(topology, curvature, {(0, 1): 1.0, (1, 2): 0.0})
    return builder.build(1, 0, state, knowledge), state


def test_observation_has_no_global_truth_fields():
    field_names = {field.name for field in dataclasses.fields(NodeObservation)}
    assert "source_versions" not in field_names
    assert "version_age" not in field_names
    assert "global_cache_versions" not in field_names


def test_observation_arrays_do_not_alias_mutable_simulator_state():
    observation, state = _observation()
    assert not np.shares_memory(observation.own_cache_versions, state.cache_versions)
    with pytest.raises(ValueError):
        observation.own_cache_versions[0] = 99


def test_zero_curvature_weight_matches_freshness_probability():
    observation, _ = _observation()
    params = dict(q_min=0.01, q_max=0.5, threshold=1.0, temperature=1.0, age_weight=0.02)
    freshness = FreshnessPolicy(**params)
    curvature = CurvatureFreshnessPolicy(curvature_weight=0.0, **params)
    assert freshness.transmission_probability(observation) == curvature.transmission_probability(observation)


def test_zero_gain_and_age_weight_reduce_to_sigmoid_baseline():
    observation, _ = _observation()
    policy = FreshnessPolicy(q_min=0.0, q_max=1.0, threshold=0.0, temperature=1.0, age_weight=0.0)
    # 手工清零自身缓存后，所有局部创新增益均为零。
    zero_observation = dataclasses.replace(observation, own_cache_versions=np.zeros(3, dtype=int))
    assert policy.transmission_probability(zero_observation) == 0.5


def test_local_congestion_and_attempt_backoff_reduce_probability():
    observation, _ = _observation()
    params = dict(q_min=0.01, q_max=0.5, threshold=1.0, temperature=1.0, age_weight=0.0)
    baseline = FreshnessPolicy(**params).transmission_probability(observation)
    backed_off = FreshnessPolicy(
        congestion_backoff_weight=1.0,
        attempt_backoff_factor=0.5,
        **params
    ).transmission_probability(dataclasses.replace(
        observation,
        congestion_ewma=np.log(3.0),
        consecutive_tx_attempts=2,
    ))
    assert 0.01 <= backed_off < baseline


def test_default_backoff_parameters_preserve_old_probability():
    observation, _ = _observation()
    params = dict(q_min=0.01, q_max=0.5, threshold=1.0, temperature=1.0, age_weight=0.02)
    default_policy = FreshnessPolicy(**params)
    explicit_disabled_policy = FreshnessPolicy(
        congestion_backoff_weight=0.0,
        attempt_backoff_factor=1.0,
        **params
    )
    assert (
        default_policy.transmission_probability(observation)
        == explicit_disabled_policy.transmission_probability(observation)
    )


def test_log_gain_compression_scales_by_log_node_count():
    observation, _ = _observation()
    policy = FreshnessPolicy(
        gain_compression="log1p",
        gain_normalization="log_node_count",
    )
    # 三节点观测中的单位版本差被映射为 log(2) / log(4)。
    assert np.allclose(policy.edge_gains(observation), np.log(2.0) / np.log(4.0))


def test_curvature_diagnostics_decompose_final_probability():
    observation, _ = _observation()
    policy = CurvatureFreshnessPolicy(
        q_min=0.01,
        q_max=0.5,
        threshold=0.5,
        temperature=0.2,
        age_weight=0.0,
        gain_compression="log1p",
        gain_normalization="log_node_count",
        congestion_backoff_weight=0.5,
        attempt_backoff_factor=0.8,
        curvature_weight=1.0,
    )
    diagnostic = policy.probability_diagnostics(dataclasses.replace(
        observation,
        congestion_ewma=0.4,
        consecutive_tx_attempts=1,
    ))
    assert diagnostic["orc_score"] > diagnostic["fresh_score"]
    assert 0.0 < diagnostic["backoff_factor"] < 1.0
    assert diagnostic["final_probability"] == pytest.approx(
        np.clip(
            diagnostic["base_probability"] * diagnostic["backoff_factor"],
            policy.q_min,
            policy.q_max,
        )
    )
