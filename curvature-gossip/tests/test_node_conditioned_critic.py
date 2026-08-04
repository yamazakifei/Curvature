"""Test full and dual no-curvature node-conditioned Critic encoders and GAE."""

from dataclasses import replace
import numpy as np

from curvature_gossip.learning.ctde_ppo import CTDEPPO
from curvature_gossip.learning.features import (
    NODE_CRITIC_LOCAL_FEATURE_NAMES,
    STAGE2_CONTEXT_FEATURE_NAMES,
    encode_node_critic_inputs,
    node_critic_feature_names,
)
from curvature_gossip.learning.trainer import _critic_config, _gae, _node_gae
from curvature_gossip.simulator.observations import NodeObservation


def _observation(node_id, own, bottlenecks):
    """Build a small immutable observation set without exposing centralized state."""
    neighbors = tuple(item for item in range(3) if item != node_id)
    return NodeObservation(
        node_id=node_id,
        own_cache_versions=np.asarray(own, dtype=np.int64),
        neighbor_ids=neighbors,
        incident_curvatures={item: 0.0 for item in neighbors},
        incident_bottleneck_importance=dict(bottlenecks),
        neighbor_cache_estimates=np.zeros((2, 3), dtype=np.int64),
        neighbor_estimate_valid=np.zeros(2, dtype=bool),
        neighbor_estimate_age=np.full(2, -1, dtype=np.int64),
        last_tx_cache_versions=np.zeros(3, dtype=np.int64),
        has_transmitted=False,
        time_since_last_tx=1,
        transmitted_previous_slot=False,
        previous_interference_power=0.0,
        previous_interference_valid=False,
        congestion_ewma=0.0,
        consecutive_tx_attempts=0,
        broadcast_debt=0.0,
        broadcast_limit=1.0,
    )


def _observations():
    return (
        _observation(0, [0, 1, 0], {1: 0.25, 2: 0.75}),
        _observation(1, [0, 0, 0], {0: 0.50, 2: 0.50}),
        _observation(2, [0, 1, 1], {0: 0.75, 1: 0.25}),
    )


def test_node_gae_uses_per_node_values_and_nonzero_bootstrap():
    rewards = np.asarray([1.0, 2.0], dtype=np.float32)
    values = np.asarray([[0.5, 1.0], [0.25, 2.0]], dtype=np.float32)
    advantages, returns = _node_gae(rewards, values, np.asarray([3.0, 4.0]), 1.0, 0.0)
    assert np.allclose(advantages, [[0.75, 2.0], [4.75, 4.0]])
    assert np.allclose(returns, [[1.25, 3.0], [5.0, 6.0]])


def test_node_gae_n1_reduces_to_scalar_gae():
    rewards = np.asarray([1.0, -0.5, 2.0], dtype=np.float32)
    values = np.asarray([0.2, 0.4, -0.1], dtype=np.float32)
    scalar_advantages, scalar_returns = _gae(rewards, values, 0.9, 0.8)
    node_advantages, node_returns = _node_gae(
        rewards, values[:, None], np.asarray([0.0], dtype=np.float32), 0.9, 0.8
    )
    assert np.allclose(node_advantages[:, 0], scalar_advantages)
    assert np.allclose(node_returns[:, 0], scalar_returns)


def test_critic_feature_layout_is_25d_and_deduplicated():
    names = node_critic_feature_names(True, True, True)
    assert len(names) == 25
    assert names[:7] == (
        "global_mean_vaoi", "global_std_vaoi", "global_tail5_mean_vaoi",
        "last_tx_ratio", "log_network_size", "update_probability", "target_tx_ratio",
    )
    assert NODE_CRITIC_LOCAL_FEATURE_NAMES == (
        "normalized_degree", "time_since_last_tx", "consecutive_tx_attempts",
        "congestion_ewma", "incident_bottleneck_max", "incident_bottleneck_mean",
        "self_information_increment_fraction", "neighbor_confidence_mean",
    )
    for excluded in (
        "neighbor_freshness_mean", "neighbor_freshness_max",
        "bottleneck_weighted_neighbor_freshness", "self_information_increment_magnitude",
    ):
        assert excluded not in names
    assert "neighbor_confidence_mean" in names
    for actor_feature in (
        "neighbor_freshness_mean", "neighbor_freshness_max",
        "neighbor_confidence_mean",
    ):
        assert actor_feature in STAGE2_CONTEXT_FEATURE_NAMES
    # The full Actor observation encoder still retains the bottleneck-weighted
    # freshness estimate; it is intentionally absent from the Critic layout.
    assert "bottleneck_weighted_neighbor_freshness" not in NODE_CRITIC_LOCAL_FEATURE_NAMES


def test_exact_innovation_features_and_permutation_equivariance():
    observations = _observations()
    version_age = np.asarray([[0, 1, 2], [3, 0, 4], [5, 6, 0]], dtype=np.float32)
    received_power = np.ones((3, 3), dtype=np.float64)
    np.fill_diagonal(received_power, 0.0)
    encoded = encode_node_critic_inputs(
        observations, version_age, 0.1, 0.1, 0.2, 3,
        mean_rx_power_mw=received_power, noise_power_mw=1.0, sinr_threshold_db=0.0,
    )
    assert encoded.critic_inputs.shape == (3, 25)
    exact_start = 7 + 8
    # Node 0 has one innovative neighbor (fraction 1/3, magnitude 1/3),
    # one non-innovative neighbor, and bottleneck weights .25/.75.
    assert np.isclose(encoded.critic_inputs[0, exact_start + 4], 1.0 / 6.0)
    assert np.isclose(encoded.critic_inputs[0, exact_start + 5], 1.0 / 3.0)
    assert np.isclose(encoded.critic_inputs[0, exact_start + 6], np.log1p(1.0 / 6.0))
    assert np.isclose(encoded.critic_inputs[0, exact_start + 7], 1.0 / 12.0)
    permuted = encode_node_critic_inputs(
        observations[::-1], version_age, 0.1, 0.1, 0.2, 3,
        mean_rx_power_mw=received_power, noise_power_mw=1.0, sinr_threshold_db=0.0,
    )
    assert np.allclose(permuted.critic_inputs, encoded.critic_inputs[::-1])


def test_node_conditioned_model_outputs_one_value_per_node_and_scalar_is_compatible():
    model = CTDEPPO(seed=91, critic_config={"architecture": "node_conditioned"})
    scalar = CTDEPPO(seed=92)
    try:
        assert model.critic_input_dim == 25
        assert model.value(np.zeros((3, 25), dtype=np.float32)).shape == (3,)
        assert scalar.value(np.zeros((2, 8), dtype=np.float32)).shape == (2,)
    finally:
        model.close()
        scalar.close()


def test_no_curvature_critic_layout_is_22d_and_ignores_curvature_values():
    names = node_critic_feature_names(True, True, True, False)
    assert len(names) == 22
    assert "incident_bottleneck_max" not in names
    assert "incident_bottleneck_mean" not in names
    assert "exact_bottleneck_weighted_innovation" not in names
    assert names[-2:] == ("mean_link_margin", "weak_link_margin")

    observations = _observations()
    changed = tuple(
        replace(
            item,
            incident_curvatures={neighbor: 1000.0 for neighbor in item.neighbor_ids},
            incident_bottleneck_importance={neighbor: 1.0 for neighbor in item.neighbor_ids},
        )
        for item in observations
    )
    version_age = np.asarray([[0, 1, 2], [3, 0, 4], [5, 6, 0]], dtype=np.float32)
    received_power = np.ones((3, 3), dtype=np.float64)
    np.fill_diagonal(received_power, 0.0)
    original = encode_node_critic_inputs(
        observations, version_age, 0.1, 0.1, 0.2, 3,
        mean_rx_power_mw=received_power, noise_power_mw=1.0, sinr_threshold_db=0.0,
        include_curvature_features=False,
    )
    altered = encode_node_critic_inputs(
        changed, version_age, 0.1, 0.1, 0.2, 3,
        mean_rx_power_mw=received_power, noise_power_mw=1.0, sinr_threshold_db=0.0,
        include_curvature_features=False,
    )
    assert original.critic_inputs.shape == (3, 22)
    assert np.allclose(original.critic_inputs, altered.critic_inputs)

    model = CTDEPPO(
        seed=94,
        critic_config={"architecture": "node_conditioned", "include_curvature_features": False},
    )
    try:
        assert model.critic_input_dim == 22
        assert model.value(np.zeros((3, 22), dtype=np.float32)).shape == (3,)
    finally:
        model.close()


def test_no_curvature_critic_config_resolves_explicit_22d_schema():
    critic = _critic_config({
        "critic": {
            "architecture": "node_conditioned",
            "include_curvature_features": False,
            "input_dim": 22,
        }
    })
    assert critic["input_dim"] == 22
    assert critic["include_curvature_features"] is False
    assert "incident_bottleneck_max" not in critic["input_feature_names"]


def test_actor_only_restore_excludes_critic_and_optimizer_variables(monkeypatch):
    """Actor-only restore builds a Saver from trainable actor variables only."""
    model = CTDEPPO(seed=93, critic_config={"architecture": "node_conditioned"})
    try:
        with model.graph.as_default():
            trainable = model.tf.get_collection(model.tf.GraphKeys.TRAINABLE_VARIABLES)
        available = {
            variable.name.split(":")[0]: variable.shape.as_list()
            for variable in trainable
        }

        class _Reader:
            def get_variable_to_shape_map(self):
                return available

        captured = {}

        class _Saver:
            def __init__(self, var_list):
                captured["names"] = tuple(var_list)

            def restore(self, session, checkpoint_prefix):
                captured["restored"] = checkpoint_prefix

        monkeypatch.setattr(model.tf.train, "NewCheckpointReader", lambda _: _Reader())
        monkeypatch.setattr(model.tf.train, "Saver", _Saver)
        restored = model.restore_actor_only("actor-only-checkpoint")
        assert restored
        assert all(name.startswith("actor/") for name in captured["names"])
        assert not any(name.startswith("critic/") for name in captured["names"])
        assert not any("Adam" in name for name in captured["names"])
    finally:
        model.close()
