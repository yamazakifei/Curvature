"""Verify strictly local Stage-2 MPNN feature encoding and directed edges."""

import numpy as np

from curvature_gossip.learning.features import encode_stage2_mpnn_observations
from curvature_gossip.simulator.observations import NodeObservation


def _observation(node_id, neighbors, estimates, valid, ages, curvatures=None, cache_size=3):
    curvatures = dict(curvatures or {})
    if cache_size == 3:
        own_cache_versions = np.asarray([3, 1, 2])
        last_tx_cache_versions = np.asarray([2, 1, 1])
    else:
        own_cache_versions = np.arange(cache_size, dtype=np.int64) + 1
        last_tx_cache_versions = np.arange(cache_size, dtype=np.int64)
    return NodeObservation(
        node_id=node_id, own_cache_versions=own_cache_versions, neighbor_ids=tuple(neighbors),
        incident_curvatures={item: float(curvatures.get(item, 0.0)) for item in neighbors},
        incident_bottleneck_importance={item: 0.5 for item in neighbors},
        neighbor_cache_estimates=np.asarray(estimates, dtype=np.int64),
        neighbor_estimate_valid=np.asarray(valid, dtype=bool), neighbor_estimate_age=np.asarray(ages, dtype=np.int64),
        last_tx_cache_versions=last_tx_cache_versions, has_transmitted=False, time_since_last_tx=2,
        transmitted_previous_slot=False, previous_interference_power=0.0, previous_interference_valid=False,
        congestion_ewma=0.0, consecutive_tx_attempts=0, broadcast_debt=0.0, broadcast_limit=1.0,
    )


def test_mpnn_encoder_uses_receiver_cache_estimates_in_directed_edges():
    first = _observation(10, [20], [[2, 1, 3]], [True], [2], {20: -2.0})
    second = _observation(20, [10], [[0, 1, 2]], [False], [-1], {10: -0.25})
    encoded = encode_stage2_mpnn_observations(
        [first, second], 0.1, 0.2, use_curvature_edge_feature=True, bmax=1.5,
    )
    assert encoded.node_context.shape == (2, 5)
    assert encoded.edge_index.dtype == np.int32
    assert np.array_equal(encoded.edge_index, np.asarray([[1, 0], [0, 1]], dtype=np.int32))
    assert np.allclose(encoded.edge_features[0], [1.0 / 3.0, np.exp(-0.1), 1.0])
    assert np.allclose(encoded.edge_features[1], [0.0, 0.0, 0.25 / 1.5])


def test_mpnn_encoder_uses_normalized_version_gap_without_changing_edge_width():
    """V3.4 replaces only the first edge column and keeps V3.3 raw-bmax curvature."""
    first = _observation(10, [20], [[1, 0, 2]], [True], [2], {20: -2.0})
    second = _observation(20, [10], [[0, 1, 2]], [False], [-1], {10: -0.25})
    encoded = encode_stage2_mpnn_observations(
        [first, second], 0.1, 0.2, use_curvature_edge_feature=True,
        bmax=20.0, edge_normalization="raw_bmax",
        edge_freshness_feature="normalized_version_gap",
    )
    expected_gap = np.mean(1.0 - np.exp(-np.asarray([2.0, 1.0, 0.0]) / 4.0))
    assert encoded.edge_features.shape == (2, 3)
    assert np.allclose(encoded.edge_features[0], [expected_gap, np.exp(-0.1), 2.0 / 20.0])
    assert np.allclose(encoded.edge_features[1], [0.0, 0.0, 0.25 / 20.0])


def test_mpnn_encoder_retains_gain_and_adds_normalized_version_gap():
    """V3.5 keeps the legacy gain fraction beside the normalized version gap."""
    first = _observation(10, [20], [[1, 0, 2]], [True], [2], {20: -2.0})
    second = _observation(20, [10], [[0, 1, 2]], [False], [-1], {10: -0.25})
    encoded = encode_stage2_mpnn_observations(
        [first, second], 0.1, 0.2, use_curvature_edge_feature=True,
        bmax=20.0, edge_normalization="raw_bmax",
        edge_freshness_feature="normalized_version_gap_with_gain",
    )
    expected_gap = np.mean(1.0 - np.exp(-np.asarray([2.0, 1.0, 0.0]) / 4.0))
    assert encoded.edge_features.shape == (2, 4)
    assert np.allclose(
        encoded.edge_features[0], [2.0 / 3.0, expected_gap, np.exp(-0.1), 2.0 / 20.0]
    )
    assert np.allclose(encoded.edge_features[1], [0.0, 0.0, 0.0, 0.25 / 20.0])


def test_mpnn_encoder_supports_local_degree_bound_curvature_edges():
    """Stage-2 local-degree normalization must use both endpoints' local degrees."""
    node_ids = list(range(5))
    observations = []
    for node_id in node_ids:
        neighbors = [other for other in node_ids if other != node_id]
        observations.append(_observation(
            node_id, neighbors, [[1, 2, 3, 4, 5]] * len(neighbors),
            [True] * len(neighbors), [0] * len(neighbors),
            {neighbor: (-8.0 if neighbor == 1 else -1.0) for neighbor in neighbors},
            cache_size=5,
        ))

    encoded = encode_stage2_mpnn_observations(
        observations, 0.1, 0.2, use_curvature_edge_feature=True,
        bmax=20.0, edge_normalization="local_degree_bound",
    )

    # For edge 1 -> 0, deg(0)=deg(1)=4, so the AF3 bound is 4+4-4=4.
    assert np.isclose(encoded.edge_features[0, 2], 1.0)
    assert np.isclose(encoded.edge_features[1, 2], 0.25)


def test_mpnn_encoder_adds_node_curvature_score_and_raw_af3_minimum():
    first = _observation(10, [20], [[2, 1, 3]], [True], [2], {20: -2.0})
    second = _observation(20, [10], [[0, 1, 2]], [False], [-1], {10: -0.25})
    encoded = encode_stage2_mpnn_observations(
        [first, second], 0.1, 0.2, use_curvature_edge_feature=True,
        bmax=4.0, use_node_curvature_score=True,
        use_raw_af3_min_edge_curvature=True,
    )
    assert encoded.node_context.shape == (2, 7)
    assert np.allclose(encoded.node_context[:, -2:], [[0.5, -2.0], [0.0625, -0.25]])


def test_mpnn_node_curvature_score_supports_local_degree_bound():
    first = _observation(10, [20], [[2, 1, 3]], [True], [2], {20: -2.0})
    second = _observation(20, [10], [[0, 1, 2]], [False], [-1], {10: -0.25})
    encoded = encode_stage2_mpnn_observations(
        [first, second], 0.1, 0.2, bmax=20.0,
        use_node_curvature_score=True, node_curvature_normalization="local_degree_bound",
    )
    # Both endpoints have degree 1, so the degree bound is max(1, 1+1-4)=1.
    assert np.allclose(encoded.node_context[:, -1], [1.0, 0.25])


def test_mpnn_encoder_disables_curvature_edge_feature_for_no_curvature_ablation():
    first = _observation(10, [20], [[2, 1, 3]], [True], [2], {20: -2.0})
    second = _observation(20, [10], [[0, 1, 2]], [False], [-1], {10: -0.25})
    encoded = encode_stage2_mpnn_observations(
        [first, second], 0.1, 0.2, use_curvature_edge_feature=False, bmax=1.5,
    )
    assert np.allclose(encoded.edge_features[:, 2], 0.0)


def test_mpnn_no_curvature_encoder_does_not_read_curvature_fields():
    first = _observation(10, [20], [[2, 1, 3]], [True], [2], {20: -999.0})
    second = _observation(20, [10], [[0, 1, 2]], [False], [-1], {10: 999.0})
    encoded = encode_stage2_mpnn_observations(
        [first, second], 0.1, 0.2, use_curvature_edge_feature=True,
        use_curvature=False,
    )
    assert np.allclose(encoded.curvature_scores, 0.0)
    assert encoded.edge_features.shape == (2, 3)
    assert np.allclose(encoded.edge_features[:, 2], 0.0)


def test_mpnn_encoder_rejects_neighbors_missing_from_batch():
    lone = _observation(10, [20], [[2, 1, 3]], [True], [2])
    try:
        encode_stage2_mpnn_observations([lone], 0.1, 0.2)
        assert False, "missing neighbor observations must be rejected"
    except ValueError as error:
        assert "neighbor" in str(error)
