"""Verify strictly local Stage-2 MPNN feature encoding and directed edges."""

import numpy as np

from curvature_gossip.learning.features import encode_stage2_mpnn_observations
from curvature_gossip.simulator.observations import NodeObservation


def _observation(node_id, neighbors, estimates, valid, ages, curvatures=None):
    curvatures = dict(curvatures or {})
    return NodeObservation(
        node_id=node_id, own_cache_versions=np.asarray([3, 1, 2]), neighbor_ids=tuple(neighbors),
        incident_curvatures={item: float(curvatures.get(item, 0.0)) for item in neighbors},
        incident_bottleneck_importance={item: 0.5 for item in neighbors},
        neighbor_cache_estimates=np.asarray(estimates, dtype=np.int64),
        neighbor_estimate_valid=np.asarray(valid, dtype=bool), neighbor_estimate_age=np.asarray(ages, dtype=np.int64),
        last_tx_cache_versions=np.asarray([2, 1, 1]), has_transmitted=False, time_since_last_tx=2,
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
