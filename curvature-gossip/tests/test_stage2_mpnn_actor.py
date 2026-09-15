"""Verify the Stage-2 MPNN graph, V3.5 curvature attention, and batching."""

import numpy as np
import pytest

from curvature_gossip.learning.ctde_ppo import CTDEPPO
from curvature_gossip.learning.features import Stage2MPNNEncodedObservations
from curvature_gossip.learning.trainer import _stage1_actor_config


def _actor(
    no_curvature=False, aggregation="mean_max", attention_uniform_mix=0.0,
    edge_freshness_feature="binary_fraction",
):
    return {
        "stage": 2,
        "curvature": {"enabled": not no_curvature, "center": 0.4, "alpha_init": 1.5, "alpha_override": 0.0 if no_curvature else 1.5},
        "residual": {
            "enabled": True, "architecture": "mpnn", "hidden_dims": [64, 64], "activation": "relu",
            "delta_max": 1.0, "zero_init_output": True, "use_stage1_reference": not no_curvature,
            "detach_stage1_reference": True, "freeze_stage1": True,
            "mpnn": {"message_hidden_dims": [32], "message_dim": 16, "update_hidden_dims": [64, 64],
                     "aggregation": aggregation, "condition_message_on_receiver": True,
                     "use_sender_node_features": False,
                     "use_curvature_edge_feature": not no_curvature, "bmax": 1.0,
                     "edge_freshness_feature": edge_freshness_feature,
                     "attention_temperature": 0.5,
                     "attention_uniform_mix": attention_uniform_mix},
        },
    }


def _encoded(offset=0, edge_width=3):
    edge_features = (
        [[1.0, 0.3, 0.8], [1.0, 0.4, 0.7]]
        if edge_width == 3 else
        [[0.6, 0.9, 0.3, 0.8], [0.7, 0.8, 0.4, 0.7]]
    )
    return Stage2MPNNEncodedObservations(
        curvature_scores=np.asarray([[0.1], [0.8]], dtype=np.float32),
        target_tx_ratios=np.full(2, 0.1, dtype=np.float32),
        node_context=np.asarray([[0.1] * 5, [0.2] * 5], dtype=np.float32),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int32),
        edge_features=np.asarray(edge_features, dtype=np.float32),
    )


def _attention_encoded():
    """Build a graph with two incoming edges for a hand-checkable softmax."""
    return Stage2MPNNEncodedObservations(
        curvature_scores=np.asarray([[0.1], [0.8], [0.4]], dtype=np.float32),
        target_tx_ratios=np.full(3, 0.1, dtype=np.float32),
        node_context=np.asarray([[0.1] * 5, [0.2] * 5, [0.3] * 5], dtype=np.float32),
        edge_index=np.asarray([[1, 2, 0], [0, 0, 1]], dtype=np.int32),
        edge_features=np.asarray([
            [1.0, 0.3, 0.0], [1.0, 0.4, 1.0], [1.0, 0.5, 0.5],
        ], dtype=np.float32),
    )


def test_mpnn_zero_output_keeps_base_and_uses_receiver_conditioned_messages():
    model = CTDEPPO(seed=41, actor_config=_actor())
    try:
        diagnostics = model.stage2_diagnostics(_encoded())
        assert diagnostics["edge_messages"].shape == (2, 16)
        assert diagnostics["aggregated_messages"].shape == (2, 32)
        assert diagnostics["residual_input"].shape == (2, 38)
        assert np.allclose(diagnostics["delta"], 0.0, atol=1e-7)
        assert np.allclose(diagnostics["q_final"], diagnostics["q_base"], atol=1e-7)
    finally:
        model.close()


def test_mpnn_batch_offsets_make_disjoint_graphs_and_no_curvature_has_37_decoder_inputs():
    config = _stage1_actor_config({"actor": _actor(True)})
    assert config["architecture_version"] == "no_curvature_stage2_mpnn_v1"
    assert config["curvature_enabled"] is False
    assert config["use_curvature_edge_feature"] is False
    assert config["edge_input_feature_names"][-1] == "disabled_zero_placeholder"
    assert config["decoder_input_dim"] == 37
    model = CTDEPPO(seed=43, actor_config=_actor(True))
    try:
        batch = model.actor_batch_inputs([_encoded(), _encoded()])
        assert np.array_equal(batch["edge_index"], np.asarray([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int32))
        assert "curvature_scores" not in batch
        diagnostics = model.stage2_diagnostics(_encoded())
        assert diagnostics["residual_input"].shape == (2, 37)
        assert np.allclose(diagnostics["q_base"], 0.1, atol=1e-7)
        altered = _encoded()
        altered.edge_features[:, 2] = 0.0
        assert np.allclose(
            model.stage2_diagnostics(_encoded())["residual_input"],
            model.stage2_diagnostics(altered)["residual_input"],
        )
    finally:
        model.close()


def test_mpnn_handles_an_empty_edge_tensor_and_restores_its_checkpoint(tmp_path):
    encoded = Stage2MPNNEncodedObservations(
        curvature_scores=np.asarray([[0.3]], dtype=np.float32), target_tx_ratios=np.asarray([0.1], dtype=np.float32),
        node_context=np.zeros((1, 5), dtype=np.float32), edge_index=np.empty((2, 0), dtype=np.int32),
        edge_features=np.empty((0, 3), dtype=np.float32),
    )
    checkpoint = tmp_path / "mpnn" / "model"
    model = CTDEPPO(seed=47, actor_config=_actor())
    try:
        before = model.predict_probabilities(encoded)
        model.save(str(checkpoint))
    finally:
        model.close()
    restored = CTDEPPO(seed=53, actor_config=_actor())
    try:
        restored.restore(str(checkpoint))
        diagnostics = restored.stage2_diagnostics(encoded)
        assert diagnostics["aggregated_messages"].shape == (1, 32)
        assert np.allclose(diagnostics["aggregated_messages"], 0.0)
        assert np.allclose(restored.predict_probabilities(encoded), before)
    finally:
        restored.close()


def test_v35_curvature_attention_weights_are_receiver_local_and_normalized():
    model = CTDEPPO(seed=59, actor_config=_actor(aggregation="curvature_attention_max"))
    try:
        diagnostics = model.stage2_diagnostics(
            _attention_encoded(), include_mpnn_messages=False, include_mpnn_attention=True
        )
        weights = diagnostics["attention_weights"]
        expected_first_receiver = np.exp(np.asarray([0.0, 1.0], dtype=np.float32) / 0.5)
        expected_first_receiver /= np.sum(expected_first_receiver)
        assert np.allclose(weights[:2], expected_first_receiver, atol=1e-6)
        assert np.isclose(np.sum(weights[:2]), 1.0, atol=1e-6)
        assert np.isclose(weights[2], 1.0, atol=1e-6)
    finally:
        model.close()


def test_v35_attention_uniform_mix_blends_with_uniform_weights():
    model = CTDEPPO(
        seed=61, actor_config=_actor(
            aggregation="curvature_attention_max", attention_uniform_mix=0.25
        )
    )
    try:
        weights = model.stage2_diagnostics(
            _attention_encoded(), include_mpnn_messages=False, include_mpnn_attention=True
        )["attention_weights"]
        softmax_weights = np.exp(np.asarray([0.0, 1.0], dtype=np.float32) / 0.5)
        softmax_weights /= np.sum(softmax_weights)
        expected = 0.75 * softmax_weights + 0.25 * 0.5
        assert np.allclose(weights[:2], expected, atol=1e-6)
    finally:
        model.close()


def test_v35_attention_handles_empty_edge_tensor():
    encoded = Stage2MPNNEncodedObservations(
        curvature_scores=np.asarray([[0.3]], dtype=np.float32),
        target_tx_ratios=np.asarray([0.1], dtype=np.float32),
        node_context=np.zeros((1, 5), dtype=np.float32),
        edge_index=np.empty((2, 0), dtype=np.int32),
        edge_features=np.empty((0, 3), dtype=np.float32),
    )
    model = CTDEPPO(seed=62, actor_config=_actor(aggregation="curvature_attention_max"))
    try:
        diagnostics = model.stage2_diagnostics(
            encoded, include_mpnn_messages=True, include_mpnn_attention=True
        )
        assert diagnostics["attention_weights"].shape == (0,)
        assert np.allclose(diagnostics["attention_entropy"], 0.0)
        assert np.allclose(diagnostics["attention_effective_degree"], 0.0)
        assert np.allclose(diagnostics["attention_max_weight"], 0.0)
        assert np.allclose(diagnostics["aggregated_messages"], 0.0)
    finally:
        model.close()


def test_v35_attention_metadata_and_no_curvature_guard():
    config = _stage1_actor_config({"actor": _actor(aggregation="curvature_attention_max")})
    assert config["architecture_version"] == "curvature_stage2_mpnn_v3_5"
    assert config["aggregation"] == "curvature_attention_max"
    assert config["attention_temperature"] == 0.5
    assert config["attention_uniform_mix"] == 0.0
    with pytest.raises(ValueError, match="requires use_curvature_edge_feature=true"):
        _stage1_actor_config({
            "actor": _actor(no_curvature=True, aggregation="curvature_attention_max")
        })


def test_v35_edge_gain_schema_keeps_version_gap_and_legacy_gain():
    actor = _actor(
        aggregation="curvature_attention_max",
        edge_freshness_feature="normalized_version_gap_with_gain",
    )
    config = _stage1_actor_config({"actor": actor})
    assert config["architecture_version"] == "curvature_stage2_mpnn_v3_5_edge_gain"
    assert config["edge_input_dim"] == 4
    assert config["message_input_dim"] == 9
    assert config["edge_input_feature_names"] == [
        "neighbor_freshness_gain", "neighbor_version_gap_normalized",
        "neighbor_estimate_confidence", "clipped_bottleneck_score",
    ]
    model = CTDEPPO(seed=63, actor_config=actor)
    try:
        encoded = _encoded(edge_width=4)
        batch = model.actor_batch_inputs([encoded])
        assert batch["edge_features"].shape == (2, 4)
        assert model.stage2_diagnostics(encoded)["residual_input"].shape == (2, 38)
    finally:
        model.close()
