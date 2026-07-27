"""Test Stage-0 local curvature baselines and realized-rate matching."""

from types import MappingProxyType

import numpy as np
import yaml

from curvature_gossip.experiments import run_experiment
from curvature_gossip.policies import CurvatureFixedAlphaPolicy
from curvature_gossip.simulator import NodeObservation


def _observation(bottlenecks):
    """Build a minimal immutable local observation for scalar-policy tests."""
    neighbors = tuple(range(len(bottlenecks)))
    return NodeObservation(
        node_id=0,
        own_cache_versions=np.zeros(1, dtype=int),
        neighbor_ids=neighbors,
        incident_curvatures=MappingProxyType({node: 0.0 for node in neighbors}),
        incident_bottleneck_importance=MappingProxyType(dict(zip(neighbors, bottlenecks))),
        neighbor_cache_estimates=np.empty((len(neighbors), 1), dtype=int),
        neighbor_estimate_valid=np.zeros(len(neighbors), dtype=bool),
        neighbor_estimate_age=np.full(len(neighbors), -1, dtype=int),
        last_tx_cache_versions=np.zeros(1, dtype=int),
        has_transmitted=False,
        time_since_last_tx=0,
        transmitted_previous_slot=False,
        previous_interference_power=0.0,
        previous_interference_valid=False,
        congestion_ewma=0.0,
        consecutive_tx_attempts=0,
        broadcast_debt=0.0,
        broadcast_limit=1.0,
    )


def test_alpha_override_zero_is_exact_uniform_baseline():
    """Stage-0 alpha override must ignore all local curvature values exactly."""
    policy = CurvatureFixedAlphaPolicy(
        target_tx_ratio=0.1, alpha=4.0, curvature_center=0.5, alpha_override=0.0
    )
    assert policy.transmission_probability(_observation([0.0, 0.2])) == 0.1
    assert policy.transmission_probability(_observation([1.0])) == 0.1


def test_fixed_alpha_probability_increases_with_local_bottleneck_score():
    """A positive fixed alpha must be monotone in the local AF3 score."""
    policy = CurvatureFixedAlphaPolicy(target_tx_ratio=0.1, alpha=2.0, curvature_center=0.0)
    assert policy.transmission_probability(_observation([0.1])) < policy.transmission_probability(_observation([0.8]))


def test_matched_rate_random_uses_reference_realized_rate(tmp_path):
    """The comparison policy must use the reference's observed, not target, rate."""
    config = {
        "experiment": {
            "id": "stage0_test", "master_seed": 7, "slots": 12,
            "topology_seeds": [0], "channel_seeds": [0], "update_seeds": [0],
        },
        "topology": {"type": "random_geometric", "params": {
            "n_nodes": 12, "area_width_m": 100.0, "area_height_m": 100.0,
            "communication_radius_m": 75.0, "min_degree": 2, "max_attempts": 1000,
        }},
        "source": {"update_probability": 0.05},
        "channel": {"rayleigh_fading": False, "shadowing_std_db": 0.0, "sinr_threshold_db": 3.0},
        "curvature": {"method": "distributed_af3", "normalization": "local_degree_bound"},
        "constraints": {"target_tx_ratio": 0.1, "per_node_cap_multiplier": 1.5},
        "policies": [
            {"name": "curvature_fixed_alpha", "type": "curvature_fixed_alpha", "params": {
                "target_tx_ratio": 0.1, "alpha": 1.0,
            }},
            {"name": "matched_rate_random", "type": "matched_rate_random", "params": {
                "reference_policy": "curvature_fixed_alpha",
            }},
        ],
        "output": {"root": str(tmp_path / "results"), "save_per_slot": False, "make_plots": False},
    }
    config_path = tmp_path / "stage0.yaml"
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    result = run_experiment(str(config_path), progress=False)
    summaries = {summary["policy"]: summary for summary in result.run_summaries}
    reference = summaries["curvature_fixed_alpha"]
    matched = summaries["matched_rate_random"]
    assert matched["matched_rate_reference_policy"] == "curvature_fixed_alpha"
    assert matched["matched_rate_probability"] == reference["actual_tx_ratio"]
    assert matched["mean_policy_probability"] == reference["actual_tx_ratio"]
    for summary in summaries.values():
        assert {"target_tx_ratio", "mean_policy_probability", "actual_tx_ratio", "mean_VAoI"} <= set(summary)
