"""Run deterministic fixed-scenario validation for the CTDE actor.

Validation constructs fresh simulators with random streams in a dedicated label
namespace.  It only calls actor inference and therefore cannot alter PPO
parameters, optimizer state, or training random-number generators.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from ..channel import ChannelParameters, PropagationModel
from ..curvature import bottleneck_importance
from ..experiments.runner import _curvature_provider
from ..policies.random_policy import UniformRandomPolicy
from ..random_streams import make_rng
from ..simulator import GossipSimulator, SimulationParameters
from ..topology import get_topology_generator
from .features import (
    encode_observations, encode_stage1_observations, encode_stage2_observations,
    encode_stage2_mpnn_observations,
)


@dataclass(frozen=True)
class ValidationScenario:
    """All exogenous and policy-action seeds for one fixed validation scenario."""

    scenario_id: str
    topology_seed: int
    source_update_seed: int
    channel_shadowing_seed: int
    channel_fading_seed: int
    policy_action_seed: int
    slots: int
    n_nodes: int
    update_probability: float
    target_tx_ratio: float


def _require_probability(name: str, value: Any, strict_lower: bool = False) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0 or (strict_lower and value == 0.0):
        raise ValueError("{} must be in {}".format(name, "(0, 1]" if strict_lower else "[0, 1]"))
    return value


def validation_scenarios(raw: Mapping) -> Tuple[ValidationScenario, ...]:
    """Parse the configuration-owned validation set without consulting training cases."""
    section = raw.get("validation", {})
    entries = section.get("scenarios", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("validation.scenarios must be a nonempty list")
    scenarios = []
    required = (
        "topology_seed", "source_update_seed", "channel_shadowing_seed",
        "channel_fading_seed", "policy_action_seed", "slots", "n_nodes",
        "update_probability", "target_tx_ratio",
    )
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or any(name not in entry for name in required):
            raise ValueError("each validation scenario must define all fixed seeds and environment fields")
        slots = int(entry["slots"])
        n_nodes = int(entry["n_nodes"])
        if slots < 1 or n_nodes < 2:
            raise ValueError("validation scenario slots must be positive and n_nodes at least two")
        scenarios.append(ValidationScenario(
            scenario_id=str(entry.get("id", index)),
            topology_seed=int(entry["topology_seed"]),
            source_update_seed=int(entry["source_update_seed"]),
            channel_shadowing_seed=int(entry["channel_shadowing_seed"]),
            channel_fading_seed=int(entry["channel_fading_seed"]),
            policy_action_seed=int(entry["policy_action_seed"]),
            slots=slots,
            n_nodes=n_nodes,
            update_probability=_require_probability("validation update_probability", entry["update_probability"]),
            target_tx_ratio=_require_probability("validation target_tx_ratio", entry["target_tx_ratio"], True),
        ))
    return tuple(scenarios)


def probability_statistics(probability_steps: Sequence[np.ndarray]) -> Dict[str, float]:
    """Summarize actor probabilities over nodes and time without mixing the two variances."""
    matrix = np.asarray(tuple(probability_steps), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("probability_steps must form a nonempty [slots, nodes] matrix")
    flat = matrix.reshape(-1)
    return {
        "action_prob_mean": float(np.mean(flat)),
        "action_prob_std": float(np.std(flat)),
        "action_prob_min": float(np.min(flat)),
        "action_prob_p10": float(np.percentile(flat, 10)),
        "action_prob_p50": float(np.percentile(flat, 50)),
        "action_prob_p90": float(np.percentile(flat, 90)),
        "action_prob_max": float(np.max(flat)),
        "action_prob_node_std_mean": float(np.mean(np.std(matrix, axis=1))),
        "action_prob_global_std": float(np.std(flat)),
        "action_prob_per_node_mean_std": float(np.std(np.mean(matrix, axis=0))),
    }


def _scenario_topology_params(raw: Mapping, n_nodes: int) -> Dict[str, Any]:
    params = dict(raw["topology"].get("params", {}))
    params["n_nodes"] = int(n_nodes)
    if "cluster_sizes" in params:
        params["cluster_sizes"] = [n_nodes // 2, n_nodes - n_nodes // 2]
    return params


def _build_simulator(raw: Mapping, scenario: ValidationScenario, policy_name: str, probability: float):
    """Build an independent but exogenously paired simulator for one policy."""
    master_seed = int(raw["experiment"].get("master_seed", 0))
    generator = get_topology_generator(raw["topology"]["type"])
    topology = generator.generate(
        make_rng(master_seed, "fixed_validation", "topology", scenario.topology_seed),
        _scenario_topology_params(raw, scenario.n_nodes),
    )
    curvature_config = raw.get("curvature", {})
    curvature = _curvature_provider(curvature_config).compute(topology)
    importance = bottleneck_importance(
        topology, curvature, curvature_config.get("normalization", "local_degree_bound")
    )
    propagation = PropagationModel(
        topology.positions,
        ChannelParameters.from_mapping(raw.get("channel", {})),
        make_rng(master_seed, "fixed_validation", "shadowing", scenario.channel_shadowing_seed),
    )
    constraints = raw.get("constraints", {})
    observation = raw.get("observation", {})
    parameters = SimulationParameters(
        slots=scenario.slots,
        update_probability=scenario.update_probability,
        target_tx_ratio=scenario.target_tx_ratio,
        per_node_cap_multiplier=float(constraints.get("per_node_cap_multiplier", 1.5)),
        congestion_ewma_beta=float(observation.get("congestion_ewma_beta", 0.8)),
        congestion_feature_scale=float(observation.get("congestion_feature_scale", 5.0)),
    )
    return GossipSimulator(
        topology, curvature, importance, propagation, UniformRandomPolicy(probability), parameters,
        make_rng(master_seed, "fixed_validation", "source_updates", scenario.source_update_seed),
        make_rng(master_seed, "fixed_validation", "fading", scenario.channel_fading_seed),
        make_rng(master_seed, "fixed_validation", "actions", policy_name, scenario.policy_action_seed),
    )


def _run_neural_scenario(
    model, raw: Mapping, scenario: ValidationScenario, policy_name: str = "nn", stage1_only: bool = False,
) -> Tuple[Mapping, Dict[str, float], np.ndarray]:
    """Sample the frozen Actor or its curvature base while recording local probabilities."""
    training = raw.get("training", {})
    simulator = _build_simulator(raw, scenario, policy_name, 0.0)
    action_rng = simulator.policy_rng
    probability_steps = []
    for slot in range(scenario.slots):
        observations = simulator.begin_step(slot)
        if model.actor_stage == 1:
            encoded = encode_stage1_observations(observations, scenario.target_tx_ratio)
        elif model.actor_stage == 2:
            residual = raw.get("actor", {}).get("residual", {})
            encoder = (encode_stage2_mpnn_observations
                       if residual.get("architecture", "mlp") == "mpnn"
                       else encode_stage2_observations)
            encoded = encoder(
                observations, scenario.target_tx_ratio, scenario.update_probability,
                float(training.get("consecutive_tx_scale", 3.0)),
                float(training.get("neighbor_confidence_time_constant", 20.0)),
                float(raw.get("observation", {}).get("congestion_feature_scale", 5.0)),
                bool(residual.get("include_scenario_context", False)),
            )
        else:
            encoded = encode_observations(
                observations, scenario.target_tx_ratio, scenario.update_probability,
                float(training.get("consecutive_tx_scale", 3.0)),
                float(training.get("neighbor_confidence_time_constant", 20.0)),
                float(raw.get("observation", {}).get("congestion_feature_scale", 5.0)),
            )
        probabilities = np.asarray(
            model.predict_stage1_probabilities(encoded) if stage1_only else model.predict_probabilities(encoded),
            dtype=np.float32,
        )
        if probabilities.shape != (scenario.n_nodes,) or not np.isfinite(probabilities).all():
            raise ValueError("Actor evaluation must return finite [N] probabilities")
        actions = action_rng.random(scenario.n_nodes) < probabilities
        simulator.complete_step(actions)
        probability_steps.append(probabilities)
    # The manual loop already completed the requested slots; use the same result fields as run().
    summary = simulator.metrics.summary()
    summary.update(simulator.tracker.summary())
    matrix = np.asarray(probability_steps, dtype=np.float32)
    return summary, probability_statistics(matrix), matrix


def _run_random_scenario(raw: Mapping, scenario: ValidationScenario, policy_name: str, probability: float) -> Tuple[Mapping, Dict[str, float]]:
    """Evaluate a fixed-probability random policy with its own fixed action stream."""
    simulator = _build_simulator(raw, scenario, policy_name, probability)
    result = simulator.run()
    constant_probabilities = np.full((scenario.slots, scenario.n_nodes), probability, dtype=np.float32)
    return result.summary, probability_statistics(constant_probabilities)


def _paired_mean_ci(values: Iterable[float]) -> Tuple[float, float, float]:
    values = np.asarray(tuple(values), dtype=float)
    if not values.size:
        raise ValueError("paired values must not be empty")
    mean = float(np.mean(values))
    half_width = 0.0 if values.size == 1 else float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size))
    return mean, mean - half_width, mean + half_width


def evaluate_fixed_validation(model, raw: Mapping, checkpoint_episode: int, checkpoint_label: str):
    """Compare the frozen NN Actor to paired fixed- and matched-rate random baselines."""
    before = model.variable_snapshot()
    scenario_rows: List[Dict[str, Any]] = []
    nn_probability_matrices = []
    deltas_matched, deltas_fixed = [], []
    for scenario in validation_scenarios(raw):
        nn_summary, nn_stats, nn_matrix = _run_neural_scenario(model, raw, scenario)
        stage1_summary = stage1_stats = stage1_matrix = None
        if model.actor_stage in (1, 2):
            stage1_summary, stage1_stats, stage1_matrix = _run_neural_scenario(
                # Reuse NN's Bernoulli stream so this is a genuinely paired action comparison.
                model, raw, scenario, policy_name="nn", stage1_only=True
            )
        matched_probability = float(np.mean(nn_matrix))
        fixed_summary, fixed_stats = _run_random_scenario(
            raw, scenario, "fixed_random", scenario.target_tx_ratio
        )
        matched_summary, matched_stats = _run_random_scenario(
            raw, scenario, "matched_random", matched_probability
        )
        nn_probability_matrices.append(nn_matrix)
        delta_matched = float(nn_summary["mean_VAoI"] - matched_summary["mean_VAoI"])
        delta_fixed = float(nn_summary["mean_VAoI"] - fixed_summary["mean_VAoI"])
        deltas_matched.append(delta_matched)
        deltas_fixed.append(delta_fixed)
        policies = [
            ("nn", nn_summary, nn_stats, matched_probability),
        ]
        if stage1_summary is not None:
            policies.append(("stage1_only", stage1_summary, stage1_stats, float(np.mean(stage1_matrix))))
        policies.extend([
            ("matched_random", matched_summary, matched_stats, matched_probability),
            ("fixed_random", fixed_summary, fixed_stats, scenario.target_tx_ratio),
        ])
        for policy_name, summary, stats, reference_probability in policies:
            row = {
                "checkpoint_episode": int(checkpoint_episode),
                "scenario_id": scenario.scenario_id,
                "policy": policy_name,
                "n_nodes": scenario.n_nodes,
                "slots": scenario.slots,
                "update_probability": scenario.update_probability,
                "target_tx_ratio": scenario.target_tx_ratio,
                "matched_rate_probability": matched_probability,
                "policy_probability": float(reference_probability),
                "mean_VAoI": float(summary["mean_VAoI"]),
                "actual_tx_ratio": float(summary["avg_tx_per_slot"] / scenario.n_nodes),
            }
            row.update(stats)
            scenario_rows.append(row)
    after = model.variable_snapshot()
    if len(before) != len(after) or any(not np.array_equal(old, new) for old, new in zip(before, after)):
        raise RuntimeError("fixed validation unexpectedly changed model or optimizer variables")

    nn_rows = [row for row in scenario_rows if row["policy"] == "nn"]
    stage1_rows = [row for row in scenario_rows if row["policy"] == "stage1_only"]
    fixed_rows = [row for row in scenario_rows if row["policy"] == "fixed_random"]
    matched_rows = [row for row in scenario_rows if row["policy"] == "matched_random"]
    # All configured V3 validation scenarios use the same N; concatenate them for global q statistics.
    all_probabilities = np.concatenate(nn_probability_matrices, axis=0)
    nn_stats = probability_statistics(all_probabilities)
    matched_mean, matched_low, matched_high = _paired_mean_ci(deltas_matched)
    fixed_mean, fixed_low, fixed_high = _paired_mean_ci(deltas_fixed)
    summary_row: Dict[str, Any] = {
        "checkpoint_episode": int(checkpoint_episode),
        "n_scenarios": len(nn_rows),
        "nn_mean_VAoI": float(np.mean([row["mean_VAoI"] for row in nn_rows])),
    }
    if stage1_rows:
        summary_row.update({
            "stage1_only_mean_VAoI": float(np.mean([row["mean_VAoI"] for row in stage1_rows])),
        })
    summary_row.update({
        "matched_random_mean_VAoI": float(np.mean([row["mean_VAoI"] for row in matched_rows])),
        "fixed_random_mean_VAoI": float(np.mean([row["mean_VAoI"] for row in fixed_rows])),
        "nn_mean_action_probability": float(np.mean([row["action_prob_mean"] for row in nn_rows])),
    })
    if stage1_rows:
        summary_row.update({
            "stage1_only_mean_action_probability": float(np.mean([row["action_prob_mean"] for row in stage1_rows])),
        })
    summary_row.update({
        "matched_random_mean_action_probability": float(np.mean([row["action_prob_mean"] for row in matched_rows])),
        "fixed_random_mean_action_probability": float(np.mean([row["action_prob_mean"] for row in fixed_rows])),
        "nn_actual_tx_ratio": float(np.mean([row["actual_tx_ratio"] for row in nn_rows])),
    })
    if stage1_rows:
        summary_row.update({
            "stage1_only_actual_tx_ratio": float(np.mean([row["actual_tx_ratio"] for row in stage1_rows])),
        })
    summary_row.update({
        "matched_random_actual_tx_ratio": float(np.mean([row["actual_tx_ratio"] for row in matched_rows])),
        "fixed_random_actual_tx_ratio": float(np.mean([row["actual_tx_ratio"] for row in fixed_rows])),
        "delta_vaoi_matched_mean": matched_mean,
        "delta_vaoi_matched_ci95_low": matched_low,
        "delta_vaoi_matched_ci95_high": matched_high,
        "delta_vaoi_fixed_mean": fixed_mean,
        "delta_vaoi_fixed_ci95_low": fixed_low,
        "delta_vaoi_fixed_ci95_high": fixed_high,
        "nn_beats_matched_fraction": float(np.mean(np.asarray(deltas_matched) < 0.0)),
    })
    summary_row.update(nn_stats)
    return summary_row, scenario_rows
