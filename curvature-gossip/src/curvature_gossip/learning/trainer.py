"""Train the V3 local-actor/centralized-critic PPO policy.

The actor uses a rollout-frozen bounded Lagrange multiplier for its budget
advantage.  The multiplier and transmission-rate EMA update only after each
complete rollout.
"""

import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ..channel import ChannelParameters, PropagationModel
from ..config import load_config
from ..curvature import bottleneck_importance
from ..experiments.runner import _curvature_provider
from ..policies.random_policy import UniformRandomPolicy
from ..random_streams import make_rng
from ..simulator import GossipSimulator, SimulationParameters
from ..topology import get_topology_generator
from .ctde_ppo import CTDEPPO
from .features import encode_global_state, encode_observations


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_simulator(raw: Mapping, episode: int, target_tx_ratio: float, node_count=None, update_probability=None):
    """Create one isolated rollout simulator for a configured training case."""
    experiment = raw["experiment"]
    training = raw.get("training", {})
    master_seed = int(experiment.get("master_seed", 0))
    topology_seed = int(training.get("topology_seed_start", 10000)) + int(episode)
    generator = get_topology_generator(raw["topology"]["type"])
    topology_params = dict(raw["topology"].get("params", {}))
    if node_count is not None:
        n_nodes = int(node_count)
        topology_params["n_nodes"] = n_nodes
        if "cluster_sizes" in topology_params:
            topology_params["cluster_sizes"] = [n_nodes // 2, n_nodes - n_nodes // 2]
    topology = generator.generate(make_rng(master_seed, "nn_topology", topology_seed), topology_params)
    curvature_config = raw.get("curvature", {})
    curvature = _curvature_provider(curvature_config).compute(topology)
    importance = bottleneck_importance(topology, curvature, curvature_config.get("normalization", "local_degree_bound"))
    channel = ChannelParameters.from_mapping(raw.get("channel", {}))
    propagation = PropagationModel(topology.positions, channel, make_rng(master_seed, "nn_shadowing", topology_seed))
    constraints = raw.get("constraints", {})
    rollout_slots = int(training.get("rollout_slots", experiment.get("slots", 200)))
    if update_probability is None:
        update_probability = float(raw.get("source", {}).get("update_probability", 0.05))
    parameters = SimulationParameters(
        slots=rollout_slots, update_probability=update_probability,
        target_tx_ratio=float(target_tx_ratio),
        per_node_cap_multiplier=float(constraints.get("per_node_cap_multiplier", 1.5)),
    )
    simulator = GossipSimulator(
        topology, curvature, importance, propagation, UniformRandomPolicy(0.0), parameters,
        make_rng(master_seed, "nn_updates", topology_seed),
        make_rng(master_seed, "nn_fading", topology_seed),
        make_rng(master_seed, "nn_policy", topology_seed),
    )
    return simulator, curvature


def _gae(rewards, values, gamma: float, gae_lambda: float):
    """Compute finite-horizon GAE with the V3-required zero terminal bootstrap."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    accumulator = 0.0
    for index in range(rewards.size - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < values.size else 0.0
        delta = rewards[index] + gamma * next_value - values[index]
        accumulator = delta + gamma * gae_lambda * accumulator
        advantages[index] = accumulator
    return advantages, advantages + values


def _standardize(values: np.ndarray) -> np.ndarray:
    """Standardize VAoI advantages once over the complete rollout time axis."""
    values = np.asarray(values, dtype=np.float32)
    return ((values - np.mean(values)) / (np.std(values) + 1e-6)).astype(np.float32)


def _budget_advantages(actions, old_probabilities, multiplier: float, multiplier_max: float) -> np.ndarray:
    """Return the bounded V3 linear Lagrangian advantage -clip(mu)*(a-q_old)."""
    actions = np.asarray(actions, dtype=np.float32)
    old_probabilities = np.asarray(old_probabilities, dtype=np.float32)
    if actions.shape != old_probabilities.shape:
        raise ValueError("actions and old_probabilities must have the same shape")
    if not np.all((actions == 0.0) | (actions == 1.0)):
        raise ValueError("actions must contain only 0 or 1")
    if np.any(~np.isfinite(old_probabilities)) or np.any((old_probabilities < 0.0) | (old_probabilities > 1.0)):
        raise ValueError("old_probabilities must lie in [0, 1]")
    if not np.isfinite(multiplier_max) or multiplier_max <= 0.0:
        raise ValueError("multiplier_max must be positive")
    if not np.isfinite(multiplier):
        raise ValueError("multiplier must be finite")
    bounded_multiplier = float(np.clip(multiplier, 0.0, multiplier_max))
    return (-bounded_multiplier * (actions - old_probabilities)).astype(np.float32)


def _update_tx_ratio_ema(previous_ema: float, rollout_tx_ratio: float, beta: float) -> float:
    """Update the rollout-level actual-transmission-ratio EMA."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("tx_ratio_ema_beta must be in [0, 1]")
    if not 0.0 <= previous_ema <= 1.0 or not 0.0 <= rollout_tx_ratio <= 1.0:
        raise ValueError("transmission ratios must be in [0, 1]")
    return float(beta * previous_ema + (1.0 - beta) * rollout_tx_ratio)


def _update_multiplier(multiplier: float, tx_ratio_ema: float, target_tx_ratio: float, learning_rate: float, multiplier_max: float, relative_tolerance: float, error_clip: float, epsilon: float = 1e-8):
    """Apply the V3 relative-error dead-zone and explicitly bounded multiplier update."""
    if not 0.0 < target_tx_ratio <= 1.0 or not 0.0 <= tx_ratio_ema <= 1.0:
        raise ValueError("transmission ratios must be in (0, 1] and [0, 1]")
    if learning_rate < 0.0 or multiplier_max <= 0.0 or relative_tolerance < 0.0 or error_clip <= 0.0:
        raise ValueError("invalid multiplier update hyperparameter")
    relative_error = (tx_ratio_ema - target_tx_ratio) / (target_tx_ratio + epsilon)
    deadzone_error = 0.0 if abs(relative_error) <= relative_tolerance else float(np.clip(relative_error, -error_clip, error_clip))
    next_multiplier = float(np.clip(multiplier + learning_rate * deadzone_error, 0.0, multiplier_max))
    return next_multiplier, float(relative_error), deadzone_error


def _write_history(output_directory: Path, history) -> None:
    with (output_directory / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    with (output_directory / "training_history.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(history), stream, indent=2, sort_keys=True)


def train_ctde(config_path: str):
    """Train and save a V3 checkpoint, metadata, and per-rollout diagnostics."""
    config = load_config(config_path)
    raw = config.raw
    training = raw.get("training", {})
    constraints = raw.get("constraints", {})
    output_root = Path(raw.get("output", {}).get("root", "result_NN"))
    output_directory = output_root / str(raw["experiment"].get("id", "nn_ctde_v3"))
    output_directory.mkdir(parents=True, exist_ok=True)
    with Path(config_path).open("r", encoding="utf-8") as source:
        (output_directory / "training_config.yaml").write_text(source.read(), encoding="utf-8")

    vaoi_reward_scale = float(training.get("vaoi_reward_scale", 0.1))
    consecutive_tx_scale = float(training.get("consecutive_tx_scale", 3.0))
    confidence_time_constant = float(training.get("neighbor_confidence_time_constant", 20.0))
    initial_multiplier = float(training.get("initial_multiplier", 0.0))
    multiplier_learning_rate = float(training.get("multiplier_learning_rate", 0.01))
    multiplier_max = float(training.get("multiplier_max", 0.15))
    tx_ratio_ema_beta = float(training.get("tx_ratio_ema_beta", 0.9))
    relative_tolerance = float(training.get("budget_relative_tolerance", 0.05))
    multiplier_error_clip = float(training.get("multiplier_error_clip", 1.0))
    if vaoi_reward_scale <= 0.0 or consecutive_tx_scale <= 0.0 or confidence_time_constant <= 0.0:
        raise ValueError("V3 reward scale, consecutive scale, and confidence time constant must be positive")
    if not 0.0 <= initial_multiplier <= multiplier_max or multiplier_learning_rate < 0.0:
        raise ValueError("invalid initial multiplier or multiplier learning rate")
    if not 0.0 <= tx_ratio_ema_beta <= 1.0 or relative_tolerance < 0.0 or multiplier_error_clip <= 0.0:
        raise ValueError("invalid V3 EMA or multiplier stability setting")

    seed = int(raw["experiment"].get("master_seed", 0))
    model = CTDEPPO(learning_rate=float(training.get("learning_rate", 3e-4)), clip_ratio=float(training.get("clip_ratio", 0.2)), entropy_coefficient=float(training.get("entropy_coefficient", 0.01)), seed=seed)
    action_rng = np.random.default_rng(seed + 7919)
    targets = [float(value) for value in training.get("target_tx_ratios", [constraints.get("target_tx_ratio", 0.1)])]
    node_counts = list(training.get("node_counts", [None])) or [None]
    update_probabilities = list(training.get("update_probabilities", [None])) or [None]
    if not targets or any(not 0.0 < value <= 1.0 for value in targets):
        raise ValueError("training.target_tx_ratios must contain values in (0, 1]")
    training_cases = [(target, node_count, update_probability) for node_count in node_counts for update_probability in update_probabilities for target in targets]
    constraint_state = {}
    history = []

    try:
        for episode in range(int(training.get("episodes", 20))):
            target, node_count, update_probability = training_cases[episode % len(training_cases)]
            simulator, curvature = _build_simulator(raw, episode, target, node_count, update_probability)
            case_key = (simulator.state.n_nodes, simulator.parameters.update_probability, target)
            if case_key not in constraint_state:
                constraint_state[case_key] = {"multiplier": initial_multiplier, "tx_ratio_ema": target}
            multiplier_used = constraint_state[case_key]["multiplier"]
            encoded_steps, actions_steps, probability_steps, old_log_steps = [], [], [], []
            value_steps, global_states, vaoi_rewards, costs, vaoi_values = [], [], [], [], []
            last_tx_ratio = 0.0
            for slot in range(simulator.parameters.slots):
                observations = simulator.begin_step(slot)
                encoded = encode_observations(observations, target, simulator.parameters.update_probability, consecutive_tx_scale, confidence_time_constant)
                global_state = encode_global_state(simulator.centralized_state(last_tx_ratio), target, simulator.parameters.update_probability, simulator.state.n_nodes)
                value = float(model.value(global_state)[0])
                actions, probabilities, old_log_probabilities = model.act(encoded, action_rng)
                outcome = simulator.complete_step(actions.astype(bool))
                encoded_steps.append(encoded)
                actions_steps.append(actions)
                probability_steps.append(probabilities.astype(np.float32))
                old_log_steps.append(old_log_probabilities)
                value_steps.append(value)
                global_states.append(global_state)
                vaoi_rewards.append(vaoi_reward_scale * outcome.reward)
                costs.append(outcome.transmission_cost)
                vaoi_values.append(outcome.mean_vaoi)
                last_tx_ratio = outcome.transmission_cost

            vaoi_advantages, returns = _gae(vaoi_rewards, value_steps, float(training.get("gamma", 0.99)), float(training.get("gae_lambda", 0.95)))
            normalized_vaoi_advantages = _standardize(vaoi_advantages)
            actor_vaoi_advantages = np.concatenate([np.full(step.node_features.shape[0], normalized_vaoi_advantages[index], dtype=np.float32) for index, step in enumerate(encoded_steps)])
            action_batch = np.concatenate(actions_steps)
            probability_batch = np.concatenate(probability_steps)
            budget_advantages = _budget_advantages(action_batch, probability_batch, multiplier_used, multiplier_max)
            actor_advantages = (actor_vaoi_advantages + budget_advantages).astype(np.float32)
            actor_batch = {"node_features": np.concatenate([step.node_features for step in encoded_steps]), "actions": action_batch, "old_log_probabilities": np.concatenate(old_log_steps), "advantages": actor_advantages}
            critic_batch = {"global_states": np.asarray(global_states, dtype=np.float32), "returns": returns}
            vaoi_batch = dict(actor_batch, advantages=actor_vaoi_advantages)
            budget_batch = dict(actor_batch, advantages=budget_advantages)
            vaoi_gradient_norm = model.policy_gradient_norm(vaoi_batch)
            budget_gradient_norm = model.policy_gradient_norm(budget_batch)
            losses = model.update(actor_batch, critic_batch, epochs=int(training.get("ppo_epochs", 4)))

            average_cost = float(np.mean(costs))
            tx_ratio_ema = _update_tx_ratio_ema(constraint_state[case_key]["tx_ratio_ema"], average_cost, tx_ratio_ema_beta)
            multiplier_next, relative_error, deadzone_error = _update_multiplier(multiplier_used, tx_ratio_ema, target, multiplier_learning_rate, multiplier_max, relative_tolerance, multiplier_error_clip)
            constraint_state[case_key] = {"multiplier": multiplier_next, "tx_ratio_ema": tx_ratio_ema}
            summary = simulator.metrics.summary()
            row = {
                "episode": episode, "mean_VAoI": float(np.mean(vaoi_values)), "avg_tx_ratio": average_cost,
                "tx_ratio_ema": tx_ratio_ema, "target_tx_ratio": target, "budget_error": tx_ratio_ema - target,
                "budget_relative_error": relative_error, "budget_error_after_deadzone": deadzone_error,
                "mean_action_probability": float(np.mean(probability_steps)), "multiplier_used": multiplier_used,
                "multiplier_next": multiplier_next, "multiplier_at_max": bool(np.isclose(multiplier_next, multiplier_max)),
                "actor_loss": losses["actor_loss"], "critic_loss": losses["critic_loss"], "entropy": losses["entropy"],
                "mean_vaoi_advantage": float(np.mean(actor_vaoi_advantages)), "std_vaoi_advantage": float(np.std(actor_vaoi_advantages)), "max_abs_vaoi_advantage": float(np.max(np.abs(actor_vaoi_advantages))),
                "mean_budget_advantage": float(np.mean(budget_advantages)), "std_budget_advantage": float(np.std(budget_advantages)), "max_abs_budget_advantage": float(np.max(np.abs(budget_advantages))),
                "vaoi_policy_gradient_norm": vaoi_gradient_norm, "budget_policy_gradient_norm": budget_gradient_norm,
                "budget_to_vaoi_gradient_ratio": budget_gradient_norm / (vaoi_gradient_norm + 1e-8),
                "n_nodes": simulator.state.n_nodes, "update_probability": simulator.parameters.update_probability,
                "max_node_activity_ratio": summary["max_node_activity_ratio"], "node_cap_violation_fraction": summary["node_cap_violation_fraction"],
                "curvature_control_messages": curvature.metadata.get("control_message_count", 0),
            }
            if not all(np.isfinite(value) for value in row.values() if isinstance(value, (float, np.floating))):
                raise FloatingPointError("V3 training diagnostics must be finite")
            history.append(row)
            _write_history(output_directory, history)
            model.save(str(output_directory / "checkpoints" / "model"))
            print("episode={episode} VAoI={vaoi:.4f} tx_ratio={tx:.4f} multiplier={multiplier:.4f}".format(episode=episode, vaoi=row["mean_VAoI"], tx=average_cost, multiplier=multiplier_next))
    finally:
        model.close()

    metadata = {
        "checkpoint": str(output_directory / "checkpoints" / "model"), "episodes": len(history),
        "target_tx_ratios": targets, "node_counts": node_counts, "update_probabilities": update_probabilities,
        "model_version": "node_only_freshness_lagrangian_v3", "vaoi_reward_scale": vaoi_reward_scale,
        "consecutive_tx_scale": consecutive_tx_scale, "neighbor_confidence_time_constant": confidence_time_constant,
        "initial_multiplier": initial_multiplier, "multiplier_learning_rate": multiplier_learning_rate,
        "multiplier_max": multiplier_max, "tx_ratio_ema_beta": tx_ratio_ema_beta,
        "budget_relative_tolerance": relative_tolerance, "multiplier_error_clip": multiplier_error_clip,
        "per_node_cap_multiplier": float(constraints.get("per_node_cap_multiplier", 1.5)),
    }
    with (output_directory / "model_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    return output_directory
