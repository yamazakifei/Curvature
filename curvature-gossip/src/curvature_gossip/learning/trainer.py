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
import yaml

from ..channel import ChannelParameters, PropagationModel
from ..config import load_config
from ..curvature import bottleneck_importance
from ..experiments.runner import _curvature_provider
from ..policies.random_policy import UniformRandomPolicy
from ..random_streams import make_rng
from ..simulator import GossipSimulator, SimulationParameters
from ..topology import get_topology_generator
from .ctde_ppo import CTDEPPO, resolve_learning_rates
from .seed_plan import build_seed_manifest
from .stage1_search import run_stage1_search
from .features import (
    encode_curvature_score, encode_global_state, encode_observations,
    encode_stage1_observations, encode_stage2_observations, encode_stage2_mpnn_observations,
    encode_node_critic_inputs, node_critic_feature_names,
    stage2_context_feature_names, STAGE2_MPNN_NODE_FEATURE_NAMES, STAGE2_MPNN_EDGE_FEATURE_NAMES,
    STAGE2_MPNN_NO_CURVATURE_EDGE_FEATURE_NAMES,
)
from .validation import evaluate_fixed_validation, probability_statistics, validation_scenarios


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
    derived_mode = str(raw.get("randomness", {}).get("mode", "legacy_explicit")) == "derived_namespaces"
    topology_seed = int(training.get("topology_seed_start", 10000)) + int(episode)
    generator = get_topology_generator(raw["topology"]["type"])
    topology_params = dict(raw["topology"].get("params", {}))
    if node_count is not None:
        n_nodes = int(node_count)
        topology_params["n_nodes"] = n_nodes
        if "cluster_sizes" in topology_params:
            topology_params["cluster_sizes"] = [n_nodes // 2, n_nodes - n_nodes // 2]
    topology_rng = (
        make_rng(master_seed, "stage2_training", "topology", int(episode))
        if derived_mode else make_rng(master_seed, "nn_topology", topology_seed)
    )
    topology = generator.generate(topology_rng, topology_params)
    curvature_config = raw.get("curvature", {})
    curvature = _curvature_provider(curvature_config).compute(topology)
    importance = bottleneck_importance(topology, curvature, curvature_config.get("normalization", "local_degree_bound"))
    channel = ChannelParameters.from_mapping(raw.get("channel", {}))
    propagation_rng = (
        make_rng(master_seed, "stage2_training", "shadowing", int(episode))
        if derived_mode else make_rng(master_seed, "nn_shadowing", topology_seed)
    )
    propagation = PropagationModel(topology.positions, channel, propagation_rng)
    constraints = raw.get("constraints", {})
    observation = raw.get("observation", {})
    rollout_slots = int(training.get("rollout_slots", experiment.get("slots", 200)))
    if update_probability is None:
        update_probability = float(raw.get("source", {}).get("update_probability", 0.05))
    parameters = SimulationParameters(
        slots=rollout_slots, update_probability=update_probability,
        target_tx_ratio=float(target_tx_ratio),
        per_node_cap_multiplier=float(constraints.get("per_node_cap_multiplier", 1.5)),
        congestion_ewma_beta=float(observation.get("congestion_ewma_beta", 0.8)),
        congestion_feature_scale=float(observation.get("congestion_feature_scale", 5.0)),
    )
    update_rng = (
        make_rng(master_seed, "stage2_training", "source_updates", int(episode))
        if derived_mode else make_rng(master_seed, "nn_updates", topology_seed)
    )
    fading_rng = (
        make_rng(master_seed, "stage2_training", "fading", int(episode))
        if derived_mode else make_rng(master_seed, "nn_fading", topology_seed)
    )
    policy_rng = (
        make_rng(master_seed, "stage2_training", "actions", int(episode))
        if derived_mode else make_rng(master_seed, "nn_policy", topology_seed)
    )
    simulator = GossipSimulator(
        topology, curvature, importance, propagation, UniformRandomPolicy(0.0), parameters,
        update_rng, fading_rng, policy_rng,
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


def _node_gae(rewards, values, bootstrap_values, gamma: float, gae_lambda: float):
    """Compute node-wise GAE with a real next-decision-state bootstrap."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    bootstrap_values = np.asarray(bootstrap_values, dtype=np.float32)
    if rewards.ndim != 1 or values.ndim != 2 or bootstrap_values.ndim != 1:
        raise ValueError("node GAE expects rewards [T], values [T,N], bootstrap [N]")
    if values.shape[0] != rewards.shape[0] or bootstrap_values.shape != (values.shape[1],):
        raise ValueError("node GAE shapes must align across time and nodes")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all() or not np.isfinite(bootstrap_values).all():
        raise ValueError("node GAE inputs must be finite")
    advantages = np.zeros_like(values, dtype=np.float32)
    accumulator = np.zeros(values.shape[1], dtype=np.float32)
    for index in range(values.shape[0] - 1, -1, -1):
        next_value = bootstrap_values if index == values.shape[0] - 1 else values[index + 1]
        delta = rewards[index] + float(gamma) * next_value - values[index]
        accumulator = delta + float(gamma) * float(gae_lambda) * accumulator
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


def _write_csv_history(path: Path, rows) -> None:
    """Persist a validation CSV with stable columns after every fixed-set evaluation."""
    if not rows:
        return
    def format_value(value):
        """Keep validation CSV values compact without changing in-memory precision."""
        if isinstance(value, (float, np.floating)) and np.isfinite(value):
            return "{:.5f}".format(float(value))
        return value
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(value) for key, value in row.items()})


def _save_checkpoint(model: CTDEPPO, output_directory: Path, name: str, episode: int, config_path: str, actor_metadata=None) -> str:
    """Save actor, critic, Adam slots, and a manifest identifying the exact training state."""
    directory = output_directory / "checkpoints" / name
    checkpoint = model.save(str(directory / "model"))
    with (directory / "checkpoint_info.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "checkpoint": checkpoint,
            "episode": int(episode),
            "config": "../../training_config.yaml",
            "source_config": str(config_path),
            "actor": _json_safe(actor_metadata or {}),
        }, stream, indent=2, sort_keys=True)
    return checkpoint


def _stage1_actor_config(raw: Mapping) -> Mapping:
    """Validate Stage-1 or Stage-2 hierarchy settings and checkpoint feature metadata."""
    actor = dict(raw.get("actor", {}))
    stage = int(actor.get("stage", 0))
    if stage not in (1, 2):
        return actor
    curvature = dict(actor.get("curvature", {}))
    curvature_enabled = bool(curvature.get("enabled", True))
    if curvature.get("score", "incident_bottleneck_max") != "incident_bottleneck_max":
        raise ValueError("Stage 1 requires curvature.score=incident_bottleneck_max")
    if curvature.get("alpha_parameterization", "softplus") != "softplus":
        raise ValueError("Stage 1 requires softplus alpha parameterization")
    residual = dict(actor.get("residual", {}))
    if stage == 1 and bool(residual.get("enabled", False)):
        raise ValueError("Stage 1 must not instantiate a residual MLP")
    if stage == 1 and not curvature_enabled:
        raise ValueError("Stage 1 requires actor.curvature.enabled=true")
    actor["curvature"] = curvature
    if stage == 1:
        actor["architecture_version"] = "curvature_stage1_v1"
        actor["input_feature_names"] = ["incident_bottleneck_max"]
        return actor
    if not bool(residual.get("enabled", False)):
        raise ValueError("Stage 2 requires residual.enabled=true")
    architecture = str(residual.get("architecture", "mlp"))
    if architecture not in ("mlp", "mpnn"):
        raise ValueError("Stage 2 residual.architecture must be 'mlp' or 'mpnn'")
    if list(residual.get("hidden_dims", [64, 64])) != [64, 64]:
        raise ValueError("Stage 2 requires residual.hidden_dims=[64, 64]")
    if residual.get("activation", "relu") != "relu" or not bool(residual.get("zero_init_output", True)):
        raise ValueError("Stage 2 requires ReLU hidden layers and a zero-initialized output")
    use_stage1_reference = bool(residual.get("use_stage1_reference", True))
    if use_stage1_reference and not bool(residual.get("detach_stage1_reference", True)):
        raise ValueError("Stage 2 with a Stage-1 reference requires detach_stage1_reference=true")
    if bool(residual.get("include_curvature_mean", False)) or bool(residual.get("include_curvature_freshness_interaction", False)):
        raise ValueError("the clean Stage-2 model excludes optional curvature augmentations")
    if not use_stage1_reference:
        override = curvature.get("alpha_override")
        if override is None or float(override) != 0.0:
            raise ValueError("Stage-2 without a Stage-1 reference requires actor.curvature.alpha_override=0.0")
    if not curvature_enabled:
        if use_stage1_reference:
            raise ValueError("no-curvature Stage 2 must disable residual.use_stage1_reference")
        if curvature.get("alpha_override") is None or float(curvature.get("alpha_override")) != 0.0:
            raise ValueError("no-curvature Stage 2 requires actor.curvature.alpha_override=0.0")
    residual.setdefault("delta_max", 1.0)
    residual.setdefault("freeze_stage1", False)
    residual["architecture"] = architecture
    actor["residual"] = residual
    if architecture == "mpnn":
        mpnn = dict(residual.get("mpnn", {}))
        required = {
            "message_hidden_dims": [32], "message_dim": 16, "update_hidden_dims": [64, 64],
            "aggregation": "mean_max", "condition_message_on_receiver": True,
            "use_sender_node_features": False,
        }
        for key, expected in required.items():
            if mpnn.get(key, expected) != expected:
                raise ValueError("Stage 2 MPNN requires mpnn.{}={!r}".format(key, expected))
        use_curvature_edge_feature = bool(mpnn.get("use_curvature_edge_feature", False))
        if not use_stage1_reference and use_curvature_edge_feature:
            raise ValueError("no-curvature Stage 2 MPNN must disable use_curvature_edge_feature")
        bmax = float(mpnn.get("bmax", 1.0))
        if not np.isfinite(bmax) or bmax <= 0.0:
            raise ValueError("Stage 2 MPNN requires mpnn.bmax to be finite and positive")
        residual["mpnn"] = dict(
            required, use_curvature_edge_feature=use_curvature_edge_feature, bmax=bmax,
        )
        node_features = list(STAGE2_MPNN_NODE_FEATURE_NAMES)
        if bool(residual.get("include_scenario_context", False)):
            node_features += ["log_network_size", "update_probability", "target_tx_ratio"]
        actor.update({
            "architecture_version": "curvature_stage2_mpnn_v1" if use_stage1_reference else "no_curvature_stage2_mpnn_v1",
            "residual_architecture": "mpnn", "node_input_feature_names": node_features,
            "edge_input_feature_names": list(
                STAGE2_MPNN_EDGE_FEATURE_NAMES if use_curvature_edge_feature
                else STAGE2_MPNN_NO_CURVATURE_EDGE_FEATURE_NAMES
            ),
            "node_input_dim": len(node_features), "edge_input_dim": 3, "message_input_dim": len(node_features) + 3,
            "message_dim": 16, "aggregation": "mean_max", "aggregated_message_dim": 32,
            "decoder_input_dim": len(node_features) + 32 + (1 if use_stage1_reference else 0),
            "condition_message_on_receiver": True, "use_sender_node_features": False,
            "use_curvature_edge_feature": use_curvature_edge_feature, "bmax": bmax,
            "input_feature_names": node_features,
            "curvature_enabled": curvature_enabled,
        })
        return actor
    context_features = list(stage2_context_feature_names(
        bool(residual.get("include_scenario_context", False))
    ))
    actor["architecture_version"] = (
        "curvature_stage2_residual_v1" if use_stage1_reference
        else "no_curvature_stage2_residual_v1"
    )
    actor["input_feature_names"] = (
        context_features + ["detached_stage1_probability"]
        if use_stage1_reference else context_features
    )
    actor["residual_architecture"] = "mlp"
    actor["curvature_enabled"] = curvature_enabled
    return actor


def _freeze_stage1_center(raw: Mapping, actor: Mapping) -> float:
    """Compute one offline mean score over configured training topologies and freeze it."""
    curvature = dict(actor["curvature"])
    center = curvature.get("center", "auto")
    if isinstance(center, (int, float)) and not isinstance(center, bool):
        if not np.isfinite(float(center)):
            raise ValueError("actor.curvature.center must be finite")
        return float(center)
    if center not in (None, "auto"):
        raise ValueError("actor.curvature.center must be a number, null, or 'auto'")
    training = raw.get("training", {})
    # By default, calibrate the fixed center from every topology used in training.
    seeds = training.get("center_topology_seeds")
    if seeds is None:
        start = int(training.get("topology_seed_start", 10000))
        seeds = list(range(start, start + int(training.get("episodes", 20))))
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("training.center_topology_seeds must be a nonempty list")
    generator = get_topology_generator(raw["topology"]["type"])
    master_seed = int(raw["experiment"].get("master_seed", 0))
    values = []
    for topology_seed in seeds:
        topology = generator.generate(make_rng(master_seed, "nn_topology", int(topology_seed)), raw["topology"].get("params", {}))
        curvature_result = _curvature_provider(raw.get("curvature", {})).compute(topology)
        importance = bottleneck_importance(topology, curvature_result, raw.get("curvature", {}).get("normalization", "local_degree_bound"))
        values.extend(max([importance[tuple(sorted((node, neighbor)))] for neighbor in topology.graph.neighbors(node)] or [0.0]) for node in topology.graph.nodes)
    return float(np.mean(values))


def _critic_config(raw: Mapping) -> Mapping:
    """Resolve Critic architecture and checkpointed feature metadata from YAML."""
    critic = dict(raw.get("critic", {}))
    architecture = str(critic.get("architecture", "scalar_global"))
    if architecture == "scalar_global":
        critic.update({"architecture": architecture, "input_dim": 8})
        return critic
    if architecture != "node_conditioned":
        raise ValueError("critic.architecture must be scalar_global or node_conditioned")
    include_scenario = bool(critic.get("include_scenario_context", True))
    include_exact = bool(critic.get("include_exact_node_features", True))
    include_channel = bool(critic.get("include_channel_features", True))
    include_curvature = bool(critic.get("include_curvature_features", True))
    hidden_dims = list(critic.get("hidden_dims", [64, 64]))
    if hidden_dims != [64, 64] or critic.get("activation", "relu") != "relu":
        raise ValueError("node-conditioned Critic requires hidden_dims=[64, 64] and relu activation")
    names = node_critic_feature_names(
        include_scenario, include_exact, include_channel, include_curvature
    )
    configured_dim = int(critic.get("input_dim", len(names)))
    if configured_dim != len(names):
        raise ValueError("critic.input_dim does not match the enabled feature groups")
    critic.update({
        "architecture": architecture,
        "hidden_dims": hidden_dims,
        "activation": "relu",
        "include_scenario_context": include_scenario,
        "include_exact_node_features": include_exact,
        "include_channel_features": include_channel,
        "include_curvature_features": include_curvature,
        "advantage_mode": str(critic.get("advantage_mode", "node_gae")),
        "bootstrap_rollout_end": bool(critic.get("bootstrap_rollout_end", True)),
        "input_dim": configured_dim,
        "input_feature_names": list(names),
    })
    if critic["advantage_mode"] != "node_gae" or not critic["bootstrap_rollout_end"]:
        raise ValueError("node-conditioned Critic requires node_gae and rollout-end bootstrap")
    return critic


def train_ctde(config_path: str):
    """Train and save a V3 checkpoint, metadata, and per-rollout diagnostics."""
    config = load_config(config_path)
    raw = config.raw
    training = raw.get("training", {})
    constraints = raw.get("constraints", {})
    output_root = Path(raw.get("output", {}).get("root", "result_NN"))
    output_directory = output_root / str(raw["experiment"].get("id", "nn_ctde_v3"))
    output_directory.mkdir(parents=True, exist_ok=True)

    # Resolve all automatic pools before model creation and persist the exact plan.
    seed_manifest = build_seed_manifest(raw)
    with (output_directory / "resolved_seed_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(seed_manifest, stream, indent=2, sort_keys=True)

    search_result = run_stage1_search(raw, output_directory)
    actor_config = _stage1_actor_config(raw)
    if search_result:
        actor_config = dict(actor_config)
        actor_curvature = dict(actor_config.get("curvature", {}))
        actor_curvature.update(search_result)
        actor_config["curvature"] = actor_curvature
        raw["actor"] = actor_config
    critic_config = _critic_config(raw)
    if (int(actor_config.get("stage", 0)) in (1, 2)
            and bool(actor_config.get("curvature", {}).get("enabled", True))):
        frozen_center = _freeze_stage1_center(raw, actor_config)
        actor_config = dict(actor_config)
        actor_config["curvature"] = dict(actor_config["curvature"], center=frozen_center)
        raw["actor"] = actor_config
    elif int(actor_config.get("stage", 0)) == 2:
        # No-curvature runs must not spend startup time computing a curvature center.
        raw["actor"] = actor_config
    with (output_directory / "training_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(raw), stream, sort_keys=False, allow_unicode=True)
    with (output_directory / "resolved_training_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(raw), stream, sort_keys=False, allow_unicode=True)

    vaoi_reward_scale = float(training.get("vaoi_reward_scale", 0.1))
    consecutive_tx_scale = float(training.get("consecutive_tx_scale", 3.0))
    confidence_time_constant = float(raw.get("observation", {}).get("neighbor_confidence_time_constant", training.get("neighbor_confidence_time_constant", 20.0)))
    congestion_feature_scale = float(raw.get("observation", {}).get("congestion_feature_scale", 5.0))
    initial_multiplier = float(training.get("initial_multiplier", 0.0))
    multiplier_learning_rate = float(training.get("multiplier_learning_rate", 0.01))
    actor_learning_rate, critic_learning_rate = resolve_learning_rates(training)
    common_mode_coefficient = float(training.get("common_mode_coefficient", 0.0))
    if not np.isfinite(common_mode_coefficient) or common_mode_coefficient < 0.0:
        raise ValueError("training.common_mode_coefficient must be finite and nonnegative")
    multiplier_max = float(training.get("multiplier_max", 0.15))
    tx_ratio_ema_beta = float(training.get("tx_ratio_ema_beta", 0.9))
    relative_tolerance = float(training.get("budget_relative_tolerance", 0.05))
    multiplier_error_clip = float(training.get("multiplier_error_clip", 1.0))
    use_budget_advantage = bool(training.get("use_budget_advantage", True))
    update_multiplier = bool(training.get("update_multiplier", True))
    validation_config = raw.get("validation", {})
    validation_enabled = bool(validation_config.get("enabled", False))
    validation_trigger = str(validation_config.get("trigger", "every_episodes"))
    checkpoint_mode = str(validation_config.get("checkpoint_mode", "periodic_and_latest"))
    validation_every = int(validation_config.get("every_episodes", 10))
    checkpoint_every = int(validation_config.get(
        "checkpoint_every_episodes", training.get("checkpoint_every_episodes", 20)
    ))
    if vaoi_reward_scale <= 0.0 or consecutive_tx_scale <= 0.0 or confidence_time_constant <= 0.0:
        raise ValueError("V3 reward scale, consecutive scale, and confidence time constant must be positive")
    if not 0.0 <= initial_multiplier <= multiplier_max or multiplier_learning_rate < 0.0:
        raise ValueError("invalid initial multiplier or multiplier learning rate")
    if not 0.0 <= tx_ratio_ema_beta <= 1.0 or relative_tolerance < 0.0 or multiplier_error_clip <= 0.0:
        raise ValueError("invalid V3 EMA or multiplier stability setting")
    if validation_enabled:
        if validation_trigger not in ("every_episodes", "train_mean_vaoi_improvement"):
            raise ValueError("validation.trigger must be every_episodes or train_mean_vaoi_improvement")
        if checkpoint_mode not in ("periodic_and_latest", "validation_best_only"):
            raise ValueError("validation.checkpoint_mode must be periodic_and_latest or validation_best_only")
        if validation_trigger == "every_episodes" and validation_every < 1:
            raise ValueError("validation.every_episodes must be positive")
        if checkpoint_mode == "periodic_and_latest" and checkpoint_every < 1:
            raise ValueError("checkpoint interval must be positive")
        validation_scenarios(raw)

    seed = int(raw["experiment"].get("master_seed", 0))
    model = CTDEPPO(
        learning_rate=float(training.get("learning_rate", 3e-4)),
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        clip_ratio=float(training.get("clip_ratio", 0.2)),
        entropy_coefficient=float(training.get("entropy_coefficient", 0.01)),
        seed=seed, actor_config=actor_config, critic_config=critic_config,
        common_mode_coefficient=common_mode_coefficient,
    )
    initial_checkpoint = training.get("initial_checkpoint")
    restored_variables = ()
    if initial_checkpoint:
        if (not model.actor_curvature_enabled
                and not critic_config.get("include_curvature_features", True)):
            raise ValueError("dual no-curvature experiments must train from scratch without initial_checkpoint")
        restored_variables = (
            model.restore_actor_only(str(initial_checkpoint))
            if bool(training.get("restore_actor_only", False))
            else model.restore_compatible(str(initial_checkpoint))
        )
    action_rng = make_rng(seed, "stage2_training", "actions", "ppo_actor") if str(raw.get("randomness", {}).get("mode", "legacy_explicit")) == "derived_namespaces" else np.random.default_rng(seed + 7919)
    targets = [float(value) for value in training.get("max_tx_ratios", [constraints.get("max_tx_ratio", 0.1)])]
    node_counts = list(training.get("node_counts", [None])) or [None]
    update_probabilities = list(training.get("update_probabilities", [None])) or [None]
    if not targets or any(not 0.0 < value <= 1.0 for value in targets):
        raise ValueError("training.max_tx_ratios must contain values in (0, 1]")
    training_cases = [(target, node_count, update_probability) for node_count in node_counts for update_probability in update_probabilities for target in targets]
    constraint_state = {}
    history = []
    validation_history = []
    validation_per_scenario = []
    best_train_vaoi = float("inf")
    best_nn_vaoi = float("inf")
    best_matched_delta = float("inf")

    def run_validation(checkpoint_episode: int, checkpoint_label: str, trigger_train_vaoi=None) -> None:
        """Evaluate frozen parameters and update the fixed-validation CSV artifacts."""
        nonlocal best_nn_vaoi, best_matched_delta
        summary_row, scenario_rows = evaluate_fixed_validation(
            model, raw, checkpoint_episode, checkpoint_label
        )
        if trigger_train_vaoi is not None:
            summary_row["trigger_train_mean_VAoI"] = float(trigger_train_vaoi)
        is_validation_best = summary_row["nn_mean_VAoI"] < best_nn_vaoi
        summary_row["is_validation_best"] = bool(is_validation_best)
        validation_history.append(summary_row)
        validation_per_scenario.extend(scenario_rows)
        _write_csv_history(output_directory / "validation_history.csv", validation_history)
        _write_csv_history(
            output_directory / "validation_per_scenario.csv", validation_per_scenario
        )
        if is_validation_best:
            best_nn_vaoi = summary_row["nn_mean_VAoI"]
            checkpoint_name = "best_validation" if checkpoint_mode == "validation_best_only" else "best"
            _save_checkpoint(model, output_directory, checkpoint_name, checkpoint_episode, config_path, actor_config)
        if checkpoint_mode == "periodic_and_latest" and summary_row["delta_vaoi_matched_mean"] < best_matched_delta:
            best_matched_delta = summary_row["delta_vaoi_matched_mean"]
            _save_checkpoint(
                model, output_directory, "best_matched", checkpoint_episode, config_path, actor_config
            )

    try:
        if validation_enabled:
            if checkpoint_mode == "periodic_and_latest":
                # Legacy mode evaluates the initialization and retains a resumable checkpoint.
                _save_checkpoint(model, output_directory, "latest", 0, config_path, actor_config)
                _save_checkpoint(model, output_directory, "episode_000000", 0, config_path, actor_config)
                run_validation(0, "initial")
        for episode in range(int(training.get("episodes", 20))):
            target, node_count, update_probability = training_cases[episode % len(training_cases)]
            simulator, curvature = _build_simulator(raw, episode, target, node_count, update_probability)
            case_key = (simulator.state.n_nodes, simulator.parameters.update_probability, target)
            if case_key not in constraint_state:
                constraint_state[case_key] = {"multiplier": initial_multiplier, "tx_ratio_ema": target}
            multiplier_used = constraint_state[case_key]["multiplier"]
            encoded_steps, actions_steps, probability_steps, old_log_steps = [], [], [], []
            base_probability_steps, residual_delta_steps = [], []
            value_steps, critic_inputs_steps, global_states = [], [], []
            vaoi_rewards, costs, vaoi_values = [], [], []
            bootstrap_values = None
            last_tx_ratio = 0.0
            for slot in range(simulator.parameters.slots):
                observations = simulator.begin_step(slot)
                if model.actor_stage == 1:
                    encoded = encode_stage1_observations(observations, target)
                elif model.actor_stage == 2:
                    residual = actor_config["residual"]
                    include_scenario_context = bool(residual.get("include_scenario_context", False))
                    if residual.get("architecture", "mlp") == "mpnn":
                        # MPNN-only edge controls stay out of the legacy MLP encoder.
                        encoded = encode_stage2_mpnn_observations(
                            observations, target, simulator.parameters.update_probability,
                            consecutive_tx_scale, confidence_time_constant, congestion_feature_scale,
                            include_scenario_context,
                            use_curvature_edge_feature=bool(residual.get("mpnn", {}).get("use_curvature_edge_feature", False)),
                            bmax=float(residual.get("mpnn", {}).get("bmax", 1.0)),
                            use_curvature=model.actor_curvature_enabled,
                        )
                    else:
                        encoded = encode_stage2_observations(
                            observations, target, simulator.parameters.update_probability,
                            consecutive_tx_scale, confidence_time_constant, congestion_feature_scale,
                            include_scenario_context,
                            model.actor_curvature_enabled,
                        )
                else:
                    encoded = encode_observations(
                        observations, target, simulator.parameters.update_probability,
                        consecutive_tx_scale, confidence_time_constant, congestion_feature_scale,
                    )
                global_state = encode_global_state(
                    simulator.centralized_state(last_tx_ratio), target,
                    simulator.parameters.update_probability, simulator.state.n_nodes,
                )
                if model.critic_architecture == "node_conditioned":
                    # Centralized state and mean channel power are training-only Critic inputs.
                    encoded_critic = encode_node_critic_inputs(
                        observations, simulator.state.version_age(), last_tx_ratio, target,
                        simulator.parameters.update_probability, simulator.state.n_nodes,
                        mean_rx_power_mw=simulator.propagation.mean_rx_power_mw,
                        noise_power_mw=simulator.propagation.parameters.noise_power_mw,
                        sinr_threshold_db=simulator.propagation.parameters.sinr_threshold_db,
                        include_scenario_context=critic_config["include_scenario_context"],
                        include_exact_node_features=critic_config["include_exact_node_features"],
                        include_channel_features=critic_config["include_channel_features"],
                        include_curvature_features=critic_config["include_curvature_features"],
                        consecutive_tx_scale=consecutive_tx_scale,
                        neighbor_confidence_time_constant=confidence_time_constant,
                        congestion_feature_scale=congestion_feature_scale,
                    )
                    value = model.value(encoded_critic.critic_inputs).astype(np.float32)
                    critic_inputs_steps.append(encoded_critic.critic_inputs)
                else:
                    value = float(model.value(global_state)[0])
                    critic_inputs_steps.append(global_state)
                actions, probabilities, old_log_probabilities = model.act(encoded, action_rng)
                if model.actor_stage == 2:
                    components = model.stage2_diagnostics(encoded)
                    base_probability_steps.append(components["q_base"].astype(np.float32))
                    residual_delta_steps.append(components["delta"].astype(np.float32))
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

            if model.critic_architecture == "node_conditioned":
                # Build the next decision state after the final transition for bootstrap only.
                bootstrap_observations = simulator.begin_step(simulator.parameters.slots)
                bootstrap_input = encode_node_critic_inputs(
                    bootstrap_observations, simulator.state.version_age(), last_tx_ratio, target,
                    simulator.parameters.update_probability, simulator.state.n_nodes,
                    mean_rx_power_mw=simulator.propagation.mean_rx_power_mw,
                    noise_power_mw=simulator.propagation.parameters.noise_power_mw,
                    sinr_threshold_db=simulator.propagation.parameters.sinr_threshold_db,
                    include_scenario_context=critic_config["include_scenario_context"],
                    include_exact_node_features=critic_config["include_exact_node_features"],
                    include_channel_features=critic_config["include_channel_features"],
                    include_curvature_features=critic_config["include_curvature_features"],
                    consecutive_tx_scale=consecutive_tx_scale,
                    neighbor_confidence_time_constant=confidence_time_constant,
                    congestion_feature_scale=congestion_feature_scale,
                )
                bootstrap_values = model.value(bootstrap_input.critic_inputs).astype(np.float32)
                value_matrix = np.asarray(value_steps, dtype=np.float32)
                vaoi_advantages, returns = _node_gae(
                    vaoi_rewards, value_matrix, bootstrap_values,
                    float(training.get("gamma", 0.99)), float(training.get("gae_lambda", 0.95)),
                )
                normalized_vaoi_advantages = _standardize(vaoi_advantages.reshape(-1)).reshape(vaoi_advantages.shape)
                actor_vaoi_advantages = np.concatenate([
                    normalized_vaoi_advantages[index].astype(np.float32, copy=False)
                    for index in range(len(encoded_steps))
                ])
                critic_batch = {
                    "critic_inputs": np.concatenate(critic_inputs_steps).astype(np.float32),
                    "returns": returns.reshape(-1).astype(np.float32),
                }
            else:
                vaoi_advantages, returns = _gae(
                    vaoi_rewards, value_steps,
                    float(training.get("gamma", 0.99)), float(training.get("gae_lambda", 0.95)),
                )
                normalized_vaoi_advantages = _standardize(vaoi_advantages)
                actor_vaoi_advantages = np.concatenate([
                    np.full(
                        (step.curvature_scores if model.actor_stage in (1, 2) else step.node_features).shape[0],
                        normalized_vaoi_advantages[index], dtype=np.float32,
                    ) for index, step in enumerate(encoded_steps)
                ])
                critic_batch = {"global_states": np.asarray(global_states, dtype=np.float32), "returns": returns}
            action_batch = np.concatenate(actions_steps)
            probability_batch = np.concatenate(probability_steps)
            budget_advantages = (
                _budget_advantages(action_batch, probability_batch, multiplier_used, multiplier_max)
                if use_budget_advantage else np.zeros_like(action_batch, dtype=np.float32)
            )
            actor_advantages = (actor_vaoi_advantages + budget_advantages).astype(np.float32)
            actor_batch = model.actor_batch_inputs(encoded_steps)
            actor_batch.update({"actions": action_batch, "old_log_probabilities": np.concatenate(old_log_steps), "advantages": actor_advantages})
            vaoi_batch = dict(actor_batch, advantages=actor_vaoi_advantages)
            budget_batch = dict(actor_batch, advantages=budget_advantages)
            vaoi_gradient_norm = model.policy_gradient_norm(vaoi_batch)
            budget_gradient_norm = model.policy_gradient_norm(budget_batch)
            common_mode_gradient_norm = model.common_mode_gradient_norm(actor_batch)
            total_actor_gradient_norm = model.total_actor_gradient_norm(actor_batch)
            losses = model.update(actor_batch, critic_batch, epochs=int(training.get("ppo_epochs", 4)))

            if model.critic_architecture == "node_conditioned":
                critic_value_matrix = np.asarray(value_steps, dtype=np.float32)
                critic_return_matrix = np.asarray(returns, dtype=np.float32)
                critic_advantage_matrix = np.asarray(vaoi_advantages, dtype=np.float32)
                bootstrap_mean = float(np.mean(bootstrap_values))
                bootstrap_std = float(np.std(bootstrap_values))
            else:
                critic_value_matrix = np.asarray(value_steps, dtype=np.float32)[:, None]
                critic_return_matrix = np.asarray(returns, dtype=np.float32)[:, None]
                critic_advantage_matrix = np.asarray(vaoi_advantages, dtype=np.float32)[:, None]
                bootstrap_mean = 0.0
                bootstrap_std = 0.0
            critic_residual = critic_return_matrix - critic_value_matrix
            return_variance = float(np.var(critic_return_matrix))
            explained_variance = 1.0 - float(np.var(critic_residual)) / (return_variance + 1e-8)

            average_cost = float(np.mean(costs))
            tx_ratio_ema = _update_tx_ratio_ema(
                constraint_state[case_key]["tx_ratio_ema"], average_cost, tx_ratio_ema_beta
            )
            multiplier_next, relative_error, deadzone_error = _update_multiplier(
                multiplier_used, tx_ratio_ema, target,
                multiplier_learning_rate if update_multiplier else 0.0,
                multiplier_max, relative_tolerance, multiplier_error_clip,
            )
            constraint_state[case_key] = {"multiplier": multiplier_next, "tx_ratio_ema": tx_ratio_ema}
            summary = simulator.metrics.summary()
            row = {
                "episode": episode, "mean_VAoI": float(np.mean(vaoi_values)), "avg_tx_ratio": average_cost,
                "tx_ratio_ema": tx_ratio_ema, "target_tx_ratio": target, "budget_error": tx_ratio_ema - target,
                "budget_relative_error": relative_error, "budget_error_after_deadzone": deadzone_error,
                "mean_action_probability": float(np.mean(probability_steps)), "multiplier_used": multiplier_used,
                "multiplier_next": multiplier_next, "multiplier_at_max": bool(np.isclose(multiplier_next, multiplier_max)),
                "actor_loss": losses["actor_loss"], "critic_loss": losses["critic_loss"], "entropy": losses["entropy"], "approx_kl": losses["approx_kl"], "clip_fraction": losses["clip_fraction"],
                "common_mode_coefficient": common_mode_coefficient,
                "common_mode_loss": losses["common_mode_loss"],
                "ppo_policy_gradient_norm": vaoi_gradient_norm,
                "total_actor_gradient_norm": total_actor_gradient_norm,
                "common_mode_gradient_norm": common_mode_gradient_norm,
                "mean_vaoi_advantage": float(np.mean(actor_vaoi_advantages)), "std_vaoi_advantage": float(np.std(actor_vaoi_advantages)), "max_abs_vaoi_advantage": float(np.max(np.abs(actor_vaoi_advantages))),
                "critic_architecture": model.critic_architecture,
                "critic_input_dim": model.critic_input_dim,
                "actor_curvature_enabled": bool(model.actor_curvature_enabled),
                "critic_curvature_enabled": bool(critic_config.get("include_curvature_features", True)),
                "curvature_ablation_mode": bool(
                    not model.actor_curvature_enabled
                    and not critic_config.get("include_curvature_features", True)
                ),
                "critic_value_mean": float(np.mean(critic_value_matrix)),
                "critic_value_std": float(np.std(critic_value_matrix)),
                "critic_within_slot_value_std_mean": float(np.mean(np.std(critic_value_matrix, axis=1))),
                "critic_within_slot_value_std_max": float(np.max(np.std(critic_value_matrix, axis=1))),
                "vaoi_advantage_within_slot_std_mean": float(np.mean(np.std(critic_advantage_matrix, axis=1))),
                "vaoi_advantage_global_std": float(np.std(critic_advantage_matrix)),
                "critic_return_mean": float(np.mean(critic_return_matrix)),
                "critic_return_std": float(np.std(critic_return_matrix)),
                "critic_explained_variance": explained_variance,
                "bootstrap_value_mean": bootstrap_mean,
                "bootstrap_value_std": bootstrap_std,
                "mean_budget_advantage": float(np.mean(budget_advantages)), "std_budget_advantage": float(np.std(budget_advantages)), "max_abs_budget_advantage": float(np.max(np.abs(budget_advantages))),
                "vaoi_policy_gradient_norm": vaoi_gradient_norm, "budget_policy_gradient_norm": budget_gradient_norm,
                "budget_to_vaoi_gradient_ratio": budget_gradient_norm / (vaoi_gradient_norm + 1e-8),
                "n_nodes": simulator.state.n_nodes, "update_probability": simulator.parameters.update_probability,
                "max_node_activity_ratio": summary["max_node_activity_ratio"], "node_cap_violation_fraction": summary["node_cap_violation_fraction"],
                "curvature_control_messages": curvature.metadata.get("control_message_count", 0),
            }
            if model.actor_stage == 1:
                diagnostics = model.stage1_diagnostics(encoded_steps[-1])
                scores = np.concatenate([item.curvature_scores[:, 0] for item in encoded_steps])
                q_base = np.concatenate(probability_steps)
                row.update({
                    "alpha_raw": float(diagnostics["alpha_raw"]),
                    "alpha_kappa": float(diagnostics["alpha_kappa"]),
                    "q_base_mean": float(np.mean(q_base)), "q_base_std": float(np.std(q_base)),
                    "q_base_min": float(np.min(q_base)), "q_base_max": float(np.max(q_base)),
                    "corr_s_kappa_q_base": float(np.corrcoef(scores, q_base)[0, 1]) if np.std(scores) > 0.0 else 0.0,
                })
            elif model.actor_stage == 2:
                diagnostics = model.stage2_diagnostics(encoded_steps[-1])
                scores = np.concatenate([item.curvature_scores[:, 0] for item in encoded_steps])
                q_base = np.concatenate(base_probability_steps)
                delta = np.concatenate(residual_delta_steps)
                delta_matrix = np.asarray(residual_delta_steps, dtype=np.float32)
                step_residual_means = np.mean(delta_matrix, axis=1)
                delta_max = float(actor_config["residual"].get("delta_max", 1.0))
                row.update({
                    "actor_architecture_version": actor_config["architecture_version"],
                    "use_stage1_reference": bool(actor_config["residual"].get("use_stage1_reference", True)),
                    "alpha_raw": float(diagnostics["alpha_raw"]),
                    "alpha_kappa": float(diagnostics["alpha_kappa"]),
                    "effective_alpha": float(diagnostics["effective_alpha"]),
                    "residual_input_dim": int(diagnostics["residual_input"].shape[1]),
                    "residual_input_feature_names": "|".join(actor_config["input_feature_names"]),
                    "q_base_mean": float(np.mean(q_base)), "q_base_std": float(np.std(q_base)),
                    "q_base_min": float(np.min(q_base)), "q_base_max": float(np.max(q_base)),
                    "residual_delta_mean": float(np.mean(delta)), "residual_delta_std": float(np.std(delta)),
                    "residual_delta_min": float(np.min(delta)), "residual_delta_max": float(np.max(delta)),
                    "residual_mean_global": float(np.mean(delta)),
                    "residual_std_global": float(np.std(delta)),
                    "residual_mean_per_step_abs_mean": float(np.mean(np.abs(step_residual_means))),
                    "residual_mean_per_step_abs_max": float(np.max(np.abs(step_residual_means))),
                    "residual_positive_fraction": float(np.mean(delta > 0.0)),
                    "residual_negative_fraction": float(np.mean(delta < 0.0)),
                    "residual_saturation_fraction": float(np.mean(np.abs(delta) >= 0.95 * delta_max)),
                    "q_final_mean": float(np.mean(probability_steps)),
                    "q_final_std": float(np.std(probability_steps)),
                    "corr_s_kappa_q_base": float(np.corrcoef(scores, q_base)[0, 1]) if np.std(scores) > 0.0 else 0.0,
                })
                if actor_config["residual"].get("architecture", "mlp") == "mpnn":
                    edge_features = np.concatenate([step.edge_features for step in encoded_steps], axis=0)
                    edge_count = int(edge_features.shape[0])
                    message = np.concatenate([model.stage2_diagnostics(step)["edge_messages"] for step in encoded_steps], axis=0)
                    aggregated = np.concatenate([model.stage2_diagnostics(step)["aggregated_messages"] for step in encoded_steps], axis=0)
                    row.update({
                        "residual_architecture": "mpnn", "node_input_dim": actor_config["node_input_dim"],
                        "edge_input_dim": actor_config["edge_input_dim"], "message_dim": actor_config["message_dim"],
                        "aggregation": actor_config["aggregation"], "aggregated_message_dim": actor_config["aggregated_message_dim"],
                        "decoder_input_dim": actor_config["decoder_input_dim"], "directed_edge_count": edge_count,
                        "mpnn_disabled_edge_feature": (
                            "disabled_zero_placeholder"
                            if not actor_config.get("use_curvature_edge_feature", False) else "none"
                        ),
                        "mean_directed_degree": float(edge_count) / simulator.state.n_nodes,
                        "mean_edge_freshness_gain": float(np.mean(edge_features[:, 0])) if edge_count else 0.0,
                        "message_l2_mean": float(np.mean(np.linalg.norm(message, axis=1))) if edge_count else 0.0,
                        "message_l2_std": float(np.std(np.linalg.norm(message, axis=1))) if edge_count else 0.0,
                        "aggregated_message_l2_mean": float(np.mean(np.linalg.norm(aggregated, axis=1))),
                        "aggregated_message_l2_std": float(np.std(np.linalg.norm(aggregated, axis=1))),
                    })
            row.update(probability_statistics(probability_steps))
            if not all(np.isfinite(value) for value in row.values() if isinstance(value, (float, np.floating))):
                raise FloatingPointError("V3 training diagnostics must be finite")
            history.append(row)
            _write_history(output_directory, history)
            completed_episodes = episode + 1
            is_train_best = row["mean_VAoI"] < best_train_vaoi
            if is_train_best:
                best_train_vaoi = row["mean_VAoI"]
            # Selection mode validates only a new training minimum, then persists only a new validation minimum.
            if validation_enabled and validation_trigger == "train_mean_vaoi_improvement" and is_train_best:
                run_validation(completed_episodes, "train_mean_vaoi_best", row["mean_VAoI"])
            elif validation_enabled and validation_trigger == "every_episodes" and completed_episodes % validation_every == 0:
                if checkpoint_mode == "periodic_and_latest":
                    _save_checkpoint(model, output_directory, "latest", completed_episodes, config_path, actor_config)
                run_validation(completed_episodes, "episode_{:06d}".format(completed_episodes))
            if checkpoint_mode == "periodic_and_latest" and completed_episodes % checkpoint_every == 0:
                _save_checkpoint(
                    model, output_directory, "episode_{:06d}".format(completed_episodes),
                    completed_episodes, config_path, actor_config,
                )
            print("episode={episode} VAoI={vaoi:.4f} tx_ratio={tx:.4f} multiplier={multiplier:.4f}".format(episode=episode, vaoi=row["mean_VAoI"], tx=average_cost, multiplier=multiplier_next))
    finally:
        if history:
            completed_episodes = len(history)
            if checkpoint_mode == "periodic_and_latest" and (not validation_enabled or completed_episodes % validation_every != 0):
                _save_checkpoint(model, output_directory, "latest", completed_episodes, config_path, actor_config)
            if checkpoint_mode == "periodic_and_latest" and completed_episodes % checkpoint_every != 0:
                _save_checkpoint(
                    model, output_directory, "episode_{:06d}".format(completed_episodes),
                    completed_episodes, config_path, actor_config,
                )
        model.close()

    selected_checkpoint = (
        output_directory / "checkpoints" / ("best_validation" if checkpoint_mode == "validation_best_only" else "latest") / "model"
    )
    metadata = {
        "checkpoint": str(selected_checkpoint), "episodes": len(history),
        "target_tx_ratios": targets, "node_counts": node_counts, "update_probabilities": update_probabilities,
        "model_version": (actor_config.get("architecture_version", "curvature_stage2_residual") if model.actor_stage == 2 else
                          "curvature_stage1_single_alpha" if model.actor_stage == 1 else
                          "node_only_freshness_lagrangian_v3"),
        "actor": _json_safe(actor_config), "critic": _json_safe(critic_config),
        "actor_curvature_enabled": bool(model.actor_curvature_enabled),
        "critic_curvature_enabled": bool(critic_config.get("include_curvature_features", True)),
        "curvature_ablation_mode": bool(
            not model.actor_curvature_enabled
            and not critic_config.get("include_curvature_features", True)
        ),
        "critic_architecture": model.critic_architecture,
        "critic_input_dim": model.critic_input_dim,
        "critic_input_feature_names": list(model.critic_feature_names),
        "mpnn_disabled_edge_feature": (
            "disabled_zero_placeholder"
            if actor_config.get("residual_architecture") == "mpnn"
            and not actor_config.get("use_curvature_edge_feature", False)
            else None
        ),
        "common_mode_coefficient": common_mode_coefficient,
        "initial_checkpoint": initial_checkpoint,
        "restored_variables": list(restored_variables), "vaoi_reward_scale": vaoi_reward_scale,
        "actor_learning_rate": actor_learning_rate, "critic_learning_rate": critic_learning_rate,
        "consecutive_tx_scale": consecutive_tx_scale, "neighbor_confidence_time_constant": confidence_time_constant,
        "initial_multiplier": initial_multiplier, "multiplier_learning_rate": multiplier_learning_rate,
        "multiplier_max": multiplier_max, "tx_ratio_ema_beta": tx_ratio_ema_beta,
        "budget_relative_tolerance": relative_tolerance, "multiplier_error_clip": multiplier_error_clip,
        "use_budget_advantage": use_budget_advantage, "update_multiplier": update_multiplier,
        "validation_enabled": validation_enabled, "validation_trigger": validation_trigger,
        "checkpoint_mode": checkpoint_mode, "validation_every_episodes": validation_every,
        "checkpoint_every_episodes": checkpoint_every, "best_train_mean_VAoI": best_train_vaoi,
        "best_validation_mean_VAoI": best_nn_vaoi,
        "per_node_cap_multiplier": float(constraints.get("per_node_cap_multiplier", 1.5)),
    }
    with (output_directory / "model_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    return output_directory
