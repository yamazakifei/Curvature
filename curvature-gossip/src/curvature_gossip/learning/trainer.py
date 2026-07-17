"""生成在线 rollout，使用双层广播约束训练共享 CTDE-PPO 策略。"""

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
from .features import encode_observations


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


def _build_simulator(
    raw: Mapping,
    episode: int,
    target_tx_ratio: float,
    node_count=None,
    update_probability=None,
):
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
    topology = generator.generate(
        make_rng(master_seed, "nn_topology", topology_seed),
        topology_params,
    )
    curvature_config = raw.get("curvature", {})
    curvature = _curvature_provider(curvature_config).compute(topology)
    importance = bottleneck_importance(
        topology,
        curvature,
        curvature_config.get("normalization", "local_degree_bound"),
    )
    channel = ChannelParameters.from_mapping(raw.get("channel", {}))
    propagation = PropagationModel(
        topology.positions,
        channel,
        make_rng(master_seed, "nn_shadowing", topology_seed),
    )
    constraints = raw.get("constraints", {})
    rollout_slots = int(training.get("rollout_slots", experiment.get("slots", 200)))
    if update_probability is None:
        update_probability = float(raw.get("source", {}).get("update_probability", 0.05))
    parameters = SimulationParameters(
        slots=rollout_slots,
        update_probability=update_probability,
        target_tx_ratio=float(target_tx_ratio),
        per_node_cap_multiplier=float(constraints.get("per_node_cap_multiplier", 1.5)),
    )
    simulator = GossipSimulator(
        topology,
        curvature,
        importance,
        propagation,
        UniformRandomPolicy(0.0),
        parameters,
        make_rng(master_seed, "nn_updates", topology_seed),
        make_rng(master_seed, "nn_fading", topology_seed),
        make_rng(master_seed, "nn_policy", topology_seed),
    )
    return simulator, curvature


def _gae(rewards, values, gamma: float, gae_lambda: float):
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


def _write_history(output_directory: Path, history) -> None:
    with (output_directory / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    with (output_directory / "training_history.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(history), stream, indent=2, sort_keys=True)


def train_ctde(config_path: str):
    """训练模型并把 checkpoint、配置和历史统一保存到 result_NN。"""
    config = load_config(config_path)
    raw = config.raw
    training = raw.get("training", {})
    constraints = raw.get("constraints", {})
    output_root = Path(raw.get("output", {}).get("root", "result_NN"))
    output_directory = output_root / str(raw["experiment"].get("id", "nn_ctde"))
    output_directory.mkdir(parents=True, exist_ok=True)
    with Path(config_path).open("r", encoding="utf-8") as source:
        (output_directory / "training_config.yaml").write_text(
            source.read(), encoding="utf-8"
        )

    seed = int(raw["experiment"].get("master_seed", 0))
    model = CTDEPPO(
        learning_rate=float(training.get("learning_rate", 3e-4)),
        clip_ratio=float(training.get("clip_ratio", 0.2)),
        entropy_coefficient=float(training.get("entropy_coefficient", 0.01)),
        q_min=float(training.get("q_min", 0.001)),
        q_max=float(training.get("q_max", 0.8)),
        seed=seed,
    )
    action_rng = np.random.default_rng(seed + 7919)
    targets = [float(value) for value in training.get(
        "target_tx_ratios", [constraints.get("target_tx_ratio", 0.1)]
    )]
    if not targets or any(not 0.0 < value <= 1.0 for value in targets):
        raise ValueError("training.target_tx_ratios must contain values in (0, 1]")
    lagrange_by_target = {
        value: float(training.get("initial_lagrange", 0.0)) for value in targets
    }
    node_counts = list(training.get("node_counts", [None])) or [None]
    update_probabilities = list(training.get("update_probabilities", [None])) or [None]
    # 笛卡尔积避免目标预算与负载或节点规模被固定配对。
    training_cases = [
        (target_value, node_count, update_probability)
        for node_count in node_counts
        for update_probability in update_probabilities
        for target_value in targets
    ]
    lagrange_lr = float(training.get("lagrange_learning_rate", 0.5))
    local_penalty = float(training.get("local_debt_penalty", 0.02))
    history = []

    try:
        for episode in range(int(training.get("episodes", 20))):
            target, node_count, update_probability = training_cases[
                episode % len(training_cases)
            ]
            lagrange = lagrange_by_target[target]
            simulator, curvature = _build_simulator(
                raw, episode, target, node_count, update_probability
            )
            encoded_steps = []
            actions_steps = []
            old_log_steps = []
            value_steps = []
            global_states = []
            rewards = []
            costs = []
            vaoi_values = []

            last_tx_ratio = 0.0
            for slot in range(simulator.parameters.slots):
                observations = simulator.begin_step(slot)
                encoded = encode_observations(observations, target)
                global_state = simulator.centralized_state(last_tx_ratio)
                value = float(model.value(global_state)[0])
                actions, _, old_log_probabilities = model.act(encoded, action_rng)
                debt_action_cost = float(np.mean([
                    observation.broadcast_debt * action
                    for observation, action in zip(observations, actions)
                ]))
                outcome = simulator.complete_step(actions.astype(bool))
                constrained_reward = (
                    outcome.reward
                    - lagrange * (outcome.transmission_cost - target)
                    - local_penalty * debt_action_cost
                )
                encoded_steps.append(encoded)
                actions_steps.append(actions)
                old_log_steps.append(old_log_probabilities)
                value_steps.append(value)
                global_states.append(global_state)
                rewards.append(constrained_reward)
                costs.append(outcome.transmission_cost)
                vaoi_values.append(outcome.mean_vaoi)
                last_tx_ratio = outcome.transmission_cost

            advantages, returns = _gae(
                rewards,
                value_steps,
                float(training.get("gamma", 0.99)),
                float(training.get("gae_lambda", 0.95)),
            )
            actor_advantages = np.concatenate([
                np.full(step.node_features.shape[0], advantages[index], dtype=np.float32)
                for index, step in enumerate(encoded_steps)
            ])
            actor_advantages = (
                actor_advantages - np.mean(actor_advantages)
            ) / (np.std(actor_advantages) + 1e-6)
            actor_batch = {
                "node_features": np.concatenate([step.node_features for step in encoded_steps]),
                "edge_features": np.concatenate([step.edge_features for step in encoded_steps]),
                "edge_mask": np.concatenate([step.edge_mask for step in encoded_steps]),
                "actions": np.concatenate(actions_steps),
                "old_log_probabilities": np.concatenate(old_log_steps),
                "advantages": actor_advantages,
            }
            critic_batch = {
                "global_states": np.asarray(global_states, dtype=np.float32),
                "returns": returns,
            }
            losses = model.update(
                actor_batch,
                critic_batch,
                epochs=int(training.get("ppo_epochs", 4)),
            )
            average_cost = float(np.mean(costs))
            lagrange = max(0.0, lagrange + lagrange_lr * (average_cost - target))
            lagrange_by_target[target] = lagrange
            summary = simulator.metrics.summary()
            row = {
                "episode": episode,
                "mean_VAoI": float(np.mean(vaoi_values)),
                "avg_tx_ratio": average_cost,
                "target_tx_ratio": target,
                "n_nodes": simulator.state.n_nodes,
                "update_probability": simulator.parameters.update_probability,
                "lagrange": lagrange,
                "actor_loss": losses["actor_loss"],
                "critic_loss": losses["critic_loss"],
                "max_node_activity_ratio": summary["max_node_activity_ratio"],
                "node_cap_violation_fraction": summary["node_cap_violation_fraction"],
                "curvature_control_messages": curvature.metadata.get(
                    "control_message_count", 0
                ),
            }
            history.append(row)
            _write_history(output_directory, history)
            model.save(str(output_directory / "checkpoints" / "model"))
            print(
                "episode={episode} VAoI={vaoi:.4f} tx_ratio={tx:.4f} "
                "lambda={lagrange:.4f}".format(
                    episode=episode,
                    vaoi=row["mean_VAoI"],
                    tx=average_cost,
                    lagrange=lagrange,
                )
            )
    finally:
        model.close()

    metadata = {
        "checkpoint": str(output_directory / "checkpoints" / "model"),
        "episodes": len(history),
        "target_tx_ratios": targets,
        "node_counts": node_counts,
        "update_probabilities": update_probabilities,
        "per_node_cap_multiplier": float(
            constraints.get("per_node_cap_multiplier", 1.5)
        ),
    }
    with (output_directory / "model_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    return output_directory
