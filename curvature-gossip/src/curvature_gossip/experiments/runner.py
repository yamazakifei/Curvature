"""运行配对外生随机性的多策略实验，持久化单次结果并计算跨种子置信区间。"""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import networkx as nx
import numpy as np
import yaml

from ..channel import ChannelParameters, PropagationModel
from ..config import ProjectConfig, load_config
from ..curvature import (
    CurvatureResult, DistributedAF3Curvature, GlobalAF3Curvature, GlobalORCCurvature,
    bottleneck_importance, canonical_edge,
)
from ..policies import create_policy
from ..random_streams import make_rng
from ..simulator import GossipSimulator, SimulationParameters
from ..topology import get_topology_generator
from ..topology.base import Topology


@dataclass(frozen=True)
class ExperimentResult:
    experiment_directory: Path
    run_summaries: Sequence[Mapping[str, Any]]
    aggregate_rows: Sequence[Mapping[str, Any]]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames=None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


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


def _curvature_provider(config: Mapping[str, Any]):
    method = config.get("method", "global_orc")
    if method == "global_orc":
        return GlobalORCCurvature(alpha=float(config.get("alpha", 0.5)))
    if method in ("af3", "global_af3"):
        return GlobalAF3Curvature()
    if method == "distributed_af3":
        return DistributedAF3Curvature()
    raise ValueError("unknown curvature method: {}".format(method))


def _policy_configs(raw: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    policies = raw.get("policies")
    if not isinstance(policies, list) or not policies:
        raise ValueError("configuration must contain a nonempty policies list")
    names = []
    for policy in policies:
        if not isinstance(policy, dict) or not isinstance(policy.get("name"), str):
            raise ValueError("each policy requires a string name")
        if policy["name"] in names:
            raise ValueError("policy names must be unique")
        names.append(policy["name"])
    for policy in policies:
        if policy.get("type", policy["name"]) == "matched_rate_random":
            reference = policy.get("params", {}).get("reference_policy")
            if not isinstance(reference, str) or reference not in names:
                raise ValueError("matched_rate_random requires params.reference_policy naming a configured policy")
            if reference == policy["name"]:
                raise ValueError("matched_rate_random cannot reference itself")
    return policies


def _run_directory(
    experiment_directory: Path,
    topology_seed: int,
    channel_seed: int,
    update_seed: int,
    policy_name: str,
    single_exogenous_pair: bool,
) -> Path:
    base = experiment_directory / str(topology_seed)
    if not single_exogenous_pair:
        base = base / "channel_{}_update_{}".format(channel_seed, update_seed)
    return base / policy_name


def _save_run(
    run_directory: Path,
    raw_config: Mapping[str, Any],
    summary: Mapping[str, Any],
    simulation_result,
    topology,
    curvature,
    importance: Mapping,
    save_per_slot: bool,
) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    with (run_directory / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(raw_config), stream, sort_keys=False, allow_unicode=True)
    with (run_directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(summary), stream, indent=2, sort_keys=True)
    if save_per_slot:
        _write_csv(run_directory / "per_slot.csv", simulation_result.per_slot)
    delay_rows = [{
        "source": record.source,
        "version": record.version,
        "generation_slot": record.generation_slot,
        "completion_slot": record.completion_slot if record.completed else "",
        "delay": record.delay if record.completed else "",
        "completed": record.completed,
    } for record in simulation_result.dissemination_records]
    _write_csv(
        run_directory / "dissemination_delays.csv", delay_rows,
        ["source", "version", "generation_slot", "completion_slot", "delay", "completed"],
    )
    # 与曲率同一行输出其非负瓶颈重要度，便于复现策略的拓扑输入。
    curvature_rows = [{
        "node_i": i,
        "node_j": j,
        "curvature": value,
        "bottleneck_importance": importance[canonical_edge(i, j)],
    } for (i, j), value in sorted(curvature.edge_values.items())]
    _write_csv(run_directory / "edge_curvature.csv", curvature_rows,
               ["node_i", "node_j", "curvature", "bottleneck_importance"])
    node_rows = []
    for node, (x_position, y_position) in enumerate(topology.positions):
        node_rows.append({
            "node": node, "x": x_position, "y": y_position,
            "label": "" if topology.node_labels is None else int(topology.node_labels[node]),
        })
    _write_csv(run_directory / "topology_nodes.csv", node_rows, ["node", "x", "y", "label"])
    edge_rows = [{
        "node_i": min(i, j), "node_j": max(i, j),
        "is_inter_cluster": bool(
            topology.node_labels is not None and topology.node_labels[i] != topology.node_labels[j]
        ),
    } for i, j in sorted(topology.graph.edges)]
    _write_csv(run_directory / "topology_edges.csv", edge_rows,
               ["node_i", "node_j", "is_inter_cluster"])


def annotate_bottleneck_importance(run_directory: str) -> Path:
    """为已保存运行的 edge_curvature.csv 补写瓶颈重要度列。

    归一化方式从同目录 resolved_config.yaml 读取，因此历史结果与当时
    仿真使用的策略输入保持一致；不会重新生成拓扑或重新计算曲率。
    """
    directory = Path(run_directory)
    curvature_path = directory / "edge_curvature.csv"
    config_path = directory / "resolved_config.yaml"
    nodes_path = directory / "topology_nodes.csv"
    edges_path = directory / "topology_edges.csv"
    required_paths = (curvature_path, config_path, nodes_path, edges_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing saved run files: {}".format(", ".join(missing)))

    with config_path.open("r", encoding="utf-8") as stream:
        resolved_config = yaml.safe_load(stream) or {}
    normalization = resolved_config.get("curvature", {}).get(
        "normalization", "global_negative_max"
    )
    with curvature_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        curvature_rows = list(reader)
    for required_field in ("node_i", "node_j", "curvature"):
        if required_field not in fieldnames:
            raise ValueError("edge_curvature.csv lacks column: {}".format(required_field))
    with nodes_path.open("r", newline="", encoding="utf-8") as stream:
        node_rows = list(csv.DictReader(stream))
    with edges_path.open("r", newline="", encoding="utf-8") as stream:
        edge_rows = list(csv.DictReader(stream))
    if not node_rows:
        raise ValueError("topology_nodes.csv must contain at least one node")

    # 从已导出的节点和边重建最小拓扑，只用于复用统一的归一化函数。
    graph = nx.Graph()
    graph.add_nodes_from(int(row["node"]) for row in node_rows)
    graph.add_edges_from((int(row["node_i"]), int(row["node_j"])) for row in edge_rows)
    n_nodes = max(graph.nodes) + 1
    topology = Topology(np.zeros((n_nodes, 2)), graph, 1.0, None, {})
    edge_values = {
        canonical_edge(int(row["node_i"]), int(row["node_j"])): float(row["curvature"])
        for row in curvature_rows
    }
    curvature = CurvatureResult("exported", edge_values, np.zeros(n_nodes), {})
    importance = bottleneck_importance(topology, curvature, normalization)
    for row in curvature_rows:
        edge = canonical_edge(int(row["node_i"]), int(row["node_j"]))
        row["bottleneck_importance"] = importance[edge]
    if "bottleneck_importance" not in fieldnames:
        fieldnames.append("bottleneck_importance")
    _write_csv(curvature_path, curvature_rows, fieldnames)
    return curvature_path


def _aggregate(run_summaries: Sequence[Mapping[str, Any]]):
    ''' 计算每个策略的平均指标和 95% 置信区间。'''
    key_metrics = [
        "mean_VAoI", "mean_max_VAoI", "p95_max_VAoI", "mean_tail_VAoI",
        "avg_tx_per_slot", "successful_decodes_per_tx", "innovative_entries_per_tx",
        "completed_update_fraction", "mean_dissemination_delay",
        "std_node_activity_ratio", "p95_node_activity_ratio",
        "max_node_activity_ratio", "node_cap_violation_fraction",
        "max_node_cap_violation", "curvature_control_message_count",
        "curvature_control_payload_neighbor_ids",
        "target_tx_ratio", "mean_policy_probability", "actual_tx_ratio",
        "matched_rate_probability",
    ]
    rows = []
    for policy_name in sorted({summary["policy"] for summary in run_summaries}):
        policy_runs = [summary for summary in run_summaries if summary["policy"] == policy_name]
        row = {
            "policy": policy_name,
            "independent_runs": len(policy_runs),
            "topology_count": len({summary["topology_seed"] for summary in policy_runs}),
            "topology_seeds": sorted({int(summary["topology_seed"]) for summary in policy_runs}),
            "channel_seeds": sorted({int(summary["channel_seed"]) for summary in policy_runs}),
            "update_seeds": sorted({int(summary["update_seed"]) for summary in policy_runs}),
        }
        for metric in key_metrics:
            values = np.asarray([
                summary[metric] for summary in policy_runs if summary.get(metric) is not None
            ], dtype=float)
            if not values.size:
                row[metric] = None
                row[metric + "_ci95"] = None
                continue
            row[metric] = float(np.mean(values))
            row[metric + "_ci95"] = (
                float(1.96 * np.std(values, ddof=1) / math.sqrt(values.size))
                if values.size > 1 else 0.0
            )
        rows.append(row)
    return rows


def run_experiment(config_path: str, overwrite: bool = False, progress: bool = True) -> ExperimentResult:
    # 读取配置文件，解析实验参数和拓扑生成器。
    config = load_config(config_path)
    raw = config.raw
    experiment = raw["experiment"]
    policies = _policy_configs(raw)
    master_seed = int(experiment.get("master_seed", 0))
    topology_seeds = list(experiment.get("topology_seeds", [0]))
    channel_seeds = list(experiment.get("channel_seeds", [0]))
    update_seeds = list(experiment.get("update_seeds", [0]))
    slots = int(experiment.get("slots", 200))
    warmup = int(experiment.get("warmup_slots", 0))
    trace_stride = int(experiment.get("trace_stride", 1))
    update_probability = float(raw.get("source", {}).get("update_probability", 0.05))
    output = raw.get("output", {})
    node_diagnostics_stride = int(output.get("node_diagnostics_stride", 0))
    constraints = raw.get("constraints", {})
    observation = raw.get("observation", {})
    target_tx_ratio = float(constraints.get("target_tx_ratio", 1.0))
    per_node_cap_multiplier = float(constraints.get("per_node_cap_multiplier", 1.0))
    output_root = Path(output.get("root", "results"))
    experiment_directory = output_root / str(experiment.get("id", "experiment"))
    single_pair = len(channel_seeds) == 1 and len(update_seeds) == 1

    expected_directories = [
        _run_directory(experiment_directory, topology_seed, channel_seed, update_seed,
                       policy["name"], single_pair)
        for topology_seed in topology_seeds
        for channel_seed in channel_seeds
        for update_seed in update_seeds
        for policy in policies
    ]
    existing = [path for path in expected_directories if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("run directory already exists: {}".format(existing[0]))

    # 生成固定拓扑、计算曲率、并对每个策略运行模拟，保存单次结果和汇总。
    run_summaries = []
    curvature_config = raw.get("curvature", {})
    channel_parameters = ChannelParameters.from_mapping(raw.get("channel", {}))
    simulation_parameters = SimulationParameters(
        slots,
        update_probability,
        warmup,
        trace_stride,
        node_diagnostics_stride,
        target_tx_ratio,
        per_node_cap_multiplier,
        float(observation.get("congestion_ewma_beta", 0.8)),
        float(observation.get("congestion_feature_scale", 5.0)),
    )
    generator = get_topology_generator(config.topology.type)

    for topology_seed in topology_seeds:
        topology = generator.generate(
            make_rng(master_seed, "topology", topology_seed), config.topology.params
        )
        # 固定拓扑只在这里计算一次曲率，随后全部策略共享结果。
        curvature = _curvature_provider(curvature_config).compute(topology)
        importance = bottleneck_importance(
            topology, curvature, curvature_config.get("normalization", "global_negative_max")
        )
        for channel_seed in channel_seeds:
            propagation = PropagationModel(
                topology.positions,
                channel_parameters,
                make_rng(master_seed, "shadowing", topology_seed, channel_seed),
            )
            for update_seed in update_seeds:
                node_diagnostic_rows = []
                completed_policy_summaries = {}
                # 依次创建策略，随后运行模拟器、保存单次结果，并将汇总添加到列表中。
                for policy_config in policies:
                    policy_name = policy_config["name"]
                    policy_type = policy_config.get("type", policy_name)
                    policy_params = dict(policy_config.get("params", {}))
                    matched_reference = None
                    matched_probability = None
                    if policy_type == "matched_rate_random":
                        matched_reference = policy_params.pop("reference_policy", None)
                        if matched_reference not in completed_policy_summaries:
                            raise ValueError(
                                "matched_rate_random policy '{}' must follow its reference policy '{}'"
                                .format(policy_name, matched_reference)
                            )
                        # Match the reference policy's realized attempt rate, not its target b.
                        matched_probability = float(
                            completed_policy_summaries[matched_reference]["actual_tx_ratio"]
                        )
                        policy_type = "random"
                        policy_params = {"tx_probability": matched_probability}
                    policy = create_policy(policy_type, policy_params)
                    simulator = GossipSimulator(
                        topology, curvature, importance, propagation, policy, simulation_parameters,
                        make_rng(master_seed, "updates", topology_seed, update_seed),
                        make_rng(master_seed, "fading", topology_seed, channel_seed, update_seed),
                        make_rng(master_seed, "policy", topology_seed, channel_seed, update_seed, policy_name),
                    )
                    if progress:
                        print("Running topology={} channel={} update={} policy={}".format(
                            topology_seed, channel_seed, update_seed, policy_name
                        ))
                    # 执行策略
                    result = simulator.run()
                    for diagnostic in result.node_diagnostics:
                        node_diagnostic_rows.append({
                            "topology_seed": topology_seed,
                            "channel_seed": channel_seed,
                            "update_seed": update_seed,
                            "policy": policy_name,
                            "slot": diagnostic["slot"],
                            "node": diagnostic["node"],
                            "fresh_score": diagnostic["fresh_score"],
                            "orc_score": diagnostic["orc_score"],
                            "base_probability": diagnostic["base_probability"],
                            "backoff_factor": diagnostic["backoff_factor"],
                            "final_probability": diagnostic["final_probability"],
                        })
                    summary = dict(result.summary)
                    summary.update({
                        "policy": policy_name, "topology_seed": topology_seed,
                        "channel_seed": channel_seed, "update_seed": update_seed,
                        "policy_type": policy_config.get("type", policy_name),
                        "matched_rate_reference_policy": matched_reference,
                        "matched_rate_probability": matched_probability,
                        "curvature_method": curvature.method,
                        "curvature_control_message_count": curvature.metadata.get(
                            "control_message_count", 0
                        ),
                        "curvature_control_payload_neighbor_ids": curvature.metadata.get(
                            "control_payload_neighbor_ids", 0
                        ),
                    })
                    run_directory = _run_directory(
                        experiment_directory, topology_seed, channel_seed, update_seed,
                        policy_name, single_pair,
                    )
                    _save_run(
                        run_directory, raw, summary, result, topology, curvature, importance,
                        bool(output.get("save_per_slot", True)),
                    )
                    run_summaries.append(summary)
                    completed_policy_summaries[policy_name] = summary
                    close_policy = getattr(policy, "close", None)
                    if callable(close_policy):
                        close_policy()

                if node_diagnostic_rows:
                    # 单一 channel/update 组合时文件直接位于 results/<id>/<topology_seed>/。
                    diagnostic_directory = _run_directory(
                        experiment_directory,
                        topology_seed,
                        channel_seed,
                        update_seed,
                        policies[0]["name"],
                        single_pair,
                    ).parent
                    diagnostic_directory.mkdir(parents=True, exist_ok=True)
                    _write_csv(
                        diagnostic_directory / "node_policy_diagnostics.csv",
                        node_diagnostic_rows,
                        [
                            "topology_seed", "channel_seed", "update_seed", "policy",
                            "slot", "node", "fresh_score", "orc_score",
                            "base_probability", "backoff_factor", "final_probability",
                        ],
                    )

    # 汇总所有策略的单次结果，保存 CSV 和 JSON，并根据配置生成绘图。
    aggregate_rows = _aggregate(run_summaries)
    experiment_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(experiment_directory / "policy_comparison.csv", aggregate_rows)
    with (experiment_directory / "policy_comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(aggregate_rows), stream, indent=2, sort_keys=True)
    if bool(output.get("make_plots", False)):
        from ..plotting import plot_experiment_results
        plot_experiment_results(
            str(experiment_directory), int(output.get("smoothing_window", 10))
        )
    if progress:
        headers = ["policy", "mean_VAoI", "mean_max_VAoI", "p95_max_VAoI",
                   "avg_tx_per_slot", "successful_decodes_per_tx"]
        print(" | ".join(headers))
        for row in aggregate_rows:
            print(" | ".join(
                str(row[key]) if key == "policy" else "{:.6g}".format(row[key])
                for key in headers
            ))
    return ExperimentResult(experiment_directory, tuple(run_summaries), tuple(aggregate_rows))
