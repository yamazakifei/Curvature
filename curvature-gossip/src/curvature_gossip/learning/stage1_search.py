"""执行 curvature-only Stage-1 的 ``b`` 粗到细搜索和 beta0 离线校准。

搜索复用项目现有 topology、curvature、channel 和 GossipSimulator。候选只改变
静态基础概率公式，所有外生随机流按固定 namespace/index 重建，从而实现配对比较。
"""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from ..channel import ChannelParameters, PropagationModel
from ..curvature import bottleneck_importance
from ..experiments.runner import _curvature_provider
from ..policies.random_policy import UniformRandomPolicy
from ..random_streams import make_rng
from ..simulator import GossipSimulator, SimulationParameters
from ..topology import get_topology_generator


@dataclass(frozen=True)
class CalibrationResult:
    center: float
    score_count: int
    score_mean: float
    score_std: float
    score_min: float
    score_max: float
    candidates: Tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class SearchScenario:
    topology_index: int
    dynamic_repeat_index: int
    topology_seed_label: int


def _sigmoid(value):
    values = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _probability(value: Any, name: str, strict: bool = True) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0 or (strict and value == 0.0):
        raise ValueError("{} must be in (0, 1]".format(name))
    return value


def coarse_base_candidates(max_tx_ratio: float, config: Mapping[str, Any]) -> Tuple[float, ...]:
    """由 Bmax 生成去重、排序且不越界的 coarse b 候选。"""
    bmax = _probability(max_tx_ratio, "max_tx_ratio")
    count = int(config.get("coarse_count", 7))
    low = float(config.get("min_fraction_of_max", 0.25))
    high = float(config.get("max_fraction_of_max", 1.0))
    if count < 1 or not np.isfinite(low) or not np.isfinite(high) or not 0.0 < low <= high:
        raise ValueError("base_candidates coarse range is invalid")
    values = np.linspace(bmax * low, bmax * high, count)
    return tuple(sorted({float(np.clip(value, np.finfo(float).eps, bmax)) for value in values}))


def alpha_candidates(config: Mapping[str, Any], fallback: Any) -> Tuple[float, ...]:
    """解析显式 alpha 列表；省略时只保留兼容回退 alpha。"""
    values = config.get("alpha_candidates", [fallback])
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("alpha_candidates must be a nonempty list")
    output = []
    for value in values:
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("alpha_candidates must be finite and nonnegative")
        if not any(math.isclose(value, previous, rel_tol=0.0, abs_tol=1e-12) for previous in output):
            output.append(value)
    return tuple(output)


def refine_base_candidates(coarse: Sequence[float], best_index: int, bmax: float, count: int) -> Tuple[float, ...]:
    """围绕某个 alpha 的 coarse 最优点生成不越界的局部 b 候选。"""
    coarse = tuple(sorted(float(value) for value in coarse))
    if not coarse or not 0.0 < bmax <= 1.0 or count < 1:
        raise ValueError("refine inputs are invalid")
    best_index = int(best_index)
    if best_index < 0 or best_index >= len(coarse):
        raise IndexError("best_index is outside coarse candidates")
    if len(coarse) == 1:
        left, right = max(np.finfo(float).eps, bmax * 0.25), bmax
    elif best_index == 0:
        left, right = max(np.finfo(float).eps, bmax * 0.25), coarse[1]
    elif best_index == len(coarse) - 1:
        left, right = coarse[-2], bmax
    else:
        left, right = coarse[best_index - 1], coarse[best_index + 1]
    left, right = max(np.finfo(float).eps, left), min(bmax, right)
    return tuple(sorted({float(value) for value in np.linspace(left, right, count) if value <= bmax + 1e-12}))


def calibrate_intercept(scores: Sequence[float], base_tx_ratio: float, alpha: float, center: float,
                        tolerance: float = 1e-6, max_iterations: int = 100) -> Dict[str, Any]:
    """用 pooled 曲率分数二分求一个公共 beta0，并返回校准诊断。"""
    values = np.asarray(scores, dtype=float).reshape(-1)
    target = _probability(base_tx_ratio, "base_tx_ratio")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("calibration scores must be nonempty and finite")
    tolerance = float(tolerance)
    max_iterations = int(max_iterations)
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("invalid intercept calibration settings")

    def mean_probability(intercept):
        return float(np.mean(_sigmoid(intercept + float(alpha) * (values - float(center)))))

    low, high = -8.0, 8.0
    while mean_probability(low) > target and low > -128.0:
        low *= 2.0
    while mean_probability(high) < target and high < 128.0:
        high *= 2.0
    converged = False
    iterations = 0
    intercept = 0.0
    for iterations in range(1, max_iterations + 1):
        intercept = (low + high) / 2.0
        actual = mean_probability(intercept)
        if abs(actual - target) <= tolerance:
            converged = True
            break
        if actual < target:
            low = intercept
        else:
            high = intercept
    actual = mean_probability(intercept)
    return {
        "base_tx_ratio": target,
        "alpha": float(alpha),
        "calibrated_intercept": float(intercept),
        "target_probability_mean": target,
        "calibration_probability_mean": actual,
        "calibration_error": actual - target,
        "iterations": int(iterations),
        "converged": bool(converged or abs(actual - target) <= tolerance),
        "tolerance": tolerance,
    }


def _topology_params(raw: Mapping[str, Any], node_count: int = None) -> Dict[str, Any]:
    params = dict(raw["topology"].get("params", {}))
    if node_count is not None:
        params["n_nodes"] = int(node_count)
        if "cluster_sizes" in params:
            params["cluster_sizes"] = [int(node_count) // 2, int(node_count) - int(node_count) // 2]
    return params


def _build_topology(raw: Mapping[str, Any], topology_index: int, pool: str):
    master_seed = int(raw["experiment"].get("master_seed", 0))
    generator = get_topology_generator(raw["topology"]["type"])
    topology = generator.generate(
        make_rng(master_seed, pool, "topology", int(topology_index)),
        _topology_params(raw),
    )
    curvature_config = raw.get("curvature", {})
    curvature = _curvature_provider(curvature_config).compute(topology)
    importance = bottleneck_importance(
        topology, curvature, curvature_config.get("normalization", "local_degree_bound")
    )
    scores = np.asarray([
        max([importance[tuple(sorted((node, neighbor)))] for neighbor in topology.graph.neighbors(node)] or [0.0])
        for node in topology.graph.nodes
    ], dtype=float)
    return topology, curvature, importance, scores


def _build_search_simulator(raw: Mapping[str, Any], scenario: SearchScenario, max_tx_ratio: float):
    """用固定外生随机流构造一个候选共享的仿真场景。"""
    master_seed = int(raw["experiment"].get("master_seed", 0))
    topology, curvature, importance, scores = _build_topology(
        raw, scenario.topology_seed_label, "stage1_search"
    )
    repeat = int(scenario.dynamic_repeat_index)
    propagation = PropagationModel(
        topology.positions,
        ChannelParameters.from_mapping(raw.get("channel", {})),
        make_rng(master_seed, "stage1_search", "shadowing", scenario.topology_index, repeat),
    )
    evaluation = dict(raw.get("actor", {}).get("curvature", {}).get("base_search", {}).get("evaluation", {}))
    slots = int(evaluation.get("slots", raw.get("experiment", {}).get("slots", 200)))
    source_probability = float(raw.get("source", {}).get("update_probability", 0.05))
    constraints = raw.get("constraints", {})
    observation = raw.get("observation", {})
    simulator = GossipSimulator(
        topology, curvature, importance, propagation, UniformRandomPolicy(0.0),
        SimulationParameters(
            slots=slots,
            update_probability=source_probability,
            target_tx_ratio=float(max_tx_ratio),
            per_node_cap_multiplier=float(constraints.get("per_node_cap_multiplier", 1.5)),
            congestion_ewma_beta=float(observation.get("congestion_ewma_beta", 0.8)),
            congestion_feature_scale=float(observation.get("congestion_feature_scale", 5.0)),
        ),
        make_rng(master_seed, "stage1_search", "source_updates", scenario.topology_index, repeat),
        make_rng(master_seed, "stage1_search", "fading", scenario.topology_index, repeat),
        make_rng(master_seed, "stage1_search", "actions", scenario.topology_index, repeat),
    )
    return simulator, scores


def _evaluate_candidate(raw: Mapping[str, Any], scenario_set: Sequence[SearchScenario], candidate: Mapping[str, Any], max_tx_ratio: float) -> Dict[str, Any]:
    """在给定配对场景上评估一个静态 curvature-only 候选。"""
    alpha = float(candidate["alpha"])
    intercept = float(candidate["calibrated_intercept"])
    center = float(candidate["center"])
    vaoi, tail, actual, probabilities = [], [], [], []
    per_scenario = []
    for scenario in scenario_set:
        simulator, scores = _build_search_simulator(raw, scenario, max_tx_ratio)
        static_probabilities = np.asarray(_sigmoid(intercept + alpha * (scores - center)), dtype=float)
        for slot in range(simulator.parameters.slots):
            simulator.begin_step(slot)
            actions = simulator.policy_rng.random(simulator.state.n_nodes) < static_probabilities
            simulator.complete_step(actions)
        summary = simulator.metrics.summary()
        mean_probability = float(np.mean(static_probabilities))
        actual_ratio = float(summary["avg_tx_per_slot"] / simulator.state.n_nodes)
        vaoi.append(float(summary["mean_VAoI"]))
        tail.append(float(summary["mean_tail_VAoI"]))
        actual.append(actual_ratio)
        probabilities.append(mean_probability)
        per_scenario.append({
            "phase": candidate["phase"],
            "base_tx_ratio": float(candidate["base_tx_ratio"]),
            "alpha": alpha,
            "calibrated_intercept": intercept,
            "topology_index": int(scenario.topology_index),
            "dynamic_repeat_index": int(scenario.dynamic_repeat_index),
            "search_probability_mean": mean_probability,
            "actual_tx_ratio": actual_ratio,
            "mean_VAoI": float(summary["mean_VAoI"]),
            "mean_tail_VAoI": float(summary["mean_tail_VAoI"]),
        })
    result = dict(candidate)
    result.update({
        "search_probability_mean": float(np.mean(probabilities)),
        "actual_tx_ratio": float(np.mean(actual)),
        "mean_VAoI": float(np.mean(vaoi)),
        "mean_tail_VAoI": float(np.mean(tail)),
        "constraint_feasible": bool(np.mean(probabilities) <= max_tx_ratio + float(candidate["max_probability_tolerance"])),
        "per_scenario": per_scenario,
    })
    return result


def _select_candidate(results: Sequence[Mapping[str, Any]], tie_relative_tolerance: float) -> Mapping[str, Any]:
    feasible = [row for row in results if row["constraint_feasible"]]
    if not feasible:
        raise ValueError("Stage-1 search found no candidate satisfying Bmax")
    metric_best = min(float(row["mean_VAoI"]) for row in feasible)
    tied = [row for row in feasible if abs(float(row["mean_VAoI"]) - metric_best) <= float(tie_relative_tolerance) * max(abs(metric_best), 1e-12)]
    return min(tied, key=lambda row: (float(row["search_probability_mean"]), float(row["base_tx_ratio"]), int(row["alpha_order"])))


def _write_search_outputs(output_directory: Path, calibration: Mapping[str, Any], results: Sequence[Mapping[str, Any]], per_scenario: Sequence[Mapping[str, Any]], selected: Mapping[str, Any], center: float) -> None:
    with (output_directory / "stage1_calibration.json").open("w", encoding="utf-8") as stream:
        json.dump(calibration, stream, indent=2, sort_keys=True)
    fields = [
        "phase", "base_tx_ratio", "alpha", "calibrated_intercept", "calibration_probability_mean",
        "search_probability_mean", "actual_tx_ratio", "mean_VAoI", "mean_tail_VAoI",
        "constraint_feasible", "rank", "selected",
    ]
    with (output_directory / "stage1_search_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(sorted(results, key=lambda item: (not item["constraint_feasible"], item["mean_VAoI"])), 1):
            output_row = {field: row.get(field, "") for field in fields}
            output_row["rank"] = rank
            output_row["selected"] = bool(row is selected)
            writer.writerow(output_row)
    scenario_fields = sorted({key for row in per_scenario for key in row})
    with (output_directory / "stage1_search_per_scenario.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scenario_fields)
        writer.writeheader()
        writer.writerows(per_scenario)
    summary = {
        "selected_base_tx_ratio": float(selected["base_tx_ratio"]),
        "selected_alpha": float(selected["alpha"]),
        "selected_intercept": float(selected["calibrated_intercept"]),
        "selected_center": float(center),
        "search_probability_mean": float(selected["search_probability_mean"]),
        "actual_tx_ratio": float(selected["actual_tx_ratio"]),
        "mean_VAoI": float(selected["mean_VAoI"]),
        "mean_tail_VAoI": float(selected["mean_tail_VAoI"]),
        "selection_metric": "mean_VAoI",
        "constraint_status": "feasible" if selected["constraint_feasible"] else "infeasible",
    }
    with (output_directory / "stage1_search_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)


def _print_selected_summary(selected: Mapping[str, Any], bmax: float, output_directory: Path) -> None:
    """在搜索结束时打印可直接用于复现实验的最优 Stage-1 配置。"""
    print(
        "[Stage-1 Search] 搜索完成，最优配置："
        " b={b:.6f}, alpha={alpha:.6f}, beta0={beta0:.6f}, c_kappa={center:.6f};"
        " search_mean_prob={prob:.6f}, actual_tx_ratio={actual:.6f},"
        " mean_VAoI={vaoi:.6f}, feasible={feasible} (Bmax={bmax:.6f})".format(
            b=float(selected["base_tx_ratio"]),
            alpha=float(selected["alpha"]),
            beta0=float(selected["calibrated_intercept"]),
            center=float(selected["center"]),
            prob=float(selected["search_probability_mean"]),
            actual=float(selected["actual_tx_ratio"]),
            vaoi=float(selected["mean_VAoI"]),
            feasible="yes" if selected["constraint_feasible"] else "no",
            bmax=float(bmax),
        ),
        flush=True,
    )
    print("[Stage-1 Search] 结果已保存到: {}".format(output_directory), flush=True)


def run_stage1_search(raw: Mapping[str, Any], output_directory: Path) -> Dict[str, Any]:
    """运行完整 Stage-1 搜索并返回注入 Actor 的冻结参数。"""
    curvature = dict(raw.get("actor", {}).get("curvature", {}))
    search = dict(curvature.get("base_search", {}))
    if not bool(search.get("enabled", False)):
        return {}
    bmax = _probability(raw["constraints"]["max_tx_ratio"], "constraints.max_tx_ratio")
    alpha_values = alpha_candidates(search, curvature.get("alpha_override", curvature.get("alpha_init", 0.0)))
    candidates_config = dict(search.get("base_candidates", {}))
    coarse = coarse_base_candidates(bmax, candidates_config)
    calibration_config = dict(search.get("calibration", {}))
    scores = []
    calibration_count = int(calibration_config.get("topology_count", 32))
    for index in range(calibration_count):
        _, _, _, topology_scores = _build_topology(raw, index, "stage1_calibration")
        scores.extend(topology_scores.tolist())
    center = float(np.mean(scores))
    calibration_entries = []
    for alpha_order, alpha in enumerate(alpha_values):
        for base in coarse:
            entry = calibrate_intercept(
                scores, base, alpha, center,
                tolerance=float(calibration_config.get("intercept_tolerance", 1e-6)),
                max_iterations=int(calibration_config.get("intercept_max_iterations", 100)),
            )
            entry["alpha_order"] = alpha_order
            calibration_entries.append(entry)

    evaluation = dict(search.get("evaluation", {}))
    total_topologies = int(evaluation.get("topology_count", 20))
    total_repeats = int(evaluation.get("dynamic_repeats", 2))
    coarse_topologies = min(total_topologies, int(evaluation.get("coarse_topology_count", total_topologies)))
    coarse_repeats = min(total_repeats, int(evaluation.get("coarse_dynamic_repeats", total_repeats)))
    coarse_scenarios = [SearchScenario(i, repeat, i) for i in range(coarse_topologies) for repeat in range(coarse_repeats)]
    full_scenarios = [SearchScenario(i, repeat, i) for i in range(total_topologies) for repeat in range(total_repeats)]
    tolerance = float(evaluation.get("max_probability_tolerance", 0.005))
    coarse_results = []
    for entry in calibration_entries:
        candidate = dict(entry, phase="coarse", center=center, max_probability_tolerance=tolerance)
        coarse_results.append(_evaluate_candidate(raw, coarse_scenarios, candidate, bmax))

    all_candidates = []
    all_candidates.extend(coarse_results)
    refine_count = int(candidates_config.get("refine_count", 5))
    if bool(candidates_config.get("refine", True)) and bool(candidates_config.get("refine_per_alpha", True)):
        for alpha_order, alpha in enumerate(alpha_values):
            alpha_rows = [row for row in coarse_results if int(row["alpha_order"]) == alpha_order]
            best_row = min(alpha_rows, key=lambda row: (not row["constraint_feasible"], row["mean_VAoI"]))
            best_index = min(range(len(coarse)), key=lambda index: abs(coarse[index] - float(best_row["base_tx_ratio"])))
            for base in refine_base_candidates(coarse, best_index, bmax, refine_count):
                if any(math.isclose(base, float(row["base_tx_ratio"]), rel_tol=0.0, abs_tol=1e-12) and int(row["alpha_order"]) == alpha_order for row in all_candidates):
                    continue
                calibrated = calibrate_intercept(
                    scores, base, alpha, center,
                    tolerance=float(calibration_config.get("intercept_tolerance", 1e-6)),
                    max_iterations=int(calibration_config.get("intercept_max_iterations", 100)),
                )
                all_candidates.append(dict(calibrated, alpha_order=alpha_order, phase="refine", center=center, max_probability_tolerance=tolerance))

    full_results = []
    for candidate in all_candidates:
        full_results.append(_evaluate_candidate(raw, full_scenarios, candidate, bmax))
    selected = _select_candidate(full_results, float(evaluation.get("tie_relative_tolerance", 0.01)))
    per_scenario = [row for candidate in full_results for row in candidate["per_scenario"]]
    calibration = {
        "topology_count": calibration_count,
        "center_statistic": str(calibration_config.get("center_statistic", "mean")),
        "center": center,
        "score_count": len(scores),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "candidates": calibration_entries,
    }
    _write_search_outputs(output_directory, calibration, full_results, per_scenario, selected, center)
    _print_selected_summary(selected, bmax, output_directory)
    return {
        "base_tx_ratio": float(selected["base_tx_ratio"]),
        "alpha_override": float(selected["alpha"]),
        "center": center,
        "calibrated_intercept": float(selected["calibrated_intercept"]),
        "freeze_stage1": True,
        "search_summary": {
            "selected_base_tx_ratio": float(selected["base_tx_ratio"]),
            "selected_alpha": float(selected["alpha"]),
            "selected_intercept": float(selected["calibrated_intercept"]),
            "selected_center": center,
            "search_probability_mean": float(selected["search_probability_mean"]),
            "actual_tx_ratio": float(selected["actual_tx_ratio"]),
            "mean_VAoI": float(selected["mean_VAoI"]),
        },
    }
