"""执行 curvature-only Stage-1 的 ``b`` 粗到细搜索和 beta0 离线校准。

搜索复用项目现有 topology、curvature、channel 和 GossipSimulator。候选只改变
静态基础概率公式，所有外生随机流按固定 namespace/index 重建，从而实现配对比较。
"""

import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
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


@dataclass(frozen=True)
class SearchTopologyCache:
    """候选之间可安全共享的静态拓扑、曲率和节点分数。"""

    topology_index: int
    topology: Any
    curvature: Any
    importance: Mapping
    scores: np.ndarray


@dataclass(frozen=True)
class SearchScenarioCache:
    """一个动态 repeat 的静态场景；simulator 本身仍然每次全新创建。"""

    topology_index: int
    dynamic_repeat_index: int
    topology_cache: SearchTopologyCache
    propagation: PropagationModel


# Windows spawn worker 时由 initializer 注入，避免每个候选重复 pickle 大型场景对象。
_SEARCH_WORKER_CONTEXT = None


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


def resolve_center_mode(curvature: Mapping[str, Any]) -> str:
    """解析中心项模式，并为旧 YAML 保留 ``auto``/数值中心的语义。"""
    mode = curvature.get("center_mode")
    if mode is None:
        center = curvature.get("center", "auto")
        if isinstance(center, (int, float)) and not isinstance(center, bool):
            mode = "fixed"
        elif center in (None, "auto"):
            mode = "calibration_mean"
        else:
            raise ValueError("actor.curvature.center must be a number, null, or 'auto'")
    mode = str(mode)
    if mode not in ("none", "fixed", "calibration_mean"):
        raise ValueError("actor.curvature.center_mode must be none, fixed, or calibration_mean")
    if mode == "fixed":
        center = curvature.get("center")
        if isinstance(center, bool) or center is None:
            raise ValueError("center_mode=fixed requires a finite numeric center")
        if not np.isfinite(float(center)):
            raise ValueError("actor.curvature.center must be finite")
    if mode == "none" and "center" in curvature and float(curvature.get("center", 0.0)) != 0.0:
        raise ValueError("center_mode=none requires center=0.0")
    return mode


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


def _build_topology_cache(raw: Mapping[str, Any], topology_index: int, pool: str) -> SearchTopologyCache:
    """只计算一次候选共享的 topology、AF3、importance 和 score 向量。"""
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
    return SearchTopologyCache(
        topology_index=int(topology_index), topology=topology, curvature=curvature,
        importance=importance, scores=scores,
    )


def _build_topology(raw: Mapping[str, Any], topology_index: int, pool: str):
    """保留旧内部调用的 tuple 返回接口。"""
    cached = _build_topology_cache(raw, topology_index, pool)
    return cached.topology, cached.curvature, cached.importance, cached.scores


def _build_search_scenario_cache(
    raw: Mapping[str, Any], topology_cache: SearchTopologyCache,
    dynamic_repeat_index: int,
) -> SearchScenarioCache:
    """为一个 repeat 构造静态传播矩阵；不把可变 simulator 放入缓存。"""
    master_seed = int(raw["experiment"].get("master_seed", 0))
    propagation = PropagationModel(
        topology_cache.topology.positions,
        ChannelParameters.from_mapping(raw.get("channel", {})),
        make_rng(
            master_seed, "stage1_search", "shadowing",
            topology_cache.topology_index, int(dynamic_repeat_index),
        ),
    )
    return SearchScenarioCache(
        topology_index=topology_cache.topology_index,
        dynamic_repeat_index=int(dynamic_repeat_index),
        topology_cache=topology_cache,
        propagation=propagation,
    )


def _build_search_simulator_from_cache(
    raw: Mapping[str, Any], scenario: SearchScenarioCache,
    max_tx_ratio: float, slots: int,
):
    """从静态 cache 创建全新的动态 simulator 和 RNG。"""
    master_seed = int(raw["experiment"].get("master_seed", 0))
    topology = scenario.topology_cache.topology
    curvature = scenario.topology_cache.curvature
    importance = scenario.topology_cache.importance
    repeat = int(scenario.dynamic_repeat_index)
    evaluation = dict(raw.get("actor", {}).get("curvature", {}).get("base_search", {}).get("evaluation", {}))
    source_probability = float(raw.get("source", {}).get("update_probability", 0.05))
    constraints = raw.get("constraints", {})
    observation = raw.get("observation", {})
    return GossipSimulator(
        topology, curvature, importance, scenario.propagation, UniformRandomPolicy(0.0),
        SimulationParameters(
            slots=int(slots),
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


def _build_search_simulator(raw: Mapping[str, Any], scenario: SearchScenario, max_tx_ratio: float, slots: int = None):
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
    slots = int(slots if slots is not None else evaluation.get("slots", raw.get("experiment", {}).get("slots", 200)))
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


def _evaluate_candidate(
    raw: Mapping[str, Any], scenario_set: Sequence[Any], candidate: Mapping[str, Any],
    max_tx_ratio: float, slots: int,
) -> Dict[str, Any]:
    """在给定配对场景上评估一个静态 curvature-only 候选。"""
    alpha = float(candidate["alpha"])
    intercept = float(candidate["calibrated_intercept"])
    center = float(candidate["center"])
    vaoi, tail, actual, probabilities = [], [], [], []
    per_scenario = []
    for scenario in scenario_set:
        if isinstance(scenario, SearchScenarioCache):
            simulator = _build_search_simulator_from_cache(raw, scenario, max_tx_ratio, slots)
            scores = scenario.topology_cache.scores
        else:
            simulator, scores = _build_search_simulator(raw, scenario, max_tx_ratio, slots)
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
            "candidate_id": int(candidate.get("candidate_id", -1)),
            "phase": candidate["phase"],
            "evaluation_phase": candidate.get("evaluation_phase", "full"),
            "base_tx_ratio": float(candidate["base_tx_ratio"]),
            "alpha": alpha,
            "calibrated_intercept": intercept,
            "topology_index": int(scenario.topology_index),
            "dynamic_repeat_index": int(scenario.dynamic_repeat_index),
            "search_probability_mean": mean_probability,
            "actual_tx_ratio": actual_ratio,
            "mean_VAoI": float(summary["mean_VAoI"]),
            "mean_tail_VAoI": float(summary["mean_tail_VAoI"]),
            "evaluated_on_full": bool(candidate.get("evaluated_on_full", True)),
            "phase_slots": int(slots),
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


def _init_search_worker(raw, scenario_set, max_tx_ratio, slots):
    """初始化进程级场景 cache；任务本身只携带一个候选。"""
    global _SEARCH_WORKER_CONTEXT
    _SEARCH_WORKER_CONTEXT = (raw, tuple(scenario_set), float(max_tx_ratio), int(slots))


def _evaluate_candidate_worker(candidate):
    """候选级 worker，并把上下文信息附加到异常，便于定位失败项。"""
    raw, scenario_set, max_tx_ratio, slots = _SEARCH_WORKER_CONTEXT
    try:
        return _evaluate_candidate(raw, scenario_set, candidate, max_tx_ratio, slots)
    except Exception as exc:
        raise RuntimeError(
            "Stage-1 candidate failed: phase={}, candidate_id={}, b={}, alpha={}".format(
                candidate.get("phase"), candidate.get("candidate_id"),
                candidate.get("base_tx_ratio"), candidate.get("alpha"),
            )
        ) from exc


def _parallel_worker_count(parallel: Mapping[str, Any], candidate_count: int) -> int:
    """解析实际 worker 数，避免 runtime JSON 只记录配置值 0。"""
    if not bool(parallel.get("enabled", False)) or candidate_count < 2:
        return 1
    configured = int(parallel.get("workers", 1))
    if configured == 1:
        return 1
    return min(os.cpu_count() or 1, candidate_count) if configured == 0 else min(configured, candidate_count)


def _evaluate_candidates(
    raw: Mapping[str, Any], scenario_set: Sequence[Any], candidates: Sequence[Mapping[str, Any]],
    max_tx_ratio: float, slots: int, parallel: Mapping[str, Any], cache_enabled: bool,
) -> List[Dict[str, Any]]:
    """按 candidate_id 稳定返回结果；并行仅跨候选，不拆散一个 rollout。"""
    prepared = [
        dict(candidate, candidate_id=candidate.get("candidate_id", index))
        for index, candidate in enumerate(candidates)
    ]
    enabled = bool(parallel.get("enabled", False))
    backend = str(parallel.get("backend", "process"))
    workers = int(parallel.get("workers", 1))
    if workers < 0:
        raise ValueError("parallel.workers must be nonnegative")
    if enabled and backend != "process":
        raise ValueError("parallel.backend must be 'process'")
    if not prepared:
        return []
    if enabled and workers != 1:
        worker_count = _parallel_worker_count(parallel, len(prepared))
        if worker_count > 1:
            chunksize = max(1, int(parallel.get("chunksize", 1)))
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_search_worker,
                initargs=(raw, tuple(scenario_set), max_tx_ratio, slots),
            ) as executor:
                return list(executor.map(_evaluate_candidate_worker, prepared, chunksize=chunksize))
    # Serial path is also the compatibility path for old YAML files.
    return [_evaluate_candidate(raw, scenario_set, candidate, max_tx_ratio, slots) for candidate in prepared]


def _select_candidate(results: Sequence[Mapping[str, Any]], tie_relative_tolerance: float) -> Mapping[str, Any]:
    feasible = [row for row in results if row["constraint_feasible"]]
    if not feasible:
        raise ValueError("Stage-1 search found no candidate satisfying Bmax")
    metric_best = min(float(row["mean_VAoI"]) for row in feasible)
    tied = [row for row in feasible if abs(float(row["mean_VAoI"]) - metric_best) <= float(tie_relative_tolerance) * max(abs(metric_best), 1e-12)]
    return min(tied, key=lambda row: (float(row["search_probability_mean"]), float(row["base_tx_ratio"]), int(row["alpha_order"])))


def _write_search_outputs(
    output_directory: Path, calibration: Mapping[str, Any], results: Sequence[Mapping[str, Any]],
    per_scenario: Sequence[Mapping[str, Any]], selected: Mapping[str, Any], center: float,
    center_mode: str, coarse_candidate_count: int, full_candidate_count: int,
) -> None:
    with (output_directory / "stage1_calibration.json").open("w", encoding="utf-8") as stream:
        json.dump(calibration, stream, indent=2, sort_keys=True)
    fields = [
        "candidate_id", "phase", "evaluation_phase", "base_tx_ratio", "alpha", "calibrated_intercept", "calibration_probability_mean",
        "search_probability_mean", "actual_tx_ratio", "mean_VAoI", "mean_tail_VAoI",
        "constraint_feasible", "coarse_rank_within_alpha", "evaluated_on_full",
        "pruned_after_coarse", "phase_slots", "rank", "selected",
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
        "selected_center_mode": center_mode,
        "selected_center": float(center),
        "search_probability_mean": float(selected["search_probability_mean"]),
        "actual_tx_ratio": float(selected["actual_tx_ratio"]),
        "mean_VAoI": float(selected["mean_VAoI"]),
        "mean_tail_VAoI": float(selected["mean_tail_VAoI"]),
        "selection_metric": "mean_VAoI",
        "constraint_status": "feasible" if selected["constraint_feasible"] else "infeasible",
        "schema_version": "v3_2",
        "coarse_candidate_count": int(coarse_candidate_count),
        "full_candidate_count": int(full_candidate_count),
    }
    with (output_directory / "stage1_search_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)


def _print_selected_summary(selected: Mapping[str, Any], bmax: float, output_directory: Path) -> None:
    """在搜索结束时打印可直接用于复现实验的最优 Stage-1 配置。"""
    print(
        "[Stage-1 Search] 搜索完成，最优配置："
        " b={b:.6f}, alpha={alpha:.6f}, beta0={beta0:.6f}, center={center:.6f};"
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
    started = time.perf_counter()
    curvature = dict(raw.get("actor", {}).get("curvature", {}))
    search = dict(curvature.get("base_search", {}))
    if not bool(search.get("enabled", False)):
        return {}
    center_mode = resolve_center_mode(curvature)
    bmax = _probability(raw["constraints"]["max_tx_ratio"], "constraints.max_tx_ratio")
    alpha_values = alpha_candidates(search, curvature.get("alpha_override", curvature.get("alpha_init", 0.0)))
    candidates_config = dict(search.get("base_candidates", {}))
    coarse = coarse_base_candidates(bmax, candidates_config)
    calibration_config = dict(search.get("calibration", {}))
    scores = []
    calibration_count = int(calibration_config.get("topology_count", 32))
    if calibration_count < 1:
        raise ValueError("calibration.topology_count must be positive")
    calibration_started = time.perf_counter()
    for index in range(calibration_count):
        cache = _build_topology_cache(raw, index, "stage1_calibration")
        scores.extend(cache.scores.tolist())
    calibration_cache_seconds = time.perf_counter() - calibration_started
    if center_mode == "none":
        center = 0.0
    elif center_mode == "fixed":
        center = float(curvature["center"])
    else:
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
    if total_topologies < 1 or total_repeats < 1:
        raise ValueError("evaluation topology_count and dynamic_repeats must be positive")
    coarse_topologies = min(total_topologies, int(evaluation.get("coarse_topology_count", total_topologies)))
    coarse_repeats = min(total_repeats, int(evaluation.get("coarse_dynamic_repeats", total_repeats)))
    if coarse_topologies < 1 or coarse_repeats < 1:
        raise ValueError("coarse topology_count and dynamic_repeats must be positive")
    coarse_slots = int(evaluation.get(
        "coarse_slots", evaluation.get("slots", raw.get("experiment", {}).get("slots", 200))
    ))
    full_slots = int(evaluation.get(
        "full_slots", evaluation.get("slots", raw.get("experiment", {}).get("slots", 200))
    ))
    if coarse_slots < 1 or full_slots < 1:
        raise ValueError("evaluation coarse_slots and full_slots must be positive")
    cache_config = dict(evaluation.get("cache", {}))
    cache_enabled = bool(cache_config.get("enabled", True))
    search_cache_started = time.perf_counter()
    if cache_enabled:
        topology_caches = {
            index: _build_topology_cache(raw, index, "stage1_search")
            for index in range(total_topologies)
        }
        full_scenarios = [
            _build_search_scenario_cache(raw, topology_caches[index], repeat)
            for index in range(total_topologies) for repeat in range(total_repeats)
        ]
        coarse_scenarios = [
            scenario for scenario in full_scenarios
            if scenario.topology_index < coarse_topologies
            and scenario.dynamic_repeat_index < coarse_repeats
        ]
    else:
        full_scenarios = [
            SearchScenario(index, repeat, index)
            for index in range(total_topologies) for repeat in range(total_repeats)
        ]
        coarse_scenarios = [
            SearchScenario(index, repeat, index)
            for index in range(coarse_topologies) for repeat in range(coarse_repeats)
        ]
    search_cache_seconds = time.perf_counter() - search_cache_started
    tolerance = float(evaluation.get("max_probability_tolerance", 0.005))
    parallel = dict(evaluation.get("parallel", {}))
    coarse_candidates = [
        dict(entry, phase="coarse", center=center, center_mode=center_mode,
             max_probability_tolerance=tolerance, phase_slots=coarse_slots,
             evaluation_phase="coarse")
        for entry in calibration_entries
    ]
    coarse_started = time.perf_counter()
    coarse_results = _evaluate_candidates(
        raw, coarse_scenarios, coarse_candidates, bmax, coarse_slots, parallel, cache_enabled,
    )
    coarse_evaluation_seconds = time.perf_counter() - coarse_started

    # 粗搜只按 alpha 保留 Top-K，其他候选仍写入 CSV 以便审计但不再跑 full。
    keep_config = dict(evaluation.get("candidate_selection", {}))
    keep_top_k = int(keep_config.get("coarse_keep_top_k_per_alpha", len(coarse)))
    if keep_top_k < 1:
        raise ValueError("candidate_selection.coarse_keep_top_k_per_alpha must be positive")
    kept_coarse = []
    for alpha_order in range(len(alpha_values)):
        rows = [row for row in coarse_results if int(row["alpha_order"]) == alpha_order]
        ordered = sorted(rows, key=lambda row: (
            not bool(row["constraint_feasible"]), float(row["mean_VAoI"]),
            float(row["base_tx_ratio"]), int(row["candidate_id"]),
        ))
        for rank, row in enumerate(ordered, 1):
            row["coarse_rank_within_alpha"] = rank
            row["evaluated_on_full"] = False
            row["pruned_after_coarse"] = rank > min(keep_top_k, len(ordered))
        kept_coarse.extend(ordered[:keep_top_k])

    refine_candidates = []
    refine_count = int(candidates_config.get("refine_count", 5))
    if bool(candidates_config.get("refine", True)) and bool(candidates_config.get("refine_per_alpha", True)):
        for alpha_order, alpha in enumerate(alpha_values):
            alpha_rows = [row for row in coarse_results if int(row["alpha_order"]) == alpha_order]
            best_row = min(alpha_rows, key=lambda row: (not row["constraint_feasible"], row["mean_VAoI"]))
            best_index = min(range(len(coarse)), key=lambda index: abs(coarse[index] - float(best_row["base_tx_ratio"])))
            for base in refine_base_candidates(coarse, best_index, bmax, refine_count):
                if any(math.isclose(base, float(row["base_tx_ratio"]), rel_tol=0.0, abs_tol=1e-12)
                       and int(row["alpha_order"]) == alpha_order
                       for row in kept_coarse + refine_candidates):
                    continue
                calibrated = calibrate_intercept(
                    scores, base, alpha, center,
                    tolerance=float(calibration_config.get("intercept_tolerance", 1e-6)),
                    max_iterations=int(calibration_config.get("intercept_max_iterations", 100)),
                )
                refine_candidates.append(dict(
                    calibrated, alpha_order=alpha_order, phase="refine", center=center,
                    center_mode=center_mode, max_probability_tolerance=tolerance,
                    phase_slots=full_slots, coarse_rank_within_alpha="",
                    evaluated_on_full=True, pruned_after_coarse=False,
                    evaluation_phase="full",
                    candidate_id=len(coarse_candidates) + len(refine_candidates),
                ))

    full_candidates = []
    for candidate in kept_coarse + refine_candidates:
        # Do not pickle coarse per-scenario diagnostics into the full task;
        # the worker recomputes and replaces them for the full rollout.
        full_candidate = {
            key: value for key, value in candidate.items() if key != "per_scenario"
        }
        full_candidate.update(
            phase_slots=full_slots, evaluated_on_full=True,
            pruned_after_coarse=False, evaluation_phase="full",
        )
        full_candidates.append(full_candidate)
    full_started = time.perf_counter()
    full_results = _evaluate_candidates(
        raw, full_scenarios, full_candidates, bmax, full_slots, parallel, cache_enabled,
    )
    full_evaluation_seconds = time.perf_counter() - full_started
    for row in full_results:
        row["evaluated_on_full"] = True
        row["pruned_after_coarse"] = False
        row.setdefault("coarse_rank_within_alpha", "")
    # Keep every coarse row plus every full row; selection is restricted to full runs.
    # This deliberately duplicates promoted candidates so both measurements remain auditable.
    audit_results = list(coarse_results) + list(full_results)
    selected = _select_candidate(full_results, float(evaluation.get("tie_relative_tolerance", 0.01)))
    per_scenario = [row for candidate in full_results for row in candidate["per_scenario"]]
    calibration = {
        "topology_count": calibration_count,
        "center_statistic": str(calibration_config.get("center_statistic", "mean")),
        "center_mode": center_mode,
        "center": center,
        "score_count": len(scores),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "candidates": calibration_entries,
    }
    _write_search_outputs(
        output_directory, calibration, audit_results, per_scenario, selected, center,
        center_mode, len(coarse_candidates), len(full_candidates),
    )
    runtime_profile = dict(evaluation.get("runtime_profile", {}))
    if bool(runtime_profile.get("enabled", False)):
        runtime_path = Path(str(runtime_profile.get("output_file", "stage1_search_runtime.json")))
        if not runtime_path.is_absolute():
            runtime_path = output_directory / runtime_path
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime = {
            "schema_version": "v3_2", "center_mode": center_mode,
            "calibration_topology_count": calibration_count,
            "search_topology_count": total_topologies,
            "dynamic_repeats": total_repeats,
            "coarse_candidate_count": len(coarse_candidates),
            "coarse_slots": coarse_slots,
            "coarse_simulation_count": len(coarse_candidates) * len(coarse_scenarios),
            "full_candidate_count": len(full_candidates),
            "full_slots": full_slots,
            "full_simulation_count": len(full_candidates) * len(full_scenarios),
            "cache_enabled": cache_enabled,
            "parallel_enabled": bool(parallel.get("enabled", False)),
            "parallel_workers": max(
                _parallel_worker_count(parallel, len(coarse_candidates)),
                _parallel_worker_count(parallel, len(full_candidates)),
            ),
            "timing_seconds": {
                "calibration_cache": calibration_cache_seconds,
                "search_cache": search_cache_seconds,
                "coarse_evaluation": coarse_evaluation_seconds,
                "full_evaluation": full_evaluation_seconds,
                "total": time.perf_counter() - started,
            },
        }
        with runtime_path.open("w", encoding="utf-8") as stream:
            json.dump(runtime, stream, indent=2, sort_keys=True)
    _print_selected_summary(selected, bmax, output_directory)
    return {
        "base_tx_ratio": float(selected["base_tx_ratio"]),
        "alpha_override": float(selected["alpha"]),
        "center_mode": center_mode,
        "center": center,
        "calibrated_intercept": float(selected["calibrated_intercept"]),
        "freeze_stage1": True,
        "search_summary": {
            "selected_base_tx_ratio": float(selected["base_tx_ratio"]),
            "selected_alpha": float(selected["alpha"]),
            "selected_intercept": float(selected["calibrated_intercept"]),
            "selected_center": center,
            "selected_center_mode": center_mode,
            "search_probability_mean": float(selected["search_probability_mean"]),
            "actual_tx_ratio": float(selected["actual_tx_ratio"]),
            "mean_VAoI": float(selected["mean_VAoI"]),
        },
    }
