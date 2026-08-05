"""统一生成 Stage-1、Stage-2 和验证场景的确定性随机流。

本模块只负责随机流命名空间、索引计划和 manifest，不携带候选参数；因此
Stage-1 不同候选可以复用完全相同的 topology、信道和动作 realization。
"""

import hashlib
from typing import Any, Dict, Mapping

import numpy as np

from ..random_streams import make_rng


SEED_DERIVATION_VERSION = "v1"
POOL_NAMESPACES = {
    "calibration": "stage1_calibration",
    "search": "stage1_search",
    "training": "stage2_training",
    "validation": "fixed_validation",
}
PROCESS_NAMESPACES = (
    "topology", "source_updates", "shadowing", "fading", "actions"
)


def derived_seed(master_seed: int, pool_namespace: str, process_namespace: str, index: int, repeat: int = 0) -> int:
    """从根 seed 派生可写入 manifest 的稳定 uint32 seed。"""
    rng = make_rng(master_seed, pool_namespace, process_namespace, int(index), int(repeat))
    return int(rng.integers(0, 2 ** 32, dtype=np.uint32))


def pool_rng(master_seed: int, pool_namespace: str, process_namespace: str, index: int, repeat: int = 0):
    """返回不受候选 ``(b, alpha)`` 影响的随机流。"""
    return make_rng(master_seed, pool_namespace, process_namespace, int(index), int(repeat))


def _config_counts(raw: Mapping[str, Any]) -> Dict[str, int]:
    curvature = dict(raw.get("actor", {}).get("curvature", {}))
    search = dict(curvature.get("base_search", {}))
    calibration = dict(search.get("calibration", {}))
    evaluation = dict(search.get("evaluation", {}))
    training = dict(raw.get("training", {}))
    validation = dict(raw.get("validation", {}))
    generation = dict(validation.get("scenario_generation", {}))
    return {
        "calibration_topologies": int(calibration.get("topology_count", 32)),
        "search_topologies": int(evaluation.get("topology_count", 20)),
        "search_repeats": int(evaluation.get("dynamic_repeats", 2)),
        "training_episodes": int(training.get("episodes", 20)),
        "validation_scenarios": int(generation.get("scenario_count", 20)),
    }


def build_seed_manifest(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """构造四个逻辑集合的可审计 seed manifest。"""
    experiment = dict(raw.get("experiment", {}))
    master_seed = int(experiment.get("master_seed", 0))
    randomness = dict(raw.get("randomness", {}))
    mode = str(randomness.get("mode", "legacy_explicit"))
    counts = _config_counts(raw)
    manifest = {
        "master_seed": master_seed,
        "seed_derivation_version": str(randomness.get("derivation_version", SEED_DERIVATION_VERSION)),
        "mode": mode,
        "pool_namespaces": dict(POOL_NAMESPACES),
        "process_namespaces": list(PROCESS_NAMESPACES),
        "pools": {},
    }

    calibration = []
    for index in range(counts["calibration_topologies"]):
        calibration.append({
            "topology_index": index,
            "topology_seed": derived_seed(master_seed, POOL_NAMESPACES["calibration"], "topology", index),
        })
    manifest["pools"]["calibration"] = {"count": len(calibration), "topologies": calibration}

    search_topologies = []
    for topology_index in range(counts["search_topologies"]):
        repeats = []
        for repeat in range(counts["search_repeats"]):
            repeats.append({
                "dynamic_repeat_index": repeat,
                "source_update_seed": derived_seed(master_seed, POOL_NAMESPACES["search"], "source_updates", topology_index, repeat),
                "shadowing_seed": derived_seed(master_seed, POOL_NAMESPACES["search"], "shadowing", topology_index, repeat),
                "fading_seed": derived_seed(master_seed, POOL_NAMESPACES["search"], "fading", topology_index, repeat),
                "action_seed": derived_seed(master_seed, POOL_NAMESPACES["search"], "actions", topology_index, repeat),
            })
        search_topologies.append({
            "topology_index": topology_index,
            "topology_seed": derived_seed(master_seed, POOL_NAMESPACES["search"], "topology", topology_index),
            "dynamic_repeats": repeats,
        })
    manifest["pools"]["search"] = {
        "topology_count": counts["search_topologies"],
        "dynamic_repeats": counts["search_repeats"],
        "topologies": search_topologies,
    }

    training = []
    for episode in range(counts["training_episodes"]):
        training.append({
            "episode_index": episode,
            "topology_seed": derived_seed(master_seed, POOL_NAMESPACES["training"], "topology", episode),
            "source_update_seed": derived_seed(master_seed, POOL_NAMESPACES["training"], "source_updates", episode),
            "shadowing_seed": derived_seed(master_seed, POOL_NAMESPACES["training"], "shadowing", episode),
            "fading_seed": derived_seed(master_seed, POOL_NAMESPACES["training"], "fading", episode),
            "action_seed": derived_seed(master_seed, POOL_NAMESPACES["training"], "actions", episode),
        })
    manifest["pools"]["training"] = {"count": len(training), "episodes": training}

    validation = dict(raw.get("validation", {}))
    scenario_generation = dict(validation.get("scenario_generation", {}))
    if str(scenario_generation.get("mode", "explicit")) == "auto":
        scenarios = []
        for index in range(counts["validation_scenarios"]):
            scenarios.append({
                "scenario_index": index,
                "topology_seed": derived_seed(master_seed, POOL_NAMESPACES["validation"], "topology", index),
                "source_update_seed": derived_seed(master_seed, POOL_NAMESPACES["validation"], "source_updates", index),
                "channel_shadowing_seed": derived_seed(master_seed, POOL_NAMESPACES["validation"], "shadowing", index),
                "channel_fading_seed": derived_seed(master_seed, POOL_NAMESPACES["validation"], "fading", index),
                "policy_action_seed": derived_seed(master_seed, POOL_NAMESPACES["validation"], "actions", index),
            })
        manifest["pools"]["validation"] = {"mode": "auto", "scenarios": scenarios}
    else:
        # Explicit validation remains byte-for-byte configuration-owned; record it for audit.
        manifest["pools"]["validation"] = {
            "mode": "explicit",
            "scenarios": list(validation.get("scenarios", [])),
        }

    manifest["manifest_fingerprint"] = hashlib.sha256(
        repr(manifest).encode("utf-8")
    ).hexdigest()
    return manifest

