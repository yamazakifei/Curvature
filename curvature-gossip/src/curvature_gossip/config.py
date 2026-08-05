"""加载并校验 YAML 项目配置，为后续仿真模块提供稳定的配置对象。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping
import copy
import warnings

import yaml


class ConfigError(ValueError):
    """表示配置缺项、类型错误或取值越界。"""


def normalize_training_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """规范化 Bmax 与 Stage-1 基础概率，并保留旧 YAML 的兼容语义。

    新配置使用 ``constraints.max_tx_ratio`` 和
    ``actor.curvature.base_tx_ratio``。旧配置的 ``target_tx_ratio`` 仍被
    解释为 Bmax；只有在没有显式基础概率时才把 Bmax 作为旧行为回退值。
    """
    resolved = copy.deepcopy(dict(raw))
    constraints = dict(resolved.get("constraints", {}))
    training = dict(resolved.get("training", {}))
    actor = dict(resolved.get("actor", {}))
    curvature = dict(actor.get("curvature", {}))

    if "max_tx_ratio" not in constraints:
        if "target_tx_ratio" in constraints:
            constraints["max_tx_ratio"] = constraints["target_tx_ratio"]
            warnings.warn(
                "旧字段 constraints.target_tx_ratio 已兼容为 Bmax；建议改用 constraints.max_tx_ratio。",
                UserWarning,
                stacklevel=2,
            )
        else:
            # Legacy non-NN experiments historically defaulted to an unconstrained rate.
            constraints["max_tx_ratio"] = 1.0

    if "max_tx_ratios" not in training:
        if "target_tx_ratios" in training:
            training["max_tx_ratios"] = training["target_tx_ratios"]
            warnings.warn(
                "旧字段 training.target_tx_ratios 已兼容为 Bmax 列表；建议改用 training.max_tx_ratios。",
                UserWarning,
                stacklevel=2,
            )
        else:
            training["max_tx_ratios"] = [constraints["max_tx_ratio"]]

    if "base_tx_ratio" not in curvature:
        curvature["base_tx_ratio"] = constraints["max_tx_ratio"]
        warnings.warn(
            "未配置 actor.curvature.base_tx_ratio，已回退为 Bmax 以保持旧行为。",
            UserWarning,
            stacklevel=2,
        )

    actor["curvature"] = curvature
    resolved["constraints"] = constraints
    resolved["training"] = training
    resolved["actor"] = actor
    return resolved


@dataclass(frozen=True)
class TopologyConfig:
    type: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class ProjectConfig:
    experiment: Mapping[str, Any]
    topology: TopologyConfig
    output: Mapping[str, Any]
    raw: Mapping[str, Any]


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError("配置项 '{}' 必须是映射".format(key))
    return value


def load_config(path: str) -> ProjectConfig:
    """读取 YAML，并执行 Milestone 1 所需的结构和拓扑参数校验。"""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError("配置文件不存在: {}".format(config_path))
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ConfigError("YAML 根节点必须是映射")

    experiment = _require_mapping(raw, "experiment")
    topology_raw = _require_mapping(raw, "topology")
    output = raw.get("output", {})
    if not isinstance(output, dict):
        raise ConfigError("配置项 'output' 必须是映射")

    topology_type = topology_raw.get("type")
    params = topology_raw.get("params", {})
    if not isinstance(topology_type, str) or not topology_type.strip():
        raise ConfigError("topology.type 必须是非空字符串")
    if not isinstance(params, dict):
        raise ConfigError("topology.params 必须是映射")
    if "master_seed" in experiment and not isinstance(experiment["master_seed"], int):
        raise ConfigError("experiment.master_seed 必须是整数")

    # 在加载阶段确认生成器已注册，尽早报告拼写错误。
    from .topology.registry import get_topology_generator

    get_topology_generator(topology_type)
    return ProjectConfig(
        experiment=dict(experiment),
        topology=TopologyConfig(topology_type, dict(params)),
        output=dict(output),
        raw=normalize_training_config(raw),
    )


def config_to_dict(config: ProjectConfig) -> Dict[str, Any]:
    """转换为可序列化字典，便于后续保存 resolved config。"""
    return {
        "experiment": dict(config.experiment),
        "topology": {"type": config.topology.type, "params": dict(config.topology.params)},
        "output": dict(config.output),
    }
