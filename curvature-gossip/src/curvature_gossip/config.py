"""加载并校验 YAML 项目配置，为后续仿真模块提供稳定的配置对象。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


class ConfigError(ValueError):
    """表示配置缺项、类型错误或取值越界。"""


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
        raw=dict(raw),
    )


def config_to_dict(config: ProjectConfig) -> Dict[str, Any]:
    """转换为可序列化字典，便于后续保存 resolved config。"""
    return {
        "experiment": dict(config.experiment),
        "topology": {"type": config.topology.type, "params": dict(config.topology.params)},
        "output": dict(config.output),
    }

