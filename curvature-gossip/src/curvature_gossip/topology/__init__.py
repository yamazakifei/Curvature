"""导出拓扑接口，并通过导入模块完成内置生成器注册。"""

from .base import Topology, TopologyGenerator
from .registry import get_topology_generator, registered_topologies
from .validation import TopologyValidationError, validate_topology

# 导入即注册；新增生成器只需定义类并在此导入。
from . import grid, random_geometric, ring, soft_two_community, two_cluster_bridge  # noqa: F401,E402

__all__ = [
    "Topology",
    "TopologyGenerator",
    "TopologyValidationError",
    "get_topology_generator",
    "registered_topologies",
    "validate_topology",
]
