"""定义拓扑数据结构和可替换的拓扑生成器接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Topology:
    positions: np.ndarray
    graph: nx.Graph
    communication_radius: float
    node_labels: Optional[np.ndarray]
    metadata: Mapping[str, Any]


class TopologyGenerator(ABC):
    @abstractmethod
    def generate(self, rng: np.random.Generator, params: Mapping[str, Any]) -> Topology:
        """使用给定 RNG 生成并校验一个固定拓扑。"""
        raise NotImplementedError

