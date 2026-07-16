"""定义未来分布式曲率估计器的本地视图和控制消息接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class LocalTopologyView:
    node_id: int
    neighbor_ids: Tuple[int, ...]


@dataclass(frozen=True)
class CurvatureMessage:
    sender: int
    receiver: Optional[int]
    kind: str
    payload: Mapping[str, Any]


class DistributedCurvatureAgent(ABC):
    @abstractmethod
    def initialize(self, node_id: int, local_view: LocalTopologyView) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_messages(self):
        raise NotImplementedError

    @abstractmethod
    def receive(self, message: CurvatureMessage) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def incident_curvatures(self):
        raise NotImplementedError

