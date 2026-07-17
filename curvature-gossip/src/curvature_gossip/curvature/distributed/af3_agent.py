"""通过一次邻居摘要交换，在每个节点本地计算 incident AF3 边曲率。"""

from typing import Dict, Tuple

from .base import (
    CurvatureMessage, DistributedCurvatureAgent, LocalTopologyView,
)


class DistributedAF3Agent(DistributedCurvatureAgent):
    """只依赖一跳邻居列表，不读取完整拓扑对象的 AF3 节点代理。"""

    def __init__(self) -> None:
        self.node_id = None
        self.neighbor_ids = tuple()  # type: Tuple[int, ...]
        self._neighbor_summaries = {}  # type: Dict[int, Tuple[int, ...]]

    def initialize(self, node_id: int, local_view: LocalTopologyView) -> None:
        if int(node_id) != int(local_view.node_id):
            raise ValueError("node_id must match local_view.node_id")
        neighbors = tuple(sorted(int(value) for value in local_view.neighbor_ids))
        if int(node_id) in neighbors or len(neighbors) != len(set(neighbors)):
            raise ValueError("local neighbor identifiers must be unique and exclude self")
        self.node_id = int(node_id)
        self.neighbor_ids = neighbors
        self._neighbor_summaries = {}

    def create_messages(self):
        if self.node_id is None:
            raise RuntimeError("agent must be initialized before creating messages")
        # 静态拓扑只需在初始化时向每个邻居发送一次本节点的一跳邻居集合。
        payload = {
            "degree": len(self.neighbor_ids),
            "neighbor_ids": self.neighbor_ids,
        }
        return tuple(
            CurvatureMessage(
                sender=self.node_id,
                receiver=neighbor,
                kind="af3_neighbor_summary",
                payload=payload,
            )
            for neighbor in self.neighbor_ids
        )

    def receive(self, message: CurvatureMessage) -> None:
        if self.node_id is None:
            raise RuntimeError("agent must be initialized before receiving messages")
        if message.receiver != self.node_id or message.sender not in self.neighbor_ids:
            raise ValueError("AF3 summaries may only be received from physical neighbors")
        if message.kind != "af3_neighbor_summary":
            raise ValueError("unsupported AF3 message kind: {}".format(message.kind))
        neighbors = tuple(sorted(int(value) for value in message.payload["neighbor_ids"]))
        if int(message.payload["degree"]) != len(neighbors):
            raise ValueError("AF3 summary degree does not match its neighbor list")
        self._neighbor_summaries[int(message.sender)] = neighbors

    def is_ready(self) -> bool:
        return (
            self.node_id is not None
            and set(self._neighbor_summaries) == set(self.neighbor_ids)
        )

    def incident_curvatures(self):
        if not self.is_ready():
            raise RuntimeError("all one-hop AF3 summaries must arrive before computation")
        own_neighbors = set(self.neighbor_ids)
        output = {}
        for neighbor in self.neighbor_ids:
            neighbor_neighbors = set(self._neighbor_summaries[neighbor])
            common_count = len(own_neighbors.intersection(neighbor_neighbors))
            output[neighbor] = float(
                4 - len(own_neighbors) - len(neighbor_neighbors) + 3 * common_count
            )
        return output
