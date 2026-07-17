"""编排局部 AF3 控制消息，并将节点计算结果适配到统一曲率接口。"""

from typing import Dict, Tuple

import numpy as np

from ...topology.base import Topology
from ..base import CurvatureProvider, CurvatureResult, canonical_edge, incident_node_means
from .af3_agent import DistributedAF3Agent
from .base import LocalTopologyView


class DistributedAF3Curvature(CurvatureProvider):
    """模拟可靠初始化控制面；每个 AF3 数值仍由对应节点本地计算。"""

    def compute(self, topology: Topology) -> CurvatureResult:
        graph = topology.graph
        agents = {}
        for node in graph.nodes:
            agent = DistributedAF3Agent()
            agent.initialize(
                node,
                LocalTopologyView(
                    node_id=int(node),
                    neighbor_ids=tuple(sorted(int(value) for value in graph.neighbors(node))),
                ),
            )
            agents[int(node)] = agent

        messages = tuple(
            message
            for agent in agents.values()
            for message in agent.create_messages()
        )
        for message in messages:
            agents[int(message.receiver)].receive(message)

        edge_values = {}  # type: Dict[Tuple[int, int], float]
        for node, agent in agents.items():
            for neighbor, value in agent.incident_curvatures().items():
                edge = canonical_edge(node, neighbor)
                if edge in edge_values and not np.isclose(edge_values[edge], value):
                    raise RuntimeError("AF3 endpoints produced inconsistent edge values")
                edge_values[edge] = float(value)

        return CurvatureResult(
            method="distributed_af3",
            edge_values=edge_values,
            node_values=incident_node_means(topology, edge_values),
            metadata={
                "formula": "4-deg(i)-deg(j)+3*common_neighbors",
                "control_message_count": len(messages),
                "control_payload_neighbor_ids": int(sum(
                    len(message.payload["neighbor_ids"]) for message in messages
                )),
            },
        )
