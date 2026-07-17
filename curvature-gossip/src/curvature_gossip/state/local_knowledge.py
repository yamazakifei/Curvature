"""维护每个节点通过成功侦听获得的邻居缓存估计。"""

from typing import Mapping

import networkx as nx
import numpy as np


class LocalKnowledge:
    def __init__(self, graph: nx.Graph, broadcast_limit: float = 1.0) -> None:
        if not 0.0 < float(broadcast_limit) <= 1.0:
            raise ValueError("broadcast_limit must be in (0, 1]")
        self.n_nodes = graph.number_of_nodes()
        # 简洁起见使用稠密数组；无效项由 valid 显式区分。
        self.neighbor_cache_estimate = np.zeros(
            (self.n_nodes, self.n_nodes, self.n_nodes), dtype=np.int64
        )
        self.neighbor_estimate_valid = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)
        self.last_tx_slot = np.full(self.n_nodes, -1, dtype=np.int64)
        self.previous_transmitted = np.zeros(self.n_nodes, dtype=bool)
        self.previous_interference_power = np.zeros(self.n_nodes, dtype=float)
        self.previous_interference_valid = np.zeros(self.n_nodes, dtype=bool)
        # 仅由节点静默时的本地能量检测更新，不依赖接收端 ACK。
        self.congestion_ewma = np.zeros(self.n_nodes, dtype=float)
        self.consecutive_tx_attempts = np.zeros(self.n_nodes, dtype=np.int64)
        # 每个节点独立维护长期广播约束的虚拟债务，无需任何全局计数器。
        self.broadcast_limit = np.full(self.n_nodes, float(broadcast_limit), dtype=float)
        self.broadcast_debt = np.zeros(self.n_nodes, dtype=float)
        self._adjacency = nx.to_numpy_array(graph, nodelist=range(self.n_nodes), dtype=bool)

    def update_from_decodes(
        self,
        decoded_senders: Mapping[int, int],
        packet_snapshot: np.ndarray,
    ) -> None:
        """接收端仅更新实际成功解码的发送邻居估计。"""
        for receiver, sender in decoded_senders.items():
            if not self._adjacency[receiver, sender]:
                raise ValueError("decoded sender must be a physical neighbor")
            self.neighbor_cache_estimate[receiver, sender] = packet_snapshot[sender]
            self.neighbor_estimate_valid[receiver, sender] = True

    def update_slot_history(
        self,
        slot: int,
        transmitted: np.ndarray,
        interference_power: np.ndarray,
        noise_power: float,
        congestion_ewma_alpha: float,
    ) -> None:
        transmitted = np.asarray(transmitted, dtype=bool)
        interference_power = np.asarray(interference_power, dtype=float)
        if interference_power.shape != (self.n_nodes,):
            raise ValueError("interference_power must have shape [N]")
        if noise_power <= 0 or not 0.0 <= congestion_ewma_alpha <= 1.0:
            raise ValueError("invalid noise power or congestion EWMA alpha")
        self.last_tx_slot[transmitted] = slot
        self.previous_transmitted = transmitted.copy()
        self.consecutive_tx_attempts[transmitted] += 1
        self.consecutive_tx_attempts[~transmitted] = 0
        self.broadcast_debt = np.maximum(
            0.0,
            self.broadcast_debt + transmitted.astype(float) - self.broadcast_limit,
        )

        # 半双工发送节点无法在发射时可靠测量信道，故不向策略暴露该值。
        silent = ~transmitted
        self.previous_interference_power.fill(np.nan)
        self.previous_interference_power[silent] = interference_power[silent]
        self.previous_interference_valid = silent
        interference_ratio = np.maximum(
            (interference_power[silent] - noise_power) / noise_power,
            0.0,
        )
        measured_congestion = np.log1p(interference_ratio)
        self.congestion_ewma[silent] = (
            congestion_ewma_alpha * self.congestion_ewma[silent]
            + (1.0 - congestion_ewma_alpha) * measured_congestion
        )
