"""实现半双工、邻域最强候选、全网干扰和至多单包解码规则。"""

from dataclasses import dataclass
from typing import Dict, Mapping

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class DecodeResult:
    decoded_senders: Mapping[int, int]
    sinr: np.ndarray
    interference_plus_noise: np.ndarray


class StrongestSignalDecoder:
    def __init__(self, graph: nx.Graph, noise_power_mw: float, sinr_threshold: float) -> None:
        self.n_nodes = graph.number_of_nodes()
        self.adjacency = nx.to_numpy_array(graph, nodelist=range(self.n_nodes), dtype=bool)
        self.noise_power_mw = float(noise_power_mw)
        self.sinr_threshold = float(sinr_threshold)
        if self.noise_power_mw <= 0 or self.sinr_threshold < 0:
            raise ValueError("noise must be positive and SINR threshold nonnegative")

    def resolve(self, active: np.ndarray, rx_power_mw: np.ndarray) -> DecodeResult:
        active = np.asarray(active, dtype=bool)
        rx_power_mw = np.asarray(rx_power_mw, dtype=float)
        if active.shape != (self.n_nodes,) or rx_power_mw.shape != (self.n_nodes, self.n_nodes):
            raise ValueError("invalid action or received-power shape")

        decoded = {}  # type: Dict[int, int]
        sinr = np.full(self.n_nodes, np.nan, dtype=float)
        interference = np.full(self.n_nodes, self.noise_power_mw, dtype=float)
        active_ids = np.flatnonzero(active)
        if active_ids.size:
            total_active_power = np.sum(rx_power_mw[active_ids, :], axis=0)
        else:
            total_active_power = np.zeros(self.n_nodes, dtype=float)

        for receiver in range(self.n_nodes):
            # 发射节点为半双工，本时隙不参与接收。
            if active[receiver]:
                interference[receiver] += total_active_power[receiver]
                continue
            candidates = active_ids[self.adjacency[active_ids, receiver]]
            if candidates.size == 0:
                interference[receiver] += total_active_power[receiver]
                continue
            powers = rx_power_mw[candidates, receiver]
            sender = int(candidates[int(np.argmax(powers))])
            desired = float(rx_power_mw[sender, receiver])
            denominator = self.noise_power_mw + float(total_active_power[receiver] - desired)
            interference[receiver] = denominator
            sinr[receiver] = desired / denominator
            if sinr[receiver] >= self.sinr_threshold:
                decoded[receiver] = sender
        return DecodeResult(decoded, sinr, interference)

