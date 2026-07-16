"""验证未知邻居状态和成功侦听后的本地估计更新。"""

import networkx as nx
import numpy as np

from curvature_gossip.state import LocalKnowledge


def test_neighbor_estimate_starts_invalid_and_updates_only_decode():
    knowledge = LocalKnowledge(nx.path_graph(3))
    packets = np.arange(9).reshape(3, 3)
    assert not knowledge.neighbor_estimate_valid.any()
    knowledge.update_from_decodes({1: 0}, packets)
    assert knowledge.neighbor_estimate_valid[1, 0]
    assert np.array_equal(knowledge.neighbor_cache_estimate[1, 0], packets[0])
    assert not knowledge.neighbor_estimate_valid[2, 1]


def test_congestion_uses_silent_measurements_and_attempt_history_only():
    knowledge = LocalKnowledge(nx.path_graph(2))
    knowledge.update_slot_history(
        slot=0,
        transmitted=np.array([True, False]),
        interference_power=np.array([9.0, 5.0]),
        noise_power=1.0,
        congestion_ewma_alpha=0.0,
    )
    assert knowledge.consecutive_tx_attempts.tolist() == [1, 0]
    assert not knowledge.previous_interference_valid[0]
    assert np.isnan(knowledge.previous_interference_power[0])
    assert knowledge.previous_interference_valid[1]
    assert knowledge.congestion_ewma[1] == np.log1p(4.0)

    knowledge.update_slot_history(
        slot=1,
        transmitted=np.array([True, False]),
        interference_power=np.array([9.0, 1.0]),
        noise_power=1.0,
        congestion_ewma_alpha=0.0,
    )
    assert knowledge.consecutive_tx_attempts.tolist() == [2, 0]
    assert knowledge.congestion_ewma[1] == 0.0
