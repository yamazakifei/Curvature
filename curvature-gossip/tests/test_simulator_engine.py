"""验证端到端时隙因果性、零更新 VAoI 和在线标量输出。"""

import networkx as nx
import numpy as np

from curvature_gossip.channel import ChannelParameters, PropagationModel
from curvature_gossip.curvature import CurvatureResult
from curvature_gossip.policies.base import DistributedBroadcastPolicy
from curvature_gossip.simulator import GossipSimulator, SimulationParameters
from curvature_gossip.topology.base import Topology


class NodeActionPolicy(DistributedBroadcastPolicy):
    def __init__(self, active_nodes):
        self.active_nodes = set(active_nodes)

    def transmission_probability(self, observation):
        return 1.0 if observation.node_id in self.active_nodes else 0.0


def _line_simulator(update_probability, slots, active_nodes):
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    topology = Topology(positions, nx.path_graph(3), 1.01, None, {})
    channel = ChannelParameters(
        tx_power_dbm=30.0, pathloss_reference_db=0.0, pathloss_exponent=0.0,
        rayleigh_fading=False, shadowing_std_db=0.0,
        noise_psd_dbm_per_hz=-100.0, bandwidth_hz=1.0, noise_figure_db=0.0,
        sinr_threshold_db=-20.0,
    )
    propagation = PropagationModel(positions, channel, np.random.default_rng(1))
    curvature = CurvatureResult("test", {(0, 1): 0.0, (1, 2): 0.0}, np.zeros(3), {})
    return GossipSimulator(
        topology, curvature, {(0, 1): 0.0, (1, 2): 0.0}, propagation,
        NodeActionPolicy(active_nodes), SimulationParameters(slots, update_probability),
        np.random.default_rng(2), np.random.default_rng(3), np.random.default_rng(4),
    )


def test_engine_prevents_same_slot_multihop_forwarding():
    simulator = _line_simulator(1.0, 1, {0, 1})
    result = simulator.run()
    # 节点 1 同时发射的冻结包不含节点 0 在该槽生成的新版本。
    assert result.final_state.cache_versions[2, 0] == 0


def test_zero_updates_keep_all_vaoi_zero():
    result = _line_simulator(0.0, 5, {0}).run()
    assert result.summary["mean_VAoI"] == 0.0
    assert result.summary["mean_max_VAoI"] == 0.0
    assert len(result.per_slot) == 5


def test_metrics_report_activity_and_decode_efficiency():
    result = _line_simulator(0.5, 4, {0}).run()
    assert result.summary["avg_tx_per_slot"] == 1.0
    assert result.summary["successful_decodes_per_tx"] >= 1.0
    assert len(result.summary["per_node_activity_ratio"]) == 3
