"""覆盖半双工、范围、最强候选、碰撞、外部干扰及信道可复现性。"""

import networkx as nx
import numpy as np

from curvature_gossip.channel import ChannelParameters, PropagationModel, StrongestSignalDecoder


def _decoder(graph, threshold=1.0, noise=1.0):
    return StrongestSignalDecoder(graph, noise_power_mw=noise, sinr_threshold=threshold)


def test_transmitter_is_half_duplex_and_no_candidate_means_no_decode():
    graph = nx.path_graph(3)
    power = np.zeros((3, 3))
    power[0, 1] = power[1, 0] = 10.0
    result = _decoder(graph).resolve(np.array([True, True, False]), power)
    assert 0 not in result.decoded_senders
    assert 1 not in result.decoded_senders
    assert 2 not in result.decoded_senders


def test_zero_interference_success_agrees_with_snr_threshold():
    graph = nx.path_graph(2)
    power = np.array([[0.0, 2.0], [2.0, 0.0]])
    assert _decoder(graph, threshold=1.5).resolve(np.array([True, False]), power).decoded_senders == {1: 0}
    assert _decoder(graph, threshold=3.0).resolve(np.array([True, False]), power).decoded_senders == {}


def test_strong_interferer_can_fail_and_out_of_range_cannot_be_decoded():
    graph = nx.Graph([(0, 1), (1, 2)])
    graph.add_node(3)  # 临时加入后连边以保证图结构对 decoder 有完整 ID。
    graph.add_edge(2, 3)
    power = np.zeros((4, 4))
    power[0, 1] = 10.0
    power[3, 1] = 20.0  # 节点 3 不在接收端 1 的邻域，但仍制造干扰。
    result = _decoder(graph, threshold=1.0).resolve(np.array([True, False, False, True]), power)
    assert result.decoded_senders == {}


def test_at_most_strongest_in_range_transmitter_is_decoded():
    graph = nx.Graph([(0, 2), (1, 2), (0, 1)])
    power = np.zeros((3, 3))
    power[0, 2] = 9.0
    power[1, 2] = 2.0
    result = _decoder(graph, threshold=1.0).resolve(np.array([True, True, False]), power)
    assert result.decoded_senders == {2: 0}


def test_full_fading_and_static_shadowing_are_reproducible():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params = ChannelParameters(shadowing_std_db=2.0, rayleigh_fading=True)
    first = PropagationModel(positions, params, np.random.default_rng(10))
    second = PropagationModel(positions, params, np.random.default_rng(10))
    first_power = first.received_power(np.random.default_rng(11))
    second_power = second.received_power(np.random.default_rng(11))
    assert np.array_equal(first.shadowing_db, second.shadowing_db)
    assert np.array_equal(first_power, second_power)
    assert np.all(np.diag(first_power) == 0.0)
