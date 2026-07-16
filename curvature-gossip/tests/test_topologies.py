"""验证五种拓扑的可复现性、连通性和严格距离一致性。"""

import networkx as nx
import numpy as np
import pytest

from curvature_gossip.topology import get_topology_generator, registered_topologies, validate_topology


CASES = {
    "random_geometric": {
        "n_nodes": 16, "area_width_m": 1.0, "area_height_m": 1.0,
        "communication_radius_m": 0.45, "max_attempts": 200,
    },
    "grid_2d": {"rows": 3, "columns": 4, "spacing_m": 2.0},
    "ring": {"n_nodes": 10, "circle_radius_m": 3.0},
    "two_cluster_bridge": {
        "n_nodes": 16, "cluster_sizes": [8, 8], "communication_radius_m": 10.0,
        "center_separation_m": 23.0, "bridge_edge_count": 1, "max_attempts": 50,
    },
    "soft_two_community": {
        "n_nodes": 40, "cluster_sizes": [20, 20], "communication_radius_m": 40.0,
        "area_width_m": 260.0, "area_height_m": 160.0,
        "cluster_radius_m": 70.0, "center_separation_m": 120.0,
        "cross_edge_count_range": [5, 15], "min_cross_endpoints_per_cluster": 3,
        "total_edge_count_range": [90, 120], "min_degree": 2,
        "require_induced_cluster_connected": True, "min_edge_connectivity": 2,
        "max_attempts": 2000,
    },
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_topology_reproducible_connected_and_distance_consistent(name):
    generator = get_topology_generator(name)
    first = generator.generate(np.random.default_rng(1234), CASES[name])
    second = generator.generate(np.random.default_rng(1234), CASES[name])
    validate_topology(first, CASES[name])
    assert nx.is_connected(first.graph)
    assert np.array_equal(first.positions, second.positions)
    assert sorted(first.graph.edges) == sorted(second.graph.edges)


def test_registry_contains_all_required_generators():
    assert set(registered_topologies()) == {
        "random_geometric", "two_cluster_bridge", "soft_two_community", "grid_2d", "ring"
    }


def test_grid_default_excludes_diagonal_edges():
    topology = get_topology_generator("grid_2d").generate(
        np.random.default_rng(0), {"rows": 2, "columns": 2, "spacing_m": 1.0}
    )
    assert topology.graph.number_of_edges() == 4


def test_ring_has_exactly_two_neighbors_per_node():
    topology = get_topology_generator("ring").generate(
        np.random.default_rng(0), {"n_nodes": 12, "circle_radius_m": 2.0}
    )
    assert set(dict(topology.graph.degree()).values()) == {2}


@pytest.mark.parametrize("bridge_count", [1, 2, 4])
def test_two_cluster_bridge_has_exact_actual_cross_edge_count(bridge_count):
    params = dict(CASES["two_cluster_bridge"])
    params.update({"n_nodes": 40, "cluster_sizes": [20, 20], "bridge_edge_count": bridge_count})
    topology = get_topology_generator("two_cluster_bridge").generate(
        np.random.default_rng(9), params
    )
    cross = [(i, j) for i, j in topology.graph.edges if topology.node_labels[i] != topology.node_labels[j]]
    assert len(cross) == bridge_count
    assert topology.metadata["inter_cluster_edges"] == sorted(cross)


def test_soft_two_community_has_distributed_cross_edges_and_no_single_edge_cut():
    params = CASES["soft_two_community"]
    topology = get_topology_generator("soft_two_community").generate(
        np.random.default_rng(9), params
    )
    cross = [
        (i, j) for i, j in topology.graph.edges
        if topology.node_labels[i] != topology.node_labels[j]
    ]
    left_endpoints = {i for i, _ in cross}
    right_endpoints = {j for _, j in cross}

    assert 5 <= len(cross) <= 15
    assert len(left_endpoints) >= 3
    assert len(right_endpoints) >= 3
    assert 90 <= topology.graph.number_of_edges() <= 120
    assert nx.edge_connectivity(topology.graph) >= 2


def test_invalid_random_geometric_reports_rejection_failure():
    with pytest.raises(ValueError, match="params"):
        get_topology_generator("random_geometric").generate(
            np.random.default_rng(0),
            {"n_nodes": 8, "communication_radius_m": 1e-6, "max_attempts": 2},
        )
