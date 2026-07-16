"""验证 ORC 键规范、有限性、桥边负曲率、归一化与可替换 AF3 后端。"""

import networkx as nx
import numpy as np

from curvature_gossip.curvature import (
    GlobalAF3Curvature, GlobalORCCurvature, bottleneck_importance, canonical_edge,
)
from curvature_gossip.topology.base import Topology


def _two_cliques_with_bridge():
    graph = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(4))
    graph.add_edge(3, 4)
    return Topology(np.column_stack((np.arange(8), np.zeros(8))), graph, 1.0, None, {})


def test_orc_is_finite_canonical_and_endpoint_order_independent():
    topology = _two_cliques_with_bridge()
    result = GlobalORCCurvature(alpha=0.5).compute(topology)
    assert set(result.edge_values) == {canonical_edge(*edge) for edge in topology.graph.edges}
    assert all(np.isfinite(value) for value in result.edge_values.values())
    assert result.value(3, 4) == result.value(4, 3)


def test_bridge_is_more_negative_than_dense_intra_cluster_edge():
    result = GlobalORCCurvature(alpha=0.5).compute(_two_cliques_with_bridge())
    assert result.value(3, 4) < result.value(0, 1)


def test_negative_curvature_importance_is_normalized():
    topology = _two_cliques_with_bridge()
    result = GlobalORCCurvature().compute(topology)
    importance = bottleneck_importance(topology, result)
    assert all(0.0 <= value <= 1.0 for value in importance.values())
    assert importance[canonical_edge(3, 4)] == 1.0


def test_af3_uses_documented_formula_and_common_interface():
    topology = _two_cliques_with_bridge()
    result = GlobalAF3Curvature().compute(topology)
    expected = 4 - topology.graph.degree(3) - topology.graph.degree(4)
    assert result.value(3, 4) == expected
    assert result.method == "global_af3"
