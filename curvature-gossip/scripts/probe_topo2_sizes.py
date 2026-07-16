"""Probe feasible topo2 node counts for the current topology constraints."""

from pathlib import Path

import yaml

from curvature_gossip.random_streams import make_rng
from curvature_gossip.topology import get_topology_generator


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "configs" / "topo2_backoff_v2.yaml").open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    generator = get_topology_generator(config["topology"]["type"])
    master_seed = config["experiment"]["master_seed"]
    for n_nodes in [60, 80, 100, 120, 150, 200]:
        params = dict(config["topology"]["params"])
        params["n_nodes"] = n_nodes
        params["cluster_sizes"] = [n_nodes // 2, n_nodes - n_nodes // 2]
        ok = []
        bad = []
        for seed in [0, 1, 2, 3, 4]:
            try:
                topology = generator.generate(make_rng(master_seed, "topology", seed), params)
                ok.append((seed, topology.graph.number_of_edges()))
            except Exception:
                bad.append(seed)
        print(n_nodes, "ok", ok, "bad", bad)


if __name__ == "__main__":
    main()
