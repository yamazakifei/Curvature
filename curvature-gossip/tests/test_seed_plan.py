"""验证统一 seed plan 的确定性、池隔离和候选参数无关性。"""

from curvature_gossip.learning.seed_plan import build_seed_manifest, pool_rng


def _raw(master_seed=20260723):
    return {
        "experiment": {"master_seed": master_seed},
        "topology": {"params": {"n_nodes": 4}},
        "actor": {"curvature": {"base_search": {
            "calibration": {"topology_count": 2},
            "evaluation": {"topology_count": 2, "dynamic_repeats": 2},
        }}},
        "training": {"episodes": 3},
        "validation": {"scenario_generation": {"mode": "auto", "scenario_count": 2}},
    }


def test_manifest_is_reproducible_and_master_seed_changes_all_automatic_pools():
    first = build_seed_manifest(_raw())
    second = build_seed_manifest(_raw())
    changed = build_seed_manifest(_raw(20260724))
    assert first == second
    assert first["manifest_fingerprint"] != changed["manifest_fingerprint"]
    for pool in ("calibration", "search", "training", "validation"):
        assert first["pools"][pool] != changed["pools"][pool]


def test_pool_and_process_namespaces_are_isolated():
    first = pool_rng(11, "stage1_search", "actions", 0).random(4)
    same = pool_rng(11, "stage1_search", "actions", 0).random(4)
    other_pool = pool_rng(11, "fixed_validation", "actions", 0).random(4)
    other_process = pool_rng(11, "stage1_search", "fading", 0).random(4)
    assert (first == same).all()
    assert not (first == other_pool).all()
    assert not (first == other_process).all()

