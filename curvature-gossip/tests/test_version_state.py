"""验证版本状态单调性、对角新鲜度、同时合并和旧版本被超越完成。"""

import numpy as np

from curvature_gossip.metrics import DisseminationTracker
from curvature_gossip.state import VersionState


def test_diagonal_freshness_and_nonnegative_age():
    state = VersionState(3)
    state.apply_source_updates(0, np.array([True, False, True]))
    assert np.array_equal(np.diag(state.cache_versions), state.source_versions)
    assert np.all(state.version_age() >= 0)
    assert np.all(np.diag(state.version_age()) == 0)


def test_cache_merge_is_monotone_and_uses_componentwise_maximum():
    state = VersionState(3)
    state.apply_source_updates(0, np.array([True, False, False]))
    versions, slots = state.packet_snapshot()
    before = state.cache_versions.copy()
    result = state.merge_decoded_snapshots({1: 0}, versions, slots)
    assert state.cache_versions[1, 0] == 1
    assert np.all(state.cache_versions >= before)
    assert result.improved_entries == 1


def test_no_same_slot_multihop_cascade_on_three_node_line():
    state = VersionState(3)
    state.apply_source_updates(0, np.array([True, False, False]))
    versions, slots = state.packet_snapshot()
    # 节点 0 发给 1、节点 1 同时发给 2；节点 1 的冻结包仍是旧缓存。
    state.merge_decoded_snapshots({1: 0, 2: 1}, versions, slots)
    assert state.cache_versions[1, 0] == 1
    assert state.cache_versions[2, 0] == 0


def test_zero_update_probability_keeps_version_age_zero():
    state = VersionState(4)
    rng = np.random.default_rng(1)
    for slot in range(5):
        state.generate_source_updates(slot, 0.0, rng)
        assert np.all(state.version_age() == 0)


def test_later_version_can_complete_an_earlier_version():
    tracker = DisseminationTracker()
    tracker.register([(0, 1, 0), (0, 2, 1)])
    caches = np.zeros((3, 3), dtype=np.int64)
    caches[:, 0] = 2
    tracker.update_completions(4, caches)
    assert [record.delay for record in tracker.records] == [4, 3]
    assert tracker.summary()["right_censored_updates"] == 0

