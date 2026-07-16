"""维护源版本与全缓存状态，并提供满足同一时隙因果性的批量合并。"""

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class CacheMergeResult:
    improved_entries: int
    total_version_improvement: int
    receiver_improvements: Mapping[int, int]


class VersionState:
    """保存全局真实状态；策略不会直接获得此对象。"""

    def __init__(self, n_nodes: int) -> None:
        if n_nodes < 2:
            raise ValueError("n_nodes must be at least two")
        self.n_nodes = n_nodes
        self.source_versions = np.zeros(n_nodes, dtype=np.int64)
        self.cache_versions = np.zeros((n_nodes, n_nodes), dtype=np.int64)
        self.cache_gen_slots = np.zeros((n_nodes, n_nodes), dtype=np.int64)

    def apply_source_updates(self, slot: int, generated: np.ndarray) -> Tuple[Tuple[int, int, int], ...]:
        """应用本时隙源更新，并立即刷新各源的对角缓存。"""
        generated = np.asarray(generated, dtype=bool)
        if generated.shape != (self.n_nodes,):
            raise ValueError("generated must have shape [N]")
        events = []
        for source in np.flatnonzero(generated):
            source = int(source)
            self.source_versions[source] += 1
            version = int(self.source_versions[source])
            self.cache_versions[source, source] = version
            self.cache_gen_slots[source, source] = slot
            events.append((source, version, slot))
        return tuple(events)

    def generate_source_updates(self, slot: int, probability: float, rng: np.random.Generator):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("update probability must be in [0, 1]")
        # 始终消费 N 个随机数，保证跨策略外生随机性对齐。
        generated = rng.random(self.n_nodes) < probability
        return generated, self.apply_source_updates(slot, generated)

    def packet_snapshot(self):
        """返回只读深拷贝，确保本时隙收到的信息不能立刻再次转发。"""
        versions = self.cache_versions.copy()
        gen_slots = self.cache_gen_slots.copy()
        versions.setflags(write=False)
        gen_slots.setflags(write=False)
        return versions, gen_slots

    def merge_decoded_snapshots(
        self,
        decoded_senders: Mapping[int, int],
        snapshot_versions: np.ndarray,
        snapshot_gen_slots: np.ndarray,
    ) -> CacheMergeResult:
        """从冻结快照同时合并所有成功数据包。"""
        updates = []
        improved_entries = 0
        total_improvement = 0
        receiver_improvements = {}
        for receiver, sender in decoded_senders.items():
            old_versions = self.cache_versions[receiver].copy()
            incoming = snapshot_versions[sender]
            mask = incoming > old_versions
            count = int(np.count_nonzero(mask))
            improvement = int(np.sum(incoming[mask] - old_versions[mask]))
            updates.append((receiver, mask, incoming.copy(), snapshot_gen_slots[sender].copy()))
            receiver_improvements[int(receiver)] = count
            improved_entries += count
            total_improvement += improvement

        # 延后写回是因果性的关键：所有接收端只读取时隙开始时的包快照。
        for receiver, mask, incoming, incoming_slots in updates:
            self.cache_versions[receiver, mask] = incoming[mask]
            self.cache_gen_slots[receiver, mask] = incoming_slots[mask]
        self._validate_invariants()
        return CacheMergeResult(improved_entries, total_improvement, receiver_improvements)

    def version_age(self) -> np.ndarray:
        age = self.source_versions[np.newaxis, :] - self.cache_versions
        if np.any(age < 0):
            raise RuntimeError("negative version age invariant violated")
        return age

    def time_age(self, slot: int) -> np.ndarray:
        return slot - self.cache_gen_slots

    def _validate_invariants(self) -> None:
        if np.any(self.cache_versions > self.source_versions[np.newaxis, :]):
            raise RuntimeError("cache version exceeds source version")
        diagonal = np.diag(self.cache_versions)
        if not np.array_equal(diagonal, self.source_versions):
            raise RuntimeError("source diagonal cache is stale")

