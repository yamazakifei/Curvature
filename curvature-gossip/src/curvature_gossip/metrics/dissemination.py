"""跟踪每个生成版本的全网完成时延，并保留未完成更新的右删失状态。"""

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class UpdateRecord:
    source: int
    version: int
    generation_slot: int
    completion_slot: int = -1

    @property
    def completed(self) -> bool:
        return self.completion_slot >= 0

    @property
    def delay(self) -> int:
        return self.completion_slot - self.generation_slot if self.completed else -1


class DisseminationTracker:
    def __init__(self) -> None:
        self._records = []  # type: List[UpdateRecord]
        self._pending = {}  # type: Dict[int, Deque[UpdateRecord]]

    def register(self, events: Iterable[Tuple[int, int, int]]) -> None:
        for source, version, slot in events:
            record = UpdateRecord(int(source), int(version), int(slot))
            self._records.append(record)
            self._pending.setdefault(int(source), deque()).append(record)

    def update_completions(self, slot: int, cache_versions: np.ndarray) -> None:
        """较新版本达到全网时，可同时完成尚未完成的旧版本。"""
        for source, pending in list(self._pending.items()):
            network_minimum = int(np.min(cache_versions[:, source]))
            # 同一源版本严格递增，只需从队首弹出已完成前缀，避免长仿真的反复全表扫描。
            while pending and pending[0].version <= network_minimum:
                pending.popleft().completion_slot = int(slot)
            if not pending:
                del self._pending[source]

    @property
    def records(self):
        return tuple(self._records)

    def summary(self):
        completed = [record.delay for record in self._records if record.completed]
        total = len(self._records)
        values = np.asarray(completed, dtype=float)
        return {
            "completed_update_fraction": float(len(completed) / total) if total else 1.0,
            "mean_dissemination_delay": float(values.mean()) if values.size else None,
            "median_dissemination_delay": float(np.median(values)) if values.size else None,
            "p95_dissemination_delay": float(np.percentile(values, 95)) if values.size else None,
            "max_dissemination_delay": float(values.max()) if values.size else None,
            "right_censored_updates": int(total - len(completed)),
            "total_generated_updates": int(total),
        }
