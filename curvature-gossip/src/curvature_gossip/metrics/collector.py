"""在线累计 VAoI、时间 AoI、尾部指标和广播效率，仅保存降采样标量轨迹。"""

import math
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..channel.decoder import DecodeResult
from ..state.version_state import CacheMergeResult, VersionState


class MetricsCollector:
    def __init__(
        self,
        n_nodes: int,
        warmup_slots: int = 0,
        trace_stride: int = 1,
        node_labels: Optional[np.ndarray] = None,
        broadcast_limits: Optional[np.ndarray] = None,
    ) -> None:
        if trace_stride < 1:
            raise ValueError("trace_stride must be positive")
        self.n_nodes = n_nodes
        self.warmup_slots = int(warmup_slots)
        self.trace_stride = int(trace_stride)
        self.node_labels = None if node_labels is None else np.asarray(node_labels)
        self.broadcast_limits = (
            np.ones(n_nodes, dtype=float)
            if broadcast_limits is None else np.asarray(broadcast_limits, dtype=float)
        )
        if self.broadcast_limits.shape != (n_nodes,):
            raise ValueError("broadcast_limits must have shape [N]")
        self.sums = defaultdict(float)
        self.measured_slots = 0
        self.max_vaoi_samples = []
        self.per_slot = []
        self.node_tx_counts = np.zeros(n_nodes, dtype=np.int64)
        self.total_transmissions = 0
        self.total_decodes = 0
        self.total_decoded_transmitters = 0
        self.total_improved_entries = 0
        self.total_version_improvement = 0

    def _cluster_metrics(self, version_age: np.ndarray) -> Dict[str, float]:
        if self.node_labels is None:
            return {}
        output = {}
        labels = sorted(np.unique(self.node_labels).tolist())
        for source_label in labels:
            for holder_label in labels:
                if source_label == holder_label:
                    continue
                holders = np.flatnonzero(self.node_labels == holder_label)
                sources = np.flatnonzero(self.node_labels == source_label)
                values = version_age[np.ix_(holders, sources)].reshape(-1)
                prefix = "cluster_{}_to_{}".format(source_label, holder_label)
                output["mean_vaoi_{}".format(prefix)] = float(np.mean(values))
                output["max_vaoi_{}".format(prefix)] = float(np.max(values))
        return output

    def record(
        self,
        slot: int,
        state: VersionState,
        actions: np.ndarray,
        decode: DecodeResult,
        merge: CacheMergeResult,
        broadcast_debt: Optional[np.ndarray] = None,
    ) -> None:
        if slot < self.warmup_slots:
            return
        version_age = state.version_age()
        time_age = state.time_age(slot)
        mask = ~np.eye(self.n_nodes, dtype=bool)
        ages = version_age[mask].astype(float)
        time_ages = time_age[mask].astype(float)
        tail_count = max(1, int(math.ceil(0.05 * ages.size)))
        tail = np.partition(ages, ages.size - tail_count)[-tail_count:]
        decoded_transmitters = len(set(decode.decoded_senders.values()))
        row = {
            "slot": int(slot),
            "mean_version_age": float(np.mean(ages)),
            "max_version_age": float(np.max(ages)),
            "p95_version_age": float(np.percentile(ages, 95)),
            "p99_version_age": float(np.percentile(ages, 99)),
            "top_5pct_mean_version_age": float(np.mean(tail)),
            "mean_time_age": float(np.mean(time_ages)),
            "num_transmitters": int(np.count_nonzero(actions)),
            "num_successful_decodes": int(len(decode.decoded_senders)),
            "num_transmitters_decoded": int(decoded_transmitters),
            "num_improved_cache_entries": int(merge.improved_entries),
            "total_version_improvement": int(merge.total_version_improvement),
        }
        if broadcast_debt is not None:
            debt = np.asarray(broadcast_debt, dtype=float)
            row.update({
                "mean_broadcast_debt": float(np.mean(debt)),
                "max_broadcast_debt": float(np.max(debt)),
                "node_debt_positive_fraction": float(np.mean(debt > 0.0)),
            })
        row.update(self._cluster_metrics(version_age))
        for key, value in row.items():
            if key != "slot":
                self.sums[key] += float(value)
        self.measured_slots += 1
        self.max_vaoi_samples.append(row["max_version_age"])
        self.node_tx_counts += np.asarray(actions, dtype=np.int64)
        self.total_transmissions += row["num_transmitters"]
        self.total_decodes += row["num_successful_decodes"]
        self.total_decoded_transmitters += decoded_transmitters
        self.total_improved_entries += merge.improved_entries
        self.total_version_improvement += merge.total_version_improvement
        if (slot - self.warmup_slots) % self.trace_stride == 0:
            self.per_slot.append(row)

    def summary(self) -> Dict[str, Any]:
        count = self.measured_slots
        if count == 0:
            raise RuntimeError("no post-warmup slots were measured")
        output = {"mean_{}".format(key): value / count for key, value in self.sums.items()}
        # 提供实验表直接使用的稳定字段名。
        activity = self.node_tx_counts / count
        output.update({
            "mean_VAoI": self.sums["mean_version_age"] / count,
            "mean_max_VAoI": self.sums["max_version_age"] / count,
            "p95_max_VAoI": float(np.percentile(self.max_vaoi_samples, 95)),
            "mean_tail_VAoI": self.sums["top_5pct_mean_version_age"] / count,
            "avg_tx_per_slot": self.total_transmissions / count,
            "successful_decodes_per_tx": self.total_decodes / self.total_transmissions if self.total_transmissions else 0.0,
            "decoded_transmitters_per_tx": self.total_decoded_transmitters / self.total_transmissions if self.total_transmissions else 0.0,
            "innovative_entries_per_tx": self.total_improved_entries / self.total_transmissions if self.total_transmissions else 0.0,
            "version_improvement_per_tx": self.total_version_improvement / self.total_transmissions if self.total_transmissions else 0.0,
            "per_node_activity_ratio": activity.tolist(),
            "per_node_broadcast_limit": self.broadcast_limits.tolist(),
            "std_node_activity_ratio": float(np.std(activity)),
            "p95_node_activity_ratio": float(np.percentile(activity, 95)),
            "max_node_activity_ratio": float(np.max(activity)),
            "node_cap_violation_fraction": float(np.mean(
                activity > self.broadcast_limits + 1e-12
            )),
            "max_node_cap_violation": float(np.max(np.maximum(
                activity - self.broadcast_limits, 0.0
            ))),
            "measured_slots": count,
        })
        return output
