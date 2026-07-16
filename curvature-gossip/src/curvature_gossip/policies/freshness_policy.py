"""计算可缩放的 freshness 广播概率，并导出概率分解诊断信息。"""

import math

import numpy as np

from ..simulator.observations import NodeObservation
from .base import DistributedBroadcastPolicy


class FreshnessPolicy(DistributedBroadcastPolicy):
    def __init__(
        self,
        source_aggregation: str = "max",
        top_fraction: float = 0.05,
        q_min: float = 0.01,
        q_max: float = 0.5,
        threshold: float = 1.0,
        temperature: float = 1.0,
        age_weight: float = 0.02,
        gain_compression: str = "none",
        gain_normalization: str = "none",
        congestion_backoff_weight: float = 0.0,
        congestion_ewma_alpha: float = 0.8,
        attempt_backoff_factor: float = 1.0,
    ) -> None:
        if source_aggregation not in ("max", "mean", "top_fraction"):
            raise ValueError("unsupported source aggregation")
        if not 0.0 <= q_min <= q_max <= 1.0 or temperature <= 0:
            raise ValueError("invalid probability range or temperature")
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        if gain_compression not in ("none", "log1p"):
            raise ValueError("gain_compression must be 'none' or 'log1p'")
        if gain_normalization not in ("none", "log_node_count"):
            raise ValueError(
                "gain_normalization must be 'none' or 'log_node_count'"
            )
        if congestion_backoff_weight < 0.0:
            raise ValueError("congestion_backoff_weight must be nonnegative")
        if not 0.0 <= congestion_ewma_alpha <= 1.0:
            raise ValueError("congestion_ewma_alpha must be in [0, 1]")
        if not 0.0 < attempt_backoff_factor <= 1.0:
            raise ValueError("attempt_backoff_factor must be in (0, 1]")
        self.source_aggregation = source_aggregation
        self.top_fraction = float(top_fraction)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.age_weight = float(age_weight)
        self.gain_compression = gain_compression
        self.gain_normalization = gain_normalization
        self.congestion_backoff_weight = float(congestion_backoff_weight)
        self.congestion_ewma_alpha = float(congestion_ewma_alpha)
        self.attempt_backoff_factor = float(attempt_backoff_factor)

    def _aggregate(self, advantages: np.ndarray) -> float:
        if self.source_aggregation == "max":
            return float(np.max(advantages))
        if self.source_aggregation == "mean":
            return float(np.mean(advantages))
        count = max(1, int(math.ceil(advantages.size * self.top_fraction)))
        selected = np.partition(advantages, advantages.size - count)[-count:]
        return float(np.mean(selected))

    def edge_gains(self, observation: NodeObservation) -> np.ndarray:
        # 新鲜度增益仅使用本地缓存与邻居估计；压缩在源聚合前逐元素执行。
        if not observation.neighbor_ids:
            return np.zeros(0, dtype=float)
        differences = np.maximum(
            observation.own_cache_versions[np.newaxis, :] - observation.neighbor_cache_estimates,
            0,
        ).astype(float)
        if self.gain_compression == "log1p":
            differences = np.log1p(differences)
        if self.gain_normalization == "log_node_count":
            # 只校正多源极值随规模的缓慢增长，避免线性除以 N 过度衰减。
            node_scale = math.log1p(max(1, observation.own_cache_versions.size))
            differences = differences / node_scale
        return np.asarray([self._aggregate(row) for row in differences], dtype=float)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)

    def node_score(self, observation: NodeObservation) -> float:
        # 节点分数取最值得广播的邻边，并保留低频发送节点的年龄补偿。
        gains = self.edge_gains(observation)
        innovation_score = float(np.max(gains)) if gains.size else 0.0
        return innovation_score + self.age_weight * observation.time_since_last_tx

    def _probability_components(self, score: float, observation: NodeObservation):
        """将分数拆成基础概率、总退避因子和最终概率。"""
        scaled = (score - self.threshold) / self.temperature
        base_probability = self.q_min + (self.q_max - self.q_min) * self._sigmoid(scaled)

        # 退避仅使用本地静默能量检测和自身发送尝试历史，不需要 ACK。
        congestion_factor = math.exp(
            -self.congestion_backoff_weight * observation.congestion_ewma
        )
        attempt_factor = self.attempt_backoff_factor ** observation.consecutive_tx_attempts
        backoff_factor = congestion_factor * attempt_factor
        final_probability = float(np.clip(
            base_probability * backoff_factor,
            self.q_min,
            self.q_max,
        ))
        return float(base_probability), float(backoff_factor), final_probability

    def probability_diagnostics(self, observation: NodeObservation):
        """返回 freshness 分数和概率分解，供降采样诊断使用。"""
        fresh_score = self.node_score(observation)
        base_probability, backoff_factor, final_probability = (
            self._probability_components(fresh_score, observation)
        )
        return {
            "fresh_score": float(fresh_score),
            "orc_score": None,
            "base_probability": base_probability,
            "backoff_factor": backoff_factor,
            "final_probability": final_probability,
        }

    def transmission_probability(self, observation: NodeObservation) -> float:
        return self.probability_diagnostics(observation)["final_probability"]
