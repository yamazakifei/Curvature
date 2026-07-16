"""实现固定路径损耗、静态阴影和逐时隙完整 Rayleigh 功率衰落矩阵。"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def dbm_to_mw(value_dbm):
    return np.power(10.0, np.asarray(value_dbm) / 10.0)


def mw_to_dbm(value_mw):
    return 10.0 * np.log10(np.asarray(value_mw))


def shannon_sinr_threshold_db(packet_bits: int, bandwidth_hz: float, slot_duration_s: float) -> float:
    """可选地由 Shannon 速率关系推导 SINR 阈值；版本 1 默认仍直接配置阈值。"""
    if packet_bits <= 0 or bandwidth_hz <= 0 or slot_duration_s <= 0:
        raise ValueError("packet length, bandwidth and slot duration must be positive")
    spectral_efficiency = packet_bits / (bandwidth_hz * slot_duration_s)
    threshold_linear = np.power(2.0, spectral_efficiency) - 1.0
    return float(10.0 * np.log10(threshold_linear))


@dataclass(frozen=True)
class ChannelParameters:
    tx_power_dbm: float = 20.0
    pathloss_reference_db: float = 40.0
    reference_distance_m: float = 1.0
    pathloss_exponent: float = 3.0
    shadowing_std_db: float = 0.0
    reciprocal_shadowing: bool = True
    rayleigh_fading: bool = True
    noise_psd_dbm_per_hz: float = -174.0
    bandwidth_hz: float = 5_000_000.0
    noise_figure_db: float = 7.0
    sinr_threshold_db: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        known = {field: values[field] for field in cls.__dataclass_fields__ if field in values}
        instance = cls(**known)
        if instance.reference_distance_m <= 0 or instance.bandwidth_hz <= 0:
            raise ValueError("reference distance and bandwidth must be positive")
        if instance.pathloss_exponent < 0 or instance.shadowing_std_db < 0:
            raise ValueError("pathloss exponent and shadowing standard deviation must be nonnegative")
        return instance

    @property
    def tx_power_mw(self) -> float:
        return float(dbm_to_mw(self.tx_power_dbm))

    @property
    def noise_power_mw(self) -> float:
        noise_dbm = (
            self.noise_psd_dbm_per_hz
            + 10.0 * np.log10(self.bandwidth_hz)
            + self.noise_figure_db
        )
        return float(dbm_to_mw(noise_dbm))

    @property
    def sinr_threshold_linear(self) -> float:
        return float(np.power(10.0, self.sinr_threshold_db / 10.0))


class PropagationModel:
    def __init__(
        self,
        positions: np.ndarray,
        parameters: ChannelParameters,
        shadowing_rng: np.random.Generator,
        shadowing_db: np.ndarray = None,
    ) -> None:
        self.positions = np.asarray(positions, dtype=float)
        self.parameters = parameters
        differences = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        self.distances = np.linalg.norm(differences, axis=2)
        n_nodes = self.positions.shape[0]
        if shadowing_db is None:
            shadowing_db = self._sample_shadowing(n_nodes, shadowing_rng)
        self.shadowing_db = np.asarray(shadowing_db, dtype=float).copy()
        if self.shadowing_db.shape != (n_nodes, n_nodes):
            raise ValueError("shadowing matrix must have shape [N, N]")

        effective_distance = np.maximum(self.distances, parameters.reference_distance_m)
        pathloss_db = (
            parameters.pathloss_reference_db
            + 10.0 * parameters.pathloss_exponent
            * np.log10(effective_distance / parameters.reference_distance_m)
            + self.shadowing_db
        )
        self.mean_rx_power_mw = parameters.tx_power_mw * np.power(10.0, -pathloss_db / 10.0)
        np.fill_diagonal(self.mean_rx_power_mw, 0.0)

    def _sample_shadowing(self, n_nodes: int, rng: np.random.Generator) -> np.ndarray:
        std = self.parameters.shadowing_std_db
        if self.parameters.reciprocal_shadowing:
            samples = rng.normal(0.0, std, size=(n_nodes, n_nodes))
            upper = np.triu(samples, 1)
            shadowing = upper + upper.T
        else:
            shadowing = rng.normal(0.0, std, size=(n_nodes, n_nodes))
            np.fill_diagonal(shadowing, 0.0)
        return shadowing

    def received_power(self, fading_rng: np.random.Generator) -> np.ndarray:
        """无论动作如何都生成完整 N×N 衰落矩阵。"""
        n_nodes = self.positions.shape[0]
        if self.parameters.rayleigh_fading:
            fading = fading_rng.exponential(1.0, size=(n_nodes, n_nodes))
        else:
            fading = np.ones((n_nodes, n_nodes), dtype=float)
        received = self.mean_rx_power_mw * fading
        np.fill_diagonal(received, 0.0)
        return received
