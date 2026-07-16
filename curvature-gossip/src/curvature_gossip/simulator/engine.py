"""运行分布式完整缓存 gossip，并按配置降采样节点概率诊断。"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from ..channel import PropagationModel, StrongestSignalDecoder
from ..curvature import CurvatureResult
from ..metrics import DisseminationTracker, MetricsCollector
from ..policies.base import DistributedBroadcastPolicy
from ..state import LocalKnowledge, VersionState
from ..topology.base import Topology
from .observations import ObservationBuilder


@dataclass(frozen=True)
class SimulationParameters:
    slots: int
    update_probability: float
    warmup_slots: int = 0
    trace_stride: int = 1
    node_diagnostics_stride: int = 0

    def __post_init__(self):
        if self.slots < 1 or not 0 <= self.warmup_slots < self.slots:
            raise ValueError("slots must be positive and warmup smaller than slots")
        if not 0.0 <= self.update_probability <= 1.0:
            raise ValueError("update_probability must be in [0, 1]")
        if self.node_diagnostics_stride < 0:
            raise ValueError("node_diagnostics_stride must be nonnegative")


@dataclass
class SimulationResult:
    summary: Mapping[str, Any]
    per_slot: tuple
    node_diagnostics: tuple
    dissemination_records: tuple
    final_state: VersionState


class GossipSimulator:
    def __init__(
        self,
        topology: Topology,
        curvature: CurvatureResult,
        bottleneck_importance: Mapping,
        propagation: PropagationModel,
        policy: DistributedBroadcastPolicy,
        parameters: SimulationParameters,
        update_rng: np.random.Generator,
        fading_rng: np.random.Generator,
        policy_rng: np.random.Generator,
    ) -> None:
        self.topology = topology
        self.curvature = curvature
        self.propagation = propagation
        self.policy = policy
        self.parameters = parameters
        self.update_rng = update_rng
        self.fading_rng = fading_rng
        self.policy_rng = policy_rng
        self.state = VersionState(topology.graph.number_of_nodes())
        self.knowledge = LocalKnowledge(topology.graph)
        self.observations = ObservationBuilder(topology, curvature, bottleneck_importance)
        channel = propagation.parameters
        self.decoder = StrongestSignalDecoder(
            topology.graph, channel.noise_power_mw, channel.sinr_threshold_linear
        )
        self.tracker = DisseminationTracker()
        self.metrics = MetricsCollector(
            self.state.n_nodes,
            warmup_slots=parameters.warmup_slots,
            trace_stride=parameters.trace_stride,
            node_labels=topology.node_labels,
        )
        self.node_diagnostics = []

    def step(self, slot: int) -> None:
        # 1. 源更新；源节点立即知道自己的新版本。
        _, events = self.state.generate_source_updates(
            slot, self.parameters.update_probability, self.update_rng
        )
        self.tracker.register(events)
        # 2. 冻结本时隙所有可能发送的数据包。
        packet_versions, packet_slots = self.state.packet_snapshot()
        # 3-4. 每个节点只基于本地观测独立计算概率并采样动作。
        probabilities = np.empty(self.state.n_nodes, dtype=float)
        diagnostics_method = getattr(self.policy, "probability_diagnostics", None)
        diagnostics_stride = self.parameters.node_diagnostics_stride
        sample_diagnostics = (
            callable(diagnostics_method)
            and diagnostics_stride > 0
            and slot >= self.parameters.warmup_slots
            and (slot - self.parameters.warmup_slots) % diagnostics_stride == 0
        )
        for node in range(self.state.n_nodes):
            observation = self.observations.build(node, slot, self.state, self.knowledge)
            if sample_diagnostics:
                # 诊断方法同时返回策略实际使用的最终概率，避免重复计算。
                diagnostics = diagnostics_method(observation)
                probability = diagnostics["final_probability"]
                self.node_diagnostics.append({
                    "slot": int(slot),
                    "node": int(node),
                    "fresh_score": diagnostics.get("fresh_score"),
                    "orc_score": diagnostics.get("orc_score"),
                    "base_probability": diagnostics.get("base_probability"),
                    "backoff_factor": diagnostics.get("backoff_factor"),
                    "final_probability": diagnostics.get("final_probability"),
                })
            else:
                probability = self.policy.transmission_probability(observation)
            probabilities[node] = np.clip(probability, 0.0, 1.0)
        actions = self.policy_rng.random(self.state.n_nodes) < probabilities
        # 5-6. 完整生成信道随机量，再按最强候选和全网干扰解码。
        received_power = self.propagation.received_power(self.fading_rng)
        decode = self.decoder.resolve(actions, received_power)
        # 7. 所有成功数据包基于冻结快照同时合并。
        merge = self.state.merge_decoded_snapshots(
            decode.decoded_senders, packet_versions, packet_slots
        )
        # 8. 仅从成功侦听的数据包更新本地邻居估计。
        self.knowledge.update_from_decodes(decode.decoded_senders, packet_versions)
        self.knowledge.update_slot_history(
            slot,
            actions,
            decode.interference_plus_noise,
            self.decoder.noise_power_mw,
            getattr(self.policy, "congestion_ewma_alpha", 0.8),
        )
        # 9. 在合并完成后更新传播完成状态与标量指标。
        self.tracker.update_completions(slot, self.state.cache_versions)
        self.metrics.record(slot, self.state, actions, decode, merge)

    def run(self) -> SimulationResult:
        for slot in range(self.parameters.slots):
            self.step(slot)
        summary = self.metrics.summary()
        summary.update(self.tracker.summary())
        return SimulationResult(
            summary=summary,
            per_slot=tuple(self.metrics.per_slot),
            node_diagnostics=tuple(self.node_diagnostics),
            dissemination_records=self.tracker.records,
            final_state=self.state,
        )
