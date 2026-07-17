"""运行分布式 gossip，并提供启发式评估与 CTDE 训练共用的分步接口。"""

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

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
    target_tx_ratio: float = 1.0
    per_node_cap_multiplier: float = 1.0

    def __post_init__(self):
        if self.slots < 1 or not 0 <= self.warmup_slots < self.slots:
            raise ValueError("slots must be positive and warmup smaller than slots")
        if not 0.0 <= self.update_probability <= 1.0:
            raise ValueError("update_probability must be in [0, 1]")
        if self.node_diagnostics_stride < 0:
            raise ValueError("node_diagnostics_stride must be nonnegative")
        if not 0.0 < self.target_tx_ratio <= 1.0:
            raise ValueError("target_tx_ratio must be in (0, 1]")
        if self.per_node_cap_multiplier <= 0.0:
            raise ValueError("per_node_cap_multiplier must be positive")


@dataclass
class SimulationResult:
    summary: Mapping[str, Any]
    per_slot: tuple
    node_diagnostics: tuple
    dissemination_records: tuple
    final_state: VersionState


@dataclass(frozen=True)
class StepOutcome:
    """训练端可见的全局结果；分布式 actor 不读取这些字段。"""

    slot: int
    reward: float
    transmission_cost: float
    mean_vaoi: float
    global_state: np.ndarray


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
        node_cap = min(
            1.0,
            parameters.target_tx_ratio * parameters.per_node_cap_multiplier,
        )
        self.knowledge = LocalKnowledge(topology.graph, broadcast_limit=node_cap)
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
            broadcast_limits=self.knowledge.broadcast_limit,
        )
        self.node_diagnostics = []
        self._pending = None

    def begin_step(self, slot: int) -> Tuple:
        """生成源更新、冻结包快照并返回所有节点的本地观测。"""
        if self._pending is not None:
            raise RuntimeError("complete_step must finish before the next begin_step")
        _, events = self.state.generate_source_updates(
            slot, self.parameters.update_probability, self.update_rng
        )
        self.tracker.register(events)
        packet_versions, packet_slots = self.state.packet_snapshot()
        observations = tuple(
            self.observations.build(node, slot, self.state, self.knowledge)
            for node in range(self.state.n_nodes)
        )
        self._pending = (int(slot), packet_versions, packet_slots, observations)
        return observations

    def complete_step(self, actions: np.ndarray) -> StepOutcome:
        """完成信道、缓存合并和指标更新，并返回 CTDE 集中训练量。"""
        if self._pending is None:
            raise RuntimeError("begin_step must be called before complete_step")
        slot, packet_versions, packet_slots, _ = self._pending
        actions = np.asarray(actions, dtype=bool)
        if actions.shape != (self.state.n_nodes,):
            raise ValueError("actions must have shape [N]")

        received_power = self.propagation.received_power(self.fading_rng)
        decode = self.decoder.resolve(actions, received_power)
        merge = self.state.merge_decoded_snapshots(
            decode.decoded_senders, packet_versions, packet_slots
        )
        self.knowledge.update_from_decodes(decode.decoded_senders, packet_versions)
        self.knowledge.update_slot_history(
            slot,
            actions,
            decode.interference_plus_noise,
            self.decoder.noise_power_mw,
            getattr(self.policy, "congestion_ewma_alpha", 0.8),
        )
        self.tracker.update_completions(slot, self.state.cache_versions)
        self.metrics.record(
            slot, self.state, actions, decode, merge, self.knowledge.broadcast_debt
        )

        global_state = self.centralized_state(float(np.mean(actions)))
        mean_vaoi = float(global_state[0])
        tx_ratio = float(global_state[4])
        self._pending = None
        return StepOutcome(
            slot=slot,
            reward=-mean_vaoi,
            transmission_cost=tx_ratio,
            mean_vaoi=mean_vaoi,
            global_state=global_state,
        )

    def centralized_state(self, last_tx_ratio: float = 0.0) -> np.ndarray:
        """构造训练 critic 的全局摘要；该方法不得被分布式策略调用。"""
        version_age = self.state.version_age().astype(float)
        ages = version_age[~np.eye(self.state.n_nodes, dtype=bool)]
        tail_count = max(1, int(np.ceil(0.05 * ages.size)))
        tail = np.partition(ages, ages.size - tail_count)[-tail_count:]
        mean_vaoi = float(np.mean(ages))
        return np.asarray([
            mean_vaoi,
            float(np.max(ages)),
            float(np.percentile(ages, 95)),
            float(np.mean(tail)),
            float(last_tx_ratio),
            float(np.mean(self.knowledge.broadcast_debt)),
        ], dtype=np.float32)

    def step(self, slot: int) -> None:
        """按现有策略计算概率；保持原有启发式实验入口兼容。"""
        observations = self.begin_step(slot)
        probabilities = np.empty(self.state.n_nodes, dtype=float)
        diagnostics_method = getattr(self.policy, "probability_diagnostics", None)
        diagnostics_stride = self.parameters.node_diagnostics_stride
        sample_diagnostics = (
            callable(diagnostics_method)
            and diagnostics_stride > 0
            and slot >= self.parameters.warmup_slots
            and (slot - self.parameters.warmup_slots) % diagnostics_stride == 0
        )
        if not sample_diagnostics:
            probabilities = np.asarray(
                self.policy.transmission_probabilities(observations), dtype=float
            )
            if probabilities.shape != (self.state.n_nodes,):
                raise ValueError("policy batch probabilities must have shape [N]")
            probabilities = np.clip(probabilities, 0.0, 1.0)
        else:
            for node, observation in enumerate(observations):
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
                probabilities[node] = np.clip(probability, 0.0, 1.0)
        actions = self.policy_rng.random(self.state.n_nodes) < probabilities
        self.complete_step(actions)

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
