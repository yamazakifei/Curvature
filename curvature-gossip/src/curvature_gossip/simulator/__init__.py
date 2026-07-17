"""导出节点本地观测；主仿真引擎在 Milestone 6 接入。"""

from .engine import GossipSimulator, SimulationParameters, SimulationResult, StepOutcome
from .observations import NodeObservation, ObservationBuilder

__all__ = [
    "GossipSimulator", "NodeObservation", "ObservationBuilder", "SimulationParameters",
    "SimulationResult", "StepOutcome",
]
