"""导出配对多策略实验执行入口。"""

from .runner import ExperimentResult, annotate_bottleneck_importance, run_experiment

__all__ = ["ExperimentResult", "annotate_bottleneck_importance", "run_experiment"]
