"""Provide the Stage-0 fixed-alpha, local AF3 broadcast baseline.

The policy is intentionally parameter-free: it converts only the node's
incident AF3 bottleneck score into a Bernoulli probability.  It is therefore
safe to evaluate before the trainable Stage-1 actor is introduced.
"""

import math

import numpy as np

from ..simulator.observations import NodeObservation
from .base import DistributedBroadcastPolicy


class CurvatureFixedAlphaPolicy(DistributedBroadcastPolicy):
    """Use a fixed curvature slope with no residual (second) layer.

    The local score is ``max_j b_ij``.  Supplying ``alpha_override=0.0`` is
    the explicit strict-degeneration path and returns ``target_tx_ratio`` for
    every node, independent of its incident curvature.
    """

    def __init__(
        self,
        target_tx_ratio: float = 0.1,
        alpha: float = 0.0,
        curvature_center: float = 0.0,
        alpha_override=None,
    ) -> None:
        self.target_tx_ratio = float(target_tx_ratio)
        self.alpha = float(alpha)
        self.curvature_center = float(curvature_center)
        self.alpha_override = None if alpha_override is None else float(alpha_override)
        if not 0.0 < self.target_tx_ratio < 1.0:
            raise ValueError("target_tx_ratio must be in (0, 1) for logit evaluation")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and nonnegative")
        if not np.isfinite(self.curvature_center):
            raise ValueError("curvature_center must be finite")
        if self.alpha_override is not None and (
            not np.isfinite(self.alpha_override) or self.alpha_override < 0.0
        ):
            raise ValueError("alpha_override must be null or a finite nonnegative value")
        self._base_logit = math.log(self.target_tx_ratio / (1.0 - self.target_tx_ratio))

    @property
    def effective_alpha(self) -> float:
        """Return the configured slope, including the strict override when present."""
        return self.alpha if self.alpha_override is None else self.alpha_override

    @staticmethod
    def _sigmoid(logit: float) -> float:
        """Evaluate a scalar sigmoid without overflow at extreme fixed slopes."""
        if logit >= 0.0:
            return float(1.0 / (1.0 + math.exp(-logit)))
        exp_logit = math.exp(logit)
        return float(exp_logit / (1.0 + exp_logit))

    def curvature_score(self, observation: NodeObservation) -> float:
        """Return the local incident AF3 bottleneck maximum, or zero if isolated."""
        values = np.asarray([
            observation.incident_bottleneck_importance[neighbor]
            for neighbor in observation.neighbor_ids
        ], dtype=float)
        if not values.size:
            return 0.0
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("incident bottleneck importance must be finite and in [0, 1]")
        return float(np.max(values))

    def transmission_probability(self, observation: NodeObservation) -> float:
        # Keep the exact zero-alpha path separate so q_i == b up to float conversion.
        if self.effective_alpha == 0.0:
            return self.target_tx_ratio
        logit = self._base_logit + self.effective_alpha * (
            self.curvature_score(observation) - self.curvature_center
        )
        return self._sigmoid(logit)
