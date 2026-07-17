"""导出局部特征编码、共享 CTDE-PPO 模型与训练入口。"""

from .features import (
    EDGE_FEATURE_DIM, GLOBAL_STATE_DIM, NODE_FEATURE_DIM, EncodedObservations,
    encode_observations,
)

__all__ = [
    "EDGE_FEATURE_DIM", "GLOBAL_STATE_DIM", "NODE_FEATURE_DIM",
    "EncodedObservations", "encode_observations",
]
