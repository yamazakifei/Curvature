"""用 SeedSequence 和稳定字符串标签派生互不干扰、与策略排列无关的随机流。"""

import hashlib
from typing import Any

import numpy as np


def _stable_uint32(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value) & 0xFFFFFFFF
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def make_rng(master_seed: int, *labels: Any) -> np.random.Generator:
    entropy = [_stable_uint32(master_seed)] + [_stable_uint32(label) for label in labels]
    return np.random.default_rng(np.random.SeedSequence(entropy))

