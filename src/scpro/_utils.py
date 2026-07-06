from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from scipy import sparse


def set_random_seed(seed: int | None) -> None:
    """Set common random seeds without importing heavy frameworks unless present."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def as_dense(x: Any, *, dtype: Any | None = np.float32) -> np.ndarray:
    """Return a dense NumPy array from dense or sparse matrix-like data."""
    if sparse.issparse(x):
        arr = x.toarray()
    else:
        arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    return dict(obj)
