"""Causal XYZ-only diagnostic filter; no state, image or expert input."""
import numpy as np


def smooth_actions(actions, alpha=0.25):
    raw = np.asarray(actions, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 29 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError('Expected finite Nx29 actions')
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError('alpha must be in (0,1]')
    result = raw.copy()
    for i in range(1, len(raw)):
        result[i, :3] = alpha * raw[i, :3] + (1-alpha) * result[i-1, :3]
    return result
