"""Pure NumPy measurements for ball-policy input sensitivity diagnostics."""
import numpy as np


def position_change_mm(a, b):
    """Euclidean position difference in millimetres for inputs in metres."""
    return np.linalg.norm(np.asarray(a)-np.asarray(b), axis=-1)*1000.
