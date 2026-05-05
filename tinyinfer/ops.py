"""
tinyinfer/ops.py
─────────────────
Pure-NumPy implementations of every op used by CyberShield.
No PyTorch, no ONNX Runtime — just numpy.

Ops:
  linear(x, W, b)         → np.ndarray
  layer_norm(x, g, b)     → np.ndarray
  relu(x)                 → np.ndarray
  dueling_aggregate(v, a) → np.ndarray   Q = V + A - mean(A)
"""

import numpy as np


def linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Fully-connected layer.
    x : (in_features,)
    W : (out_features, in_features)
    b : (out_features,)
    """
    return W @ x + b


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """
    LayerNorm over the full vector x (matches PyTorch LayerNorm default).
    """
    mean = x.mean()
    var  = x.var()
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def dueling_aggregate(value: np.ndarray, advantage: np.ndarray) -> np.ndarray:
    """
    Dueling DQN Q-value aggregation:
      Q[i] = value[0] + advantage[i] − mean(advantage)
    """
    return value[0] + advantage - advantage.mean()
