"""
TinyInfer — Zero-dependency ONNX-style inference engine for CyberShield.
Pure Python + NumPy. No PyTorch required at runtime.
"""

from .loader import load, info
from .model  import CyberShieldModel, DuelingDQN, WafResult
from .ops    import linear, layer_norm, relu, dueling_aggregate

__version__ = "1.0.0"
__all__ = [
    "CyberShieldModel", "DuelingDQN", "WafResult",
    "load", "info",
    "linear", "layer_norm", "relu", "dueling_aggregate",
]
