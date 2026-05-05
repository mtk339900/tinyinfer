"""
tinyinfer/model.py
───────────────────
CyberShield model — pure NumPy inference, no PyTorch.

Classes:
  DuelingDQN         — generic Dueling DQN with the same architecture
                       as the PyTorch version (Linear→LN→ReLU ×2,
                       then value + advantage heads)
  CyberShieldModel   — loads weights and exposes waf_forward(),
                       net_forward(), and predict()
"""

import numpy as np
from typing import Dict, NamedTuple
from .ops import linear, layer_norm, relu, dueling_aggregate
from .loader import load

# ── Constants ─────────────────────────────────────────────────
WAF_ACTIONS = {0: "ALLOW", 1: "BLOCK", 2: "SCRUB", 3: "RATE_LIMIT", 4: "CAPTCHA"}
NET_ACTIONS = {0: "ALLOW", 1: "BLOCK", 2: "RATE_LIMIT", 3: "ALERT"}


class WafResult(NamedTuple):
    action_id:   int
    action_name: str
    confidence:  float          # max(Q) − min(Q)
    q_values:    np.ndarray     # shape (action_dim,)


# ── DuelingDQN ────────────────────────────────────────────────
class DuelingDQN:
    """
    Mirrors PyTorch DuelingDQN exactly:

      features:
        Linear(in → 512) → LayerNorm(512) → ReLU → Dropout(no-op)
        Linear(512 → 256) → LayerNorm(256) → ReLU → Dropout(no-op)

      value:
        Linear(256 → 128) → ReLU → Linear(128 → 1)

      advantage:
        Linear(256 → 128) → ReLU → Linear(128 → action_dim)

      Q = value + advantage - mean(advantage)
    """

    def __init__(self, weights: Dict[str, np.ndarray], prefix: str):
        """
        prefix: e.g. "waf_dqn", "network_dqn", "server_dqn"
        """
        def w(key): return weights[f"{prefix}.{key}"]

        # Feature trunk
        self.f0_W  = w("features.0.weight")
        self.f0_b  = w("features.0.bias")
        self.ln1_g = w("features.1.weight")
        self.ln1_b = w("features.1.bias")
        self.f4_W  = w("features.4.weight")
        self.f4_b  = w("features.4.bias")
        self.ln5_g = w("features.5.weight")
        self.ln5_b = w("features.5.bias")

        # Value head
        self.v0_W = w("value.0.weight")
        self.v0_b = w("value.0.bias")
        self.v2_W = w("value.2.weight")
        self.v2_b = w("value.2.bias")

        # Advantage head
        self.a0_W = w("advantage.0.weight")
        self.a0_b = w("advantage.0.bias")
        self.a2_W = w("advantage.2.weight")
        self.a2_b = w("advantage.2.bias")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x : (in_features,) float32
        Returns Q-values : (action_dim,) float32
        """
        # Feature trunk
        h = relu(layer_norm(linear(x, self.f0_W, self.f0_b), self.ln1_g, self.ln1_b))
        h = relu(layer_norm(linear(h, self.f4_W, self.f4_b), self.ln5_g, self.ln5_b))

        # Value head
        v = relu(linear(h, self.v0_W, self.v0_b))
        v = linear(v, self.v2_W, self.v2_b)        # shape (1,)

        # Advantage head
        a = relu(linear(h, self.a0_W, self.a0_b))
        a = linear(a, self.a2_W, self.a2_b)         # shape (action_dim,)

        return dueling_aggregate(v, a)

    __call__ = forward


# ── CyberShield ───────────────────────────────────────────────
class CyberShieldModel:
    """
    Load and run CyberShield inference without PyTorch.

    Usage
    -----
    from tinyinfer import CyberShieldModel

    model = CyberShieldModel("cyber_shield_v7.tinyinfer")
    result = model.waf_forward(features)   # features: np.ndarray shape (26,)
    print(result.action_name, result.confidence)
    """

    def __init__(self, model_path: str):
        self.weights = load(model_path)
        self.waf     = DuelingDQN(self.weights, "waf_dqn")
        self.net     = DuelingDQN(self.weights, "network_dqn")
        self.srv     = DuelingDQN(self.weights, "server_dqn")

    # ── WAF DQN (26 features → 5 actions) ────────────────────
    def waf_forward(self, features: np.ndarray) -> WafResult:
        """
        features : np.ndarray shape (26,) — output of extract_features_fast()
        """
        features = np.asarray(features, dtype=np.float32)
        q = self.waf(features)
        action_id   = int(q.argmax())
        confidence  = float(q.max() - q.min())
        return WafResult(
            action_id   = action_id,
            action_name = WAF_ACTIONS[action_id],
            confidence  = confidence,
            q_values    = q,
        )

    # ── Network DQN (22 features → 4 actions) ─────────────────
    def net_forward(self, features: np.ndarray) -> WafResult:
        features = np.asarray(features, dtype=np.float32)
        q = self.net(features)
        action_id = int(q.argmax())
        return WafResult(
            action_id   = action_id,
            action_name = NET_ACTIONS[action_id],
            confidence  = float(q.max() - q.min()),
            q_values    = q,
        )

    # ── High-level predict ────────────────────────────────────
    def predict(self, features: np.ndarray) -> dict:
        """
        Full inference: returns a dict with action, confidence, q_values.
        Compatible with CyberShield v7 pipeline.
        """
        r = self.waf_forward(features)
        return {
            "action_id":   r.action_id,
            "action":      r.action_name,
            "confidence":  r.confidence,
            "q_values":    r.q_values.tolist(),
            "blocked":     r.action_id != 0,
        }
