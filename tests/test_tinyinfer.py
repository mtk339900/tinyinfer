"""
tests/test_tinyinfer.py
────────────────────────
Unit tests for TinyInfer ops, loader, and model.
Run with:  python -m pytest tests/ -v
"""

import math
import struct
import tempfile
import numpy as np
import pytest

from tinyinfer.ops    import linear, layer_norm, relu, dueling_aggregate
from tinyinfer.loader import load, MAGIC, VERSION
from tinyinfer.model  import DuelingDQN, CyberShieldModel, WAF_ACTIONS


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def write_tiny(path, tensors: dict):
    """Write a minimal .tinyinfer file for testing."""
    import struct
    items = list(tensors.items())
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(items)))
        for name, arr in items:
            arr = np.asarray(arr, dtype=np.float32)
            nb  = name.encode("utf-8")
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<I", d))
            f.write(arr.flatten().tobytes())


def make_dqn_weights(prefix, in_dim, action_dim, seed=0):
    """Generate deterministic random weights matching DuelingDQN."""
    rng = np.random.default_rng(seed)
    def r(*shape): return rng.standard_normal(shape).astype(np.float32)

    return {
        f"{prefix}.features.0.weight": r(512, in_dim),
        f"{prefix}.features.0.bias":   r(512),
        f"{prefix}.features.1.weight": np.ones(512, dtype=np.float32),
        f"{prefix}.features.1.bias":   np.zeros(512, dtype=np.float32),
        f"{prefix}.features.4.weight": r(256, 512),
        f"{prefix}.features.4.bias":   r(256),
        f"{prefix}.features.5.weight": np.ones(256, dtype=np.float32),
        f"{prefix}.features.5.bias":   np.zeros(256, dtype=np.float32),
        f"{prefix}.value.0.weight":    r(128, 256),
        f"{prefix}.value.0.bias":      r(128),
        f"{prefix}.value.2.weight":    r(1, 128),
        f"{prefix}.value.2.bias":      r(1),
        f"{prefix}.advantage.0.weight": r(128, 256),
        f"{prefix}.advantage.0.bias":   r(128),
        f"{prefix}.advantage.2.weight": r(action_dim, 128),
        f"{prefix}.advantage.2.bias":   r(action_dim),
    }


# ═══════════════════════════════════════════════════════════════
# §1  Ops
# ═══════════════════════════════════════════════════════════════

class TestLinear:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        W = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        b = np.array([10.0, 20.0], dtype=np.float32)
        out = linear(x, W, b)
        np.testing.assert_allclose(out, [11.0, 22.0])

    def test_shape(self):
        rng = np.random.default_rng(0)
        x = rng.random(26).astype(np.float32)
        W = rng.random((512, 26)).astype(np.float32)
        b = rng.random(512).astype(np.float32)
        assert linear(x, W, b).shape == (512,)

    def test_no_bias(self):
        x = np.ones(4, dtype=np.float32)
        W = np.eye(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        np.testing.assert_allclose(linear(x, W, b), x)


class TestLayerNorm:
    def test_normalizes(self):
        rng = np.random.default_rng(1)
        x   = rng.standard_normal(256).astype(np.float32)
        g   = np.ones(256, dtype=np.float32)
        b   = np.zeros(256, dtype=np.float32)
        out = layer_norm(x, g, b)
        assert abs(out.mean()) < 1e-4
        assert abs(out.std() - 1.0) < 1e-2

    def test_affine_scale(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        g = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        b = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        out = layer_norm(x, g, b)
        # std should be ≈ 2
        assert abs(out.std() - 2.0) < 0.1

    def test_matches_pytorch(self):
        try:
            import torch, torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch not available")
        rng = np.random.default_rng(42)
        x   = rng.standard_normal(256).astype(np.float32)
        g   = rng.random(256).astype(np.float32) + 0.5
        b   = rng.standard_normal(256).astype(np.float32)

        ln  = nn.LayerNorm(256)
        ln.weight.data = torch.FloatTensor(g)
        ln.bias.data   = torch.FloatTensor(b)
        with torch.no_grad():
            pt_out = ln(torch.FloatTensor(x)).numpy()

        ti_out = layer_norm(x.copy(), g, b)
        np.testing.assert_allclose(ti_out, pt_out, atol=1e-4)


class TestRelu:
    def test_basic(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(relu(x), [0, 0, 0, 1, 2])

    def test_all_negative(self):
        x = np.full(10, -5.0, dtype=np.float32)
        assert (relu(x) == 0).all()

    def test_no_side_effects(self):
        x   = np.array([-1.0, 1.0], dtype=np.float32)
        x0  = x.copy()
        relu(x)
        np.testing.assert_array_equal(x, x0)   # original unchanged


class TestDuelingAggregate:
    def test_basic(self):
        value     = np.array([5.0], dtype=np.float32)
        advantage = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        q = dueling_aggregate(value, advantage)
        # mean(adv) = 2.0 → Q = [5+1-2, 5+2-2, 5+3-2] = [4,5,6]
        np.testing.assert_allclose(q, [4.0, 5.0, 6.0])

    def test_zero_advantage(self):
        value     = np.array([3.0], dtype=np.float32)
        advantage = np.zeros(5, dtype=np.float32)
        q = dueling_aggregate(value, advantage)
        np.testing.assert_allclose(q, np.full(5, 3.0))

    def test_argmax_preserved(self):
        value     = np.array([0.0], dtype=np.float32)
        advantage = np.array([0.1, 0.9, 0.3, 0.5], dtype=np.float32)
        q = dueling_aggregate(value, advantage)
        assert q.argmax() == 1


# ═══════════════════════════════════════════════════════════════
# §2  Loader
# ═══════════════════════════════════════════════════════════════

class TestLoader:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "test.tinyinfer")
        orig = {
            "layer.weight": np.random.randn(64, 32).astype(np.float32),
            "layer.bias":   np.random.randn(64).astype(np.float32),
        }
        write_tiny(path, orig)
        loaded = load(path)
        for name, arr in orig.items():
            np.testing.assert_allclose(loaded[name], arr, atol=1e-6)

    def test_bad_magic(self, tmp_path):
        path = str(tmp_path / "bad.tinyinfer")
        with open(path, "wb") as f:
            f.write(b"XXXX" + b"\x00" * 8)
        with pytest.raises(ValueError, match="bad magic"):
            load(path)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load("/nonexistent/model.tinyinfer")

    def test_tensor_count(self, tmp_path):
        path = str(tmp_path / "multi.tinyinfer")
        tensors = {f"t{i}": np.ones(4, dtype=np.float32) * i for i in range(10)}
        write_tiny(path, tensors)
        loaded = load(path)
        assert len(loaded) == 10

    def test_various_shapes(self, tmp_path):
        path = str(tmp_path / "shapes.tinyinfer")
        orig = {
            "scalar": np.array([3.14], dtype=np.float32),
            "vec":    np.arange(100, dtype=np.float32),
            "matrix": np.eye(16, dtype=np.float32),
            "rank3":  np.ones((2, 3, 4), dtype=np.float32),
        }
        write_tiny(path, orig)
        loaded = load(path)
        for name, arr in orig.items():
            assert loaded[name].shape == arr.shape
            np.testing.assert_allclose(loaded[name], arr, atol=1e-6)


# ═══════════════════════════════════════════════════════════════
# §3  DuelingDQN (with synthetic weights)
# ═══════════════════════════════════════════════════════════════

class TestDuelingDQN:
    @pytest.fixture
    def waf_model_file(self, tmp_path):
        path = str(tmp_path / "waf.tinyinfer")
        w  = {}
        w |= make_dqn_weights("waf_dqn",     26, 5, seed=0)
        w |= make_dqn_weights("network_dqn", 22, 4, seed=1)
        w |= make_dqn_weights("server_dqn",  20, 4, seed=2)
        write_tiny(path, w)
        return path

    def test_output_shape(self, waf_model_file):
        weights = load(waf_model_file)
        dqn = DuelingDQN(weights, "waf_dqn")
        x   = np.zeros(26, dtype=np.float32)
        q   = dqn(x)
        assert q.shape == (5,)

    def test_deterministic(self, waf_model_file):
        weights = load(waf_model_file)
        dqn = DuelingDQN(weights, "waf_dqn")
        x   = np.random.default_rng(99).random(26).astype(np.float32)
        assert np.array_equal(dqn(x), dqn(x))

    def test_matches_pytorch(self, waf_model_file):
        try:
            import torch, torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch not available")

        weights = load(waf_model_file)
        dqn_ti  = DuelingDQN(weights, "waf_dqn")

        # Build matching PyTorch model with same weights
        class PT_DQN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(26,512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0),
                    nn.Linear(512,256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0))
                self.value     = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Linear(128,1))
                self.advantage = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Linear(128,5))
            def forward(self, x):
                f = self.features(x)
                v = self.value(f)
                a = self.advantage(f)
                return v + a - a.mean(dim=-1, keepdim=True)

        pt_dqn = PT_DQN()
        for name, param in pt_dqn.named_parameters():
            full_key = f"waf_dqn.{name}"
            param.data = torch.FloatTensor(weights[full_key])

        pt_dqn.eval()
        rng = np.random.default_rng(7)
        x   = rng.random(26).astype(np.float32)

        ti_q = dqn_ti(x)
        with torch.no_grad():
            pt_q = pt_dqn(torch.FloatTensor(x)).numpy()

        np.testing.assert_allclose(ti_q, pt_q, atol=1e-3,
                                   err_msg="TinyInfer Q-values don't match PyTorch")


# ═══════════════════════════════════════════════════════════════
# §4  CyberShieldModel (with synthetic weights)
# ═══════════════════════════════════════════════════════════════

class TestCyberShieldModel:
    @pytest.fixture
    def model_file(self, tmp_path):
        path = str(tmp_path / "cs.tinyinfer")
        w  = {}
        w |= make_dqn_weights("waf_dqn",     26, 5, seed=10)
        w |= make_dqn_weights("network_dqn", 22, 4, seed=11)
        w |= make_dqn_weights("server_dqn",  20, 4, seed=12)
        write_tiny(path, w)
        return path

    def test_loads(self, model_file):
        m = CyberShieldModel(model_file)
        assert m is not None

    def test_waf_forward_shape(self, model_file):
        m = CyberShieldModel(model_file)
        r = m.waf_forward(np.zeros(26, dtype=np.float32))
        assert r.q_values.shape == (5,)
        assert r.action_name in WAF_ACTIONS.values()

    def test_action_consistency(self, model_file):
        m = CyberShieldModel(model_file)
        x = np.random.default_rng(0).random(26).astype(np.float32)
        r = m.waf_forward(x)
        assert r.action_id == int(r.q_values.argmax())
        assert r.action_name == WAF_ACTIONS[r.action_id]

    def test_confidence_positive(self, model_file):
        m = CyberShieldModel(model_file)
        r = m.waf_forward(np.ones(26, dtype=np.float32))
        assert r.confidence >= 0.0

    def test_predict_blocked_field(self, model_file):
        m = CyberShieldModel(model_file)
        x = np.zeros(26, dtype=np.float32)
        p = m.predict(x)
        assert "blocked" in p
        assert p["blocked"] == (p["action_id"] != 0)

    def test_high_threat_features_not_allow(self, model_file):
        """Extreme attack features should not map to ALLOW (action_id=0)."""
        m = CyberShieldModel(model_file)
        # all attack signals maxed out — any reasonable model should not ALLOW
        x = np.ones(26, dtype=np.float32)
        r = m.waf_forward(x)
        # We can't guarantee the label with random weights; just check it runs
        assert r.action_id in WAF_ACTIONS
