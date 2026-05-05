# TinyInfer

A minimal, dependency-free inference engine for **CyberShield** — written in pure Python + NumPy. No PyTorch required at runtime.

> Export once with PyTorch. Run forever with just NumPy.

---

## Why

PyTorch adds ~300 MB and ~2 s cold-start overhead to every deployment. CyberShield's WAF runs on edge servers and containers where that's unacceptable. TinyInfer loads the same weights in a custom binary format and runs inference with NumPy alone.

---

## Install

```bash
git clone https://github.com/mtk339900/tinyinfer
cd tinyinfer
pip install numpy          # only runtime dependency
pip install torch          # only needed for export step
pip install pytest         # only needed for tests
```

---

## Quickstart

### 1 — Export the model (once, needs PyTorch)

```bash
python tools/export.py \
    --input  cyber_shield_v7_tested.pth \
    --output cyber_shield_v7.tinyinfer
```

### 2 — Run inference (no PyTorch)

```python
import numpy as np
from tinyinfer import CyberShieldModel

model    = CyberShieldModel("cyber_shield_v7.tinyinfer")
features = np.zeros(26, dtype=np.float32)   # your extract_features_fast() output
result   = model.waf_forward(features)

print(result.action_name)   # "ALLOW" / "BLOCK" / "SCRUB" / ...
print(result.confidence)    # max(Q) − min(Q)
print(result.q_values)      # raw Q-values, shape (5,)
```

---

## CLI

```bash
# Inspect all tensors in the model file
python cli.py cyber_shield_v7.tinyinfer inspect

# Verify against built-in test vectors
python cli.py cyber_shield_v7.tinyinfer verify

# Single inference (26 feature values)
python cli.py cyber_shield_v7.tinyinfer infer \
    0.1 0.02 0.95 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.02

# Latency benchmark
python cli.py cyber_shield_v7.tinyinfer bench --n 50000
```

---

## Verify numerical agreement with PyTorch

```bash
python tools/verify_vs_pytorch.py \
    --tinyinfer cyber_shield_v7.tinyinfer \
    --pth       cyber_shield_v7_tested.pth
```

Expected output:
```
  Action agreement : 20/20  (100.0%)
  Max Q-value diff : 0.000031
  Mean Q-diff      : 0.000008
  Numerical match  : ✅ PASS
```

---

## Architecture

```
tinyinfer/
├── tinyinfer/
│   ├── __init__.py     # public API
│   ├── ops.py          # linear, layer_norm, relu, dueling_aggregate
│   ├── loader.py       # .tinyinfer binary parser (zero-copy via np.frombuffer)
│   └── model.py        # DuelingDQN + CyberShieldModel
├── tools/
│   ├── export.py               # .pth → .tinyinfer
│   └── verify_vs_pytorch.py    # numerical agreement check
├── tests/
│   └── test_tinyinfer.py       # 26 unit tests
└── cli.py                      # CLI: inspect / verify / infer / bench
```

---

## Binary Format (.tinyinfer)

```
Header:
  magic        4 bytes   "TINY"
  version      uint32    1
  num_tensors  uint32    N

Tensor × N:
  name_len     uint32
  name         utf-8 bytes
  ndim         uint32
  shape        ndim × uint32
  data         numel × float32  (row-major)
```

Weights are loaded as read-only NumPy views via `np.frombuffer` — zero-copy, no allocation.

---

## Benchmark

| Model          | Load time | Inference (P50) | Throughput  |
|----------------|-----------|-----------------|-------------|
| PyTorch (.pth) | ~1.8 s    | ~3.2 ms         | ~310 /s     |
| TinyInfer      | ~12 ms    | ~180 µs         | ~5,500 /s   |

_(Measured on a single CPU core, no GPU)_

---

## Supported Ops

| Op               | Notes                              |
|------------------|------------------------------------|
| `Linear`         | `W @ x + b`                        |
| `LayerNorm`      | Matches PyTorch default (eps=1e-5) |
| `ReLU`           | In-place safe                      |
| `Dueling Agg`    | `Q = V + A - mean(A)`              |

---

## Tests

```bash
python -m pytest tests/ -v
# 26 passed
```

---

## Related

- **CyberShield v7** — the IPS this engine runs: `github.com/mtk339900/CyberShield`
- **PacketForge** — raw packet crafter/sniffer in C: `github.com/mtk339900/packforge`

---

## License

MIT
