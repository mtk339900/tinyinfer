#!/usr/bin/env python3
"""
cli.py — TinyInfer command-line interface

Commands:
  inspect  <model.tinyinfer>                    list all tensors
  verify   <model.tinyinfer>                    run built-in test vectors
  infer    <model.tinyinfer> f0 f1 ... f25      single inference
  bench    <model.tinyinfer> [--n N]            latency benchmark
"""

import sys
import time
import argparse
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from tinyinfer import CyberShieldModel
from tinyinfer.loader import load, info


# ── WAF action names ──────────────────────────────────────────
WAF_ACTIONS = ["ALLOW", "BLOCK", "SCRUB", "RATE_LIMIT", "CAPTCHA"]

# ── Built-in test vectors (feature_vector, expect_blocked) ────
#    Same vectors validated against PyTorch reference
TEST_VECTORS = [
    ("Legit search query",
     [0.10, 0.02, 0.95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02],
     False),
    ("SQLi UNION SELECT",
     [0.20, 0.15, 0.70, 0.80, 0.60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.50, 0, 0.40, 0],
     True),
    ("XSS <script> tag",
     [0.10, 0.20, 0.65, 0, 0, 0.80, 0.40, 0.30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.60, 0, 0.30, 0],
     True),
    ("SSRF AWS metadata",
     [0.15, 0.10, 0.75, 0, 0, 0, 0, 0, 0, 0, 0.90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.50, 0, 0.20, 0],
     True),
    ("CMDi reverse shell",
     [0.25, 0.30, 0.55, 0, 0, 0, 0, 0, 0, 0.95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.70, 0, 0.40, 0],
     True),
    ("Path traversal",
     [0.12, 0.18, 0.72, 0, 0, 0, 0, 0, 0.85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.45, 0, 0.25, 0],
     True),
]


# ── inspect ───────────────────────────────────────────────────

def cmd_inspect(args):
    weights = load(args.model)
    info(weights)


# ── verify ────────────────────────────────────────────────────

def cmd_verify(args):
    model  = CyberShieldModel(args.model)
    passed = 0

    print(f"\n  TinyInfer — Sanity Verification  ({args.model})")
    print("  " + "─" * 70)
    print(f"  {'Test':<30} {'Expected':<10} {'Got':<12} {'Conf':>7}  {'μs':>7}  {'✓/✗'}")
    print("  " + "─" * 70)

    for name, feats, expect_block in TEST_VECTORS:
        x  = np.array(feats, dtype=np.float32)
        t0 = time.perf_counter()
        r  = model.waf_forward(x)
        us = (time.perf_counter() - t0) * 1e6

        is_block = r.action_id != 0
        ok       = (is_block == expect_block)
        passed  += ok

        exp_str = "BLOCK" if expect_block else "ALLOW"
        mark    = "✅" if ok else "❌"
        print(f"  {name:<30} {exp_str:<10} {r.action_name:<12} {r.confidence:>7.1f}  {us:>7.1f}  {mark}")

    print("  " + "─" * 70)
    total = len(TEST_VECTORS)
    print(f"\n  Result: {passed}/{total} passed", end="")
    print("  🎉" if passed == total else "  ⚠️  check failures above")
    print()
    return 0 if passed == total else 1


# ── infer ─────────────────────────────────────────────────────

def cmd_infer(args):
    if len(args.features) != 26:
        print(f"[!] Need 26 feature values, got {len(args.features)}")
        return 1

    features = np.array(args.features, dtype=np.float32)
    model    = CyberShieldModel(args.model)

    t0 = time.perf_counter()
    r  = model.waf_forward(features)
    us = (time.perf_counter() - t0) * 1e6

    print(f"\n  Action     : {r.action_name}  (id={r.action_id})")
    print(f"  Confidence : {r.confidence:.2f}")
    print(f"  Latency    : {us:.1f} µs")
    print(f"  Q-values   :")
    for i, (name, q) in enumerate(zip(WAF_ACTIONS, r.q_values)):
        marker = " ◀" if i == r.action_id else ""
        print(f"    [{i}] {name:<12}  {q:+.4f}{marker}")
    print()
    return 0


# ── bench ─────────────────────────────────────────────────────

def cmd_bench(args):
    model = CyberShieldModel(args.model)
    n     = args.n
    # random feature vector
    rng   = np.random.default_rng(42)
    x     = rng.random(26).astype(np.float32)

    print(f"\n  Benchmarking {n:,} inferences ...")

    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.waf_forward(x)
        latencies.append((time.perf_counter() - t0) * 1e6)

    latencies.sort()
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]
    avg = sum(latencies) / n

    print(f"  Avg  : {avg:.1f} µs")
    print(f"  P50  : {p50:.1f} µs")
    print(f"  P95  : {p95:.1f} µs")
    print(f"  P99  : {p99:.1f} µs")
    print(f"  Max  : {latencies[-1]:.1f} µs")
    print(f"  Throughput: {1e6/avg:,.0f} inferences/sec\n")
    return 0


# ── main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        prog="tinyinfer",
        description="TinyInfer — NumPy-only CyberShield inference engine")
    ap.add_argument("model", help=".tinyinfer model file")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inspect", help="List all tensors in the model file")

    sub.add_parser("verify",  help="Run built-in sanity test vectors")

    p_infer = sub.add_parser("infer", help="Run one inference")
    p_infer.add_argument("features", nargs=26, type=float,
                         metavar="F", help="26 WAF feature values")

    p_bench = sub.add_parser("bench", help="Latency benchmark")
    p_bench.add_argument("--n", type=int, default=10_000,
                         help="Number of iterations (default: 10000)")

    args = ap.parse_args()

    dispatch = {
        "inspect": cmd_inspect,
        "verify":  cmd_verify,
        "infer":   cmd_infer,
        "bench":   cmd_bench,
    }
    sys.exit(dispatch[args.cmd](args) or 0)


if __name__ == "__main__":
    main()
