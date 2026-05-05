#!/usr/bin/env python3
"""
tools/verify_vs_pytorch.py
───────────────────────────
Runs TinyInfer and PyTorch side-by-side on the full CyberShield
test suite and reports Q-value agreement (max diff, mean diff).

Usage:
    python tools/verify_vs_pytorch.py \
        --tinyinfer  cyber_shield_v7.tinyinfer \
        --pth        cyber_shield_v7_tested.pth

Requires: torch (only for this script)
"""

import sys, argparse, time
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))


# ── Load PyTorch model ────────────────────────────────────────

def load_pytorch(pth_path: str):
    import torch, torch.nn as nn

    class DuelingDQN(nn.Module):
        def __init__(self, state_dim, action_dim, dropout=0.3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(state_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(512, 256),       nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            )
            self.value     = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
            self.advantage = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))

        def forward(self, x):
            f = self.features(x)
            v = self.value(f)
            a = self.advantage(f)
            return v + a - a.mean(dim=-1, keepdim=True)

    class UnifiedCyberShield(nn.Module):
        def __init__(self):
            super().__init__()
            self.waf_dqn     = DuelingDQN(26, 5)
            self.network_dqn = DuelingDQN(22, 4)
            self.server_dqn  = DuelingDQN(20, 4)
            self.meta_controller = nn.Sequential(
                nn.Linear(793,512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512,256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256,128))
            self.action_head     = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,7))
            self.threat_head     = nn.Sequential(nn.Linear(128,32), nn.ReLU(), nn.Linear(32,5))
            self.confidence_head = nn.Sequential(nn.Linear(128,16), nn.ReLU(), nn.Linear(16,3))
        def forward(self, x): return x

    ckpt  = torch.load(pth_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model = UnifiedCyberShield()
    model.load_state_dict(state)
    model.eval()
    return model


# ── Test payloads → feature vectors (simplified) ──────────────

FEATURE_VECTORS = [
    # (label, feat_26)
    ("Legit: normal search",
     [0.10, 0.02, 0.95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02]),
    ("Legit: JSON API payload",
     [0.08, 0.05, 0.90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01]),
    ("SQLi: UNION SELECT",
     [0.20, 0.15, 0.70, 0.80, 0.60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.50, 0, 0.40, 0]),
    ("SQLi: error-based extractvalue",
     [0.18, 0.12, 0.72, 0.70, 0.45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0.45, 0, 0.35, 0]),
    ("XSS: <script> basic",
     [0.10, 0.20, 0.65, 0, 0, 0.80, 0.40, 0.30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.60, 0, 0.30, 0]),
    ("XSS: SVG onload",
     [0.08, 0.25, 0.62, 0, 0, 0.70, 0.90, 0.20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.65, 0, 0.28, 0]),
    ("XSS: sendBeacon steal",
     [0.12, 0.18, 0.68, 0, 0, 0.40, 0.20, 0.85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.55, 0, 0.25, 0]),
    ("CMDi: semicolon cat",
     [0.15, 0.22, 0.68, 0, 0, 0, 0, 0, 0, 0.80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.55, 0, 0.30, 0]),
    ("CMDi: reverse shell",
     [0.25, 0.30, 0.55, 0, 0, 0, 0, 0, 0, 0.95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.70, 0, 0.40, 0]),
    ("Traversal: ../etc/passwd",
     [0.12, 0.18, 0.72, 0, 0, 0, 0, 0, 0.85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.45, 0, 0.25, 0]),
    ("Traversal: double-encode",
     [0.14, 0.22, 0.69, 0, 0, 0, 0, 0, 0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.60, 0, 0, 0, 0.50, 0, 0.28, 0]),
    ("SSRF: localhost",
     [0.10, 0.08, 0.80, 0, 0, 0, 0, 0, 0, 0, 0.80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.40, 0, 0.20, 0]),
    ("SSRF: AWS metadata",
     [0.15, 0.10, 0.75, 0, 0, 0, 0, 0, 0, 0, 0.90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.50, 0, 0.20, 0]),
    ("SSTI: Jinja2 basic",
     [0.08, 0.30, 0.60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.50, 0, 0.30, 0]),
    ("SSTI: Jinja2 RCE MRO chain",
     [0.20, 0.28, 0.58, 0, 0, 0, 0, 0, 0, 0.50, 0, 0, 0.95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.75, 0, 0.45, 0]),
    ("NoSQL: $where JS exec",
     [0.10, 0.15, 0.72, 0, 0, 0, 0, 0, 0, 0, 0, 0.85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.45, 0, 0.22, 0]),
    ("LDAP: wildcard dump",
     [0.12, 0.25, 0.62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.80, 0, 0, 0, 0, 0, 0, 0, 0.40, 0, 0.22, 0]),
    ("Deser: Java magic bytes",
     [0.14, 0.10, 0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.90, 0, 0, 0, 0, 0, 0, 0.50, 0, 0.25, 0]),
    ("Deser: Python pickle RCE",
     [0.18, 0.12, 0.70, 0, 0, 0, 0, 0, 0, 0.40, 0, 0, 0, 0, 0, 0.85, 0, 0, 0, 0, 0, 0, 0.60, 0, 0.35, 0]),
    ("PromptInj: ignore instructions",
     [0.20, 0.05, 0.88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95, 0, 0, 0, 0, 0.50, 0, 0.20, 0]),
]


def main():
    ap = argparse.ArgumentParser(description="Verify TinyInfer matches PyTorch")
    ap.add_argument("--tinyinfer", required=True)
    ap.add_argument("--pth",       required=True)
    args = ap.parse_args()

    import torch
    from tinyinfer import CyberShieldModel

    WAF_ACTIONS = ["ALLOW", "BLOCK", "SCRUB", "RATE_LIMIT", "CAPTCHA"]

    print("\n[*] Loading TinyInfer model ...")
    ti_model = CyberShieldModel(args.tinyinfer)

    print("[*] Loading PyTorch model ...")
    pt_model = load_pytorch(args.pth)

    print(f"\n  {'Test':<38}  {'PT':^8}  {'TI':^8}  {'MaxΔQ':>8}  {'Match'}")
    print("  " + "─" * 75)

    max_diffs   = []
    action_matches = 0

    with torch.no_grad():
        for label, feats in FEATURE_VECTORS:
            x_np = np.array(feats, dtype=np.float32)
            x_pt = torch.FloatTensor(feats).unsqueeze(0)

            # PyTorch
            pt_q      = pt_model.waf_dqn(x_pt).numpy()[0]
            pt_action = int(pt_q.argmax())

            # TinyInfer
            ti_r      = ti_model.waf_forward(x_np)
            ti_q      = ti_r.q_values
            ti_action = ti_r.action_id

            max_diff = float(np.abs(pt_q - ti_q).max())
            max_diffs.append(max_diff)
            match = (pt_action == ti_action)
            action_matches += match

            mark = "✅" if match else "❌"
            print(f"  {label:<38}  {WAF_ACTIONS[pt_action]:^8}  "
                  f"{WAF_ACTIONS[ti_action]:^8}  {max_diff:>8.4f}  {mark}")

    n = len(FEATURE_VECTORS)
    print("  " + "─" * 75)
    print(f"\n  Action agreement : {action_matches}/{n}  ({action_matches/n:.1%})")
    print(f"  Max Q-value diff : {max(max_diffs):.6f}")
    print(f"  Mean Q-diff      : {sum(max_diffs)/len(max_diffs):.6f}")
    print(f"  Numerical match  : {'✅ PASS' if max(max_diffs) < 0.01 else '⚠️  check diffs'}\n")


if __name__ == "__main__":
    main()
