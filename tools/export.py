#!/usr/bin/env python3
"""
tools/export.py
────────────────
Convert a CyberShield PyTorch .pth checkpoint into a
self-contained .tinyinfer binary (no PyTorch needed at inference).

Usage:
    python tools/export.py \
        --input  cyber_shield_v7_tested.pth \
        --output cyber_shield_v7.tinyinfer

Binary layout
─────────────
  Header:
    magic       4 bytes  "TINY"
    version     uint32   1
    num_tensors uint32   N

  For each tensor:
    name_len    uint32
    name        utf-8 bytes
    ndim        uint32
    shape       ndim × uint32
    data        numel × float32  (row-major)
"""

import argparse
import os
import struct
import sys
from pathlib import Path


def load_state_dict(path: str) -> dict:
    try:
        import torch
    except ImportError:
        print("[!] PyTorch required for export. Install with: pip install torch")
        sys.exit(1)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in ckpt:
                return {k: v.float().numpy() for k, v in ckpt[key].items()}
        # dict might directly be the state dict
        first = next(iter(ckpt.values()))
        if hasattr(first, "numpy"):
            return {k: v.float().numpy() for k, v in ckpt.items()}

    raise ValueError(f"Cannot find state_dict in checkpoint. Keys: {list(ckpt.keys())}")


def write_tinyinfer(state: dict, out_path: str) -> None:
    import numpy as np

    tensors = list(state.items())
    n       = len(tensors)

    with open(out_path, "wb") as f:
        # Header
        f.write(b"TINY")
        f.write(struct.pack("<I", 1))   # version
        f.write(struct.pack("<I", n))   # num_tensors

        total_params = 0
        for name, arr in tensors:
            arr = np.asarray(arr, dtype=np.float32)
            name_bytes = name.encode("utf-8")

            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", arr.ndim))
            for dim in arr.shape:
                f.write(struct.pack("<I", int(dim)))
            f.write(arr.flatten().tobytes())

            total_params += arr.size
            print(f"  {name:<60} {str(arr.shape)}")

    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"\n  ✓  {n} tensors | {total_params:,} parameters | {size_mb:.1f} MB")
    print(f"  ✓  Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Export CyberShield .pth → .tinyinfer (no PyTorch at runtime)")
    ap.add_argument("--input",  required=True, help="Input .pth checkpoint")
    ap.add_argument("--output", required=True, help="Output .tinyinfer file")
    args = ap.parse_args()

    if not Path(args.input).exists():
        print(f"[!] File not found: {args.input}")
        sys.exit(1)

    print(f"[*] Loading {args.input} ...")
    state = load_state_dict(args.input)
    print(f"[*] Exporting {len(state)} tensors ...\n")
    write_tinyinfer(state, args.output)


if __name__ == "__main__":
    main()
