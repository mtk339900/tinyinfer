#!/usr/bin/env python3
"""
export_tinyinfer.py
───────────────────
Converts a CyberShield .pth checkpoint into a self-contained
.tinyinfer binary that the C engine can load with zero Python.

Binary layout
─────────────
  [Header]
    magic      : 4 bytes  "TINY"
    version    : uint32   1
    num_tensors: uint32   N

  [Tensor × N]
    name_len   : uint32
    name       : name_len bytes (UTF-8, no null)
    ndim       : uint32
    shape      : ndim × uint32
    data       : prod(shape) × float32 (row-major)

Usage:
    python3 export_tinyinfer.py \
        --input  cyber_shield_v7_tested.pth \
        --output cyber_shield_v7.tinyinfer
"""

import argparse, struct, sys, os
import torch


def load_state(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in ckpt:
                return ckpt[key]
        # maybe the dict IS the state dict
        first_val = next(iter(ckpt.values()))
        if isinstance(first_val, torch.Tensor):
            return ckpt
    raise ValueError(f"Cannot find state_dict in checkpoint: {list(ckpt.keys())}")


def write_tinyinfer(state: dict, out_path: str) -> None:
    tensors = [(k, v.float().numpy()) for k, v in state.items()]
    n = len(tensors)

    with open(out_path, "wb") as f:
        # Header
        f.write(b"TINY")                          # magic
        f.write(struct.pack("<I", 1))              # version
        f.write(struct.pack("<I", n))              # num_tensors

        total_params = 0
        for name, arr in tensors:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", arr.ndim))
            for dim in arr.shape:
                f.write(struct.pack("<I", dim))
            data = arr.flatten().tobytes()
            f.write(data)
            total_params += arr.size
            print(f"  [{name:<60}] {tuple(arr.shape)}")

        size_mb = os.path.getsize(out_path) / (1024 ** 2)
        print(f"\n  ✓  {n} tensors  |  {total_params:,} parameters  |  {size_mb:.1f} MB")
        print(f"  ✓  Written → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Export PyTorch .pth → .tinyinfer")
    ap.add_argument("--input",  required=True, help="Input .pth file")
    ap.add_argument("--output", required=True, help="Output .tinyinfer file")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading {args.input} ...")
    state = load_state(args.input)
    print(f"[*] Exporting {len(state)} tensors ...\n")
    write_tinyinfer(state, args.output)


if __name__ == "__main__":
    main()
