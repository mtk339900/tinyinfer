"""
tinyinfer/loader.py
────────────────────
Loads a .tinyinfer binary weight file into a plain dict:
  { "layer.weight": np.ndarray, ... }

Binary layout (same format written by tools/export.py):
  Header : magic(4) + version(uint32) + num_tensors(uint32)
  Tensor : name_len(uint32) + name(utf-8) + ndim(uint32)
           + shape(ndim×uint32) + data(numel×float32)
"""

import struct
import numpy as np
from pathlib import Path
from typing import Dict

MAGIC   = b"TINY"
VERSION = 1


def load(path: str | Path) -> Dict[str, np.ndarray]:
    """
    Load a .tinyinfer file.

    Returns
    -------
    weights : dict[str, np.ndarray]
        Mapping of tensor name → float32 NumPy array (read-only view).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    raw  = path.read_bytes()
    pos  = 0

    def read_u32():
        nonlocal pos
        v = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        return v

    # ── Header ───────────────────────────────────────────────
    magic = raw[pos:pos+4]
    pos  += 4
    if magic != MAGIC:
        raise ValueError(f"Not a .tinyinfer file (bad magic: {magic!r})")

    version = read_u32()
    if version != VERSION:
        raise ValueError(f"Unsupported .tinyinfer version: {version}")

    n = read_u32()

    # ── Tensors ───────────────────────────────────────────────
    weights: Dict[str, np.ndarray] = {}
    for _ in range(n):
        name_len = read_u32()
        name     = raw[pos:pos+name_len].decode("utf-8")
        pos     += name_len

        ndim  = read_u32()
        shape = tuple(read_u32() for _ in range(ndim))
        numel = int(np.prod(shape)) if shape else 1

        byte_size = numel * 4
        arr = np.frombuffer(raw, dtype=np.float32, count=numel, offset=pos).reshape(shape)
        pos += byte_size

        weights[name] = arr   # read-only view into the raw bytes

    return weights


def info(weights: Dict[str, np.ndarray]) -> None:
    """Print a summary of all loaded tensors."""
    total = 0
    print(f"{'Name':<60} {'Shape':<20} {'Params':>10}")
    print("─" * 94)
    for name, arr in weights.items():
        n = arr.size
        total += n
        print(f"{name:<60} {str(arr.shape):<20} {n:>10,}")
    print("─" * 94)
    print(f"{'TOTAL':<60} {'':20} {total:>10,}")
