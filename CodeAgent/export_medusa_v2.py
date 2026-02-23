#!/usr/bin/env python3
# convert_medusa_to_mlx.py
# ============================================================
# Convert PyTorch Medusa head weights (.pth) to MLX-compatible .npz
#
# Features:
#  - CLI args for input/output paths
#  - Supports selecting either:
#       (a) a single .pth file
#       (b) "best checkpoint" auto-pick from a prefix (e.g., medusa_heads_kd_topk.best_prefix_acc_2.pth)
#  - Optional key remapping (for compatibility if module naming changes)
#  - Optional dtype control (float16/float32)
#  - Optional sanity check print / stats
# ============================================================

import argparse
import os
import re
from typing import Dict, Any, Optional

import numpy as np
import torch


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """
    Robustly load a PyTorch state_dict.
    Supports either:
      - direct state_dict saved by torch.save(model.state_dict(), ...)
      - full checkpoint with 'state_dict' field
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        # likely already a state_dict
        sd = obj
    else:
        raise ValueError(f"Unsupported checkpoint format at: {path}")
    # ensure tensors
    for k, v in sd.items():
        if not torch.is_tensor(v):
            raise ValueError(f"Non-tensor item in state_dict: {k} -> {type(v)}")
    return sd


def _maybe_strip_prefix(key: str, prefix: Optional[str]) -> str:
    if prefix and key.startswith(prefix):
        return key[len(prefix):]
    return key


def _apply_regex_rename(key: str, pattern: Optional[str], repl: Optional[str]) -> str:
    if pattern and repl is not None:
        return re.sub(pattern, repl, key)
    return key


def convert_medusa_to_mlx(
    pth_path: str,
    output_path: str,
    dtype: str = "float32",
    strip_prefix: Optional[str] = None,
    rename_pattern: Optional[str] = None,
    rename_repl: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Convert state_dict tensors to NumPy arrays and save as .npz for MLX.
    """
    if verbose:
        print(f"Loading PyTorch weights from: {pth_path}")

    state_dict = _load_state_dict(pth_path)

    if dtype not in ("float16", "float32"):
        raise ValueError("dtype must be 'float16' or 'float32'")

    np_dtype = np.float16 if dtype == "float16" else np.float32

    mlx_weights: Dict[str, Any] = {}
    exported = 0

    for key, tensor in state_dict.items():
        new_key = _maybe_strip_prefix(key, strip_prefix)
        new_key = _apply_regex_rename(new_key, rename_pattern, rename_repl)

        arr = tensor.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        mlx_weights[new_key] = arr
        exported += 1

        if verbose:
            print(f"Exported: {key} -> {new_key} | shape={tuple(arr.shape)} | dtype={arr.dtype}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, **mlx_weights)

    if verbose:
        print(f"✅ Saved {exported} tensors to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch Medusa weights (.pth) to MLX .npz format"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input .pth file (e.g., medusa_heads_kd_topk.best_prefix_acc_2.pth)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output .npz file (e.g., mlx_medusa_heads.npz)"
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32"],
        help="Output numpy dtype. float32 is safest; float16 is smaller/faster to load."
    )
    parser.add_argument(
        "--strip-prefix",
        default=None,
        help="Optionally strip a prefix from all keys (e.g., 'medusa.')"
    )
    parser.add_argument(
        "--rename-pattern",
        default=None,
        help="Optional regex pattern to rename keys (advanced)."
    )
    parser.add_argument(
        "--rename-repl",
        default=None,
        help="Replacement string for --rename-pattern."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less logging."
    )

    args = parser.parse_args()

    convert_medusa_to_mlx(
        pth_path=args.input,
        output_path=args.output,
        dtype=args.dtype,
        strip_prefix=args.strip_prefix,
        rename_pattern=args.rename_pattern,
        rename_repl=args.rename_repl,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

"""
python export_medusa_v2.py \
  -i medusa_heads_kd_topk.best_prefix_acc_2.pth \
  -o mlx_medusa_heads.npz \
  --dtype float32
  
#If your keys have an unwanted prefix (e.g., medusa.blocks.0...):
python export_medusa_v2.py \
  -i medusa_heads_kd_topk.best_prefix_acc_2.pth \
  -o mlx_medusa_heads.npz \
  --strip-prefix "medusa."

#Regex rename (advanced, only if you really need it):
python export_medusa_v2.py \
  -i medusa_heads_kd_topk.best_prefix_acc_2.pth \
  -o mlx_medusa_heads.npz \
  --rename-pattern "^blocks\." \
  --rename-repl "blocks."
"""