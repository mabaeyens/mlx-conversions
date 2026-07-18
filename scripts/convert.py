#!/usr/bin/env python3
"""Quantize a HuggingFace model to MLX format and optionally upload it.

Usage:
    uv run python scripts/convert.py \\
        --hf-path mistralai/Ministral-3-3B-Base-2512 \\
        --upload-to mlx-community/Ministral-3-3B-Base-2512-4bit \\
        --q-bits 4 --q-group-size 64
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi
from mlx_lm.convert import convert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="Source model repo, e.g. mistralai/Ministral-3-3B-Base-2512")
    parser.add_argument("--upload-to", help="Destination repo, e.g. mlx-community/Ministral-3-3B-Base-2512-4bit. Omit to convert only.")
    parser.add_argument("--q-bits", type=int, default=4, help="Quantization bits (default: 4)")
    parser.add_argument("--q-group-size", type=int, default=64, help="Quantization group size (default: 64)")
    parser.add_argument("--mlx-path", help="Local output directory. Defaults to a temp directory.")
    parser.add_argument("--dry-run", action="store_true", help="Convert only, skip upload even if --upload-to is set")
    args = parser.parse_args()

    out_dir = Path(args.mlx_path) if args.mlx_path else Path(tempfile.mkdtemp(prefix="mlx-conv-"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[convert] {args.hf_path} -> {out_dir} (q_bits={args.q_bits}, group_size={args.q_group_size})")
    start = time.monotonic()
    convert(
        args.hf_path,
        mlx_path=str(out_dir),
        quantize=True,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
    )
    elapsed = time.monotonic() - start
    size_gb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e9
    print(f"[convert] done in {elapsed / 60:.1f} min, output size {size_gb:.2f} GB")

    if args.upload_to and not args.dry_run:
        print(f"[upload] {out_dir} -> {args.upload_to}")
        api = HfApi()
        api.create_repo(args.upload_to, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=args.upload_to)
        print(f"[upload] done: https://huggingface.co/{args.upload_to}")
    elif not args.mlx_path:
        print(f"[convert] no --upload-to given, output left at {out_dir} (not cleaned up)")

    if not args.mlx_path and args.upload_to and not args.dry_run:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
