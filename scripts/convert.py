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
import glob
import shutil
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from mlx_lm.utils import (
    create_model_card,
    load,
    quantize_model,
    save_config,
    save_model,
)


def convert(hf_path: str, mlx_path: Path, q_bits: int, q_group_size: int) -> None:
    """Re-implements mlx_lm.convert.convert()'s quantize+save steps, but avoids
    its save() -> hf_repo_to_path() call, which requires the ENTIRE source repo
    snapshot to already be cached locally (local_files_only=True) even though it
    only ever reads *.py and generation_config.json from it. Mistral-family repos
    ship a large redundant `consolidated.safetensors` file (a duplicate of the
    sharded weights in a different layout) that isn't part of mlx_lm's normal
    fetch, so that strict completeness check fails there and would otherwise
    force downloading several extra GB just to satisfy a check whose result is
    never used."""
    print("[INFO] Loading")
    model, tokenizer, config = load(hf_path, return_config=True, lazy=True)

    print("[INFO] Quantizing")
    model, config = quantize_model(model, config, q_group_size, q_bits, mode="affine")

    save_model(mlx_path, model, donate_model=True)
    save_config(config, config_path=mlx_path / "config.json")
    tokenizer.save_pretrained(mlx_path)

    # Only fetch the two patterns save() actually uses, instead of requiring
    # the full snapshot (which would pull consolidated.safetensors unnecessarily).
    local_src = Path(snapshot_download(hf_path, allow_patterns=["*.py", "generation_config.json"]))
    for pattern in ["*.py", "generation_config.json"]:
        for file in glob.glob(str(local_src / pattern)):
            shutil.copy(file, mlx_path)

    create_model_card(mlx_path, hf_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="Source model repo, e.g. mistralai/Ministral-3-3B-Base-2512")
    parser.add_argument("--upload-to", help="Destination repo, e.g. mlx-community/Ministral-3-3B-Base-2512-4bit. Omit to convert only.")
    parser.add_argument("--q-bits", type=int, default=4, help="Quantization bits (default: 4)")
    parser.add_argument("--q-group-size", type=int, default=64, help="Quantization group size (default: 64)")
    parser.add_argument("--mlx-path", help="Local output directory. Defaults to a temp directory.")
    parser.add_argument("--dry-run", action="store_true", help="Convert only, skip upload even if --upload-to is set")
    args = parser.parse_args()

    # mlx_lm's save_model() creates mlx_path itself; hand it a path that does
    # not exist yet (only ensure the parent does).
    if args.mlx_path:
        out_dir = Path(args.mlx_path)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="mlx-conv-"))
        out_dir.rmdir()

    print(f"[convert] {args.hf_path} -> {out_dir} (q_bits={args.q_bits}, group_size={args.q_group_size})")
    start = time.monotonic()
    convert(args.hf_path, out_dir, args.q_bits, args.q_group_size)
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
