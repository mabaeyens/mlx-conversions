#!/usr/bin/env python3
"""Quantize a HuggingFace vision-language model to MLX format (via mlx-vlm).

Use this instead of scripts/convert.py whenever the source model is a
vision-language architecture (e.g. Mistral3ForConditionalGeneration / the
Ministral-3 family, which pairs a language model with a vision tower and
multimodal projector). scripts/convert.py uses mlx_lm, which is text-only:
it silently loads and quantizes only the language_model weights and drops
the vision_tower/multi_modal_projector entirely, with no error or warning.
Check a model's config.json for "vision_config" (or an architecture ending
in "ForConditionalGeneration") before picking which script to use.

Usage:
    uv run python scripts/convert_vlm.py \\
        --hf-path mistralai/Ministral-3-3B-Base-2512 \\
        --upload-to mlx-community/Ministral-3-3B-Base-2512-4bit \\
        --q-bits 4 --q-group-size 64
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_vlm.convert import convert as mlx_vlm_convert
from mlx_vlm.utils import create_model_card, upload_to_hub

# mlx_vlm's own get_model_path() fetches every "*.safetensors" file, which
# includes Mistral-family repos' redundant `consolidated.safetensors` (a
# duplicate of the sharded weights in a different layout -- not needed for
# loading, but ~as large as the real weights). Pre-fetching ourselves with
# this file excluded, then handing convert() the resolved local directory
# instead of the repo id, makes it skip its own (unrestricted) download.
_VLM_DEFAULT_ALLOW_PATTERNS = [
    "*.json", "*.safetensors", "*.py", "*.model", "*.tiktoken", "*.txt", "*.jinja",
]


def convert(hf_path: str, mlx_path: Path, q_bits: int, q_group_size: int) -> None:
    print("[INFO] Pre-fetching snapshot (excluding redundant consolidated.safetensors)")
    local_snapshot = snapshot_download(
        hf_path,
        allow_patterns=_VLM_DEFAULT_ALLOW_PATTERNS,
        ignore_patterns=["consolidated.safetensors"],
    )

    mlx_vlm_convert(
        local_snapshot,
        mlx_path=str(mlx_path),
        quantize=True,
        q_bits=q_bits,
        q_group_size=q_group_size,
        upload_repo=None,
    )

    # convert() saw a local path (not the hf_path string) for its `hf_path`
    # argument, so its own create_model_card() call treated this as a
    # from-scratch conversion with no known source and wrote a generic,
    # unattributed card. Overwrite it with one that correctly links back to
    # the real source repo.
    create_model_card(mlx_path, hf_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="Source model repo, e.g. mistralai/Ministral-3-3B-Base-2512")
    parser.add_argument("--upload-to", help="Destination repo, e.g. mlx-community/Ministral-3-3B-Base-2512-4bit. Omit to convert only.")
    parser.add_argument("--q-bits", type=int, default=4, help="Quantization bits for the language model (default: 4). Vision/projector modules are never quantized (mlx_vlm policy).")
    parser.add_argument("--q-group-size", type=int, default=64, help="Quantization group size (default: 64)")
    parser.add_argument("--mlx-path", help="Local output directory. Defaults to a temp directory.")
    parser.add_argument("--dry-run", action="store_true", help="Convert only, skip upload even if --upload-to is set")
    args = parser.parse_args()

    if args.mlx_path:
        out_dir = Path(args.mlx_path)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="mlx-vlm-conv-"))
        out_dir.rmdir()

    print(f"[convert] {args.hf_path} -> {out_dir} (q_bits={args.q_bits}, group_size={args.q_group_size})")
    start = time.monotonic()
    convert(args.hf_path, out_dir, args.q_bits, args.q_group_size)
    elapsed = time.monotonic() - start
    size_gb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e9
    print(f"[convert] done in {elapsed / 60:.1f} min, output size {size_gb:.2f} GB")

    if args.upload_to and not args.dry_run:
        print(f"[upload] {out_dir} -> {args.upload_to}")
        upload_to_hub(str(out_dir), args.upload_to)
        print(f"[upload] done: https://huggingface.co/{args.upload_to}")


if __name__ == "__main__":
    main()
