#!/usr/bin/env python3
"""Publish 6-bit, 8-bit, and bf16 (unquantized) variants of the Ministral-3
Base family, in addition to the existing 4-bit ones. For each: convert,
upload, verify (structural + functional), write a real model card, then
delete the local copy per CLAUDE.md's policy (delete once uploaded+verified).

Skips any (size, precision) combo already on mlx-community. Runs sizes
sequentially, largest last, to respect the 32GB M5's memory budget -- no
two conversions run concurrently.
"""
from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent))
import convert_vlm  # noqa: E402
import write_model_cards  # noqa: E402
from verify import functional_check, structural_check  # noqa: E402

SIZES = ["3B", "8B", "14B"]
PRECISIONS = [
    {"suffix": "6bit", "bits": 6, "group_size": 64, "no_quantize": False},
    {"suffix": "8bit", "bits": 8, "group_size": 64, "no_quantize": False},
    {"suffix": "bf16", "bits": None, "group_size": None, "no_quantize": True},
]

OUTPUT_ROOT = Path(__file__).parent.parent / "output_precisions"


def repo_exists(repo: str) -> bool:
    try:
        HfApi().model_info(repo)
        return True
    except Exception:
        return False


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


class _Tee:
    """Writes to the real stdout AND captures everything, so we can both show
    live progress and parse mlx_vlm's own "Quantized model with X bits per
    weight" log line afterward -- computing that number ourselves from the
    safetensors header is wrong (packed/quantized tensors' raw shapes don't
    reflect logical parameter count without knowing per-tensor group_size)."""

    def __init__(self, real):
        self.real = real
        self.buffer = []

    def write(self, s):
        self.real.write(s)
        self.buffer.append(s)

    def flush(self):
        self.real.flush()

    def getvalue(self):
        return "".join(self.buffer)


def convert_capturing_bpw(source, mlx_path, bits, group_size, quantize):
    """Runs convert_vlm.convert(), returns the bits-per-weight mlx_vlm itself
    reported (None for --no-quantize / bf16 passthrough, where there's no
    quantization step to report one)."""
    import re

    tee = _Tee(sys.stdout)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        convert_vlm.convert(source, mlx_path, q_bits=bits or 4, q_group_size=group_size or 64, quantize=quantize)
    finally:
        sys.stdout = old_stdout
    m = re.search(r"Quantized model with ([\d.]+) bits per weight", tee.getvalue())
    return float(m.group(1)) if m else None


def process(size: str, precision: dict) -> None:
    source = f"mistralai/Ministral-3-{size}-Base-2512"
    repo = f"mlx-community/Ministral-3-{size}-Base-2512-{precision['suffix']}"

    if repo_exists(repo):
        print(f"[skip] {repo} already exists")
        return

    mlx_path = OUTPUT_ROOT / f"Ministral-3-{size}-Base-2512-{precision['suffix']}"
    print(f"\n=== {repo} ===")
    try:
        print(f"[convert] {source} -> {mlx_path} (quantize={not precision['no_quantize']}, bits={precision['bits']})")
        avg_bpw = convert_capturing_bpw(
            source, mlx_path,
            bits=precision["bits"], group_size=precision["group_size"],
            quantize=not precision["no_quantize"],
        )
        size_gb = dir_size_gb(mlx_path)
        bpw_str = f"{avg_bpw:.3f}" if avg_bpw is not None else "16 (bf16, unquantized)"
        print(f"[convert] done, output size {size_gb:.2f}GB, avg {bpw_str} bits/weight")

        print(f"[upload] {mlx_path} -> {repo}")
        from mlx_vlm.utils import upload_to_hub
        upload_to_hub(str(mlx_path), repo)
        print(f"[upload] done: https://huggingface.co/{repo}")

        print("[verify] structural check")
        struct_ok = structural_check(source, repo)
        print("[verify] functional check")
        func_ok = functional_check(repo, is_vlm=True)
        verified = struct_ok and func_ok
        print(f"[verify] {'PASS' if verified else 'FAIL'}: {repo}")

        print("[card] writing")
        write_model_cards.render_and_upload(
            repo, source, precision["bits"], precision["group_size"],
            f"{size_gb:.2f}GB", bpw_str, verified,
        )

        if verified:
            print(f"[cleanup] removing local copy {mlx_path}")
            shutil.rmtree(mlx_path, ignore_errors=True)
        else:
            print(f"[cleanup] SKIPPED (verification failed) -- local copy kept at {mlx_path} for debugging")

    except Exception:
        print(f"[ERROR] {repo} failed:")
        traceback.print_exc()
        print(f"[ERROR] local copy (if any) left at {mlx_path} for debugging")


def main() -> None:
    for size in SIZES:
        for precision in PRECISIONS:
            process(size, precision)
    print("\n=== all done ===")


if __name__ == "__main__":
    main()
