#!/usr/bin/env python3
"""Retry: 14B-bf16's convert+upload+structural-check already succeeded; only
the functional check (generation) failed with a Metal OOM (model needs
~26.6GB, close to this 32GB machine's ~25.6GB default GPU memory ceiling).
Retrying in isolation with a clean machine to see if it was transient
memory pressure or a hard limit."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import write_model_cards
from verify import functional_check, structural_check

SOURCE = "mistralai/Ministral-3-14B-Base-2512"
REPO = "mlx-community/Ministral-3-14B-Base-2512-bf16"
MLX_PATH = Path(__file__).parent.parent / "output_precisions" / "Ministral-3-14B-Base-2512-bf16"


def main() -> None:
    print("[verify] structural check (already passed, re-confirming)")
    struct_ok = structural_check(SOURCE, REPO)
    print("[verify] functional check")
    func_ok = functional_check(REPO, is_vlm=True)
    verified = struct_ok and func_ok
    print(f"[verify] {'PASS' if verified else 'FAIL'}: {REPO}")

    size_gb = sum(f.stat().st_size for f in MLX_PATH.rglob("*") if f.is_file()) / 1e9 if MLX_PATH.exists() else 27.92

    print("[card] writing")
    write_model_cards.render_and_upload(
        REPO, SOURCE, None, None, f"{size_gb:.2f}GB", "16 (bf16, unquantized)", verified,
    )

    if verified and MLX_PATH.exists():
        print(f"[cleanup] removing local copy {MLX_PATH}")
        shutil.rmtree(MLX_PATH, ignore_errors=True)
    else:
        print(f"[cleanup] SKIPPED -- kept at {MLX_PATH} for debugging" if MLX_PATH.exists() else "[cleanup] no local copy found")

    print(f"\n{'DONE' if verified else 'DONE WITH FAILED VERIFICATION'}: https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
