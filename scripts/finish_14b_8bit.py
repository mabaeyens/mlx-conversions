#!/usr/bin/env python3
"""One-off resume: the 14B-8bit conversion completed locally but its upload
was interrupted mid-transfer (background task killed), leaving an empty
shell repo on HF (.gitattributes only, 0 commits). Local files are intact
(scripts/convert_all_precisions.py already ran the convert step), so this
just finishes upload -> verify -> card -> cleanup without reconverting."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import write_model_cards
from verify import functional_check, structural_check

SOURCE = "mistralai/Ministral-3-14B-Base-2512"
REPO = "mlx-community/Ministral-3-14B-Base-2512-8bit"
MLX_PATH = Path(__file__).parent.parent / "output_precisions" / "Ministral-3-14B-Base-2512-8bit"


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def compute_bpw(size_gb: float) -> float:
    """The mlx_vlm-printed bpw line was lost (stdout block-buffering ate it
    when the process was killed mid-run) -- recompute it the correct way:
    total compressed bits / total LOGICAL param count from the source's
    (unquantized) safetensors index. Do not use the quantized model's own
    packed tensor shapes for this -- that undercounts (see the 24.79-vs-7.537
    bug in convert_all_precisions.py's history)."""
    import requests

    resp = requests.get(f"https://huggingface.co/{SOURCE}/raw/main/model.safetensors.index.json")
    resp.raise_for_status()
    weight_map = resp.json()["weight_map"]
    total_params = 0
    seen_shards = {}
    for fname in set(weight_map.values()):
        r = requests.get(f"https://huggingface.co/{SOURCE}/resolve/main/{fname}", headers={"Range": "bytes=0-4000000"})
        header_len = int.from_bytes(r.content[:8], "little")
        header = r.content[8:8 + header_len]
        import json as _json
        seen_shards[fname] = _json.loads(header)
    for tensor_name, fname in weight_map.items():
        meta = seen_shards[fname].get(tensor_name)
        if meta and "shape" in meta:
            n = 1
            for d in meta["shape"]:
                n *= d
            total_params += n
    total_bits = size_gb * 1e9 * 8
    return total_bits / total_params


def main() -> None:
    size_gb = dir_size_gb(MLX_PATH)
    print(f"[resume] local copy present: {size_gb:.2f}GB")
    bpw = compute_bpw(size_gb)
    bpw_str = f"{bpw:.3f}"
    print(f"[resume] recomputed bits/weight: {bpw_str}")

    print(f"[upload] {MLX_PATH} -> {REPO}")
    from mlx_vlm.utils import upload_to_hub
    upload_to_hub(str(MLX_PATH), REPO)
    print(f"[upload] done: https://huggingface.co/{REPO}")

    print("[verify] structural check")
    struct_ok = structural_check(SOURCE, REPO)
    print("[verify] functional check")
    func_ok = functional_check(REPO, is_vlm=True)
    verified = struct_ok and func_ok
    print(f"[verify] {'PASS' if verified else 'FAIL'}: {REPO}")

    print("[card] writing")
    write_model_cards.render_and_upload(
        REPO, SOURCE, 8, 64, f"{size_gb:.2f}GB", bpw_str, verified,
    )

    if verified:
        print(f"[cleanup] removing local copy {MLX_PATH}")
        shutil.rmtree(MLX_PATH, ignore_errors=True)
    else:
        print(f"[cleanup] SKIPPED (verification failed) -- kept at {MLX_PATH} for debugging")

    print(f"\n{'DONE' if verified else 'DONE WITH FAILED VERIFICATION'}: https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
