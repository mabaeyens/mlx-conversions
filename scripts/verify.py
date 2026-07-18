#!/usr/bin/env python3
"""Verify a published MLX conversion before trusting it / deleting the local copy.

Two checks, per specs/model-verification.md:

1. Structural: compare tensor-prefix sets between the source repo and the
   converted output. Catches silently-dropped components (e.g. mlx_lm
   dropping a vision tower with no error) that a generation test alone would
   NOT catch -- the language backbone can work fine while an entire modality
   is missing.
2. Functional: actually load the published repo and run generation. For
   vision-language models, this includes a real image, not just text --
   presence of vision_tower weights doesn't prove the vision pathway works.

Usage:
    uv run python scripts/verify.py --repo mlx-community/Ministral-3-3B-Base-2512-4bit
    uv run python scripts/verify.py --repo mlx-community/Ministral-3-3B-Base-2512-4bit --source mistralai/Ministral-3-3B-Base-2512
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import requests


def _weight_prefixes(repo: str) -> set[str]:
    """Top-level module prefixes for a repo's weights, from its safetensors
    index (sharded) or by listing single-file safetensors tensor names."""
    index_url = f"https://huggingface.co/{repo}/raw/main/model.safetensors.index.json"
    resp = requests.get(index_url)
    if resp.status_code == 200:
        weight_map = resp.json()["weight_map"]
        return {k.split(".")[0] for k in weight_map}

    # Single-shard models have no index file -- inspect the safetensors header directly.
    file_url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
    resp = requests.get(file_url, headers={"Range": "bytes=0-100000"})
    resp.raise_for_status()
    header_len = int.from_bytes(resp.content[:8], "little")
    header = json.loads(resp.content[8:8 + header_len])
    return {k.split(".")[0] for k in header if k != "__metadata__"}


def structural_check(source_repo: str, target_repo: str) -> bool:
    print(f"[structural] {source_repo} vs {target_repo}")
    source_prefixes = _weight_prefixes(source_repo)
    target_prefixes = _weight_prefixes(target_repo)
    missing = source_prefixes - target_prefixes
    print(f"  source prefixes: {sorted(source_prefixes)}")
    print(f"  target prefixes: {sorted(target_prefixes)}")
    if missing:
        print(f"  FAIL: target is missing component(s) present in source: {sorted(missing)}")
        return False
    print("  OK: no components dropped")
    return True


def _make_test_image() -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 256), color=(70, 130, 180))
    draw = ImageDraw.Draw(img)
    draw.rectangle([64, 64, 192, 192], fill=(220, 20, 60))
    path = Path(tempfile.gettempdir()) / "mlx_verify_test_image.png"
    img.save(path)
    return str(path)


def functional_check(target_repo: str, is_vlm: bool) -> bool:
    print(f"[functional] loading {target_repo}")
    if is_vlm:
        from mlx_vlm import generate, load

        model, processor = load(target_repo)

        print("  generating (text-only)...")
        text_result = generate(model, processor, "Hello, how are you?", verbose=False)
        text_ok = bool(text_result.text and text_result.text.strip())
        print(f"    -> {text_result.text!r}")

        print("  generating (with test image)...")
        image_result = generate(model, processor, "Describe this image in one sentence.",
                                 image=_make_test_image(), verbose=False)
        image_ok = bool(image_result.text and image_result.text.strip())
        print(f"    -> {image_result.text!r}")

        if not (text_ok and image_ok):
            print("  FAIL: empty or missing generation output")
            return False
        print("  OK: both text-only and image+text generation produced output")
        return True
    else:
        from mlx_lm import generate, load

        model, tokenizer = load(target_repo)
        prompt = "Hello, how are you?"
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        result = generate(model, tokenizer, prompt=prompt, verbose=False)
        print(f"  -> {result!r}")
        if not (result and result.strip()):
            print("  FAIL: empty generation output")
            return False
        print("  OK: generation produced output")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Published mlx-community repo to verify")
    parser.add_argument("--source", help="Original source repo. Defaults to the target's base_model front-matter field.")
    parser.add_argument("--skip-structural", action="store_true")
    parser.add_argument("--skip-functional", action="store_true")
    args = parser.parse_args()

    config_resp = requests.get(f"https://huggingface.co/{args.repo}/raw/main/config.json")
    config_resp.raise_for_status()
    config = config_resp.json()
    is_vlm = "vision_config" in config

    source = args.source
    if source is None:
        card_resp = requests.get(f"https://huggingface.co/{args.repo}/raw/main/README.md")
        card_resp.raise_for_status()
        for line in card_resp.text.splitlines():
            if line.strip().startswith("base_model:"):
                source = line.split(":", 1)[1].strip()
                break
    if source is None:
        print("Could not determine source repo -- pass --source explicitly.")
        sys.exit(1)

    print(f"repo={args.repo} source={source} is_vlm={is_vlm}\n")

    ok = True
    if not args.skip_structural:
        ok = structural_check(source, args.repo) and ok
    if not args.skip_functional:
        ok = functional_check(args.repo, is_vlm) and ok

    print(f"\n{'PASS' if ok else 'FAIL'}: {args.repo}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
