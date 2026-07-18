#!/usr/bin/env python3
"""Scan mistralai's HF catalog for models not yet converted to mlx-community.

Standalone version of the logic embedded in the weekly cloud routine
(specs/conversion-queue-automation.md) -- kept here so it can be run and
iterated on locally without needing a routine invocation.

Usage:
    uv run python scripts/scan_candidates.py
    uv run python scripts/scan_candidates.py --max-params 16
"""
from __future__ import annotations

import argparse
import re

import requests

EXCLUDE_PATTERN = re.compile(r"guard|moderation|embed|ocr|pixtral|gguf|fp8|onnx|voxtral|tts|realtime", re.IGNORECASE)

# HF's search is a literal substring match, not fuzzy -- "ministral" does not
# contain "mistral" as a substring, so a single query silently misses half the
# family. Search every plausible stem separately and merge results.
MLX_COMMUNITY_SEARCH_TERMS = ["mistral", "ministral", "mathstral", "mamba-codestral", "codestral", "magistral", "devstral"]


def params_billions(model_id: str) -> float | None:
    m = re.search(r"(\d+)x(\d+)B", model_id, re.IGNORECASE)
    if m:
        return float(m.group(1)) * float(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)B", model_id, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def base_name(model_id: str) -> str:
    name = model_id.split("/")[-1].lower()
    name = re.sub(r"-(gguf|awq|gptq|fp8|bf16|mlx.*|\d+bit.*|eagle|reasoning|dwq|dynamic)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[-_]v?\d+\.\d+$", "", name)
    return name


def fetch_models(author: str, search: str | None = None, limit: int = 200) -> list[dict]:
    params = {"author": author, "limit": limit}
    if search:
        params["search"] = search
    resp = requests.get("https://huggingface.co/api/models", params=params)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-params", type=float, default=16, help="Max parameter count in billions (default: 16)")
    args = parser.parse_args()

    mistralai_models = fetch_models("mistralai", limit=100)

    mlx_ids: set[str] = set()
    for term in MLX_COMMUNITY_SEARCH_TERMS:
        for m in fetch_models("mlx-community", search=term):
            mlx_ids.add(m["id"])
    mlx_base_names = {base_name(mid) for mid in mlx_ids}

    candidates = []
    for m in mistralai_models:
        model_id = m["id"]
        if EXCLUDE_PATTERN.search(model_id):
            continue
        p = params_billions(model_id)
        if p is None or p > args.max_params:
            continue
        if base_name(model_id) in mlx_base_names:
            continue
        candidates.append((model_id, p, m.get("downloads", 0)))

    if not candidates:
        print("No new candidates found.")
        return

    print(f"Found {len(candidates)} new candidate(s):")
    for model_id, p, downloads in sorted(candidates, key=lambda x: -x[2]):
        print(f"  {model_id} | ~{p}B | downloads={downloads}")


if __name__ == "__main__":
    main()
