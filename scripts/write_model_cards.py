#!/usr/bin/env python3
"""Apply the mlx-model-card skill's template to a published mlx-community
conversion -- a real description, quantization/provenance notes, an honest
verification-status note (style modeled on mlx-community/Inkling-mlx-4bit),
and a family table, replacing the bare auto-generated card (title +
pip-install snippet only).

Usage (single model):
    uv run python scripts/write_model_cards.py --repo mlx-community/Ministral-3-3B-Base-2512-4bit \\
        --source mistralai/Ministral-3-3B-Base-2512 --bits 4 --group-size 64 \\
        --output-size 2.80GB --avg-bpw 5.756 --verified

Or with no args, regenerates cards for all currently-known Ministral-3 Base
variants (see MODELS below) -- edit that dict as new variants are published.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

FAMILY_ROWS = []
for size in ["3B", "8B", "14B"]:
    FAMILY_ROWS.append((f"Ministral 3 {size} Base 2512", "Base pre-trained",
                         f"mlx-community/Ministral-3-{size}-Base-2512-4bit"))
    FAMILY_ROWS.append((f"Ministral 3 {size} Instruct 2512", "Instruct post-trained",
                         f"mlx-community/Ministral-3-{size}-Instruct-2512-4bit"))
    FAMILY_ROWS.append((f"Ministral 3 {size} Reasoning 2512", "Reasoning capable",
                         f"mlx-community/Ministral-3-{size}-Reasoning-2512-4bit"))


def family_table(highlight_repo: str) -> str:
    header = "| Model | Type | mlx-community (4-bit) |\n|---|---|---|\n"
    rows = []
    for name, kind, mlx_repo in FAMILY_ROWS:
        bold = highlight_repo == mlx_repo
        name_cell = f"**{name}**" if bold else name
        kind_cell = f"**{kind}**" if bold else kind
        rows.append(f"| {name_cell} | {kind_cell} | [{mlx_repo}](https://huggingface.co/{mlx_repo}) |")
    return header + "\n".join(rows)


# Known published variants, for the no-args bulk-regenerate path. Add a new
# entry here whenever a new precision variant is published.
MODELS = {
    "mlx-community/Ministral-3-3B-Base-2512-4bit": {
        "source": "mistralai/Ministral-3-3B-Base-2512", "bits": 4, "group_size": 64,
        "size_desc": "the smallest model in the Ministral 3 family",
        "output_size": "2.80GB", "avg_bpw": "5.756", "verified": True,
    },
    "mlx-community/Ministral-3-8B-Base-2512-4bit": {
        "source": "mistralai/Ministral-3-8B-Base-2512", "bits": 4, "group_size": 64,
        "size_desc": "the mid-size model in the Ministral 3 family",
        "output_size": "6.44GB", "avg_bpw": "5.745", "verified": True,
    },
    "mlx-community/Ministral-3-14B-Base-2512-4bit": {
        "source": "mistralai/Ministral-3-14B-Base-2512", "bits": 4, "group_size": 64,
        "size_desc": "the largest model in the Ministral 3 family",
        "output_size": "9.47GB", "avg_bpw": "5.416", "verified": True,
    },
}

FRONT_MATTER_TEMPLATE = """---
library_name: mlx
language:
- en
- fr
- es
- de
- it
- pt
- nl
- zh
- ja
- ko
- ar
license: apache-2.0
inference: false
extra_gated_description: If you want to learn more about how we process your personal
  data, please read our <a href="https://mistral.ai/terms/">Privacy Policy</a>.
tags:
- mistral-common
- mlx
- ministral
- ministral-3
- vision-language
- multimodal
- quantized
- edge
- {bit_tag}
- base-model
pipeline_tag: image-text-to-text
base_model: {source}
---"""

VERIFIED_NOTE = """> **Community note.** Structural check confirms the vision tower and
> multimodal projector were carried over intact (not dropped, which is a real
> failure mode for text-only conversion tools on vision-language models).
> Functional check confirms both text-only and image+text generation produce
> coherent output. Converted and verified by a single maintainer running
> local MLX tooling -- not independently reviewed by anyone else; please open
> a discussion if you hit anything unexpected."""

UNVERIFIED_NOTE = """> **Community note.** Structural check confirms the vision tower and
> multimodal projector were carried over intact (not dropped, which is a real
> failure mode for text-only conversion tools on vision-language models).
> **Functional check (actually running generation) has not been completed
> yet** -- weights are structurally complete but end-to-end generation
> hasn't been confirmed. Please open a discussion with results if you try it
> before this note is updated."""

BODY_TEMPLATE = """
# {repo}

{size_desc_cap}, **{name}** is a vision-language model: a text backbone paired
with a vision encoder, supporting image understanding alongside text. This is
the **base pre-trained** checkpoint — not instruction- or chat-tuned. For
chat/instruction-following use cases, use the
[Instruct variant](https://huggingface.co/mlx-community/{instruct_repo_name})
instead; this base checkpoint is intended for custom post-training/fine-tuning.

{verification_note}

This is an MLX conversion of [`{source}`](https://huggingface.co/{source}),
converted with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Refer to the
[original model card](https://huggingface.co/{source}) for the full
description, capabilities, and license terms.

## Heads up

- **Base model, not instruct-tuned** — expect raw completion behavior, not
  chat-following. Don't expect it to follow instructions well.
- **Vision retained at full precision** — only the language backbone is
  quantized; the vision tower and multimodal projector are untouched bf16,
  per mlx-vlm's standard policy of not quantizing multimodal modules.
- **Output size on disk: {output_size}**

## Provenance

- Source: [`{source}`](https://huggingface.co/{source}) (BF16)
- Language model layers: **{bits}-bit** affine quantization, group_size={group_size}
- Vision tower + multimodal projector: kept at full precision (not quantized)
- Blended average: **{avg_bpw} bits per weight** across all parameters

## Ministral 3 family

{family_table}

## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model {repo} --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
```

For text-only prompts, omit `--image`.

## License

Apache 2.0 — see the [original model card](https://huggingface.co/{source}) for
the full license text and any usage terms.
"""


def bit_tag(bits: int | None) -> str:
    return "bf16" if bits is None else f"{bits}-bit"


def render_and_upload(repo: str, source: str, bits: int | None, group_size: int | None,
                       output_size: str, avg_bpw: str, verified: bool) -> None:
    api = HfApi()
    size = source.split("-")[2]  # e.g. "3B"
    name = f"Ministral 3 {size} Base 2512"
    instruct_repo_name = repo.replace("Base", "Instruct").split("/")[-1]

    front_matter = FRONT_MATTER_TEMPLATE.format(source=source, bit_tag=bit_tag(bits))
    body = BODY_TEMPLATE.format(
        repo=repo,
        size_desc_cap="This is",
        name=name,
        source=source,
        instruct_repo_name=instruct_repo_name,
        bits=bits if bits is not None else "16 (bf16, unquantized)",
        group_size=group_size if group_size is not None else "n/a",
        avg_bpw=avg_bpw,
        output_size=output_size,
        family_table=family_table(repo),
        verification_note=VERIFIED_NOTE if verified else UNVERIFIED_NOTE,
    )
    content = front_matter + "\n" + body

    local_path = Path(f"/tmp/{repo.split('/')[-1]}-README.md")
    local_path.write_text(content)

    print(f"[card] uploading README.md -> {repo}")
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo="README.md", repo_id=repo)
    print(f"[card] done: https://huggingface.co/{repo}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo")
    parser.add_argument("--source")
    parser.add_argument("--bits", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--output-size")
    parser.add_argument("--avg-bpw")
    parser.add_argument("--verified", action="store_true")
    args = parser.parse_args()

    if args.repo:
        render_and_upload(args.repo, args.source, args.bits, args.group_size,
                           args.output_size, args.avg_bpw, args.verified)
        return

    for repo, info in MODELS.items():
        render_and_upload(repo, info["source"], info["bits"], info["group_size"],
                           info["output_size"], info["avg_bpw"], info["verified"])


if __name__ == "__main__":
    main()
