#!/usr/bin/env python3
"""One-off: apply the mlx-model-card skill's template to the 3 published
Ministral-3 Base models, replacing their bare auto-generated cards (title +
pip-install snippet only) with a real description, quantization notes, and
family table pulled from the source model's own card."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

FAMILY_ROWS = []
for size in ["3B", "8B", "14B"]:
    FAMILY_ROWS.append((f"Ministral 3 {size} Base 2512", "Base pre-trained",
                         f"mistralai/Ministral-3-{size}-Base-2512",
                         f"mlx-community/Ministral-3-{size}-Base-2512-4bit"))
    FAMILY_ROWS.append((f"Ministral 3 {size} Instruct 2512", "Instruct post-trained",
                         f"mistralai/Ministral-3-{size}-Instruct-2512",
                         f"mlx-community/Ministral-3-{size}-Instruct-2512-4bit"))
    FAMILY_ROWS.append((f"Ministral 3 {size} Reasoning 2512", "Reasoning capable",
                         f"mistralai/Ministral-3-{size}-Reasoning-2512",
                         f"mlx-community/Ministral-3-{size}-Reasoning-2512-4bit"))


def family_table(highlight_repo: str) -> str:
    header = "| Model | Type | mlx-community (4-bit) |\n|---|---|---|\n"
    rows = []
    for name, kind, _src, mlx_repo in FAMILY_ROWS:
        bold = highlight_repo == mlx_repo
        name_cell = f"**{name}**" if bold else name
        kind_cell = f"**{kind}**" if bold else kind
        rows.append(f"| {name_cell} | {kind_cell} | [{mlx_repo}](https://huggingface.co/{mlx_repo}) |")
    return header + "\n".join(rows)


MODELS = {
    "mlx-community/Ministral-3-3B-Base-2512-4bit": {
        "source": "mistralai/Ministral-3-3B-Base-2512",
        "size_desc": "the smallest model in the Ministral 3 family",
        "output_size": "2.80GB",
        "avg_bpw": "5.756",
    },
    "mlx-community/Ministral-3-8B-Base-2512-4bit": {
        "source": "mistralai/Ministral-3-8B-Base-2512",
        "size_desc": "the mid-size model in the Ministral 3 family",
        "output_size": "6.44GB",
        "avg_bpw": "5.745",
    },
    "mlx-community/Ministral-3-14B-Base-2512-4bit": {
        "source": "mistralai/Ministral-3-14B-Base-2512",
        "size_desc": "the largest model in the Ministral 3 family",
        "output_size": "9.47GB",
        "avg_bpw": "5.416",
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
pipeline_tag: image-text-to-text
base_model: {source}
---"""

BODY_TEMPLATE = """
# {repo}

{size_desc_cap}, **{name}** is a vision-language model: a text backbone paired
with a vision encoder, supporting image understanding alongside text. This is
the **base pre-trained** checkpoint — not instruction- or chat-tuned. For
chat/instruction-following use cases, use the
[Instruct variant](https://huggingface.co/mlx-community/{instruct_repo_name})
instead; this base checkpoint is intended for custom post-training/fine-tuning.

This is an MLX conversion of [`{source}`](https://huggingface.co/{source}),
converted with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Refer to the
[original model card](https://huggingface.co/{source}) for the full
description, capabilities, and license terms.

## Quantization notes

- Language model layers: 4-bit, group_size=64, affine mode
- Vision tower and multimodal projector are kept at **full precision** — only
  the language backbone is quantized, per mlx-vlm's standard policy of not
  quantizing multimodal modules
- Blended average: **{avg_bpw} bits per weight** across all parameters
- Output size on disk: **{output_size}**

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


def main() -> None:
    api = HfApi()
    for repo, info in MODELS.items():
        source = info["source"]
        size = source.split("-")[2]  # e.g. "3B"
        name = f"Ministral 3 {size} Base 2512"
        instruct_repo_name = repo.replace("Base", "Instruct").split("/")[-1]

        front_matter = FRONT_MATTER_TEMPLATE.format(source=source)
        body = BODY_TEMPLATE.format(
            repo=repo,
            size_desc_cap=info["size_desc"][0].upper() + info["size_desc"][1:],
            name=name,
            source=source,
            instruct_repo_name=instruct_repo_name,
            avg_bpw=info["avg_bpw"],
            output_size=info["output_size"],
            family_table=family_table(repo),
        )
        content = front_matter + "\n" + body

        local_path = Path(f"/tmp/{repo.split('/')[-1]}-README.md")
        local_path.write_text(content)

        print(f"[card] uploading README.md -> {repo}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo="README.md",
            repo_id=repo,
        )
        print(f"[card] done: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
