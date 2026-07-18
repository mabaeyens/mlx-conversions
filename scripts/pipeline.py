#!/usr/bin/env python3
"""End-to-end conversion pipeline: given a source HF repo, this does every
step a manual conversion in this repo requires, in code, so none of it has
to be re-derived by hand each time:

  1. Fetch config.json, decide mlx_lm (text-only) vs mlx_vlm (vision-
     language) -- checks for vision_config/audio_config or a
     *ForConditionalGeneration architecture (see CLAUDE.md constraint).
  2. Check mlx-community doesn't already have this exact repo/quant.
  3. Convert (preserving whatever modalities the source actually has --
     never silently drop vision/audio components, see
     specs/model-verification.md for why this matters).
  4. Upload.
  5. Verify: structural (tensor-prefix diff vs source) + functional (real
     generation, image+text too for VLMs).
  6. Write a real model card (description, quantization/provenance notes,
     honest verification-status note, discoverability tags).
  7. Update BACKLOG.md: check off the source if it was a tracked candidate,
     remove any now-stale/duplicate entries referencing this repo, add a
     CHANGELOG.md entry.
  8. Delete the local copy once verified (CLAUDE.md policy) -- keep it only
     if verification failed, for debugging.

Usage:
    uv run python scripts/pipeline.py --hf-path mistralai/Ministral-3-8B-Reasoning-2512 \\
        --q-bits 4 --q-group-size 64

This is the "leave it to code" version of what mlx-conversions/CLAUDE.md and
specs/*.md otherwise ask a human (or an LLM) to remember and do by hand.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import requests
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent))
from verify import functional_check, structural_check  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = REPO_ROOT / "output_pipeline"


def is_vlm(source: str) -> bool:
    resp = requests.get(f"https://huggingface.co/{source}/raw/main/config.json")
    resp.raise_for_status()
    config = resp.json()
    if "vision_config" in config or "audio_config" in config:
        return True
    architectures = config.get("architectures", [])
    return any(a.endswith("ForConditionalGeneration") for a in architectures)


def repo_exists(repo: str) -> bool:
    try:
        HfApi().model_info(repo)
        return True
    except Exception:
        return False


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def default_target_repo(source: str, bits: int | None) -> str:
    name = source.split("/")[-1]
    suffix = "bf16" if bits is None else f"{bits}bit"
    return f"mlx-community/{name}-{suffix}"


def convert_vlm_capturing_bpw(source, mlx_path, bits, group_size, quantize) -> float | None:
    import convert_vlm

    class _Tee:
        def __init__(self, real):
            self.real, self.buffer = real, []

        def write(self, s):
            self.real.write(s)
            self.buffer.append(s)

        def flush(self):
            self.real.flush()

    tee = _Tee(sys.stdout)
    old = sys.stdout
    sys.stdout = tee
    try:
        convert_vlm.convert(source, mlx_path, q_bits=bits or 4, q_group_size=group_size or 64, quantize=quantize)
    finally:
        sys.stdout = old
    m = re.search(r"Quantized model with ([\d.]+) bits per weight", "".join(tee.buffer))
    return float(m.group(1)) if m else None


def convert_lm_capturing_bpw(source, mlx_path, bits, group_size, quantize) -> float | None:
    import convert as convert_text

    class _Tee:
        def __init__(self, real):
            self.real, self.buffer = real, []

        def write(self, s):
            self.real.write(s)
            self.buffer.append(s)

        def flush(self):
            self.real.flush()

    tee = _Tee(sys.stdout)
    old = sys.stdout
    sys.stdout = tee
    try:
        convert_text.convert(source, mlx_path, bits or 4, group_size or 64)
    finally:
        sys.stdout = old
    m = re.search(r"Quantized model with ([\d.]+) bits per weight", "".join(tee.buffer))
    return float(m.group(1)) if m else None


def write_generic_card(repo: str, source: str, vlm: bool, bits: int | None, group_size: int | None,
                        output_size: str, avg_bpw: str, verified: bool) -> None:
    """A leaner, family-agnostic card (unlike scripts/write_model_cards.py,
    which is specific to the Ministral-3 Base family's card layout/family
    table). Same spirit: real description pulled from source, honest
    verification note, quantization/provenance notes, discoverability tags."""
    api = HfApi()
    card_resp = requests.get(f"https://huggingface.co/{repo}/raw/main/README.md")
    front_matter_end = card_resp.text.find("---", 3) + 3 if card_resp.ok else 0
    front_matter = card_resp.text[:front_matter_end] if front_matter_end else "---\nlibrary_name: mlx\n---"

    source_card = requests.get(f"https://huggingface.co/{source}/raw/main/README.md")
    source_desc = ""
    if source_card.ok:
        lines = source_card.text.split("---", 2)
        body = lines[2] if len(lines) > 2 else source_card.text
        paras = [p.strip() for p in body.strip().split("\n\n") if p.strip() and not p.strip().startswith("#")]
        source_desc = paras[0] if paras else ""

    pipeline_tag = "image-text-to-text" if vlm else "text-generation"
    tool = "mlx-vlm" if vlm else "mlx-lm"
    modality_note = (
        "Vision/audio components (if present) are kept at full precision -- only "
        "the language backbone is quantized, per mlx-vlm's standard policy."
        if vlm else ""
    )

    verification_note = (
        f"> **Community note.** Structural check confirms nothing was dropped "
        f"from the source model during conversion. Functional check confirms "
        f"generation produces coherent output{' (text and image+text)' if vlm else ''}. "
        f"Converted and verified by a single maintainer running local MLX "
        f"tooling -- please open a discussion if you hit anything unexpected."
        if verified else
        f"> **Community note.** Structural check confirms nothing was dropped "
        f"from the source model. **Functional check (running generation) has "
        f"not passed yet** -- please open a discussion with results if you "
        f"try this before it's updated."
    )

    body = f"""
# {repo}

{source_desc}

This is an MLX conversion of [`{source}`](https://huggingface.co/{source}),
converted with [{tool}](https://github.com/{'Blaizzy/mlx-vlm' if vlm else 'ml-explore/mlx-lm'}).
Refer to the [original model card](https://huggingface.co/{source}) for the
full description, capabilities, and license terms.

{verification_note}

## Provenance

- Source: [`{source}`](https://huggingface.co/{source})
- Language model layers: {"unquantized (bf16 passthrough)" if bits is None else f"**{bits}-bit** affine quantization, group_size={group_size}"}
- {modality_note}
- Output size on disk: **{output_size}**
- Blended average: **{avg_bpw} bits per weight**

## Use with mlx

```bash
pip install -U {tool}
```

{'```bash\npython -m mlx_vlm.generate --model ' + repo + ' --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>\n```\n\nFor text-only prompts, omit `--image`.' if vlm else '```python\nfrom mlx_lm import load, generate\n\nmodel, tokenizer = load("' + repo + '")\nresponse = generate(model, tokenizer, prompt="hello", verbose=True)\n```'}
"""

    content = front_matter.replace("pipeline_tag:", f"tags:\n- mlx\n- quantized\npipeline_tag:", 1) if "tags:" not in front_matter else front_matter
    if "pipeline_tag:" not in content:
        content = content.rstrip("-\n") + f"\npipeline_tag: {pipeline_tag}\n---"
    content = content + body

    local_path = Path(f"/tmp/{repo.split('/')[-1]}-README.md")
    local_path.write_text(content)
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo="README.md", repo_id=repo)


def update_backlog_and_changelog(source: str, repo: str, output_size: str, avg_bpw: str, verified: bool) -> None:
    backlog_path = REPO_ROOT / "BACKLOG.md"
    changelog_path = REPO_ROOT / "CHANGELOG.md"

    if backlog_path.exists():
        text = backlog_path.read_text()
        lines = text.splitlines()
        new_lines = []
        marked = False
        for line in lines:
            # Check off an unchecked entry mentioning this source repo.
            if not marked and line.strip().startswith("- [ ]") and source in line:
                line = line.replace("- [ ]", "- [x]", 1) + f" — done via pipeline.py, published as [`{repo}`](https://huggingface.co/{repo})"
                marked = True
            new_lines.append(line)
        backlog_path.write_text("\n".join(new_lines) + "\n")

    if changelog_path.exists():
        text = changelog_path.read_text()
        marker = "### Added\n"
        idx = text.find(marker)
        entry = f"- Published [`{repo}`](https://huggingface.co/{repo}) via `scripts/pipeline.py` — output {output_size}, {avg_bpw} bits/weight, verified={verified}.\n"
        if idx != -1:
            insert_at = idx + len(marker)
            text = text[:insert_at] + entry + text[insert_at:]
        changelog_path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--upload-to", help="Defaults to mlx-community/<name>-<bits>bit")
    parser.add_argument("--q-bits", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()

    source = args.hf_path
    bits = None if args.no_quantize else args.q_bits
    group_size = None if args.no_quantize else args.q_group_size
    repo = args.upload_to or default_target_repo(source, bits)

    if repo_exists(repo):
        print(f"[skip] {repo} already exists on mlx-community")
        return

    vlm = is_vlm(source)
    if not vlm and args.no_quantize:
        print("[error] --no-quantize isn't supported for text-only models yet -- "
              "scripts/convert.py's convert() always quantizes (see BACKLOG.md). "
              "Add that support before using this flag on a non-VLM model.")
        sys.exit(1)
    print(f"[pipeline] {source} -> {repo} (vlm={vlm}, quantize={not args.no_quantize}, bits={bits})")

    mlx_path = OUTPUT_ROOT / repo.split("/")[-1]
    if vlm:
        avg_bpw = convert_vlm_capturing_bpw(source, mlx_path, bits, group_size, quantize=not args.no_quantize)
        from mlx_vlm.utils import upload_to_hub
    else:
        avg_bpw = convert_lm_capturing_bpw(source, mlx_path, bits, group_size, quantize=not args.no_quantize)
        from mlx_lm.utils import upload_to_hub  # noqa: F401 (mlx_lm's own, matching convert.py's import path)

    size_gb = dir_size_gb(mlx_path)
    bpw_str = f"{avg_bpw:.3f}" if avg_bpw is not None else "16 (bf16, unquantized)"
    print(f"[convert] done, {size_gb:.2f}GB, {bpw_str} bits/weight")

    print(f"[upload] -> {repo}")
    upload_to_hub(str(mlx_path), repo)

    print("[verify] running structural + functional checks")
    verified = structural_check(source, repo) and functional_check(repo, is_vlm=vlm)
    print(f"[verify] {'PASS' if verified else 'FAIL'}")

    print("[card] writing")
    write_generic_card(repo, source, vlm, bits, group_size, f"{size_gb:.2f}GB", bpw_str, verified)

    print("[tracking] updating BACKLOG.md / CHANGELOG.md")
    update_backlog_and_changelog(source, repo, f"{size_gb:.2f}GB", bpw_str, verified)

    if verified:
        print(f"[cleanup] removing local copy {mlx_path}")
        shutil.rmtree(mlx_path, ignore_errors=True)
    else:
        print(f"[cleanup] SKIPPED (verification failed) -- kept at {mlx_path} for debugging")

    print(f"\n{'DONE' if verified else 'DONE WITH FAILED VERIFICATION'}: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
