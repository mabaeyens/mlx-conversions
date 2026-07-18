# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Published the full 6-bit/8-bit/bf16 precision matrix for the Ministral-3 Base family
  (3B/8B/14B), via `scripts/convert_all_precisions.py`:
  [3B-6bit](https://huggingface.co/mlx-community/Ministral-3-3B-Base-2512-6bit) (7.537 bpw),
  [3B-8bit](https://huggingface.co/mlx-community/Ministral-3-3B-Base-2512-8bit) (9.319 bpw),
  [3B-bf16](https://huggingface.co/mlx-community/Ministral-3-3B-Base-2512-bf16),
  [8B-6bit](https://huggingface.co/mlx-community/Ministral-3-8B-Base-2512-6bit) (7.529 bpw),
  [8B-8bit](https://huggingface.co/mlx-community/Ministral-3-8B-Base-2512-8bit) (9.312 bpw),
  [8B-bf16](https://huggingface.co/mlx-community/Ministral-3-8B-Base-2512-bf16),
  [14B-6bit](https://huggingface.co/mlx-community/Ministral-3-14B-Base-2512-6bit) (7.257 bpw),
  [14B-8bit](https://huggingface.co/mlx-community/Ministral-3-14B-Base-2512-8bit) (9.117 bpw),
  [14B-bf16](https://huggingface.co/mlx-community/Ministral-3-14B-Base-2512-bf16).
  All passed structural + functional verification except `14B-bf16`, which passed
  structural but hits a reproducible Metal out-of-memory error during functional
  (generation) verification on this 32GB M5 — see Fixed below.
- `scripts/pipeline.py` + the `mlx-convert` Claude Code skill — end-to-end conversion
  automation: pick `convert.py` vs `convert_vlm.py` from the source's `config.json`,
  convert preserving all modalities, verify (structural + functional), write a real
  model card, upload, update `BACKLOG.md`/this changelog, delete the local copy once
  verified. Only smoke-tested on the skip-if-exists path so far — a real fresh
  conversion through this path, and `--no-quantize` support for text-only models,
  are still unproven (tracked in `BACKLOG.md`).
- `scripts/convert_all_precisions.py` — convert a model at multiple quantization
  precisions in one pass.
- Initial repo scaffold: README, CLAUDE.md, BACKLOG, conversion script (`scripts/convert.py`).
- Published [`mlx-community/Ministral-3-3B-Base-2512-4bit`](https://huggingface.co/mlx-community/Ministral-3-3B-Base-2512-4bit) — first conversion, 4-bit/group_size=64, output 1.95GB. Source download + quantize took well under a minute once cached (initial cold run, including the ~6GB download, was a few minutes).
- Published [`mlx-community/Ministral-3-8B-Base-2512-4bit`](https://huggingface.co/mlx-community/Ministral-3-8B-Base-2512-4bit) — 4-bit/group_size=64, output 4.79GB, 2.5 min end-to-end (cold, including download).
- Published [`mlx-community/Ministral-3-14B-Base-2512-4bit`](https://huggingface.co/mlx-community/Ministral-3-14B-Base-2512-4bit) — 4-bit/group_size=64, output 7.62GB, 4 min end-to-end (cold, including download). This was the tightest fit on the 32GB M5 per the original plan, but ran with no observed memory pressure or swapping — conversion itself is comfortably within budget even at 14B.

### Fixed
- `14B-bf16`'s upload was interrupted mid-transfer by an out-of-band process kill,
  leaving an empty shell repo on `mlx-community` (`.gitattributes` only, 0 commits).
  Found that `convert_all_precisions.py`/`pipeline.py`'s `repo_exists()` check would
  have treated that shell as "already done" on every future retry, permanently
  skipping it — fixed by requiring at least one `.safetensors` file in the repo
  listing, not just a successful `model_info()` call. Resumed the interrupted
  upload from the (already fully converted) local copy via a one-off script
  (no reconversion needed) and confirmed the weights land correctly.
- `14B-bf16` reproducibly fails functional (generation) verification on this 32GB
  M5 — needs ~26.6GB GPU memory, over the ~25.6GB default Metal ceiling, confirmed
  twice including on an otherwise-idle machine. Added an honest
  `HARDWARE_LIMITED_NOTE` card variant in `write_model_cards.py` (rather than
  reusing the generic "hasn't passed yet" wording, which would wrongly imply a
  pending fix) — the weights are an unquantized dtype/format conversion with no
  quantization math applied, so there's no reason to doubt correctness, just no
  way to confirm it on this hardware. Local copy kept per the unverified-models
  policy.
- **Critical: the 3 published Ministral-3 Base models were incomplete.** `scripts/convert.py` uses `mlx_lm`, which is text-only — for a vision-language model like Ministral-3 (text backbone + vision tower + multimodal projector), `mlx_lm.utils.load()` silently loads and quantizes only the language backbone and drops the rest, with no error. Verified via tensor-prefix comparison: source has 458 tensors (`language_model`, `vision_tower`, `multi_modal_projector`); the mlx_lm-converted output had 602 tensors, all `language_model`-only, zero vision-related. Added `scripts/convert_vlm.py` (uses `mlx_vlm`, which is architecture-aware and keeps the vision tower + projector at full precision per its own quantization policy) and re-converted + re-published all 3 models. New output sizes: 3B 2.80GB (was 1.95GB), 8B 6.44GB (was 4.79GB), 14B 9.47GB (was 7.62GB). See `specs/model-verification.md` for the structural check that should catch this class of bug before publishing next time, and the new `CLAUDE.md` constraint on picking `convert.py` vs `convert_vlm.py` based on the source model's `config.json`.
- `scripts/convert.py` no longer calls `mlx_lm.convert.convert()` directly. That wrapper's `save()` step calls `hf_repo_to_path()` with `local_files_only=True`, which requires the *entire* source repo snapshot to be cached — including files mlx_lm never actually reads (just `*.py` and `generation_config.json`). Mistral-family repos ship a large redundant `consolidated.safetensors` (a duplicate of the sharded weights in a different layout, ~7.7GB for the 3B) that isn't part of mlx_lm's normal fetch, so this check failed and would have forced downloading that extra file for nothing. Replaced with a manual replication of `convert()`'s steps (`load` → `quantize_model` → `save_model`/`save_config`/tokenizer save → narrow `snapshot_download` for the two patterns actually needed → `create_model_card`), which fetches only what's used. `scripts/convert_vlm.py` has the same problem and the same fix (mlx_vlm's `get_model_path()` allows all `*.safetensors` files by default, which would fetch the redundant file too — pre-fetch with `ignore_patterns` instead, then hand `mlx_vlm.convert.convert()` the resolved local path).
- Also fixed: the script previously pre-created the output directory before calling `convert()`/`save_model()`, which errors if the path already exists — now only the parent directory is created ahead of time.
- **Second correctness bug on the same 3 models, found running `scripts/verify.py` for the first time:** `mlx_vlm.convert()`'s internal `processor.save_pretrained(mlx_path)` regenerates `processor_config.json`, `tokenizer_config.json`, and `special_tokens_map.json` in a form plain `transformers.AutoProcessor.from_pretrained()` can't correctly reload — it resolves to a bare `PixtralImageProcessor` with no `.tokenizer` attribute instead of the correct composite `PixtralProcessor`, and separately a tokenizer-reconstruction `TypeError` in `add_tokens`. Confirmed this doesn't reproduce on the original source repo or on an existing (non-ours) mlx-community conversion of a sibling model, and disappears entirely when the *source's original* versions of those 3 files are restored verbatim — they're pure metadata, untouched by quantization, so there's no correctness cost to skipping mlx_vlm's regeneration of them. `scripts/convert_vlm.py` now does this automatically after every conversion; hot-fixed the 3 already-published repos via `scripts/fix_processor_metadata.py` (weights were fine, only these 3 files needed replacing, so no reconversion was needed).

### Added
- `scripts/verify.py` — structural (tensor-prefix diff vs source) + functional (real generation, text-only and image+text for VLMs) verification, per `specs/model-verification.md`. Required adding `torch`/`torchvision` as dependencies purely so `transformers`' `PixtralProcessor` class can be instantiated (an import-time check unrelated to any actual PyTorch computation — mlx-vlm never uses PyTorch at runtime).
- `scripts/scan_candidates.py` — standalone version of the mistralai-vs-mlx-community candidate scan, so it's runnable locally and not only embedded in the weekly routine's prompt.
- `--no-quantize` flag on `scripts/convert_vlm.py` for bf16 passthrough (no quantization at all), for the planned multi-precision variants.
- All 3 Ministral-3 Base models now verified PASS (structural + functional, both checks) after the processor-metadata fix above.

### Changed
- Published richer model cards for all 3 Ministral-3 Base models, replacing the bare auto-generated boilerplate (title + pip-install snippet, no description) with a real description, "Heads up"/"Provenance" sections (style modeled on `mlx-community/Inkling-mlx-4bit`), an honest verification-status callout, expanded discovery tags (`ministral`, `ministral-3`, `vision-language`, `multimodal`, `quantized`, `edge`, `base-model`, `{bits}-bit`), and a Ministral-3 family table linking sibling mlx-community conversions. Done via the `mlx-model-card` skill (`claude-skills/mlx/mlx-model-card/SKILL.md`) and `scripts/write_model_cards.py`.
- Added a weekly scheduled cloud routine (Saturdays, 9am Madrid / 07:00 UTC) that re-runs the mistralai-vs-mlx-community candidate scan and appends new findings to this repo's `BACKLOG.md` directly. See `specs/conversion-queue-automation.md`.
- New policy (now in `CLAUDE.md`): once a model is uploaded AND passes `scripts/verify.py`, its local copy is deleted. Freed ~33GB after verifying all 3 Base models (`output/` — the superseded mlx_lm-only conversions — and `output_vlm/` were both removed).
