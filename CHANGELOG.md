# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Initial repo scaffold: README, CLAUDE.md, BACKLOG, conversion script (`scripts/convert.py`).
- Published [`mlx-community/Ministral-3-3B-Base-2512-4bit`](https://huggingface.co/mlx-community/Ministral-3-3B-Base-2512-4bit) — first conversion, 4-bit/group_size=64, output 1.95GB. Source download + quantize took well under a minute once cached (initial cold run, including the ~6GB download, was a few minutes).
- Published [`mlx-community/Ministral-3-8B-Base-2512-4bit`](https://huggingface.co/mlx-community/Ministral-3-8B-Base-2512-4bit) — 4-bit/group_size=64, output 4.79GB, 2.5 min end-to-end (cold, including download).
- Published [`mlx-community/Ministral-3-14B-Base-2512-4bit`](https://huggingface.co/mlx-community/Ministral-3-14B-Base-2512-4bit) — 4-bit/group_size=64, output 7.62GB, 4 min end-to-end (cold, including download). This was the tightest fit on the 32GB M5 per the original plan, but ran with no observed memory pressure or swapping — conversion itself is comfortably within budget even at 14B.

### Fixed
- **Critical: the 3 published Ministral-3 Base models were incomplete.** `scripts/convert.py` uses `mlx_lm`, which is text-only — for a vision-language model like Ministral-3 (text backbone + vision tower + multimodal projector), `mlx_lm.utils.load()` silently loads and quantizes only the language backbone and drops the rest, with no error. Verified via tensor-prefix comparison: source has 458 tensors (`language_model`, `vision_tower`, `multi_modal_projector`); the mlx_lm-converted output had 602 tensors, all `language_model`-only, zero vision-related. Added `scripts/convert_vlm.py` (uses `mlx_vlm`, which is architecture-aware and keeps the vision tower + projector at full precision per its own quantization policy) and re-converted + re-published all 3 models. New output sizes: 3B 2.80GB (was 1.95GB), 8B 6.44GB (was 4.79GB), 14B 9.47GB (was 7.62GB). See `specs/model-verification.md` for the structural check that should catch this class of bug before publishing next time, and the new `CLAUDE.md` constraint on picking `convert.py` vs `convert_vlm.py` based on the source model's `config.json`.
- `scripts/convert.py` no longer calls `mlx_lm.convert.convert()` directly. That wrapper's `save()` step calls `hf_repo_to_path()` with `local_files_only=True`, which requires the *entire* source repo snapshot to be cached — including files mlx_lm never actually reads (just `*.py` and `generation_config.json`). Mistral-family repos ship a large redundant `consolidated.safetensors` (a duplicate of the sharded weights in a different layout, ~7.7GB for the 3B) that isn't part of mlx_lm's normal fetch, so this check failed and would have forced downloading that extra file for nothing. Replaced with a manual replication of `convert()`'s steps (`load` → `quantize_model` → `save_model`/`save_config`/tokenizer save → narrow `snapshot_download` for the two patterns actually needed → `create_model_card`), which fetches only what's used. `scripts/convert_vlm.py` has the same problem and the same fix (mlx_vlm's `get_model_path()` allows all `*.safetensors` files by default, which would fetch the redundant file too — pre-fetch with `ignore_patterns` instead, then hand `mlx_vlm.convert.convert()` the resolved local path).
- Also fixed: the script previously pre-created the output directory before calling `convert()`/`save_model()`, which errors if the path already exists — now only the parent directory is created ahead of time.

### Changed
- Published richer model cards for all 3 Ministral-3 Base models, replacing the bare auto-generated boilerplate (title + pip-install snippet, no description) with a real description, quantization notes (what's quantized vs kept full-precision), and a Ministral-3 family table linking sibling mlx-community conversions. Done via a new reusable skill, `mlx-model-card` (`claude-skills/mlx/mlx-model-card/SKILL.md`).
- Added a weekly scheduled cloud routine (Saturdays, 9am Madrid / 07:00 UTC) that re-runs the mistralai-vs-mlx-community candidate scan and appends new findings to this repo's `BACKLOG.md` directly. See `specs/conversion-queue-automation.md`.
