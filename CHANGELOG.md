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
- `scripts/convert.py` no longer calls `mlx_lm.convert.convert()` directly. That wrapper's `save()` step calls `hf_repo_to_path()` with `local_files_only=True`, which requires the *entire* source repo snapshot to be cached — including files mlx_lm never actually reads (just `*.py` and `generation_config.json`). Mistral-family repos ship a large redundant `consolidated.safetensors` (a duplicate of the sharded weights in a different layout, ~7.7GB for the 3B) that isn't part of mlx_lm's normal fetch, so this check failed and would have forced downloading that extra file for nothing. Replaced with a manual replication of `convert()`'s steps (`load` → `quantize_model` → `save_model`/`save_config`/tokenizer save → narrow `snapshot_download` for the two patterns actually needed → `create_model_card`), which fetches only what's used.
- Also fixed: the script previously pre-created the output directory before calling `convert()`/`save_model()`, which errors if the path already exists — now only the parent directory is created ahead of time.
