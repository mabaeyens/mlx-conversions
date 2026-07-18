## Project
Converts HuggingFace models to MLX quantized format and publishes them to mlx-community. Python 3.11+, mlx-lm, standalone — no dependency on mira-core or any other project.

## Reference docs
| File | When to read it |
|---|---|
| README.md | Setup and how to run a conversion |
| BACKLOG.md | Current conversion queue, what's next |
| CHANGELOG.md | What's already been converted and published |

## Commands
```bash
uv sync
uv run python scripts/convert.py --hf-path <repo> --upload-to <mlx-community/name> --q-bits 4 --q-group-size 64
```

## Constraints
- Conversion only runs on Apple Silicon (Metal) — never attempt a CUDA/cloud-GPU path for this repo.
- Default to 4-bit, group_size=64 unless a specific model family's existing mlx-community releases establish a different convention to match.
- Check `mlx-community` doesn't already have the target model/quant before converting — avoid duplicate work.
- No dependency on mira-core, mira-mlx, or any other project in this workspace — this repo must stand alone.

## Working style
Solo maintainer, low-frequency iteration. Prefer plain scripts over frameworks. Log every conversion (source repo, quant settings, output size, time taken) in CHANGELOG.md as it happens, not after.
