## Project
Converts HuggingFace models to MLX quantized format and publishes them to mlx-community. Python 3.11+, mlx-lm + mlx-vlm, standalone — no dependency on mira-core or any other project.

## Reference docs
| File | When to read it |
|---|---|
| README.md | Setup and how to run a conversion |
| BACKLOG.md | Current conversion queue, what's next |
| CHANGELOG.md | What's already been converted and published |
| specs/model-verification.md | What to check before considering a conversion done |
| specs/conversion-queue-automation.md | Multi-precision convention + candidate-scanning process |

## Commands
```bash
uv sync
uv run python scripts/convert.py --hf-path <repo> --upload-to <mlx-community/name> --q-bits 4 --q-group-size 64      # text-only models
uv run python scripts/convert_vlm.py --hf-path <repo> --upload-to <mlx-community/name> --q-bits 4 --q-group-size 64  # vision-language models
```

## Constraints
- Conversion only runs on Apple Silicon (Metal) — never attempt a CUDA/cloud-GPU path for this repo.
- **Before converting any new model, check its source `config.json`.** If it has `vision_config`/`audio_config`, or `architectures` ending in `ForConditionalGeneration`, it's a multimodal model — use `scripts/convert_vlm.py` (mlx-vlm), not `scripts/convert.py` (mlx-lm). `mlx_lm.utils.load()` silently drops any non-text-decoder weights with no error; using it on a multimodal model produces an incomplete conversion missing entire modalities. This happened once already (2026-07-18, Ministral-3 Base family) — see specs/model-verification.md.
- Default to 4-bit, group_size=64 unless a specific model family's existing mlx-community releases establish a different convention to match.
- Check `mlx-community` doesn't already have the target model/quant before converting — avoid duplicate work.
- No dependency on mira-core, mira-mlx, or any other project in this workspace — this repo must stand alone.
- Before publishing, do the structural check in specs/model-verification.md (compare tensor-prefix sets between source and output) — a successful conversion + plausible file size is not sufficient evidence nothing was dropped.
- For vision-language conversions specifically: `mlx_vlm.convert()`'s internal `processor.save_pretrained()` call regenerates `processor_config.json`/`tokenizer_config.json`/`special_tokens_map.json` in a form plain `transformers.AutoProcessor` can't reload (found 2026-07-18 on the Pixtral/Mistral3 processor). `scripts/convert_vlm.py` restores the source's originals verbatim after conversion — don't remove that step, and if writing a new conversion path, keep it.
- **Once a model is uploaded AND has passed `scripts/verify.py` (structural + functional), delete its local copy** (`rm -rf` the output directory) — don't keep converted weights on disk after they're confirmed safe on the Hub. Keep the local copy only while a model is still unverified or mid-debugging.

## Working style
Solo maintainer, low-frequency iteration. Prefer plain scripts over frameworks. Log every conversion (source repo, quant settings, output size, time taken) in CHANGELOG.md as it happens, not after.
