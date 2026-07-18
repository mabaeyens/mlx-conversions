# Backlog

## Conversion queue

Found by diffing `mistralai`'s HF catalog against `mlx-community`'s existing
conversions (2026-07-18). The Instruct/Reasoning checkpoints across 3B/8B/14B
are already fully covered by mlx-community (multiple bit-depths each) — the
open gap is the Base (pretrained, non-chat-tuned) checkpoints:

- [ ] `mistralai/Ministral-3-3B-Base-2512` (~3B, ~6GB bf16) — first target, smallest, validates the workflow end-to-end
- [ ] `mistralai/Ministral-3-8B-Base-2512` (~8B, ~16GB bf16)
- [ ] `mistralai/Ministral-3-14B-Base-2512` (~14B, ~28GB bf16) — tightest fit on 32GB M5, expect some paging during conversion

Default quant: 4-bit, group_size=64 (matches the audit in mira-core's
`docs/model-cache.md` — uniform 4-bit beat mixed-precision on Apple Silicon).
Optionally also publish 6-bit + bf16 for the 3B/8B to match mlx-community's
usual multi-precision convention for a model family; skip bf16 for the 14B
unless there's a specific reason (28GB is expensive to host/re-download for a
niche base-model audience).

## Not yet done

- [ ] Confirm mlx-community org push access (or plan: push to personal namespace first, request transfer)
- [ ] Write `scripts/convert.py` (wraps `mlx_lm.convert` + `huggingface-cli upload`)
- [ ] Run first conversion (3B base) end-to-end, log actual time/RAM behavior in CHANGELOG.md
