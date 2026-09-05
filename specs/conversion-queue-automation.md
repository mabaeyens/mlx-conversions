# Conversion Queue Automation

## Status

**Removed 2026-09-05.** The recurring cloud trigger (`trig_01X1Z7iv5vwZVLwyG423YY82`,
Saturdays 07:00 UTC) was decommissioned after its first real run: the
execution environment it ran in blocks `huggingface.co` at the network
egress layer, so the scan could never fetch data — every run would have
failed identically. The trigger needs manual deletion from the claude.ai/code
scheduled-tasks UI (no tool available from within a session can remove a
durable cloud trigger). `scripts/scan_candidates.py` still implements the
logic below and works fine for a manual, local run in an environment that
can reach the HF API — it's just no longer invoked automatically.

The candidate-scan logic below exists only as ad hoc research done inline
in a chat session (2026-07-18), formalized into a repeatable process and,
briefly, a recurring job (see above for why that job is gone).

## Part 1 — Multi-precision variants

mlx-community's convention for a popular model family is to publish across
several bit-depths (commonly 3/4/5/6/8-bit + bf16), not just one. Our
existing conversions (Ministral-3 Base 3B/8B/14B) only published 4-bit.

**Plan:**
- Publish additional 6-bit and bf16 variants for the 3B and 8B (small enough
  that hosting multiple precisions isn't wasteful).
- Skip additional precisions for the 14B unless there's a specific reason —
  bf16 at that size (~28GB, before accounting for the vision tower being
  full-precision in every quantized variant already, which shrinks the gap
  between "4-bit" and "bf16" versions less than it would for a text-only
  model) is expensive to host and re-download for what's a niche
  (non-instruction-tuned) audience.
- Use `scripts/convert_vlm.py` with `--q-bits 6` for the 6-bit variant; for
  bf16, use `--dequantize`-equivalent (skip quantization entirely — check
  whether `convert_vlm.py` needs a `--no-quantize` flag added, since
  currently `quantize=True` is hardcoded in `convert()`).

## Part 2 — Periodic candidate scan

The gap-finding process that surfaced the original 3-model queue (diffing
`mistralai`'s HF catalog against what `mlx-community` already has) was done
once, by hand, in a chat session. It should repeat on a schedule so new
`mistralai` releases don't sit unconverted indefinitely.

### The scan logic (to formalize into a script)

1. Fetch `mistralai`'s full model catalog:
   `https://huggingface.co/api/models?author=mistralai&sort=downloads&direction=-1&limit=100`
2. Filter out: gated/guard/moderation models, embeddings, OCR/vision-only
   (note: vision-*language* models like Ministral-3 are still in scope —
   only pure vision-encoder or moderation models get excluded), GGUF/FP8/ONNX
   duplicates, anything with no parseable parameter count.
3. Filter to what's "affordable" on this machine: params ≤ ~16B (the M5's
   32GB unified memory handles conversion comfortably up to 14B per the
   Ministral-3 run's actual timings; leave headroom rather than pushing to
   the exact ceiling).
4. For each remaining candidate, check whether `mlx-community` already has a
   matching conversion — search `mlx-community`'s catalog by every plausible
   name variant (the model name itself, e.g. both `mistral` and `ministral`
   as separate substrings — HF's search is not fuzzy enough to match one
   from a query for the other, which caused a false-positive gap in the
   first manual run of this scan).
5. Report new candidates: name, param count, why it's not yet covered.

### Recurring schedule

**Removed** — see Status above. Was run every Saturday via a durable cloud
trigger (not the session-only `CronCreate`, which is capped at 7 days and
dies with the session); decommissioned because the execution environment
couldn't reach `huggingface.co`. Re-adding this would need an execution
environment with HF API access, not just re-creating the trigger.

Report format: a short list of new candidates (if any), added to
`BACKLOG.md`'s conversion queue directly, plus a one-line note if the scan
found nothing new (so silence isn't ambiguous between "nothing new" and "the
job stopped running").

## Not yet done

- [ ] Write `scripts/scan_candidates.py` implementing the Part 2 logic as a
  standalone, re-runnable script (currently only exists as an inline
  research pass)
- [ ] Add a `--no-quantize` (bf16 passthrough) option to `scripts/convert.py`
  and `scripts/convert_vlm.py` for the multi-precision plan
- [ ] Publish 6-bit + bf16 variants for the 3B/8B Ministral-3 Base models
- [x] ~~Set up the actual recurring Saturday job~~ — done, then removed
  2026-09-05 (see Status above)
- [ ] Decide whether to extend the scan beyond `mistralai` to other orgs
  once the current family is fully covered across precisions
