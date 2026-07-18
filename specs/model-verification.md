# Model Verification

## Status

Draft. Not yet implemented as a script — this documents what "done" should
mean for a conversion, informed by a real bug found 2026-07-18.

## Why this exists

The first pass at converting the Ministral-3 Base family (3B/8B/14B) was
published to mlx-community after only checking that `mlx_lm.convert` exited
cleanly and produced a plausible file size. It was wrong: `mlx_lm` is
text-only, and Ministral-3 is a vision-language model (text backbone +
vision tower + multimodal projector). The conversion silently dropped the
entire vision tower (218 of 458 source tensors) and multimodal projector,
with no error — `mlx_lm.utils.load()` just loads whatever it recognizes and
ignores the rest. The published models were live on a public community org
for several hours before this was caught, purely by chance (reading a
source model's own README description mentioned "vision capabilities",
which prompted checking whether the conversion actually kept them).

**A pure generation smoke test would NOT have caught this.** The language
backbone alone still loads and generates coherent text — the bug is an
entire missing modality, not degraded quality in the parts that exist. This
is why verification needs both a structural check and a functional check;
neither alone is sufficient.

## What "verified" should mean before publishing

### 1. Structural check (catches silently-dropped components)

Compare the converted output against the source repo *before* even running
generation:

- Fetch the source repo's `model.safetensors.index.json` (or scan its
  `.safetensors` files) and collect the set of top-level module prefixes
  (e.g. `language_model`, `vision_tower`, `multi_modal_projector`, `sam_model`,
  `audio_tower` — anything beyond the plain text decoder).
- Compare against the converted output's own tensor prefixes.
- If the source has prefixes the output doesn't, that's a hard failure —
  something was silently dropped. Don't publish; figure out which
  conversion tool (`mlx_lm` vs `mlx_vlm`) actually preserves that content.
- Cross-check `config.json`: does the source's config have `vision_config`,
  `audio_config`, or similar multimodal blocks the output's config lacks?
  A stripped config is the same signal from a different angle.

This check is cheap (no model loading, no compute) and should run
immediately after conversion, before quantization time is even "spent" on
something that'll need to be redone.

### 2. Functional check (catches broken-but-structurally-complete conversions)

Once structure matches, actually run the model:

- Text-only models (`mlx_lm`): `mlx_lm.generate` with a short prompt,
  confirm the output is coherent (not garbage tokens, not immediate EOS,
  not a crash).
- Vision-language models (`mlx_vlm`): `mlx_vlm.generate` with both a
  text-only prompt and an image + prompt, confirm both produce coherent
  output — a structural check alone doesn't prove the vision pathway
  actually *works*, only that the weights are present.
- For quantized models specifically: sanity-check that quantized layers
  didn't degrade to nonsense (a garbled-but-technically-non-crashing output
  is a real failure mode, not just "coherent vs not").

### 3. What to record

Once both checks pass, note in `CHANGELOG.md` (already the convention for
this repo):
- Conversion tool used (`mlx_lm` vs `mlx_vlm`) and why
- Structural check result (prefixes matched, nothing dropped)
- A one-line example of the functional check's output, so a future reader
  doesn't have to re-derive "does this actually work" from scratch

## How to pick the right tool up front (avoid needing this check at all)

Before converting any new model, check its source `config.json`:
- `vision_config`, `audio_config` present, or `architectures` ending in
  `ForConditionalGeneration` → use `scripts/convert_vlm.py` (mlx-vlm)
- Otherwise → `scripts/convert.py` (mlx-lm) is fine

This is now a `BACKLOG.md`/`CLAUDE.md` constraint, not just a suggestion —
getting this wrong is exactly what happened here.

## Not yet done

- [ ] Write an actual `scripts/verify.py` that automates the structural
  check (tensor-prefix diff) so it doesn't rely on remembering to check
  manually
- [ ] Decide whether the functional check needs a real image asset checked
  into the repo (for VLM models) or can use a generated/placeholder one
- [ ] Retroactively verify all 3 already-corrected Ministral-3 Base models
  with the functional check (structural check has already been done ad hoc
  for the 3B; 8B/14B were converted with the same fixed script but not
  individually re-verified tensor-by-tensor)
