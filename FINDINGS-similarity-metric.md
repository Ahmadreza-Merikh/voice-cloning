# Handoff: the similarity metric is not measuring voice cloning

> **Closed, 3 August 2026.** Every task below is resolved; see `FINDINGS.md`
> for the outcome and the numbers that replaced these. Two figures quoted here
> did not survive re-measurement on a corpus that could be checked: the
> different-speaker floor is 0.132 rather than 0.288, and the same-speaker
> ceiling 0.689 rather than 0.647. Kept for the reasoning, not the values.

Supervisor report, 30 July 2026. Supersedes `REVIEW-best-of-n.md`, which is now
fully closed (see §1).

Measured on an M1 Pro with `speechbrain/spkrec-ecapa-voxceleb` and the four
`mana` reference clips. Everything here is reproducible with
`scripts/measure_ceiling.py`.

---

## 0. What I changed, so nothing surprises you

**Added `scripts/measure_ceiling.py`** — a measurement tool. It imports the
project read-only, renders into `outputs/ceiling/`, and writes nothing to the
voice library or to config. It is not on any runtime path.

**Rewrote this document.**

**I touched no source.** `git diff HEAD -- src configs tests README.md` is empty.
Every change proposed below is unimplemented and yours to make.

One bug of mine, already fixed in the script: I guessed
`SimilarityResult.is_model` when the field is `reliable`. If you extend the
script, the attribute names it depends on are `SimilarityResult.{score, backend,
reliable}`, `QualityScores.{speaker_similarity, word_error_rate}`,
`CloneResult.{audio, output_path, scores}`.

---

## 1. Best-of-N: closed

`9502920` anchors the similarity reference point inside the contending WER tier
and clamps the gap. I re-derived the ranking standalone and ran every case: **23
assertions pass**, including `test_a_disqualified_take_does_not_anchor_the_similarity_scale`
with the 0.8230 / 0.7031 / 0.7029 counterexample, both translation-parametrised
families, and the no-negative-keys invariant. `ruff` is clean across
`src tests scripts`.

The docstring addition explaining *why* the anchor cannot be the batch maximum is
the part that keeps this from regressing. Nothing further is open on best-of-N.

---

## 2. The finding

| what was compared | score |
|---|---|
| clone vs recordings it was **never conditioned on** | **0.534** |
| two genuine recordings of the same human, same comparison | 0.647 |
| clone vs **its own reference** — what the project reports today | 0.788 |
| **inflation** | **0.254** |

Held-out detail: 0.510 / 0.592 / 0.501 against `mana_50-90`, `mana_504-4`,
`mana_559-81`, cloning from `mana_102-61`. F5, seed 42, one take, phonemes off,
WER 18.2%.

Human baseline over six pairs: 0.662 raw (sd 0.042), 0.647 through the pipeline's
preprocessing (sd 0.058). The preprocessed figure is the comparable one, because
`Studio._evaluate` scores the clone against the *processed* reference.

The inflation is **twelve times** the 0.0205 ECAPA noise this project measured. It
is not noise.

### Why

`_evaluate` scores the clone against the clip it was conditioned on. That shares
the room, the microphone, the noise floor and the loudness normalisation.
Zero-shot cloning reproduces channel alongside voice, and ECAPA encodes channel
as well as identity — so "clone vs own reference" is systematically easier than
"recording A vs recording B", and the two numbers are not on one scale.

The first version of this measurement made exactly that mistake and produced an
impossibility: the clone scored 0.788 against a human ceiling of 0.647, i.e. the
synthetic voice resembled Mana more than Mana does. Scoring against held-out
recordings restores the expected ordering — 0.534 < 0.647, a gap of 5.5x noise.

### What the metric is still good for

Same-reference scoring is not worthless. It catches a render that collapsed, and
it is a valid *relative* comparison between engines or settings on a fixed
reference, because the channel term is then constant. It is simply not a measure
of identity transfer, and must stop being reported as one.

---

## 3. Do these now — no new data needed

**T1. Make the reported number say which comparison it is.**
`QualityScores.speaker_similarity` is measured against the conditioning
reference. Either rename it to `reference_similarity`, or keep the name and
document precisely what it measures. Files: `src/voiceclone/pipelines/results.py`,
`Studio._evaluate`.

**T2. Stop asserting uncalibrated verdicts.** `QualityScores.similarity_verdict`
returns:

```python
if value >= 0.80: return "excellent match"
if value >= 0.70: return "strong match"
if value >= 0.60: return "fair match"
if value >= 0.45: return "weak match"
```

Two genuine recordings of one human score 0.647 and are reported as a **"fair
match"**. This reaches users through both the UI and the CLI. The bands cannot be
re-derived until T5, but they can stop claiming more than they know — say in the
docstring and in the README table that they are calibrated against same-reference
scoring and do not describe identity transfer. Same table in `README.md`, plus
the "above 0.75 is a strong match" line.

**T3. Add held-out scoring.** When the voice library holds more than one clip of
a speaker, score the render against the clips that were *not* used as the prompt
and report that number alongside. `scripts/measure_ceiling.py` already does this;
lift the logic. This is the number that describes voice cloning.

**T4. Reconcile the two F5 figures in `README.md`** — 0.825 in "Measured
results", 0.730 in the engine table. Both are same-reference. Say which
configuration each belongs to, or drop one. Neither is the cloning number.

**T5. The reference-transcript confidence gate is too lax.**
`transcribe_reference` warns only when `not transcript.is_confident`. For
`mana_102-61` it produced `... باعث ایجاد عوارز جانبی` — `عوارز` for `عوارض` —
beginning mid-sentence with `یا`, and raised no warning. F5 conditions on this
text and the README states a wrong transcript measurably degrades output. Either
the confidence signal is not sensitive enough, or a clip that starts mid-sentence
needs its own check.

---

## 4. Blocked on audio only Ahmadreza can supply

**T6. There is no floor.** Every clip available is one speaker, so nothing
establishes what a *different* person scores. Without it, 0.534 cannot be judged:
a stranger at ~0.45 means cloning is barely above chance; a stranger at ~0.10
means 0.534 carries real identity information.

It needs a few clips of any second speaker in `voices/` under a different
filename prefix (`ali_1.wav`); the script detects the grouping and reports
cross-speaker pairs automatically. **Every threshold in this project — current or
proposed — is unfounded until this exists.**

External calibration does not substitute. The model card reports 0.8% EER on
VoxCeleb1-test (cleaned), so the encoder discriminates well in general, but a
community answer suggesting a 0.7–0.75 same/different threshold "when scaled from
[0,1]" is ambiguous about whether that is raw cosine or `(cos+1)/2` — either well
above or well below both figures here. The empirical floor on this project's own
audio and pipeline is worth more.

---

## 5. Caveats on the 0.534

- **The corrupt transcript above may make it understate F5.** Rerun with a
  hand-corrected reference transcript before treating it as settled.
- **The held-out clips may not be independent.** The filenames
  (`mana_102-61`, `mana_50-90`, `mana_504-4`, `mana_559-81`) look like slice
  indices into one source. If all four come from one session they share the
  channel that inflates the same-reference score, and 0.534 is itself inflated —
  the true figure would be lower. Worth establishing provenance.
- **The human baseline is low for the audio quality.** 0.647 on clips measuring
  SNR 23–29 dB and quality 95–100 is low for same-speaker ECAPA. Worth
  confirming all four clips really are one speaker.
- **n = 1 on every axis** — one reference, one sentence, one seed, one engine. Do
  not quote 0.534 as *the* figure. The *structure* of the finding — that a large
  channel inflation exists — is robust, because it was measured within one run
  against one voice.

---

## 6. Order

1. T6 (needs Ahmadreza) — everything numeric depends on the scale it establishes.
2. T1, T2, T4, T5 — honesty fixes, no new data required.
3. T3 — the real metric.
4. Rerun part 3 with a corrected transcript and with `moss`, for held-out numbers
   worth comparing between engines.

Reproduce:

```bash
.venv/bin/python scripts/measure_ceiling.py --calibrate-only   # bounds, seconds
.venv/bin/python scripts/measure_ceiling.py --engines f5       # + held-out score
```

---

Sources for §4's external notes:
[model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) ·
[threshold discussion](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/discussions/7)
