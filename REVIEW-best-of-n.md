# Review: best-of-N selection (HEAD = c6262b9)

Two defects and one API trap. Everything else from the previous round landed
correctly — this document covers only what still needs work.

Nothing in the repo has been modified. All numbers below are reproduced, not
estimated; the reproduction snippets run standalone with no model weights.

---

## Bug 1 — similarity bucketing does not deliver the property it documents

**File:** `src/voiceclone/pipelines/studio.py`, `_candidate_rank`

The docstring states:

> Both metrics are quantised to their own measurement noise before they are
> compared, because neither resolves differences below it and sorting on what a
> metric cannot resolve is sorting on nothing.

The WER half achieves this exactly. `max(wer, wer_floor)` maps *everything* at
or below the floor onto the identical value, so no pair below the floor can ever
be discriminated. That is a real equivalence.

The similarity half does not. `-round(similarity / similarity_step)` is a grid
with its origin fixed at zero, so whether two takes tie depends on where they
happen to fall relative to arbitrary bucket boundaries — not on how far apart
they are.

### Reproduction (`similarity_step = 0.02`)

| pair | separation | buckets | result |
|---|---|---|---|
| 0.7899 vs 0.7901 | 0.0002 | 39, 40 | **ranked** |
| 0.7900 vs 0.7840 | 0.0060 | 40, 39 | **ranked** |
| 0.7901 vs 0.7999 | 0.0098 | 40, 40 | tied |

The first pair is 49× closer than the third and it is the one that gets ranked.
0.0002 is roughly 100× below the 0.0205 spread the config cites as ECAPA's own
noise. So the actual behaviour is: *takes may or may not tie, depending on their
absolute position on a grid whose origin has no physical meaning.*

### The affected test is grid-dependent, not property-based

```python
def test_similarity_inside_ecapa_noise_is_a_tie(self) -> None:
    assert self._tied([(0.20, 0.788), (0.20, 0.782)]) == 2
```

This passes only because 0.788 and 0.782 happen to share a bucket. Add 0.002 to
both — same separation, same physical claim — and it fails:

```
(0.788, 0.782)  ->  buckets 39, 39  ->  tied = 2   PASS
(0.790, 0.784)  ->  buckets 40, 39  ->  tied = 1   FAIL
```

A test for "differences below the noise floor are a tie" must hold under
translation. This one does not, so it is not testing that claim.

### Fix: make the tolerance relative to the best observed score

Count how many whole `step`s below the best take each take sits. Everything
within one step of the best lands in bucket 0 and ties. This is exact for both
the winner and the tie count, which is all that is consumed.

```python
def _candidate_rank(
    wer: float | None,
    similarity: float | None,
    best_similarity: float | None,
    wer_floor: float,
    similarity_step: float,
) -> tuple[float, float]:
    """...

    *Similarity* is compared as whole ``similarity_step`` intervals below the
    best take in the batch, so whether two takes tie depends on how far apart
    they are and not on where an absolute grid falls. Rounding on a fixed grid
    was tried first and does not survive translation: with step 0.02, two takes
    at 0.7899 and 0.7901 land in different buckets while 0.7901 and 0.7999 share
    one.
    """
    ranked_wer = max(wer, wer_floor) if wer is not None else wer_floor
    if similarity is None or best_similarity is None or similarity_step <= 0:
        ranked_similarity = float("inf") if similarity is None else 0.0
    else:
        ranked_similarity = float(int((best_similarity - similarity) / similarity_step))
    return (ranked_wer, ranked_similarity)
```

`_render_best_of` computes `best_similarity = max(...)` over the scored takes
before ranking, and the test helpers take the same extra argument.

Verified: **all 11 existing assertions still pass unchanged**, plus the shifted
variant that currently fails. The fix strengthens the property without changing
any documented behaviour.

Two notes while touching this:

- `float("inf")` replaces the current `0.0` for unscored similarity. Today an
  unscored take gets `0.0`, which happens to rank last only because every real
  score maps to a *negative* number. Once the sign convention changes, `0.0`
  would mean "best". The asymmetry documented in the docstring — unscored
  similarity ranks last, unscored WER ranks at the floor — needs a sentinel that
  does not depend on the sign of the other branch.
- The current return annotation is `tuple[float, float]` but `-round(...)`
  yields `int`. Harmless, but the cast above makes it honest.

---

## Bug 2 — best-of-N leaks from the Text-to-Speech tab into Speech-to-Speech

**Files:** `src/voiceclone/ui/app.py` (`on_generate_tts`, `on_generate_s2s`),
`src/voiceclone/pipelines/studio.py` (`_convert_cascade`)

`on_generate_tts` writes the setting onto the shared config:

```python
# Per-request, so one render does not change the setting for the next.
studio.config.evaluation.candidates = max(1, int(best_of))
```

The comment is only true within the Text-to-Speech tab. `on_generate_s2s` never
assigns `candidates`, and `_convert_cascade` calls `self.synthesize(...)`, which
reads `self.config.evaluation.candidates`.

**Consequence:** render once in Text-to-Speech with Takes = 4, then run a
Speech-to-Speech conversion in `cascade` mode. It silently renders four takes —
roughly four times the wait, plus round-trip ASR and speaker scoring per take —
with no control for it in that tab and no indication that it is happening. The
setting persists for the rest of the process.

There is also a concurrency hazard independent of the leak: Gradio can serve
requests concurrently, and mutating `studio.config` from a request handler is a
data race on shared state.

**Fix:** stop mutating shared config. Make it a parameter.

1. `Studio.synthesize(..., candidates: int | None = None)`, falling back to
   `self.config.evaluation.candidates` when `None`.
2. `Studio.convert(..., candidates: int | None = None)`, passed through to
   `_convert_cascade` → `synthesize`.
3. `on_generate_tts` passes `candidates=max(1, int(best_of))` instead of
   assigning to config; delete the misleading comment.
4. Add a Takes slider to the Speech-to-Speech tab and wire it the same way. The
   cascade path is ordinary synthesis, so it benefits exactly as much — and its
   `seedvc` mode should ignore the value rather than pretending to honour it.
5. `command_tts` in `cli.py` can keep assigning to its own config (one config per
   process, no leak), but passing the argument is more consistent.

---

## Trap — `candidate_similarity_step: 0` silently disables similarity ranking

```python
if similarity is None or similarity_step <= 0:
    ranked_similarity = 0.0
```

Reading `configs/default.yaml`, `0` looks like "do not quantise — use the raw
score". It actually means "ignore similarity entirely", collapsing selection to
lowest-clamped-WER with an arbitrary tie-break. Either give `0` the reading a
user expects (no quantisation), or reject non-positive values in config
validation with a message pointing at the intended knob.

---

## Not a defect, but worth labelling

`README.md` now claims MOSS 0.819 vs F5 0.730 speaker similarity. The 0.089 gap
is comfortably outside the 0.0205 noise the project just measured, so the claim
survives its own new standard — good. The accompanying pronunciation claim
(بهار، کردند، باغ، شکفت) is n=1: one sentence, one reference, one seed. Worth
marking as a single observation rather than a general finding, on the same
principle that retired the temperature-sweep conclusion.

Coverage note: `test_a_clean_take_beats_a_marginally_cleaner_but_duller_one` was
removed in this round. It was the only three-take case exercising "all takes
inside the WER floor, similarity decides". Minor, but cheap to restore.

---

## Priority

1. **Bug 2** — a real, silently-triggered 4× cost regression that reaches users.
2. **Bug 1** — selection is currently ranking on measurement noise part of the
   time, which is the exact failure the design set out to avoid.
3. The trap and the labelling.
