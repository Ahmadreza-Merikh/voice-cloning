# Review: best-of-N selection (HEAD = eff648c)

Third round. The two defects from the previous review are addressed; one has a
residual that shows up in exactly the scenario the design exists for. This
document is a full rewrite and supersedes the previous version.

Nothing in the repo has been modified. `ruff check src tests` is clean. The
ranking suite — 21 assertions including the parametrised cases — was reproduced
standalone (no model weights) and all of it passes.

---

## Closed

**Best-of-N leaking between tabs.** Fixed properly rather than patched. The
setting is now a `candidates` parameter threaded through
`synthesize` → `convert` → `_convert_cascade`, the config write is gone from
`on_generate_tts`, Speech-to-Speech has its own slider, and `seedvc` is
documented as ignoring the value instead of silently paying for it. The
`synthesize` docstring records *why* it is a parameter — shared `Studio`,
concurrent requests — which is the part that stops it regressing.

**`candidate_similarity_step: 0`.** Now means "no quantisation, compare raw
gaps", which is what the number reads like, and
`test_zero_step_compares_raw_scores` pins it.

**README labelling.** The n=1 pronunciation observation is marked as such, and
the 0.089 similarity gap now carries its own noise-margin justification.

**Coverage.** `test_all_inside_the_floor_leaves_similarity_to_decide` restores
the three-take case dropped last round.

**Tests.** Better than what was asked for. Parametrising the tie property over
`shift = [0.0, 0.002, 0.05, -0.1, 0.199]` turns "must survive translation" from
a review comment into an executable claim. That is the right shape for this
property, and `test_a_hair_apart_is_never_ranked` / `test_a_wide_gap_is_never_tied`
pin the two specific pairs the old grid got backwards.

---

## Open — the similarity reference point is the global maximum, not the maximum
## within the winning WER tier

**File:** `src/voiceclone/pipelines/studio.py`, `_render_best_of`

```python
observed = [row[1] for row in scored if row[1] is not None]
best_similarity = max(observed) if observed else None
```

`best_similarity` is taken across *every* take. But WER comes first in the
lexicographic key, so takes above the floor can never win. When the
most-similar take is one of those, the grid origin is anchored to a take that is
already disqualified, and the translation-dependence the fix removed comes back
for the takes that are actually competing.

This is not a corner case. It is the `r = -0.06` situation the docstring
describes — *"the single most similar take was also the least intelligible"* —
which is the reason WER ranks first at all.

### Reproduction

Take 0 is the real grid data point from the docstring (0.823 similarity at 26.7%
WER). Takes 1 and 2 are two clean takes a hair apart.

| take | WER | similarity | key at HEAD |
|---|---|---|---|
| 0 | 0.267 | 0.8230 | `(0.267, 0.0)` ← disqualified by WER |
| 1 | 0.000 | 0.7031 | `(0.15, 5.0)` |
| 2 | 0.000 | 0.7029 | `(0.15, 6.0)` |

`best_similarity = 0.823`, contributed by the take that cannot win. Takes 1 and
2 are **0.0002 apart — roughly 100× below the 0.0205 noise the step encodes —
and land in different buckets.** `tied` reports 1, so nothing warns either.

Slide the pair a little and the verdict flips, which is the same failure the
fixed grid had; only the origin moved.

### Fix

Anchor the reference point inside the tier that can actually win. Two passes, no
circularity:

```python
# Rank the primary key first, then take the similarity reference point from the
# takes that tie on it. Anchoring on the global maximum reintroduces the grid
# problem whenever the most similar take is disqualified by WER - which is the
# r = -0.06 case this ordering exists for.
ranked_wers = [
    max(row[0], floor) if row[0] is not None else floor
    for row in scored
]
contending = min(ranked_wers)
observed = [
    row[1]
    for row, ranked in zip(scored, ranked_wers)
    if row[1] is not None and ranked == contending
]
best_similarity = max(observed) if observed else None
```

And clamp the gap in `_candidate_rank`, so a take outside the winning tier
cannot score better than the tier's best on the similarity axis:

```python
gap = max(0.0, best_similarity - similarity)
```

The clamp is inert today because WER dominates lexicographically, but without it
out-of-tier takes carry negative similarity keys — a mine for anyone who later
reorders the tuple or exposes the full sorted list.

Verified: **all 21 existing assertions pass unchanged**, and the reproduction
above becomes `tied = 2`, which is the correct answer.

### Also

`if best_similarity is None: return (ranked_wer, 0.0)` is unreachable.
`best_similarity` is `None` only when no take scored, and in that case every
take hits the `similarity is None` branch above it. Harmless, but it is dead
code presented as a live case.

---

## Priority

One change, in `_render_best_of`, plus the one-line clamp in `_candidate_rank`.
The behaviour it corrects is narrow — it needs a batch where the most similar
take is disqualified by WER *and* two contenders straddle a bucket boundary —
but it is silent when it happens, and the first of those two conditions is the
premise the whole ordering is built on.
