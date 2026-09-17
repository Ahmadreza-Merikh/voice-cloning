# Project status

A handover file. It says where the work is, what is already answered, what is
running elsewhere, and what the next concrete step is - enough for someone (or
some later session) to resume without re-deriving any of it. No credentials.

Last updated: 2026-09-18, after the audit pass and SOTA_ROADMAP.md.

## Stage

Measuring candidate engines on one fixed benchmark, and repairing the analysis
that the earlier measurements were reported with. Not training anything yet.

## Last healthy commit

`0c8e2ac` - "Correct the record: the licences, the statistics, and the two new
engines". Tree clean, 228 tests pass, `ruff check src scripts tests` clean.

## What is measured

Six runs on the same 27 cells (9 speakers x 3 sentences, seed 42, one reference
clip each), all scored by the same CPU-side ECAPA-TDNN:

| run | engine | held-out | CER | where |
|---|---|---|---|---|
| `omnivoice` | OmniVoice | 0.5833 | 2.6% | Kaggle T4 |
| `moss` | MOSS-TTS | 0.5587 | not transcribed | Kaggle T4 |
| `finetuned-bestof4` | F5-TTS continued | 0.5462 | 2.2% | local |
| `fish` | Fish S2 Pro | 0.5368 | 9.5% | Kaggle T4, fp16 |
| `higgs` | Higgs TTS 3 | 0.5312 | 2.9% | Kaggle T4, fp16, ported |
| `baseline` | F5-TTS as found | 0.5196 | 2.9% | local |

Results live in `eval/results/<label>.json`; renders in
`outputs/benchmark/<label>/`. Compare them with:

```
.venv/bin/python scripts/compare_runs.py omnivoice fish higgs baseline
```

After clustering by speaker, exactly one comparison clears both bars:
OmniVoice over the F5 baseline, +0.064 [+0.025, +0.102]. Everything else is a
direction or not separable. "Not separable" is not "equal": with nine speakers
this design cannot see a difference below about 0.04.

## Remote jobs

| kernel | state | what it holds |
|---|---|---|
| `marsmanam/persian-voice-fish-smoke` v5 | complete | the `fish` run, 27/27 |
| `marsmanam/persian-voice-higgs-smoke` v2 | complete | the `higgs` run, 27/27 |
| `marsmanam/persian-voice-eval` (dataset) | current | the harness the kernels mount |

Nothing is queued. Both results are already pulled into `eval/results/`, so
neither needs re-running; a re-push would spend GPU quota for nothing.

Kaggle notes that cost sessions before: a dataset push must land *before* the
kernel that mounts it, `kernels output` returns nothing while a kernel is
running, and the log is only reachable through `api.kernels_logs`.

## Corrections from the audit

1. **done** `ed19ad0` - speaker-clustered statistics (`scripts/compare_runs.py`).
   Only one comparison in the project still clears both bars.
2. **done** `d134b75` - `results.py` recalibrated from the early 39-pair sample
   to the benchmark's 900 and 90 pairs, the unevidenced session bands dropped,
   and a test pins the constants to `eval/results/baseline.json`.
3. **done** `0c8e2ac` - licences corrected (OmniVoice weights are CC-BY-NC, not
   Apache-2.0; the F5 base is CC-BY-NC too), Fish and Higgs added to the record,
   and the corpus-selection and rounding caveats written down.
4. **open** - `scripts/build_eval_corpus.py` still selects speakers with the
   encoder that grades the results. Documented, not fixed; fixing it means
   admitting speakers on evidence that is not the judge, which belongs with the
   corpus widening in SOTA_ROADMAP.md §3.
5. **open** - only the Higgs engine pins a model revision in config. The others
   should, now that the exact shas are recorded in SOTA_ROADMAP.md §4.

## Next concrete step

`SOTA_ROADMAP.md` §8 sets the order. The first two items that need no decision
from the user: measure `MOSS-TTS-Local-Transformer-v1.5` (the only unrestricted
candidate, and the variant measured here is the older one), and test
multi-reference conditioning on the existing corpus, which needs no training and
is the first evidence for or against the channel hypothesis.

## Blockers

None technical. Two standing limits: no new spend or GPU purchase, and no new
upload of the user's private audio, without the user saying so.
