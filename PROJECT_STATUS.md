# Project status

A handover file. It says where the work is, what is already answered, what is
running elsewhere, and what the next concrete step is - enough for someone (or
some later session) to resume without re-deriving any of it. No credentials.

Last updated: 2026-09-18, after the Fish S2 Pro run finished.

## Stage

Measuring candidate engines on one fixed benchmark, and repairing the analysis
that the earlier measurements were reported with. Not training anything yet.

## Last healthy commit

`ed19ad0` - "Stop treating 27 cells as 27 independent observations". Tree clean,
214 tests plus 13 worker tests pass, `ruff check src scripts tests` clean.

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

## Open corrections, in order

1. **done** - speaker-clustered statistics (`scripts/compare_runs.py`).
2. `src/voiceclone/pipelines/results.py` still carries an early small-sample
   calibration (0.288 / 0.544 / 0.647 / 0.800) while the benchmark measures
   0.132 / 0.508 / 0.689 on 900 and 90 pairs. The app's verdict strings are
   computed from the stale set.
3. The corpus drops speakers using the same ECAPA that scores the results
   (`scripts/build_eval_corpus.py`, `verify()`). That biases the absolute scale
   - ceiling up, floor down - though not the comparison between engines, which
   all see the same speakers.
4. The manifest calls each Common Voice clip a separate "session". Common Voice
   publishes no session field; the evidenced statement is "a different clip".
5. FINDINGS and README still quote the cell-level statistics.

## Next concrete step

Item 2 above: replace the calibration constants in `results.py` with the
measured ones, scope them to this corpus and encoder, and drop the
same-session band that has no evidence behind it.

## Blockers

None technical. Two standing limits: no new spend or GPU purchase, and no new
upload of the user's private audio, without the user saying so.
