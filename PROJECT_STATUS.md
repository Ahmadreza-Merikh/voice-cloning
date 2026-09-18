# Project status

A handover file: where the work is, what is already answered, what is running
elsewhere, and the next concrete step - enough to resume without re-deriving
anything. No credentials.

Last updated: 2026-09-18, after the second review's corrections (up to the
multi-reference manifest), before MOSS v1.5.

## Stage

Development-set measurement and repair of the analysis. Not training. An
independent review has not accepted any SOTA claim.

## Last healthy commit

`69c17a8` - multi-reference manifest. Tree clean; every test passes
(`.venv/bin/python -m pytest -q`); `ruff check src scripts tests` clean.

## Measured (development set: 9 speakers x 3 sentences, seed 42)

| run | engine | held-out | CER | scored in |
|---|---|---|---|---|
| `omnivoice` | OmniVoice | 0.5833 | 2.6% | kaggle eval-run 2026-09-12 |
| `moss` / `moss-local` | MOSS v1.0 (bf16 on T4) | 0.5587 | 6.5% (Mac) | kaggle moss-run / Mac |
| `finetuned-bestof4` | F5 continued | 0.5462 | 2.2% | Mac |
| `fish` | Fish S2 Pro (fp16, `--half`) | 0.5368 | 9.5% | kaggle fish v5 |
| `higgs` | Higgs TTS 3 port (fp16) | 0.5312 | 2.9% | kaggle higgs v2 |
| `baseline` | F5 as found | 0.5196 | 2.9% | Mac |

`scripts/compare_runs.py` gives the speaker-level intervals; CER is judged only
inside one scoring environment (`eval/results/_provenance.json` for older files,
`scoring_environment` in new ones).

## Remote jobs

| kernel | state | needs re-running? |
|---|---|---|
| `marsmanam/persian-voice-fish-smoke` v5 | complete | no - results pulled |
| `marsmanam/persian-voice-higgs-smoke` v2 | complete | no - results pulled |
| `marsmanam/persian-voice-eval-run` | complete | no |
| `marsmanam/persian-voice-moss-run` | complete | no |
| dataset `marsmanam/persian-voice-eval` | current at `244c46f`-era workers | re-stage before the next kernel |

Nothing is queued. Logs: `kaggle/fetch_log.py marsmanam/<slug>`.

## This round (second review), each its own commit

1. `79fec9d` comparison tool: no interval under 5 speakers or zero spread,
   exact t quantile, failures/missing/non-finite/duplicate keys reported not
   dropped, per-cell scale control, CER judged per scoring environment
2. `56a963e` benchmark records `scoring_environment` (recogniser precision is
   cpu/int8 on the Mac, cuda/float16 on GPU) and whether the ASR floor was
   measured or read from cache
3. `c13ab5c` wording: 0.0205's real provenance (dee1ace, perturbation
   sensitivity of one render), post-hoc rule labelled, probabilistic language,
   selection bias and channel claims narrowed, MOSS CER provenance
4. `7944081` Kaggle log fetcher kept in the repo
5. `0f71c71` roadmap: test set sized for one primary comparison with pilot
   uncertainty; dev/test split; identity from sources, not encoders; MOSS v1.5
   official recipe and licences at pinned revisions
6. `69c17a8` multi-reference manifest: leak-free, shared evaluation pool;
   supports only 4 of 9 speakers, so a descriptive pilot at most

## Next concrete step

MOSS-TTS-Local-Transformer-v1.5 at its official recipe (SOTA_ROADMAP.md §7):
a new out-of-process worker (48 kHz stereo kept on disk, codes passed to the
processor so torchaudio.load is not needed, language tag "Persian" recorded as
an inference), a kernel that smoke-tests one render with load time, per-cell
time and memory on each T4, and only then the 27-cell development benchmark.
bf16 stays bf16: if Turing cannot run it, stop and record the blocker.

## Blockers and standing limits

No technical blocker known. No new spend, GPU purchase, acceptance of legal
terms (e.g. ParsVoice), upload of the user's private audio, heavy local
training/inference, or git push without the user.
