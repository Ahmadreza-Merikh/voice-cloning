# Project status

A handover file: where the work is, what is already answered, what is running
elsewhere, and the next concrete step - enough to resume without re-deriving
anything. No credentials.

Last updated: 2026-09-18, MOSS v1.5 v1/v2 hung in codec loading; CPU probe did not reproduce; T4-image probe running.

## Stage

Development-set measurement and repair of the analysis. Not training. An
independent review has not accepted any SOTA claim.

## Last healthy commit

`d5fca50` plus the dataset commit after it - gated MOSS v1.5 kernel. Tree
clean; every test passes (`.venv/bin/python -m pytest -q`); `ruff check src
scripts tests` clean.

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
| `marsmanam/persian-voice-moss15` v1 | complete: worker gate timed out in codec loading, all later stages skipped | no - recorded in `eval/runs/persian-voice-moss15-v1.json` |
| `marsmanam/persian-voice-moss15` v2 | complete: worker gate timed out again, no stack dump ever printed | no - `eval/runs/persian-voice-moss15-v2.json` |
| `marsmanam/persian-voice-moss15-probe` v1 | complete, CPU image: codec loaded in 20 s drained, 13 s inherited - hang not reproduced | no - `eval/runs/persian-voice-moss15-probe-v1.json` |
| `marsmanam/persian-voice-moss15-probe` **v2** | **pushed 2026-09-18, T4 image, ten-minute budgets, worker's full load path** | **no - read `probe.json` first** |
| dataset `marsmanam/persian-voice-eval` | current: gates, float-WAV v1.5 worker, compare_runs | verified by download before the push |

The v1.5 kernel writes `results/gates.json` whatever happens: each stage's
pass/fail and reason. Read that before anything else. Logs: `kaggle/fetch_log.py marsmanam/<slug>`.

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

Read `persian-voice-moss15-probe` v2 (v1, on the CPU image, did not reproduce the hang): `results` are in `probe.json` (reading
decided in advance in `kaggle/build_moss15_probe_kernel.py`) and the
per-minute wait channels are in the log. Only then push the T4 kernel again
(`kaggle/build_moss15_kernel.py`, session v3), which already runs its worker
drained and monitored.


Wait for `persian-voice-moss15` v1; then `kaggle/fetch_log.py
marsmanam/persian-voice-moss15` and pull `results/` (gates.json,
moss15_smoke.json, moss15-smoke.json, baseline-moss15-env.json and, if the
budget gate passed, moss15.json). A gate failure is a finding: record it with
the log line, fix the cause, and only then push a new version.

Engineering since the last update: `d5fca50` stage gates (skips recorded as
failures, verdicts defined by outputs, budget from measured time and peak
memory), process-group timeouts, float-WAV output and peak-memory records in
the v1.5 worker; `1a60672` blind listening-test builder and scorer (prepares
material only - no listener has been run and nothing is scored); `ef8ea3b`
FINDINGS records that the four older out-of-process workers wrote 16-bit PCM.

Review of 948d2fa, fixed: `0795bbb` listening scorer (explicit "none" for
misread, invalid values never votes, duplicate files/listener ids counted once,
responses bound to the test by name and key fingerprint, references from the
runs' own provenance, a speakers x listeners bootstrap for a final test);
`81cc65f` MOSS v1.5 native archive per session, keyed by the whole recipe,
never overwritten. The running v1 kernel predates `81cc65f`: its native smoke
render of cv02/s1 will be replaced by the full run's (harness renders of both
are kept).

Open, not urgent: engines accept `timeout_seconds` but `_read_reply` never
enforces it, so a hung worker is only bounded by the kernel's process-group
timeout, not by the engine.

## Blockers and standing limits

No technical blocker known. No new spend, GPU purchase, acceptance of legal
terms (e.g. ParsVoice), upload of the user's private audio, heavy local
training/inference, or git push without the user.
