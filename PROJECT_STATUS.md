# Project status

A handover file: where the work is, what is already answered, what is running
elsewhere, and the next concrete step - enough to resume without re-deriving
anything. No credentials.

Last updated: 2026-09-18 23:32 +0330 (20:02 UTC). `persian-voice-moss15` **v4
RUNNING**, pushed 19:56:56 UTC - do not push it again while it runs. v3 is
complete and recorded.

## Stage

Development-set measurement. MOSS-TTS-Local-Transformer-v1.5 has its first
full measurement (v3): every gate passed, and at that recipe it does not beat
the F5 baseline. v3 sent the language tag "Persian"; the official string,
found afterwards, is "Persian (Farsi)" - v4 changes only that. No SOTA claim
has been accepted by the independent review.

## Last healthy commit

`a390618` (before this file's commit). Tree clean; `.venv/bin/python -m pytest`
gives 293 passed, 1 skipped (the skip needs Linux `/proc`; the same check ran
on Kaggle in v3 and worked); `ruff check src scripts tests` clean.

## Measured (development set: 9 speakers x 3 sentences, seed 42)

| run | engine | held-out | CER | scored in |
|---|---|---|---|---|
| `omnivoice` | OmniVoice | 0.5833 | 2.6% | kaggle eval-run 2026-09-12 |
| `moss` / `moss-local` | MOSS v1.0 (bf16 on T4) | 0.5587 | 6.5% (Mac) | kaggle moss-run / Mac |
| `finetuned-bestof4` | F5 continued | 0.5462 | 2.2% | Mac |
| `fish` | Fish S2 Pro (fp16, `--half`) | 0.5368 | 9.5% | kaggle fish v5 |
| `higgs` | Higgs TTS 3 port (fp16) | 0.5312 | 2.9% | kaggle higgs v2 |
| `moss15` | MOSS v1.5 (bf16 on T4), tag "Persian" - not the official string | 0.5244 | 9.0% | kaggle moss15 v3 |
| `baseline` | F5 as found | 0.5196 | 2.9% | Mac |

`scripts/compare_runs.py` gives the speaker-level intervals; CER is judged only
inside one scoring environment (`eval/results/_provenance.json` for older files,
`scoring_environment` in new ones). Only OmniVoice - baseline clears both 0 and
the 0.0205 reference line: +0.064 [+0.025, +0.102]. MOSS v1.5 (v3): +0.005
[-0.028, +0.037] over the baseline, -0.059 [-0.112, -0.006] against OmniVoice,
-0.034 [-0.067, -0.001] against MOSS v1.0; CER +5.8 points [+3.9, +7.8] over
the baseline scored in its own session.

## Remote jobs (live state read 2026-09-18 23:26 +0330)

| kernel | state | re-run? |
|---|---|---|
| `persian-voice-fish-smoke` v5, `-higgs-smoke` v2, `-eval-run`, `-moss-run` | complete | no - results pulled |
| `persian-voice-moss15` v1 | complete: worker gate timed out, silent a second into the codec's progress bar | no - `eval/runs/persian-voice-moss15-v1.json` |
| `persian-voice-moss15` v2 | complete: same, and no faulthandler dump ever appeared | no - `eval/runs/persian-voice-moss15-v2.json` |
| `persian-voice-moss15-probe` v1 | complete, CPU image: codec loaded both ways | no - `...-probe-v1.json` |
| `persian-voice-moss15-probe` v2 | complete, 1428 s (14:41-15:05 UTC): the worker was blocked behind the runner's display write | no - `...-probe-v2.json` + `eval/runs/evidence/moss15-probe-v2/` |
| `persian-voice-moss15` v3 | complete, ~18 min: all seven gates passed, 27/27 cells | no - `eval/runs/persian-voice-moss15-v3.json`, evidence in `eval/runs/evidence/moss15-v3/` |
| **`persian-voice-moss15` v4** | **RUNNING since 19:56:56 UTC** - v3 with only the language string changed to "Persian (Farsi)"; results as `moss15-fa`, `moss15-fa-smoke`, `baseline-moss15v4-env` | **no - wait for it** |
| dataset `persian-voice-eval` | **v13** (19:55 UTC), code at `cdc4b36`; all 55 code and config files verified identical by download | - |

v1, v2 and both probes mounted dataset v11 (code at `f21ca87`), whose runner
had the display write inside its read loop; v3 mounted v12 (`89dc14e`).

## What probe v2 established (details and every number: `eval/runs/persian-voice-moss15-probe-v2.json`, recomputed by `kaggle/analyse_probe_v2.py`)

* **measured**: the runner's tee holds 187,498 bytes of the worker stage; the
  notebook showed 123,075 of them. The 64,423 in between - less than one 64 KiB
  read, starting exactly where the display stops - is the chunk the reader was
  stuck writing to the notebook's raw stdout. The worker sat in `pipe_write` in
  all 20 reports (RSS unchanged), the driver in `pipe_read`, ~29 GB free.
  print() output kept arriving the whole time
* **reproduced without a model**: the old reader deadlocks the same way
  against a display nobody reads; the new runner passes the same job
* **not established**: why the notebook's raw stdout stopped taking data in
  that stage after carrying 640 KB earlier
* **inference only**: v1 and v2 of the main kernel stalled the same way - they
  kept no tee to show it

## Corrections (kept as history, not erased)

* The status of 18:35 +0330 called probe v2 "still RUNNING past every budget -
  probably stuck", and the report said "more than 2 h, since about 16:27".
  Wrong: it had run 24 minutes, inside its budgets, and was finishing. It was
  never stopped or deleted.
* `312b757` (bounded waits) was motivated by "a process survived SIGKILL". Probe
  v2 shows no such thing: the group died on SIGTERM. The bounded wait stays as
  correct engineering; it explains nothing here.
* The probe builder assumed its T4 image had torch 2.9.1+cu128; it had
  2.10.0+cu128. The main kernel's v1/v2 image did have 2.9.1.
* The old runner's last line said "process group ended" whatever had happened;
  a timeout was reported as exit -1.

## This round (third review), each its own commit

1. `90f2f9c` probe v2 recorded: service version and `probe.json`, run time vs
   push time, real environment, the tee-vs-display analysis, evidence files
   and the script that recomputes them; the probe builder marked historical
2. `4341b0f` runner: the reader writes only to an unbuffered log file; the
   display has its own thread, a bounded queue that drops, collapsed progress
   bars, a byte limit plus the last 40 lines; the waiting thread never writes
   to the display. `Run` separates timeout / exit code / signal; the group's
   state is what happened, and members a finished command left behind are
   ended. On Linux the monitor names the descriptor a blocked process waits
   on. `scripts/runner_check.py` runs the stalled job, model-free, on the
   notebook's own display before anything downloads. Every notebook
   subprocess has its output captured
3. `89dc14e` MOSS v1.5's official runtime (pyproject at `934d6826`): torch
   2.9.1+cu128, torchaudio 2.9.1+cu128, transformers 5.0.0, safetensors 0.6.2
   installed from PyTorch's cu128 index and checked - the run stops on any
   difference; the pyproject's other pins keep the image's versions, and
   `environment.json` lists every package against its official version
4. `87c33e2` the tokenizer's "incorrect regex pattern" warning is a false
   alarm for MOSS v1.5 (transformers 5.0.0 fires it for any local tokenizer
   whose config lacks `transformers_version`); the suggested regex leaves all
   nine Persian test sentences identical and changes only mixed-case Latin
   words, so the official behaviour stays (`scripts/check_moss15_tokenizer.py`)
5. `587af3d` v3 recorded: all gates passed; results, evidence and the per-cell
   errors (mostly the first word of the sentence mispronounced or dropped)
6. `cdc4b36` the official Persian tag, "Persian (Farsi)" (card table and the
   official v1.5 app at `934d6826`); session 4 changes only that, with its
   reading stated in the builder before the run
7. `a390618` README: Fish's CER against the same-session baseline

Earlier rounds, for reference: second review `79fec9d` `56a963e` `c13ab5c`
`7944081` `0f71c71` `69c17a8`; gates `d5fca50`; listening test `1a60672`
`0795bbb`; v1.5 native archive `81cc65f`; FINDINGS on 16-bit PCM `ef8ea3b`.

## Next concrete step

When v4 ends (a background watcher reports it): pull the output
(`kaggle kernels output marsmanam/persian-voice-moss15`), read `results/gates.json`
and `runs.json`, copy `moss15-fa*.json` and `baseline-moss15v4-env.json` into
`eval/results/`, and record `eval/runs/persian-voice-moss15-v4.json`. Then read
it the way the builder's docstring fixed in advance: CER against its own
session's baseline; against v3 directly only if the two sessions' baseline
controls agree cell for cell in CER too; the tag matters only if the
speaker-level interval of (v4 - v3) excludes 0. Then write MOSS v1.5 into
FINDINGS, README and the roadmap once, with both sessions.

After that, the second official cloning mode (continuation with the reference
transcript) is the next single variable - it needs the official app's prompt
construction, since the transformers processor has no transcript field.

## Open

* engines accept `timeout_seconds` but `_read_reply` never enforces it
* `build_eval_corpus.py` still selects speakers with the judging ECAPA
  (documented, not fixed)
* the multi-reference pilot supports only 4 of 9 speakers; the listening test
  is built but has had no listeners
* for the user: commercial or non-commercial use; ParsVoice access (its terms
  are the user's to accept); who could listen

## Standing limits

No new spend, GPU purchase, acceptance of legal terms (e.g. ParsVoice), upload
of the user's private audio, heavy local training/inference, or git push
without the user. Routine fixes and bounded runs in the free quota are allowed.
