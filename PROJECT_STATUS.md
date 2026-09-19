# Project status

A handover file: where the work is, what is already answered, what is running
elsewhere, and the next concrete step - enough to resume without re-deriving
anything. No credentials.

Last updated: 2026-09-19 04:27 +0330 (00:57 UTC). **Running remotely:**
`marsmanam/persian-voice-omnivoice-probe` **version 2**, pushed 00:52:42 UTC,
RUNNING at 00:53:10 - one question: do gradient checkpointing and DDP bucket
views (memory only, arithmetic unchanged) make OmniVoice's official SDPA
recipe fit a T4 pair, at what seconds per step, and do they leave the first
loss and first-step gradients unchanged (plan and budget, written before the
push: `eval/runs/persian-voice-omnivoice-probe-v2.json`). Version 1 is
complete: the official recipe runs out of memory on a T4 in its first forward
pass (`eval/runs/persian-voice-omnivoice-probe-v1.json`). Do not resubmit or
stop version 2; read its state first. Dataset `persian-voice-eval` is at the
version staged from `007af09`, verified by download.

## Stage

The user's goal as of 2026-09-19: research and the best quality first; the
commercial question no longer blocks. Current work: adapting OmniVoice - the
highest identity score on the development set, not a proven SOTA - to Persian
by its official recipe, on independent data (SOTA_ROADMAP §3, §5, §8).

Before that: development-set measurement. MOSS-TTS-Local-Transformer-v1.5 is measured at its
official recipe (v4, tag "Persian (Farsi)"): every gate passed, and it is not
above the F5 baseline, below OmniVoice and below MOSS v1.0. The tag string made
no detectable difference against v3 ("Persian"). No SOTA claim; the
independent review has accepted none.

## Last healthy commit

`7d1c9f9` (before this file's commit). Tree clean; `.venv/bin/python -m pytest`
gives 293 passed, 1 skipped (the skip needs Linux `/proc`; the same check ran
on Kaggle in v3 and v4 and worked); `ruff check src scripts tests` clean.

## Measured (development set: 9 speakers x 3 sentences, seed 42)

| run | engine | held-out | CER | scored in |
|---|---|---|---|---|
| `omnivoice` | OmniVoice | 0.5833 | 2.6% | kaggle eval-run 2026-09-12 |
| `moss` / `moss-local` | MOSS v1.0 (bf16 on T4) | 0.5587 | 6.5% (Mac) | kaggle moss-run / Mac |
| `finetuned-bestof4` | F5 continued | 0.5462 | 2.2% | Mac |
| `fish` | Fish S2 Pro (fp16, `--half`) | 0.5368 | 9.5% | kaggle fish v5 |
| `higgs` | Higgs TTS 3 port (fp16) | 0.5312 | 2.9% | kaggle higgs v2 |
| `moss15` | MOSS v1.5 (bf16 on T4), tag "Persian" - not the official string | 0.5244 | 9.0% | kaggle moss15 v3 |
| `moss15-fa` | MOSS v1.5, official recipe, tag "Persian (Farsi)" | 0.5066 | 10.9% | kaggle moss15 v4 |
| `baseline` | F5 as found | 0.5196 | 2.9% | Mac |

`scripts/compare_runs.py` gives the speaker-level intervals; CER is judged only
inside one scoring environment (`eval/results/_provenance.json` for older files,
`scoring_environment` in new ones). Only OmniVoice - baseline clears both 0 and
the 0.0205 reference line: +0.064 [+0.025, +0.102] - and OmniVoice - MOSS v1.5
(official), +0.077 [+0.032, +0.122]. MOSS v1.5 (official) - baseline: -0.013
[-0.053, +0.027]; - MOSS v1.0: -0.052 [-0.095, -0.009]; CER +7.7 points
[+2.8, +12.7] over the baseline scored in its own session. Tag effect (v4 -
v3): similarity -0.018 [-0.046, +0.010], CER +1.9 [-3.4, +7.2], both include 0
(the two sessions' baseline controls match in every CER cell and transcript).

## Remote jobs (live state read 2026-09-18 23:54 +0330)

| kernel | state | re-run? |
|---|---|---|
| `persian-voice-fish-smoke` v5, `-higgs-smoke` v2, `-eval-run`, `-moss-run` | complete | no - results pulled |
| `persian-voice-moss15` v1 | complete: worker gate timed out, silent a second into the codec's progress bar | no - `eval/runs/persian-voice-moss15-v1.json` |
| `persian-voice-moss15` v2 | complete: same, and no faulthandler dump ever appeared | no - `eval/runs/persian-voice-moss15-v2.json` |
| `persian-voice-moss15-probe` v1 | complete, CPU image: codec loaded both ways | no - `...-probe-v1.json` |
| `persian-voice-moss15-probe` v2 | complete, 1428 s (14:41-15:05 UTC): the worker was blocked behind the runner's display write | no - `...-probe-v2.json` + `eval/runs/evidence/moss15-probe-v2/` |
| `persian-voice-moss15` v3 | complete, ~18 min: all seven gates passed, 27/27 cells | no - `eval/runs/persian-voice-moss15-v3.json`, evidence in `eval/runs/evidence/moss15-v3/` |
| `persian-voice-moss15` v4 | complete, ~17 min: all seven gates passed, 27/27 cells; only the tag changed from v3 | no - `eval/runs/persian-voice-moss15-v4.json`, evidence in `eval/runs/evidence/moss15-v4/` |
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
8. `09b6373` v4 recorded and read as fixed in advance: the tag made no
   detectable difference
9. `7d1c9f9` MOSS v1.5 written into FINDINGS (§8, with the statistics table),
   README (a row of its own; v1.0's "only unrestricted licence" and "two
   candidates unmeasured" corrected) and the roadmap

Earlier rounds, for reference: second review `79fec9d` `56a963e` `c13ab5c`
`7944081` `0f71c71` `69c17a8`; gates `d5fca50`; listening test `1a60672`
`0795bbb`; v1.5 native archive `81cc65f`; FINDINGS on 16-bit PCM `ef8ea3b`.

## Next concrete step

Nothing is running. The next single variable, if MOSS v1.5 stays in play, is
its second official cloning mode: "continuation", which takes the reference
clip's transcript (the official app's `continuation_clone`; the server docs say
it gives better speaker similarity). The transformers processor has no field
for it, so read the official app's `streaming` module at `934d6826` first and
build the prompt exactly as it does; state the reading before the run; one
session, gated like v3/v4, about 20 minutes of T4 x2.

Whether to spend on that, or go straight to the test set with the candidates
already measured, depends on the user's commercial / non-commercial answer:
MOSS v1.5 is the only candidate here with every component checked Apache-2.0,
and it is currently the weakest on this development set.

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
