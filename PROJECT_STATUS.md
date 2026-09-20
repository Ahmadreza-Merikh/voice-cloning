# Project status

A handover file: where the work is, what is already answered, what is running
elsewhere, and the next concrete step - enough to resume without re-deriving
anything. No credentials.

Last updated: 2026-09-20 23:55 +0330 (2026-09-20 20:25 UTC).

**Running remotely:** `marsmanam/persian-voice-omnivoice-train-b` v1, pushed
2026-09-20 16:05:47 UTC, RUNNING - it mounts session a's tokens and
`checkpoint-2000` and resumes the official recipe. Built with `--hours=6.5`
(not the usual 11.25) because 7.5 GPU-hours are left this week and unused
quota expires: the recipe is untouched, the session simply stops at a lower
step and c resumes. Expected about 1,390 more steps, reaching about 3,400 of
5,000. Do not re-push or cancel it; read its output only after it ends.

**Session a v2 is complete** (pushed 2026-09-19 20:54:15 UTC, 11.25 h budget):
every gate passed and 2,000 of the recipe's 5,000 steps are trained and
checkpointed. Downloads took 4.2 minutes, every step on the first attempt -
the stall that cost v1 a whole session did not recur. Data: 87,499 train clips
/ 127.85 h and 2,251 valid, nothing left out, no Neyshekar test shard
downloaded. Tokenising: 88 shards, 0 failures, 79 min at 97x real time.
Training: 8.69 h on the T4 pair at 15.547 s a step (11% slower than the
13.98 s the budget assumed), peak 13.4 GiB of 14.56, saves every 500 steps,
`checkpoint-2000` plus `export-1000-float16` and `export-2000-float16` kept,
11.1 GiB of output. [measured] train loss: mean 4.367 over the first 100 steps,
4.212 over the last 100; validation loss flat (4.263 -> 4.280 over 144 recorded
batches). [hypothesis] that is consistent with early adaptation at lr 1e-5 and
40% of the recipe; nothing here shows the model is better at Persian - only the
sealed test answers that, after step 5,000. Record and evidence:
`eval/runs/persian-voice-omnivoice-adapt.json`,
`eval/runs/evidence/omnivoice-train-a-v2/`.

**GPU quota** [measured, Kaggle's own `kaggle quota`, 2026-09-20]: 22.50 h used
of 30, 7.50 h left, resets **2026-09-26 00:00 UTC**. The API does report it
after all - `KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle quota` - so the
user no longer has to read it off the site. The OAuth token expired on the
20th; `kaggle auth login --force` is what renews it, and only the user runs it.

Dataset `persian-voice-eval` is at the version staged from `b103fa6`
(2026-09-19 20:45 UTC), verified by download: 251 files byte-identical, the
split file unpacked by Kaggle with identical content. The training code in it
(`omnivoice_adapt.py`, `prepare_adaptation_data.py`, the split, the gates) is
unchanged since `007af09`; new are the bounded downloads and the development
and test sessions' code.

## A Persian-specific model appeared, and was measured (2026-09-20)

* `mehdi-hf/pocket-tts-farsi-v2`, announced on the 19th and packaged as
  `nimaone/persian_tts`: 109.5 M parameters, **973 h of Persian from 2,978
  speakers**, phoneme input through Homo-GE2PE, CPU-speed, CC-BY-NC
* added as the `pocket` engine by its own card's recipe (`ff8101e`) and scored
  on the development set beside the stored OmniVoice renders re-scored in the
  same session
* [measured] held-out similarity **0.333** against OmniVoice's **0.583**;
  paired over the same 27 cells **-0.250**, 7.6 standard errors from zero,
  worse in 26 of 27, and below this corpus's 0.508 different-speaker line. CER
  8.9% against OmniVoice's 2.8%, with 2 of 27 renders the recogniser could not
  read at all
* the integration was checked before the result was believed: prompt lengths of
  4 s and 6 s change little (0.412 / 0.388 against 0.381), and **the authors'
  own released samples score 0.321-0.368 against their own prompts on this
  project's ECAPA** - below what this integration gets, so the wiring is not
  the cause
* so OmniVoice stays the model being adapted. What Pocket-TTS wins is cost
  (0.45 GiB, no GPU, faster than real time on a CPU) and it is the first engine
  here that can synthesise from Homo-GE2PE phonemes
* record: `eval/runs/persian-voice-pocket-tts-v1.json`, FINDINGS §10

## The sealed test gained two arms (2026-09-20, registered before any render)

* as first written the test compares the adaptation against the model it
  started from and answers nothing else - and the development set that chose
  that base is very probably contaminated (8 of 9 speakers are Common Voice)
* so `407ffa5` registers **F5-TTS Persian** and **Pocket-TTS Farsi v2** as
  pre-registered *secondary* arms on the same sealed corpora, same twelve
  texts, same seeds, same reference and held-out clips. The primary comparison
  and its reading are untouched
* they run in their own session (`kaggle/build_omnivoice_arms_kernel.py`,
  slug `persian-voice-omnivoice-arms`), because the test session's interpreter
  carries OmniVoice's pinned stack and installing `f5-tts` or `pocket-tts`
  beside it would change the environment the primary comparison is measured in
* read on **held-out similarity only**: similarity reproduced across machines
  in every test here, CER did not, so CER is compared between the two arms and
  never against the test session
* budget: about 2-2.5 GPU-hours, after the primary test, after the reset
* what it buys: if an arm beats the adapted model on speakers nobody has seen,
  the project says so - which is the one thing the test as written could not
  have revealed

## Stage

The user's goal as of 2026-09-19: research and the best quality first; the
commercial question no longer blocks. The work: adapt OmniVoice - the highest
identity score on the development set, not a proven SOTA, and very probably
trained on eight of its nine speakers - to Persian by its official recipe, on
independent data, with the test decided before any training (SOTA_ROADMAP §3,
§5, §8). No improvement is claimed until the sealed test comparison exists.

## Last healthy commit

`0346c86` (before this file's commit). Tree clean; `.venv/bin/python -m pytest`
gives 354 passed, 1 skipped (the skip needs Linux `/proc`, and ran on Kaggle);
`ruff check src scripts tests` clean.

## The adaptation, where it stands

* **recipe** [source]: `k2-fsa/OmniVoice` at `08be0b4c`; the config the
  official script runs has lr **1e-5** (the examples README's 5e-5 is stale
  since `a65149e0`, 2026-04-22), 5,000 steps, 8,192 tokens per GPU on two GPUs,
  bf16; SDPA config for GPUs older than Ampere; runtime from `uv.lock`; base
  `c5fdb5cc`; tokenizer `528e871c`, sha256 equal to the copy the base bundles
* **data** [measured]: `eval/manifests/adaptation_splits_v1.jsonl.gz`, drawn
  from metadata only - no audio, no encoder. Train 87,499 clips, 127.85 h,
  3,835 speakers (Common Voice 26 cleaned 41.70 h + Neyshekar v6 86.16 h);
  valid 2,251 clips, 3.68 h; development speakers excluded; no speaker in two
  splits; no shared normalised text. Test sets sealed: primary the 28 usable
  Neyshekar test speakers (not on OmniVoice's published training list),
  secondary 30 Common Voice ones (15/15 by self-reported gender)
* **power** [measured]: margin and true effect now separate; primary H0 Δ ≤ 0
  at one-sided alpha 0.025, planned for δ = 0.03: 87% power at the pilot's
  spread (0.050), 36% at its upper end (0.096)
  (`eval/manifests/power_design_v1.json`)
* **probe v1** [measured]: the official recipe does not fit a T4 - both
  configs run out of memory at once, SDPA in its first forward pass with 14 GiB
  allocated. The out-of-memory fallback never ran because of a bug in the
  notebook (it read only the last 25 lines of the launcher's log)
* **probe v2** [measured]: with gradient checkpointing and DDP bucket views it
  trains - 13.98 s a step, 12.3 GiB of 14.56 - and on one GPU the checkpointed
  path gave the same first loss and first-step gradients as the official one
  to every printed digit. Budget: about 23-26 GPU-hours in three sessions
* **full run**: session a v1 stalled in its downloads and was cancelled
  (above); b and c are built
  (`kaggle/omnivoice-train-b`, `-c`) and are pushed only after the previous
  session's output exists
* **the float16 export path** [measured, tiny model]: the probe stopped before
  any save, so the export had never run; `scripts/check_omnivoice_export.py`
  exports a 10.8M-parameter OmniVoice the way the sessions do and loads it with
  `from_pretrained`: the published model's files, both tokenizers, all 26
  tensors equal. On this Mac's torch 2.14 / transformers 5.17 (the OmniVoice
  code is identical to the pinned commit's; Kaggle's pinned torch and
  transformers are not what ran). Session a's first real export is at step
  1,000
* **sealed test** [built, not pushed]: `kaggle/omnivoice-test`
  (`kaggle/build_omnivoice_test_kernel.py`), planned and budgeted before any
  test clip was read (`eval/runs/persian-voice-omnivoice-test.json`): both test
  sets written by `scripts/prepare_test_corpus.py`, twelve texts x seeds 42
  and 43, published model then candidate in one session, read by
  `scripts/read_sealed_test.py`; about 6 h of one T4 pair
* **reference rule amended** [measured, metadata only]: the split's longest
  clip would have been cut by the reference window (over 11.5 s) while keeping
  the whole clip's transcript - 13 of the 28 Neyshekar references. The test
  reference is now the longest clip of 6.5-11.4 s; still 28 Neyshekar
  speakers, no Common Voice role changed (SOTA_ROADMAP §3.2). A Neyshekar test
  speaker has no reserve, so a clip failing the 50 ms duration check would drop
  them; Neyshekar's listed durations are multiples of 60 ms, but all 780 of its
  clips in probe v2 passed that check, and session a's data summary will show
  it for all 56,000

## The app on this Mac (2026-09-19, the user's request)

* OmniVoice is the default engine (`2c0b429`), behind `memory_guard`: the
  recogniser and the engine are never loaded together; the app falls back to F5
  where OmniVoice is not installed
* [measured] the guard alone was not enough: a guarded render reached warning
  memory pressure while the weights loaded (5.1 GB already compressed, no swap)
* `c084be2`, `04fda8f`: macOS's pressure level is read (`voiceclone.memory`);
  nothing starts at warning; OmniVoice is not started below 6 GB available (an
  estimate from three warnings at 3.05-3.37 GB available) and its worker is
  killed the moment warning arrives; the worker loads a verified float16 copy
  (bit for bit, 1.2 GB; made on this Mac in 5.3 s at 0.59 GB resident, record
  in `~/.cache/persian-voice-cloning/omnivoice/float16/<revision>/SOURCE.json`)
* [measured] after both: the load completed without pressure; with 4.4 GB
  available the render reached warning two seconds in and the app stopped it
  cleanly. F5 runs in-process: it is refused at warning, but a load
  already under way cannot be stopped the same way
* `eeb0483` (2026-09-20, the user's request): the transcript no longer has to
  be typed. The reference is transcribed *before* the engine loads and the
  recogniser is released first, which the measurement behind it
  (`scripts/measure_recogniser_release.py`,
  `eval/runs/evidence/recogniser-release/measurement.json`) shows gives back
  the 1.3 GB it took - twice in a row, the process returning to 0.9 GB, with
  the 0.6 GB over its 0.3 GB start being imported runtime, not a leak. The
  speech-to-speech cascade is allowed again on the same rule (both
  transcriptions happen before the load); best-of-N is still refused, because
  it transcribes each take while the engine holds memory. Typing the transcript
  still skips the 6-9 s transcription
* [measured] 2026-09-20 08:44, with the user's apps closed and 8.16 GB
  available: the first **complete** OmniVoice render on this Mac, through that
  new path - reference 1.1 s, transcription 7.2 s, load 5.3 s, synthesis 13.8 s,
  evaluation 4.3 s, no pressure warning at any point, 6.53 GB available at the
  end (`eval/runs/evidence/recogniser-release/omnivoice-render-on-the-mac.json`,
  log beside it). Nobody has listened to the render. The recogniser's own
  transcript of the reference had three wrong words, which is the argument for
  typing it where the words are known

## Remote jobs (live state read 2026-09-19 01:14 UTC)

| kernel | state | re-run? |
|---|---|---|
| `persian-voice-fetch-probe` v1 | complete in 8 min, CPU only: both download modes fetched all 21 GiB | no - `eval/runs/persian-voice-fetch-probe-v1.json` |
| `persian-voice-omnivoice-train-b` v1 | **RUNNING** since 2026-09-20 16:05 UTC, 6.5 h budget | no - read its output when it ends |
| `persian-voice-omnivoice-train-a` v2 | complete: 2,000/5,000 steps, `checkpoint-2000` kept | no - `eval/runs/evidence/omnivoice-train-a-v2/` |
| `persian-voice-omnivoice-train-a` v1 | CANCEL_ACKNOWLEDGED (read 18:10 UTC); log ends at 138 s in the download cell | no - superseded by v2 |
| `persian-voice-omnivoice-probe` v2 | complete, about 14 min: fits with memory-only changes | no - `eval/runs/persian-voice-omnivoice-probe-v2.json`, evidence in `eval/runs/evidence/omnivoice-probe-v2/` |
| `persian-voice-omnivoice-probe` v1 | complete, about 12 min: official recipe out of memory; fallback skipped by a bug | no - `...-probe-v1.json`, evidence in `.../omnivoice-probe-v1/` |
| `persian-voice-moss15` v1-v4, `-moss15-probe` v1-v2 | complete | no - `eval/runs/persian-voice-moss15-*.json` |
| `persian-voice-fish-smoke` v5, `-higgs-smoke` v2, `-eval-run`, `-moss-run` | complete | no - results pulled |

## Next concrete steps

0. Done: session a v2 trained 2,000 steps and session b is running with the
   6.5 h that were left this week. Left: read b's output when it ends, record
   it, and after the quota resets on 2026-09-26 build session c with the usual
   11.25 h (`--hours` only lowers one build; the default is unchanged) to
   finish at step 5,000, then the development session, then the sealed test.
1. When session a ends: read its gates (`session/gates.json` in its output);
   if `train` passed, push `kaggle/omnivoice-train-b` (it mounts a's tokens and
   checkpoint); then c the same way. If a session fails a gate, read why before
   anything is pushed again.
2. When c has ended: re-stage the dataset (`kaggle/build_dataset.py`) with
   this round's code and verify it by download, as for `007af09`.
3. Then one development-set session (`kaggle/omnivoice-dev`) - the five
   float16 exports and the published model scored in the same kernel - and
   the choice by the reading fixed in SOTA_ROADMAP §8.
4. The one test comparison: push `kaggle/omnivoice-test` after the choice
   (built; plan, reading and budget recorded before any test clip was read).

MOSS v1.5's "continuation" mode stays in reserve; it is not run for the
licence alone.

## Measured (development set: 9 speakers x 3 sentences, seed 42)

| run | engine | held-out | CER | scored in |
|---|---|---|---|---|
| `omnivoice` | OmniVoice | 0.5833 | 2.6% | kaggle eval-run 2026-09-12 |
| `moss` / `moss-local` | MOSS v1.0 (bf16 on T4) | 0.5587 | 6.5% (Mac) | kaggle moss-run / Mac |
| `finetuned-bestof4` | F5 continued | 0.5462 | 2.2% | Mac |
| `fish` | Fish S2 Pro (fp16, `--half`) | 0.5368 | 9.5% | kaggle fish v5 |
| `higgs` | Higgs TTS 3 port (fp16) | 0.5312 | 2.9% | kaggle higgs v2 |
| `moss15` | MOSS v1.5, tag "Persian" | 0.5244 | 9.0% | kaggle moss15 v3 |
| `moss15-fa` | MOSS v1.5, official recipe | 0.5066 | 10.9% | kaggle moss15 v4 |
| `baseline` | F5 as found | 0.5196 | 2.9% | Mac |

Every file above has a recipe beside it (`<label>.recipe.json`) - model and
codec revisions, settings, reference clips' sha256, session, results sha256 -
reconstructed for these older files with each field's source named. Intervals
by `scripts/compare_runs.py`, speaker-level. OmniVoice - baseline +0.064
[+0.025, +0.102] and OmniVoice - MOSS v1.5 +0.077 [+0.032, +0.122] clear 0 and
0.0205. MOSS v1.5 - baseline -0.013 [-0.053, +0.027]: no detectable difference.
CER only inside one scoring environment: MOSS v1.0 - OmniVoice +3.7 [+1.4,
+6.0] on this Mac (same recogniser settings and packages, documented in
FINDINGS §8); CER is the recogniser's reading - nobody has listened.

## This round (fourth review), each its own commit

1. `259a4c1` claims cut back to what the intervals and the recogniser support;
   the "same recogniser" basis for Mac CER documented; new caveat that
   OmniVoice very probably trained on eight of the nine development speakers
2. `7f8a116` power design with margin and true effect separate
   (`scripts/power_design.py`); primary comparison declared
3. `7191293` public Persian data surveyed; Neyshekar found; two-source split
   by speaker, without an encoder (`scripts/build_adaptation_splits.py`)
4. `60e9004` OmniVoice's official recipe pinned; the lr discrepancy resolved
   from the repository's history
5. `1732d58`, `d7a549c` the probe and its data writer; Kaggle unpacks `.gz`
   inside datasets, so the split file is checked by content hash
6. `17c8de0` every benchmark run writes its recipe beside its results, once;
   OmniVoice and F5 report the revision and checkpoint they loaded
7. `c7baf98`, `007af09`, `4c11444` probe v1 and v2 recorded; memory-only
   techniques with a gradient-fingerprint check; the runbook
   (`docs/OMNIVOICE-ADAPTATION.md`)
8. `592c64d` recipes reconstructed for the seventeen quoted results
9. `65e5f14` the full run's sessions, with plan and budget before the push
10. `29abd0b` the development-set session that will choose among the exports
    (built, not pushed)
11. `cfa3e1a`, `f1990a0` the twelve test sentences, checked against all 68,532
    distinct texts of the split, the development transcripts and the benchmark
12. `cb083f2` the benchmark renders a written text file whole and records the
    window each reference went through
13. `6d12038` the sealed test sets as corpora: references the pipeline keeps
    whole, the split's reserve rule recomputed and checked against the split
14. `7dafc14` the test's reading as code, pinned by tests on made-up results
15. `4683028` the development session takes its code from the dataset only (the
    training outputs carry older copies of the repository)
16. `d667c6b` the sealed test session, built
17. `2e13f0a` its pre-registration; roadmap, runbook and FINDINGS caveat
18. `cbb447e` the dataset list lacked the test session's three scripts - it
    would have failed after its downloads; staged now, with a "code" gate and a
    test that follows every OmniVoice notebook's scripts to the staged list
19. `32ddfd9` the export path checked on a tiny model
20. `0346c86` the OmniVoice engine waits for its worker with the configured
    deadline and starts a new worker after a hang or a crash - before, one hung
    render would have lost a whole test-set run, and one crash every later cell

Earlier rounds, for reference: third review `90f2f9c` `4341b0f` `89dc14e`
`87c33e2` `587af3d` `cdc4b36` `a390618` `09b6373` `7d1c9f9`; second review
`79fec9d` `56a963e` `c13ab5c` `7944081` `0f71c71` `69c17a8`.

## Corrections (kept as history, not erased)

* 2026-09-19: this file said session a being RUNNING past its first hour
  "suggests (does not show) that the data stage passed". It had not started:
  the session was stalled in its download cell from 138 s. A running status
  says nothing about progress when a cell prints nothing - which is why the
  downloads now report as they go and are bounded

* 2026-09-19: the split's test reference rule (the longest clip) would have
  given 13 of the 28 Neyshekar test speakers a reference the pipeline cuts
  while keeping its whole transcript; amended before any test clip was read
  (`6d12038`, `2e13f0a`)
* 2026-09-19: the MOSS v1.5 records said the tag "was not" v3's problem, that
  one render "is unintelligible", that v1.5 "mispronounces", that the
  recogniser "behaved identically" across sessions, and FINDINGS said all three
  failed MOSS attempts were the runner's fault. Corrected (`259a4c1`): no
  detectable effect is not no effect; transcripts are not listening; identical
  controls cover their 27 inputs; the runner cause is documented for probe v2
  and inferred for v1 and v2
* 2026-09-19: the roadmap planned "one-sided, margin 0.03" from the row for a
  true 0.03 - a design that cannot succeed (`7f8a116`)
* 2026-09-19: the roadmap gave OmniVoice's adaptation lr as 5e-5, copied from a
  stale README table; the executed config says 1e-5 (`60e9004`)
* 2026-09-19: probe v1's notebook missed its own out-of-memory and skipped the
  fallback; fixed and re-run as v2
* 2026-09-18: the status of 18:35 +0330 called MOSS probe v2 "stuck for more
  than 2 h"; it had run 24 minutes and was finishing
* `312b757` (bounded waits) was motivated by "a process survived SIGKILL";
  MOSS probe v2 showed the group died on SIGTERM
* the MOSS probe builder assumed torch 2.9.1 on its image; it had 2.10.0
* the old runner's last line said "process group ended" whatever happened

## Open

* Fish, Higgs and Chatterbox accept `timeout_seconds` but their `_read_reply`
  never enforces it (OmniVoice does since `0346c86`; offered as a separate task)
* the Studio keeps a supplied transcript when the reference window cuts a
  clip over 11.5 s (it re-transcribes only for an engine's own limit) - a user
  with a long reference and a typed transcript gets a mismatched prompt. Left
  as it is until the sealed test has run, so the development inputs do not
  change mid-comparison; benchmark runs now record the window
* `build_eval_corpus.py` still selects development speakers with the judging
  ECAPA (documented; the new splits do not)
* the multi-reference pilot supports only 4 of 9 speakers; the listening test
  is built but has had no listeners
* for the user: ParsVoice and `Thomcles/Persian-Farsi-Speech` are gated - their
  terms are the user's to accept, and neither is needed to start; who could
  listen

## Standing limits

No new spend, GPU purchase, acceptance of legal terms, upload of the user's
private audio, heavy local training or inference, or git push without the
user. Routine fixes and runs in the free quota are allowed; every long run's
budget is written down before it is pushed.
