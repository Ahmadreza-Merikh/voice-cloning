# The history rewrite of 2026-09-30

Before the repository's first push to GitHub (2026-09-30, about 13:07-13:18
+0330) the user rewrote its history: all 234 commits got new hashes and the
reflog was cleared. The last commit's tree is unchanged - no file in the working
tree was touched by the rewrite - and older trees cannot be compared, because the
old objects are gone. Commit messages are kept, and author dates look kept except
on the last two commits, which now read 13:10:00 and 13:11:00.

The records, PROJECT_STATUS.md, FINDINGS.md and the recipes beside results
cite the **old** hashes, and so do Kaggle outputs made before the rewrite (for
example the dataset's PROVENANCE.md, `staged from 542b2a5`). They are left as
written; this table gives the commit each one now is. Each old hash was matched
to the new history by its exact commit message, read from the session
transcripts where it was printed; the four marked `description` were matched by
the description beside them in PROJECT_STATUS.md instead.

| old | new | date | commit | cited in |
|---|---|---|---|---|
| `c1c4ca4` | `bac8a13` | 2026-08-27 | Keep remote checkpoints out of git | `eval/results/finetuned-bestof4.recipe.json` |
| `ff585b9` | `ef3363e` | 2026-09-12 | Record what the control measured across two machines | `eval/results/omnivoice.recipe.json` |
| `3694f6b` | `509f7e5` | 2026-09-15 | Make the MOSS run survive the load hang instead of betting on its cause | `eval/results/moss.recipe.json` |
| `72e1c55` | `6cdea5f` | 2026-09-18 | Read the reference with soundfile, and keep torchvision out of the way | `eval/results/higgs.recipe.json` |
| `3ee3440` | `ef8655a` | 2026-09-18 | Give the Fish transformer a card to itself, and say where the memory went | `eval/results/fish.recipe.json` |
| `79fec9d` | `6dce45d` | 2026-09-18 | Make the comparison tool refuse the certainty it used to manufacture | `FINDINGS.md`, `PROJECT_STATUS.md` |
| `56a963e` | `8f616a5` | 2026-09-18 | Record what scored each run, starting with the recogniser's precision | `FINDINGS.md`, `PROJECT_STATUS.md` |
| `c13ab5c` | `3e40a57` | 2026-09-18 | Say what 0.0205 is, and stop the documents claiming more than intervals can | `PROJECT_STATUS.md` |
| `7944081` | `3b7d45a` | 2026-09-18 | Keep the Kaggle log fetcher in the repo, not in a scratch directory | `PROJECT_STATUS.md` |
| `0f71c71` | `ef1b19a` | 2026-09-18 | Rewrite the roadmap around a test set sized for its own question | `PROJECT_STATUS.md` |
| `69c17a8` | `20b331a` | 2026-09-18 | Fix the multi-reference experiment in a file, and find it cannot run yet | `PROJECT_STATUS.md` |
| `d5fca50` | `750e8d8` | 2026-09-18 | Gate every stage of the MOSS v1.5 notebook on the one before it actually working | `eval/runs/persian-voice-moss15-v1.json` |
| `f21ca87` | `920060e` | 2026-09-18 | Test whether MOSS v1.5 hangs on an undrained pipe, on a CPU box, before spending more GPU | `eval/runs/persian-voice-moss15-probe-v2.json` |
| `fa7578f` | `d77f2f6` | 2026-09-18 | The CPU probe did not reproduce the hang; repeat it on the T4 image it happened on | `eval/runs/persian-voice-moss15-probe-v2.json`, `kaggle/build_moss15_probe_kernel.py`, `tests/test_kernel_gates.py` |
| `312b757` | `a6147ee` | 2026-09-18 | Never wait without a limit for a process group to die | `PROJECT_STATUS.md`, `eval/runs/persian-voice-moss15-probe-v2.json` |
| `96c64a8` | `df197fa` | 2026-09-18 | Status: the T4-image probe looks stuck, and stopping it is the user's call | `eval/runs/persian-voice-moss15-probe-v2.json` |
| `90f2f9c` | `3cbc414` | 2026-09-18 | Record the T4 probe: the worker was blocked behind the runner's display write, not hung in loading | `PROJECT_STATUS.md` |
| `4341b0f` | `f094ceb` | 2026-09-18 | Keep a command's output in a file and never let the display hold the command up | `PROJECT_STATUS.md` |
| `89dc14e` | `385831a` | 2026-09-18 | Install MOSS v1.5's official runtime instead of trusting the image, and record every version | `PROJECT_STATUS.md`, `eval/results/baseline-moss15-env.recipe.json`, `eval/results/moss15.recipe.json`, `eval/runs/persian-voice-moss15-v3.json`, `scripts/reconstruct_recipes.py` |
| `87c33e2` | `6d16999` | 2026-09-18 | MOSS v1.5's tokenizer warning is a false alarm for Persian: keep the official behaviour | `PROJECT_STATUS.md` |
| `587af3d` | `642ba0a` | 2026-09-18 | Record MOSS v1.5's first full run: every gate passed, and it does not beat the baseline | `PROJECT_STATUS.md` |
| `cdc4b36` | `f15a518` | 2026-09-18 | Send MOSS v1.5 the official Persian tag, 'Persian (Farsi)', and test that one change | `PROJECT_STATUS.md`, `eval/results/baseline-moss15v4-env.recipe.json`, `eval/results/moss15-fa.recipe.json`, `eval/runs/persian-voice-moss15-v4.json`, `scripts/reconstruct_recipes.py` |
| `a390618` | `5fdeaf7` | 2026-09-18 | README: state Fish's CER against the baseline scored in the same session | `PROJECT_STATUS.md` |
| `09b6373` | `c73195b` | 2026-09-18 | Record MOSS v1.5 with the official Persian tag: no detectable change, still below the baseline | `PROJECT_STATUS.md` |
| `7d1c9f9` | `257de18` | 2026-09-18 | Write MOSS v1.5's measurement into FINDINGS, README and the roadmap | `PROJECT_STATUS.md` |
| `259a4c1` | `f413232` | 2026-09-19 | Say only what the intervals and the recogniser support | `PROJECT_STATUS.md` |
| `7f8a116` | `88f5e00` | 2026-09-19 | Size the test for a margin and a true effect kept apart | `PROJECT_STATUS.md` |
| `7191293` | `23fbadb` | 2026-09-19 | Split two CC0 Persian corpora for adapting OmniVoice, by speaker and without an encoder | `PROJECT_STATUS.md` |
| `60e9004` | `b5cb681` | 2026-09-19 | Record OmniVoice's official adaptation recipe at a pinned commit | `PROJECT_STATUS.md` |
| `1732d58` | `e3ba4ec` | 2026-09-19 | Probe OmniVoice's official adaptation recipe on a T4 pair before any long run | `PROJECT_STATUS.md` |
| `d7a549c` | `4675e8e` | 2026-09-19 | Accept the split file Kaggle unpacks, and check it against its recorded hash | `PROJECT_STATUS.md`, `eval/runs/persian-voice-omnivoice-probe-v1.json` |
| `17c8de0` | `9cd95fd` | 2026-09-19 | Write each benchmark run's recipe beside its results, once | `PROJECT_STATUS.md` |
| `c7baf98` | `9005502` | 2026-09-19 | Record the OmniVoice probe: the official recipe does not fit a T4 at 8,192 tokens | `PROJECT_STATUS.md` |
| `007af09` | `d0a563d` | 2026-09-19 | Probe version 2: the memory-only techniques, with a check that they change nothing | `PROJECT_STATUS.md`, `eval/runs/persian-voice-omnivoice-probe-v2.json` |
| `592c64d` | `32affdc` | 2026-09-19 | Reconstruct the recipes behind the results the documents quote | `PROJECT_STATUS.md` |
| `4c11444` | `2d3db59` | 2026-09-19 | Record probe version 2: the official recipe trains on a T4 pair with memory-only changes | `PROJECT_STATUS.md` |
| `65e5f14` | `63ad936` | 2026-09-19 | Run the full OmniVoice adaptation in chained sessions, with its budget fixed first | `PROJECT_STATUS.md` |
| `29abd0b` | `c223a75` | 2026-09-19 | Build the development-set session that will choose among the adaptation's exports | `PROJECT_STATUS.md` |
| `cfa3e1a` | `a9efadc` | 2026-09-19 | Write the twelve test sentences and check none was ever trained on | `PROJECT_STATUS.md` |
| `f1990a0` | `49719cb` | 2026-09-19 | Roadmap: the test sentences exist and are checked | `PROJECT_STATUS.md` |
| `cb083f2` | `b61b7de` | 2026-09-19 | Benchmark: render a written text file whole, and record the reference window each run used | `FINDINGS.md`, `PROJECT_STATUS.md` |
| `6d12038` | `b2bb6ea` | 2026-09-19 | Write the sealed test sets as benchmark corpora, with references the pipeline keeps whole (description) | `PROJECT_STATUS.md`, `eval/runs/persian-voice-omnivoice-test.json` |
| `7dafc14` | `d57e786` | 2026-09-19 | Write the sealed test's reading as code, before any test result exists | `PROJECT_STATUS.md`, `eval/runs/persian-voice-omnivoice-test.json` |
| `4683028` | `ae35582` | 2026-09-19 | Development session: take the code from the dataset, not from a training session's output | `PROJECT_STATUS.md` |
| `d667c6b` | `0550b24` | 2026-09-19 | Build the sealed test session: the chosen export and the published model on both test sets (description) | `PROJECT_STATUS.md`, `eval/runs/persian-voice-omnivoice-test.json` |
| `2e13f0a` | `c7f6332` | 2026-09-19 | Pre-register the sealed test: roles, seeds, reading and budget, before any test clip is read | `PROJECT_STATUS.md` |
| `cbb447e` | `d200103` | 2026-09-19 | Stage the test session's scripts, and check every OmniVoice notebook against the dataset | `PROJECT_STATUS.md` |
| `32ddfd9` | `cee04f1` | 2026-09-19 | Check the float16 export path on a tiny model before the first real export exists | `PROJECT_STATUS.md` |
| `0346c86` | `683c3e0` | 2026-09-19 | OmniVoice engine: wait for the worker with a deadline, and start a new one after losing it (description) | `PROJECT_STATUS.md` |
| `2c0b429` | `f953bfb` | 2026-09-19 | Make OmniVoice the default, behind a memory guard for 16 GB Macs | `PROJECT_STATUS.md` |
| `c084be2` | `5811f05` | 2026-09-19 | Stop at the Mac's memory-pressure warning, and let OmniVoice load a verified float16 copy (description) | `PROJECT_STATUS.md` |
| `04fda8f` | `6209a63` | 2026-09-19 | Build the float16 copy in 64 MB pieces with the page cache bypassed | `PROJECT_STATUS.md` |
| `1da16c3` | `97023d3` | 2026-09-19 | Build a CPU-only session that downloads session a's inputs twice and times them | `eval/runs/persian-voice-fetch-probe-v1.json` |
| `b103fa6` | `1d64da7` | 2026-09-20 | Record the CPU-only download probe and fetch over plain HTTP in the GPU sessions | `PROJECT_STATUS.md`, `eval/runs/persian-voice-omnivoice-adapt.json` |
| `eeb0483` | `8b77082` | 2026-09-20 | Transcribe the reference before a guarded engine loads, then release the recogniser | `PROJECT_STATUS.md` |
| `ff8101e` | `0dd7d65` | 2026-09-20 | Add Pocket-TTS Farsi v2 as an engine, driven by its own card's recipe | `FINDINGS.md`, `PROJECT_STATUS.md`, `eval/runs/persian-voice-omnivoice-test.json`, `eval/runs/persian-voice-pocket-tts-v1.json` |
| `407ffa5` | `54318da` | 2026-09-20 | Register F5 and Pocket-TTS as secondary arms of the sealed test, before anything is rendered | `PROJECT_STATUS.md` |
| `a40801f` | `c6c071d` | 2026-09-21 | Say beside the transcript which window of the recording it covers | `eval/runs/persian-voice-asr-settings-v1.json` |
| `6f3335d` | `8b15d42` | 2026-09-21 | Default to the quality-first inference budgets | `eval/runs/persian-voice-omnivoice-test.json`, `kaggle/build_omnivoice_arms_kernel.py` |
| `5c7d40b` | `3b3ae09` | 2026-09-25 | Leave English text's punctuation and numbers in English | `PROJECT_STATUS.md` |
| `cf62d2d` | `890b6f6` | 2026-09-25 | Build the LongCat probe's Kaggle session on its own dataset | `PROJECT_STATUS.md` |
| `a996118` | `8183add` | 2026-09-25 | Give a newly created dataset the licence field datasets create requires | `eval/results/longcat-1b-probe-en.recipe.json`, `eval/results/longcat-1b-probe-fa.recipe.json`, `eval/results/longcat-3.5b-probe-en.recipe.json`, `eval/results/omnivoice-probe-en.recipe.json`, `eval/results/omnivoice-probe-fa.recipe.json` |
| `968af67` | `12030f7` | 2026-09-25 | Pin the F5 arm to the settings it was registered with | `PROJECT_STATUS.md` |
| `7d94b45` | `8cef94e` | 2026-09-25 | Build the development session for sessions a, b and c, as version 2 | `eval/runs/persian-voice-omnivoice-dev.json` |
| `1af6894` | `ad6a4df` | 2026-09-27 | Declare session c's budget before its push: 11.25 h of a full 30 | `eval/runs/persian-voice-omnivoice-adapt.json` |
| `542b2a5` | `c3369e5` | 2026-09-30 | Record session c: the recipe's 5,000 steps are trained, and the validation loss is flat throughout | `PROJECT_STATUS.md`, `eval/runs/persian-voice-omnivoice-dev.json` |
