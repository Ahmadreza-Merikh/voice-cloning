# Improving the checkpoint on a remote GPU

This is the recipe for the one lever measurement has left standing. It is
written to be run **unchanged**. If something here does not fit the hardware you
have, the answer is different hardware, not smaller numbers — every value below
is doing work, and the section at the end says what each one costs if you cut it.

## Why this and not something cheaper

Everything cheaper has been measured and does nothing.

| Lever | Result |
|---|---|
| `trim_silence`, `denoise` | inside noise |
| `cfg_strength` 1.5 / 2.5, `nfe_step` 64 | inside noise |
| Reference preprocessing | moves the speaker embedding by 0.037 — not the problem |
| Chatterbox (different architecture) | 0.389 held-out against F5's 0.520 |

Six F5 configurations span 0.510–0.524, a range smaller than the encoder's own
0.0205 noise. The inference side is exhausted; what is left is the weights.

## The hypothesis being tested

The Persian checkpoint (`Lumos675/F5_TTS_Persian`, whose model card is a licence
line and nothing else) **reads Persian better than the corpus it is scored
against** — 2.9% CER against a 7.8% recogniser floor — and **transfers identity
poorly** — 0.520 held-out against a 0.689 human ceiling.

Excellent language, weak voice, is the signature of training on many hours from
few speakers: the model learned Persian and lost the ability to generalise
across voices. If that is what happened, continuing training on **many speakers**
should recover speaker generalisation without giving back the pronunciation.

**This is a hypothesis, not a promise.** It has not been shown that 0.520 is
closable at all; no engine measured on this corpus has done better. Run it, then
measure it — and if held-out similarity does not move by more than 0.0205, the
answer is that it did not work, not that it needs more epochs.

---

## 1. Prepare the data (local, cheap)

```bash
.venv/bin/python scripts/prepare_finetune_data.py --out data/finetune-fa --hours 25
```

Selects for **speaker diversity, not hours** — at most 6 clips per speaker,
round-robin — and **excludes every evaluation speaker**, so the benchmark cannot
end up measuring memorisation. Produces `wavs/`, `metadata.csv` (`id|transcript`)
and `dataset_info.json`.

**What comes out: 8797 clips, 2739 speakers, 12.5 h.** Not the 25 h asked for —
the dataset lists 34134 clips but publishes 9985, and that is the whole supply.
This is a ceiling, not a cut: an average of 3.2 clips per speaker is *more*
speaker-diverse than 25 h from a smaller pool would have been, and diversity is
the intervention. The one way past the ceiling is more corpora — see the note at
the end of this section.

Half the audio at the same epoch count would also be half the gradient updates,
so the epoch count doubles to 80 to keep the training budget the recipe was
written for. That is holding the plan steady against a smaller corpus, not
adding to it.

The run takes hours over anonymous Hub requests and will be rate-limited. It
retries, and rerunning skips whatever is already on disk, so an interrupted run
resumes rather than restarts. Set `HF_TOKEN` to be throttled less. If listing the
repository is what keeps failing, cache the listing once and pass it back with
`--repo-files`.

> An earlier serial version of this script died mid-download leaving 1157 wavs
> and no `metadata.csv`. Audio without transcripts trains nothing — check that
> both `metadata.csv` and `dataset_info.json` exist before uploading anything.

**If 12.5 h turns out not to be enough**, the next corpus is
`Thomcles/Persian-Farsi-Speech`, which is gated and needs an accepted licence
plus `hf auth login`. Reach for it only after a run at 12.5 h has been measured —
otherwise there is no way to tell which change did the work.

## 2. Upload

Authenticate once. The OAuth flow is what the current CLI wants; a legacy
`~/.kaggle/kaggle.json` still authenticates reads but not every endpoint:

```bash
.venv/bin/kaggle auth login
```

Then publish the prepared directory as a **private** dataset — it is 2.7 GB of
Common Voice audio and there is no reason for it to be public:

```bash
.venv/bin/kaggle datasets init -p data/finetune-fa
# set title and id in data/finetune-fa/dataset-metadata.json:
#   "id": "marsmanam/persian-f5-multispeaker"
.venv/bin/kaggle datasets create -p data/finetune-fa --dir-mode zip
```

`--dir-mode zip` matters: the default skips subdirectories entirely, which would
upload `metadata.csv` and none of the audio. Kaggle unpacks the archive on its
side, and the notebook copes with either layout.

Check `dataset_info.json` reports the clip count you expect before uploading. A
partial download produces a dataset that trains without complaint on whatever
fraction arrived.

## 3. Train

The notebook is generated, not hand-written — hand-written is how the two traps
below got into this document in the first place:

```bash
.venv/bin/python kaggle/build_kernel.py
KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/train
```

It runs on **GPU T4 ×2**, internet on, with the uploaded dataset attached, and
carries the hyperparameters unchanged. Every cell source lives in
`kaggle/build_kernel.py`, which compile-checks each one before writing the
`.ipynb`.

### Two ways this silently does nothing

Both were in the earlier version of this recipe. Neither raises an error; both
produce a run that looks healthy and finishes having learned nothing.

**`--finetune` is required.** `finetune_cli` only assigns `ckpt_path` inside
`if args.finetune:`. Without the flag, `--pretrain` is ignored outright and the
model trains from random initialisation.

**The published checkpoint carries its own training state.** `model_last.pt`
holds `update: 420000` next to the optimizer and scheduler.
`Trainer.load_checkpoint` branches on that key: when it is present the optimizer
and scheduler are restored and the starting update is 420000. `train()` then
computes `skipped_epoch = 420000 // steps_per_epoch`, which is thousands, so
`range(skipped_epoch, epochs)` is empty and the loop never runs. The fix is to
save only the EMA weights:

```python
state = torch.load(raw_ckpt, map_location="cpu", weights_only=True)
torch.save({"ema_model_state_dict": state["ema_model_state_dict"]}, pretrain)
```

With no `update` key at the top level, `load_checkpoint` takes the other branch,
loads the weights, and returns 0. The notebook asserts this rather than trusting
it.

A third, smaller one: `prepare_csv_wavs` demands a header line reading exactly
`audio_file|text` and rejects relative paths, and in finetune mode it copies the
*Emilia Chinese/English* vocabulary in as `vocab.txt`. The notebook translates
the manifest and points the trainer at the Persian vocabulary explicitly.

### The command it runs

```bash
accelerate launch --mixed_precision=fp16 \
  -m f5_tts.train.finetune_cli \
  --exp_name F5TTS_v1_Base \
  --dataset_name persian_multispeaker \
  --finetune \
  --pretrain "$PRETRAIN_EMA_ONLY" \
  --tokenizer custom --tokenizer_path "$VOCAB" \
  --learning_rate 7.5e-5 \
  --batch_size_per_gpu 12000 --batch_size_type frame \
  --max_samples 64 \
  --grad_accumulation_steps 1 --max_grad_norm 1 \
  --epochs 80 \
  --num_warmup_updates 2000 \
  --save_per_updates 5000 --last_per_updates 2000 \
  --keep_last_n_checkpoints 2 \
  --logger none
```

**Why these values.** `learning_rate`, `batch_size_per_gpu`, `max_samples`,
`grad_accumulation_steps` and `max_grad_norm` are exactly what produced the
checkpoint being continued — verified against its `setting.json`, which reads
`7.5e-05`, `12000`, `frame`, `64`, `1`, `1`, `fp16`. Changing them mid-run
changes the optimisation dynamics against weights already adapted to them.
`batch_size_type frame` keeps the effective batch constant across clip lengths,
which matters when clips run 2.5–10 s. Warmup is lower than the original 6418
because this is a continuation, not a cold start.

**Checkpoints must be redirected.** The trainer writes to
`<f5_tts>/../../ckpts/<dataset_name>`, which on Kaggle is inside the Python
installation and is discarded when the session ends. The notebook symlinks it to
`/kaggle/working/ckpts` before training starts.

Expect roughly 6–10 h on 2×T4 for the 12.5 h of audio at 80 epochs. Kaggle allows
30 GPU-hours a week and 12 h per session, so this fits in one — and
`--save_per_updates 5000` means a session that times out anyway resumes rather
than restarts.

**Watch the first log lines.** `vocab : 2609`, and a loss that starts low
because the weights are already trained rather than at random-initialisation
levels, are what confirm the checkpoint actually loaded.

## 4. Bring it back and measure it

```bash
KAGGLE_CONFIG_DIR=~/.kaggle-oauth \
  .venv/bin/kaggle kernels output marsmanam/persian-f5-multispeaker-train -p remote/
```

**Rename it before pointing anything at it.** The engine converts a training
checkpoint to EMA-only safetensors and caches the result as
`<stem>_ema.safetensors` beside the input, so a second `model_last.pt` in a
directory that already has one will be read from the first one's cache and the
benchmark will measure the old weights while reporting the new label. Give it a
name of its own:

```bash
mv remote/model_last.pt remote/persian_multispeaker_40ep.pt
```

Then measure:

```bash
VOICECLONE__ENGINES__F5__CHECKPOINT=$PWD/remote/persian_multispeaker_40ep.pt \
  .venv/bin/python scripts/benchmark.py --label finetuned --device mps

.venv/bin/python scripts/benchmark.py --compare baseline finetuned
```

The vocabulary does not change — the run continues on the same 2609-entry
Persian tokenizer — so leave `VOICECLONE__ENGINES__F5__VOCAB` alone unless the
checkpoint came from somewhere else.

Both the override and the checkpoint conversion are verified: the environment
variable reaches `config.engines.f5.checkpoint`, and `slim_f5_checkpoint`
already prefers `ema_model_state_dict`, drops `initted`/`step`, and strips the
`ema_model.` prefix — which is exactly the layout the trainer writes.

The number that decides it is **held-out similarity**, against a baseline of
**0.520** and a ceiling of **0.689**. A move smaller than 0.0205 is not a move.
Watch CER at the same time: if similarity rises while CER climbs past the 7.8%
floor, the model has traded away the axis that was already won.

Sanity-check the first render before trusting the aggregate. A checkpoint that
loaded wrong does not usually crash — it produces confident, fluent noise.

---

## What each cut would cost

Given in case someone is tempted by a smaller machine. None of these is
recommended; they are here so the cost is visible rather than discovered later.

| Cut | What it costs |
|---|---|
| `batch_size_per_gpu` below 12000 | Changes optimisation dynamics against weights tuned at this batch. Compensating with `grad_accumulation_steps` keeps the effective batch but not the throughput. |
| Fewer epochs | Speaker generalisation is the slow part; pronunciation was learned first and is already good. Stopping early keeps what is already working and drops what is being fixed. |
| Fewer speakers / fewer hours | Directly attacks the hypothesis. Speaker diversity *is* the intervention. |
| Single-speaker data (e.g. Mana-TTS) | Produces a good single-voice reader, not a cloner. It would raise same-speaker scores and lower zero-shot ones — the opposite of what is being fixed. |
| Smaller architecture | Not available: the Persian vocabulary and weights are for `F5TTS_v1_Base`. |

**Do not run this on the laptop.** F5-TTS at this batch size needs more memory
than a 16 GB M1 Pro has to spare, and a run that swaps is a measurement of the
swap. A `PreToolUse` hook blocks it, which is the intended behaviour.
