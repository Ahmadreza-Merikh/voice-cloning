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
the intervention. It shortens training to roughly 3–5 h on 2×T4, which fits one
Kaggle session. The one way past the ceiling is more corpora — see the note at
the end of this section.

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

Then pack it:

```bash
cd data && tar -czf finetune-fa.tar.gz finetune-fa && cd ..
```

## 2. Upload

```bash
pip install kaggle                       # already in .venv
# ~/.kaggle/kaggle.json from https://www.kaggle.com/settings/account
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets init -p data/finetune-fa
# edit data/finetune-fa/dataset-metadata.json: set title and id (marsmanam/persian-f5-multispeaker)
kaggle datasets create -p data/finetune-fa --dir-mode tar
```

## 3. Train

Kaggle notebook, **GPU T4 ×2**, internet on, the prepared dataset attached.

```python
!pip -q install f5-tts

import os, json
os.environ["WANDB_DISABLED"] = "true"

# The Persian checkpoint to continue from, and its vocabulary. The vocabulary
# is not optional: it is a character tokenizer extended with the Persian
# alphabet, and the base F5 vocabulary has no Persian in it at all.
from huggingface_hub import hf_hub_download
ckpt  = hf_hub_download("Lumos675/F5_TTS_Persian", "model_last.pt")
vocab = hf_hub_download("Lumos675/F5_TTS_Persian", "vocab.txt")
```

Then, from a shell cell:

```bash
accelerate launch --mixed_precision=fp16 \
  -m f5_tts.train.finetune_cli \
  --exp_name F5TTS_v1_Base \
  --dataset_name persian_multispeaker \
  --pretrain "$CKPT" \
  --tokenizer custom --tokenizer_path "$VOCAB" \
  --learning_rate 7.5e-5 \
  --batch_size_per_gpu 12000 --batch_size_type frame \
  --max_samples 64 \
  --grad_accumulation_steps 1 --max_grad_norm 1 \
  --epochs 40 \
  --num_warmup_updates 2000 \
  --save_per_updates 5000 --last_per_updates 2000 \
  --logger none
```

**Why these values.** `learning_rate`, `batch_size_per_gpu`, `max_samples`,
`grad_accumulation_steps` and `max_grad_norm` are exactly what produced the
checkpoint being continued (its `setting.json`) — changing them mid-run changes
the optimisation dynamics against weights already adapted to them.
`batch_size_type frame` keeps the effective batch constant across clip lengths,
which matters when clips run 2.5–10 s. Warmup is lower than the original 6418
because this is a continuation, not a cold finetune. fp16 is what a T4 gives;
it is what the original used too.

Expect roughly 3–5 h on 2×T4 for the 12.5 h of audio at 40 epochs. Kaggle allows
30 GPU-hours a week and 12 h per session, so this fits in one — and
`--save_per_updates 5000` means a session that times out anyway resumes rather
than restarts.

## 4. Bring it back and measure it

```bash
# put model_last.pt and vocab.txt somewhere local, then:
VOICECLONE__ENGINES__F5__CHECKPOINT=/path/to/model_last.pt \
VOICECLONE__ENGINES__F5__VOCAB=/path/to/vocab.txt \
  .venv/bin/python scripts/benchmark.py --label finetuned --device mps

.venv/bin/python scripts/benchmark.py --compare baseline finetuned
```

The number that decides it is **held-out similarity**, against a baseline of
**0.520** and a ceiling of **0.689**. A move smaller than 0.0205 is not a move.
Watch CER at the same time: if similarity rises while CER climbs past the 7.8%
floor, the model has traded away the axis that was already won.

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
