# Adapting OmniVoice to Persian: the full recipe, as commands

This is the runbook for the one training run the roadmap plans (SOTA_ROADMAP
§8): OmniVoice's own adaptation recipe, unchanged, on the two CC0 corpora split
in §3.2. It is written to run on any machine with two CUDA GPUs and enough
memory, in the order given. Nothing here reduces the recipe to fit a smaller
card; where a card is too small, the answer is a bigger card.

## What is pinned

| what | where | pin |
|---|---|---|
| code | `k2-fsa/OmniVoice` | `08be0b4ccbac3e13e374e86fbfead4b4cac343e2` |
| runtime | its `uv.lock` | torch 2.8.0+cu128, torchaudio 2.8.0+cu128, transformers 5.3.0, accelerate 1.13.0, webdataset 1.0.2, safetensors 0.7.0, tokenizers 0.22.2, huggingface-hub 1.7.2 |
| base weights | `k2-fsa/OmniVoice` | `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` |
| audio tokenizer | `eustlb/higgs-audio-v2-tokenizer` | `528e871c2a26c4f0f7773b9754e2e1acae20899d`, `model.safetensors` sha256 `fe7c5e87...` (the same file OmniVoice bundles) |
| config | `examples/config/train_config_finetune.json` (Ampere or newer) or `train_config_finetune_sdpa.json` (older GPUs) | git blob `f80f573b` / `4b0a543e` |
| data | `pymmdrza/Common-Voice-Speech-26.0-Persian-Clean` | `142e9bb40b56d9491faa0ec0987ee8d408bd9c05` |
| | `shekar-ai/neyshekar` | `30696ada746a66bb3047c984744b5ecfd60fe4e9` |
| split | `eval/manifests/adaptation_splits_v1.jsonl.gz` | content sha256 `85be52bd...` (in its `.summary.json`) |

The recipe itself (from the config, not from `examples/README.md`, whose 5e-5
is stale): learning rate 1e-5, AdamW with weight decay 0.01, cosine schedule
with 1% warmup, 5,000 steps, 8,192 tokens per GPU per step on two GPUs, no
gradient accumulation, gradient norm clipped at 1.0, bf16 mixed precision,
language tag in 80% of prompts, a checkpoint and an evaluation loss every 500
steps.

## Hardware

**[measured]** A Kaggle T4 (15 GB, Turing) cannot hold the recipe as
published: both official configurations run out of memory in the first step,
SDPA already in the first forward pass with 14 GiB allocated
(`eval/runs/persian-voice-omnivoice-probe-v1.json`). With gradient
checkpointing and DDP bucket views it trains: 13.98 s per step, 12.3 GiB
reserved of 14.56, and on one GPU the checkpointed path gave the same first
loss and first-step gradients as the official one
(`eval/runs/persian-voice-omnivoice-probe-v2.json`). So a T4 pair runs the
full recipe in about 19.4 hours of training plus about 3.6 of tokenising.

**[source]** The authors trained on H800s (80 GB) with `flex_attention`, which
the repository says needs Ampere or newer. A pair of bf16-capable cards with
at least 40 GB each (A100 40/80 GB, L40S, H100) runs the default configuration
as published; that is the recommendation when a T4 pair cannot. Smaller cards
(24 GB L4, for example) are not known to fit and would first need the probe.

## 1. Environment

```bash
git clone https://github.com/k2-fsa/OmniVoice && cd OmniVoice
git checkout 08be0b4ccbac3e13e374e86fbfead4b4cac343e2
uv sync                                   # installs exactly uv.lock
```

## 2. Data

From this repository, on the same machine (the audio is about 19 GB of
parquet to download and 23 GB of WAV to write; nothing is decoded or
resampled here - the tokenizer resamples to 24 kHz itself):

```bash
huggingface-cli download pymmdrza/Common-Voice-Speech-26.0-Persian-Clean --repo-type dataset \
  --revision 142e9bb40b56d9491faa0ec0987ee8d408bd9c05 --include "data/*" --local-dir data/cv26
huggingface-cli download shekar-ai/neyshekar --repo-type dataset \
  --revision 30696ada746a66bb3047c984744b5ecfd60fe4e9 \
  --include "data/train-*" "data/validation-*" --local-dir data/neyshekar
```

Neyshekar's test shards are never downloaded: its test speakers are this
project's primary test set, sealed. The Common Voice test clips sit inside the
same shards as training clips, so all of that source is downloaded, and
`prepare_adaptation_data.py` refuses to write a test clip.

```bash
for split in train valid; do
  python scripts/prepare_adaptation_data.py \
    --splits eval/manifests/adaptation_splits_v1.jsonl.gz --split $split \
    --cv-dir data/cv26 --neyshekar-dir data/neyshekar \
    --out data/adapt/$split --summary data/adapt/$split.summary.json
done
```

Expected: train 33,148 + 54,351 clips (127.85 h), valid 365 + 1,886 (3.68 h),
nothing left out. Without `--limit` the script stops if any clip of the split
is missing.

## 3. Tokenise (the official stage 0)

```bash
for split in train valid; do
  python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl data/adapt/$split/$split.jsonl \
    --tar_output_pattern data/tokens/$split/audios/shard-%06d.tar \
    --jsonl_output_pattern data/tokens/$split/txts/shard-%06d.jsonl \
    --tokenizer_path <local snapshot of eustlb/higgs-audio-v2-tokenizer @ 528e871c> \
    --nj_per_gpu 3 --shuffle True
done
```

**[measured]** On a T4 pair: 34x real time on a 2.1 h slice, start-up
included - about 4 hours for the training split at that rate.

## 4. Train (the official stage 1)

`data_config.json`, in the official format:

```json
{"train": [{"manifest_path": ["data/tokens/train/data.lst"]}],
 "dev":   [{"manifest_path": ["data/tokens/valid/data.lst"]}]}
```

The official launch line, with this repository's wrapper in place of
`-m omnivoice.cli.train` (it builds the same trainer with the same functions;
it adds only session budgets, float16 exports and a record - see its
docstring):

```bash
accelerate launch --gpu_ids 0,1 --num_processes 2 scripts/omnivoice_adapt.py \
  --train_config examples/config/train_config_finetune.json \
  --data_config data_config.json --output_dir exp/omnivoice_fa \
  --set init_from_checkpoint=<local snapshot of k2-fsa/OmniVoice @ c5fdb5cc> \
  --export-every 1000 --audio-tokenizer <that snapshot>/audio_tokenizer \
  --record exp/omnivoice_fa/record.json
```

On a GPU older than Ampere use `train_config_finetune_sdpa.json`. On a card
that cannot hold the recipe, `--memory checkpointing+bucket-view` (and, if
that is still short, `+single-tensor`) changes memory only - but only after
the probe has shown the losses and first-step gradients unchanged on that
card. To run in sessions, add `--session-seconds <budget>` and
`--set keep_last_n_checkpoints=1`, and resume with
`--set resume_from_checkpoint=exp/omnivoice_fa/checkpoint-<step>`.

## 5. Choose, then test

The reading was fixed before any run (SOTA_ROADMAP §8): the float16 exports
at 1,000-5,000 steps are scored on the development set with
`scripts/benchmark.py --engine omnivoice` and
`VOICECLONE__ENGINES__OMNIVOICE__MODEL_ID=<export>`, beside the published
model scored in the same session; the export with the highest held-out
similarity whose CER interval against the published model does not lie above
zero is the candidate. Only then are the candidate and the published model
scored, once, on the sealed test sets, and only that comparison can say
whether the adaptation helped.
