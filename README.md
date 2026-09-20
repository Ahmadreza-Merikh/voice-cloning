# Persian Voice Cloning Studio

High-fidelity zero-shot voice cloning for Persian. Give it a few seconds of
someone's voice and either text or an audio file, and it produces speech in that
voice.

> **On phonemes.** This project was built around feeding IPA to a phoneme-capable
> engine. Measurement did not support it: MOSS-TTS accepts IPA, but its phoneme
> channel is trained on English (CMUdict) and Chinese pinyin, and Persian IPA is out
> of distribution — the same sentence renders in 7.3 s from Persian script and
> collapses to 1.3 s of nonsense from IPA (100% WER against 40%). `apply_to_engines`
> is therefore empty by default and every engine receives Persian script. The G2P
> still earns its keep for analysis, homograph inspection and the WER metric.

```
                                 ┌─────────────────────┐
   reference voice ──────────────▶  audio preprocessing │──┐
   (any format, any length)      │  analyse → clean →   │  │  clean 8-12s
                                 │  best-window select  │  │  voice prompt
                                 └─────────────────────┘  │
                                                          ▼
   text ──▶ normalise ──▶ numbers ──▶ G2P ──▶ chunk ──▶ ┌───────────┐
            (Persian)     (→ words)   (→IPA)            │  engine   │──▶ speech
                                                        └───────────┘
   audio ─▶ ASR ─▶ (same text pipeline) ────────────────────▲
        └─▶ or direct acoustic voice conversion ────────────┘
```

---

## Why this exists

Most TTS systems handle Persian badly, and the reason is orthographic rather than
acoustic. Persian script simply does not write down the information a synthesiser
needs:

| Problem | Example | What a naive TTS does |
|---|---|---|
| Short vowels are unwritten | `کرم` = *kerm* (worm), *karam* (generosity), *krem* (cream) | guesses |
| Ezafe is invisible | `کتاب من` is pronounced *ketâb**e** man* | usually omits the /e/ |
| Homographs | `گل` = *gol* (flower) or *gel* (mud) | guesses |
| Arabic/Persian glyphs mix | `ي` vs `ی`, `ك` vs `ک` | out-of-vocabulary tokens |
| Numbers | `۱۴۰۳` | reads digits, or nothing |

This project fixes each of them **before** the model sees the text:

1. **Normalisation** unifies Arabic letterforms, digits, ZWNJ and punctuation.
2. **Verbalisation** turns numbers, dates, times, currency and percentages into words.
3. **Grapheme-to-phoneme conversion** resolves short vowels, ezafe and homographs,
   emitting IPA — used for inspection and measurement, not currently for synthesis
   (see the note at the top).
4. The engine synthesises from normalised Persian script.

Step 3 uses a [Homo-GE2PE](https://huggingface.co/MahtaFetrat/Homo-GE2PE-Persian) based
Persian G2P system (3.98% PER, 76.9% homograph accuracy). It is optional — without it
the system still runs, reading raw Persian script, and tells you so in Diagnostics.

---

## Features

- **Text to speech** — clone a voice from a reference clip and make it say anything.
- **Speech to speech** — two strategies:
  - *re-speak* (ASR → Persian text pipeline → TTS): best pronunciation, cleans up a poor source
  - *voice conversion* (Seed-VC): keeps the original rhythm, emphasis and emotion
- **Serious audio preprocessing** — quality analysis with a 0-100 score, denoising,
  vocal isolation, VAD trimming, automatic best-window selection, loudness normalisation
- **Objective quality feedback** — calibrated similarity scoring on every render, so
  "did this work?" has a number rather than an opinion
- **Long-form stability** — sentence-aware chunking with equal-power crossfades, so
  minute ten sounds like minute one
- **Voice library** — save processed references with their transcripts and reuse them
- **Professional Gradio UI** — RTL Persian layout, quality presets, live diagnostics
- **CLI + Python API** — same pipeline as the UI, scriptable
- **Graceful degradation** — every optional dependency is lazily imported; missing
  ones disable one feature and are reported, never crash the app

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk | 8 GB | 20 GB |
| Compute | CPU | NVIDIA GPU (CUDA) |

Apple Silicon works and is used by default, but see the `Bus error: 10` note under
[Troubleshooting](#troubleshooting) — PyTorch's Metal backend has an intermittent
crash that `--device cpu` avoids.

`ffmpeg` is optional but strongly recommended — without it, mp3/m4a/video inputs may
not decode.

---

## Quick start

```bash
cd "Voice Cloning"

# 1. Environment (creates .venv, installs everything)
./scripts/setup_env.sh              # add --full for optional extras

# 2. Model weights for F5-TTS (~5.5 GB), the engine used where OmniVoice is not installed
.venv/bin/python scripts/download_models.py

# 3. OmniVoice, the default engine, in its own environment (3.3 GB of weights on first use)
uv venv --python python3.11 ~/.cache/persian-voice-cloning/omnivoice/.venv
VIRTUAL_ENV=~/.cache/persian-voice-cloning/omnivoice/.venv uv pip install omnivoice

# 4. Launch
.venv/bin/python app.py
```

Then open <http://127.0.0.1:7860>.

<details>
<summary>Manual install</summary>

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install -r requirements-optional.txt   # optional
python scripts/download_models.py
python app.py
```
</details>

Check what your machine can actually run at any time:

```bash
.venv/bin/python -m voiceclone.cli doctor
```

---

## Engines

| Engine | Persian | Phoneme input | Size | Speed | Notes |
|---|---|---|---|---|---|
| **F5-TTS** (`f5`) | community finetune | no (grapheme) | 5.4 GB | fast | Used where OmniVoice is not installed, and for speech-to-speech. ~330 M params, comfortable on CPU/MPS. Measured: held-out 0.520 stock, 0.546 continued on 2739 speakers with best-of-4; CER 2.2%, a figure whose small differences are within recogniser noise (FINDINGS §9). |
| **Chatterbox** (`chatterbox`) | community T3 finetune | no (grapheme) | ~3.5 GB | medium | Speaker path is language-agnostic — the part Persian data is *not* needed for. Own venv (torch 2.6). |
| **OmniVoice** (`omnivoice`) | official (1 of 600+) | no (grapheme) | 3.3 GB | medium | **Default** since 2026-09-19; on a 16 GB Mac it runs behind a memory guard (below). k2-fsa. Code Apache-2.0, **weights CC-BY-NC**. **Strongest identity transfer measured here: held-out 0.583**; over the best F5 configuration +0.037, 95% interval [+0.005, +0.069] across nine speakers - a direction, not a size. Own venv. |
| **MOSS-TTS** (`moss`) | official (1 of 31 languages) | accepts IPA, but not usefully for Persian | 6.1 GB | slower | Model Apache-2.0 (its codec's licence was not checked). 3.06B params, pinned to transformers 5.0.x (own environment). Measured: held-out 0.559, not shown to differ from the best F5 or from OmniVoice; CER 6.5% against OmniVoice's 2.8%, +3.7 points [+1.4, +6.0], both Kaggle renders re-scored on this Mac two minutes apart with the same recogniser settings and packages (what that rests on: FINDINGS §8). This is the older variant; v1.5 is the next row. |
| **MOSS-TTS v1.5** (`moss15`) | official, as "Persian (Farsi)" (1 of 31) | no | 9.1 GB + 8.5 GB codec | medium | Model, codec and code all Apache-2.0 at the pinned revisions. Runs out of process on its official stack (torch 2.9.1, transformers 5.0.0); on T4s it needs two cards, the codec on the second. Measured at the official recipe: held-out 0.507, no detectable difference from the F5 baseline in either direction (-0.013 [-0.053, +0.027]) and below OmniVoice (-0.077 [-0.122, -0.032]); CER +7.7 points over the baseline scored in the same session (FINDINGS §8). |
| **Fish S2 Pro** (`fish`) | third tier of 80+ | no | 11 GB | slow | Fish Audio Research License, non-commercial. Measured on a T4 in fp16: held-out 0.537, and CER 9.5% - against the F5 baseline re-scored in the same session, +6.4 points [+4.2, +8.6], worse for all nine speakers (CER is only compared inside one scoring environment, FINDINGS §9). Needs a CUDA card; own venv. |
| **Higgs TTS 3** (`higgs`) | claimed top tier of 102 | no | 10 GB | medium | Boson research/non-commercial licence. Measured on a T4 in fp16 through a transformers port whose weights hash identical to Boson's: held-out 0.531, CER 2.9%. Needs a CUDA card; own venv. |
| **Seed-VC** (`seedvc`) | language-agnostic | n/a | ~1 GB | medium | Acoustic voice conversion for speech-to-speech. |

**Which should I use?**

- Anything where you want the words right → **F5-TTS**. The recogniser reads
  its renders at 2.9% CER, against 7.8% for the corpus's own recordings - a
  measure of what the recogniser can read, not a check by ear. OmniVoice's CER
  is not detectably different (FINDINGS §8).
- Where **timbre** matters → **OmniVoice**, on the measurement in this
  repository: held-out 0.583 against 0.546 for the best F5 configuration,
  +0.037 [+0.005, +0.069] over the nine speakers and better in 22 of 27 cells -
  on a development set whose Common Voice speakers OmniVoice very probably
  heard in training (FINDINGS §9). Nothing was
  trained for it — picking a base that already had Persian in it beat nine
  GPU-hours of continuing F5 on 2739 speakers. F5 clones in context, so its
  identity transfer is bound up with how well someone trained it on Persian,
  and measurement puts it 0.170 under the human ceiling. `chatterbox` remains
  here as the other way at the same axis: its voice encoder and S3Gen carry
  identity and never see Persian text, so only T3 — the 0.5B text-to-token
  model — is language-specific.
- You want the *source recording's* delivery preserved → **Seed-VC**.

The earlier claim here that MOSS beat F5 on similarity (0.819 vs 0.730) was
measured against the **conditioning clip**, on one sentence with one seed. That
comparison rewards reproducing the prompt's room and microphone as much as its
voice, and it has been withdrawn rather than restated.

Switch the default back in `configs/default.yaml`:

```yaml
engines:
  tts: f5
```

> **Seed-VC needs its own environment.** It pins `transformers==4.46` and `numpy==1.26`,
> which cannot coexist with what F5-TTS requires, so it is driven out-of-process:
> `./scripts/setup_env.sh --seedvc`. The re-speak cascade needs no extra setup.

---

## Usage

### Web interface

```bash
python app.py                                  # http://127.0.0.1:7860
python app.py --port 8080 --share              # public link
python app.py --engine moss --device cpu
python app.py --language en                    # English UI chrome
```

Four tabs: **Text to Speech**, **Speech to Speech**, **Voice Library**, **Diagnostics**.

A good first run:

1. Drop a reference clip into **Text to Speech** and press **Analyse reference**.
   You get a quality score, a list of what is wrong with the recording, and an
   auto-filled transcript.
2. Fix anything flagged as an issue (noise, clipping, band-limiting) — the clone can
   never be better than the reference.
3. Type Persian text, pick **High quality**, and generate.
4. Read the badges under the player. **VS PROMPT** compares the render with the
   clip it cloned from and shares its channel; **SPEAKER** appears when you supply
   held-out audio and is the one that measures identity.

### Command line

```bash
# text to speech
voiceclone tts "سلام، حال شما چطور است؟" -r reference.wav -o out.wav

# Render 4 takes and keep the best. Autoregressive engines vary more between
# seeds than any sampling knob moves them, so this is the cheapest quality
# win available - but it is slow: measured at 6 m 48 s for one ~6 s sentence
# on an M1 Pro (MOSS/MPS), against ~2 m for a single take. Each take pays for
# round-trip ASR and speaker scoring on top of synthesis.
voiceclone tts "متن شما" -r reference.wav --best-of 4 -o out.wav

# long text from a file, with the MOSS engine
voiceclone tts @script.txt -r reference.wav -e moss -o narration.wav

# speech to speech: re-speak the content in a new voice
voiceclone s2s source.wav -r reference.wav -o out.wav

# speech to speech: keep the original delivery, change only the timbre
voiceclone s2s source.wav -r reference.wav --mode seedvc -o out.wav

# inspect the Persian text pipeline without synthesising anything
voiceclone text "قیمت ۲۵۰۰۰ تومان است" -e moss

# analyse a reference recording
voiceclone check reference.wav --save cleaned.wav

# what is installed?
voiceclone doctor

# voice library
voiceclone voices add "Narrator" -r reference.wav
voiceclone voices list
```

Running from a checkout without installing: `python -m voiceclone.cli ...`
(with `src` on `PYTHONPATH`), or just `pip install -e .`.

### Python API

```python
from voiceclone import Studio

studio = Studio()

result = studio.synthesize(
    "سلام، این یک آزمایش شبیه‌سازی صداست.",
    "reference.wav",
    engine="moss",
    speed=1.0,
    seed=42,                    # reproducible
)

print(result.output_path)                      # outputs/tts_moss_....wav
print(result.scores.reference_similarity)      # vs the prompt: e.g. 0.81
print(result.scores.held_out_similarity)       # vs other clips: the honest one
print(result.report())                         # full human-readable summary

# Speech to speech
converted = studio.convert("source.wav", "reference.wav", mode="seedvc")

# Reuse a processed reference across many renders (it is cached automatically)
reference = studio.prepare_reference("reference.wav")
print(reference.report_after.summary())
```

Lower-level pieces are usable on their own:

```python
from voiceclone.text import TextPipeline, normalize, expand_numbers
from voiceclone.audio import load_audio, analyze, preprocess_reference

normalize("قيمت ٢٥٠٠ تومان")     # 'قیمت دو هزار و پانصد تومان'
analyze(load_audio("ref.wav")).summary()
```

---

## Getting the best quality

The single biggest factor is the **reference recording**. In order of impact:

1. **8-12 seconds** of one speaker, continuous, no music or second voice.
2. **Quiet room.** Below ~20 dB SNR the clone inherits the noise. The analyser tells you.
3. **Full bandwidth.** A telephone or heavily compressed source caps how bright the
   clone can ever sound — no amount of processing recovers frequencies that were
   never recorded.
4. **No clipping.** Distortion is copied faithfully.
5. **Neutral, representative delivery.** Zero-shot cloning copies speaking style along
   with timbre; a shouted reference produces a shouty clone.

Then, in the app:

- Pick **OmniVoice** for Persian. In this repository's measurements it has the
  highest identity score (held-out 0.583; the development set may favour it,
  FINDINGS §9), and its CER on this Mac, 2.8%, is not detectably different from
  F5's 2.2-2.9% (-0.0 points [-1.1, +1.0] against the baseline). All of those
  were transcribed here with the same recogniser settings and packages - what
  that rests on is in FINDINGS §8 - and CER is the recogniser's reading, not a
  listening test. MOSS-TTS v1.0, which this line used to recommend, has the
  highest CER of the engines scored on this Mac (6.5%, +3.7 [+1.4, +6.0] over
  OmniVoice). Fish S2 Pro and MOSS v1.5 were scored on Kaggle; their CER is only
  compared with the baseline scored in the same session.
  Persian G2P does not help rendering - see the note on phonemes at the top.
  On a 16 GB Mac it is a tight fit. Its worker peaks around 2.9 GB, which leaves
  room only while the recogniser is not loaded as well: measured, a render with
  the transcript supplied peaked at 3.4 GB and never raised memory pressure,
  while the same render with the transcript left empty - so the recogniser had
  to load - reached warning pressure and was stopped. So OmniVoice runs behind a
  memory guard (`engines.omnivoice.memory_guard`), whose rule is *one at a
  time*, not *type it yourself*: the reference is transcribed first - by
  "Analyse", by saving a voice, or by the render itself - and the recogniser is
  released before OmniVoice loads. Measured on 2026-09-20
  (`scripts/measure_recogniser_release.py`,
  `eval/runs/evidence/recogniser-release/`): loading faster-whisper
  `large-v3-turbo` and transcribing takes 1.3 GB and releasing gives 1.3 GB
  back, twice in a row, the process returning to 0.9 GB each time; the 0.6 GB
  above its 0.3 GB start is the runtime the first transcription imports, not a
  leak, so a second cycle costs nothing further and there is no need to restart
  the app. What is still refused is best-of-N, which transcribes each take
  *while* the engine holds memory. Typing the transcript yourself is still
  worth it where you know the words: it skips the 6-9 s transcription and an
  exact transcript is what the engine conditions on. Keep browsers light either
  way. **The guard is not enough on its own:** on 2026-09-19, with
  other apps holding 5.1 GB in the memory compressor and no swap, a guarded
  render - transcript typed, no recogniser loaded - reached warning pressure
  while OmniVoice's weights were loading (available memory fell to 3.2 GB) and
  was stopped. Whether it fits depends on what else is open. Two things were
  added the same day. The app reads macOS's own memory-pressure level: it
  starts nothing at warning, will not start OmniVoice with less than 6 GB
  available (`engines.omnivoice.min_available_gb`, an estimate - warning came
  each time available memory fell to 3.05-3.37 GB), and stops the worker the
  moment warning arrives, with a message saying what to close. And the worker
  loads a verified float16 copy of the weights, 1.2 GB instead of 2.45 GB
  (`scripts/omnivoice_float16.py`: every tensor bit for bit the float32
  original cast to float16, which is what the worker does on load anyway).
  Measured after both: the load completed without pressure; with 4.4 GB
  available at the start, the render itself reached warning two seconds in and
  the app stopped it. With 6 GB free, whether a render completes has not been
  measured yet. Close heavy apps first.
- The **Maximum quality** preset (64 ODE steps) measured no better than the
  default: held-out 0.517 against 0.520, inside the encoder's noise. It costs time
  and buys nothing measurable here.
- F5-TTS conditions on a **reference transcript**. Whisper's own transcripts
  measured no worse than the corpus's ground truth (0.514 against 0.520, inside
  the noise), so auto-transcription is fine. A transcript that is actually wrong was
  never tested - correct one if you can see it is.
- Fix the seed once you like a take, so you can reproduce it.

Reading the numbers — **two scales, and the difference matters.**

By default a render is scored against *the clip it was conditioned on*. A
zero-shot clone reproduces that clip's channel — room, microphone, noise floor —
alongside the voice, and ECAPA encodes channel as well as identity. So that
number runs roughly **0.26 high** and is not a measure of identity transfer. Pass
`--held-out other1.wav other2.wav` (other recordings of the same speaker, not
used as the prompt) to get the figure that is.

Calibrated on a **verified** corpus of nine speakers — Common Voice 26 Persian
plus one studio voice — 45 clips, ECAPA-TDNN:

| Comparison | Score |
|---|---|
| different speakers | **0.132** (900 pairs, highest 0.508) |
| same speaker, different recordings | **0.689** (90 pairs, lowest 0.365) |

*Verified* is load-bearing. Common Voice `client_id` is an account, not a
person: households share them and one contributor may record on a laptop one
week and a phone in a car the next. Of 21 candidate speakers, **12 were
rejected** — one account's own two recordings scored 0.099 against each other,
lower than most different-speaker pairs, and two accounts scored 0.671 against
*each other*. A speaker is kept only when their worst pair against themselves
beats their best pair against anyone else.

| Held-out similarity | Meaning |
|---|---|
| ≥ 0.689 | as close as another recording of the same speaker |
| 0.508 – 0.689 | above every different-speaker pair measured |
| 0.132 – 0.508 | inside the range different speakers occupy — not distinguishable |
| < 0.132 | no resemblance |

Rebuild the corpus with `scripts/build_eval_corpus.py`, reproduce the
calibration with `scripts/benchmark.py --calibrate-only`.

### On WER, and why this project reports CER

**Word error rate does not work for Persian.** Compounds have no fixed
spacing — `کتابخانه`, `کتاب خانه` and `کتاب‌خانه` are the same word — and the
recogniser picks whichever its training data favoured. Word scoring counts that
choice as a deletion plus an insertion.

Measured here: Whisper transcribing **genuine human speech** from this corpus,
against the corpus's own transcripts, scores **37.4% WER**. One flawless render
scored 55.6% WER purely over where the spaces landed; the same render is 2.2%
CER. Every WER figure in older revisions of this document was partly measuring
typography.

`character_error_rate` is whitespace-insensitive and folds the Arabic
letterforms and harakat that ASR never emits. **On real human speech from this
corpus it scores 7.8%** — that is the floor a render is read against, not zero.

### Measured results

F5-TTS, 9 speakers × 3 sentences, one seed, M1 Pro on MPS. Held-out scoring
throughout — the render against recordings of that speaker it was never
conditioned on:

| Metric | Result | Read against |
|---|---|---|
| **Held-out similarity** | **0.520** ± 0.083 | ceiling 0.689, floor 0.132 |
| Position on the scale | **69%** stranger → same person | |
| **Round-trip CER** | **2.9%** | 7.8% on the corpus's real speech |
| Similarity to the reference clip | 0.654 | inflated +0.135 by shared channel |

**By the recogniser, the words are not where the quality is going.** It reads
the renders *more* accurately than the human recordings they are measured
against - a measure of what the recogniser can read, not a check of
pronunciation by ear, which nobody has done systematically. The gap is
identity: 0.520 sits 0.170 below the human ceiling, eight times the 0.0205
perturbation-sensitivity figure (what that figure is and is not: FINDINGS §8),
and inside the band where different speakers still occur.

Things that did **not** move it (each run changes exactly one thing, same
corpus, sentences, seed and device):

| Change | Held-out | vs baseline |
|---|---|---|
| baseline | 0.520 | — |
| `trim_silence: false` | 0.524 | +0.004 |
| `denoise: none` | 0.510 | −0.010 |
| `cfg_strength: 2.5` | 0.518 | −0.002 |
| `cfg_strength: 1.5` | 0.521 | +0.002 |
| `nfe_step: 64` | 0.517 | −0.003 |
| ASR reference transcripts | 0.514 | −0.005 |

**Six configurations span 0.014 — narrower than the 0.0205
perturbation-sensitivity figure.** Nothing on the inference side moves identity transfer. Reference
preprocessing was not the problem either: it moves a clip's speaker embedding
by only 0.037 (0.963 self-similarity), and `denoise: auto` already resolves to
a no-op on clean audio.

The one inference-side thing that **did**:

| Change | Held-out | CER | Cost |
|---|---|---|---|
| `evaluation.candidates: 4` | **0.533** | 2.8% | 4× |

Paired cell by cell against the baseline, best-of-4 changed the chosen take in
10 of 27 cases and improved 9 of them (mean +0.046, best +0.118).

### What closed part of the gap

Continuing the checkpoint on **2739 speakers, 12.47 h** of Common Voice Persian
— selected for diversity, evaluation speakers excluded, 78 epochs on 2×T4:

| Configuration | Held-out | CER | Paired vs baseline |
|---|---|---|---|
| baseline | 0.520 | 2.9% | — |
| best-of-4 | 0.533 | 2.8% | +0.0135, 2.2 SE |
| finetuned | 0.539 | 2.7% | +0.0199, 4.4 SE |
| **finetuned + best-of-4** | **0.546** | **2.2%** | **+0.0266, 4.6 SE** |

The full stack is the first configuration to clear the 0.0205 reference line
(fixed before the benchmark existed, commit `dee1ace`; per-speaker intervals in
FINDINGS §8 put this difference at [+0.015, +0.038], clear of 0 but overlapping
that line), and CER does not pay for it. The CER column moves in the
right direction, but re-scoring identical audio shifts it on its own, so these
tenths sit at the recogniser's resolution rather than above it — see the caveat
in FINDINGS §9. Paired over the same 27 cells it changed every one, improving
23 and costing 4.

**The two levers overlap.** Independently they total +0.034; together they
deliver +0.027. Best-of-4's own contribution on top of the finetune is +0.0067
at 1.7 standard errors — not resolved — and it changes 7 of 27 cells where on
the baseline it changed 10. The finetune appears to have absorbed some of what
seed selection was recovering.

**What is left is most of it.** 0.546 against a 0.689 ceiling is still 0.143
short, seven times the encoder's noise, and the gap is not spread evenly: cv09
lands within 0.071 of its own ceiling while cv11 and mana remain more than 0.19
away.

### Changing the base closed more than training it did

[OmniVoice](https://github.com/k2-fsa/OmniVoice) (k2-fsa; code Apache-2.0, weights CC-BY-NC) has
Persian among its training languages. Run at its published inference settings,
one take per cell, on the same speakers, sentences, seed and reference clips,
with every run below re-scored on one machine so the comparison is paired:

| Run | Held-out | vs OmniVoice, paired |
|---|---|---|
| F5 baseline | 0.520 | +0.0637, 5.4 SE, better in 23/27 |
| F5 continued | 0.540 | +0.0438, 3.7 SE, better in 22/27 |
| F5 continued + best-of-4 | 0.546 | +0.0371, 3.4 SE, better in 22/27 |
| **OmniVoice, single take** | **0.583** | — |

It wins seven speakers of nine, and reaches 84.8% of each speaker's own ceiling
against 79.2% for the best F5 configuration and 75.2% for the baseline — so the
gain survives normalising away how self-consistent each speaker's recordings
happen to be. Nothing was trained for it. The continuation cost nine GPU-hours and bought
+0.0266; changing the base bought +0.0371 for free. On this corpus the choice
of base mattered more than tuning it — and the better base was found only
because the landscape was checked against live sources instead of recalled.

**This is not a state-of-the-art claim.** OmniVoice is someone else's model,
the scale is local to this corpus and comparable to no published number,
nine development speakers are too few to rank close engines, eight of them
were very probably in OmniVoice's training data (FINDINGS §9), and no human has
scored any of it. Of the other candidates measured, MOSS-TTS v1.0 was not shown
to differ from it on identity and had a clearly higher CER - a recogniser
measurement, nobody has listened; Fish S2 Pro, Higgs TTS 3 and MOSS-TTS v1.5 at
its official recipe scored lower on identity, with speaker-level intervals that
exclude zero (FINDINGS §8). What this repository contributes is the instrument
that could tell the difference.

See [`FINDINGS.md`](FINDINGS.md) for the full investigation and
[`docs/REMOTE-TRAINING.md`](docs/REMOTE-TRAINING.md) for the recipe.
