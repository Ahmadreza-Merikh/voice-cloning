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
- **Objective quality feedback** — speaker-similarity scoring on every render, so
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

# 2. Model weights (~5.5 GB for the default engine)
.venv/bin/python scripts/download_models.py

# 3. Launch
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
| **F5-TTS** (`f5`) | community finetune | no (grapheme) | 5.4 GB | fast | Default. ~330 M params, comfortable on CPU/MPS. |
| **MOSS-TTS** (`moss`) | official (1 of 20 languages) | accepts IPA, but not usefully for Persian | 13.5 GB | slower | Apache-2.0. Best measured similarity. Pinned to transformers 5.0.x. |
| **Seed-VC** (`seedvc`) | language-agnostic | n/a | ~1 GB | medium | Acoustic voice conversion for speech-to-speech. |

**Which should I use?**

- Iterating on wording, or on a modest machine → **F5-TTS**.
- Final renders, or anything where timbre matters → **MOSS-TTS on Persian script**.
  Measured against the same reference and sentence: 0.819 speaker similarity versus
  F5's 0.730, and it got بهار، کردند، باغ، شکفت right where F5 got all four wrong.
  It needs its own environment (`transformers>=5.0,<5.1`) and ~13.5 GB of weights.
- You want the *source recording's* delivery preserved → **Seed-VC**.

Switch the default in `configs/default.yaml`:

```yaml
engines:
  tts: moss
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
4. Read the **speaker similarity** score under the player. Above 0.75 is a strong match.

### Command line

```bash
# text to speech
voiceclone tts "سلام، حال شما چطور است؟" -r reference.wav -o out.wav

# Render 4 takes and keep the best. Autoregressive engines vary more between
# seeds than any sampling knob moves them, so this is the cheapest quality
# win available. Costs 4x the time.
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
print(result.scores.speaker_similarity)        # e.g. 0.81
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

- Enable **Use Persian G2P** and pick **MOSS-TTS**. This is where Persian pronunciation
  is won or lost.
- Use the **Maximum quality** preset for final renders (64 ODE steps).
- Give F5-TTS an accurate **reference transcript**. It conditions on it, and a wrong
  transcript measurably degrades output. Auto-transcription is good; checking it is better.
- Fix the seed once you like a take, so you can reproduce it.

Reading the numbers:

| Speaker similarity | Meaning |
|---|---|
| ≥ 0.80 | excellent — hard to distinguish |
| 0.70 – 0.80 | strong — clearly the same speaker |
| 0.60 – 0.70 | fair — related but noticeably off |
| < 0.60 | the clone did not take; fix the reference |

### Measured results

From an actual run on an M1 Pro (16 GB), F5-TTS at the *High quality* preset
(48 ODE steps), 11.7 s reference, one Persian sentence:

| Metric | Result |
|---|---|
| Speaker similarity (ECAPA-TDNN) | **0.825** — excellent match |
| Round-trip WER (Whisper re-transcribes the output) | **0.0%** — every word came back correct |
| Synthesis, MPS | 56 s for 10.2 s of audio (~5.5× realtime) |
| Synthesis, CPU | 294 s for the same (~29× realtime) |

Enable the round-trip check yourself with `evaluation.transcribe_check: true`;
it re-runs ASR on the render and reports WER against what you asked for. The
first run of a session is much slower than the numbers above — it pays for the
checkpoint conversion, the vocoder download and a cold cache.

The G2P output for that sentence, showing the invisible vowels being recovered:

```
گلِ زیبای باغ ما امروز شکفت. چای و ژاله و جان.
→ gole zibɒje bɒqe mɒ emruz ʃekoft tʃɒj væ ʒɒle væ dʒɒn
        ^^      ^^                 ezafe vowels that the script never writes
```

---

## Configuration

Everything lives in [`configs/default.yaml`](configs/default.yaml), which is heavily
commented. Override it three ways, in increasing precedence:

```bash
# 1. a custom file
python app.py --config configs/my-setup.yaml

# 2. environment variables
VOICECLONE__RUNTIME__DEVICE=cpu VOICECLONE__ENGINES__TTS=moss python app.py

# 3. CLI flags
python app.py --device cpu --engine moss
```

Settings worth knowing:

```yaml
audio:
  reference:
    target_seconds: 10.0        # length of the voice prompt
    auto_select_window: true    # pick the best window of a long recording
  preprocess:
    denoise: auto               # auto | deepfilternet | noisereduce | none
    separate_vocals: false      # Demucs — for references taken from music/video

text:
  g2p:
    enabled: auto               # auto | true (require) | false
    repo_path: auto             # path to the Persian G2P project
    apply_to_engines: []        # empty: see the note at the top of this file

engines:
  f5:
    nfe_step: 48                # flow-matching steps; higher = better, slower
  moss:
    model_id: OpenMOSS-Team/MOSS-TTS-Local-Transformer   # 1.7B, ~3.5 GB at bf16
    # 24 GB+ GPU? Use the 8B flagship:
    # model_id: OpenMOSS-Team/MOSS-TTS
```

### Connecting the Persian G2P

The G2P bridge looks for the project automatically in `~/Desktop/G2P Research`,
`~/G2P Research` and a few other locations. Point it explicitly with either:

```yaml
text:
  g2p:
    repo_path: /path/to/G2P Research
```

```bash
export VOICECLONE_G2P_REPO="/path/to/G2P Research"
```

The **Diagnostics** tab reports which backend loaded, or exactly why it did not.

---

## Project structure

```
Voice Cloning/
├── app.py                      # Gradio entry point
├── configs/default.yaml        # every setting, commented
├── requirements.txt            # core runtime
├── requirements-optional.txt   # optional extras
├── scripts/
│   ├── setup_env.sh            # environment bootstrap
│   └── download_models.py      # weight downloader + checkpoint slimming
├── src/voiceclone/
│   ├── config.py               # typed, layered configuration
│   ├── device.py               # CUDA / MPS / CPU resolution
│   ├── models.py               # hf:// URI resolution and caching
│   ├── errors.py               # typed, actionable errors
│   ├── cli.py                  # command-line interface
│   ├── audio/
│   │   ├── io.py               # decode/encode, resampling
│   │   ├── analysis.py         # SNR, bandwidth, clipping → quality score
│   │   ├── denoise.py          # DeepFilterNet / noisereduce / Demucs
│   │   ├── vad.py              # Silero / energy VAD, silence handling
│   │   ├── segment.py          # best reference-window selection
│   │   ├── loudness.py         # EBU R128 normalisation, limiting
│   │   ├── preprocess.py       # the reference pipeline
│   │   └── postprocess.py      # crossfade joining, mastering
│   ├── text/
│   │   ├── normalize_fa.py     # Persian normalisation
│   │   ├── numbers_fa.py       # number/date/currency verbalisation
│   │   ├── g2p.py              # Homo-GE2PE bridge → IPA
│   │   ├── chunker.py          # sentence-aware chunking
│   │   └── pipeline.py         # the text pipeline
│   ├── engines/
│   │   ├── base.py             # engine interfaces + registry
│   │   ├── f5_tts.py           # F5-TTS
│   │   ├── moss_tts.py         # MOSS-TTS
│   │   ├── asr.py              # Whisper
│   │   └── vc_seedvc.py        # Seed-VC (out-of-process)
│   ├── pipelines/
│   │   ├── studio.py           # the orchestrator
│   │   ├── results.py          # result objects
│   │   └── voices.py           # voice library
│   ├── evaluation/similarity.py# speaker similarity, WER
│   └── ui/                     # Gradio app, theme, strings
└── tests/                      # 145 tests, no weights or network needed
```

Design rules the code follows:

- **Nothing heavy at import time.** Engines register themselves cheaply; models load
  on first use. `import voiceclone` stays fast.
- **Optional dependencies are always lazy** and always have a fallback or a clear
  message. There is no import that can break the app.
- **The UI holds no logic.** It marshals values into `Studio`, which the CLI and the
  Python API also use, so all three behave identically.
- **Errors are typed and carry a fix.** `DependencyMissingError` prints the pip command.

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -q        # 151 tests, seconds, no downloads
.venv/bin/python -m pytest tests/ -v        # verbose
.venv/bin/ruff check src tests              # lint
```

Tests cover Persian normalisation and number verbalisation, chunking and paragraph
handling, audio analysis/loudness/VAD/assembly, config layering, the engine registry
and the voice library. Anything needing model weights is marked `slow` and skipped
by default.

---

## Troubleshooting

**`No module named 'f5_tts'`**
`pip install f5-tts`, or `./scripts/setup_env.sh`.

**MOSS-TTS will not load**
It needs `transformers>=5.0`: `pip install -U transformers accelerate`. The 1.7B model
needs ~6.2 GB of download and ~3.5 GB of memory. Run
`python scripts/download_models.py --engine moss` first.

**Out of memory**
Use the F5-TTS engine, set `runtime.device: cpu`, lower `engines.f5.nfe_step`, or
shorten `text.chunk.max_chars`.

**Very slow on Apple Silicon**
Partly expected — MPS has no kernel for some operations and falls back to CPU. Use the
*Balanced* or *Fast preview* preset while iterating, *Maximum quality* for the final take.
The first run of a session is also much slower than later ones: it pays for the
checkpoint conversion, the vocoder download and a cold page cache.

**`Bus error: 10` / the process dies mid-synthesis on a Mac**
An upstream PyTorch bug, not this project. The Metal backend's `abs()` kernel
(`abs_kernel_mps`) intermittently faults, which kills the process outright — so it
cannot be caught and retried in Python. It is non-deterministic and more likely under
memory pressure. Observed on torch 2.13 / M1 Pro. The fix is to move off Metal:

```bash
python app.py --device cpu
voiceclone tts "..." -r ref.wav --device cpu
```

Or make it permanent in `configs/default.yaml`:

```yaml
runtime:
  device: cpu
```

Output is bit-for-bit the same idea, just slower. Closing other applications also
reduces how often it happens.

**The clone does not sound like the reference**
Press **Analyse reference** first. A score below 60 explains most failures. Also check
the reference is a single speaker with no background music, and try 10-12 seconds.

**Persian words are mispronounced**
Switch to the MOSS engine, which measured better on exactly this. Do *not* reach for
phonemes — handing MOSS Persian IPA makes it worse, not better (see the note at the
top). Try `--best-of 4` as well: seed variance moves pronunciation more than any
sampling knob does.

Do not write the ezafe explicitly either. `کتابِ من` is careful Persian, but GE2PE
emits a vowel for the kasra *and* applies its own ezafe rule, so the marked spelling
used to come out doubled (`ketaebe` for `ketabe`). The pipeline now strips a
word-final kasra before the G2P sees it; word-internal harakat are kept, since they
disambiguate homographs. MOSS also reads an explicit kasra as /i/, so `شهرِ` becomes
*shahri*.

**Numbers are read as digits**
`text.expand_numbers` must be `true`. Check with `voiceclone text "your text"`.

**`ffmpeg not found` / a file will not decode**
`brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Ubuntu).

**Seed-VC exits non-zero**
It needs its own environment: `./scripts/setup_env.sh --seedvc`. Or use `--mode cascade`,
which needs nothing extra.

---

## Responsible use

Voice cloning is powerful and easy to misuse.

- **Only clone voices you have explicit permission to clone.** Someone's voice is
  personal, and in many jurisdictions it is legally protected.
- **Label synthetic audio as synthetic.** Do not pass it off as a real recording.
- **Never** use it for impersonation, fraud, harassment, or to fabricate statements.

Cloning a voice without consent is unacceptable regardless of what the software allows.

---

## Acknowledgements

- [F5-TTS](https://github.com/SWivid/F5-TTS) — flow-matching TTS
- [F5-TTS Persian finetune](https://huggingface.co/Lumos675/F5_TTS_Persian)
- [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) — OpenMOSS / MOSI.AI (Apache-2.0)
- [Homo-GE2PE](https://huggingface.co/MahtaFetrat/Homo-GE2PE-Persian) — Persian G2P with
  homograph disambiguation ([paper](https://huggingface.co/papers/2505.12973))
- [Seed-VC](https://github.com/Plachtaa/seed-vc) — zero-shot voice conversion
- [Whisper](https://github.com/openai/whisper) / [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad), [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), [Demucs](https://github.com/facebookresearch/demucs)

Model weights carry their own licences — MOSS-TTS is Apache-2.0, Seed-VC and some
F5-TTS checkpoints are not commercially licensed. Check before shipping anything.

## Licence

MIT for this project's code. See `pyproject.toml`.
