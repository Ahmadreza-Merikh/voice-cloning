# Where the quality is actually going

Investigation of 3 August 2026. Supersedes `FINDINGS-similarity-metric.md`
(every task in it is closed below) and `REVIEW-best-of-n.md`.

Everything here is reproducible:

```bash
.venv/bin/python scripts/build_eval_corpus.py     # assemble + verify the corpus
.venv/bin/python scripts/benchmark.py --label baseline
.venv/bin/python scripts/benchmark.py --compare baseline trim-off cfg25
```

---

## 1. The finding

**Persian pronunciation is not the problem. Identity transfer is.**

F5-TTS, nine verified speakers, three sentences, one seed, scored against
recordings it was never conditioned on:

| | measured | read against |
|---|---|---|
| Held-out speaker similarity | **0.520** ± 0.083 | ceiling **0.689**, floor **0.132** |
| Position on that scale | **69%** stranger → same person | |
| Round-trip CER | **2.9%** | **7.8%** on the corpus's own human speech |

The renders come back *more* intelligible than the human recordings they are
measured against. Meanwhile similarity sits 0.170 below the human ceiling —
eight times the encoder's 0.0205 noise — and inside the band where genuinely
different speakers still occur.

Any further effort spent on Persian text handling is spent on the axis that is
already won.

---

## 2. Three measurement errors had to be fixed first

None of them would have announced itself. Each produces a confident number.

### 2.1 WER was scoring typography, not speech

Persian compounds have no fixed spacing: `کتابخانه`, `کتاب خانه` and
`کتاب‌خانه` are one word, and the recogniser picks whichever its training data
favoured. Word scoring counts that choice as a deletion plus an insertion.

A flawless render came back at **55.6% WER** on that alone — every sound in the
right place. The same render is **2.2% CER**.

The scale of the problem: **Whisper transcribing genuine human speech from this
corpus, against the corpus's own transcripts, scores 37.4% WER.** Every WER
figure this project previously published was partly measuring where Whisper put
its spaces.

`character_error_rate` is whitespace-insensitive and folds the Arabic
letterforms and harakat that ASR never emits. On real human speech it scores
**7.8%** — that, not zero, is what a perfect render is read against.

### 2.2 The calibration corpus could not be checked

The floor and ceiling this project quoted (0.288 / 0.647) came from audio that
no longer existed on disk, measured on a single speaker. `build_eval_corpus.py`
rebuilds a versioned one from Common Voice 26 Persian — which has real speaker
labels — plus one studio voice.

### 2.3 Speaker labels are not ground truth

Common Voice `client_id` is an *account*, not a person. Of 21 candidate
speakers, **12 failed verification**:

- one account's own two recordings scored **0.099** against each other — lower
  than most different-speaker pairs
- two different accounts scored **0.671** against *each other* — as high as a
  typical same-speaker pair

A speaker is kept only when their worst pair against themselves beats their best
pair against anyone else. Keeping the rest would have put a false ceiling on the
floor and a false floor under the ceiling.

The verified scale is much wider than the old one: **0.132** (900 cross-speaker
pairs, highest 0.508) to **0.689** (90 same-speaker pairs, lowest 0.365).

---

## 3. What was tried, and what it bought

One change per run. Same corpus, sentences, seed and device throughout; only
the named flag differs.

| Run | Held-out | vs baseline | CER | Verdict |
|---|---|---|---|---|
| baseline | 0.520 | — | 2.9% | |
| `trim_silence: false` | 0.524 | +0.004 | 3.4% | inside noise |
| `denoise: none` | 0.510 | −0.010 | 7.0% | inside noise |
| `cfg_strength: 2.5` | 0.518 | −0.002 | 3.2% | inside noise |
| `cfg_strength: 1.5` | 0.521 | +0.001 | — | inside noise |

**Every F5 knob is inert.** Nothing available on the inference side moves
identity transfer, which is what one would expect if the limit is the model
rather than how it is being driven.

Reference preprocessing was separately exonerated by direct measurement: the
pipeline's own preprocessing moves a clip's speaker embedding by only 0.037
(0.963 self-similarity), and `denoise: auto` already resolves to a no-op on
audio this clean. Render pacing is also normal — 12.3–13.1 characters per
second, within the range for Persian read speech, and uncorrelated with
similarity.

---

## 4. The architecture hypothesis, and how far it got

F5 clones *in context*: one model reads the reference audio, the reference
transcript and the target text together, so identity transfer is entangled with
how well someone trained it on Persian — here, a community finetune whose model
card is a licence line and nothing else.

Chatterbox separates them. A voice encoder and S3Gen carry identity and never
see Persian text; only T3, a 0.5B text-to-token model, is language-specific,
and that is the single file `Thomcles/Chatterbox-TTS-Persian-Farsi` replaces.
The component this project is weakest on would be the component Persian data is
*not* needed for.

**Status: not yet demonstrated.** With the base T3 — which cannot read Persian —
the engine produced usable audio for two of four speakers and garbage for the
other two. On the two that worked it scored 0.582 and 0.567 against F5's 0.563
and 0.556: better, but inside the noise, so not yet a win. Where it collapsed
(86–91% CER) similarity collapsed with it, which is expected — a speaker
embedding of non-speech means nothing.

The Persian T3 is gated and needs an account that has accepted its licence.

### A trap worth recording

Chatterbox ships two stacks. The English tokenizer holds 704 tokens and, checked
directly, **not one character of Arabic script** — it cannot represent Persian
at all. The multilingual one holds 2352 and covers every Persian letter.

The Persian finetune distributes a bare `t3_fa.safetensors` with no note of
which base it belongs to, and **loading it onto the wrong one does not raise**:
the text embedding is simply the wrong height, `load_state_dict(strict=False)`
drops the weights, and the output is fluent-sounding gibberish indistinguishable
from a bad model. The worker now matches the checkpoint's text-embedding height
against each stack and refuses to guess.

---

## 5. Closing the previous handoff

Every task in `FINDINGS-similarity-metric.md`:

| | |
|---|---|
| T1 name the comparison | done — `reference_similarity` vs `held_out_similarity` |
| T2 stop uncalibrated verdicts | done — bands re-derived from the verified corpus |
| T3 held-out scoring | done — it is the primary metric everywhere |
| T4 reconcile the two F5 figures | done — both withdrawn, neither was a cloning score |
| T5 reference-transcript gate | measured, see below |
| T6 no floor exists | done — 0.132 over 900 cross-speaker pairs |

T5's specific complaint stands: `transcribe_reference` warned that a clip began
mid-sentence but said nothing about `عوارز` for `عوارض`, a wrong word delivered
confidently. The benchmark feeds F5 ground-truth transcripts from the corpus;
`--ref-transcript asr` measures what production loses by feeding it Whisper's
guess instead.

---

## 6. Caveats

- **One seed per configuration.** Autoregressive and flow-matching engines vary
  between seeds; the differences reported as "inside noise" would need several
  seeds each to be called *equal* rather than *not shown to differ*.
- **Three sentences.** Enough to average out one unlucky text, not enough to
  characterise an engine across registers.
- **Common Voice is crowdsourced.** Reference clips are 8.7–9.9 s of variable
  channel quality. A studio corpus would raise every absolute number; it should
  not change the ordering, and `mana` is included as one studio point of
  comparison.
- **The ceiling is a property of this corpus.** 0.689 is what these speakers
  score against themselves across their own recordings. A corpus recorded in one
  session would sit higher and make every clone look worse by comparison.
