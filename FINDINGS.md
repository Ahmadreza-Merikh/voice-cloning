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
| `cfg_strength: 1.5` | 0.521 | +0.002 | 3.3% | inside noise |
| `nfe_step: 64` | 0.517 | −0.003 | 3.1% | inside noise |
| ASR reference transcripts | 0.514 | −0.005 | 3.1% | inside noise |

**Every F5 knob is inert.** Six configurations span 0.510 to 0.524 — a range of
0.014, smaller than the encoder's own 0.0205 noise. Nothing available on the
inference side moves identity transfer, which is what one would expect if the
limit is the model rather than how it is being driven. Doubling the ODE budget
from 32 to 64 does not help either, so the flow-matching solver is not the
bottleneck.

### The one thing that worked: best-of-4

| Run | Held-out | CER | Cost |
|---|---|---|---|
| baseline (one take) | 0.520 | 2.9% | 15 min |
| best-of-4 | **0.533** | 2.8% | 60 min |

The aggregate difference is +0.014, *below* the 0.0205 noise figure — but that
figure bounds a single render comparison, and these two runs are **paired**: same
speakers, same sentences, same seeds. Best-of-4 renders seed 42 as one of its
four takes, so whenever seed 42 was already the best take the result is an exact
tie, and those ties dilute the mean rather than arguing against the effect.

Cell by cell over 27 pairs:

| | |
|---|---|
| unchanged | 17 |
| better | **9** — mean +0.046, best +0.118 |
| worse | 1 — −0.046 |

**It changed the outcome in 10 cells and got 9 of them right** (sign test on the
changed cells, p ≈ 0.02; mean difference 2.2 standard errors from zero). CER did
not suffer — 2.9% to 2.8%.

So the selector works, and it is the only intervention measured here that does.
It is also the most expensive: **4× the wall-clock for +0.014**, which buys back
about 8% of the 0.170 gap. Worth switching on for a final render, not for
iteration — which is what `evaluation.candidates` already defaults to.

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

**Status: the hypothesis is not supported by either proxy available so far.**

*Probe one — base T3, Persian text.* The engine produced usable audio for two
of four speakers and garbage for the other two. On the two that worked it scored
0.582 and 0.567 against F5's 0.563 and 0.556: better, but inside the noise. Where
it collapsed (86–91% CER) similarity collapsed with it, which is expected — a
speaker embedding of non-speech means nothing.

*Probe two — base T3, English text.* This isolates the speaker path from the
inability to read Persian: give the model text it can actually pronounce, clone
a Persian speaker, and score against that speaker's Persian held-out clips.
Speaker embeddings are largely text-independent, so this should approximate what
the architecture can carry for these voices.

| | held-out |
|---|---|
| Chatterbox, English text, base T3 | **0.389** (0.438 excluding one collapsed render) |
| F5, Persian text | **0.520** |

That is 0.131 *worse* than F5, six times the encoder's noise. The architecture
argument predicted the opposite.

**Caveat, and it is a real one.** The English probe changes engine and language
together. Cross-lingual cloning is known to be harder — the render carries an
English accent the reference never had, and some of that distance is language
rather than speaker path. So this is evidence against the hypothesis, not proof.

The clean test is Chatterbox with the Persian T3 on Persian text, which needs
the gated checkpoint. Until then the honest position is: **the architecture
argument is unconfirmed, and the two proxies that could be run both point the
wrong way.** F5 remains the better measured engine.

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

T5 is now **measured, and the worry does not survive it.** The concern was that
F5 conditions on a transcript of the reference and that a wrong one degrades
output — `transcribe_reference` warned a clip began mid-sentence but said nothing
about `عوارز` for `عوارض`, a wrong word delivered confidently.

Feeding F5 Whisper's transcripts instead of the corpus's ground truth, over all
nine speakers:

| Reference transcript | Held-out | CER |
|---|---|---|
| corpus ground truth | 0.520 | 2.9% |
| Whisper | 0.514 | 3.1% |

Both differences are inside the noise. The transcripts are good enough that the
production path loses nothing measurable by using them. The gate could still be
tightened, but not on quality grounds — there is no quality here to recover.

---

---

## 6. The finetune: the first thing to move identity transfer

Continuing the Persian checkpoint on **2739 speakers, 12.47 h** of Common Voice
Persian — selected for speaker diversity, evaluation speakers excluded — is the
first intervention measured here that moves the axis the investigation named.

| | baseline | finetuned |
|---|---|---|
| Held-out similarity | 0.520 | **0.539** |
| Reference similarity | 0.654 | 0.680 |
| Round-trip CER | 2.9% | **2.7%** |
| Position on the 0.132–0.689 scale | 69.2% | **73.0%** |

Paired over the same 27 cells — same speakers, sentences and seed:

| | |
|---|---|
| mean difference | **+0.0199**, 4.4 standard errors from zero |
| better | **23** — mean +0.027, best +0.069 |
| worse | 4 — mean −0.021, worst −0.030 |
| unchanged | 0 |

**On the pre-registered threshold: the aggregate misses it, by 0.0006.** The
rule set in advance was a move larger than 0.0205, and +0.0199 is not that. What
that figure bounds, though, is a *single* render comparison; the paired mean over
27 cells has a standard error of 0.0045, and the effect sits 4.4 of them from
zero. Twenty-three of twenty-seven cells improved. The distinction between an
aggregate threshold and a paired test was written into the tool before this run,
for best-of-4, so it is not a goalpost moved to fit a result.

The honest reading is that **the effect is real, and it is small**: 0.019 of the
0.170 gap, about 11%. Held-out similarity sits 0.150 under the human ceiling —
seven times the encoder's noise, still inside the band where genuinely different
speakers occur.

CER did not pay for it: 2.9% to 2.7%, both far under the 7.8% recogniser floor.
The model did not trade away the axis that was already won.

### Per speaker

| speaker | held-out | ceiling | gap |
|---|---|---|---|
| cv09 | 0.691 | 0.781 | −0.090 |
| cv02 | 0.573 | 0.679 | −0.106 |
| cv04 | 0.573 | 0.702 | −0.129 |
| cv03 | 0.546 | 0.661 | −0.115 |
| cv11 | 0.539 | 0.757 | −0.218 |
| mana | 0.529 | 0.739 | −0.209 |
| cv07 | 0.493 | 0.680 | −0.187 |
| cv08 | 0.465 | 0.607 | −0.142 |
| cv10 | 0.446 | 0.595 | −0.149 |

The gap is not uniform. cv09 lands within 0.090 of its own ceiling; cv11 and
mana remain more than 0.2 short. Whatever is left is not evenly distributed
across voices, which is a lead rather than a conclusion.

### What was actually trained

- 78 of 80 epochs. The run ended at update 15,000 of 15,440 when Kaggle's
  20 GB working disk filled while saving — `--keep_last_n_checkpoints 2`
  deletes *after* writing, and the write is what failed. `model_15000.pt` was
  truncated to 2.05 GB by that; `model_last.pt`, update 14,000, is intact and is
  what was measured.
- Loss 1.61 → 0.52.
- Effective batch held at 12000 frames per GPU via 3000 × 4 accumulation, since
  12000 in one pass does not fit a T4. Deviation from the original recipe is a
  reweighting of the loss mean across unequal micro-batches, nothing else.
- Probe tensors moved 1.42% and 1.57% relative; the text embedding only 0.046%,
  which is what a continuation on an already-Persian vocabulary should look like.

### Not yet tried

Best-of-4 scored 0.533 on its own and is independent of the weights, so the two
should compose. Neither has been measured together.

## 7. Caveats

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
