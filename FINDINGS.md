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

**By the recogniser's reading, the Persian comes out right. The gap is identity transfer.**

F5-TTS, nine verified speakers, three sentences, one seed, scored against
recordings it was never conditioned on:

| | measured | read against |
|---|---|---|
| Held-out speaker similarity | **0.520** ± 0.083 | ceiling **0.689**, floor **0.132** |
| Position on that scale | **69%** stranger → same person | |
| Round-trip CER | **2.9%** | **7.8%** on the corpus's own human speech |

The recogniser transcribes the renders *more* accurately than the human
recordings they are measured against. That is a measurement of what the
recogniser can read, not a listening judgement: nobody has checked the
pronunciation by ear. Meanwhile similarity sits 0.170 below the human ceiling —
eight times the 0.0205 perturbation-sensitivity figure (§8) — and inside the band where genuinely
different speakers still occur.

On these two measures, further effort on Persian text handling goes to the
smaller of the two gaps.

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
0.014, smaller than the 0.0205 perturbation-sensitivity figure (§8). Nothing available on the
inference side moves identity transfer, which is what one would expect if the
limit is the model rather than how it is being driven. Doubling the ODE budget
from 32 to 64 does not help either, so the flow-matching solver is not the
bottleneck.

### The one thing that worked: best-of-4

| Run | Held-out | CER | Cost |
|---|---|---|---|
| baseline (one take) | 0.520 | 2.9% | 15 min |
| best-of-4 | **0.533** | 2.8% | 60 min |

The aggregate difference is +0.014, *below* the 0.0205 perturbation-sensitivity figure — but that
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
not suffer, but 2.9% to 2.8% is smaller than the recogniser's own run-to-run
wobble (§9): read it as "no cost", not as an improvement.

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

**On the threshold fixed before the benchmark existed (commit `dee1ace`, 2026-07-30): the aggregate misses it, by 0.0006.** The
rule was a move larger than 0.0205, and +0.0199 is not that. What
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

### Composed with best-of-4

They do stack, and they overlap.

| Configuration | Held-out | CER | Paired vs baseline |
|---|---|---|---|
| baseline | 0.520 | 2.9% | — |
| best-of-4 | 0.533 | 2.8% | +0.0135, 2.2 SE |
| finetuned | 0.539 | 2.7% | +0.0199, 4.4 SE |
| **finetuned + best-of-4** | **0.546** | **2.2%** | **+0.0266, 4.6 SE** |

The full stack is the first configuration to clear the 0.0205
threshold outright, and CER does not pay for it. The CER column moves the way a
CER-ranked selector on better weights should, but that reading is weaker than it
looks: re-scoring identical audio moves the aggregate by itself (§9), and one
replication is not enough to put an error bar on a 0.7-point difference.
Paired against the baseline it changed all 27 cells, improving 23.

But the levers are **sub-additive**. Independently they sum to +0.034; together
they deliver +0.027. Best-of-4's own contribution on top of the finetune is
+0.0067 at 1.7 standard errors - not resolved - and it changes the chosen take
in 7 of 27 cells where on the baseline it changed 10.

That is the shape expected if the finetune narrowed the spread between seeds:
selecting among four takes recovers less when the takes agree more. It is a
consistent reading of two numbers, not a measurement of seed variance, which
has not been made.

**What remains is most of the gap.** 0.546 against a 0.689 ceiling leaves
0.143, seven times the encoder's noise. The per-speaker spread is the standing
lead: cv09 sits 0.071 from its own ceiling, cv11 and mana more than 0.19.

---

## 7. The per-speaker spread was the reference clip

At 0.546 overall the gap to each speaker's own ceiling ranges from 0.071 (cv09)
to 0.217 (mana) — three times the spread. `scripts/analyse_speaker_gap.py`
tests three explanations for it. **All three fail**, which is worth recording
because each would have implied a different fix.

### Mean-voice regression: refuted, and backwards

The obvious hypothesis is that the model pulls every render toward the centre of
the voices it was trained on, so distinctive speakers lose most. Measured
against the centroid of the nine real voices:

| | |
|---|---|
| renders pulled *toward* the centre | **1 of 9** |
| mean displacement | **−0.080** (away) |

Renders sit *further* from the centre of real speech than the references they
were cloned from. Whatever the model is doing, averaging voices is not it.

### Distinctiveness does not predict the loss

If some voices were simply harder for sitting far from the training
distribution, distance from the centroid would track the gap. Spearman rho over
the nine speakers is **+0.067** — nothing. mana is the *least* distinctive voice
in the set and has the largest gap; cv11 is the most distinctive and has the
second largest.

### One shared artefact: real, weak, and not a lever

Since renders move away from real speech in some direction, that direction might
be shared — a "this is synthetic" signature carried regardless of speaker. The
per-speaker offsets do align, weakly: mean pairwise cosine **+0.147** across 36
pairs.

Projecting it out — fitted leave-one-speaker-out, so no speaker's own offset
informs its own correction — recovers **+0.0167**, improving 8 of 9. That is
about 12% of the remaining gap, from a direction that genuinely generalises
across voices.

**It is not a fix, and should not be used as one.** Subtracting a direction from
an embedding changes the score, not the audio. Applying it would improve the
metric while leaving the render exactly as it sounds — which is the definition
of gaming the measurement. It is recorded as a diagnostic: a shared artefact
exists, and it is a minority of the problem.

*(The similarities in that test run higher than the headline numbers — 0.66
against 0.55 — because they compare each render to the mean of a speaker's
held-out clips rather than pair by pair. Comparable within the test, not to the
benchmark.)*

### What that leaves

The spread is mostly **speaker-specific and unexplained**. It is not the model
averaging voices, not the voices being unusual, and not one common artefact. So
the remaining levers are the model and the data, not post-processing.

The cheapest untested explanation was the **reference clip**. It has now been
tested, and it does not merely contribute — it accounts for the whole spread.

### The spread was not a property of the voices

Section 7 ended by naming the reference clip as the cheapest untested
explanation. It is the explanation, and it is worse than expected: **the
per-speaker ranking does not survive changing which clip the model is
conditioned on.**

Same weights, same sentences, same seed, same device. Only the reference clip
differs — index 1 instead of the manifest's index 0, with the swapped clip
moving into the held-out set so each speaker is still scored against four
recordings and its own ceiling recomputed.

| speaker | gap, ref 0 | gap, ref 1 | change |
|---|---|---|---|
| cv09 | **0.090** | 0.190 | **+0.100** |
| cv02 | 0.106 | 0.203 | +0.097 |
| cv03 | 0.115 | 0.138 | +0.022 |
| cv04 | 0.129 | 0.155 | +0.026 |
| cv08 | 0.142 | 0.117 | −0.026 |
| cv10 | 0.149 | 0.120 | −0.029 |
| cv07 | 0.187 | 0.154 | −0.033 |
| cv11 | **0.218** | 0.175 | −0.044 |
| mana | 0.209 | 0.213 | +0.004 |

**Rank correlation between the two orderings: −0.033.** cv09 was the
best-cloned speaker and becomes the seventh; cv11 was the worst and becomes the
sixth. The mean absolute change is 0.042, twice the encoder's noise, and cv09
alone moves 0.100 — seventy per cent of the average gap, from swapping one
input file.

That single swap looked like it erased the speaker effect. Running all five
clips shows it did not — see below. What it did erase is any confidence in a
ranking built on **one** reference each.

### All five clips, and two corrections to the above

Every clip as the reference, for every speaker. Eight of the 45 combinations
never rendered: below about 5 s the clip fails the pipeline's 3 s
usable-speech guard outright, which is worth stating plainly as the strongest
duration effect there is.

On the five speakers where all five clips survive:

| | variance | sd |
|---|---|---|
| within speaker (reference choice) | 0.00171 | 0.041 |
| between speaker (voice identity) | 0.00227 | 0.048 |

**Ratio 1.33.** Both are real and they are the same size. Voice identity is not
an illusion — it is simply no larger than the noise from choosing a reference,
which is why one draw per speaker produced a ranking that inverted on the second
draw. Mean within-speaker spread from best to worst clip is **0.092**, most of
the 0.143 gap.

**Correction 1.** The "erased" reading above was one draw over-interpreted. The
right statement is that per-speaker rankings need averaging over references, not
that speakers do not differ.

**Correction 2, and this one was an error rather than an overstatement.**
Section 7 reported duration against gap at rho +0.001 and concluded length was
irrelevant. That used 18 points from two runs, all of which had already passed
the 3 s guard — range restriction, on a variable whose most important effect is
the cliff at the bottom. Over all 37 surviving cells:

| | Spearman rho |
|---|---|
| duration vs similarity, all survivors | **+0.231** |
| duration vs similarity, within speaker | **+0.337** |

Longer references are moderately better, and short ones do not work at all. The
earlier claim was wrong. It is still not a rule — the longest clip is the best
one for only 3 of 9 speakers — so length is a weak predictor above the floor and
a decisive one below it.

### What survives, and what this costs

**The finetune result is unaffected.** baseline, finetuned and
finetuned-bestof4 all used the same reference clips, so the paired comparisons
between them hold reference choice constant. +0.0266 at 4.6 standard errors
stands.

**The absolute numbers carry uncertainty that was not reported.** The ±0.076
quoted on 0.546 is spread across speakers and sentences; reference choice adds
about ±0.041 per speaker on top, which the run-level means (0.539, 0.526, 0.530,
0.511, 0.557) show directly.

**Reference selection is the largest untapped lever measured so far.** An oracle
picking each speaker's best clip gains **+0.028** over the default — as much as
the finetune and best-of-4 combined. That figure is an upper bound: taking the
maximum of a few noisy draws is biased upward.

### Can the good clip be picked in advance? Not from anything measurable here

Thirteen properties of a reference clip, correlated within speaker against the
similarity it actually produced. Within speaker, because between speakers the
voices differ as much as the references do — a pooled correlation would mostly
measure which speaker a clip belongs to.

| feature | mean rho | consistent |
|---|---|---|
| speech seconds (after VAD) | **+0.575** | 7 of 8 |
| voiced fraction | +0.500 | 6 of 8 |
| speech ratio | +0.450 | 6 of 8 |
| typicality (closeness to the speaker's other clips) | +0.450 | 7 of 8 |
| **the project's own quality score** | **−0.362** | 6 of 8 |
| raw duration | +0.337 | 7 of 8 |
| SNR | −0.175 | 5 of 8 |

Speech content is what matters — actual speech seconds ranks better than raw
duration, which is why the earlier duration finding was real but weak: silence
was diluting it.

**The project's 0–100 usability score points the wrong way.** It is built to
judge whether a recording is clean enough to use, and on that job it is fine —
every clip here scores "excellent". As a way to choose between clips it is worse
than a coin, and nothing should be selecting references with it.

### The correlations do not survive contact with the decision

Ranking is not picking. Each rule below chooses the clip it ranks first, and is
scored on what that clip actually achieved:

| rule | gain over the default clip | share of the headroom | picked the true best |
|---|---|---|---|
| voiced fraction | **+0.0074** | 27% | 5 of 8 |
| speech seconds | +0.0052 | 19% | 4 of 8 |
| speech ratio | +0.0036 | 13% | 4 of 8 |
| raw duration | +0.0000 | 0% | 3 of 8 |
| typicality | **−0.0108** | −40% | 3 of 8 |
| oracle | +0.0272 | 100% | 8 of 8 |

The best rule captures **27% of the available gain and lands at +0.007 — a third
of the 0.0205 perturbation-sensitivity figure (§8).** A rank correlation of +0.575 over four clips still
puts the wrong one first most of the time, and only the first pick ships.

Note also that the strongest correlation and the best realised gain are
different features. With eight speakers and three to five clips each, that
difference is not resolvable — which is itself the point: none of these is
reliable enough to build on.

*(Typicality is worse than useless despite correlating at +0.450, and it carries
a caveat besides: it is measured against the same recordings the render is later
scored against, so part of its apparent signal is the evaluation setup rather
than the clip.)*

### Where that leaves reference selection

The +0.027 is real and it is the largest single lever measured on this corpus —
larger than the finetune. It is **not reachable by inspection**. Choosing a
reference by measurement means rendering from each candidate and comparing,
which is the same trade best-of-N already makes on the seed axis.

That comparison is made directly below, and the reference axis wins.

### Selecting over references beats selecting over seeds

If no property of a clip identifies the good one, take the trade best-of-N
already makes: render from several and choose by measurement.

**The trap this has to avoid.** The obvious selector — score each render against
the speaker's other recordings, keep the best — *is* the evaluation metric.
Selecting on it and reporting it would inflate the result by exactly the amount
the selection chose, then present the inflation as a gain.

So the evaluation clip is withheld from the decision. For each held-out clip h:
candidates are every reference except h; each is scored against the speaker's
clips excluding both itself and h; the chosen render is then reported against h
alone. h never touches the choice. Rotating h over all five clips and averaging
gives what the rule is worth on a recording the selector has not seen.

No new rendering — the five reference sweeps were already on disk.

| | |
|---|---|
| decisions | 129 |
| default | 0.5461 |
| **selected** | **0.5634** |
| oracle | 0.5812 |
| **gain** | **+0.0173**, 3.9 standard errors |
| oracle headroom captured | **49%** |
| changed the pick | 78 of 129 — 55 better, 23 worse, 51 chose the same clip |

Per speaker the gain is uneven and roughly tracks how much room there was:
cv11 +0.038, mana +0.037, cv10 +0.029, cv09 +0.021, cv03 +0.022, down to cv02
+0.000 and cv08 −0.007.

### Against the seed axis

| axis | gain | evidence | cost |
|---|---|---|---|
| seeds (best-of-4) | +0.0135 | 2.2 SE | 4× render |
| **references** | **+0.0173** | **3.9 SE** | N× render |

More gain and far firmer, for the same kind of spend. The reason is visible in
the headroom: seed choice moves a render less than reference choice does, so
searching the reference axis has more to find.

Two things this does not show. The two selectors have not been **combined** —
every render here is seed 42, so choosing over both axes at once is untested and
its gain is not the sum of these. And the comparison is not perfectly like for
like: best-of-4 was measured on four seeds, this on up to five references.

*(The absolute numbers here sit slightly above the headline 0.539 because each
decision is scored against a single withheld recording rather than four. Read
the gain, not the level.)*

---

## 8. What exists, checked rather than recalled

Every result above rests on one base, and that base was never compared against
anything current. The choice came from recalled knowledge that placed F5-TTS at
the frontier. On 11 September 2026 the landscape was checked live instead —
Hugging Face's API, model cards, papers — and the recalled picture turned out to
be months stale.

### Current models that list Persian

Verified against each model's own card or repository:

| model | size | licence | Persian | runs on this 16 GB Mac |
|---|---|---|---|---|
| [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) | 3.3 GB | code Apache-2.0, **weights CC-BY-NC** | yes | yes — MPS documented |
| [MOSS-TTS-Local-Transformer](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer) | 3.06B, bf16 | Apache-2.0 | yes | marginally — 6.1 GB of weights, and transformers 5.0.x |
| [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | 5B | research / non-commercial | yes | no |
| [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b) | ~4B | research / non-commercial | yes | no |
| SWivid/F5-TTS (the base this project continued) | 330M | — | **no** | yes |

Also found: [ParsVoice](https://arxiv.org/abs/2510.10774), the largest public
multi-speaker Persian TTS corpus, latest version 26 May 2026. A zero-shot XTTS
adapted on it reports naturalness MOS 3.6 and speaker-similarity MOS 4.0. It is
exactly the kind of corpus this project's discussion had assumed did not exist.

### Why no leaderboard answers "what is state of the art" here

- [Voice Arena](https://voicearena.com/tts-methodology) measures naturalness
  only and says so explicitly — it does not measure voice-cloning similarity. It
  covers six languages; Persian is not one.
- The multilingual cloning benchmarks — [CV3-Eval](https://arxiv.org/pdf/2505.17589)
  (9 languages), LEMAS-TTS (10), Seed-TTS-eval (zh, en) — include no Persian.
- Published speaker similarity uses WavLM-based extractors. PilotTTS reports
  0.862 and 0.815. This corpus's human ceiling on ECAPA is 0.689, so on a shared
  scale those systems would be clone-better-than-the-person — which shows only
  that the scales are different instruments.

So for Persian zero-shot cloning there is no external answer to "which is best".
The calibrated benchmark in this repository is the instrument that can give one,
and running current models through it is, as far as a search finds, the first
such comparison.

### First candidate: OmniVoice

Chosen because it is the only one that fits this machine without cutting
anything. From [its paper](https://arxiv.org/abs/2604.00688), not recalled:

- FLEURS Persian CER **1.90%** against 1.57% for the ground-truth recordings
- on identical training data it beats F5-TTS on every speaker-similarity set its
  authors report: 0.729 vs 0.655, 0.741 vs 0.664, 0.777 vs 0.750

Integrated as engine `omnivoice`, out-of-process in its own environment, with
upstream's published inference settings. Its paper's numbers say nothing about
this corpus, so the benchmark decided.

### What the benchmark decided

Nine speakers, three sentences, seed 42, the same reference clip each engine
gets — every configuration below scored by one embedder and one recogniser on
one machine, so the comparison is paired cell by cell.

| Run | Held-out | vs OmniVoice, paired | CER |
|---|---|---|---|
| F5 baseline | 0.520 | +0.0637, 5.4 SE, better in 23/27 | 3.1% |
| F5 continued on 2739 speakers | 0.540 | +0.0438, 3.7 SE, better in 22/27 | 3.1% |
| F5 continued + best-of-4 | 0.546 | +0.0371, 3.4 SE, better in 22/27 | 2.7% |
| **OmniVoice, single take** | **0.583** | — | 2.6% |

All three gaps clear the 0.0205 reference line at the cell level (§8 redoes this per speaker); against the strongest
F5 configuration the margin is about 1.8× the encoder's noise. Per speaker
OmniVoice wins seven of nine, losing only `cv02` by 0.016 and `cv03` by 0.006.
On the calibrated scale it sits at 81% of the way from stranger to same person,
against 69% for the F5 baseline.

Speakers differ in how self-consistent their own recordings are, so the raw
mean mixes clone quality with corpus quality. Measured against each speaker's
*own* ceiling the ordering is unchanged and the margin survives:

| Run | share of each speaker's own ceiling |
|---|---|
| F5 baseline | 75.2% |
| F5 continued | 78.2% |
| F5 continued + best-of-4 | 79.2% |
| **OmniVoice** | **84.8%** |

That normalisation also reorders which speakers are *hard*. `cv08` and `cv10`
look like OmniVoice's biggest wins in raw terms (+0.121 and +0.067), but both
have low ceilings — 0.607 and 0.595 — so there was less room there and
OmniVoice nearly saturates it, reaching 97% and 88%. `mana`, mid-table at 0.571
raw, is the furthest any speaker sits from its own ceiling at 77%. Treat these
ratios as direction rather than precision: each speaker's ceiling comes from
ten pairs, so it carries its own noise, and a ratio of two noisy numbers
carries more.

**The result is not that this project built something better.** OmniVoice is
k2-fsa's model, run at its published settings with no training of any kind. Its
code is Apache-2.0; its *weights* are CC-BY-NC, which its model card states
plainly ("due to constraints from its training data (e.g., Emilia)") and which
this document got wrong until it was checked. The same check found that the F5
base this project has been continuing is CC-BY-NC-4.0 as well. What this project contributed is the instrument that could tell.
The honest comparison is uncomfortable: nine GPU-hours of continuation on 2739
speakers bought +0.027, and changing the base model bought +0.037 for free -
both with speaker-level intervals that exclude zero but overlap the 0.0205
reference line, once the statistics are done per speaker rather than per cell
(below).
On this corpus, choosing the base mattered more than tuning it — and the only
reason the better base was found at all is that the landscape was checked live
instead of recalled.

Two caveats travel with the number. It is one seed and one take per cell, where
the F5 best-of-4 figure is the best of four; and CER differences this small are
at the recogniser's resolution (§9), so the CER column should be read as "no
worse", not as better.

### Second candidate: MOSS-TTS

[MOSS-TTS-Local-Transformer](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer)
(Apache-2.0, 3.06B parameters) lists Persian among its twenty training
languages. It needs transformers 5.0.x and runs in-process, so it was measured
in a Kaggle kernel of its own - after re-scoring the baseline renders there
reproduced the established 0.5196 exactly, CER included.

It took about ten GPU-hours to get a number. Two runs froze on their first cell
and printed nothing more: the second one's log places the stall in
transformers' threaded weight loading, at parameter 866 of 1600 of MOSS's audio
tokenizer. A controlled load test in clean processes could not reproduce it -
seven loads, none hung - so the cause is not established. The run that
succeeded disabled threaded loading (`HF_DEACTIVATE_ASYNC_LOAD`) and ran under a
watchdog that would have killed and retried a silent process; it finished on
its first attempt in 24 minutes. That is consistent with the thread pool being
the cause and does not show it.

That run exposed a gap in the benchmark itself. With MOSS occupying the card no
speech recogniser could allocate - faster-whisper, then its Transformers
fallback, on every cell - and the run reported zero failures, with similarity
for all 27 cells and a transcription for none. `benchmark.py` now says so. The
renders were re-scored on this Mac instead, where every paired similarity
statistic came out identical to the Kaggle scoring, so the table below uses the
Mac's recogniser for every run - the only way CER may be compared (§9).

What "the same recogniser" rests on, checked on 2026-09-19 rather than assumed:
every run in the table was transcribed on this Mac by faster-whisper
`large-v3-turbo` as cpu/int8, language `fa`, beam 5, VAD on. That settings block
has not changed since the repository's first commit (2026-07-28);
`engines/asr.py` last changed on 2026-08-02 and the CER normalisation
(`evaluation/similarity.py`) at 03:00 on 2026-08-03, both before the first
result here (the baseline, 03:17 that day); and the recogniser's packages -
faster-whisper 1.2.1, ctranslate2 4.8.1 - were installed on 2026-07-28 and have
not been reinstalled since (their install records are dated that day). Two
differences remain. The F5 runs were transcribed in-process from the rendered
waveform, MOSS and OmniVoice later from the saved 16-bit files, two minutes
apart on 2026-09-16. And re-scoring identical files on this Mac reproduced CER
in 25 of 27 cells, so differences of a few tenths of a point are inside that.

| Run, all scored on the Mac | Held-out | share of own ceiling | CER |
|---|---|---|---|
| F5 baseline | 0.520 | 75.2% | 2.9% |
| F5 continued + best-of-4 | 0.546 | 79.2% | 2.2% |
| MOSS-TTS | 0.559 | 81.2% | 6.5% |
| **OmniVoice** | **0.583** | **84.8%** | 2.8% |

Speaker-level differences, 95% t interval over the nine speakers (the rule in
"What the statistics actually support", below; this table first reported
cell-level standard errors, which overstate the evidence):

| MOSS minus | similarity | CER, points |
|---|---|---|
| F5 baseline | +0.039 [+0.007, +0.071] | +3.7 [+1.7, +5.6] |
| F5 continued + best-of-4 | +0.013 [-0.023, +0.048] | +4.4 [+2.1, +6.6] |
| OmniVoice | -0.025 [-0.069, +0.020] | +3.7 [+1.4, +6.0] |

On identity MOSS is not shown to differ from the best F5 configuration or from
OmniVoice. Its CER is higher than every other run's scored by this recogniser:
three and a half to four and a half points, higher than OmniVoice's in 18 of 27
cells and lower in 3. CER is what the recogniser makes of a render. Nobody has
listened to these, so how much of the gap is mispronunciation and how much is a
voice the recogniser finds harder to read is not established. In each of its
five worst cells the transcript drops or garbles the first word - "او گفت که
فردا صبح" comes back as "که فرداس رو" - and several have more damage after it.
Part of any Persian CER is the recogniser choosing between homophones such as
بهار and بحار, which OmniVoice's transcripts show too; that affects both engines
and does not account for the gap.

OmniVoice's CER is not detectably different from either F5 run's: -0.0 points
[-1.1, +1.0] against the baseline, +0.6 [-0.0, +1.3] against the best F5.

So on this instrument OmniVoice has the highest identity score measured, and a
CER no engine here is shown to beat. That is a statement about this instrument
- nine speakers, one seed, a local scale, no systematic human listening - not a
world ranking. And the instrument may favour it: see the training-data caveat
in §9.

### The last two candidates: Fish S2 Pro and Higgs TTS 3

Both were listed and left unmeasured because their licences are research-only
and neither fits this Mac. That decides what may be *shipped*, not what may be
*known*, so both were measured on a T4 in float16 - neither card has a bfloat16
unit, and `torch.cuda.is_bf16_supported()` returns True there anyway because it
counts the emulated path, which is worth knowing before trusting it.

[Fish S2 Pro](https://huggingface.co/fishaudio/s2-pro) (revision
`1de9996b`, 11 GB) runs through upstream's own dual-AR code. Two things had to
change to make it run at all, both recorded in `scripts/fish_worker.py`: the
reference is read with soundfile, because torchaudio 2.9 delegates to torchcodec
and no FFmpeg shared library on that image satisfies it; and the transformer is
built with `max_length=4096`, because at the checkpoint's declared 32768 the
causal mask alone is a gibibyte of triangle. Measured on the card: transformer
8.51 GiB, KV cache 0.57, codec 3.85 - which is why the codec sits on the second
T4.

[Higgs TTS 3](https://huggingface.co/bosonai/higgs-tts-3-4b) claims Persian in
the top tier of its own model card, the 85 languages it puts under "WER/CER
under 5", and tops its own eleven-system multilingual table. Boson ships it for
SGLang-Omni or vLLM-Omni, both of which want compute capability 8.0 against the
T4's 7.5, and the one public attempt to force that path returned WAV files whose
every sample was -32768. It was measured instead through
`multimodalart/higgs-audio-v3-tts-4b-transformers` at revision `30f01593`,
whose `model.safetensors` and `tokenizer.json` have sha256 hashes identical to
Boson's own - verified again inside the run. The *weights* are therefore the
vendor's; the generation loop is not, and a port that differed would be measured
here as the model. Its CER landing at 2.9%, consistent with the card's
under-5 claim, is the only evidence available that the port is faithful.

| Run | Held-out | share of own ceiling | CER | licence of the weights |
|---|---|---|---|---|
| **OmniVoice** | **0.583** | 84.8% | 2.6% | CC-BY-NC |
| MOSS-TTS (older variant) | 0.559 | 81.2% | 6.5%, scored on the Mac | Apache-2.0 |
| F5 continued + best-of-4 | 0.546 | 79.2% | 2.2% | base is CC-BY-NC |
| Fish S2 Pro | 0.537 | 77.7% | **9.5%** | research only |
| Higgs TTS 3 | 0.531 | 77.3% | 2.9% | research only |
| F5 baseline | 0.520 | 75.2% | 2.9% | base is CC-BY-NC |
| MOSS-TTS v1.5, official recipe (below) | 0.507 | 73.5% | 10.9%, same session's baseline 3.1% | Apache-2.0 |

Neither new candidate changes the ordering. Against the best F5
configuration, Higgs is -0.015 [-0.048, +0.018] and Fish -0.009 [-0.036,
+0.018]: both intervals include zero, and with nine speakers that is weak
evidence either way (below).

CER is only compared inside one scoring environment (§9), and each Kaggle
kernel re-scored the F5 baseline renders in its own session, which gives every
remote run a same-environment comparator. Against it, Fish is +6.4 CER points
[+4.2, +8.6] - worse for all nine speakers, consistent with its own model card
putting Persian in the third tier - while Higgs (-0.2 [-1.4, +0.9]) and
OmniVoice (-0.5 [-1.5, +0.5]) have intervals that include zero. The CER
columns above mix environments: MOSS's 6.5% was scored on this Mac from the
Kaggle renders (`moss-local`, because no recogniser fit beside MOSS on the
card), and against the Mac-scored baseline it is +3.7 points [+1.7, +5.6].
`eval/results/_provenance.json` records where every run was scored and why.

### MOSS-TTS v1.5, at its official recipe

The newer variant, `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` at
`be7766a6` - Apache-2.0 for the model, the codec (`MOSS-Audio-Tokenizer-v2` at
`f6e20e54`) and the code (`OpenMOSS/MOSS-TTS` at `934d6826`) - was measured on a
T4 pair at the card's recipe: the repository's runtime pins for everything
the model's code imports (torch 2.9.1+cu128, torchaudio 2.9.1+cu128,
transformers 5.0.0, safetensors 0.6.2), installed and checked by the notebook
rather than trusted to the image - the pyproject's other pins were left at the
image's versions and differ: torchcodec 0.10.0 (official 0.8.1, never called),
numpy 2.0.2 (2.1.0), tqdm 4.67.3 (4.67.1), einops 0.8.2 (0.8.1), scipy 1.16.3
(1.16.2), all listed in each run's `environment.json`;
bfloat16, which Turing runs emulated; SDPA, the card's own fallback below
compute 8.0; the recommended sampling (temperature 1.7, top-p 0.8, top-k 25,
repetition penalty 1.0, 4096 new tokens); 48 kHz stereo, scored like every
other engine on its channel mean. Cloning goes through the processor's
`build_user_message(text, reference=[audio], language)`, as in the card's
transformers example.

Three attempts failed before the first render. For the last of them, the T4
probe, the cause is documented: the runner that read the worker's output also
wrote it to the notebook display, the display stopped taking data, and the
worker blocked on a full pipe - the runner's own copy of what it read shows it
(`eval/runs/persian-voice-moss15-probe-v2.json`). The first two, in the main
kernel, kept no such copy; that they stalled the same way is an inference from
identical symptoms, not a finding. With the runner fixed, the model loads in
about 55 s and renders a sentence in 8-10 s.

Two sessions differ in one string. The first sent the language tag "Persian",
by analogy with the card's examples; the official string, found afterwards in
the card's language table and in the official v1.5 app, is "Persian (Farsi)",
and the second sent that. Their baseline controls agree in every cell -
similarity, CER and transcript. That shows the recogniser gave the same output
for those 27 inputs in both sessions, not that it would for every input; on
that evidence the two are compared directly:

| | held-out | share of own ceiling | CER (same session's baseline: 3.1%) |
|---|---|---|---|
| tag "Persian" | 0.524 | 75.8% | 9.0% |
| tag "Persian (Farsi)", the official string | 0.507 | 73.5% | 10.9% |

The tag made no detectable difference: -0.018 [-0.046, +0.010] in similarity,
+1.9 CER points [-3.4, +7.2]. Both intervals include zero and are wide, so this
does not show the tag has no effect - only that nine speakers did not detect
one. Against the F5 baseline the official configuration is -0.013 [-0.053,
+0.027] in similarity - no detectable difference in either direction - and +7.7
CER points [+2.8, +12.7] in its own session; against OmniVoice -0.077 [-0.122,
-0.032], an interval that also clears the 0.0205 line; against MOSS v1.0 -0.052
[-0.095, -0.009]. In the recogniser's transcripts the errors gather at the start
of the sentence - a first word garbled or dropped, or words added before it -
and one render (`cv09`/s3) is transcribed 68.6% away from its target. Nobody
has listened to these renders, so whether that one is unintelligible to a
person, and how much of the CER is mispronunciation, is not established.

So on this development set's similarity measure, the card's claim that v1.5
with a language tag beats v1.0 is not reproduced: v1.5 scored lower, with an
interval that excludes zero. CER was scored in different environments for the
two and is not compared. Three things keep this from being a verdict on the
model: nine speakers; renders that are single samples at temperature 1.7; and
metrics that are not the card's. And one official mode is untested:
"continuation", which also takes the reference clip's transcript and which the
official server documentation says gives better speaker similarity. The
transformers processor at this revision has no field for it.

### What the statistics actually support

Every comparison above was previously reported over 27 cells as if they were 27
independent observations. They are not: three cells share a speaker, a
reference clip and a voice. `scripts/compare_runs.py` now averages each
speaker's cells and puts a t interval over the nine speaker differences, which
roughly doubles every interval. This rule was adopted on 2026-09-18, after the
results below had been seen; it is a correction, not a pre-registration.

| comparison | speaker mean | 95% interval | interval vs 0 | interval vs +/-0.0205 |
|---|---|---|---|---|
| OmniVoice - F5 baseline | +0.064 | [+0.025, +0.102] | excludes 0 | clear of it |
| OmniVoice - Higgs | +0.052 | [+0.012, +0.093] | excludes 0 | overlaps it |
| OmniVoice - Fish | +0.046 | [+0.011, +0.082] | excludes 0 | overlaps it |
| OmniVoice - best F5 | +0.037 | [+0.005, +0.069] | excludes 0 | overlaps it |
| MOSS - F5 baseline | +0.039 | [+0.007, +0.071] | excludes 0 | overlaps it |
| best F5 - F5 baseline | +0.027 | [+0.015, +0.038] | excludes 0 | overlaps it |
| OmniVoice - MOSS | +0.025 | [-0.020, +0.069] | includes 0 | - |
| Higgs - best F5 | -0.015 | [-0.048, +0.018] | includes 0 | - |
| OmniVoice - MOSS v1.5 | +0.077 | [+0.032, +0.122] | excludes 0 | clear of it |
| MOSS - MOSS v1.5 | +0.052 | [+0.009, +0.095] | excludes 0 | overlaps it |
| MOSS v1.5 - F5 baseline | -0.013 | [-0.053, +0.027] | includes 0 | - |

How to read it. An interval that excludes 0 means the data would be unusual if
the true difference were zero, for these nine speakers; it is not a proof of
direction, and with the other comparisons in the table some such result is
expected by chance. Two rows' intervals also clear +/-0.0205: OmniVoice over
the F5 baseline, and OmniVoice over MOSS v1.5.

What 0.0205 is matters here, because earlier versions of this document called
it "the encoder's noise". It is the largest change in ECAPA score measured
across identity-preserving perturbations of **one** render - 85-95% crops,
+/-1 dB of gain, dither at -60 dBFS - with sd 0.0055, recorded on 2026-07-30 in
commit `dee1ace`, four days before the benchmark produced its first result.
So its value was not chosen by looking at benchmark results. But it is a
sensitivity figure from a single render: it is not re-scoring repeatability
(re-scoring identical audio reproduces every cell), not seed-to-seed
generation variance, not clip-to-clip variance, and not a threshold anyone has
shown a listener can or cannot hear. It is kept as a practical reference line,
not a decision bar with a measured justification.

Nothing here says two engines are equal. The design is weak: taking the
speaker-level spread of paired differences as about 0.048, nine speakers give
roughly a 59% chance of an interval excluding zero if the true difference is
0.04, 38% at 0.03 and 20% at 0.02 - and that spread is itself estimated from
nine speakers, with a 95% interval of about [0.032, 0.092]. An interval that
includes zero in this table is therefore weak evidence of no difference.

## 9. Caveats

- **The nine speakers were chosen by the encoder that grades the results.**
  `scripts/build_eval_corpus.py` keeps a speaker only when their worst pair
  against themselves beats their best pair against anyone else, measured with
  the same ECAPA-TDNN that produces every number in this document. That rule
  exists for a good reason - a Common Voice `client_id` is an account, not a
  person, and one candidate's own two clips scored 0.099 against each other -
  but it is selection on the outcome variable. It pushes the ceiling up and the
  floor down relative to an unselected sample, so "84.8% of the human ceiling"
  is a statement about nine speakers who passed an ECAPA coherence test.
  Comparing engines on this set is still well-defined - they all meet the same
  speakers - but it is conditional on the set: a selection made by this encoder
  can favour whichever engine best preserves what this encoder listens to, so
  neither the ranking nor its generalisation is protected from the bias. And a
  speaker dropped for incoherence may be a failure of the encoder on that audio
  rather than evidence that the account is two people; the record should not
  say which.
- **OmniVoice has very probably trained on eight of the nine speakers.**
  Checked on 2026-09-19. OmniVoice's paper (arXiv 2604.00688v3, Appendix A)
  lists Common Voice among its training data, without a version, and its
  language table gives Persian 366.07 hours - close to Common Voice Persian's
  372.21 validated hours in v23 (2025-09-05), which the Hub account the
  official training docs link to for their manifests (`zhu-han`) published in
  December 2025 (`zhu-han/commonvoice_v23`). All eight Common Voice speakers here
  already had validated clips in v22 (2025-06-20), and releases are cumulative.
  So the voices - possibly the very clips used as references and held-out
  recordings - were very likely in its training set. That is an inference: the
  paper publishes no manifest. It can favour OmniVoice on this development set
  over engines that never heard these people, and none of the other engines'
  training data was checked for the same thing. `mana` comes from Mana-TTS,
  which is not on OmniVoice's list. Of the 3,791 speakers in the Common Voice
  26 subset this project uses, only 4 are absent from v22, so no test set drawn
  from it can avoid the problem.
- **"Different recording session" was an assumption and has been withdrawn.**
  Common Voice publishes no session field. Earlier versions of this project
  labelled every clip its own session and carried a same-session band of 0.800
  into `results.py`; what is actually evidenced is "a different clip by the
  same speaker". What is measured directly is a gap: across five engines a
  render scores 0.11 to 0.16 higher against the clip it was conditioned on
  than against that speaker's other recordings. That the gap is recording
  channel, and that it is the same mechanism in every engine, are hypotheses.
- **Cells are clustered, and the intervals in §8 now say so.** Three cells
  share a speaker; nine share a sentence. Every statistic here is over nine
  speaker-level differences, not 27 cell-level ones.
- **Scores are recorded to four decimals.** `benchmark.py` rounds each cell's
  similarity before writing it, and a run mean is the mean of those rounded
  numbers. Where this document says two scorings agree "exactly", what is
  demonstrated is that every rounded cell matched, checked cell by cell by key
  since `79fec9d` (equal means alone prove nothing - opposite differences
  cancel). Two raw scores that round to the same four decimals can differ by up
  to 1e-4. The encoder runs on CPU in every environment, which is consistent
  with the agreement but does not by itself prove the raw outputs identical.
- **Out-of-process renders passed through 16-bit PCM.** The OmniVoice, Fish,
  Higgs and Chatterbox workers write their render with soundfile's default for
  WAV, which is 16-bit PCM, and the engine reads it back; F5 and MOSS v1.0 run
  in-process and never do. Quantisation at 16 bits sits around -96 dBFS, far
  below anything these renders contain, so no effect on the scores is expected
  - but it is a difference in the processing path between engines, and it has
  not been measured. The MOSS v1.5 worker writes 32-bit float. Changing the
  others would change every future measurement relative to the recorded ones,
  so it is written down here rather than done silently.
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
- **The recogniser is not deterministic, so small CER differences are noise.**
  Re-scoring one run's existing renders — identical audio, nothing re-generated —
  reproduced held-out similarity exactly in 27 of 27 cells, but CER matched in
  only 25 of 27, differing by up to 2.86 points in a single cell and moving the
  aggregate from 2.9% to 3.0%. Every CER difference of a few tenths quoted above
  is therefore at or below the instrument's resolution, and the ones flagged in
  §3 and §6 have been softened accordingly. What survives is the gap that
  carries the claim: 2.9% against the 7.8% the same recogniser scores on these
  speakers' real recordings. Similarity is unaffected — it reproduced exactly,
  and it is what every ranking in this document rests on.
- **Across machines, similarity is exact and CER is not.** All three F5 runs
  were re-scored on a Kaggle T4: the same wav files, a different device and a
  separately installed embedder and recogniser. Held-out similarity reproduced
  *exactly* in 27 of 27 cells for all three, maximum difference 0.00000 — which
  is what makes a cross-device comparison of similarity legitimate at all. CER
  did not: it rose by 0.27, 0.38 and 0.49 points (1.8, 2.1 and 2.4 standard
  errors). But 22 to 24 of 27 cells were identical each time, so the difference
  is carried by three or four cells rather than by a drift across the set, and
  that per-cell instability is the same size as re-scoring on one machine (25
  of 27), which at first reading suggested the recogniser's own
  non-determinism rather than anything about the GPU.

  A second remote scoring settled it, and against that reading. The kernel was
  re-run and the same three F5 runs re-scored again: every cell came back
  *identical* to the first remote scoring — 27 of 27 on similarity **and** on
  CER, mean difference +0.0000, maximum 0.000. The remote scorer is
  deterministic, while the same operation on this Mac is not. That removes
  run-to-run variation as the explanation and leaves the installation itself:
  different builds of the recogniser, or the machine it runs on. The similarity
  half of this has a mechanism, found later in the code rather than inferred:
  `evaluation/similarity.py` loads the encoder with `run_opts={"device": "cpu"}`,
  so `--device cuda` never reaches the similarity path. The CER half has one
  too, and it retires the vaguer explanation above: `engines/asr.py` runs the
  recogniser as cpu/int8 without a GPU and cuda/float16 with one, so Mac and
  CUDA CER were never the same measurement. Each run now records this
  (`scoring_environment`, since `56a963e`). The practical
  rule is therefore narrower than "CER is noisy". CER may be compared between
  runs scored on the *same* machine, and never across machines. Similarity may
  be compared anywhere — it reproduced exactly in every test of both kinds.
