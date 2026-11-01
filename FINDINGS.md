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
`eval/results/_provenance.json` records where every run was scored and why,
and `eval/results/<label>.recipe.json` what produced it: model and codec
revisions, settings, precision, the reference clips' sha256, the session and
the results file's own sha256. For the files quoted here, which predate
2026-09-19, the recipes are reconstructed (`scripts/reconstruct_recipes.py`),
every field naming its source and anything unknown written as unknown; runs
since then write their own, once.

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
- **A reference longer than 11.5 s is cut, and its transcript is not.**
  `audio/preprocess.py` prepares a clip longer than 1.15 times the 10 s target
  by choosing a window, while the benchmark keeps giving the transcript written
  for the whole clip; the Studio re-transcribes only when an engine's own limit
  cuts the clip (F5's 12 s). On the development set this touches one reference:
  `mana` (11.89 s) is prepared by the current code from 0.06 s to 11.49 s, the
  whole speech run the voice-activity detector found, so no word is known to be
  missing - but the 0.4 s cut from its end has not been checked by ear. The
  sealed test avoids the case by rule (SOTA_ROADMAP §3.2), and every benchmark
  run since `cb083f2` records the window each reference went through.
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

---

## 10. The Persian-specific model, measured rather than argued about

On 19 September 2026 a Persian TTS with voice cloning was announced that runs
offline on a CPU (`nimaone/persian_tts`). The packaging is not the interesting
part: underneath it is
[`mehdi-hf/pocket-tts-farsi-v2`](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2),
109.5 M parameters distilled from Kyutai's pocket-tts and trained on **973
hours of Persian from 2,978 speakers** — 7.6 times the 128 hours this project
is adapting OmniVoice on — reading romanised phonemes through Homo-GE2PE
rather than Persian script. Its authors report, on 300 held-out Common Voice
pairs, speaker similarity **0.859**, median per-item WER 0.333 and UTMOS 3.11,
and their run report is unusually careful: seed spread, a warning that mean WER
cannot rank checkpoints, and an instruction not to compare against their own v1
number because it was measured on contaminated data.

None of that answers the question this project asks, because it is a different
encoder against a different reference on a different set. So the model was
added as an engine (`ff8101e`, by its own card's recipe) and scored here.

**Development set, 9 speakers × 3 sentences × seed 42, both scored in the same
session on this Mac** — the OmniVoice renders are the stored ones re-scored, so
similarity and CER are like-for-like:

| | held-out similarity | to the prompt | CER | WER | not transcribed |
|---|---|---|---|---|---|
| OmniVoice (base) | **0.583** ± 0.075 | 0.741 | **2.8%** | 24.8% | 0 |
| Pocket-TTS Farsi v2 | 0.333 ± 0.196 | 0.410 | 8.9% | 33.2% | 2 of 27 |

Paired over the same 27 cells: **−0.2503**, 7.6 standard errors from zero,
worse in 26 of them. On this corpus 0.333 is *below* the highest
different-speaker pair (0.508) — these renders are not reliably distinguishable
from a different person's, while the corpus's own recordings reach 0.689.

### The integration was doubted before the result was believed

Three checks, because a negative result about someone else's model is worth
less than the effort of trying to break it:

1. **Prompt length.** This model continues from a prompt rather than embedding
   it once, and its training utterances averaged 3.8 s, so the 11.4 s reference
   could have been the handicap. On 4 speakers: 4 s gives 0.412, 6 s gives
   0.388, the whole window 0.381. Better, and nowhere near the gap.
2. **Its own released samples.** The three demos shipped beside the weights,
   rendered by the authors from their own ~5 s prompts, score **0.321, 0.368
   and 0.346** against those prompts on this project's ECAPA — *below* the
   0.410–0.500 this integration gets. Whatever the gap is, it is not the wiring.
3. **The card's three flags.** Loaded from the shipped `model.yaml`, the model
   reports `capitalize_first_letter`, `append_terminal_punctuation` and
   `pad_with_spaces_for_short_inputs` all false — the settings whose absence
   silently deletes or corrupts the first word of every chunk.

The renders are intelligible: the recogniser reads them back as the intended
sentences at 2–14% CER per render, against a 7.8% floor on the corpus's own
human speech. What is low is identity transfer, on this encoder.

### What it is better at

Cost, by a distance. 0.45 GiB of weights, about 4 s to load, and roughly 2 s of
speech per 1.4 s of CPU — no GPU, no 6 GB of free memory, no second 3 GB
environment. It is also the only engine here that synthesises from Homo-GE2PE
output, the family of the Persian G2P work beside this project; §9's note that
nothing here can read that G2P's phonemes is no longer true.

### What this does not say

Their 0.859 is neither reproduced nor disputed here — it is not on this scale.
Nobody has listened to any of these renders. CER is compared only inside this
one session. And eight of the nine development speakers are Common Voice, which
both models were very probably trained on: that caveat cuts both ways, and is
why the sealed test exists. Record:
`eval/runs/persian-voice-pocket-tts-v1.json`.


## 11. LongCat-AudioDiT, measured before anyone trains it on Persian

On 25 September 2026 the user asked whether LongCat-AudioDiT-1B, "the best
voice-cloning engine in the world", should replace OmniVoice as the model
this project adapts, or be joined with the best Persian TTS. Checked live that
day (SOTA_ROADMAP section 4): it reads Chinese and English only, ships no
training code, and its lead is the 3.5B's WavLM similarity on Seed-TTS-eval,
which other scorers and noisy prompts do not reproduce. An adaptation would
mean writing its training recipe and teaching it a language from nothing, so
before that, one question was measured: given a Persian speaker's clip and
text both models read - English - whose render is closer to that speaker's
other recordings?

**Design, fixed before any render** (`eval/runs/persian-voice-longcat-probe.json`):
Neyshekar's 25 usable valid speakers (not the development set, not the sealed
tests, not on OmniVoice's training list), six English sentences, seed 42,
OmniVoice (published, `language=en`) against LongCat-1B at upstream's
settings (fp32, the README's 16-step APG call) and the 3.5B with its
transformer on the second T4, all in one session and one interpreter. The bar:
LongCat-1B's whole 95% interval over speakers above 0.0205.

| run | held-out | against its prompt | CER |
|---|---|---|---|
| OmniVoice, English | 0.521 | 0.659 | 0.1% |
| **LongCat-1B, English** | **0.589** | 0.740 | 3.5% |
| LongCat-3.5B, English | 0.597 | 0.747 | 2.3% |
| OmniVoice, Persian (2 sentences, exploratory) | 0.601 | 0.738 | 2.2% |
| LongCat-1B, Persian (2 sentences, exploratory) | 0.607 | 0.784 | 81.6% |

This corpus's scale: different speakers 0.172, the same speaker 0.710.

**The pre-registered reading.** LongCat-1B minus OmniVoice: **+0.068 [+0.055,
+0.080]**, higher for all 25 speakers (the smallest margin +0.015) and in 135
of 150 cells, both arms' renders the requested English (one LongCat render of
150 above WER 0.5). By the rule, LongCat clears the line: **an adaptation is
worth costing.** The 3.5B adds nothing that matters: +0.008 [-0.001, +0.017]
over the 1B. LongCat reads English a little worse (WER +3.8 points [+1.7,
+5.9]) and cannot read Persian at all untrained (CER 81.6% against 2.2%).

The explanation proposed on the same day - that its Seed-TTS-eval lead is the
prompt recording copied, which this project's held-out metric would not
reward - **is not supported**: LongCat's prompt-minus-held-out gap exceeds
OmniVoice's by +0.014 [-0.001, +0.029]. Its lead is about as large against
the speaker's other clips as against the prompt.

**What qualifies it - found after the result, so it decides nothing.**
OmniVoice reading *Persian* on the same speakers scores 0.601, higher than its
own English for all 25 of them (+0.080 [+0.059, +0.100]), and LongCat-1B
reading English is level with it (-0.012 [-0.030, +0.006]). LongCat's
unintelligible Persian lands there too (+0.006 [-0.008, +0.020]). So the
English margin is about the size of OmniVoice's own English penalty: on the
language this project needs, OmniVoice already stands where LongCat stands in
English. The bar a Persian LongCat would have to clear is OmniVoice's Persian,
0.601 on these speakers, not its English. Whether the penalty is the `en`
language tag (passed per OmniVoice's card) or the language itself is untested,
and the Persian runs used two sentences against six.

**What this does not say.** How LongCat would clone once it reads Persian - an
adaptation can move identity, and nothing here predicts which way. Whether any
of these differences is audible: nobody has listened. Record:
`eval/runs/persian-voice-longcat-probe.json`, evidence in
`eval/runs/evidence/longcat-probe-v1/`.

## 12. The sealed test: the adaptation did not carry over to unseen speakers

The question the whole adaptation was for: does OmniVoice, trained further on
Persian by its official recipe, clone *unseen* Persian speakers better than the
published model? It was answered on 30 September 2026 by the test fixed on 19
September, before any training (`eval/runs/persian-voice-omnivoice-test.json`,
SOTA_ROADMAP section 3.1): the candidate the development set chose by a rule
written in advance - the float16 export at step 1,000 of 5,000 - against the
published weights, the weights being the only difference, on two sealed sets
nobody had rendered: Neyshekar's test speakers (primary, not on OmniVoice's
training list) and Common Voice test speakers (secondary, very probably seen by
OmniVoice). Twelve texts, seeds 42 and 43, both models scored in one session per
set.

The first session scored the secondary set; its primary runs stopped at the
sixth speaker on a harness error - the Studio refused ney06's reference (2.7 s
of speech after trimming, under its 3 s minimum) and the benchmark crashed
instead of noting it. With that fixed and before any primary score was read,
the primary set was rerun alone in a session of its own; ney06 is left out by
the pre-registered rule for a speaker with failed cells.

| set | speakers | published | step 1,000 | difference [95%] | CER, points |
|---|---|---|---|---|---|
| **Neyshekar (primary, unseen)** | 27 of 28 | 0.5959 | 0.5972 | **+0.0014 [-0.0042, +0.0069]** | -0.11 [-0.75, +0.53] |
| Common Voice (secondary) | 30 of 30 | 0.5162 | 0.5202 | +0.0040 [+0.0002, +0.0079] | -0.56 [-1.15, +0.03] |

**The pre-registered reading, as the code gave it.** Primary: no detectable
improvement in held-out similarity - the interval's lower end is not above 0
(one-sided p 0.31) - and CER no different. Secondary: H0 rejected on that set,
a gain of 0.004, a fifth of the 0.0205 noise line. Nothing clears the practical
line anywhere.

**What it means.** The spread over speakers was 0.014 against the 0.050 the
power design planned for, so the test resolved far more finely than planned:
the interval's upper end, +0.007, excludes the 0.03 it was designed to detect.
On speakers nobody trained on, 5,000 steps of the official recipe on 128 hours
of Persian did not measurably improve zero-shot cloning, and did not change how
well the recogniser can read the output. The development set's +0.025 did not
carry over; that set is eight Common Voice speakers OmniVoice had very probably
trained on, and the secondary set, from the same source, keeps a trace of the
same thing. So the uplift the development set showed was, as §9 warned it
might be, familiarity with the source rather than better cloning. By the rule
written before the training, no improvement is claimed.

**What it does not say.** Only the step-1,000 export was tested, as the rule
fixes; the later exports scored no better on the development set. Similarity
and CER are an encoder's and a recogniser's readings; nobody has listened. And
the published OmniVoice on these unseen speakers - 0.596 on Neyshekar, 0.516 on
Common Voice - is what the arms session compared F5-TTS and Pocket-TTS against.
Evidence: `eval/runs/evidence/omnivoice-test-v1/` (secondary) and
`eval/runs/evidence/omnivoice-test-primary-v1/` (primary).

**Was OmniVoice the right base? Yes - the arms, registered 20 September.** The
development set that chose OmniVoice was contaminated too, so F5-TTS Persian (at
its registered 48 steps) and Pocket-TTS Farsi v2 rendered the same sealed cells
(`eval/runs/evidence/omnivoice-arms-v2/`, read by `scripts/read_arms.py`):

| set | OmniVoice (step 1,000) | F5-TTS | Pocket-TTS | OmniVoice minus F5 | OmniVoice minus Pocket |
|---|---|---|---|---|---|
| Neyshekar (unseen) | 0.597 | 0.485 | 0.376 | **+0.112 [+0.091, +0.133]** | **+0.221 [+0.193, +0.249]** |
| Common Voice | 0.520 | 0.471 | 0.317 | +0.049 [+0.034, +0.064] | +0.203 [+0.172, +0.234] |

Every interval clears zero: by the registered reading, the base choice survives
contact with unseen speakers. On speakers nobody trained on, OmniVoice's lead
over F5 is *larger* than on the development set (+0.112 against +0.064), so the
contaminated set understated it rather than manufactured it. Between the arms,
F5 reads more intelligibly than Pocket (CER -5.4 points [-9.0, -1.8]). So the
project's best measured model for cloning unseen Persian speakers is OmniVoice -
the published weights, which the adaptation matched but did not beat.

## 13. Pronunciation, heard: ezafe and short vowels, and the model follows its marks

The voice listening test was deferred on 3 October at the user's request:
pronunciation first. To find the errors before fixing them, the user listened,
blind to which was which, to the A/B/C renders of 24 September - 14 hard
sentences read by OmniVoice on plain script (A), on a diacritised twin (B), and
by the IPA model on phonemes (C) - and marked every word said wrong
(`eval/runs/persian-voice-pronunciation-review-v1.json`, registered before
listening).

| | readings with no wrong word | words marked wrong | picked best |
|---|---|---|---|
| A, plain script | 3 of 14 | 19 of 199 (9.5%) | 8 |
| B, diacritised twin | 3 of 11 | 10 of 97 (10.3%) | 4 |
| C, IPA model | 0 of 14 | 27 of 199 (13.6%) | 0 |

**What plain script gets wrong** (A, classified from the listener's notes):
9 of the 19 errors are **ezafe** - added where none belongs (دانشگاه، مرد،
غروب، ساعت، عاملی) or missing where it belongs (باران شدید، کتاب خوب، پایان
مسابقه، ایستگاه قطار) - and 8 are **short vowels**, two of them homographs
(chatr read chetr, ghal'e ghel'e, amanat emanat, gel gol, koshti keshti).

**The model reads the marks it is given.** Where a twin's diacritics were
right, B cleared errors A made in the same sentence (chatr, the ezafe of باران
and خوب, قلعه). Where they were wrong - four places, found and listed before
anyone listened - it read them as written. A damma came out as a long u twice.
**The IPA route is closed:** worst on every count, never picked, and to this
listener its accent sounds foreign even where the words are right.

So the lever is the text in front of the model: ezafe and the short vowels a
word needs, written into its input from a source accurate enough that the marks
are right, plus a way to say where no ezafe belongs. One listener, fourteen
chosen-hard sentences, one voice: this names the error types, it measures no
system. These sentences are now development material; a fix is measured on new
ones written before it.

## 14. Which text tools can write the ezafe and the vowels

Section 13 put the fix in the text: mark what plain script leaves out, from a
source accurate enough that the marks are right. On 3 October the candidate
sources were measured on text alone - no audio, no GPU - against two golds:
SentenceBench (400 annotated sentences; a word's ezafe read off its annotated
pronunciation) and the 14 review sentences (`eval/manifests/frontend_review14_gold_v1.json`).
Registered before any component ran (`eval/runs/persian-voice-frontend-components-v1.json`);
amended before any score, because HomoRich - the training data named for
Homo-GE2PE and the ALBERT model - turned out to hold 98 of SentenceBench's 100
ManaTTS sentences and 79 of its 88 Common Voice ones. The choice is read on the
213 sentences found in no component's named training data.

| ezafe, per word | unseen (213 sentences) | review's 14 sentences |
|---|---|---|
| hazm ParsBERT tagger (spaCy) | 1.7% wrong, F1 0.93 | 0 of 198 |
| ALBERT (`abreza/persian-ezafe-albert`) | 2.5%, F1 0.90 | 4 |
| Homo-GE2PE (the app's G2P) | 3.0%, F1 0.88 | 8 |
| hazm CRF tagger | 4.8%, F1 0.83 | 6 |
| plain script - no ezafe anywhere | 12.6% | 48 |

**The ezafe source is hazm's ParsBERT tagger.** Against ALBERT the difference,
-0.8 points [-1.6, 0.0], only reaches zero, so the registered tie-break applied:
it is right on all of A's heard ezafe errors (ALBERT adds one after مرد, as
OmniVoice did). On the 14 sentences it makes no ezafe error at all. About a
third of its 25 errors on the benchmark are the benchmark's own slips. It
costs a separate environment (Python 3.12, spaCy, an older transformers),
655 MB of weights and about 4 GB of memory while it runs.

**No vowel source is good enough to mark every word.** Homo-GE2PE gets the
vowels of 88.7% of the unseen words right (homographs 74%); KaamelDict answers
only the 52% of words it lists one way, 93% of them right. Part of that is the
benchmark's slips and split words, but nothing approaches the 96% that
OmniVoice reached alone in the review, which the registration set as the bar.
On the 14 sentences Homo-GE2PE is right on 8 of A's 9 vowel errors - and wrong
on six words A read right (کنار, پنجره, the second کشتی, روبه‌رویی, دست‌کم,
مسافران). So step 2 marks vowels only where they are needed. The 12 readings
Claude wrote that a component disputed went to the user blind; the user agreed
with all 12, so the 14-sentence numbers stand.

## 15. Writing the marks in: the ezafe helps, the sukun does not, the vowels are undecided

Step 2 put the section 14 choices in front of OmniVoice. Twelve new hard
sentences (147 words, `eval/manifests/frontend_listening_v1.json`) were rendered
four ways from one voice and seed: A plain, E with the tagger's ezafes written
in, EN with a sukun closing nouns, adjectives and numbers it gives no ezafe, ENV
with the first short vowel of nouns and adjectives whose KaamelDict readings
agree. Registered before the front-end ran on them
(`eval/runs/persian-voice-frontend-listening-v1.json`); the Mac's memory guard
refused the renders, so they were made in one Kaggle T4 session, an amendment
written before the push. The user listened blind, each condition three times in
each slot, loudness matched.

| | words marked wrong (of 147) | readings with none wrong | picked best | mean CER |
|---|---|---|---|---|
| A plain | 12 | 3 of 12 | 1 | 3.1% |
| E + ezafe | **9** | 4 | **4** | 3.9% |
| EN + sukun | 12 | 5 | 2 | 3.9% |
| ENV + first vowels | 10 | 5 | 3 | 2.9% |

Two sentences were picked "no difference". By the registered rules each step is
kept only if it lowers the words marked wrong: **the ezafe is kept** (12 -> 9),
**the sukun is dropped** (9 -> 12), the first vowels are kept against EN
(12 -> 10). No step raised CER by the 2 points that would flag harm. The ezafe
and the vowels without the sukun were never rendered together, so only the
ezafe goes into the app now; the vowels wait for that combination to be heard.

How firm this is: one listener and differences of two or three words. In four
sentences a step changed nothing, and the renders came out bit-identical (seed
42 is deterministic on the GPU), so the listener unknowingly heard the same
audio twice: three pairs were judged alike, one differed by a word. Splitting
each step's changes by whether the word itself was marked (Claude's reading;
there are no notes): the ezafe fixed 4 marked words, all missing ezafes
(باریک, خونه, طولانی, ماه), and broke 2 - the compound مدیرعامل, heard right
only without it. The sukun stopped two plausible spurious ezafes (کوچک, غروب)
and broke سبد and شروع. The first vowels fixed 4 marked words (خشک, سبد, شهر,
هیئت) and broke 4 that carried the right vowel (مهربان, بزرگ, عذرخواهی, شرکت):
their net gain came from words they did not touch. Still wrong in all four
forms: the homograph کرم (no agreed reading, so unmarked), تبر, برنگشته.

## 16. The ezafe in the app, and the tagger without spaCy

Section 15's ezafe went into the app's text pipeline for OmniVoice on 3
October. As published, hazm's tagger is a spaCy pipeline that peaked at 4.3 GB
for 13.5 s to tag one sentence on the M1 Pro - most of it spaCy's import and
spacy-transformers' load - too much to sit beside a loaded OmniVoice on a 16 GB
Mac. So its tensors were read out of the pipeline's files once and are run with
transformers alone, repeating what spacy-transformers does (strided spans,
wordpieces aligned to tokens by character offsets, each token the mean of its
wordpieces, the output layer). Registered before it ran
(`eval/runs/persian-voice-ezafe-tagger-v1.json`), it gives the pipeline's
decision on every one of 3,355 words in 414 sentences and every tag of the 12
listening sentences, at 0.86 GB resident and about 30 ms a sentence.

In the app it loads for one text and is released, and is not started with less
than 1.5 GB free. A final h with no reading to go by is pronounced after ا or و
and in 33 listed words (توجه, شبیه, گره ...), silent otherwise. No ezafe joins
مدیر to عامل. For OmniVoice only, the normaliser now keeps أ and writes ۀ as
هٔ. Through the app at its default settings the 12 listening sentences give
the E text that was heard in 11 and differ in n12 only by that مدیر - a form
nobody has heard yet. Every other engine gets the text it got before.

## 17. Homograph vowels help; a sukun before adverbs does not pass

Step 3 went back to the errors the user heard in the app: an ezafe OmniVoice
added on its own before an adverb (تعاملات معمولاً), and homographs read the
other way (محرم, اسرار). Twelve sentences (the user's and eleven new, none in
HomoRich, SentenceBench or the adaptation data) with 23 of HomoRich's
homographs that the first vowel tells apart, in a 2 x 2 on top of the app's
ezafe: H, the first short vowel of each such homograph from Homo-GE2PE's
reading of the sentence; S, a sukun on a noun, adjective or number the tagger
gives no ezafe when an adverb follows. Registered before any front-end ran on
the sentences (`eval/runs/persian-voice-frontend-listening-v2.json`); rendered
in one Kaggle session (the first stopped on a path bug before the engine
loaded); heard blind by the user.

| | words marked wrong (of 242) | homographs wrong (of 23) | picked best | mean CER | held-out similarity |
|---|---|---|---|---|---|
| E (the app) | 27 | 14 | 1 | 2.4% | 0.599 |
| EH | 25 | 8 | 3 | 2.0% | 0.608 |
| ES | 27 | 15 | 3 | 2.1% | 0.602 |
| EHS | 25 | 8 | 4 | 2.5% | 0.608 |

**H is kept** (50 words against 54) and nothing flags it: CER +0.05 points,
similarity +0.007 [-0.003, +0.018]. It halves the homographs heard wrong.
**S ties** (52 against 52) and is dropped. Read beside it: on the words it
marked S fixed 7 and broke 1, while words it left moved 2 against 8 - with one
seed every changed text renders anew, and the registered count takes both.

What the words say. H fixed شکر (shokr), سر (sar) twice, both مهر, پر and
تن, and broke only the preposition به, all four times it marked it (بِه);
unmarked, به was never wrong. Five homographs marked right were still heard
wrong in every form (محرم, خلق, مسکن, عرق, سحر), and Homo-GE2PE's four wrong
readings (both کشتی, سرّ, the painkiller مسکن) fixed nothing. The listener
noted the sugar شکر said shekr even with its first vowel marked. And روی with
the app's written ezafe (رویِ) was heard wrong in 11 of 12 readings - every
form, so not H or S - where step 2's one رویِ was heard right; no plain form
was rendered here to say whether the kasra causes it.

## 18. The kasra on روی: the rule drops it, on one mark

Step 3 heard the app's رویِ wrong in 11 of 12 readings, in every form, and
could not say whether the written kasra caused it. Ten new sentences with 12
روی were each rendered twice (seed 42, mana_000): R1, the app's text (رویِ),
and R0, the same without that kasra. The check was registered before any
front-end ran on them (`eval/runs/persian-voice-rui-check-v1.json`), rendered
in one Kaggle session and heard blind, by a listener recorded as پارسا rather
than the one who heard steps 1 to 3. The page's database kept only an empty
start; the answers came as its answer code.

| | روی marked wrong (of 12) | words marked wrong (of 114) | picked best | mean CER |
|---|---|---|---|---|
| R1 (رویِ, the app) | 1 | 12 | 0 | 3.8% |
| R0 (روی) | 0 | 11 | 0 | 3.1% |

"No difference" was picked for all ten sentences; held-out similarity R0 − R1
is +0.005 [−0.021, +0.035].

**The rule drops the kasra.** R0 has fewer روی wrong (0 against 1) and no more
words wrong (11 against 12), and nothing flags it. One mark decides it: r03's
روی in R1, noted as said roy. Every other word was marked the same in both
forms. To this listener the kasra made almost no difference. Step 3's listener
heard royi in 11 of 12; this one heard the app's رویِ wrong once in 12. The
listener and the sentences both differ, and this check cannot say which
mattered.

Other errors were heard the same way in both forms:
- **An ezafe nobody wrote, on four words:** باران before گوش, پیرمرد before
  هر, حیاط before نقاشی, and دیوار before باغ را. For دیوار the tagger was
  right to give none. This is the same kind of error as the user's تعاملات
  معمولاً, which step 3's sukun did not stop.
- **Wrong vowels in ordinary words:**
  - طاقچه and نیمکت with a vowel added inside (طاقِچه, نیمِکَت)
  - بشقابش with e for o
  - زانویش as زانوییش
  - می‌نشیند without its last a
  - پنجهٔ پا said panj-e with a pause

**In the app**, the tagger's ezafe is no longer written on روی (`NO_EZAFE_ON`
in `src/voiceclone/text/ezafe_fa.py`). A kasra the writer puts there stays.
Through the app at its default settings, with the real tagger and Homo-GE2PE,
the ten sentences give the R0 that was heard (`scripts/check_rui_in_app.py`).

## 19. The model trained on Persian reads better, and the app now uses it

Four rounds on the text left the model's own errors:
- an ezafe it adds where none is written
- wrong vowels inside ordinary words

The Persian adaptation (section 12) had been measured only by similarity and
CER, and CER cannot see either error. Twelve new sentences were written to
hold both: 19 words after which no ezafe belongs, and 17 ordinary words whose
short vowels plain script does not show. The check was registered before any
front-end ran on them (`eval/runs/persian-voice-adapted-reading-v1.json`).
Three sets of weights read each sentence from the same app text:
- P: the published model
- A1: the adaptation's export at step 1,000, the one the sealed test measured
- A5: the export at step 5,000

One Kaggle session rendered them in about ten minutes. The user heard them
blind, with each form four times in each slot.

| | words marked wrong (of 114) | readings with none wrong | picked best | mean CER | held-out similarity |
|---|---|---|---|---|---|
| P, published | 15 | 3 of 12 | 1 | 4.8% | 0.590 |
| A1, step 1,000 | **12** | **5** | **4** | **3.7%** | 0.580 |
| A5, step 5,000 | 14 | 2 | 1 | 5.1% | 0.590 |

"No difference" was picked for the other six sentences.

**By the registered rule the app now reads with A1.** Both exports have fewer
words marked wrong than P and neither is flagged; A1 has fewer. Its similarity
on this voice is 0.010 lower, but the interval crosses zero (−0.025 to +0.005).
On the sealed test's 27 unseen speakers it matched the published model
(+0.001 [−0.004, +0.007]). Three words in 114, from one listener, one voice and
one seed, is a small difference. More training did not read better either: A5,
at five times the steps, sits between the two.

**Where the difference is: the ezafe the model adds.** By the notes, it was
heard on 8 words in P, 5 in A1 and 8 in A5. Four came in every form:
- پیر before درختان
- مدرسه before توپ
- کاغذی before پیشانی
- برق before صف

Every form also said the written کنارِ as kenareye. The vowels did not move:
کتلت, می‌پزد, چکش and کلاغ were wrong in all three. Both silent-h ezafes
(همسایهٔ, رانندهٔ) were right in all three.

**In the app.** OmniVoice reads with A1 (`engines.omnivoice.weights: auto`)
once `scripts/fetch_adapted_weights.py` has fetched it from the training
session's output (about 2 GB) and its sha256 matches the one the sealed test
and this check heard. On a machine without it, the app reads with the
published model and says so in the output's warnings. `weights: published`
keeps the old weights; `weights: adapted` refuses to start without the new
ones.

## 20. The user's news text: three of eight errors were the app's

On 25 October the user rendered a tech news paragraph, first with A1 and then
with the published model. It covered Apple's launch dates, Mark Gurman,
MacBook Pro, OLED, M6 and iPad mini 8. The user named eight words said wrong:
- اپل as epel
- جمعه as jam'e
- اکتبر as ektabar
- نوامبر as navambr
- OLED as olad
- مجهز as majahaz
- M6 as me shesh
- کند as kond

**Three came from the text the app wrote, and are fixed:**
- **اپل:** Homo-GE2PE read it as the epaulette (epol), and the app wrote اِپل.
- **M6:** the number was verbalised glued to its letter, as Mشش.
- **کند:** the first-vowel mark the app wrote, کُند, is how the adjective is spelled.

Now letters touching a number are parted from it. One to three letters before
a number are spelled as Persian letter names (اِم شش). Known units after a
number are named (8GB as هشت گیگابایت). اپل and کند are no longer marked.

**The other five are the model's, in both models:** the short vowels of
ordinary and borrowed words (جمعه, اکتبر, نوامبر, مجهز) and a Latin word
(OLED). News text with loanwords, names and numbers was never in a listening
test. The sentences of sections 15 to 19 are everyday narrative.

## 21. Vowels written on every word make news worse, even when every mark is right

The errors left in the user's news text were the model's own short vowels, so
the next check wrote them in. Eight new news-style sentences (138 words) were
read in a 2 x 2 (`eval/runs/persian-voice-news-vowels-v1.json`, registered
before any front-end ran):
- **text:** B, the app's text; or V, which adds the first short vowel of every
  word of three or more letters as Homo-GE2PE reads the sentence
- **weights:** P (published) or A (A1)

The sentences hold Western months, borrowed names and formal words. V's 60
marks all matched the first vowels written by hand before any front-end ran;
7 words were left unmarked. Heard blind by the user:

| | words marked wrong (of 138) | readings with none wrong | mean CER |
|---|---|---|---|
| PB | 13 | 2 of 8 | 5.2% |
| PV | 21 | 1 | 5.8% |
| AB | 14 | 2 | 5.0% |
| AV | 19 | 1 | 6.0% |

**V is not adopted.** It added 13 words wrong (40 against 27): 8 under P and 5
under A. Per word it broke 18 of the words it marked and fixed 6, with marks
that were all right. The notes say how: the model reads some marked vowels
long or otherwise wrong, for example:
- اَست said as est
- مِلی said as mili for melli
- وَرزشگاه said with vor-
- نُوامبر said as noovambr

Section 13 heard a damma come out as a long u; here it happens on common
words. Marks help where the model would otherwise choose wrongly between two
words (the homographs, section 17). On words it already reads, they are a
risk.

**The weights tie on news, 34 against 33,** so the app keeps A1 as the rule
says. Its edge on everyday narrative (section 19) does not show here. Left in
every form:
- دوازدهم and اکتبر
- نوامبر
- فوتبال said with an ezafe
- جشنواره and بهمن
- مردم, کولرها and پرمصرف

These are about one word in ten on news text.

## 22. ParsVoice-XTTS: some words right, the voice and the rest worse

**The model.** ParsVoice-XTTS is the most Persian-trained cloning model found
on 2026-10-25 (SOTA_ROADMAP section 4). It is XTTS-v2 adapted on about
2,000 h of Persian audiobooks. It was set against the app (OmniVoice A1 with
the app's text) on ten new news sentences, registered before any front-end
ran (`eval/runs/persian-voice-xtts-check-v1.json`). It was rendered on this
Mac's CPU from its weights alone: the published file is a 5.7 GB training
checkpoint, read memory-mapped once its pickle was seen to hold only tensors.
It used the fork it was trained with and its own notebook's settings.

**Scored on this Mac by one recogniser:**
- CER is 12.2% against the app's 4.3%.
- Held-out similarity is 0.544 against 0.587: -0.043 [-0.083, -0.010].

Both clear the registered harm lines, so it is flagged twice. Three of its ten
readings ran far past their text's length.

**Heard.** The user stopped after three sentences: its quality is very poor,
and its readings are easy to pick out, though it may say some words more
correctly. On those three it had 2 words wrong against the app's 4, and the
app was picked as the better reading twice of three.

**Decision.** It is not adopted. A model trained on twenty times this
project's Persian hours reads some words better and everything else worse.

## 23. Training on transcripts with the ezafe written stops the model adding one

**The lever.** The largest error left after section 22 was an ezafe the model
adds where none is written. Text-side marks could not stop it (sections 15, 17
and 21). The model learned from transcripts where the ezafe is never written,
so it guesses one. The E run repeats the adaptation (section 12) with one
change: every transcript rewritten through the app's ezafe tagger, 102,062
ezafes written into 45,907 of 66,023 texts. Same 128 h, same official recipe of
5,000 steps, registered before any transcript was marked
(`eval/runs/persian-voice-omnivoice-ezafe-train.json`). It took three Kaggle
sessions, about 22.5 GPU-hours. Its training and validation losses stayed
beside the first run's throughout: the loss cannot tell the two apart.

**The check.** Twelve new sentences were written after training: six news and
six everyday narrative, with 33 words after which no ezafe belongs. One
sentence was reordered before any render because the app's tagger wrote an
ezafe on one of its traps. Three sets of weights read the same app text:
- A1: the first adaptation at step 1,000, the app's weights until now
- E1: the E run at step 1,000
- E5: the E run at step 5,000

One Kaggle session rendered them. The user heard them blind, each form four
times in each slot.

| | words marked wrong (of 165) | readings with none wrong | picked best | trap words marked (of 33) | mean CER | held-out similarity |
|---|---|---|---|---|---|---|
| A1, the app until now | 16 | 3 of 12 | 0 | 11 | **2.6%** | **0.631** |
| E1, step 1,000 | 6 | 7 | 1 | 2 | 2.8% | 0.623 |
| E5, step 5,000 | **3** | **9** | **8** | **1** | 3.5% | 0.609 |

"No difference" was picked for the other three sentences.

**By the registered rule the app now reads with E1.** Both E exports have far
fewer words marked wrong than A1. E5 has the fewest, but its similarity on this
voice is 0.023 below A1's with the interval wholly below zero (−0.040 to
−0.005), so it is flagged and waits for the user's say. E1 is not flagged: CER
+0.3 points, similarity −0.009 [−0.022, +0.005].

**Where the difference is: the ezafe nobody wrote.** By the notes, A1 said one
on 10 words: nine traps (درصد, ایران, پیش, شیراز, شام, عصر, اتوبوس, راه,
مسافران) and the verb کرد (karde). E1 and E5 said none. What is left in the E
forms:
- short vowels inside words: بندرعباس in every form, ژاپنی and کوچکم in E1,
  کمربندهایشان in both
- an e inside a compound: نفس‌گیر (nafasegir) in E1, دیروقت (direveghat) in both

One listener, one voice and one seed, but 16 against 6 and 3, and 11 trap
words against 2 and 1, is the largest difference any check has heard.

**In the app.** OmniVoice reads with E1 (`engines.omnivoice.weights: auto`)
once `scripts/fetch_adapted_weights.py` has fetched it from the E run's first
session (`persian-voice-omnivoice-ezafe-a`, about 2 GB) and its sha256 matches
the one this check heard. The app's text already writes the tagger's ezafe,
the convention E was trained on.
