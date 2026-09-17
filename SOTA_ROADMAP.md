# Road to a defensible Persian voice-cloning result

What would have to be true to claim a state-of-the-art Persian zero-shot voice
clone, what is true now, and what each next step costs. Every line is marked:

* **[measured]** - produced by this project's own runs, with the file to check
* **[source]** - taken from a vendor's own card, repository or paper, checked
  live on 2026-09-18, with the quote or field named
* **[hypothesis]** - not established; written down so it can be tested or dropped

Nothing here is a plan to train yet. It is the decision material for choosing
what to train, on what, and how anyone would know it worked.

## 1. The claim, and what would settle it

A defensible claim needs three things this project does not yet have:

1. an instrument that can see the difference being claimed (§3)
2. a test set that was not used to choose anything (§3)
3. at least one human listening result, collected blind (§3.4)

Until then the honest form of every statement is the one §8 of FINDINGS.md now
uses: a direction, an interval, and the size of the effect the design could not
have seen.

## 2. Where the work actually stands

**[measured]** Six configurations, one benchmark, 27 cells each
(`eval/results/*.json`, compared by `scripts/compare_runs.py`):

| run | held-out | share of ceiling | CER | weights licence |
|---|---|---|---|---|
| OmniVoice | 0.583 | 84.8% | 2.6% | CC-BY-NC **[source]** |
| MOSS-TTS (older variant) | 0.559 | 81.2% | not transcribed | Apache-2.0 **[source]** |
| F5 continued + best-of-4 | 0.546 | 79.2% | 2.2% | base is CC-BY-NC **[source]** |
| Fish S2 Pro | 0.537 | 77.7% | 9.5% | research only **[source]** |
| Higgs TTS 3 | 0.531 | 77.3% | 2.9% | research only **[source]** |
| F5 baseline | 0.520 | 75.2% | 2.9% | base is CC-BY-NC **[source]** |

**[measured]** Clustered by speaker, one comparison in the whole project clears
both the non-zero bar and the instrument-noise bar: OmniVoice over the F5
baseline, +0.064 [+0.025, +0.102]. OmniVoice over the best F5 configuration,
over Fish and over Higgs are directions with intervals that reach down into the
noise. Nothing here shows any two engines are equal.

**The licence finding changes the shape of the problem.** **[source]** The model
that leads this table is CC-BY-NC, as is the F5 base the project has been
continuing, as is ParsVoice, as are Fish and Higgs. The only unrestricted
weights among the candidates are MOSS-TTS (Apache-2.0) - and the MOSS variant
measured here is the older one. So:

* if the result may be non-commercial, the base to beat is OmniVoice
* if it must be shippable, the whole table collapses to MOSS-TTS, and the
  version that should be measured has not been

That question is the user's to answer and it decides most of what follows.

## 3. The instrument, and what it cannot currently see

### 3.1 Sample size

**[measured]** The speaker-level standard deviation of paired differences is
about 0.048 across five engine pairs. With a paired t test at 80% power:

| difference to detect | speakers needed |
|---|---|
| 0.05 | 8 |
| 0.04 | 12 |
| 0.03 | 21 |
| 0.02 | 46 |

The corpus has nine. Every unresolved comparison in §2 is between 0.01 and
0.05, which is exactly the band nine speakers cannot reach. **Widening the
corpus to roughly 24 speakers is the single change that would make the next
comparison decisive**, and it costs no GPU time - only corpus building and
re-scoring.

### 3.2 What the corpus is, and what it is not

**[measured]** Nine speakers, 45 clips: eight from a Common Voice 26 Persian
re-upload, one from Mana-TTS. Reference clips are 8.7-9.9 s.

**[measured]** `scripts/build_eval_corpus.py` selects speakers with the same
ECAPA-TDNN that scores the results, keeping a speaker only when their worst
self-pair beats their best cross-pair. That is selection on the outcome
variable: it lifts the ceiling, lowers the floor, and makes "84.8% of the human
ceiling" a statement about speakers who passed an ECAPA test. The next corpus
should decide inclusion on evidence that is not the judge - published speaker
metadata, or a second encoder of a different family (WavLM-based, say) used only
for admission.

**[measured]** Every configuration decision in this project - best-of-4,
reference selection, CFG strength, NFE steps - was made on these same nine
speakers, and the headline numbers are reported on them too. There is no held-out
split. The next corpus needs one: a development half for choosing anything, and
a test half that is scored once per candidate and never used to select.

### 3.3 What the texts cover

**[measured]** Three sentences, all written register. The within-speaker spread
across them is large - for `cv07` the OmniVoice-over-baseline difference is
+0.124, +0.016, +0.050 on the three - so text choice is a real source of
variance, not a formality.

Proposed set, six categories, five sentences each:

| category | why |
|---|---|
| formal written | the current three; continuity with every past run |
| conversational | the actual use case, and no engine has been tested on it |
| numbers, dates, prices | Persian numeral reading is a known failure mode |
| proper nouns, borrowed words | where a grapheme model has least support |
| homographs and ezafe | already the design principle of s1-s3, extended |
| very short (< 5 words) and long (> 25 words) | duration models fail at both ends |

### 3.4 Listening

**[hypothesis]** Nothing in this project has been listened to systematically,
and cosine similarity is not a perceptual scale. A blind test is the only way to
learn whether a 0.03 difference is audible at all - which is also what would
justify or retire the 0.0205 threshold as a decision bar.

Minimum viable design: pairs of renders from two systems for the same speaker
and text, presented in random order beside one real recording of that speaker,
three questions scored separately - *which sounds more like the same person*,
*which sounds more natural*, *did either misread the text*. Tooling belongs in
this repository; recruiting listeners does not happen without the user.

## 4. Candidate bases

| base | licence **[source]** | measured | official fine-tuning path | keep or drop |
|---|---|---|---|---|
| OmniVoice 3.3 GB | code Apache-2.0, weights CC-BY-NC | **[measured]** best here | yes - `examples/run_finetune.sh`, JSONL of `{id, audio_path, text, language_id}`, 5,000 steps at lr 5e-5 against 300,000 from scratch **[source]** | keep if non-commercial is acceptable |
| MOSS-TTS-Local-Transformer-v1.5, 9.1 GB | Apache-2.0 | **no** | not checked yet | measure before anything else if the result must be shippable |
| MOSS-TTS-Local-Transformer, 6.1 GB | Apache-2.0 | **[measured]** 0.559 | not checked | superseded by v1.5; keep only as the measured reference point |
| F5-TTS continued | base CC-BY-NC | **[measured]** 0.546 | this project already has one | keep as the control, not as the destination |
| Fish S2 Pro | research only | **[measured]** 0.537, CER 9.5% | out of scope | drop - worst Persian pronunciation measured and the most restrictive licence |
| Higgs TTS 3 | research only | **[measured]** 0.531 | out of scope | drop for training; keep the engine, since it is the only one whose vendor claims Persian in a top tier and it may be worth re-measuring on a stronger instrument |

**[source]** Exact revisions of what was measured, for reproducibility:
`k2-fsa/OmniVoice` `c5fdb5cc`, `OpenMOSS-Team/MOSS-TTS-Local-Transformer`
`12aa734e`, `fishaudio/s2-pro` `1de9996b`, the Higgs port `30f01593` (weights
sha256 identical to `bosonai/higgs-tts-3-4b`), `SWivid/F5-TTS` `84e5a410`,
`Lumos675/F5_TTS_Persian` `cc322511`. Only the Higgs engine currently pins its
revision in config; the others should.

## 5. Data

**[source]** ParsVoice (`MohammadJRanjbar/ParsVoice`, card checked 2026-09-18):
CC-BY-NC-4.0, gated behind an access request, 273 parquet shards, size category
1M-10M. The paper (arXiv 2510.10774) reports 2,200 hours, 1,815 automatically
inferred speaker ids, 1.36M aligned segments, sourced from **audiobooks**, with
a fine-tuned XTTSv2 reaching naturalness MOS 3.6 and speaker-similarity MOS 4.0.

Three consequences:

* **[source]** the licence is non-commercial, so training on it cannot produce
  shippable weights whatever the base licence is
* **[hypothesis]** audiobook speech is read, performed and cleanly recorded;
  the target use is a person recording themselves on a phone or laptop. Data
  like this should improve pronunciation and prosody and do little for channel
  robustness, which §6 argues is the actual weakness
* **[measured]** access needs the user to accept terms and supply a token; no
  account action has been taken on their behalf

**[hypothesis]** What the target actually needs is several recordings of the
same speaker in *different* channels. Common Voice has that structurally - one
contributor, many clips, plausibly different devices - and is CC0 upstream.
Whether the pairing is real is exactly the assumption §9 of FINDINGS withdrew,
so it would have to be established rather than assumed.

## 6. The hypothesis worth testing

**[measured]** Every engine loses most of its score to the same thing: a render
scores 0.11-0.16 higher against the clip it was conditioned on than against the
same speaker's other recordings. The gap between a clone and the human ceiling,
0.10-0.16 depending on engine, is the same size as the channel term.

**[hypothesis]** Therefore the limiting factor is not "the model does not know
this voice" but "the model reproduces the prompt's channel along with the
voice", and a training objective that separates them would move the number that
matters more than more data of the same kind would.

Two concrete forms, both testable and neither novel enough to assume:

1. **Two-channel pairing.** Train on pairs where the prompt and the target are
   different recordings of the same speaker, so reproducing the prompt's channel
   earns nothing. Needs speaker-linked multi-recording data, which is §5's open
   question.
2. **Multi-reference conditioning.** Condition on two or three clips of the
   speaker at once. Cheap to test *without any training at all*: the benchmark
   already holds four clips per speaker, and an engine that accepts a longer or
   concatenated prompt can be measured this week.

Form 2 should be run first precisely because it costs nothing and would give the
hypothesis its first evidence. **[hypothesis]** If multi-reference conditioning
moves held-out similarity, channel entanglement is real and form 1 is worth GPU
time; if it does not, the hypothesis is weakened before anything is trained.

## 7. Resources, and what needs asking first

**[measured]** What is already known about cost on free hardware: OmniVoice
renders 27 cells in about 25 minutes on a T4; Higgs loads in 17 s and renders at
about twice real time; Fish needs both T4s (transformer 8.5 GiB, codec 3.9 GiB)
and takes 10 minutes for 27 cells. Kaggle allows about 30 GPU-hours a week on
T4 x2.

**[source]** OmniVoice's own fine-tuning example assumes two GPUs and 5,000
steps; it states no VRAM figure, and the batch size lives in
`config/train_config_finetune.json`, which has not been read yet. Estimating the
cost properly means reading that config and running one smoke step - not
guessing.

Standing constraints, unchanged: no new spend, no GPU purchase, and no new
upload of the user's private audio without the user saying so. A full-quality
training run goes to remote hardware sized for the recipe; the recipe is not
shrunk to fit a laptop or a T4.

## 8. Order of work

1. Decide commercial or non-commercial. It selects the base and it is not a
   technical question.
2. Measure `MOSS-TTS-Local-Transformer-v1.5` on the current benchmark. Cheap,
   and it is the only unrestricted candidate.
3. Multi-reference conditioning on the existing corpus. No training, and it is
   the first evidence for or against §6.
4. Widen the corpus to about 24 speakers with a development/test split and the
   six text categories. This is what makes step 5 decisive.
5. Re-measure the surviving candidates on the widened set, once each.
6. Only then choose a base and a recipe, read its training config, size the
   hardware, and ask.

## 9. Open questions for the user

* Commercial use or not?
* Is a ParsVoice access request something they want to make themselves?
* Is a listening test something they are willing to organise, even with five
  listeners?
