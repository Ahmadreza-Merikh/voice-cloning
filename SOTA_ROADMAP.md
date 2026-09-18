# Road to a defensible Persian voice-cloning result

What would have to be true to claim a state-of-the-art Persian zero-shot voice
clone, what is true now, and what each next step costs. Every statement is
marked:

* **[measured]** - produced by this project's own runs, with the file to check
* **[source]** - taken from a vendor's own card, repository or paper, checked
  live on 2026-09-18, with the revision or field named
* **[hypothesis]** - not established; written down so it can be tested or dropped

Nothing here is a plan to train yet. It is the decision material for choosing
what to train, on what, and how anyone would know it worked. An independent
review has not accepted any SOTA claim, and nothing below makes one.

## 1. The claim, and what would settle it

A defensible claim needs four things this project does not have yet:

1. a test set of speakers that was never used to choose anything (§3.2)
2. a design with a stated chance of detecting the difference being claimed,
   sized from the pilot's uncertainty rather than its point estimate (§3.1)
3. identity labels whose validity does not rest on the judging encoder (§3.2)
4. at least one blind human listening result (§3.4)

Until then every statement takes the form FINDINGS §8 now uses: an interval,
what it excludes, and how weak the design is.

## 2. Where the work stands

**[measured]** Six configurations on one development benchmark - nine speakers,
three sentences, one seed, one reference clip each (`eval/results/*.json`,
compared by `scripts/compare_runs.py`):

| run | held-out similarity | CER, and where it was scored | weights licence **[source]** |
|---|---|---|---|
| OmniVoice | 0.583 | 2.6%, Kaggle eval kernel | CC-BY-NC |
| MOSS-TTS v1.0 | 0.559 | 6.5%, this Mac (`moss-local`) | Apache-2.0; codec not checked |
| F5 continued + best-of-4 | 0.546 | 2.2%, this Mac | base CC-BY-NC |
| Fish S2 Pro | 0.537 | 9.5%, Kaggle Fish kernel | research only |
| Higgs TTS 3 (ported, fp16) | 0.531 | 2.9%, Kaggle Higgs kernel | research only |
| F5 baseline | 0.520 | 2.9%, this Mac | base CC-BY-NC |

CER is only comparable inside one scoring environment, because the recogniser
runs cpu/int8 on the Mac and cuda/float16 on a GPU (`engines/asr.py`). Inside
their environments: Fish is +6.4 points [+4.2, +8.6] against the baseline
re-scored in the same kernel; MOSS v1.0 is +3.7 [+1.7, +5.6] against the Mac
baseline; Higgs and OmniVoice have intervals that include zero.

**[measured]** On similarity, per speaker with a 95% t interval, one comparison
has an interval clear of both zero and the 0.0205 reference line: OmniVoice
over the F5 baseline, +0.064 [+0.025, +0.102]. OmniVoice over the best F5,
over Fish and over Higgs have intervals that exclude zero and overlap the line.
The design's power is low - about 59% for a true 0.04 difference, 38% at 0.03 -
so intervals that include zero are weak evidence either way.

**[measured]** The Higgs number is for a transformers port in fp16 on a T4, not
Boson's serving stack in bf16. Identical weights (sha256) and a CER near the
card's claim do not show that the port's generation loop or the precision
change leave quality unchanged.

## 3. The instrument

### 3.1 How big the test has to be

**[measured]** The speaker-level standard deviation of paired similarity
differences is about 0.048 across five engine pairs - estimated from nine
speakers, so its own 95% interval is roughly [0.032, 0.092]. Test speakers
needed for 80% power, two-sided alpha 0.05, exact noncentral t:

| true difference | spread 0.048 | spread 0.092 (pilot's upper bound) |
|---|---|---|
| 0.04 | 14 | 44 |
| 0.03 | 23 | 76 |
| 0.02 | 48 | 168 |

With three primary comparisons and a Bonferroni split (alpha 0.0167 each), 0.03
needs 30 to 102. Eighty percent power is a planning figure, not a guarantee: it
is the chance of an interval excluding zero *if* the true difference and the
spread are as assumed.

**Proposal.** Declare one primary comparison before building the test set -
for example "the chosen candidate against OmniVoice, one-sided, margin 0.03" -
and size the test for it at about 30 speakers, which covers the pilot spread up
to roughly 0.055. Every other comparison is exploratory, reported with a Holm
correction and labelled as such. If the test set's own spread comes out higher,
say so and report the interval; do not add speakers after looking.

**[measured]** Adding speakers is not free. Each new speaker needs new renders
from every engine under test. Measured on a T4, including scoring: OmniVoice 6
s a cell, Higgs 10 s, Fish 20 s, MOSS v1.0 50 s. Thirty speakers x twelve texts
x two seeds is 720 cells per engine: about 1.2 GPU-hours for OmniVoice, 10 for
MOSS v1.0. Kaggle's weekly T4 allowance is about 30 hours, so a four-engine
comparison is a planned week, not a rescoring job.

### 3.2 Development and test, kept apart

**[measured]** Every configuration decision so far - best-of-4, reference
choice, CFG, NFE steps, and every "which engine is best" - was made on the
current nine speakers. They are a **development set** and stay one. A test set
is new speakers, disjoint from development, scored once per final candidate,
and never used to choose a recipe, a setting or an engine.

**[measured]** `scripts/build_eval_corpus.py` selects speakers with the same
ECAPA-TDNN that scores the results. That makes comparisons conditional on
speakers who pass an ECAPA coherence test, and it can bias both the absolute
scale and the ranking. The test set must not repeat this. Swapping in a second
encoder for admission would not fix it either: an encoder is not evidence that
two clips are one person. Identity has to come from the source - published,
verified speaker ids, or recordings made for the purpose - with encoders used
at most to *flag* a speaker for a human to check, and every flag reported.

### 3.3 Texts

**[measured]** The within-speaker spread across the three current sentences is
large (for `cv07`, OmniVoice over baseline is +0.124, +0.016, +0.050), so text
is a real source of variance. Proposed test texts, twelve per speaker, two in
each category: formal written, conversational, numbers/dates/prices, proper
nouns and borrowed words, homographs and ezafe, and very short (< 5 words) or
long (> 25 words). None may appear in any prompt or in any development text.

### 3.4 Listening

**[hypothesis]** Cosine similarity is not a perceptual scale; whether a 0.03
difference is audible is unknown. A blind test is the only way to find out,
and it is also the only thing that could turn the 0.0205 reference line into a
justified decision bar - or retire it.

Design: for one speaker and one text, two renders from two systems in random
order beside a real recording of that speaker, three separate questions -
*same person as the recording?*, *natural?*, *any misread words?* - with the
real recording drawn from the evaluation pool, never the prompt. The tooling
belongs here; recruiting listeners needs the user, and no result is written
down until listening has actually happened.

## 4. Candidate bases

| base | licences **[source]** | measured | official fine-tuning path | keep or drop |
|---|---|---|---|---|
| OmniVoice | code Apache-2.0, weights CC-BY-NC | **[measured]** 0.583 | yes: `examples/run_finetune.sh`, JSONL `{id, audio_path, text, language_id}`, 5,000 steps at lr 5e-5 | keep if non-commercial is acceptable |
| MOSS-TTS-Local-Transformer-v1.5 | model, codec and code all Apache-2.0 (below) | **no** | not checked | measure next (§7) |
| MOSS-TTS-Local-Transformer v1.0 | model Apache-2.0; codec licence not checked | **[measured]** 0.559, CER +3.7 | not checked | superseded; kept as a reference point |
| F5-TTS continued | base CC-BY-NC | **[measured]** 0.546 | this project has one | control, not destination |
| Fish S2 Pro | research only | **[measured]** 0.537, CER +6.4 | out of scope | drop - worst Persian CER measured and the narrowest licence |
| Higgs TTS 3 | research only | **[measured]**, ported fp16 only | out of scope | drop for training; re-measure only through an official path |

"Unrestricted" is a claim about every component, so it is only made where
every component was checked. For MOSS v1.5 **[source]**:

* model `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` at `be7766a6`,
  Apache-2.0, 9.1 GB
* codec `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` at `f6e20e54`, Apache-2.0,
  8.5 GB in fp32 on disk, loaded as bf16 by the processor by default
* code `OpenMOSS/MOSS-TTS` at `934d6826`, Apache-2.0
* remote code read at those revisions: imports torch, torchaudio,
  transformers, safetensors and the standard library; no network, shell,
  pickle or eval

So among the candidates audited here, MOSS v1.5 is the one whose model, codec
and code were all checked and found Apache-2.0. That is not the same as "the
only commercial option": other models were not audited, and training data
licences (ParsVoice is CC-BY-NC) bind the result independently.

Exact revisions measured so far: `k2-fsa/OmniVoice` `c5fdb5cc`,
`OpenMOSS-Team/MOSS-TTS-Local-Transformer` `12aa734e`, `fishaudio/s2-pro`
`1de9996b`, the Higgs port `30f01593`, `SWivid/F5-TTS` `84e5a410`,
`Lumos675/F5_TTS_Persian` `cc322511`. Only Higgs pins its revision in config.

## 5. Data

**[source]** ParsVoice (`MohammadJRanjbar/ParsVoice`, card fields checked
2026-09-18): CC-BY-NC-4.0, gated behind an access request, 273 parquet shards.
The paper (arXiv 2510.10774) reports 2,200 hours, 1,815 speaker ids that were
*automatically inferred*, 1.36M segments from audiobooks, and a fine-tuned
XTTSv2 at naturalness MOS 3.6 and speaker-similarity MOS 4.0. The card body
could not be read without accepting the terms, which is the user's decision.

What that means for use:

* it cannot produce shippable weights, whatever the base
* its speaker ids are inferred, so it is not a source of identity ground truth
  for a test set (§3.2)
* **[hypothesis]** read audiobook speech should help pronunciation and prosody
  more than robustness to a phone in a room, which is the actual use case;
  phone recordings and conversational speech are not represented, and would
  need a separate source

## 6. A hypothesis worth testing

**[measured]** Every engine scores 0.11 to 0.16 higher against the clip it was
conditioned on than against the same speaker's other clips, and that is about
the size of its gap to the development set's same-speaker mean.

**[hypothesis]** Part of that gap is recording channel carried from the prompt
into the render. Other explanations - continuation of the prompt's prosody,
level or content - have not been separated from it, and nothing shows the
mechanism is the same in every engine.

Two ways to probe it:

1. **Multi-reference conditioning, no training.** Condition on clips from two
   different recordings instead of one. It needs a fixed manifest first
   (`eval/manifests/`, §8): prompt clips and evaluation clips fully disjoint;
   one evaluation pool shared by every condition; single- and multi-reference
   prompts with *equal* total speech, and a separately named condition for a
   larger budget. An improvement would be consistent with the hypothesis
   without proving it - more diverse prompt material helps for other reasons
   too.
2. **Two-recording training pairs.** Prompt and target from different
   recordings of one speaker, so copying the prompt's channel earns nothing.
   It needs identity-validated multi-recording data, which is §3.2's problem.

## 7. MOSS-TTS v1.5: what the official recipe is

**[source]** From the model card and `pyproject.toml` at the revisions above:

* `torch==2.9.1+cu128`, `torchaudio==2.9.1+cu128`, `torchcodec==0.8.1`,
  `transformers==5.0.0`, loaded with `trust_remote_code=True`
* `dtype = torch.bfloat16 if device == "cuda" else torch.float32`
* FlashAttention 2 only on compute capability 8.0+; otherwise SDPA on CUDA -
  an official fallback, so a T4 uses SDPA without leaving the recipe
* 48 kHz stereo output, 12 codebooks fixed (`n_vq_for_inference` must not change)
* recommended sampling: audio_temperature 1.7, top_p 0.8, top_k 25, repetition
  penalty 1.0, max_new_tokens 4096
* a language tag is recommended when known. The examples use English names
  ("French", "Chinese"); the Persian string is not documented, so "Persian" is
  an inference by analogy and is recorded as one

**[measured]** On a T4 the question is bf16, which Turing has no unit for. The
v1.0 run's own log shows it loaded `torch.bfloat16` with transformers 5.0.0 on
the T4 and completed 27 cells, so emulated bf16 has run this model family
there. That is the official precision, run slowly - not a precision change.
The plan keeps it: model on one T4, codec (bf16, about 4.3 GB) on the other,
and if bf16 fails on Turing, the run stops and the blocker is written down
rather than switched to fp16 without the user.

Audio: rendered 48 kHz stereo is kept on disk as rendered. The harness reads a
render as mono by averaging its channels (`Audio.to_mono`, `audio/io.py`), and
the similarity scorer resamples to 16 kHz (`EMBEDDING_SAMPLE_RATE`,
`evaluation/similarity.py`) - the same path every engine's audio takes. A
stereo engine is therefore scored on its channel mean, which is worth stating
next to its numbers.

## 8. Order of work

1. Build and validate the multi-reference manifest on the development set.
   No GPU.
2. Smoke-test MOSS v1.5 on a T4 at the official recipe, recording load time,
   per-cell time and memory on each card. Then, only if it renders valid
   audio, the 27-cell development benchmark.
3. Run the multi-reference conditions on the development set for the engines
   that accept the prompt budget, with the manifest fixed beforehand.
4. Decide commercial or not (the user's question) and declare the primary test
   comparison.
5. Build the independent test set: about 30 identity-validated speakers,
   twelve texts each, disjoint from development. Budget it (§3.1).
6. Score each final candidate on it once.

## 9. Open questions for the user

* Commercial use or not? It selects the base, and it binds the data too.
* A ParsVoice access request, if wanted, is theirs to make.
* Would they organise a listening test, even with five listeners?
