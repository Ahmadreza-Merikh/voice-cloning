# Road to a defensible Persian voice-cloning result

What would have to be true to claim a state-of-the-art Persian zero-shot voice
clone, what is true now, and what each next step costs. Every statement is
marked:

* **[measured]** - produced by this project's own runs, with the file to check
* **[source]** - taken from a vendor's own card, repository or paper, checked
  live on 2026-09-18, with the revision or field named
* **[hypothesis]** - not established; written down so it can be tested or dropped

It is the decision material for choosing what to train, on what, and how
anyone would know it worked; the first training it plans is the Persian
adaptation of OmniVoice by its official recipe (§8), which has not run yet. An
independent review has not accepted any SOTA claim, and nothing below makes
one.

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

**[measured]** Seven configurations on one development benchmark - nine speakers,
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
| MOSS-TTS v1.5, official recipe | 0.507 | 10.9%, Kaggle v1.5 kernel | Apache-2.0, codec and code too |

CER is only comparable inside one scoring environment, because the recogniser
runs cpu/int8 on the Mac and cuda/float16 on a GPU (`engines/asr.py`). Inside
their environments: Fish is +6.4 points [+4.2, +8.6] against the baseline
re-scored in the same kernel; MOSS v1.5 +7.7 [+2.8, +12.7] likewise; MOSS
v1.0 is +3.7 [+1.7, +5.6] against the Mac baseline; Higgs and OmniVoice have
intervals that include zero.

**[measured]** On similarity, per speaker with a 95% t interval, one comparison
has an interval clear of both zero and the 0.0205 reference line: OmniVoice
over the F5 baseline, +0.064 [+0.025, +0.102] - and, since MOSS v1.5 was
measured, OmniVoice over it, +0.077 [+0.032, +0.122]. OmniVoice over the best F5,
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
differences - OmniVoice against each of the five other engines measured on all
27 cells - pools to 0.050, and because nine speakers estimate it loosely its
95% interval is 0.034-0.096 (`scripts/power_design.py`, output in
`eval/manifests/power_design_v1.json`; the five pairs share their speakers, so
the interval uses one pair's 8 degrees of freedom).

A superiority claim carries two numbers that must be kept apart:

* the **margin** m that the null hypothesis is stated against, H0: Δ ≤ m.
  m = 0 asks whether the candidate is better at all; m = 0.0205 whether it
  clears the practical reference line;
* the **true effect** δ that the design is sized for - the difference assumed
  to exist when asking how often the interval's lower end would clear m.

Power depends on δ - m. This section's earlier proposal, "one-sided, margin
0.03", sized from the row for a true difference of 0.03, mixed the two: with
δ = m the chance of success equals alpha whatever the number of speakers.

Speakers for 80% power, one-sided alpha 0.025 - the lower end of a two-sided 95%
interval - by exact noncentral t:

| margin m | true effect δ | spread 0.050 | spread 0.096 (upper end) |
|---|---|---|---|
| 0 | 0.03 | 24 | 82 |
| 0 | 0.05 | 10 | 31 |
| 0.0205 | 0.04 | 54 | 191 |
| 0.0205 | 0.05 | 25 | 85 |
| 0.03 | 0.03 | impossible | impossible |
| 0.03 | 0.06 | 24 | 82 |

Eighty percent is a planning figure, not a guarantee: it is the chance of
success *if* the true difference and the spread are as assumed.

**Declared on 2026-09-19, before any test render.** The primary comparison is
the Persian adaptation of OmniVoice minus OmniVoice as published, on held-out
similarity, paired per speaker: H0 Δ ≤ 0 (m = 0), one-sided alpha 0.025,
planned for a true effect of δ = 0.03, on the primary test set: the 28
Neyshekar test speakers that have a reference and four held-out clips (§3.2) -
87% power at spread 0.050 and 36% at 0.096. The 30 Common Voice test speakers
are a secondary test set, because OmniVoice has very probably heard them.
(Amended the same day, still before any test render, when Neyshekar was found:
the first version of this paragraph named the Common Voice set.) Secondary, reported
and labelled as such: whether the lower end also clears m = 0.0205 (85% power
with 28 speakers only if δ = 0.05 and the spread is 0.050). CER is a guard, not a target: the
adapted model's CER against the published model's, scored in one session, is
reported with its interval, and a CER interval above zero counts against the
adaptation whatever its similarity. Every other comparison is exploratory,
with a Holm correction. The spread between a model and its own adaptation may
not match the spread between different engines; if the test's own spread is
higher, the interval is reported as it is and no speakers are added after
looking.

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

**[measured] The split, drawn 2026-09-19** by `scripts/build_adaptation_splits.py`
from source metadata only - no audio read, no encoder - into
`eval/manifests/adaptation_splits_v1.jsonl.gz` (one line per clip, with the
source's own clip and speaker ids) and its `.summary.json`:

| source | split | clips | speakers | hours |
|---|---|---|---|---|
| Common Voice 26 (cleaned) | train | 33,148 | 3,703 | 41.70 |
| | valid | 365 | 40 | 0.53 |
| | test (secondary, sealed) | 427 | 30, 15 per self-reported gender | 0.58 |
| | test reserve (sealed) | 127 | 10 | 0.18 |
| | development speakers, excluded | 59 | 8 | 0.10 |
| | train clips dropped for shared text | 8 | 7 | 0.01 |
| Neyshekar v6 | train | 54,351 | 132 | 86.16 |
| | valid (its validation split) | 1,886 | 26 | 3.15 |
| | test (primary, sealed; its test split) | 2,149 | 30, 28 with the clips a test needs | 3.88 |
| | train clips dropped for shared text | 3,893 | 98 | 5.84 |

No speaker is in two splits; no training text equals, after CER
normalisation, the text of any development, valid or test clip; no training
text contains or is contained in a benchmark sentence. What this cannot rule
out: one person behind two accounts, or recording on both platforms. And the
Common Voice test speakers are very probably in OmniVoice's own training data
(FINDINGS §9), which is why Neyshekar's are primary.

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
| OmniVoice | code Apache-2.0, weights CC-BY-NC | **[measured]** 0.583 | yes: `examples/run_finetune.sh`, JSONL `{id, audio_path, text, language_id}`, 5,000 steps at lr 1e-5 (§8; the README's 5e-5 is stale) | the adaptation candidate (§8) |
| MOSS-TTS-Local-Transformer-v1.5 | model, codec and code all Apache-2.0 (below) | **[measured]** 0.507 at the official recipe; CER +7.7 over the baseline in its session | not checked | below OmniVoice and v1.0 on similarity here; its "continuation" mode, with the reference transcript, is untested - kept in reserve, not run for the licence alone |
| MOSS-TTS-Local-Transformer v1.0 | model Apache-2.0; codec licence not checked | **[measured]** 0.559, CER +3.7 | not checked | superseded; kept as a reference point |
| F5-TTS continued | base CC-BY-NC | **[measured]** 0.546 | this project has one | control, not destination |
| Fish S2 Pro | research only | **[measured]** 0.537, CER +6.4 | out of scope | drop - CER +6.4 over its session's baseline, and the narrowest licence |
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

**[source]** Public Persian speech, surveyed on the Hugging Face Hub on
2026-09-19 (search by language, task and name; each candidate's card, licence
field, gating and files read at its current revision). "In OmniVoice" means
listed in its paper's complete list of training data (arXiv 2604.00688v3,
Appendix A).

| source | licence, access | size | speaker ids | in OmniVoice | use here |
|---|---|---|---|---|---|
| Common Voice 26 Persian, cleaned (`pymmdrza/...-26.0-Persian-Clean` @ `142e9bb4`) | CC0, ungated | 34,134 clips, 43.1 h, 3,791 speakers, 24 kHz | `client_id` | yes, very probably (Common Voice) | train, valid, secondary test |
| Neyshekar v6 (`shekar-ai/neyshekar` @ `30696ada`; arXiv 2609.14542) | CC0, ungated; card forbids identifying speakers | 62,279 clips, 99.0 h, 190 speakers, 16 kHz read speech, 24% informal | anonymous stable ids, speaker-disjoint official splits | not listed | train, valid, **primary test** |
| Common Voice Persian, full (official: Mozilla Data Collective; ungated mirrors such as `fsicoli/common_voice_22_0`) | CC0; official download needs an account and terms | 372 validated h, 4,600+ speakers | `client_id` | yes | not used: the user's terms to accept, and it is what OmniVoice trained on |
| FLEURS `fa_ir` (`google/fleurs`) | CC-BY-4.0, ungated | about 10 h of training data per language (card) | none | yes | not used: no speaker ids |
| YODAS2, Sidon-denoised (`sarulab-speech/yodas2_sidon`, `fa000`) | CC-BY-3.0, ungated | 9.3 GB | none (YouTube videos) | not as Persian (its YODAS-Granary covers European languages) | possible later training data; not a test source |
| Ganjoor recitations (`Reza2kn/ganjoor-recitations`) | CC-BY-3.0, ungated | 30,133 recitations, 1,276 h, full-length | reciter per recording | not listed | possible later; poetry, unsegmented |
| WorldSpeech `fa_ir` (`disco-eth/WorldSpeech`) | CC-BY-NC-4.0, ungated | 27.6 h, broadcast and parliament | not checked | not listed | not used yet |
| Persian accents benchmark (`MR3z4/...`) | CC-BY-NC-SA-4.0, ungated | 279 files, 4.4 h, overlapping speech | not checked | not listed | evaluation only, if ever |
| ParsVoice (`MohammadJRanjbar/ParsVoice`) | CC-BY-NC-4.0, **gated** | 2,177.9 h, 1,803 *inferred* speakers, audiobooks | inferred | not listed | the user's terms to accept |
| Persian-Farsi-Speech (`Thomcles/...`) | CC-BY-4.0, **gated** | 417 h TTS subset, merged from several sources | not checked | not listed | the user's terms to accept |
| Mana-TTS (`MahtaFetrat/Mana-TTS`) | CC0, ungated | 114 h, one speaker | one | not listed | never: it is development speaker `mana` |
| Telegram audiobooks (`Reza2kn/telegram-audiobook-chizzled`) | "other"; its card says no licence or consent is established | 14,430 h | none | - | not used: not permitted |

What the adaptation trains on, then, is the two ungated CC0 sources with
speaker ids, split as in §3.2: 127.9 hours from 3,835 speakers. The largest
Persian TTS corpus, ParsVoice, and the 417-hour `Thomcles` set are gated, and
accepting their terms is the user's decision; neither is needed to start.

**[source]** ParsVoice, for the record: the card reports 2,177.9 hours, 1,803
automatically inferred speaker ids and 1,877 source audiobooks, FLAC 16 kHz;
annotations CC-BY-NC-4.0. What that means for use:

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
   (`eval/manifests/`, §9): prompt clips and evaluation clips fully disjoint;
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
* a language tag is recommended when known, and for Persian it is the string
  "Persian (Farsi)": the card's language table, and the official v1.5 app's
  `LANGUAGE_TAG_CHOICES` (`clis/moss_tts_local_v1.5_app.py` at `934d6826`),
  which puts it into the prompt unchanged. The first full run used "Persian",
  an analogy with the card's examples, before that was found
* the transformers processor has no field for the reference transcript. The
  official app and the SGLang-Omni server also offer a "continuation" mode that
  takes one, "for better speaker similarity" - a second official mode, not
  measured here

**[measured]** On a T4 the question is bf16, which Turing has no unit for. The
v1.0 run's own log shows it loaded `torch.bfloat16` with transformers 5.0.0 on
the T4 and completed 27 cells, so emulated bf16 has run this model family
there. That is the official precision, run slowly - not a precision change.
The plan keeps it: model on one T4, codec (bf16, about 4.3 GB) on the other,
and if bf16 fails on Turing, the run stops and the blocker is written down
rather than switched to fp16 without the user.

**[measured]** Loading the processor under transformers 5.0.0 logs that the
tokenizer has "an incorrect regex pattern" and suggests `fix_mistral_regex=True`.
It is a false alarm for this model: the check in transformers 5.0.0
(`TokenizersBackend._patch_mistral_regex`) fires for any local tokenizer whose
`config.json` lacks `transformers_version`, and MOSS v1.5's does; the tokenizer
is Qwen2's, not Mistral's. The suggested regex leaves all nine Persian test
sentences - the benchmark's five and four with digits, half-spaces and
quotation marks - token-for-token identical, and changes only mixed-case Latin
words ("iPhone" becomes "i" + "Phone"), away from how the model was trained.
Under 5.0.0 the flag also raises a TypeError. So the official behaviour -
warning ignored - is kept (`scripts/check_moss15_tokenizer.py`).

**[measured]** Kaggle's images move: on 2026-09-18 the main kernel's image had
torch 2.9.1+cu128 and the probe kernel's had 2.10.0+cu128. The notebook
therefore installs torch, torchaudio, transformers and safetensors at the
official versions - everything the remote code imports - stops on any
difference, and writes every package against its official version to
`environment.json`. The pyproject's other pins keep the image's versions: they
are the repository's own tooling, and torchcodec serves only `torchaudio.load`,
which the worker does not call.

Audio: rendered 48 kHz stereo is kept on disk as rendered. The harness reads a
render as mono by averaging its channels (`Audio.to_mono`, `audio/io.py`), and
the similarity scorer resamples to 16 kHz (`EMBEDDING_SAMPLE_RATE`,
`evaluation/similarity.py`) - the same path every engine's audio takes. A
stereo engine is therefore scored on its channel mean, which is worth stating
next to its numbers.

## 8. OmniVoice: the official adaptation recipe

**[source]** `k2-fsa/OmniVoice` at `08be0b4c` (master, 2026-08-24; package
0.2.1), read file by file on 2026-09-19:

* the entry point, `examples/run_finetune.sh`: stage 0 tokenises a JSONL of
  `{id, audio_path, text, language_id}` with
  `omnivoice.scripts.extract_audio_tokens` and `eustlb/higgs-audio-v2-tokenizer`
  into WebDataset shards; stage 1 runs `accelerate launch --num_processes 2
  -m omnivoice.cli.train` on two GPUs
* `examples/config/train_config_finetune.json`: initialise from
  `k2-fsa/OmniVoice`; learning rate **1e-5**; AdamW, weight decay 0.01; cosine
  schedule with 1% warmup; 5,000 steps; 8,192 tokens per GPU per step, no
  gradient accumulation; gradient norm clipped at 1.0; bf16 mixed precision
  with TF32; the language tag in 80% of prompts (`language_ratio` 0.8); prompt
  ratio 0-0.3, mask ratio 0-1, condition drop 0.1, codebook weights
  [8, 8, 6, 6, 4, 4, 2, 2]; seed 42; an evaluation loss and a checkpoint every
  500 steps, all kept
* the attention backend: `flex_attention` by default, which `docs/training.md`
  says needs "a compatible GPU (e.g. NVIDIA Ampere or newer)"; for others the
  repository ships `train_config_finetune_sdpa.json` - the same hyperparameters
  with SDPA attention and length-grouped padded batches (samples of 50-2,000
  tokens, at most 64 to a batch, at most 8,192 padded tokens). A T4 is Turing,
  so the SDPA configuration is the official path there, as SDPA was for MOSS
  v1.5
* **the learning rate.** `examples/README.md` still tabulates 5e-5 for
  fine-tuning, and this roadmap copied it. The config the script runs says 1e-5.
  The history settles it: the initial commit (`c8c0a625`, 2026-04-01) had
  5e-5 in both; `a65149e0` ("Support training with SDPA", 2026-04-22) changed
  the config to 1e-5 and added the SDPA config at 1e-5, and the README table
  was never updated. What the official script executes is 1e-5, so that is the
  recipe; the README's 5e-5 is documentation drift, recorded, not followed
* the runtime, from `uv.lock`: torch 2.8.0+cu128, torchaudio 2.8.0+cu128,
  transformers 5.3.0, accelerate 1.13.0, webdataset 1.0.2, safetensors 0.7.0,
  tokenizers 0.22.2, huggingface-hub 1.7.2. DeepSpeed is not in the lock,
  though the trainer can use it
* the base: `k2-fsa/OmniVoice` at `c5fdb5cc`, the revision every OmniVoice
  number here was measured on - 2.45 GB of float32, 612M parameters on a
  Qwen3-0.6B backbone; CC-BY-NC
* the audio tokenizer: `eustlb/higgs-audio-v2-tokenizer` at `528e871c`, whose
  `model.safetensors` has the same sha256 (`fe7c5e87...`) as the copy bundled
  with OmniVoice's weights, so training tokens and inference decoding use one
  tokenizer; the bundled copy carries the Boson Higgs Audio 2 Community License

**What this project changes, and why none of it is a quality change:**

* the data: the two CC0 sources of §5, split as in §3.2, `language_id` "fa"
  (Persian's id in OmniVoice's language table)
* the SDPA configuration, because a T4 is not Ampere - unless the probe shows
  `flex_attention` running on it
* the base and the tokenizer loaded from local snapshots of the pinned
  revisions instead of their moving names: the same weights, pinned
* checkpoint retention: ten checkpoints of about 10 GB each (weights plus
  optimizer state) do not fit Kaggle's 20 GB of output, so the latest full
  state is kept for resuming, plus a float16 copy of the weights - which is how
  inference loads them - every 1,000 steps
* session boundaries: a Kaggle session is at most 12 hours, so a run that does
  not fit stops cleanly after an official checkpoint and the next session
  resumes through the official `resume_from_checkpoint`. The official resume
  restarts the data order at the first epoch, so an interrupted run is not
  bit-identical to an uninterrupted one; that is the official trainer's
  behaviour, recorded rather than patched
* memory: if 8,192 tokens per T4 does not fit, the only changes allowed are
  gradient checkpointing (recomputation) and DDP gradient bucket views, which
  change memory, not arithmetic. Fewer tokens per step, gradient accumulation
  in their place, 8-bit optimisers, LoRA, fp16 or fewer steps are recipe
  changes and need the user

**The probe first** (`kaggle/build_omnivoice_adapt_kernel.py`; one session,
behind the model-free runner check). It installs the pinned runtime; tokenises
a slice of the training split with the official script - never a test clip;
loads the pinned base; runs a few optimizer steps at the full 8,192 tokens per
GPU, first with the default `flex_attention` configuration and then with the
SDPA one, recording peak memory per GPU, seconds per step and the loss; and
only if the official configuration does not fit, the same again with the
memory techniques above, checking that the first-step loss is unchanged by
them. Its numbers decide:

* how long the full run takes: 5,000 steps at the measured rate, plus
  tokenising 128 hours
* whether that fits the free quota (about 30 GPU-hours a week) in sessions of
  up to 12 hours - recorded before the run is sent
* if a T4 pair cannot hold the recipe: the full-recipe command for hardware
  that can (bf16-capable, e.g. two A100s or L4s), with nothing reduced

**Reading, fixed before the run.** The float16 exports at steps 1,000 to 5,000
are each rendered and scored on the development set (9 speakers x 3
sentences, as every engine was), and the one with the highest held-out
similarity whose CER interval against the published model does not lie above
zero is the candidate. Choosing among five on the development set is what it
is for. That candidate and the published model are then scored once on the
sealed test sets (§3.1). No improvement is claimed before that has happened.

## 9. Order of work

1. Build and validate the multi-reference manifest on the development set.
   No GPU.
2. ~~Smoke-test MOSS v1.5 on a T4 at the official recipe, then the 27-cell
   development benchmark.~~ Done on 2026-09-18 (kernel v3 and v4, all gates
   passed): 0.507 with the official tag, no detectable difference from the
   baseline (-0.013 [-0.053, +0.027]). Open: its
   "continuation" mode with the reference transcript - one variable, with the
   official app's prompt construction.
3. Run the multi-reference conditions on the development set for the engines
   that accept the prompt budget, with the manifest fixed beforehand.
4. ~~Declare the primary test comparison~~ (§3.1, 2026-09-19); the user has
   set research and the best quality as the goal for now, so the commercial
   question no longer blocks.
5. ~~Build the independent test set~~: drawn 2026-09-19 (§3.2), 28 usable
   Neyshekar speakers primary, 30 Common Voice secondary, both sealed. Still to
   write: the twelve test texts per speaker (§3.3), checked against every
   training text.
6. The OmniVoice adaptation (§8): the probe, then - if it fits - the full run
   in sessions, then the development-set choice among its exports.
7. Score each final candidate on the sealed test sets once.

## 10. Open questions for the user

* Commercial use: set aside by the user on 2026-09-19 - research and the best
  quality first. It will still select the base and bind the data if it
  returns.
* A ParsVoice access request, if wanted, is theirs to make.
* Would they organise a listening test, even with five listeners?
