# Project status

A handover file: where the work is, what is already answered, what is running
elsewhere, and the next concrete step - enough to resume without re-deriving
anything. No credentials.

Last updated: 2026-10-25 +0330.

## The XTTS check: ParsVoice-XTTS not adopted (2026-10-27)

FINDINGS section 22; `eval/runs/persian-voice-xtts-check-v1.json`, result
recorded. ParsVoice-XTTS is flagged twice: CER 12.2% against the app's 4.3%,
and similarity -0.043 [-0.083, -0.010]. The user stopped listening after
three sentences: very poor quality, easy to pick out, some words more correct
(2 wrong against 4 on those three). The app stays as it is (OmniVoice A1). The
XTTS environment and weights stay in `~/.cache/persian-voice-cloning/xtts`
(2.1 GB plus 1 GB environment) until the user wants them removed.

**Next (the user's call):** Fish Audio Drama 3 has an official demo Space,
`fishaudio/Drama-3`, running on ZeroGPU. The same news sentences could be
rendered through it without a GPU of this project's own, but that sends the
text and a reference clip to a third-party service, so it needs the user's
yes.

## The news vowels check heard: no vowels on every word; weights tie on news (2026-10-25)

FINDINGS section 21; `eval/runs/persian-voice-news-vowels-v1.json`, result
recorded. Eight news sentences in a 2 x 2. Words wrong (of 138):

| | P | A |
|---|---|---|
| B | 13 | 14 |
| V | 21 | 19 |

- **V (the first vowel on every word) is not adopted:** 40 against 27. Its
  marks were all right, and the model read some of them long or wrong.
- **The weights tie (34 against 33):** the app keeps A1.
- **Left in every form:** loanwords and months (اکتبر, نوامبر, فوتبال),
  دوازدهم, and some common words. That is about one word in ten on news.

**Next (the user's call):** nothing is running. Writing marks has reached its
limit: the ezafe and the homograph vowels help, more marks hurt. What is left
is the model's own reading. Two levers:
- look for a base that reads Persian news better, checked live first, and test
  it by ear on these sentences
- train OmniVoice to read marks as written

## The user's news text: three app errors fixed (2026-10-25)

The user rendered a tech news paragraph with A1 and with the published model,
was unhappy with both, and named eight words said wrong (FINDINGS section 20).

**Fixed (three were the app's own):**
- اپل was marked اِپل from Homo-GE2PE's wrong reading.
- M6 was verbalised glued, as Mشش.
- کند was marked کُند, the adjective's spelling.

Letters touching numbers are now parted and spelled (`numbers_fa.py`), and
اپل and کند are off the homograph list.

**Still open, the model's own:** short vowels of ordinary and borrowed words
(جمعه, اکتبر, نوامبر, مجهز) and Latin words (OLED). Waiting on the user's word
on which model read the paragraph better (outputs 17:10 A1, 17:19 published).

## The adapted reading check heard: the app reads with the step-1,000 export (2026-10-24)

The question: does OmniVoice, trained further on 128 h of Persian, read the
app's text better than the published model? Its pronunciation had never been
heard, and CER cannot see the errors that were left: an ezafe the model adds
on its own, and wrong vowels inside ordinary words.

The check was registered before any front-end ran (f0ffa32,
`eval/runs/persian-voice-adapted-reading-v1.json`, result recorded; FINDINGS
section 19). Twelve new sentences, with 19 ezafe traps and 17 vowel traps,
were read from the same app text by three sets of weights. Kaggle
`persian-voice-adapted-reading-run` v1 rendered them; every gate passed. The
user heard them blind on https://claude.ai/artifact/QoQz55Cxfb1YLqiQkYGJB2,
and the page saved the answers itself.

| | words wrong (of 114) | none wrong | picked best | CER | similarity |
|---|---|---|---|---|---|
| P, published | 15 | 3 | 1 | 4.8% | 0.590 |
| A1, step 1,000 | 12 | 5 | 4 | 3.7% | 0.580 |
| A5, step 5,000 | 14 | 2 | 1 | 5.1% | 0.590 |

By the registered rule the app's weights become A1. Neither export was
flagged. A1's similarity on this voice is −0.010 [−0.025, +0.005]; on the
sealed test's unseen speakers it was the same as P's. The gain is in the
ezafes the model adds (8 words in P, 5 in A1); the vowels did not move. It is
three words, one listener, one voice and one seed.

**In the app:** `engines.omnivoice.weights: auto` reads with A1 from
`~/.cache/persian-voice-cloning/omnivoice/adapted/omnivoice-fa-1000-float16`,
which `scripts/fetch_adapted_weights.py` fetches from train-a's output and
verifies by sha256 (fetched on this Mac). Without A1, the app reads with the
published model and puts a warning in the output. `weights: published` keeps
the old weights.

**Still wrong in every form:**
- ezafes nobody wrote: پیر, مدرسه, کاغذی, برق before the next word
- کنارِ said as kenareye
- the vowels of کتلت, می‌پزد, چکش and کلاغ

**Next (the user's call):** nothing is running. The open lever is a model that
does not guess the ezafe: training with the tagger's ezafe written into the
transcripts, so that an unwritten kasra means none. That is a full training
run, about 24 GPU-hours as for the adaptation, and needs its own decision.
Listening to the app on real texts with A1 is the cheap step before it.

## The روی check heard: the app leaves روی plain (2026-10-04)

The check was registered before any front-end ran
(`eval/runs/persian-voice-rui-check-v1.json`, result recorded; FINDINGS
section 18). Ten sentences with 12 روی compared R1 (the app's رویِ) against
R0 (no kasra on روی). They were heard blind by a listener recorded as پارسا;
steps 1 to 3 were heard by احمدرضا. Page:
https://claude.ai/artifact/DnBHQbqRWXLt2innub42uo. Its database kept only an
empty start, so the answers came as the page's answer code
(`eval/runs/evidence/rui-review-v1/answers-listener-1.json`).

| | روی wrong (of 12) | words wrong (of 114) | mean CER |
|---|---|---|---|
| R1 (رویِ) | 1 | 12 | 3.8% |
| R0 (روی) | 0 | 11 | 3.1% |

"No difference" was picked for all 10. Similarity R0 − R1 is +0.005
[−0.021, +0.035]. The registered rule drops the kasra, but on one mark, so to
this listener the kasra matters little either way.

**In the app:** `NO_EZAFE_ON = {"روی"}` in `src/voiceclone/text/ezafe_fa.py`.
Through the app the 10 sentences give the R0 that was heard, 10 of 10
(`scripts/check_rui_in_app.py`). Tests: 626 passed, 1 skipped.

**Also heard, the same in both forms:**
- An ezafe nobody wrote on four words: باران, پیرمرد, حیاط, and دیوار before
  باغ را. This is the تعاملات معمولاً kind, which step 3's sukun did not stop.
- Wrong vowels inside ordinary words: طاقچه, نیمکت, بشقابش, زانویش, می‌نشیند,
  پنجهٔ.

**Next (the user's call):** nothing is running.

## Homograph vowels in the app; the روی check rendering (2026-10-04)

**In the app** (c73ddf8, the user's go-ahead after step 3): for OmniVoice, after
the ezafe, the first short vowel of each of HomoRich's 190 homographs that it
tells apart (`src/voiceclone/text/homographs_data.py`, CC0; not به), read from
the sentence by Homo-GE2PE (`src/voiceclone/text/homographs_fa.py`, step 1's
alignment carried over and tested against it; `text.homographs` in the
config). Loaded per text and released (0.66 GB peak, 5.5 s to load, about 1 s
a sentence after), not started under 1.0 GB free. Through the app with the real
tagger and Homo-GE2PE, step 3's 12 sentences give the EH that was heard, به
unmarked (`scripts/check_homographs_in_app.py`). Known from step 3: Homo-GE2PE
reads about 4 in 23 homographs wrong, and the mark is then wrong (نشسته in the
روی check's r07 is one: nashaste for neshaste).

**The روی check** (registered 24fe06b, `eval/runs/persian-voice-rui-check-v1.json`):
10 sentences, 12 روی; R1 = the app today (رویِ), R0 = the same without the kasra
on روی. Kaggle `persian-voice-rui-check-run` (texts written by the notebook,
step 3's code dataset). Heard and read: see the section above.

## Step 3 heard: homograph vowels kept, sukun before adverbs dropped (2026-10-04)

Registered before any front-end ran (`eval/runs/persian-voice-frontend-listening-v2.json`,
result recorded; FINDINGS section 17). Twelve sentences (the user's and eleven
new) with 23 HomoRich homographs, 2 x 2 on top of the app's ezafe: H = the
first short vowel of each homograph from Homo-GE2PE in context, S = a sukun on
a noun before an adverb. Kaggle `persian-voice-frontend-listening-v2-run`
version 2 (version 1 rendered nothing: a relative-path bug, fixed in
bfd6405; the code dataset could not be re-uploaded from this Mac, so v2 passed
the path absolute). Page: https://claude.ai/artifact/RdNBNBgQ5VZum2PG5N4rfh

Words wrong of 242: E 27, EH 25, ES 27, EHS 25; homographs wrong of 23: E 14,
EH 8, ES 15, EHS 8. **H kept and unflagged** (CER +0.05 points, similarity
+0.007 [-0.003, +0.018]); **S a tie, dropped**. H broke only the preposition
به, all four times it marked it. روی with the written ezafe (رویِ) was heard
wrong in 11 of 12 readings, in every form - not yet known whether the kasra
causes it (`scripts/read_frontend_listening_v2.py` -> `eval/runs/evidence/frontend-review-v2/reading.json`).

**Next (the user's call):**
1. H into the app for OmniVoice: Homo-GE2PE's first vowel on HomoRich's 191
   homographs (`eval/manifests/homorich_homographs_v1.json`, CC0), recommended
   without به (every mark on it was heard wrong). Homo-GE2PE is 0.7 GB, run
   before the speech model like the tagger; off with a note where it is absent.
2. A short blind check of روی with and without the written ezafe.

## The ezafe is in the app for OmniVoice (2026-10-03 18:20 +0330)

What the user approved after step 2 (FINDINGS section 16):

- **The tagger without spaCy** (`src/voiceclone/text/ezafe_tagger.py`): hazm's
  ParsBERT tagger read out of its spaCy files once into
  `~/.cache/persian-voice-cloning/ezafe-tagger/eba1d2b30cc4` (624 MB) and run
  with transformers. Registered check (`eval/runs/persian-voice-ezafe-tagger-v1.json`):
  the same decision on all 3,355 words of 414 sentences and all 147 tags of
  the 12 listening sentences; 0.86 GB resident against the pipeline's 4.3 GB.
  `scripts/check_ezafe_tagger.py` repeats it.
- **In the text pipeline** (`src/voiceclone/text/ezafe_fa.py`, `text.ezafe` and
  `text.marks_engines` in `configs/default.yaml`): for OmniVoice only, the
  ezafe is written as کتابِ / خانهٔ / دانشجوی; loaded per text, released after,
  not started under 1.5 GB free (auto: plain script and a note instead). A
  final h is pronounced after ا/و and in 33 listed words (`marks_fa.PRONOUNCED_H`);
  no ezafe on مدیر before عامل. The normaliser keeps أ and writes ۀ as هٔ for
  OmniVoice. `scripts/check_ezafe_in_app.py`: the 12 listening sentences give
  the heard E text in 11, n12 differs only by that مدیر (not yet heard).
- Unchanged: every other engine's text; the registered render scripts turn
  both off (`ab_omnivoice_ipa.render_config`). From now on an OmniVoice
  benchmark carries the ezafe unless `text.ezafe.enabled: false` - earlier
  OmniVoice numbers are all plain script.
- KaamelDict is GPL: it only checked the h list; none of it ships.

**Next (optional, the user's call):** the first vowels - render E against E +
first vowels on new sentences and hear them blind, as step 2's registration
requires before any vowel ships.

## Step 2 heard: the ezafe goes in, the sukun is dropped, the vowels wait (2026-10-03 17:05 +0330)

The front-end listening test (`eval/runs/persian-voice-frontend-listening-v1.json`,
two amendments before any render, result recorded; FINDINGS section 15): 12 new
hard sentences in four forms - A plain, E +ezafe (hazm's ParsBERT tagger), EN
+sukun where no ezafe, ENV +first short vowels - rendered in one Kaggle T4
session (the Mac's memory guard refused; `marsmanam/persian-voice-frontend-listening-run`
v1, 0.13 GPU-hours, every gate passed), heard blind by the user on
https://claude.ai/artifact/UbAgAE7tPmtY9Y2cknpLWC.

Words marked wrong of 147: A 12, E 9, EN 12, ENV 10; picked best A 1, E 4,
EN 2, ENV 3, no difference 2; mean CER 3.1 / 3.9 / 3.9 / 2.9%, nothing flagged.
By the registered rules: **the ezafe is kept, the sukun dropped**, the first
vowels kept against EN - but ezafe + vowels without the sukun was never
rendered, so only the ezafe ships now. Read beside it: one listener, two- or
three-word differences; the listener unknowingly heard four bit-identical pairs
and judged three alike; the vowels fixed 4 of their own words and broke 4, so
their net gain is not theirs. Evidence: `eval/runs/evidence/frontend-listening-v1/`
(renders), `eval/runs/evidence/frontend-review-v1/` (key, items, page, answers,
score).

**Next (the user's go-ahead first):**
1. The ezafe into the app: the tagger as a worker in its own environment
   (`~/.cache/persian-voice-cloning/frontend/.venv`, about 4 GB while it runs -
   so it runs and exits before OmniVoice loads), `marks_fa.with_ezafe` on what
   it predicts, and the normaliser fixes found on the way (أ to ا, which made
   تأخیر misread; ۀ to a bare ه, which drops a written ezafe). Watch: the tagger
   links compounds such as مدیرعامل, which the listener heard right only
   without the ezafe.
2. Optional, before any vowel ships: render E against E + first vowels on new
   sentences and hear them blind, as the registration requires.

## Text tools measured: hazm's ParsBERT tagger for the ezafe; no vowel source for every word (2026-10-03)

Step 1 of the pronunciation fix, on text only (no audio, no GPU), registered
before any component ran (`eval/runs/persian-voice-frontend-components-v1.json`,
FINDINGS section 14). Gold: SentenceBench, read on the 213 sentences in no
component's named training data (HomoRich holds nearly all its ManaTTS and
Common Voice sentences - amendment written before any score), and the 14
review sentences (`eval/manifests/frontend_review14_gold_v1.json`).

- **Ezafe: hazm's ParsBERT tagger** (`roshan-research/hazm-parsbert-postagger`
  through spaCy, full tags) - 1.7% of words wrong on the unseen sentences,
  none wrong on the 14 review sentences, right on every ezafe error OmniVoice
  made. ALBERT 2.5%, Homo-GE2PE 3.0%, hazm's CRF 4.8%, plain script 12.6%.
  It runs in its own environment, `~/.cache/persian-voice-cloning/frontend/.venv`
  (Python 3.12, hazm 0.12.1, spaCy 3.8.16, transformers 4.53.2): 655 MB, ~4 GB
  while running, 32 s for 400 sentences on CPU.
- **Vowels: nothing is accurate enough to mark every word.** Homo-GE2PE 88.7%
  of words (homographs 74%), KaamelDict 93% of the 52% it answers - against the
  registered 96% bar. Vowels get marked only where needed.
- Not run: `PedramR/canine-fa-diacritizer` is gated (the request comes from
  the user's Hugging Face account); `persian-byt5-g2p` reads a word at a time.

The 12 readings Claude wrote for the 14 sentences that a component disputed
went to the user blind; the user agreed with all 12
(`eval/runs/evidence/frontend-components-v1/adjudication-answers.json`), so the
numbers stand. Homo-GE2PE's vowels there: right on 8 of OmniVoice's 9 heard
vowel errors, wrong on 6 words OmniVoice read right.

**Next (the user's go-ahead first): step 2** - put the tagger in front of
OmniVoice: write each predicted ezafe into the text (kasra, or ی / هٔ after a
vowel), try a mark for "no ezafe here" (the review's spurious ones fell at
phrase ends), and mark vowels only on homographs and known-risky words. Measure
by listening and CER on new hard sentences written before the fix, against
plain script.

## The pronunciation review is answered: ezafe and short vowels (2026-10-03)

The user listened blind to the 14 A/B/C sentences and marked every wrong word
(`eval/runs/persian-voice-pronunciation-review-v1.json`, FINDINGS section 13).
Plain script (A, what the app uses): 3 of 14 readings without an error, 19 of
199 words wrong, picked best 8 times; **9 of its errors are ezafe** (5 added,
4 missing) and **8 short vowels**. Diacritised twins (B) cleared A's errors
where their marks were right and were read as written where wrong. **The IPA
model (C) is closed:** worst on every count, never picked, accent foreign.

**Next (the user's go-ahead first):** a text front-end that writes ezafe and
the needed short vowels into OmniVoice's input. Candidates found on Hugging
Face on 2026-10-03, untested here: `abreza/persian-ezafe-albert` (ezafe),
`PedramR/canine-fa-diacritizer` (vowels; 8.3% diacritic error on its own test,
word-final ezafe noted as hard), and Homo-GE2PE already in the project. Measure
each on these 14 sentences first (the listener's notes give the right
readings), then the whole front-end on new hard sentences written before it,
by listening and CER.

## The first listening test is built and deferred (2026-10-03)

Nobody had listened to any render; the user asked for a listening test.
Registered before anyone listened (`eval/runs/persian-voice-listening-v1.json`):
two blind A/B tests on unseen Neyshekar speakers - OmniVoice published against
the adapted export (14 trials, where the metric found nothing) and OmniVoice
against F5-TTS (7 trials, the largest gap measured) - each trial a held-out
real recording, both renders in hidden balanced order, loudness-matched, three
questions (same person, natural, misread words). Built by
`scripts/build_listening_test.py` (now able to use the sealed sets); keys in
`eval/runs/evidence/listening-v1/`.

The page: https://claude.ai/artifact/7yj1WvanE1A5pih2vLCMy7 - private until the
user shares it. Listeners who can write to its database (the owner, people
invited by email as editors) are saved automatically, each seeing only their
own; anyone else gets an answer code at the end to send to the user.
**Deferred the same day, at the user's request, before anyone answered:** fix
Persian pronunciation first, then rebuild this test with the fixed system's
renders. [measured] On the sealed set the recogniser reads the speakers' own
recordings at CER 6.4% and OmniVoice at 9.9%, so about 3.5 points of error are
not the recogniser's. When the test runs: the user recruits listeners (five or
more for the declared crossed interval). Then read the database (ArtifactData `list` of `responses`, with
`out_dir`) and any codes, run `scripts/export_listening_answers.py` with both
keys, then `scripts/score_listening_test.py` per test, and record what it says
- nothing is written as a result before that.

## The sealed test is answered: no detectable improvement on unseen speakers

[measured, read by the pre-registered code] **Primary set** - 27 unseen
Neyshekar speakers (ney06 left out: its reference is refused by the configured
preparation): the export at step 1,000 minus the published model is
**+0.0014 held-out similarity [-0.0042, +0.0069]**, one-sided p 0.31, CER
-0.11 points [-0.75, +0.53]. **H0 of no improvement is not rejected**, and
the interval's upper end excludes the 0.03 the test was designed to detect.
**Secondary set** - 30 Common Voice speakers: +0.0040 [+0.0002, +0.0079], H0
rejected on that set, a fifth of the 0.0205 noise line. So the adaptation - the
official recipe's 5,000 steps on 128 h of Persian, about 24 GPU-hours of training sessions - did not
measurably improve zero-shot cloning of unseen Persian speakers; the
development set's +0.025 was familiarity with Common Voice. No improvement is
claimed. The published OmniVoice stays what the app uses. FINDINGS section 12;
record `eval/runs/persian-voice-omnivoice-test.json` (`result`).

How it was run: version 1 (`persian-voice-omnivoice-test`, 10:34-13:37 UTC,
2.90 GPU-hours) scored the secondary set; its primary runs stopped on a harness
error at ney06 (`KeyError: 'window'`, fixed in `130a649`), before either wrote
a results file. The primary set was rerun alone with the fix
(`persian-voice-omnivoice-test-primary`, 14:42-17:03 UTC, 2.16 GPU-hours),
registered before any primary score was read
(`amendment_2026_09_30_primary_rerun`).

## The arms are read: OmniVoice is the right base (2026-10-03)

`persian-voice-omnivoice-arms` v2 (pushed 2026-09-30 17:36 UTC, 6.3 h, every
gate passed; version 1 had stopped in an environment cell built for the
OmniVoice sessions) rendered F5-TTS Persian (48 steps, as registered) and
Pocket-TTS Farsi v2 on the same sealed cells. Read by `scripts/read_arms.py`:
on the 27 unseen Neyshekar speakers OmniVoice's candidate is above F5 by
**+0.112 [+0.091, +0.133]** and above Pocket by **+0.221 [+0.193, +0.249]**;
on Common Voice +0.049 and +0.203. Every interval clears zero, so by the
registered reading **the base choice survives contact with unseen speakers**.
F5 reads more intelligibly than Pocket (CER -5.4 points). FINDINGS section 12;
`arms_result` in `eval/runs/persian-voice-omnivoice-test.json`.

**Where the whole OmniVoice question stands:** the published OmniVoice is the
best model measured here for cloning unseen Persian speakers (0.596 on
Neyshekar), far above F5 and Pocket; the adaptation (about 24 GPU-hours of
training) matched it and did not beat it. Nobody has listened to any of it.

The Mac's Kaggle calls were refused from 2026-09-30 19:11 UTC with a token not
yet expired, and worked again on 2026-10-03 without a new login; the cause is
not known. The arms result was read then.

**Next: the user's decisions.** The LongCat question deferred to this point
(below; recommended 2026-09-30: not now - small expected gain over OmniVoice's
0.601, no training code, a language to teach from nothing - and listening and
waiting for LongCat's own training code first), and whether to organise a
listening test. A new quota week began 2026-10-03: 30 GPU-hours until
2026-10-10.

## Decided 2026-09-25: the OmniVoice plan goes on, LongCat waits for the test

The user chose the recommendation after the probe: finish the OmniVoice
adaptation (session c, the development session, the sealed test, the arms)
and decide on a LongCat-1B adaptation only after the sealed test, with a
pilot's bar at 0.601 in Persian.

Checked before session c's push: its notebook rebuilds byte-identical,
`.venv/bin/python -m pytest` passes, `persian-voice-eval` is still the version
of 2026-09-19 20:45 UTC that a and b ran, and a and b read COMPLETE on Kaggle.
[measured, `kaggle quota`, 19:40 UTC] 0.67 GPU-hours left, reset 2026-09-26
00:00 UTC, so c is pushed right after the reset with its usual 11.25 h.

Also checked what the later sessions read against their registration: only
F5's configured `nfe_step` had moved (48 to 64 on the 21st). The arm is pinned
to the registered 48 (`968af67`, amendment in
`eval/runs/persian-voice-omnivoice-test.json`); FINDINGS section 3 measured 64
as no better. Everything else - OmniVoice's settings, the scoring recogniser,
the 12 test texts and 3 development sentences after the text pipeline - is as
registered.

## The LongCat probe (2026-09-25): complete

`marsmanam/persian-voice-longcat-probe-run` v1 ran 14:03-15:55 UTC (about 1.9
of the week's last 2.51 GPU-hours), every gate passed, all 550 cells rendered (3 x 150 English, 2 x 50 Persian).
[measured] **By the rule fixed before the run, LongCat-1B clears the line**:
held-out similarity minus OmniVoice's, both reading English from Persian
speakers' references, +0.068 [+0.055, +0.080], all 25 speakers - so an
adaptation is worth costing. The 3.5B adds +0.008 [-0.001, +0.017] over the
1B. The prompt-copying explanation of its lead is not supported (+0.014
[-0.001, +0.029]). It cannot read Persian untrained (CER 81.6%).

[measured, post hoc - decides nothing] OmniVoice reading **Persian** on the
same speakers scores 0.601: +0.080 over its own English, and level with
LongCat-1B's English (-0.012 [-0.030, +0.006]). So the bar for a Persian
LongCat is OmniVoice's Persian, not its English; whether OmniVoice's English
penalty is its `en` tag is untested. FINDINGS section 11; record
`eval/runs/persian-voice-longcat-probe.json`; evidence
`eval/runs/evidence/longcat-probe-v1/`; renders in
`remote/longcat-probe-v1/renders` (not in git). Nobody has listened.

**Next, the user's call:** whether to cost a LongCat-1B adaptation (training
code written from its flow-matching loss, fp32 recipe, and a pilot whose bar is
0.601 in Persian). The OmniVoice plan is unchanged: session c after the reset.

The user asked whether LongCat-AudioDiT-1B should replace OmniVoice as the
model adapted to Persian, or be joined with "the best Persian TTS". Checked
live (SOTA_ROADMAP section 4): Chinese and English only, no training code,
and its SOTA claim is the 3.5B's WavLM similarity, which other scorers and
noisy prompts do not reproduce; no benchmark covers Persian, and a Persian TTS
has no seam into LongCat. So one measurement decides it, pre-registered in
`eval/runs/persian-voice-longcat-probe.json`: OmniVoice against LongCat-1B
(primary) and -3.5B (secondary) on English text, Neyshekar's 25 valid
speakers, held-out similarity, read by `scripts/read_longcat_probe.py`; an
adaptation is costed only if LongCat-1B's whole interval is above 0.0205.
Nothing in the OmniVoice plan changes.

Built and committed (`5c7d40b`..`cf62d2d`): the English-text fix in the text
pipeline (it wrote "،" and Persian number words into English), the `longcat`
engine and worker (upstream's inference.py; checked end to end on a tiny random
model on this Mac, where moving the transformer to another device gave
bit-identical audio), the probe corpus, texts, reader and the kernel
(`kaggle/build_longcat_probe_kernel.py`). The kernel mounts its **own** dataset,
`persian-voice-longcat-probe`, so `persian-voice-eval`
stays as the pending sessions verified it.

Unrelated, found on the way: `ruff check src scripts tests` fails on HEAD in
files from the IPA work of 2026-09-23 (`build_omnivoice_ipa.py`,
`omnivoice_ipa.py`, `ipa_normalize.py`); left as found.

**Nothing of the adaptation is running remotely.** The early development session
(`persian-voice-omnivoice-dev` v1, 2026-09-20 22:32 UTC, 17 minutes, 0.31
GPU-hours) is complete, and it is the first evidence here that **the
adaptation moves identity transfer at all**:

| | held-out | minus published | 95% interval |
|---|---|---|---|
| F5 baseline, re-scored as the scale control | 0.5196 | - | (0.520 on the Mac: the scale reproduces) |
| OmniVoice published | 0.5837 | - | - |
| export 1,000 | **0.6083** | +0.0245 | [+0.0029, +0.0462] |
| export 2,000 | 0.5919 | +0.0082 | [-0.0132, +0.0296] |
| export 3,000 | 0.6004 | +0.0167 | [+0.0012, +0.0321] |

[measured] All three are above the published model and two intervals exclude
zero; every CER interval includes zero with a slightly negative mean, so
nothing was paid in intelligibility. **But the best export is the earliest**,
the three sit inside each other's intervals, and only step 1,000's uplift
clears the encoder's 0.0205 noise line - so nothing here says training further
helps, which matches the loss flattening after step 2,000. And the development
set is the contaminated one: an uplift on Common Voice speakers can be domain
familiarity rather than better cloning. The sealed test answers that. This run
**chooses nothing** - the candidate is picked after session c, over every
export, by the same fixed rule (`eval/runs/persian-voice-omnivoice-dev-early-v1.json`).

It also established that the development notebook works end to end before nine
GPU-hours are spent ahead of it.

Sessions a and b are complete andSessions a and b are complete and
**3,000 of the recipe's 5,000 steps are trained**. Session b (pushed
2026-09-20 16:05:47 UTC with a 6.5 h budget) mounted a's tokens and
`checkpoint-2000`, resumed through the official `resume_from_checkpoint`,
trained 1,000 steps in 4.61 h at 16.367 s a step and stopped itself at
`checkpoint-3000` because the next save would not have fitted; it kept that
checkpoint and `export-3000-float16` and cost about 4.7 GPU-hours. Its
downloads took 0.8 minutes and nothing was tokenised again.

[measured] **The loss has stopped falling.** Session a went 4.367 to 4.212
over its 2,000 steps; session b sat between 4.220 and 4.268 over its 1,000,
and validation moved 4.323 to 4.294. Nothing in that curve says the model is
still learning - and nothing in it says the model is better at Persian either.
The development session, which scores every export against the published
weights, is what answers that; if no export beats the base, the base is what
goes to the sealed test and the project says the adaptation did not work.

**Next:** the weekly quota resets 2026-09-26 00:00 UTC with 2.82 GPU-hours
left before it. Then session c with the usual 11.25 h budget (2,000 steps at
16.4 s is about 9.1 h of training; build it with no `--hours` override), then
the development session, the sealed test, and the arms session.

**Session a v2 is complete** (pushed 2026-09-19 20:54:15 UTC, 11.25 h budget):
every gate passed and 2,000 of the recipe's 5,000 steps are trained and
checkpointed. Downloads took 4.2 minutes, every step on the first attempt -
the stall that cost v1 a whole session did not recur. Data: 87,499 train clips
/ 127.85 h and 2,251 valid, nothing left out, no Neyshekar test shard
downloaded. Tokenising: 88 shards, 0 failures, 79 min at 97x real time.
Training: 8.69 h on the T4 pair at 15.547 s a step (11% slower than the
13.98 s the budget assumed), peak 13.4 GiB of 14.56, saves every 500 steps,
`checkpoint-2000` plus `export-1000-float16` and `export-2000-float16` kept,
11.1 GiB of output. [measured] train loss: mean 4.367 over the first 100 steps,
4.212 over the last 100; validation loss flat (4.263 -> 4.280 over 144 recorded
batches). [hypothesis] that is consistent with early adaptation at lr 1e-5 and
40% of the recipe; nothing here shows the model is better at Persian - only the
sealed test answers that, after step 5,000. Record and evidence:
`eval/runs/persian-voice-omnivoice-adapt.json`,
`eval/runs/evidence/omnivoice-train-a-v2/`.

**GPU quota** [measured, Kaggle's own `kaggle quota`, 2026-09-20]: 22.50 h used
of 30, 7.50 h left, resets **2026-09-26 00:00 UTC**. The API does report it
after all - `KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle quota` - so the
user no longer has to read it off the site. The OAuth token expired on the
20th; `kaggle auth login --force` is what renews it, and only the user runs it.

Dataset `persian-voice-eval` is at the version staged from `b103fa6`
(2026-09-19 20:45 UTC), verified by download: 251 files byte-identical, the
split file unpacked by Kaggle with identical content. The training code in it
(`omnivoice_adapt.py`, `prepare_adaptation_data.py`, the split, the gates) is
unchanged since `007af09`; new are the bounded downloads and the development
and test sessions' code.

## A Persian-specific model appeared, and was measured (2026-09-20)

* `mehdi-hf/pocket-tts-farsi-v2`, announced on the 19th and packaged as
  `nimaone/persian_tts`: 109.5 M parameters, **973 h of Persian from 2,978
  speakers**, phoneme input through Homo-GE2PE, CPU-speed, CC-BY-NC
* added as the `pocket` engine by its own card's recipe (`ff8101e`) and scored
  on the development set beside the stored OmniVoice renders re-scored in the
  same session
* [measured] held-out similarity **0.333** against OmniVoice's **0.583**;
  paired over the same 27 cells **-0.250**, 7.6 standard errors from zero,
  worse in 26 of 27, and below this corpus's 0.508 different-speaker line. CER
  8.9% against OmniVoice's 2.8%, with 2 of 27 renders the recogniser could not
  read at all
* the integration was checked before the result was believed: prompt lengths of
  4 s and 6 s change little (0.412 / 0.388 against 0.381), and **the authors'
  own released samples score 0.321-0.368 against their own prompts on this
  project's ECAPA** - below what this integration gets, so the wiring is not
  the cause
* so OmniVoice stays the model being adapted. What Pocket-TTS wins is cost
  (0.45 GiB, no GPU, faster than real time on a CPU) and it is the first engine
  here that can synthesise from Homo-GE2PE phonemes
* record: `eval/runs/persian-voice-pocket-tts-v1.json`, FINDINGS §10

## The sealed test gained two arms (2026-09-20, registered before any render)

* as first written the test compares the adaptation against the model it
  started from and answers nothing else - and the development set that chose
  that base is very probably contaminated (8 of 9 speakers are Common Voice)
* so `407ffa5` registers **F5-TTS Persian** and **Pocket-TTS Farsi v2** as
  pre-registered *secondary* arms on the same sealed corpora, same twelve
  texts, same seeds, same reference and held-out clips. The primary comparison
  and its reading are untouched
* they run in their own session (`kaggle/build_omnivoice_arms_kernel.py`,
  slug `persian-voice-omnivoice-arms`), because the test session's interpreter
  carries OmniVoice's pinned stack and installing `f5-tts` or `pocket-tts`
  beside it would change the environment the primary comparison is measured in
* read on **held-out similarity only**: similarity reproduced across machines
  in every test here, CER did not, so CER is compared between the two arms and
  never against the test session
* budget: about 2-2.5 GPU-hours, after the primary test, after the reset
* what it buys: if an arm beats the adapted model on speakers nobody has seen,
  the project says so - which is the one thing the test as written could not
  have revealed

## Stage

The user's goal as of 2026-09-19: research and the best quality first; the
commercial question no longer blocks. The work: adapt OmniVoice - the highest
identity score on the development set, not a proven SOTA, and very probably
trained on eight of its nine speakers - to Persian by its official recipe, on
independent data, with the test decided before any training (SOTA_ROADMAP §3,
§5, §8). No improvement is claimed until the sealed test comparison exists.

## Last healthy commit

`0346c86` (before this file's commit). Tree clean; `.venv/bin/python -m pytest`
gives 354 passed, 1 skipped (the skip needs Linux `/proc`, and ran on Kaggle);
`ruff check src scripts tests` clean.

## The adaptation, where it stands

* **recipe** [source]: `k2-fsa/OmniVoice` at `08be0b4c`; the config the
  official script runs has lr **1e-5** (the examples README's 5e-5 is stale
  since `a65149e0`, 2026-04-22), 5,000 steps, 8,192 tokens per GPU on two GPUs,
  bf16; SDPA config for GPUs older than Ampere; runtime from `uv.lock`; base
  `c5fdb5cc`; tokenizer `528e871c`, sha256 equal to the copy the base bundles
* **data** [measured]: `eval/manifests/adaptation_splits_v1.jsonl.gz`, drawn
  from metadata only - no audio, no encoder. Train 87,499 clips, 127.85 h,
  3,835 speakers (Common Voice 26 cleaned 41.70 h + Neyshekar v6 86.16 h);
  valid 2,251 clips, 3.68 h; development speakers excluded; no speaker in two
  splits; no shared normalised text. Test sets sealed: primary the 28 usable
  Neyshekar test speakers (not on OmniVoice's published training list),
  secondary 30 Common Voice ones (15/15 by self-reported gender)
* **power** [measured]: margin and true effect now separate; primary H0 Δ ≤ 0
  at one-sided alpha 0.025, planned for δ = 0.03: 87% power at the pilot's
  spread (0.050), 36% at its upper end (0.096)
  (`eval/manifests/power_design_v1.json`)
* **probe v1** [measured]: the official recipe does not fit a T4 - both
  configs run out of memory at once, SDPA in its first forward pass with 14 GiB
  allocated. The out-of-memory fallback never ran because of a bug in the
  notebook (it read only the last 25 lines of the launcher's log)
* **probe v2** [measured]: with gradient checkpointing and DDP bucket views it
  trains - 13.98 s a step, 12.3 GiB of 14.56 - and on one GPU the checkpointed
  path gave the same first loss and first-step gradients as the official one
  to every printed digit. Budget: about 23-26 GPU-hours in three sessions
* **full run**: session a v1 stalled in its downloads and was cancelled
  (above); b and c are built
  (`kaggle/omnivoice-train-b`, `-c`) and are pushed only after the previous
  session's output exists
* **the float16 export path** [measured, tiny model]: the probe stopped before
  any save, so the export had never run; `scripts/check_omnivoice_export.py`
  exports a 10.8M-parameter OmniVoice the way the sessions do and loads it with
  `from_pretrained`: the published model's files, both tokenizers, all 26
  tensors equal. On this Mac's torch 2.14 / transformers 5.17 (the OmniVoice
  code is identical to the pinned commit's; Kaggle's pinned torch and
  transformers are not what ran). Session a's first real export is at step
  1,000
* **sealed test** [built, not pushed]: `kaggle/omnivoice-test`
  (`kaggle/build_omnivoice_test_kernel.py`), planned and budgeted before any
  test clip was read (`eval/runs/persian-voice-omnivoice-test.json`): both test
  sets written by `scripts/prepare_test_corpus.py`, twelve texts x seeds 42
  and 43, published model then candidate in one session, read by
  `scripts/read_sealed_test.py`; about 6 h of one T4 pair
* **reference rule amended** [measured, metadata only]: the split's longest
  clip would have been cut by the reference window (over 11.5 s) while keeping
  the whole clip's transcript - 13 of the 28 Neyshekar references. The test
  reference is now the longest clip of 6.5-11.4 s; still 28 Neyshekar
  speakers, no Common Voice role changed (SOTA_ROADMAP §3.2). A Neyshekar test
  speaker has no reserve, so a clip failing the 50 ms duration check would drop
  them; Neyshekar's listed durations are multiples of 60 ms, but all 780 of its
  clips in probe v2 passed that check, and session a's data summary will show
  it for all 56,000

## The app on this Mac (2026-09-19, the user's request)

* OmniVoice is the default engine (`2c0b429`), behind `memory_guard`: the
  recogniser and the engine are never loaded together; the app falls back to F5
  where OmniVoice is not installed
* [measured] the guard alone was not enough: a guarded render reached warning
  memory pressure while the weights loaded (5.1 GB already compressed, no swap)
* `c084be2`, `04fda8f`: macOS's pressure level is read (`voiceclone.memory`);
  nothing starts at warning; OmniVoice is not started below 6 GB available (an
  estimate from three warnings at 3.05-3.37 GB available) and its worker is
  killed the moment warning arrives; the worker loads a verified float16 copy
  (bit for bit, 1.2 GB; made on this Mac in 5.3 s at 0.59 GB resident, record
  in `~/.cache/persian-voice-cloning/omnivoice/float16/<revision>/SOURCE.json`)
* [measured] after both: the load completed without pressure; with 4.4 GB
  available the render reached warning two seconds in and the app stopped it
  cleanly. F5 runs in-process: it is refused at warning, but a load
  already under way cannot be stopped the same way
* `eeb0483` (2026-09-20, the user's request): the transcript no longer has to
  be typed. The reference is transcribed *before* the engine loads and the
  recogniser is released first, which the measurement behind it
  (`scripts/measure_recogniser_release.py`,
  `eval/runs/evidence/recogniser-release/measurement.json`) shows gives back
  the 1.3 GB it took - twice in a row, the process returning to 0.9 GB, with
  the 0.6 GB over its 0.3 GB start being imported runtime, not a leak. The
  speech-to-speech cascade is allowed again on the same rule (both
  transcriptions happen before the load); best-of-N is still refused, because
  it transcribes each take while the engine holds memory. Typing the transcript
  still skips the 6-9 s transcription
* [measured] 2026-09-20 08:44, with the user's apps closed and 8.16 GB
  available: the first **complete** OmniVoice render on this Mac, through that
  new path - reference 1.1 s, transcription 7.2 s, load 5.3 s, synthesis 13.8 s,
  evaluation 4.3 s, no pressure warning at any point, 6.53 GB available at the
  end (`eval/runs/evidence/recogniser-release/omnivoice-render-on-the-mac.json`,
  log beside it). Nobody has listened to the render. The recogniser's own
  transcript of the reference had three wrong words, which is the argument for
  typing it where the words are known

## Remote jobs (live state read 2026-09-19 01:14 UTC)

| kernel | state | re-run? |
|---|---|---|
| `persian-voice-fetch-probe` v1 | complete in 8 min, CPU only: both download modes fetched all 21 GiB | no - `eval/runs/persian-voice-fetch-probe-v1.json` |
| `persian-voice-omnivoice-train-b` v1 | **RUNNING** since 2026-09-20 16:05 UTC, 6.5 h budget | no - read its output when it ends |
| `persian-voice-omnivoice-train-a` v2 | complete: 2,000/5,000 steps, `checkpoint-2000` kept | no - `eval/runs/evidence/omnivoice-train-a-v2/` |
| `persian-voice-omnivoice-train-a` v1 | CANCEL_ACKNOWLEDGED (read 18:10 UTC); log ends at 138 s in the download cell | no - superseded by v2 |
| `persian-voice-omnivoice-probe` v2 | complete, about 14 min: fits with memory-only changes | no - `eval/runs/persian-voice-omnivoice-probe-v2.json`, evidence in `eval/runs/evidence/omnivoice-probe-v2/` |
| `persian-voice-omnivoice-probe` v1 | complete, about 12 min: official recipe out of memory; fallback skipped by a bug | no - `...-probe-v1.json`, evidence in `.../omnivoice-probe-v1/` |
| `persian-voice-moss15` v1-v4, `-moss15-probe` v1-v2 | complete | no - `eval/runs/persian-voice-moss15-*.json` |
| `persian-voice-fish-smoke` v5, `-higgs-smoke` v2, `-eval-run`, `-moss-run` | complete | no - results pulled |

## Next concrete steps

0. Done: session a v2 trained 2,000 steps and session b is running with the
   6.5 h that were left this week. Left: read b's output when it ends, record
   it, and after the quota resets on 2026-09-26 build session c with the usual
   11.25 h (`--hours` only lowers one build; the default is unchanged) to
   finish at step 5,000, then the development session, then the sealed test.
1. When session a ends: read its gates (`session/gates.json` in its output);
   if `train` passed, push `kaggle/omnivoice-train-b` (it mounts a's tokens and
   checkpoint); then c the same way. If a session fails a gate, read why before
   anything is pushed again.
2. When c has ended: re-stage the dataset (`kaggle/build_dataset.py`) with
   this round's code and verify it by download, as for `007af09`.
3. Then one development-set session (`kaggle/omnivoice-dev`) - the five
   float16 exports and the published model scored in the same kernel - and
   the choice by the reading fixed in SOTA_ROADMAP §8.
4. The one test comparison: push `kaggle/omnivoice-test` after the choice
   (built; plan, reading and budget recorded before any test clip was read).

MOSS v1.5's "continuation" mode stays in reserve; it is not run for the
licence alone.

## Measured (development set: 9 speakers x 3 sentences, seed 42)

| run | engine | held-out | CER | scored in |
|---|---|---|---|---|
| `omnivoice` | OmniVoice | 0.5833 | 2.6% | kaggle eval-run 2026-09-12 |
| `moss` / `moss-local` | MOSS v1.0 (bf16 on T4) | 0.5587 | 6.5% (Mac) | kaggle moss-run / Mac |
| `finetuned-bestof4` | F5 continued | 0.5462 | 2.2% | Mac |
| `fish` | Fish S2 Pro (fp16, `--half`) | 0.5368 | 9.5% | kaggle fish v5 |
| `higgs` | Higgs TTS 3 port (fp16) | 0.5312 | 2.9% | kaggle higgs v2 |
| `moss15` | MOSS v1.5, tag "Persian" | 0.5244 | 9.0% | kaggle moss15 v3 |
| `moss15-fa` | MOSS v1.5, official recipe | 0.5066 | 10.9% | kaggle moss15 v4 |
| `baseline` | F5 as found | 0.5196 | 2.9% | Mac |

Every file above has a recipe beside it (`<label>.recipe.json`) - model and
codec revisions, settings, reference clips' sha256, session, results sha256 -
reconstructed for these older files with each field's source named. Intervals
by `scripts/compare_runs.py`, speaker-level. OmniVoice - baseline +0.064
[+0.025, +0.102] and OmniVoice - MOSS v1.5 +0.077 [+0.032, +0.122] clear 0 and
0.0205. MOSS v1.5 - baseline -0.013 [-0.053, +0.027]: no detectable difference.
CER only inside one scoring environment: MOSS v1.0 - OmniVoice +3.7 [+1.4,
+6.0] on this Mac (same recogniser settings and packages, documented in
FINDINGS §8); CER is the recogniser's reading - nobody has listened.

## This round (fourth review), each its own commit

1. `259a4c1` claims cut back to what the intervals and the recogniser support;
   the "same recogniser" basis for Mac CER documented; new caveat that
   OmniVoice very probably trained on eight of the nine development speakers
2. `7f8a116` power design with margin and true effect separate
   (`scripts/power_design.py`); primary comparison declared
3. `7191293` public Persian data surveyed; Neyshekar found; two-source split
   by speaker, without an encoder (`scripts/build_adaptation_splits.py`)
4. `60e9004` OmniVoice's official recipe pinned; the lr discrepancy resolved
   from the repository's history
5. `1732d58`, `d7a549c` the probe and its data writer; Kaggle unpacks `.gz`
   inside datasets, so the split file is checked by content hash
6. `17c8de0` every benchmark run writes its recipe beside its results, once;
   OmniVoice and F5 report the revision and checkpoint they loaded
7. `c7baf98`, `007af09`, `4c11444` probe v1 and v2 recorded; memory-only
   techniques with a gradient-fingerprint check; the runbook
   (`docs/OMNIVOICE-ADAPTATION.md`)
8. `592c64d` recipes reconstructed for the seventeen quoted results
9. `65e5f14` the full run's sessions, with plan and budget before the push
10. `29abd0b` the development-set session that will choose among the exports
    (built, not pushed)
11. `cfa3e1a`, `f1990a0` the twelve test sentences, checked against all 68,532
    distinct texts of the split, the development transcripts and the benchmark
12. `cb083f2` the benchmark renders a written text file whole and records the
    window each reference went through
13. `6d12038` the sealed test sets as corpora: references the pipeline keeps
    whole, the split's reserve rule recomputed and checked against the split
14. `7dafc14` the test's reading as code, pinned by tests on made-up results
15. `4683028` the development session takes its code from the dataset only (the
    training outputs carry older copies of the repository)
16. `d667c6b` the sealed test session, built
17. `2e13f0a` its pre-registration; roadmap, runbook and FINDINGS caveat
18. `cbb447e` the dataset list lacked the test session's three scripts - it
    would have failed after its downloads; staged now, with a "code" gate and a
    test that follows every OmniVoice notebook's scripts to the staged list
19. `32ddfd9` the export path checked on a tiny model
20. `0346c86` the OmniVoice engine waits for its worker with the configured
    deadline and starts a new worker after a hang or a crash - before, one hung
    render would have lost a whole test-set run, and one crash every later cell

Earlier rounds, for reference: third review `90f2f9c` `4341b0f` `89dc14e`
`87c33e2` `587af3d` `cdc4b36` `a390618` `09b6373` `7d1c9f9`; second review
`79fec9d` `56a963e` `c13ab5c` `7944081` `0f71c71` `69c17a8`.

## Corrections (kept as history, not erased)

* 2026-09-30: every commit hash in this file and in the records before this
  date is from the history as it was before the user rewrote it for GitHub.
  They are left as written; `docs/HISTORY-REWRITE-2026-09-30.md` maps each one
  to its commit now.

* 2026-09-19: this file said session a being RUNNING past its first hour
  "suggests (does not show) that the data stage passed". It had not started:
  the session was stalled in its download cell from 138 s. A running status
  says nothing about progress when a cell prints nothing - which is why the
  downloads now report as they go and are bounded

* 2026-09-19: the split's test reference rule (the longest clip) would have
  given 13 of the 28 Neyshekar test speakers a reference the pipeline cuts
  while keeping its whole transcript; amended before any test clip was read
  (`6d12038`, `2e13f0a`)
* 2026-09-19: the MOSS v1.5 records said the tag "was not" v3's problem, that
  one render "is unintelligible", that v1.5 "mispronounces", that the
  recogniser "behaved identically" across sessions, and FINDINGS said all three
  failed MOSS attempts were the runner's fault. Corrected (`259a4c1`): no
  detectable effect is not no effect; transcripts are not listening; identical
  controls cover their 27 inputs; the runner cause is documented for probe v2
  and inferred for v1 and v2
* 2026-09-19: the roadmap planned "one-sided, margin 0.03" from the row for a
  true 0.03 - a design that cannot succeed (`7f8a116`)
* 2026-09-19: the roadmap gave OmniVoice's adaptation lr as 5e-5, copied from a
  stale README table; the executed config says 1e-5 (`60e9004`)
* 2026-09-19: probe v1's notebook missed its own out-of-memory and skipped the
  fallback; fixed and re-run as v2
* 2026-09-18: the status of 18:35 +0330 called MOSS probe v2 "stuck for more
  than 2 h"; it had run 24 minutes and was finishing
* `312b757` (bounded waits) was motivated by "a process survived SIGKILL";
  MOSS probe v2 showed the group died on SIGTERM
* the MOSS probe builder assumed torch 2.9.1 on its image; it had 2.10.0
* the old runner's last line said "process group ended" whatever happened

## Open

* Fish, Higgs and Chatterbox accept `timeout_seconds` but their `_read_reply`
  never enforces it (OmniVoice does since `0346c86`; offered as a separate task)
* the Studio keeps a supplied transcript when the reference window cuts a
  clip over 11.5 s (it re-transcribes only for an engine's own limit) - a user
  with a long reference and a typed transcript gets a mismatched prompt. Left
  as it is until the sealed test has run, so the development inputs do not
  change mid-comparison; benchmark runs now record the window
* `build_eval_corpus.py` still selects development speakers with the judging
  ECAPA (documented; the new splits do not)
* the multi-reference pilot supports only 4 of 9 speakers; the listening test
  is built but has had no listeners
* for the user: ParsVoice and `Thomcles/Persian-Farsi-Speech` are gated - their
  terms are the user's to accept, and neither is needed to start; who could
  listen

## Standing limits

No new spend, GPU purchase, acceptance of legal terms, upload of the user's
private audio, heavy local training or inference, or git push without the
user. Routine fixes and runs in the free quota are allowed; every long run's
budget is written down before it is pushed.
