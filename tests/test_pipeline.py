"""Config layering, chunking, the text pipeline, engines and the voice library."""

from __future__ import annotations

import numpy as np
import pytest

from voiceclone.audio.io import Audio
from voiceclone.config import AppConfig, deep_merge
from voiceclone.engines import engine_status, list_tts_engines, list_vc_engines
from voiceclone.engines.asr import Transcript
from voiceclone.errors import EngineNotAvailableError, VoiceCloneError
from voiceclone.models import parse_hf_uri
from voiceclone.pipelines.results import QualityScores
from voiceclone.pipelines.studio import _candidate_rank
from voiceclone.pipelines.voices import VoiceLibrary, slugify
from voiceclone.text.chunker import chunk_text, estimate_duration, split_sentences
from voiceclone.text.g2p import compact_word_to_ipa, find_g2p_repo
from voiceclone.text.pipeline import TextPipeline


class TestConfig:
    def test_defaults_load(self, config: AppConfig) -> None:
        assert config.audio.sample_rate == 24000
        assert config.engines.tts == "omnivoice"
        assert config.engines.omnivoice.memory_guard is True
        assert config.engines.f5.nfe_step == 64
        assert config.engines.moss.temperature == 0.6
        assert config.engines.seedvc.diffusion_steps == 75
        assert config.engines.asr.vad_filter is True
        assert config.engines.asr.vad_speech_pad_ms == 400

    def test_nested_types_are_coerced(self, config: AppConfig) -> None:
        assert isinstance(config.engines.f5.nfe_step, int)
        assert isinstance(config.audio.preprocess.highpass_hz, float)

    def test_yaml_null_becomes_none(self, config: AppConfig) -> None:
        assert config.audio.preprocess.lowpass_hz is None

    def test_copy_with_dotted_override(self, config: AppConfig) -> None:
        updated = config.copy_with(**{"engines.tts": "moss", "audio.sample_rate": 22050})
        assert updated.engines.tts == "moss"
        assert updated.audio.sample_rate == 22050
        assert config.engines.tts != "moss"  # original untouched

    def test_copy_with_rejects_bad_path(self, config: AppConfig) -> None:
        with pytest.raises(VoiceCloneError):
            config.copy_with(**{"engines.tts.nope": 1})

    def test_deep_merge(self) -> None:
        merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
        assert merged == {"a": {"b": 1, "c": 3}}

    def test_paths_resolve(self, config: AppConfig) -> None:
        assert config.paths.resolved("cache_dir").is_absolute()


class TestChunker:
    def test_sentence_split(self) -> None:
        sentences = split_sentences("جمله اول. جمله دوم؟ جمله سوم!")
        assert len(sentences) == 3

    def test_short_text_is_one_chunk(self) -> None:
        chunks = chunk_text("سلام دنیا.", max_chars=200)
        assert len(chunks) == 1
        assert chunks[0].gap_before_ms == 0

    def test_long_text_splits(self) -> None:
        text = " ".join(["این یک جمله آزمایشی نسبتا بلند است."] * 30)
        chunks = chunk_text(text, max_chars=120)
        assert len(chunks) > 1
        assert all(len(chunk.text) <= 160 for chunk in chunks)

    def test_paragraphs_get_longer_gaps(self) -> None:
        text = (
            "این نخستین بند از متن آزمایشی ما است.\n\n"
            "و این هم بند دوم که کاملا جدا نوشته شده است."
        )
        chunks = chunk_text(text, sentence_gap_ms=180, paragraph_gap_ms=400)
        assert len(chunks) == 2
        assert chunks[1].gap_before_ms == 400
        assert chunks[1].is_paragraph_start

    def test_paragraph_boundary_survives_runt_merging(self) -> None:
        # The trailing paragraph is shorter than min_chars, but merging it into
        # the previous paragraph would erase structure the author asked for.
        chunks = chunk_text("این یک بند کامل و به اندازه کافی بلند است.\n\nکوتاه.", min_chars=30)
        assert len(chunks) == 2
        assert chunks[1].is_paragraph_start

    def test_first_chunk_never_has_a_gap(self) -> None:
        chunks = chunk_text("اول.\n\nدوم.\n\nسوم.")
        assert chunks[0].gap_before_ms == 0

    def test_indices_are_sequential(self) -> None:
        chunks = chunk_text(" ".join(["جمله آزمایشی."] * 40), max_chars=60)
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))

    def test_runt_tail_is_merged(self) -> None:
        chunks = chunk_text("یک جمله نسبتا بلند برای آزمایش تقسیم متن. کوتاه.", max_chars=50, min_chars=20)
        assert all(len(chunk.text) >= 10 for chunk in chunks)

    def test_empty_text(self) -> None:
        assert chunk_text("") == []

    def test_duration_estimate_grows_with_length(self) -> None:
        assert estimate_duration("سلام") < estimate_duration("سلام دنیا چطوری خوبی")

    def test_unpunctuated_wall_of_text_still_splits(self) -> None:
        text = "کلمه " * 200
        chunks = chunk_text(text, max_chars=100)
        assert len(chunks) > 1


class TestTextPipeline:
    def test_prepare_persian(self, config: AppConfig, persian_text: str) -> None:
        prepared = TextPipeline(config.text).prepare(persian_text, engine_name="f5")
        assert prepared.language == "fa"
        assert prepared.chunks
        assert "۲۵۰۰۰" not in prepared.normalized  # numbers were verbalised
        assert prepared.estimated_seconds > 0

    def test_grapheme_engine_gets_no_phonemes(self, config: AppConfig, persian_text: str) -> None:
        pipeline = TextPipeline(config.text)
        prepared = pipeline.prepare(persian_text, engine_name="f5")
        assert not prepared.used_phonemes
        assert all(not chunk.phonemes for chunk in prepared.chunks)

    def test_should_use_phonemes_follows_config(self) -> None:
        # The name is the contract: whatever is listed gets phonemes, and
        # nothing else does. Which engines are listed is a tuning decision
        # that has already changed once, so do not hard-code it here.
        # Local config: the `config` fixture is session-scoped, so mutating
        # it would leak this setting into every later test.
        local = AppConfig.load()
        local.text.g2p.apply_to_engines = ["moss"]
        pipeline = TextPipeline(local.text)
        assert pipeline.should_use_phonemes("moss")
        assert not pipeline.should_use_phonemes("f5")

    def test_no_engine_gets_phonemes_by_default(self, config: AppConfig) -> None:
        # Nothing in this project reads Persian IPA better than Persian
        # script: MOSS's phoneme channel is English/pinyin-trained and
        # collapses on it, and F5's tokenizer is grapheme-only.
        pipeline = TextPipeline(config.text)
        assert not pipeline.should_use_phonemes("moss")
        assert not pipeline.should_use_phonemes("f5")

    def test_empty_input(self, config: AppConfig) -> None:
        prepared = TextPipeline(config.text).prepare("")
        assert prepared.chunks == []

    def test_synth_text_prefers_phonemes(self, config: AppConfig) -> None:
        prepared = TextPipeline(config.text).prepare("سلام دنیا", engine_name="f5")
        chunk = prepared.chunks[0]
        assert chunk.synth_text == chunk.text
        chunk.phonemes = "sælɒm donjɒ"
        assert chunk.synth_text == "sælɒm donjɒ"

    def test_summary_is_renderable(self, config: AppConfig, persian_text: str) -> None:
        assert TextPipeline(config.text).prepare(persian_text).summary()


class TestG2PBridge:
    """The GE2PE symbol table, pinned against real model output.

    Each expectation below was produced by running the actual G2P model; the
    counter-intuitive ones (`j`, `c`, `$`, `;`) are exactly the mappings that
    are easy to get wrong and silently mispronounce.
    """

    def test_compact_to_ipa_vowels(self) -> None:
        # GE2PE writes the long back vowel as `a` and the short one as `/`.
        assert compact_word_to_ipa("ab") == "ɒ b"      # آب
        assert compact_word_to_ipa("/b") == "æ b"

    @pytest.mark.parametrize(
        ("compact", "expected", "word"),
        [
            ("$ekoft", "ʃ e k o f t", "شکفت"),
            (";ale", "ʒ ɒ l e", "ژاله"),
            ("jan", "dʒ ɒ n", "جان"),      # `j` is ج, not the glide
            ("yar", "j ɒ r", "یار"),        # the glide is `y`
            ("cay", "tʃ ɒ j", "چای"),       # `c` is چ, not /k/
            ("q/nd", "q æ n d", "قند"),
            ("@absb", "ɒ b s b", "alef onset is dropped"),
            ("ketabe", "k e t ɒ b e", "کتابِ (ezafe already in GE2PE output)"),
        ],
    )
    def test_verified_symbol_table(self, compact: str, expected: str, word: str) -> None:
        assert compact_word_to_ipa(compact) == expected, word

    def test_latin_digraphs_still_work(self) -> None:
        # For hand-written romanisation fed in directly.
        assert "ʃ" in compact_word_to_ipa("sh")
        assert "tʃ" in compact_word_to_ipa("ch")
        assert "x" in compact_word_to_ipa("kh")

    def test_bookkeeping_symbols_are_dropped(self) -> None:
        assert compact_word_to_ipa("ke1tab@") == compact_word_to_ipa("ketab")

    def test_empty(self) -> None:
        assert compact_word_to_ipa("") == ""

    def test_repo_discovery_returns_path_or_none(self) -> None:
        found = find_g2p_repo()
        assert found is None or found.is_dir()


class TestEngineRegistry:
    def test_engines_are_registered(self) -> None:
        assert "f5" in list_tts_engines()
        assert "moss" in list_tts_engines()
        assert "seedvc" in list_vc_engines()

    def test_status_shape(self) -> None:
        rows = engine_status()
        assert rows
        for row in rows:
            assert {"kind", "name", "available", "reason"} <= set(row)

    def test_unknown_engine_raises(self, config: AppConfig) -> None:
        from voiceclone.device import get_device_spec
        from voiceclone.engines import get_tts_engine

        with pytest.raises(EngineNotAvailableError):
            get_tts_engine("nope", config, get_device_spec("cpu"))

    def test_capabilities_are_declared(self) -> None:
        from voiceclone.engines import F5TTSEngine, MossTTSEngine

        assert F5TTSEngine.capabilities.requires_reference_text
        assert not F5TTSEngine.capabilities.accepts_phonemes
        assert MossTTSEngine.capabilities.accepts_phonemes


class TestModelURIs:
    def test_parse(self) -> None:
        assert parse_hf_uri("hf://owner/repo/file.pt") == ("owner/repo", "file.pt")
        assert parse_hf_uri("hf://a/b/c/d.safetensors") == ("a/b", "c/d.safetensors")

    def test_malformed(self) -> None:
        with pytest.raises(VoiceCloneError):
            parse_hf_uri("hf://owner/repo")

    def test_not_a_uri(self) -> None:
        with pytest.raises(VoiceCloneError):
            parse_hf_uri("/local/path.pt")


class TestVoiceLibrary:
    def test_slugify(self) -> None:
        assert slugify("My Voice") == "My-Voice"
        assert slugify("صدای من") == "صدای-من"
        assert slugify("  ") .startswith("voice-")

    def test_save_and_load(self, tmp_path, speech_audio: Audio) -> None:
        library = VoiceLibrary(tmp_path)
        profile = library.save("Narrator", speech_audio, transcript="سلام", quality_score=80.0)
        assert profile.slug == "Narrator"

        loaded_profile, loaded_audio = library.load(profile.slug)
        assert loaded_profile.transcript == "سلام"
        assert loaded_audio.num_frames > 0

    def test_list_and_choices(self, tmp_path, speech_audio: Audio) -> None:
        library = VoiceLibrary(tmp_path)
        library.save("A", speech_audio)
        library.save("B", speech_audio)
        assert len(library.list()) == 2
        assert len(library.choices()) == 2

    def test_delete(self, tmp_path, speech_audio: Audio) -> None:
        library = VoiceLibrary(tmp_path)
        library.save("Temp", speech_audio)
        assert library.delete("Temp")
        assert not library.delete("Temp")

    def test_missing_voice_raises(self, tmp_path) -> None:
        with pytest.raises(VoiceCloneError):
            VoiceLibrary(tmp_path).load("ghost")

    def test_empty_audio_rejected(self, tmp_path) -> None:
        with pytest.raises(VoiceCloneError):
            VoiceLibrary(tmp_path).save("Empty", Audio(np.zeros(0, dtype=np.float32), 24000))


class TestUIConstruction:
    def test_app_builds(self, config: AppConfig) -> None:
        pytest.importorskip("gradio")
        from voiceclone.ui import build_app

        assert build_app(config) is not None


class TestEvaluationMetrics:
    """WER must measure pronunciation, not typography."""

    def test_punctuation_and_zwnj_do_not_count_as_errors(self) -> None:
        from voiceclone.evaluation.similarity import word_error_rate

        # What we asked for, versus what Whisper actually returns: no
        # punctuation, and ZWNJ-joined words split apart.
        requested = "سلام، این یک آزمایش شبیه‌سازی صدا است. امیدوارم واضح باشد."
        heard = "سلام این یک آزمایش شبیه سازی صدا است امیدوارم واضح باشد"
        assert word_error_rate(requested, heard) == 0.0

    def test_a_real_substitution_is_still_caught(self) -> None:
        from voiceclone.evaluation.similarity import word_error_rate

        score = word_error_rate("صدا طبیعی و واضح باشد", "صدا عجیب و واضح باشد")
        assert score is not None and score > 0.0

    def test_empty_inputs(self) -> None:
        from voiceclone.evaluation.similarity import word_error_rate

        assert word_error_rate("", "چیزی") is None
        assert word_error_rate("چیزی", "") is None

    def test_compound_spacing_wrecks_wer_but_not_cer(self) -> None:
        """The case that made this metric necessary.

        Persian compounds have no fixed spacing, and Whisper picks its own.
        This render was correct - every sound in the right place - and word
        scoring called it 55% wrong purely over where the spaces landed.
        """
        from voiceclone.evaluation.similarity import character_error_rate, word_error_rate

        requested = "دیروز در کتابخانه شهر، کتابی درباره تاریخ ایران خواندم."
        heard = "دیروز در کتاب خانه شهر کتابی در باره تاریخ ایران خاندم."

        assert word_error_rate(requested, heard) > 0.5
        assert character_error_rate(requested, heard) < 0.05

    def test_cer_still_catches_real_errors(self) -> None:
        from voiceclone.evaluation.similarity import character_error_rate

        assert character_error_rate("صدا طبیعی و واضح باشد", "صدا طبیعی و واضح باشد") == 0.0
        # A render that came apart should not hide behind whitespace folding.
        assert character_error_rate("صدا طبیعی و واضح باشد", "سلام چطوری") > 0.5

    def test_cer_ignores_arabic_letterforms_and_harakat(self) -> None:
        """ASR emits neither, so neither can be an error the render made."""
        from voiceclone.evaluation.similarity import character_error_rate

        assert character_error_rate("کتابِ یاسمین", "كتاب ياسمين") == 0.0

    def test_cer_empty_inputs(self) -> None:
        from voiceclone.evaluation.similarity import character_error_rate

        assert character_error_rate("", "چیزی") is None
        assert character_error_rate("چیزی", "") is None

    def test_similarity_backends_reported(self) -> None:
        from voiceclone.evaluation.similarity import available_backends

        assert set(available_backends()) == {"speechbrain", "resemblyzer"}


class TestBestOfNSelection:
    """The ranking rule, exercised without loading any model.

    Getting this backwards is silent: you still get audio, it is just the
    wrong take. Both metrics are quantised to their measured noise first -
    CER clamped at 8% (Whisper scores 7.8% transcribing this project's real
    human clips against their own transcripts), similarity counted in 0.02
    steps below the contending best (ECAPA's spread across identity-preserving
    perturbations). Numbers below are from real runs.

    The primary metric was word error rate until WER was measured at 37.4% on
    genuine human speech here - Persian compounds have no fixed spacing and
    Whisper picks its own, so the ranking was partly sorting on whitespace.
    """

    FLOOR = 0.08
    STEP = 0.02

    @classmethod
    def _keys(cls, scored: list[tuple[float | None, float | None]]) -> list[tuple[float, float]]:
        # Mirrors Studio._render_best_of: the similarity reference comes from
        # the takes that tie on the primary key, never the batch maximum.
        ranked = [max(w, cls.FLOOR) if w is not None else cls.FLOOR for w, _ in scored]
        contending = min(ranked)
        observed = [
            sim
            for (_, sim), r in zip(scored, ranked, strict=True)
            if sim is not None and r == contending
        ]
        best = max(observed) if observed else None
        return [_candidate_rank(w, sim, best, cls.FLOOR, cls.STEP) for w, sim in scored]

    @classmethod
    def _pick(cls, scored: list[tuple[float | None, float | None]]) -> int:
        keys = cls._keys(scored)
        return min(range(len(scored)), key=lambda i: keys[i])

    @classmethod
    def _tied(cls, scored: list[tuple[float | None, float | None]]) -> int:
        keys = cls._keys(scored)
        return keys.count(min(keys))

    def test_similarity_decides_inside_the_noise_band(self) -> None:
        # 0% vs 6.7% CER is Whisper disagreeing with itself; 0.76 vs 0.83 is
        # wider than ECAPA's noise, so it is a real preference.
        assert self._pick([(0.000, 0.76), (0.067, 0.83)]) == 1

    def test_a_real_mistake_outranks_any_similarity_gain(self) -> None:
        # Live best-of-4: take 0 is more similar but drops a word.
        assert self._pick([(0.333, 0.788), (0.200, 0.700)]) == 1

    def test_the_most_similar_take_loses_when_it_is_broken(self) -> None:
        # From the temperature grid: 0.823 at 26.7% WER was the least
        # intelligible render in it.
        assert self._pick([(0.267, 0.823), (0.000, 0.786)]) == 1

    def test_all_inside_the_floor_leaves_similarity_to_decide(self) -> None:
        # Three takes, every CER inside the recogniser's noise, so the raw
        # error ordering must not matter at all. These are real baseline
        # figures: renders here land at 1.5-4.4% CER against a 7.8% floor.
        assert self._pick([(0.015, 0.70), (0.044, 0.83), (0.000, 0.75)]) == 1

    def test_least_broken_take_survives_when_everything_is_bad(self) -> None:
        # Four takes of a collapsed IPA render; similarity means nothing
        # there, so the one that kept the most words wins.
        assert self._pick([(1.00, 0.526), (1.00, 0.645), (0.933, 0.515)]) == 2

    def test_similarity_inside_ecapa_noise_is_a_tie(self) -> None:
        # The 0.788/0.782 pair I once agonised over is inside one step.
        assert self._tied([(0.20, 0.788), (0.20, 0.782)]) == 2

    @pytest.mark.parametrize("shift", [0.0, 0.002, 0.05, -0.1, 0.199])
    def test_the_tie_test_survives_translation(self, shift: float) -> None:
        # The property is about separation, so sliding both scores must not
        # change the verdict. A fixed grid failed exactly here: 0.788/0.782
        # tied but 0.790/0.784 - same gap - did not.
        assert self._tied([(0.20, 0.788 + shift), (0.20, 0.782 + shift)]) == 2

    @pytest.mark.parametrize("shift", [0.0, 0.002, 0.05, -0.1])
    def test_a_gap_wider_than_the_step_ranks_under_translation(self, shift: float) -> None:
        # 0.646 vs 0.671 from the reference sweep - more than one step apart.
        assert self._tied([(0.20, 0.646 + shift), (0.20, 0.671 + shift)]) == 1

    def test_a_hair_apart_is_never_ranked(self) -> None:
        # The fixed grid separated these two, 100x below ECAPA's noise.
        assert self._tied([(0.20, 0.7899), (0.20, 0.7901)]) == 2

    def test_a_wide_gap_is_never_tied(self) -> None:
        # ...while tying these, fifty times further apart.
        assert self._tied([(0.20, 0.7901), (0.20, 0.8199)]) == 1

    def test_unscored_wer_ranks_at_the_floor(self) -> None:
        # ASR unavailable is not evidence of a bad take.
        assert self._pick([(None, 0.88), (0.05, 0.70)]) == 0

    def test_an_unscored_take_beats_a_broken_one(self) -> None:
        assert self._pick([(None, 0.60), (0.40, 0.95)]) == 0

    def test_unscored_similarity_does_not_win_on_that_axis(self) -> None:
        # Asymmetric with CER on purpose: the floor is a real "no worse than"
        # bound, whereas an unscored take cannot be *shown* to match.
        assert self._pick([(0.05, None), (0.05, 0.70)]) == 1

    def test_everything_ties_when_no_similarity_is_available(self) -> None:
        # ECAPA missing entirely: CER decides, and the tie is visible so the
        # caller can say so rather than implying the winner was better. All
        # three sit inside the floor, so none of them is measurably worse.
        assert self._tied([(0.05, None), (0.07, None), (0.02, None)]) == 3

    def test_a_disqualified_take_does_not_anchor_the_similarity_scale(self) -> None:
        # The r = -0.06 case: the most similar take (0.823) is the least
        # intelligible, so CER rules it out. Anchoring the similarity scale on
        # it split two clean contenders 0.0002 apart into different buckets -
        # a hundred times below the noise the step encodes.
        scored = [(0.267, 0.8230), (0.000, 0.7031), (0.000, 0.7029)]
        assert self._tied(scored) == 2
        assert self._pick(scored) in {1, 2}

    def test_no_take_carries_a_negative_similarity_key(self) -> None:
        # A take outside the contending tier sits above its best. Inert while
        # CER leads the tuple, but it must not depend on that.
        keys = self._keys([(0.267, 0.8230), (0.000, 0.7031)])
        assert all(key[1] >= 0 for key in keys)

    def test_zero_step_compares_raw_scores(self) -> None:
        # 0 reads like "do not quantise", so it must not mean "ignore".
        keys = [
            _candidate_rank(0.20, sim, 0.7901, self.FLOOR, 0.0)
            for sim in (0.7899, 0.7901)
        ]
        assert keys[1] < keys[0]


class TestSimilarityVerdicts:
    """The two similarity scales, pinned against what the benchmark measured.

    ECAPA-TDNN on CPU, every clip through the same peak-normalising path:
    different speakers average 0.132 and never exceed 0.508 over 900 pairs, two
    different clips of one speaker average 0.689 over 90. Scoring a clone
    against its own prompt adds 0.11 to 0.16 over scoring it against the
    speaker's other clips, so that comparison cannot speak to identity - why it
    adds that much is not established.
    """

    def test_the_constants_match_the_calibration_that_was_measured(self) -> None:
        # The app's verdicts and the benchmark's scale drifted apart once
        # already: results.py kept an early 39-pair sample while the benchmark
        # had moved to 900. Anything that re-runs the calibration has to come
        # back through here.
        import json
        from pathlib import Path

        from voiceclone.pipelines import results as R

        calibration = json.loads(
            (Path(__file__).resolve().parents[1] / "eval/results/baseline.json").read_text()
        )["calibration"]
        assert pytest.approx(calibration["floor"], abs=0.001) == R.SIMILARITY_DIFFERENT_SPEAKER
        assert pytest.approx(calibration["cross_max"], abs=0.001) == R.SIMILARITY_DIFFERENT_SPEAKER_MAX
        assert pytest.approx(calibration["ceiling"], abs=0.001) == R.SIMILARITY_SAME_SPEAKER

    def test_held_out_inside_the_stranger_range_is_not_a_match(self) -> None:
        # A real F5 render: 0.456 held-out, inside the different-speaker range.
        scores = QualityScores(reference_similarity=0.731, held_out_similarity=0.456)
        assert "not distinguishable" in scores.held_out_verdict

    def test_held_out_at_the_human_ceiling_is_a_match(self) -> None:
        # 0.689 is the mean same-speaker pair; at or above it, the render is as
        # close as one of the speaker's own other recordings.
        scores = QualityScores(held_out_similarity=0.70)
        assert "same speaker" in scores.held_out_verdict

    def test_a_human_pair_is_never_called_a_fair_match(self) -> None:
        # Two genuine recordings of one person score around 0.689. The old bands
        # returned "fair match" for that, which reached users in both front ends.
        assert "fair" not in QualityScores(held_out_similarity=0.689).held_out_verdict

    def test_reference_verdict_never_claims_identity(self) -> None:
        # This comparison shares the prompt's channel and cannot show identity,
        # so no wording here may imply a speaker judgement.
        for value in (0.95, 0.80, 0.70, 0.55, 0.30):
            verdict = QualityScores(reference_similarity=value).reference_verdict
            assert "speaker" not in verdict.lower()

    def test_summary_leads_with_the_honest_number(self) -> None:
        text = QualityScores(reference_similarity=0.731, held_out_similarity=0.456).summary()
        assert text.index("held-out") < text.index("reference clip")

    def test_reference_only_summary_flags_what_it_is_not(self) -> None:
        text = QualityScores(reference_similarity=0.731).summary()
        assert "not identity transfer" in text

    def test_unmeasured_scores_say_so(self) -> None:
        scores = QualityScores()
        assert scores.held_out_verdict == "not measured"
        assert scores.reference_verdict == "not measured"


class TestReferenceTranscriptGate:
    """Both gates on the transcript F5 conditions on."""

    def test_a_mid_sentence_start_is_caught(self) -> None:
        # The real case: high confidence, corrupt transcript, no warning.
        transcript = Transcript(text="یا بدون نسخ مصرف می‌کنید و همراه داشته باشید.", avg_logprob=-0.3)
        assert transcript.starts_mid_sentence
        assert transcript.is_confident  # which is why confidence alone missed it

    def test_a_whole_utterance_passes(self) -> None:
        transcript = Transcript(text="گل زیبای باغ ما امروز شکفت.", avg_logprob=-0.3)
        assert not transcript.starts_mid_sentence

    def test_the_confidence_floor_is_tighter_than_it_was(self) -> None:
        # -0.9 passed the old -1.0 threshold.
        assert not Transcript(text="سلام", avg_logprob=-0.9).is_confident
        assert Transcript(text="سلام", avg_logprob=-0.3).is_confident

    def test_missing_confidence_is_not_treated_as_failure(self) -> None:
        assert Transcript(text="سلام").is_confident
