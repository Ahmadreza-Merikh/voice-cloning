"""Config layering, chunking, the text pipeline, engines and the voice library."""

from __future__ import annotations

import numpy as np
import pytest

from voiceclone.audio.io import Audio
from voiceclone.config import AppConfig, deep_merge
from voiceclone.engines import engine_status, list_tts_engines, list_vc_engines
from voiceclone.errors import EngineNotAvailableError, VoiceCloneError
from voiceclone.models import parse_hf_uri
from voiceclone.pipelines.voices import VoiceLibrary, slugify
from voiceclone.text.chunker import chunk_text, estimate_duration, split_sentences
from voiceclone.text.g2p import compact_word_to_ipa, find_g2p_repo
from voiceclone.text.pipeline import TextPipeline


class TestConfig:
    def test_defaults_load(self, config: AppConfig) -> None:
        assert config.audio.sample_rate == 24000
        assert config.engines.tts in {"f5", "moss"}

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

    def test_similarity_backends_reported(self) -> None:
        from voiceclone.evaluation.similarity import available_backends

        assert set(available_backends()) == {"speechbrain", "resemblyzer"}


class TestBestOfNSelection:
    """The gate-then-rank rule, exercised without loading any model.

    Getting this backwards is silent: you still get audio, it is just the
    wrong take. Whisper's own floor on real Persian speech is ~11% WER, so
    ranking candidates by WER ranks noise - and speaker similarity, measured
    across a temperature/seed grid, is uncorrelated with it (r = -0.06). The
    most similar take in that grid was the *least* intelligible one.
    """

    @staticmethod
    def _pick(scored: list[tuple[float | None, float]], gate: float = 0.15) -> int:
        """Mirror of Studio._render_best_of's decision, over indices."""
        rows = [(wer, sim, index) for index, (wer, sim) in enumerate(scored)]
        passed = [r for r in rows if r[0] is None or r[0] <= gate]
        if passed:
            return max(passed, key=lambda r: r[1])[2]
        return min(rows, key=lambda r: (r[0] if r[0] is not None else 1.0, -r[1]))[2]

    def test_most_similar_take_wins_when_all_are_intelligible(self) -> None:
        assert self._pick([(0.05, 0.77), (0.10, 0.82), (0.02, 0.75)]) == 1

    def test_a_broken_take_loses_however_similar_it_sounds(self) -> None:
        # The real case: 0.823 similarity at 26.7% WER beat everything on
        # similarity alone, and it is the take that mangles the sentence.
        assert self._pick([(0.267, 0.823), (0.00, 0.786)]) == 1

    def test_wer_does_not_rank_below_the_gate(self) -> None:
        # 0% vs 6.7% is inside Whisper's own error floor - similarity decides.
        assert self._pick([(0.000, 0.76), (0.067, 0.81)]) == 1

    def test_least_broken_take_survives_when_everything_trips_the_gate(self) -> None:
        assert self._pick([(0.40, 0.90), (0.22, 0.60), (0.31, 0.85)]) == 1

    def test_wer_ties_in_the_fallback_break_on_similarity(self) -> None:
        # Two takes at 20.0% WER really happened; picking by list order made
        # the right choice by luck, which is not a property worth keeping.
        assert self._pick([(0.20, 0.70), (0.20, 0.78)]) == 1

    def test_a_real_error_beats_a_marginal_similarity_gain(self) -> None:
        # Straight from a live best-of-4: no take reached the floor, so the
        # fallback ran. Take 0 is +0.006 more similar but drops a word
        # (33.3% vs 20.0%). Widening the gate to admit both would buy that
        # rounding error at the cost of an audible mistake.
        assert self._pick([(0.333, 0.788), (0.200, 0.782)]) == 1

    def test_unscored_takes_are_not_discarded(self) -> None:
        # ASR unavailable => WER is None => the take still competes.
        assert self._pick([(None, 0.88), (0.05, 0.70)]) == 0
