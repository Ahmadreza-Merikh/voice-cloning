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

    def test_should_use_phonemes_follows_config(self, config: AppConfig) -> None:
        pipeline = TextPipeline(config.text)
        assert pipeline.should_use_phonemes("moss")
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
