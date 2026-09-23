"""The ``omnivoice_ipa`` experiment: canonical IPA in, old paths untouched.

Groups, added as the layers land:

* :class:`TestNormalizer` - every mapping row of the convention, the
  unknown-symbol report, stress, punctuation, and the flat/compact modes;
* :class:`TestRawPathPreserved` - ``omnivoice``/``f5`` still receive exactly
  what they received before this module existed;
* engine registration, capability, config and build-script tests follow.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from voiceclone.config import AppConfig
from voiceclone.errors import EngineNotAvailableError
from voiceclone.text.g2p import G2PResult, words_to_ipa
from voiceclone.text.ipa_normalize import (
    CANONICAL_ENGINES,
    NormalizedIPA,
    normalize_ge2pe_ipa_for_omnivoice,
    normalizer_for,
)
from voiceclone.text.pipeline import TextPipeline


class TestNormalizer:
    def test_alef_onset_becomes_glottal_stop(self) -> None:
        # از -> GE2PE compact "@/z": without the compact stream this ʔ is lost.
        result = normalize_ge2pe_ipa_for_omnivoice([("از", "æ z")], ["@/z"])
        assert result.text == "ʔˈaz"
        assert ("@", "ʔ", 1) in [(c.source, c.target, c.count) for c in result.changes]
        assert result.source == "compact"

    @pytest.mark.parametrize(
        ("orth", "phones", "compact", "expected"),
        [
            # vowels: GE2PE's length-blind inventory into the goldens' one
            ("کتاب", "k e t ɒ b e", "ketabe", "ketaːbˈe"),
            ("بید", "b i d", "bid", "bˈiːd"),      # i is always long
            ("بود", "b u d", "bud", "bˈuːd"),      # ... and so is u
            ("اَست", "æ s t", "/st", "ˈast"),      # æ is the short a
            # consonants: tap and script-g, never ASCII
            ("بزرگ", "b o z o r g", "bozorg", "bozˈoɾɡ"),
            # affricates always carry the tie bar
            ("چای", "tʃ ɒ j", "cay", "t͡ʃˈaːj"),
            ("جان", "dʒ ɒ n", "jan", "d͡ʒˈaːn"),
            # stress goes immediately before the LAST vowel letter
            ("خانهٔ", "x ɒ n e j e", "xaneye", "xaːnejˈe"),
            ("روی", "r u j e", "ruye", "ɾuːjˈe"),
        ],
    )
    def test_mapping_rows(self, orth: str, phones: str, compact: str, expected: str) -> None:
        result = normalize_ge2pe_ipa_for_omnivoice([(orth, phones)], [compact])
        assert result.text == expected, orth

    def test_changes_are_counted_per_symbol(self) -> None:
        result = normalize_ge2pe_ipa_for_omnivoice([("بزرگ", "b o z o r g")], ["bozorg"])
        pairs = {(c.source, c.target): c.count for c in result.changes}
        assert pairs[("r", "ɾ")] == 1
        assert pairs[("g", "ɡ")] == 1

    def test_changes_sorted_most_frequent_first(self) -> None:
        # رررگگ is nonsense phonetically but pins the ordering contract.
        result = normalize_ge2pe_ipa_for_omnivoice([("x", "r r r g g")], ["rrrgg"])
        counts = [change.count for change in result.changes]
        assert counts == sorted(counts, reverse=True)
        assert result.changes[0].source == "r"

    def test_unknown_symbols_are_kept_and_reported(self) -> None:
        result = normalize_ge2pe_ipa_for_omnivoice([("واژه", "qˤ e")], None)
        assert "qˤ" in result.unknown
        assert "qˤ" in result.text  # kept, not dropped
        assert any("Unknown symbol" in note for note in result.notes)

    def test_punctuation_is_restored_from_orthography(self) -> None:
        # GE2PE's compact drops the full stop; the goldens keep it as its own
        # token, so it is recovered from the orthography and spaced off.
        result = normalize_ge2pe_ipa_for_omnivoice([("است.", "æ s t")], ["@/st"])
        assert result.text == "ʔˈast ."
        assert result.stats["punct_restored"] == 1
        assert ("@", "ʔ", 1) in [(c.source, c.target, c.count) for c in result.changes]

    def test_persian_comma_becomes_ascii(self) -> None:
        result = normalize_ge2pe_ipa_for_omnivoice([("بلند،", "b o l nd")], ["boland"])
        pairs = {(c.source, c.target) for c in result.changes}
        assert ("،", ",") in pairs
        assert result.text.endswith(" ,")

    @pytest.mark.parametrize("orth", ["گفت:", "«کتاب»", "گفت؛"])
    def test_punctuation_the_goldens_drop_is_dropped_and_logged(self, orth: str) -> None:
        result = normalize_ge2pe_ipa_for_omnivoice([(orth, "g o f t")], ["goft"])
        dropped = {(c.source, c.target) for c in result.changes if c.target == ""}
        assert any(source for source, target in dropped if source in orth)
        assert all(char not in result.text for char in "،:؛«»")

    def test_stress_digit_is_removed_and_logged(self) -> None:
        result = normalize_ge2pe_ipa_for_omnivoice([("کتاب", "k e t ɒ b")], ["ke1tab"])
        assert result.text == "ketˈaːb"
        assert result.stats["stress_digit_removed"] == 1
        assert ("1", "", 1) in [(c.source, c.target, c.count) for c in result.changes]
        assert any("stress digit" in note for note in result.notes)

    def test_flat_output_cannot_take_stress_and_says_so(self) -> None:
        # The flat fallback returns one entry for the whole sentence: word
        # boundaries (and therefore per-word stress) are gone. Reported, not
        # guessed.
        result = normalize_ge2pe_ipa_for_omnivoice([("یک جمله بلند", "g o l m æ n")], None)
        assert result.source == "flat"
        assert "ˈ" not in result.text
        assert result.stats["stress_skipped"] == 1
        assert any("no word boundaries" in note for note in result.notes)

    def test_ipa_fallback_notes_the_missing_compact_stream(self) -> None:
        result = normalize_ge2pe_ipa_for_omnivoice(
            [("از", "æ z"), ("عمل", "æ m æ l")], None
        )
        assert result.source == "ipa"
        assert any("alef onset" in note for note in result.notes)
        # ... and the mapping itself still runs: ɒ/æ/r/g rows apply.
        assert "a" in result.text

    def test_stress_is_not_inserted_twice(self) -> None:
        result = normalize_ge2pe_ipa_for_omnivoice([("x", "ʔestenfˈuːɾd")], None)
        assert result.text == "ʔestenfˈuːɾd"
        assert result.stats.get("stress_already_present") == 1

    def test_canonical_input_survives_unchanged(self) -> None:
        # Already-canonical phones pass through: length is not doubled.
        result = normalize_ge2pe_ipa_for_omnivoice([("x", "biːd")], None)
        assert result.text == "bˈiːd"
        assert "ːː" not in result.text

    def test_empty_input(self) -> None:
        result = normalize_ge2pe_ipa_for_omnivoice([])
        assert result.text == ""
        assert result.source == "empty"
        assert result.notes

    def test_text_is_the_join_of_words(self) -> None:
        result = normalize_ge2pe_ipa_for_omnivoice(
            [("از", "æ z"), ("کتاب", "k e t ɒ b e")], ["@/z", "ketabe"]
        )
        assert result.text == " ".join(result.words)

    def test_to_dict_is_json_ready(self) -> None:
        result = normalize_ge2pe_ipa_for_omnivoice([("از", "æ z")], ["@/z"])
        payload = json.loads(json.dumps(result.to_dict()))
        assert set(payload) == {"text", "words", "changes", "unknown", "notes", "stats", "source"}
        assert payload["text"] == "ʔˈaz"
        assert {"source", "target", "count"} <= set(payload["changes"][0])

    def test_report_is_a_dataclass_contract(self) -> None:
        assert isinstance(
            normalize_ge2pe_ipa_for_omnivoice([("از", "æ z")], ["@/z"]), NormalizedIPA
        )


class TestNormalizerRegistry:
    def test_only_the_fine_tune_is_canonical(self) -> None:
        assert CANONICAL_ENGINES == {"omnivoice_ipa"}
        assert normalizer_for("omnivoice_ipa") is normalize_ge2pe_ipa_for_omnivoice
        assert normalizer_for("OmniVoice_IPA") is normalize_ge2pe_ipa_for_omnivoice

    @pytest.mark.parametrize("engine", ["omnivoice", "f5", "moss", "chatterbox", "", None])
    def test_every_other_engine_gets_nothing(self, engine: str | None) -> None:
        assert normalizer_for(engine or "") is None


class TestG2PResult:
    def test_compact_words_defaults_empty(self) -> None:
        assert G2PResult("text", "phonemes", [("a", "b")], "backend").compact_words == []

    def test_old_positional_construction_still_works(self) -> None:
        # The field is additive: nothing that constructs G2PResult today moves.
        result = G2PResult("t", "p", [("x", "y")], "b")
        assert (result.text, result.phonemes, result.words, result.backend) == (
            "t",
            "p",
            [("x", "y")],
            "b",
        )


class _FakeG2P:
    """A pinned G2P: same shapes as the real one, no model to load."""

    backend = "fake (pinned output)"
    status = "fake"

    def __init__(self, words: list[tuple[str, str]], compact: list[str]) -> None:
        self._words = words
        self._compact = compact

    def phonemize(self, text: str) -> G2PResult:
        phonemes = " ".join(phones for _, phones in self._words)
        return G2PResult(text, phonemes, list(self._words), self.backend, list(self._compact))


class TestRawPathPreserved:
    """The old engines must see GE2PE exactly as they did before, no matter
    what the normaliser would have done."""

    @staticmethod
    def _pipeline(fake: _FakeG2P) -> TextPipeline:
        pipeline = TextPipeline(AppConfig.load().text)
        pipeline._g2p = fake  # injected before the property would build one
        return pipeline

    def test_omnivoice_still_gets_raw_ge2pe_ipa(self) -> None:
        pipeline = self._pipeline(_FakeG2P([("از", "æ z")], ["@/z"]))
        prepared = pipeline.prepare("از", engine_name="omnivoice", use_phonemes=True)
        assert prepared.used_phonemes
        # words_to_ipa's exact historical output: no ʔ, no ˈ, æ untouched.
        assert prepared.chunks[0].phonemes == words_to_ipa([("از", "æ z")]) == "æz"

    def test_f5_still_gets_raw_ge2pe_ipa(self) -> None:
        pipeline = self._pipeline(_FakeG2P([("کتاب", "k e t ɒ b e")], ["ketabe"]))
        prepared = pipeline.prepare("کتاب", engine_name="f5", use_phonemes=True)
        assert prepared.chunks[0].phonemes == "ketɒbe"  # no canonicalisation

    def test_the_fine_tune_gets_canonical_ipa(self) -> None:
        pipeline = self._pipeline(_FakeG2P([("از", "æ z")], ["@/z"]))
        prepared = pipeline.prepare("از", engine_name="omnivoice_ipa", use_phonemes=True)
        assert prepared.chunks[0].phonemes == "ʔˈaz"
        assert any("canonicalised" in note for note in prepared.notes)

class TestGraphemeRoutingUntouchedWithoutExplicitRequest:
    def test_f5_gets_no_phonemes_from_config(self) -> None:
        # The config lists only the fine-tune; f5 stays on script.
        pipeline = TextPipeline(AppConfig.load().text)
        pipeline._g2p = _FakeG2P([("از", "æ z")], ["@/z"])
        prepared = pipeline.prepare("از", engine_name="f5")
        assert not prepared.used_phonemes
        assert prepared.chunks[0].phonemes == ""


class TestConfig:
    def test_default_engine_unchanged(self, config: AppConfig) -> None:
        assert config.engines.tts == "omnivoice"
        assert config.engines.vc == "cascade"

    def test_apply_to_engines_lists_only_the_fine_tune(self, config: AppConfig) -> None:
        assert config.text.g2p.apply_to_engines == ["omnivoice_ipa"]

    def test_dataclass_default_matches_the_yaml(self) -> None:
        # A config built without the YAML must route the same way.
        from voiceclone.config import G2PConfig

        assert G2PConfig().apply_to_engines == ["omnivoice_ipa"]

    def test_fine_tune_section_exists_with_expected_values(self, config: AppConfig) -> None:
        from voiceclone.config import OmniVoiceIPAConfig

        settings = config.engines.omnivoice_ipa
        assert isinstance(settings, OmniVoiceIPAConfig)
        assert isinstance(settings, type(config.engines.omnivoice))  # a subclass
        assert settings.model_dir == "omnivoice-ipa"
        assert settings.float16_copy is False
        # provenance: where the diff came from, not what the worker loads
        assert settings.model_id == "christopherthompson81/omnivoice-ipa-onnx"

    @pytest.mark.parametrize(
        "field",
        [
            "python",
            "language",
            "dtype",
            "num_step",
            "guidance_scale",
            "t_shift",
            "denoise",
            "postprocess_output",
            "layer_penalty_factor",
            "position_temperature",
            "class_temperature",
            "timeout_seconds",
            "memory_guard",
            "min_available_gb",
        ],
    )
    def test_generation_defaults_identical_to_omnivoice(
        self, config: AppConfig, field: str
    ) -> None:
        # An A/B comparison may differ only in weights and input text.
        parent = getattr(config.engines.omnivoice, field)
        child = getattr(config.engines.omnivoice_ipa, field)
        assert child == parent, field

    def test_should_use_phonemes_follows_the_engine_name(self, config: AppConfig) -> None:
        from voiceclone.text.pipeline import TextPipeline

        pipeline = TextPipeline(config.text)
        assert pipeline.should_use_phonemes("omnivoice_ipa") is True
        for engine in ("omnivoice", "f5", "moss", "chatterbox", "pocket"):
            assert pipeline.should_use_phonemes(engine) is False, engine


class TestEngineRegistration:
    def test_registered_under_its_own_name(self) -> None:
        from voiceclone.engines import list_tts_engines

        engines = list_tts_engines()
        assert "omnivoice_ipa" in engines
        assert engines["omnivoice_ipa"].name == "omnivoice_ipa"

    def test_accepts_phonemes_is_the_only_capability_that_changed(self) -> None:
        from voiceclone.engines import list_tts_engines

        engines = list_tts_engines()
        parent = engines["omnivoice"].capabilities
        child = engines["omnivoice_ipa"].capabilities
        assert child.accepts_phonemes is True
        assert parent.accepts_phonemes is False  # unchanged
        assert engines["f5"].capabilities.accepts_phonemes is False
        assert engines["moss"].capabilities.accepts_phonemes is True  # unchanged
        # everything else identical: the comparison must not drift on it
        for field in (
            "requires_reference_text",
            "native_persian",
            "supports_speed",
            "supports_seed",
            "approx_download_gb",
            "max_reference_seconds",
        ):
            assert getattr(child, field) == getattr(parent, field), field

    def test_get_engine_builds_one(self, config: AppConfig) -> None:
        from voiceclone.device import get_device_spec
        from voiceclone.engines import get_tts_engine

        engine = get_tts_engine("omnivoice_ipa", config, get_device_spec("cpu"))
        assert engine.name == "omnivoice_ipa"
        assert engine.capabilities.accepts_phonemes is True


class TestModelPaths:
    def test_default_dir_is_under_the_cache(self) -> None:
        from voiceclone.engines.omnivoice_ipa import default_model_dir

        assert default_model_dir().name == "omnivoice-ipa"
        assert "persian-voice-cloning" in str(default_model_dir())

    def test_relative_model_dir_resolves_under_cache_dir(self) -> None:
        from voiceclone.engines.omnivoice_ipa import ipa_model_dir

        # Fresh load: the session `config` fixture must not be mutated here.
        config = AppConfig.load()
        config.engines.omnivoice_ipa.model_dir = "somewhere-else"
        resolved = ipa_model_dir(config)
        assert resolved.name == "somewhere-else"
        assert str(resolved).startswith(str(config.paths.cache_dir).replace("~", str(Path.home())))

    def test_absolute_model_dir_wins_over_cache_dir(self) -> None:
        from voiceclone.engines.omnivoice_ipa import ipa_model_dir

        config = AppConfig.load()
        config.engines.omnivoice_ipa.model_dir = "/tmp/custom-ipa-weights"
        assert str(ipa_model_dir(config)) == "/tmp/custom-ipa-weights"


class TestEngineAvailability:
    """The weights are not installed: report it, never crash on it."""

    @staticmethod
    def _with_env_ok(monkeypatch: pytest.MonkeyPatch) -> None:
        # Availability of the *weights* is what is under test; the venv check
        # is the parent's and must not decide these answers.
        from voiceclone.engines.omnivoice_tts import OmniVoiceEngine

        monkeypatch.setattr(
            OmniVoiceEngine, "is_available", classmethod(lambda cls: (True, ""))
        )

    def test_missing_weights_report_unavailable_with_the_build_command(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from voiceclone.engines import omnivoice_ipa

        self._with_env_ok(monkeypatch)
        monkeypatch.setattr(omnivoice_ipa, "_candidate_model_dirs", lambda: [tmp_path / "nowhere"])
        available, reason = omnivoice_ipa.OmniVoiceIPAEngine.is_available()
        assert available is False
        assert "build_omnivoice_ipa" in reason
        assert "IPA-adapted" in reason or "not built" in reason

    def test_built_weights_are_available(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from voiceclone.engines import omnivoice_ipa

        self._with_env_ok(monkeypatch)
        (tmp_path / omnivoice_ipa.PROVENANCE_NAME).write_text("{}", encoding="utf-8")
        monkeypatch.setattr(omnivoice_ipa, "_candidate_model_dirs", lambda: [tmp_path])
        assert omnivoice_ipa.OmniVoiceIPAEngine.is_available() == (True, "")

    def test_missing_weights_raise_before_any_worker_starts(
        self, tmp_path: Path
    ) -> None:
        # Instance-level guard: the configured directory (may be a custom one
        # the classmethod cannot see) is checked before the process spawns.
        from voiceclone.engines.omnivoice_ipa import OmniVoiceIPAEngine

        config = AppConfig.load()
        config.engines.omnivoice_ipa.model_dir = str(tmp_path / "not-built")
        stub = SimpleNamespace(config=config)

        with pytest.raises(EngineNotAvailableError) as excinfo:
            OmniVoiceIPAEngine._require_weights(stub)  # type: ignore[arg-type]
        assert "build_omnivoice_ipa" in f"{excinfo.value} {excinfo.value.hint or ''}"
        # no subprocess, no torch load - the guard ran first

    def test_model_source_is_the_configured_directory(self) -> None:
        from voiceclone.engines.omnivoice_ipa import OmniVoiceIPAEngine, ipa_model_dir

        config = AppConfig.load()
        config.engines.omnivoice_ipa.model_dir = str(Path("/tmp/some-dir"))
        stub = SimpleNamespace(config=config)

        settings = OmniVoiceIPAEngine._settings(stub)  # type: ignore[arg-type]
        source = OmniVoiceIPAEngine._model_source(stub, settings)  # type: ignore[arg-type]
        assert source == str(ipa_model_dir(config))
        # never the diff repo: the worker would find no model files there
        assert "omnivoice-ipa-onnx" not in source


class TestPhonemeFlagWarning:
    def test_metadata_records_whether_ipa_was_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The parent synthesize is replaced: this tests the wrapper's contract
        # (flag propagated, raw-text warning) without a worker process.
        from voiceclone.device import get_device_spec
        from voiceclone.engines.base import SynthesisResult
        from voiceclone.engines.omnivoice_ipa import OmniVoiceIPAEngine
        from voiceclone.engines.omnivoice_tts import OmniVoiceEngine

        captured: dict[str, object] = {}

        def fake_synthesize(self, text, reference, **kwargs):  # noqa: ANN001
            captured.update(kwargs)
            captured["text"] = text
            return SynthesisResult(
                audio=None, engine="omnivoice_ipa", seed=0, metadata={}
            )

        monkeypatch.setattr(OmniVoiceEngine, "synthesize", fake_synthesize)
        engine = OmniVoiceIPAEngine(AppConfig.load(), get_device_spec("cpu"))

        result = engine.synthesize("ʔˈaz", None, is_phonemes=True)
        assert result.metadata["is_phonemes"] is True
        assert captured["is_phonemes"] is True

        result = engine.synthesize("از", None)  # raw text: warned, recorded
        assert result.metadata["is_phonemes"] is False

    def test_synthesis_result_metadata_is_mutable_enough(self) -> None:
        from voiceclone.engines.base import SynthesisResult

        result = SynthesisResult(audio=None, engine="x", seed=0, metadata={})
        result.metadata["is_phonemes"] = True
        assert result.metadata["is_phonemes"] is True
