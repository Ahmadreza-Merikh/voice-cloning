"""The ``omnivoice_ipa`` experiment: canonical IPA in, old paths untouched.

Groups, added as the layers land:

* :class:`TestNormalizer` - every mapping row of the convention, the
  unknown-symbol report, stress, punctuation, and the flat/compact modes;
* :class:`TestRawPathPreserved` - ``omnivoice``/``f5`` still receive exactly
  what they received before this module existed;
* engine registration, capability, config and build-script tests follow;
* :class:`TestABHarness` - the A/B/C harness: sentence assembly, job
  building, the phoneme capture, and the manifest plumbing.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def _load_build_script():
    """The build script as a module (no onnx/network needed to import it)."""
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "build_omnivoice_ipa_under_test", root / "scripts" / "build_omnivoice_ipa.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Register before executing: dataclasses resolves string annotations
    # (from `from __future__ import annotations`) through sys.modules, and
    # @dataclass on ModulePlan raises without it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def build_script():
    return _load_build_script()


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
        assert {"omnivoice_ipa"} == CANONICAL_ENGINES
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
        # An A/B comparison may differ only in weights, input text and the
        # fine-tune card's language tag (the dedicated test below).
        parent = getattr(config.engines.omnivoice, field)
        child = getattr(config.engines.omnivoice_ipa, field)
        assert child == parent, field

    def test_ipa_path_is_language_agnostic_per_the_fine_tune_card(
        self, config: AppConfig
    ) -> None:
        # The single documented divergence from the base engine: the published
        # card conditions the fine-tune on <|lang_start|>None<|lang_end|> and
        # says "the IPA stream carries everything". It never trained with the
        # base ``fa`` tag next to IPA text, so the IPA path must not send one;
        # the base engine keeps upstream's named-language default untouched.
        assert config.engines.omnivoice.language == "fa"
        assert config.engines.omnivoice_ipa.language is None
        # The worker's wire format is a string; upstream's _resolve_language
        # maps the literal "None" to language-agnostic mode.
        assert str(config.engines.omnivoice_ipa.language) == "None"

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


class TestBuildScript:
    """The fold: math, safetensors plumbing, and the contracts between them.

    Everything here runs without onnx, without a network and without the
    2.45 GB file - the script is written so its correctness can be pinned by
    small fixtures, and the real build re-checks itself against the base.
    """

    def test_provenance_constant_matches_the_engine(self) -> None:
        from voiceclone.engines.omnivoice_ipa import PROVENANCE_NAME as engine_marker

        assert _load_build_script().PROVENANCE_NAME == engine_marker == "IPA_ADAPTATION.json"

    def test_default_out_matches_the_engine_default_dir(self) -> None:
        from voiceclone.engines.omnivoice_ipa import default_model_dir

        script = _load_build_script()
        assert Path(script.DEFAULT_OUT).expanduser() == default_model_dir()

    def test_imports_nothing_from_the_project(self) -> None:
        # The build must run in a bare interpreter: numpy + onnx only.
        # (Mentions of the project's paths in comments are fine - imports are not.)
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "build_omnivoice_ipa.py").read_text(encoding="utf-8")
        assert "from voiceclone" not in source
        assert "import voiceclone" not in source
        assert "from .." not in source

    def test_parse_safetensors_header(self, tmp_path: Path) -> None:
        import struct

        header = json.dumps(
            {"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
        ).encode("utf-8")
        path = tmp_path / "t.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 24)
        offset, parsed = _load_build_script().parse_safetensors_header(path)
        assert offset == 8 + len(header)
        assert parsed["w"]["shape"] == [2, 3]

    def test_fold_direct_orientation_is_exact(self) -> None:
        script = _load_build_script()
        rng = np.random.default_rng(0)
        weight = rng.normal(size=(8, 4)).astype(np.float32)
        a = rng.normal(size=(2, 4)).astype(np.float32)  # (rank, in)
        b = rng.normal(size=(8, 2)).astype(np.float32)  # (out, rank)
        expected = weight + 2.5 * (b @ a)
        returned = script.fold_lora_delta(weight, a, b, 2.5)
        assert returned is weight
        assert np.allclose(weight, expected, atol=1e-6)

    def test_fold_transposed_storage_lands_the_same_way(self) -> None:
        script = _load_build_script()
        rng = np.random.default_rng(1)
        weight = rng.normal(size=(8, 4)).astype(np.float32)
        a = rng.normal(size=(4, 2)).astype(np.float32)  # (in, rank)
        b = rng.normal(size=(2, 8)).astype(np.float32)  # (rank, out)
        expected = weight + 1.5 * ((a @ b).T)
        script.fold_lora_delta(weight, a, b, 1.5)
        assert np.allclose(weight, expected, atol=1e-6)

    def test_fold_refuses_shapes_it_cannot_explain(self) -> None:
        script = _load_build_script()
        weight = np.zeros((8, 4), dtype=np.float32)
        a = np.zeros((3, 5), dtype=np.float32)
        b = np.zeros((7, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="no orientation"):
            script.fold_lora_delta(weight, a, b, 2.0)

    def test_fold_refuses_non_float32_weights(self) -> None:
        script = _load_build_script()
        weight = np.zeros((4, 4), dtype=np.float16)
        a = np.zeros((2, 4), dtype=np.float32)
        b = np.zeros((4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="dtype"):
            script.fold_lora_delta(weight, a, b, 2.0)

    def test_apply_embed_rows_selects_without_accumulating(self) -> None:
        script = _load_build_script()
        table = np.zeros((10, 4), dtype=np.float32)
        idx = np.array([1, 5, 9], dtype=np.int32)
        rows = np.ones((3, 4), dtype=np.float16)
        script.apply_embed_rows(table, idx, rows)
        assert np.all(table[[1, 5, 9]] == 1.0)
        assert np.all(table[[0, 2, 3, 4]] == 0.0)  # untouched: no scatter-add

    def test_apply_embed_rows_rejects_out_of_range_indices(self) -> None:
        script = _load_build_script()
        table = np.zeros((10, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="out of range"):
            script.apply_embed_rows(table, np.array([3, 99]), np.ones((2, 4), np.float16))

    def test_apply_embed_rows_rejects_shape_mismatch(self) -> None:
        script = _load_build_script()
        table = np.zeros((10, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="do not fit"):
            script.apply_embed_rows(table, np.array([1, 2, 3]), np.ones((2, 4), np.float16))

    def test_plan_modules_pairs_factors(self) -> None:
        script = _load_build_script()
        arrays = {
            "llm.layers.0.self_attn.q_proj.lora_A": np.zeros((2, 4), np.float32),
            "llm.layers.0.self_attn.q_proj.lora_B": np.zeros((8, 2), np.float32),
            "llm.layers.1.mlp.down_proj.lora_A": np.zeros((2, 8), np.float32),
            "llm.layers.1.mlp.down_proj.lora_B": np.zeros((4, 2), np.float32),
            "embed_rows": np.zeros((3, 4), np.float16),
            "embed_idx": np.array([1, 2, 3], np.int32),
        }
        plans = script.plan_modules(arrays)
        assert [p.weight_key for p in plans] == [
            "llm.layers.0.self_attn.q_proj.weight",
            "llm.layers.1.mlp.down_proj.weight",
        ]
        assert plans[0].a_name.endswith(".lora_A") and plans[0].b_name.endswith(".lora_B")

    def test_plan_modules_refuses_an_unpaired_factor(self) -> None:
        script = _load_build_script()
        with pytest.raises(SystemExit, match="unpaired"):
            script.plan_modules({"x.lora_A": np.zeros((2, 2), np.float32)})

    @staticmethod
    def _matching_header() -> dict:
        return {
            "llm.x.weight": {"dtype": "F32", "shape": [8, 4], "data_offsets": [0, 128]},
            "llm.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [10, 4],
                "data_offsets": [128, 288],
            },
        }

    @staticmethod
    def _matching_arrays() -> dict:
        return {
            "llm.x.lora_A": np.zeros((2, 4), np.float32),
            "llm.x.lora_B": np.zeros((8, 2), np.float32),
            "embed_rows": np.zeros((3, 4), np.float16),
            "embed_idx": np.array([1, 5, 9], np.int32),
        }

    def test_validate_targets_accepts_a_matching_diff(self, build_script) -> None:
        arrays = self._matching_arrays()
        plans = build_script.plan_modules(arrays)
        assert build_script.validate_targets(self._matching_header(), plans, arrays) == [
            "llm.x.weight"
        ]

    def test_validate_targets_rejects_a_missing_weight(self, build_script) -> None:
        arrays = self._matching_arrays()
        header = {
            "llm.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [10, 4],
                "data_offsets": [0, 160],
            }
        }
        with pytest.raises(SystemExit, match="not in the base weights"):
            build_script.validate_targets(header, build_script.plan_modules(arrays), arrays)

    def test_validate_targets_rejects_an_index_outside_the_table(self, build_script) -> None:
        arrays = self._matching_arrays()
        arrays["embed_idx"] = np.array([1, 5, 99], np.int32)  # table has 10 rows
        with pytest.raises(SystemExit, match="embed_idx range"):
            build_script.validate_targets(
                self._matching_header(), build_script.plan_modules(arrays), arrays
            )

    def test_validate_targets_rejects_a_shape_mismatch(self, build_script) -> None:
        arrays = self._matching_arrays()
        arrays["llm.x.lora_A"] = np.zeros((3, 5), np.float32)  # matches neither side
        arrays["llm.x.lora_B"] = np.zeros((7, 6), np.float32)
        with pytest.raises(SystemExit, match="no orientation"):
            build_script.validate_targets(
                self._matching_header(), build_script.plan_modules(arrays), arrays
            )

    def test_sha256_matches_hashlib(self, tmp_path: Path) -> None:
        import hashlib as _hashlib

        path = tmp_path / "blob"
        path.write_bytes(b"the diff, hypothetically")
        expected = _hashlib.sha256(b"the diff, hypothetically").hexdigest()
        assert _load_build_script().sha256_of(path) == expected


def _load_ab_script():
    """The A/B harness as a module (importing it must render nothing)."""
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "ab_omnivoice_ipa_under_test", root / "scripts" / "ab_omnivoice_ipa.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Register before executing, same as the build script: Sentence/Job are
    # @dataclass under `from __future__ import annotations` and resolve
    # their string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def ab_script():
    return _load_ab_script()


class _DriftingG2P:
    """A G2P whose replay disagrees with what prepare was fed.

    The first call (prepare) is the pinned truth; the second (capture's
    replay) returns different phones, which must surface as a drift flag
    rather than be smoothed over.
    """

    backend = "fake (drifts)"
    status = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def phonemize(self, text: str) -> G2PResult:
        self.calls += 1
        if self.calls == 1:
            words, compact = [("از", "æ z")], ["@/z"]
        else:
            words, compact = [("از", "b z")], ["bz"]
        phonemes = " ".join(phones for _, phones in words)
        return G2PResult(text, phonemes, words, self.backend, compact)


class _StubStudio:
    """Studio-shaped double: records the call contract, writes dummy wavs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def synthesize(self, text, reference, **kwargs):  # noqa: ANN001, ANN201
        self.calls.append({"text": text, "reference": reference, **kwargs})
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"RIFF....")
        prepared = SimpleNamespace(
            used_phonemes=kwargs["use_phonemes"],
            chunks=[SimpleNamespace(synth_text=text)],
            notes=["canonicalised"],
        )
        return SimpleNamespace(
            output_path=out,
            duration=1.5,
            text=prepared,
            warnings=["a warning"],
            timings={"total": 1.5},
        )


class TestABHarness:
    """``scripts/ab_omnivoice_ipa.py``: what the comparison is made of.

    Every test here runs without an engine, a worker, or a network call -
    the harness must be fully describable before anything renders.
    """

    # -- fixtures ---------------------------------------------------------
    @staticmethod
    def _sealed() -> dict:
        # Distinct on purpose: proves t09-t12 are read from the manifest,
        # not re-typed into the harness.
        return {
            "texts": [
                {"key": key, "category": "cat", "text": f"متن {key} از مانیفست"}
                for key in ("t09", "t10", "t11", "t12")
            ]
        }

    @staticmethod
    def _corpus() -> dict:
        return {
            "entries": [
                {"speaker": "mana", "role": "reference", "file": "mana_000.wav",
                 "transcript": "متن مرجع", "duration": 11.8},
                {"speaker": "mana", "role": "held_out", "file": "mana_001.wav",
                 "transcript": "یک جمله بلند واقعی از گفت‌وگوی مرجع"},
            ]
        }

    def _sentences(self, ab_script) -> list:
        return ab_script.build_sentences(self._sealed(), self._corpus())

    @staticmethod
    def _real_manifests() -> tuple[dict, dict]:
        root = Path(__file__).resolve().parents[1]
        sealed = json.loads(
            (root / "eval" / "manifests" / "test_texts_v1.json").read_text(encoding="utf-8")
        )
        corpus = json.loads(
            (root / "eval" / "corpus" / "manifest.json").read_text(encoding="utf-8")
        )
        return sealed, corpus

    # -- sentence assembly ------------------------------------------------
    def test_sentences_merge_three_sources_in_order(self, ab_script) -> None:
        sentences = self._sentences(ab_script)
        assert [s.id for s in sentences] == [
            "s01", "s02", "s03", "s04", "s05", "s06", "s07", "s08", "s09",
            "t09", "t10", "t11", "t12", "r01",
        ]
        by_id = {s.id: s for s in sentences}
        # Sealed texts come from the manifest: the fixture text is unique to it.
        assert by_id["t11"].text == "متن t11 از مانیفست"
        assert by_id["t11"].source == "sealed"
        assert by_id["s01"].source == "probe"
        # r01 is the speaker's first held-out transcript, not the reference.
        assert by_id["r01"].text == "یک جمله بلند واقعی از گفت‌وگوی مرجع"
        assert by_id["r01"].source == "corpus"

    def test_missing_sealed_key_exits_rather_than_shrinking_the_set(self, ab_script) -> None:
        sealed = {"texts": self._sealed()["texts"][:2]}  # t09, t10 only
        with pytest.raises(SystemExit, match="sealed manifest is missing t11, t12"):
            ab_script.build_sentences(sealed, self._corpus())

    def test_no_variant_set_is_enforced_not_documented(
        self, ab_script, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # s01 loses its diacritised twin: the default run would silently
        # skip B, so build_sentences must refuse instead.
        reduced = {k: v for k, v in ab_script.DIACRITIZED.items() if k != "s01"}
        monkeypatch.setattr(ab_script, "DIACRITIZED", reduced)
        with pytest.raises(SystemExit, match="disagree with NO_VARIANT"):
            self._sentences(ab_script)

    def test_probe_texts_carry_no_vowel_marks(self, ab_script) -> None:
        # Harakat only - `DIACRITICS` would also match the ezafe hamza in
        # «هفتهٔ», which is ordinary orthography, not a vowel mark.
        harakat = set("ًٌٍَُِّْٰ")
        for key, _, text in ab_script.PROBES:
            assert not harakat & set(text), f"{key} leaked a vowel mark"

    def test_variant_tables_agree_with_the_sentence_universe(self, ab_script) -> None:
        sealed, corpus = self._real_manifests()
        sentences = ab_script.build_sentences(sealed, corpus, speaker="mana")
        ids = {s.id for s in sentences}
        assert set(ab_script.DIACRITIZED) <= ids  # no stray, never-used key
        unmarked = {s.id for s in sentences if not s.diacritized}
        assert unmarked == set(ab_script.NO_VARIANT) & ids == {"s09", "t12", "r01"}

    def test_variants_differ_from_plains_but_say_the_same_words(self, ab_script) -> None:
        # Condition B must speak *the same words* - only marked. CER strips
        # the marks, so A and B end up score-identical by construction.
        from voiceclone.evaluation.similarity import normalize_for_cer

        sealed, corpus = self._real_manifests()
        by_id = {
            s.id: s for s in ab_script.build_sentences(sealed, corpus, speaker="mana")
        }
        harakat = set("ًٌٍَُِّْٰ")
        for key, variant in ab_script.DIACRITIZED.items():
            plain = by_id[key].text
            assert variant != plain, key
            assert harakat & set(variant), f"{key} variant carries no marks"
            assert normalize_for_cer(variant) == normalize_for_cer(plain), key

    def test_select_sentences_filters_and_rejects_unknown_ids(self, ab_script) -> None:
        sentences = self._sentences(ab_script)
        picked = ab_script.select_sentences(sentences, "s01, t11")
        assert [s.id for s in picked] == ["s01", "t11"]  # manifest order kept
        assert ab_script.select_sentences(sentences, None) == sentences
        with pytest.raises(SystemExit, match="unknown sentence id"):
            ab_script.select_sentences(sentences, "s99")

    def test_reference_and_held_out_lookup(self, ab_script) -> None:
        corpus = self._corpus()
        assert ab_script.reference_entry(corpus, "mana")["file"] == "mana_000.wav"
        held = ab_script.held_out_entries(corpus, "mana")
        assert [e["file"] for e in held] == ["mana_001.wav"]
        with pytest.raises(SystemExit, match="no reference clip"):
            ab_script.reference_entry(corpus, "nobody")

    # -- jobs -------------------------------------------------------------
    def test_build_jobs_makes_three_conditions_per_sentence(self, ab_script) -> None:
        sentences = self._sentences(ab_script)
        jobs = ab_script.build_jobs(sentences, seed=7)
        assert len(jobs) == 3 * len(sentences)
        assert len({j.wav_name for j in jobs}) == len(jobs)  # no clobbering

        by_id = {s.id: s for s in sentences}
        marked = by_id["s01"]
        a, b, c = (j for j in jobs if j.id == "s01")
        assert (a.condition, a.engine, a.use_phonemes, a.input_text, a.seed) == (
            "A", "omnivoice", False, marked.text, 7
        )
        assert b.input_text == marked.diacritized and b.skipped is None
        assert (c.condition, c.engine, c.use_phonemes, c.input_text) == (
            "C", "omnivoice_ipa", True, marked.diacritized
        )
        # The G2P reads the diacritised twin where one exists - and the plain
        # text where no twin was written, so s09/t12/r01 keep their own input.
        c_plain = next(j for j in jobs if j.id == "s09" and j.condition == "C")
        assert c_plain.input_text == by_id["s09"].text
        # An unmarked sentence still gets all three jobs: B is recorded, kept.
        skip_b = next(j for j in jobs if j.id == "s09" and j.condition == "B")
        assert skip_b.skipped == "no diacritised variant written for this sentence"
        assert skip_b.input_text == ""

    def test_jobs_follow_the_declared_condition_tables(self, ab_script) -> None:
        # ENGINE_FOR/PHONEMES_FOR are the single source; a hardcoded drift
        # between them and the built jobs would silently change the test.
        jobs = ab_script.build_jobs(self._sentences(ab_script))
        for job in jobs:
            assert job.engine == ab_script.ENGINE_FOR[job.condition]
            assert job.use_phonemes == ab_script.PHONEMES_FOR[job.condition]
        assert ab_script.CONDITIONS == ("A", "B", "C")
        assert ab_script.PHONEMES_FOR == {"A": False, "B": False, "C": True}

    def test_grouping_releases_between_engines(self, ab_script) -> None:
        jobs = ab_script.build_jobs(self._sentences(ab_script))
        groups = ab_script.group_jobs_by_engine(jobs)
        # First-seen order: every A/B render finishes (and is released)
        # before the fine-tune's worker would start.
        assert list(groups) == ["omnivoice", "omnivoice_ipa"]
        assert len(groups["omnivoice"]) == 28  # 14 A + 14 B (3 of them skips)
        assert len(groups["omnivoice_ipa"]) == 14
        assert sum(len(g) for g in groups.values()) == len(jobs)

    def test_expected_text_scores_each_condition_on_its_own_input(self, ab_script) -> None:
        by_id = {s.id: s for s in self._sentences(ab_script)}
        marked, unmarked = by_id["s01"], by_id["s09"]
        assert ab_script.expected_text(marked, "A") == marked.text
        assert ab_script.expected_text(marked, "B") == marked.diacritized
        assert ab_script.expected_text(marked, "C") == marked.diacritized
        assert ab_script.expected_text(unmarked, "B") == unmarked.text
        assert ab_script.expected_text(unmarked, "C") == unmarked.text

    # -- G2P / normaliser capture ----------------------------------------
    def test_capture_records_raw_stream_and_canonical_ipa(self, ab_script) -> None:
        pipeline = TextPipeline(AppConfig.load().text)
        pipeline._g2p = _FakeG2P([("از", "æ z")], ["@/z"])
        capture = ab_script.capture_ipa("از", pipeline)

        assert capture["used_phonemes"] is True
        chunk = capture["chunks"][0]
        assert chunk["chunk_text"] == "از"
        assert chunk["raw_ge2pe_compact"] == "@/z"           # what GE2PE said
        assert chunk["raw_ge2pe_words"] == ["از\tæ z"]
        assert chunk["canonical_ipa"] == "ʔˈaz"               # what C is fed
        assert chunk["matches_pipeline"] is True
        norm = chunk["normalization"]
        assert norm["source"] == "compact"
        assert ("@", "ʔ") in {(c["source"], c["target"]) for c in norm["changes"]}
        assert norm["unknown"] == []

    def test_capture_flags_a_replay_that_disagrees_with_prepare(self, ab_script) -> None:
        pipeline = TextPipeline(AppConfig.load().text)
        pipeline._g2p = _DriftingG2P()
        chunk = ab_script.capture_ipa("از", pipeline)["chunks"][0]
        # Both strings are kept: the manifest can show what diverged.
        assert chunk["canonical_ipa"] == "ʔˈaz"
        assert chunk["normalization"]["text"] == "bz"
        assert chunk["matches_pipeline"] is False

    def test_capture_reports_disabled_g2p_instead_of_phantom_phonemes(self, ab_script) -> None:
        config = AppConfig.load().copy_with(**{"text.g2p.enabled": False})
        pipeline = TextPipeline(config.text)
        capture = ab_script.capture_ipa("از", pipeline)
        assert capture["used_phonemes"] is False  # main() aborts on this
        assert capture["chunks"][0]["error"] == "g2p unavailable"
        assert capture["chunks"][0].get("canonical_ipa", "") == ""

    # -- config, leak check, manifest -------------------------------------
    def test_render_config_turns_off_only_evaluation(self, ab_script, config) -> None:
        before = config.evaluation.enabled
        render_cfg = ab_script.render_config(config)
        assert render_cfg.evaluation.enabled is False
        assert config.evaluation.enabled is before  # copy_with, not a mutation
        assert render_cfg.engines.tts == config.engines.tts == "omnivoice"
        assert render_cfg.text.g2p.apply_to_engines == ["omnivoice_ipa"]
        assert render_cfg.engines.omnivoice.num_step == config.engines.omnivoice.num_step
        assert render_cfg.evaluation.transcribe_check == config.evaluation.transcribe_check

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ("", "متن", False),
            ("متن", "متن", True),
            # 20 letters inside 40: at least 20 and at least half -> hit.
            ("x" * 20, "y" * 10 + "x" * 20 + "y" * 10, True),
            # 19 letters: under the floor, however well contained.
            ("x" * 19, "y" * 10 + "x" * 19 + "y" * 9, False),
            # 30 of 100 letters: contained, but less than half the longer.
            ("x" * 30, "y" * 70 + "x" * 30, False),
            ("الف", "ببب", False),
        ],
    )
    def test_overlap_rule(self, ab_script, a: str, b: str, expected: bool) -> None:
        assert ab_script.overlaps(a, b) is expected

    def test_probe_leak_check_reports_hits_and_checks_only_probes(
        self, ab_script, tmp_path: Path
    ) -> None:
        import gzip

        probe_text = ab_script.PROBES[0][2]
        sentences = [
            ab_script.Sentence("s01", "probe", probe_text, None, "probe"),
            # Same text, but sealed: must NOT be reported - only probes are
            # new sentences that could have leaked into training.
            ab_script.Sentence("t09", "sealed", probe_text, None, "sealed"),
        ]
        splits = tmp_path / "splits.jsonl.gz"
        records = [
            {"text": probe_text, "split": "train", "clip": "c1.wav"},
            {"text": "a completely unrelated english sentence for the split",
             "split": "valid", "clip": "c2.wav"},
        ]
        with gzip.open(splits, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        report = ab_script.check_probe_overlap(
            sentences, splits_path=splits, include_extra_sources=False
        )
        assert report["available"] is True
        assert report["checked"] == 2
        assert report["probes_checked"] == 1
        assert report["overlaps"] == ["s01 overlaps train clip c1.wav"]

    def test_probe_leak_check_survives_a_missing_split(self, ab_script, tmp_path: Path) -> None:
        report = ab_script.check_probe_overlap(
            [], splits_path=tmp_path / "absent.jsonl.gz", include_extra_sources=False
        )
        assert report["available"] is False
        assert "not found" in report["reason"]
        assert report["overlaps"] == []

    def test_write_manifest_is_atomic_and_keeps_persian_readable(
        self, ab_script, tmp_path: Path
    ) -> None:
        path = tmp_path / "nested" / "manifest.json"
        data = {"renders": [{"input_text": "او گل قرمز را از گل باغچه بیرون کشید."}],
                "events": []}
        ab_script.write_manifest(path, data)
        raw = path.read_text(encoding="utf-8")
        assert json.loads(raw) == data
        assert "گل" in raw  # not \uXXXX-escaped, and no .tmp left behind
        assert not path.with_suffix(path.suffix + ".tmp").exists()

    def test_render_entry_records_a_skip_without_a_wav(self, ab_script) -> None:
        skip = ab_script.Job(
            "s09", "B", "omnivoice", False, "",
            42, "no diacritised variant written for this sentence",
        )
        entry = ab_script.render_entry(skip, reference_file="mana_000.wav")
        assert entry["wav"] is None
        assert entry["input_text"] is None  # empty string is not an input
        assert "no diacritised variant" in entry["skipped"]
        assert entry["reference"] == "mana_000.wav" and entry["seed"] == 42

        render = ab_script.Job("s01", "C", "omnivoice_ipa", True, "ʔˈaz", 42)
        entry = ab_script.render_entry(render, reference_file="mana_000.wav")
        assert entry["input_text"] == "ʔˈaz"
        assert entry["use_phonemes_requested"] is True
        assert entry["skipped"] is None

    def test_settings_fields_exist_on_both_engine_configs(self, ab_script) -> None:
        # The manifest promises these fields for both engines; a rename in
        # either config would otherwise surface as a KeyError at render time.
        from voiceclone.config import OmniVoiceConfig, OmniVoiceIPAConfig

        for field in ab_script._SETTINGS_FIELDS:
            assert hasattr(OmniVoiceConfig(), field), field
            assert hasattr(OmniVoiceIPAConfig(), field), field

    # -- CLI --------------------------------------------------------------
    def test_parse_args_defaults_and_flags(self, ab_script) -> None:
        args = ab_script.parse_args([])
        assert (args.seed, args.speaker, args.cer, args.only) == (42, "mana", False, None)
        assert args.out.endswith("omnivoice-ipa-ab-v1")

        args = ab_script.parse_args(["--only", "s01,t10", "--cer", "--seed", "7"])
        assert (args.only, args.cer, args.seed) == ("s01,t10", True, 7)

    # -- rendering contract (stub studio, no engine) ----------------------
    def test_render_group_renders_skips_and_persists_each_step(
        self, ab_script, tmp_path: Path
    ) -> None:
        sentences = ab_script.select_sentences(self._sentences(ab_script), "s01,s09")
        jobs = ab_script.build_jobs(sentences, seed=42)
        studio = _StubStudio()
        manifest, manifest_path = {"renders": [], "events": []}, tmp_path / "manifest.json"
        reference = tmp_path / "ref.wav"
        reference.write_bytes(b"RIFF")

        ab_script.render_group(
            studio, jobs,
            out_dir=tmp_path, reference_path=reference,
            reference_text="متن مرجع", reference_file="mana_000.wav",
            manifest=manifest, manifest_path=manifest_path,
        )

        # 6 jobs, one recorded skip -> 5 renders, all else held identical.
        assert len(jobs) == 6 and sum(1 for j in jobs if j.skipped) == 1
        assert len(studio.calls) == 5
        assert [c["engine"] for c in studio.calls] == [
            "omnivoice", "omnivoice", "omnivoice_ipa", "omnivoice", "omnivoice_ipa",
        ]
        assert [c["use_phonemes"] for c in studio.calls] == [
            False, False, True, False, True,
        ]
        for call in studio.calls:
            assert call["seed"] == 42 and call["candidates"] == 1
            assert call["reference_text"] == "متن مرجع"
            assert call["reference"] == str(reference)

        rendered = [j for j in jobs if not j.skipped]
        for job, call in zip(rendered, studio.calls, strict=True):
            assert Path(call["output_path"]) == tmp_path / "wavs" / job.wav_name
            assert Path(call["output_path"]).is_file()

        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert len(on_disk["renders"]) == len(jobs)  # skip persisted too
        skipped = [e for e in on_disk["renders"] if e.get("skipped")]
        assert len(skipped) == 1 and skipped[0]["wav"] is None
        entry = next(e for e in on_disk["renders"] if e["condition"] == "C")
        assert entry["used_phonemes"] is True and entry["duration_s"] == 1.5
        assert entry["fed_texts"] and entry["warnings"] == ["a warning"]

    def test_score_cer_runs_after_unload_and_ignores_skips(
        self, ab_script, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import voiceclone.audio.io as audio_io

        monkeypatch.setattr(audio_io, "load_audio", lambda path: ("loaded", path))
        sentence = ab_script.Sentence("s01", "probe", "متن آزمایشی", None, "probe")
        studio = SimpleNamespace(
            asr=SimpleNamespace(
                transcribe=lambda audio: SimpleNamespace(text=sentence.text)
            )
        )
        wav = tmp_path / "s01_A.wav"
        wav.write_bytes(b"RIFF")
        manifest = {
            "events": [],
            "renders": [
                {"id": "s01", "condition": "A", "wav": str(wav)},
                {"id": "s01", "condition": "B", "wav": None, "skipped": "no variant"},
            ],
        }
        path = tmp_path / "manifest.json"
        ab_script.score_cer(studio, manifest, [sentence], manifest_path=path)

        assert manifest["events"][0] == {"event": "cer_started", "engines_unloaded": True}
        assert manifest["renders"][0]["cer"] == 0.0  # identical words
        assert manifest["renders"][0]["asr_text"] == sentence.text
        assert "cer" not in manifest["renders"][1]  # skipped: no wav, untouched
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk["renders"][0]["cer"] == 0.0
