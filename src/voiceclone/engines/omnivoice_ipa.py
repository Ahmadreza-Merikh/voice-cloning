"""OmniVoice IPA - the fine-tune whose native input is Persian phonemes.

What it is
----------
``christopherthompson81/omnivoice-ipa-onnx`` is ``k2-fsa/OmniVoice`` continued
on IPA produced by ``vernacula-phonemizer`` (Persian goldens:
``csharp/goldens/fa.tsv``). It ships as a *diff* against the base - LoRA
factors plus an embedding delta - which ``scripts/build_omnivoice_ipa.py``
folds into a local copy of the base weights once. The merged directory is the
model this engine loads; ``IPA_ADAPTATION.json`` beside it records exactly
which diff went in (sha256, lora scale, row counts).

Licences differ: the base is Apache-2.0, the fine-tune CC-BY-NC-4.0
(non-commercial). The merged weights carry the fine-tune's licence.

Why a subclass and not a flag
-----------------------------
Everything mechanical - worker, venv, generation defaults, memory guard - is
inherited unchanged from :class:`OmniVoiceEngine` via two hooks
(``_settings``/``_model_source``), so a render under this engine differs from
one under ``omnivoice`` only in weights and in the phonemes the text pipeline
hands over. What differs semantically is one capability: this engine wants
canonical IPA (:mod:`voiceclone.text.ipa_normalize`), and says so.

Availability
------------
The weights are *not* downloaded with the rest: they are built. Until
``scripts/build_omnivoice_ipa.py`` has run, this engine reports unavailable
with the command to run - and nothing crashes.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import AppConfig, OmniVoiceIPAConfig
from ..errors import EngineNotAvailableError
from ..logging_utils import get_logger
from .base import EngineCapabilities, SynthesisResult, register_tts
from .omnivoice_tts import OmniVoiceEngine

log = get_logger("engines.omnivoice_ipa")

#: Written by scripts/build_omnivoice_ipa.py next to the merged weights; its
#: presence IS the "built" marker (and it carries the provenance record).
#: Keep the literal in sync with that script - tests pin both.
PROVENANCE_NAME = "IPA_ADAPTATION.json"


def default_model_dir() -> Path:
    """Where the merged weights live with no config in hand (diagnostics)."""
    settings = OmniVoiceIPAConfig()
    path = Path(settings.model_dir).expanduser()
    if path.is_absolute():
        return path
    return Path("~/.cache/persian-voice-cloning").expanduser() / path


def ipa_model_dir(config: AppConfig) -> Path:
    """The configured merged-weights directory (relative -> under cache_dir)."""
    path = Path(config.engines.omnivoice_ipa.model_dir).expanduser()
    if path.is_absolute():
        return path
    return Path(config.paths.cache_dir).expanduser() / path


def _candidate_model_dirs() -> list[Path]:
    """Directories worth checking before declaring the weights missing.

    The classmethod ``is_available`` has no config to look at, so it checks
    the default and - best effort - the directory the YAML config points at.
    The instance path (``_load``) always checks the config it was built with.
    """
    candidates = [default_model_dir()]
    try:
        candidates.append(ipa_model_dir(AppConfig.load()))
    except Exception:  # a broken config is not this engine's problem to report
        pass
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _build_hint(model_dir: Path) -> str:
    return (
        f"Expected {model_dir / PROVENANCE_NAME}. The weights are not downloaded: "
        "the published diff is folded into a local copy of the base once, with\n"
        "  <python> scripts/build_omnivoice_ipa.py"
    )


@register_tts
class OmniVoiceIPAEngine(OmniVoiceEngine):
    """The IPA fine-tune: same worker, canonical phonemes as input."""

    name = "omnivoice_ipa"
    display_name = "OmniVoice IPA (fine-tune, reads Persian IPA)"
    capabilities: EngineCapabilities = replace(
        OmniVoiceEngine.capabilities,
        accepts_phonemes=True,
        # Everything else - reference transcript, speed, seed, size, native
        # Persian - is the parent's: a comparison must not drift on it.
    )

    # -- availability ------------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        available, reason = OmniVoiceEngine.is_available()  # env + worker only
        if not available:
            return available, reason
        for model_dir in _candidate_model_dirs():
            if (model_dir / PROVENANCE_NAME).is_file():
                return True, ""
        searched = ", ".join(str(path) for path in _candidate_model_dirs())
        return False, (
            "The IPA-adapted OmniVoice weights are not built yet "
            f"(searched: {searched}).\n" + _build_hint(default_model_dir())
        )

    # -- config hooks ------------------------------------------------------
    def _settings(self) -> OmniVoiceIPAConfig:
        return self.config.engines.omnivoice_ipa

    def _model_source(self, settings: OmniVoiceIPAConfig) -> str:
        # Always the merged local directory - never the diff repo id; the
        # worker would find no model files there.
        return str(ipa_model_dir(self.config))

    def _require_weights(self) -> Path:
        """Instance-level check: the *configured* directory must be built.

        ``is_available`` can only see default/YAML locations; this runs before
        any worker process starts, so a custom ``model_dir`` still fails with
        the same clear error instead of a cryptic load failure.
        """
        model_dir = ipa_model_dir(self.config)
        if not (model_dir / PROVENANCE_NAME).is_file():
            raise EngineNotAvailableError(
                "The IPA fine-tune weights are missing.", hint=_build_hint(model_dir)
            )
        return model_dir

    def _load(self) -> None:
        model_dir = self._require_weights()
        super()._load()
        # Record which diff went into the weights, for the render's provenance.
        try:
            adaptation = json.loads((model_dir / PROVENANCE_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            adaptation = {}
        if adaptation:
            self._loaded_model = {
                **getattr(self, "_loaded_model", {}),
                "fine_tune": {
                    "repo": adaptation.get("diff_repo", ""),
                    "diff_sha256": str(adaptation.get("diff_sha256", ""))[:12],
                    "lora_scale": adaptation.get("lora_scale"),
                },
            }

    # -- inference ---------------------------------------------------------
    def synthesize(
        self,
        text: str,
        reference: Any,
        *,
        reference_text: str | None = None,
        speed: float = 1.0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SynthesisResult:
        phonemes = bool(kwargs.get("is_phonemes"))
        if not phonemes:
            log.warning(
                "%s received raw Persian text rather than canonical IPA; its "
                "phoneme input path is not engaged (text.g2p produced no "
                "phonemes for this render).",
                self.display_name,
            )
        result = super().synthesize(
            text,
            reference,
            reference_text=reference_text,
            speed=speed,
            seed=seed,
            **kwargs,
        )
        # Says which channel the model actually saw; the A/B manifest stores it.
        result.metadata["is_phonemes"] = phonemes
        return result


__all__ = ["PROVENANCE_NAME", "OmniVoiceIPAEngine", "default_model_dir", "ipa_model_dir"]
