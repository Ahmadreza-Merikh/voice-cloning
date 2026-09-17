"""Typed configuration objects backed by ``configs/default.yaml``.

Layering, lowest precedence first:

1. the dataclass defaults in this module
2. ``configs/default.yaml``
3. a user YAML (``--config`` / ``$VOICECLONE_CONFIG``)
4. environment overrides ``VOICECLONE__SECTION__KEY=value``
5. explicit keyword arguments at call sites

Nothing here imports torch or any engine, so it stays cheap to load.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import sys
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cache
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

import yaml

from .errors import ConfigError
from .logging_utils import get_logger

log = get_logger("config")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
ENV_PREFIX = "VOICECLONE__"

T = TypeVar("T")


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
@dataclass
class PathsConfig:
    cache_dir: str = "~/.cache/persian-voice-cloning"
    output_dir: str = "outputs"
    voices_dir: str = "voices"

    def resolved(self, name: str, *, base: Path | None = None) -> Path:
        """Expand ``~``/env vars and anchor relative paths at the repo root."""
        raw = getattr(self, name)
        path = Path(os.path.expandvars(str(raw))).expanduser()
        if not path.is_absolute():
            path = (base or PROJECT_ROOT) / path
        return path

    def ensure(self, name: str, *, base: Path | None = None) -> Path:
        path = self.resolved(name, base=base)
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class RuntimeConfig:
    device: str = "auto"
    dtype: str = "auto"
    seed: int = 0
    num_threads: int = 0
    mps_fallback: bool = True


@dataclass
class ReferenceAudioConfig:
    target_seconds: float = 10.0
    min_seconds: float = 3.0
    max_seconds: float = 20.0
    auto_select_window: bool = True
    min_quality_score: float = 25.0


@dataclass
class PreprocessConfig:
    to_mono: bool = True
    remove_dc: bool = True
    highpass_hz: float | None = 55.0
    lowpass_hz: float | None = None
    separate_vocals: bool = False
    denoise: str = "auto"
    denoise_strength: float = 0.8
    dereverb: bool = False
    trim_silence: bool = True
    vad_backend: str = "auto"
    vad_threshold: float = 0.5
    max_internal_silence: float = 0.35
    vad_pad_ms: int = 120
    loudness_lufs: float | None = -23.0
    peak_dbfs: float = -1.5


@dataclass
class OutputAudioConfig:
    loudness_lufs: float | None = -18.0
    peak_dbfs: float = -1.0
    fade_ms: int = 12
    join_crossfade_ms: int = 40
    sentence_gap_ms: int = 180
    paragraph_gap_ms: int = 380
    format: str = "wav"
    bit_depth: int = 24


@dataclass
class AudioConfig:
    sample_rate: int = 24000
    reference: ReferenceAudioConfig = field(default_factory=ReferenceAudioConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    output: OutputAudioConfig = field(default_factory=OutputAudioConfig)


@dataclass
class G2PConfig:
    enabled: str | bool = "auto"
    repo_path: str | None = "auto"
    inject_ezafe: bool = True
    output: str = "ipa"
    #: Empty: no engine here reads Persian IPA better than Persian script.
    #: See configs/default.yaml for the measurement.
    apply_to_engines: list[str] = field(default_factory=list)


@dataclass
class ChunkConfig:
    max_chars: int = 220
    max_seconds: float = 22.0
    min_chars: int = 24


@dataclass
class TextConfig:
    language: str = "auto"
    normalize: bool = True
    expand_numbers: bool = True
    strip_bracketed_tags: bool = True
    g2p: G2PConfig = field(default_factory=G2PConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)


@dataclass
class F5Config:
    model_arch: str = "F5TTS_v1_Base"
    checkpoint: str = "hf://Lumos675/F5_TTS_Persian/model_last.pt"
    vocab: str = "hf://Lumos675/F5_TTS_Persian/vocab.txt"
    fallback_checkpoint: str = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    vocoder: str = "vocos"
    nfe_step: int = 48
    cfg_strength: float = 2.0
    sway_sampling_coef: float = -1.0
    speed: float = 1.0
    cross_fade_duration: float = 0.15


@dataclass
class MossConfig:
    model_id: str = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
    #: Below upstream's 1.0 - see configs/default.yaml for the measurements.
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 4096


@dataclass
class ASRConfig:
    backend: str = "faster-whisper"
    model: str = "large-v3-turbo"
    compute_type: str = "auto"
    language: str = "fa"
    beam_size: int = 5
    vad_filter: bool = True


@dataclass
class ChatterboxConfig:
    """Chatterbox, driven out-of-process in its own virtualenv.

    ``t3_checkpoint`` swaps the text-to-speech-token transformer for a Persian
    finetune. The speaker path - voice encoder and S3Gen - stays as trained and
    never sees Persian text, which is why this engine's identity transfer does
    not depend on how much Persian data anyone had.
    """

    python: str | None = "auto"
    t3_checkpoint: str | None = "hf://Thomcles/Chatterbox-TTS-Persian-Farsi/t3_fa.safetensors"
    #: The multilingual stack requires a language tag and Persian is not one
    #: of its 23. Arabic shares the script and is the default until a
    #: measurement says another tag reads Persian better.
    language_id: str = "ar"
    #: Emotion intensity. 0.5 is upstream's neutral; higher is more theatrical
    #: and, in upstream's own guidance, less stable.
    exaggeration: float = 0.5
    #: Classifier-free guidance on the token model. Upstream notes lower values
    #: slow the delivery, which matters for a fidelity task more than a demo.
    cfg_weight: float = 0.5
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    min_p: float = 0.05
    top_p: float = 1.0
    timeout_seconds: int = 900


@dataclass
class OmniVoiceConfig:
    """OmniVoice, driven out-of-process in its own virtualenv.

    Generation defaults are upstream's ``omnivoice-infer`` values, which are
    what arXiv 2604.00688 measured with. Changing any of them is a departure
    from the published configuration and belongs in a benchmark label.
    """

    python: str | None = "auto"
    model_id: str = "k2-fsa/OmniVoice"
    #: Upstream: performance is slightly better when the language is named.
    language: str = "fa"
    #: Upstream's own inference dtype, which the paper measured with. It loads
    #: straight onto Metal: what crashed was transformers' threaded weight
    #: loading, and the worker switches that thread pool off before importing
    #: it. Loading directly also halves the peak - 1.97 GB against 4.39 GB.
    dtype: str = "float16"
    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    denoise: bool = True
    postprocess_output: bool = True
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    timeout_seconds: int = 900


@dataclass
class SeedVCConfig:
    repo_path: str | None = "auto"
    python: str | None = "auto"
    diffusion_steps: int = 50
    length_adjust: float = 1.0
    inference_cfg_rate: float = 0.7
    f0_condition: bool = False
    auto_f0_adjust: bool = False
    semi_tone_shift: int = 0
    timeout_seconds: int = 900


@dataclass
class EnginesConfig:
    tts: str = "f5"
    vc: str = "cascade"
    f5: F5Config = field(default_factory=F5Config)
    moss: MossConfig = field(default_factory=MossConfig)
    chatterbox: ChatterboxConfig = field(default_factory=ChatterboxConfig)
    omnivoice: OmniVoiceConfig = field(default_factory=OmniVoiceConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    seedvc: SeedVCConfig = field(default_factory=SeedVCConfig)


@dataclass
class EvaluationConfig:
    enabled: bool = True
    speaker_similarity: bool = True
    transcribe_check: bool = False
    #: Render this many takes and keep the best. 1 disables the machinery.
    candidates: int = 1
    #: Round-trip CER is clamped here before candidates are compared, so takes
    #: inside the recogniser's noise are ranked on similarity instead. Measured:
    #: Whisper scores 7.8% CER transcribing this project's *real human* clips
    #: against their own transcripts - see `_candidate_rank`.
    candidate_cer_floor: float = 0.08
    #: Similarity differences smaller than this are treated as a tie.
    #: Identity-preserving perturbations of one render (85-95% crops, +/-1 dB,
    #: -60 dBFS dither) move the ECAPA score by up to 0.0205 - see
    #: `_candidate_rank`.
    candidate_similarity_step: float = 0.02


@dataclass
class UIConfig:
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False
    language: str = "fa"
    theme: str = "soft"
    max_upload_mb: int = 200
    show_diagnostics: bool = True


@dataclass
class AppConfig:
    """Root configuration object passed around the whole application."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    text: TextConfig = field(default_factory=TextConfig)
    engines: EnginesConfig = field(default_factory=EnginesConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # -- construction ------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        overrides: dict[str, Any] | None = None,
        use_env: bool = True,
    ) -> AppConfig:
        """Build a config from the default YAML plus optional layers."""
        data: dict[str, Any] = {}

        if DEFAULT_CONFIG_PATH.is_file():
            data = _read_yaml(DEFAULT_CONFIG_PATH)

        user_path = path or os.environ.get("VOICECLONE_CONFIG")
        if user_path:
            user_file = Path(user_path).expanduser()
            if not user_file.is_file():
                raise ConfigError(f"Config file not found: {user_file}")
            data = deep_merge(data, _read_yaml(user_file))
            log.info("Loaded config overrides from %s", user_file)

        if use_env:
            env_data = _env_overrides()
            if env_data:
                data = deep_merge(data, env_data)
                log.debug("Applied %d environment override(s)", len(env_data))

        if overrides:
            data = deep_merge(data, overrides)

        return from_mapping(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def copy_with(self, **overrides: Any) -> AppConfig:
        """Return a deep copy with ``section.key`` style overrides applied.

        >>> cfg.copy_with(**{"engines.tts": "moss", "audio.sample_rate": 22050})
        """
        data = self.to_dict()
        for dotted, value in overrides.items():
            cursor: dict[str, Any] = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                nxt = cursor.get(part)
                if not isinstance(nxt, dict):
                    raise ConfigError(f"Cannot override {dotted!r}: {part!r} is not a section.")
                cursor = nxt
            cursor[parts[-1]] = value
        return from_mapping(AppConfig, data)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(loaded).__name__}: {path}")
    return loaded


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` without mutating either."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _coerce_scalar(raw: str) -> Any:
    """Interpret an environment string as YAML (so ints/bools/null work)."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _env_overrides() -> dict[str, Any]:
    """Collect ``VOICECLONE__A__B=value`` variables into a nested dict."""
    result: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX) :].split("__") if p]
        if not parts:
            continue
        cursor = result
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):  # pragma: no cover - malformed env
                break
        else:
            cursor[parts[-1]] = _coerce_scalar(raw)
    return result


@cache
def _resolved_hints(cls: type) -> dict[str, Any]:
    """Real type objects for a dataclass.

    ``from __future__ import annotations`` turns every annotation into a
    string, which would make ``field.type`` useless for coercion. Resolving
    them once per class (and caching) keeps the conversion honest.
    """
    try:
        return typing.get_type_hints(cls, include_extras=False, globalns=vars(sys.modules[cls.__module__]))
    except Exception:  # pragma: no cover - exotic forward refs
        log.debug("Could not resolve type hints for %s", cls.__name__, exc_info=True)
        return {f.name: Any for f in fields(cls)}


def _is_optional(annotation: Any) -> tuple[bool, Any]:
    """Unwrap ``X | None`` into ``(True, X)``."""
    origin = get_origin(annotation)
    if origin is None:
        return False, annotation
    args = get_args(annotation)
    if type(None) in args:
        remaining = [a for a in args if a is not type(None)]
        if len(remaining) == 1:
            return True, remaining[0]
        return True, remaining[0] if remaining else Any
    return False, annotation


def _convert(value: Any, annotation: Any, *, context: str) -> Any:
    """Best-effort coercion of a YAML value to the annotated type."""
    optional, inner = _is_optional(annotation)
    if value is None:
        if optional:
            return None
        return None  # keep explicit nulls; dataclass default already applied

    if is_dataclass(inner) and isinstance(value, dict):
        return from_mapping(inner, value)

    origin = get_origin(inner)
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            value = [value]
        return list(value)

    # Unions such as `str | bool` (used by g2p.enabled) pass through as-is.
    if origin is not None:
        return value

    if inner in (int, float, str, bool):
        if inner is bool:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        try:
            return inner(value)
        except (TypeError, ValueError):
            log.warning("Could not coerce %s=%r to %s; keeping raw value.", context, value, inner.__name__)
            return value

    return value


def from_mapping(cls: type[T], data: dict[str, Any] | None) -> T:
    """Instantiate a (possibly nested) dataclass from a plain mapping."""
    if not is_dataclass(cls):  # pragma: no cover - programming error
        raise ConfigError(f"{cls!r} is not a dataclass")
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected mapping for {cls.__name__}, got {type(data).__name__}")

    hints = _resolved_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(cls)}

    for unknown in set(data) - known:
        log.warning("Ignoring unknown config key %s.%s", cls.__name__, unknown)

    for f in fields(cls):
        if f.name not in data:
            continue
        annotation = hints.get(f.name, Any)
        value = _convert(data[f.name], annotation, context=f"{cls.__name__}.{f.name}")
        # An explicit YAML null on a non-optional field means "use the
        # dataclass default" rather than "set it to None".
        if value is None and not _is_optional(annotation)[0]:
            continue
        kwargs[f.name] = value

    return cls(**kwargs)  # type: ignore[return-value]


def load_config(path: str | Path | None = None, **overrides: Any) -> AppConfig:
    """Convenience wrapper: ``load_config(engines__tts="moss")``-ish."""
    dotted = {k.replace("__", "."): v for k, v in overrides.items()}
    cfg = AppConfig.load(path)
    return cfg.copy_with(**dotted) if dotted else cfg
