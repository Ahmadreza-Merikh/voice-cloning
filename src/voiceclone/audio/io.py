"""Audio loading, resampling and writing.

Everything downstream works on a single representation: ``Audio`` - float32
samples in ``[-1, 1]``, mono unless explicitly stated, with a known sample
rate. Container/codec quirks are contained in this module.
"""

from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import AudioLoadError
from ..logging_utils import get_logger

log = get_logger("audio.io")

#: Extensions we attempt to decode. Anything else is handed to the fallbacks.
AUDIO_EXTENSIONS = frozenset(
    {
        ".wav", ".wave", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".m4a",
        ".aac", ".wma", ".aiff", ".aif", ".aifc", ".caf", ".webm", ".mp4",
        ".mkv", ".mov", ".avi", ".au", ".snd",
    }
)

_SUBTYPE_FOR_DEPTH = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}


@dataclass
class Audio:
    """A block of PCM audio.

    Attributes:
        samples: float32 array, shape ``(n,)`` for mono or ``(channels, n)``.
        sample_rate: samples per second.
        source: where it came from, for logging and provenance.
    """

    samples: np.ndarray
    sample_rate: int
    source: str | None = None

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)
        # Normalise (n, channels) -> (channels, n) for multichannel input.
        if samples.ndim == 2 and samples.shape[0] > samples.shape[1]:
            samples = samples.T
        self.samples = np.ascontiguousarray(samples)

    # -- properties --------------------------------------------------------
    @property
    def is_mono(self) -> bool:
        return self.samples.ndim == 1

    @property
    def channels(self) -> int:
        return 1 if self.is_mono else int(self.samples.shape[0])

    @property
    def num_frames(self) -> int:
        return int(self.samples.shape[-1])

    @property
    def duration(self) -> float:
        return self.num_frames / float(self.sample_rate) if self.sample_rate else 0.0

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.samples))) if self.num_frames else 0.0

    @property
    def rms(self) -> float:
        if not self.num_frames:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples), dtype=np.float64)))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.num_frames

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Audio({self.duration:.2f}s @ {self.sample_rate}Hz, "
            f"{self.channels}ch, peak={self.peak:.3f})"
        )

    # -- transforms (all return new objects) -------------------------------
    def copy(self) -> Audio:
        return replace(self, samples=self.samples.copy())

    def with_samples(self, samples: np.ndarray) -> Audio:
        return replace(self, samples=np.asarray(samples, dtype=np.float32))

    def to_mono(self) -> Audio:
        if self.is_mono:
            return self
        return self.with_samples(np.mean(self.samples, axis=0))

    def resample(self, target_rate: int) -> Audio:
        if target_rate == self.sample_rate or self.num_frames == 0:
            return self
        resampled = resample_array(self.samples, self.sample_rate, target_rate)
        return replace(self, samples=resampled, sample_rate=int(target_rate))

    def slice_seconds(self, start: float, end: float | None = None) -> Audio:
        a = max(0, int(round(start * self.sample_rate)))
        b = self.num_frames if end is None else min(self.num_frames, int(round(end * self.sample_rate)))
        if b <= a:
            return self.with_samples(np.zeros(0, dtype=np.float32))
        return self.with_samples(self.samples[..., a:b])

    def pad(self, before_s: float = 0.0, after_s: float = 0.0) -> Audio:
        pre = int(round(max(0.0, before_s) * self.sample_rate))
        post = int(round(max(0.0, after_s) * self.sample_rate))
        if not pre and not post:
            return self
        pad_width = [(0, 0)] * (self.samples.ndim - 1) + [(pre, post)]
        return self.with_samples(np.pad(self.samples, pad_width, mode="constant"))


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------
def resample_array(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Band-limited resampling, preferring the highest-quality backend present.

    Order of preference: ``soxr`` (VHQ) > ``librosa`` (soxr/kaiser) >
    ``scipy.signal.resample_poly`` > naive linear interpolation.
    """
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)

    samples = np.asarray(samples, dtype=np.float32)

    try:
        import soxr  # type: ignore

        # soxr expects (n,) or (n, channels).
        if samples.ndim == 1:
            out = soxr.resample(samples, source_rate, target_rate, quality="VHQ")
        else:
            out = soxr.resample(samples.T, source_rate, target_rate, quality="VHQ").T
        return np.ascontiguousarray(out, dtype=np.float32)
    except Exception:
        log.debug("soxr unavailable/failed; trying librosa", exc_info=True)

    try:
        import librosa  # type: ignore

        out = librosa.resample(samples, orig_sr=source_rate, target_sr=target_rate, res_type="soxr_hq")
        return np.ascontiguousarray(out, dtype=np.float32)
    except Exception:
        log.debug("librosa resample failed; trying scipy", exc_info=True)

    try:
        from math import gcd

        from scipy.signal import resample_poly  # type: ignore

        divisor = gcd(int(source_rate), int(target_rate))
        up, down = int(target_rate // divisor), int(source_rate // divisor)
        axis = samples.ndim - 1
        out = resample_poly(samples, up, down, axis=axis)
        return np.ascontiguousarray(out, dtype=np.float32)
    except Exception:
        log.warning("Falling back to linear resampling; install `soxr` for better quality.")

    ratio = target_rate / float(source_rate)
    n_out = int(math.floor(samples.shape[-1] * ratio))
    positions = np.linspace(0, samples.shape[-1] - 1, n_out, dtype=np.float64)
    if samples.ndim == 1:
        return np.interp(positions, np.arange(samples.shape[-1]), samples).astype(np.float32)
    stacked = [np.interp(positions, np.arange(samples.shape[-1]), ch) for ch in samples]
    return np.asarray(stacked, dtype=np.float32)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _load_soundfile(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return data, int(sample_rate)


def _load_torchaudio(path: Path) -> tuple[np.ndarray, int]:
    import torchaudio  # type: ignore

    waveform, sample_rate = torchaudio.load(str(path))
    return waveform.numpy(), int(sample_rate)


def _load_av(path: Path) -> tuple[np.ndarray, int]:
    """Decode anything ffmpeg can read, via PyAV. Handles video containers."""
    import av  # type: ignore

    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise AudioLoadError(f"No audio stream found in {path.name}")
        sample_rate = int(stream.codec_context.sample_rate or 0)
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=sample_rate or 48000)
        blocks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for converted in resampler.resample(frame) or []:
                blocks.append(converted.to_ndarray().reshape(-1))
        if not blocks:
            raise AudioLoadError(f"Decoded zero audio frames from {path.name}")
        return np.concatenate(blocks), sample_rate or 48000


def _load_pydub(path: Path) -> tuple[np.ndarray, int]:
    from pydub import AudioSegment  # type: ignore

    segment = AudioSegment.from_file(str(path))
    raw = np.array(segment.get_array_of_samples())
    if segment.channels > 1:
        raw = raw.reshape((-1, segment.channels)).T
    max_value = float(1 << (8 * segment.sample_width - 1))
    return raw.astype(np.float32) / max_value, int(segment.frame_rate)


def load_audio(
    path: str | Path | tuple[int, np.ndarray] | Audio,
    *,
    target_rate: int | None = None,
    mono: bool = True,
) -> Audio:
    """Load audio from a path, an ``(sr, samples)`` tuple, or pass through.

    Gradio hands back ``(sample_rate, numpy_array)`` tuples, so accepting them
    here keeps the UI layer free of conversion code.
    """
    if isinstance(path, Audio):
        audio = path
    elif isinstance(path, tuple) and len(path) == 2:
        sample_rate, samples = path
        samples = np.asarray(samples)
        # Gradio delivers int16/int32 for uploaded files.
        if np.issubdtype(samples.dtype, np.integer):
            max_value = float(np.iinfo(samples.dtype).max)
            samples = samples.astype(np.float32) / max_value
        audio = Audio(samples.astype(np.float32), int(sample_rate), source="<memory>")
    else:
        file_path = Path(str(path)).expanduser()
        if not file_path.is_file():
            raise AudioLoadError(f"Audio file not found: {file_path}")
        if file_path.stat().st_size == 0:
            raise AudioLoadError(f"Audio file is empty: {file_path.name}")

        errors: list[str] = []
        loaders = (
            ("soundfile", _load_soundfile),
            ("torchaudio", _load_torchaudio),
            ("av", _load_av),
            ("pydub", _load_pydub),
        )
        data = None
        for name, loader in loaders:
            try:
                data = loader(file_path)
                log.debug("Decoded %s with %s", file_path.name, name)
                break
            except Exception as exc:  # noqa: PERF203 - each backend may fail
                errors.append(f"{name}: {exc}")

        if data is None:
            raise AudioLoadError(
                f"Could not decode '{file_path.name}'.",
                hint=(
                    "Install ffmpeg (brew install ffmpeg / apt install ffmpeg) for broad "
                    "format support. Backend errors:\n    " + "\n    ".join(errors)
                ),
            )
        audio = Audio(data[0], data[1], source=str(file_path))

    if audio.num_frames == 0:
        raise AudioLoadError(f"Audio contains no samples: {audio.source or 'input'}")

    if mono:
        audio = audio.to_mono()
    if target_rate:
        audio = audio.resample(target_rate)

    # Guard against NaN/Inf from broken files - they poison every later stage.
    if not np.all(np.isfinite(audio.samples)):
        log.warning("Non-finite samples in %s; replacing with silence.", audio.source)
        audio = audio.with_samples(np.nan_to_num(audio.samples, nan=0.0, posinf=0.0, neginf=0.0))

    return audio


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------
def save_audio(
    audio: Audio,
    path: str | Path,
    *,
    audio_format: str | None = None,
    bit_depth: int = 24,
) -> Path:
    """Write audio to disk, creating parent directories as needed."""
    out_path = Path(str(path)).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = (audio_format or out_path.suffix.lstrip(".") or "wav").lower()
    if out_path.suffix.lstrip(".").lower() != fmt:
        out_path = out_path.with_suffix(f".{fmt}")

    samples = audio.samples
    # soundfile wants (frames, channels).
    data = samples if samples.ndim == 1 else samples.T

    if fmt == "mp3":
        return _save_mp3(audio, out_path)

    import soundfile as sf

    subtype = _SUBTYPE_FOR_DEPTH.get(int(bit_depth), "PCM_24")
    if fmt == "flac" and subtype == "FLOAT":
        subtype = "PCM_24"  # FLAC has no float subtype
    try:
        sf.write(str(out_path), data, audio.sample_rate, subtype=subtype)
    except Exception:
        log.debug("soundfile write failed with subtype=%s; retrying default", subtype, exc_info=True)
        sf.write(str(out_path), data, audio.sample_rate)

    return out_path


def _save_mp3(audio: Audio, out_path: Path) -> Path:
    """MP3 export via pydub/ffmpeg, falling back to WAV when unavailable."""
    try:
        from pydub import AudioSegment  # type: ignore

        pcm = np.clip(audio.to_mono().samples, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(audio.sample_rate)
            handle.writeframes(pcm16.tobytes())
        buffer.seek(0)
        AudioSegment.from_wav(buffer).export(str(out_path), format="mp3", bitrate="256k")
        return out_path
    except Exception:
        log.warning("MP3 export unavailable (needs ffmpeg); writing WAV instead.")
        return save_audio(audio, out_path.with_suffix(".wav"), audio_format="wav")


def is_audio_file(path: str | Path) -> bool:
    return Path(str(path)).suffix.lower() in AUDIO_EXTENSIONS


def to_gradio(audio: Audio) -> tuple[int, np.ndarray]:
    """Convert to the ``(sample_rate, int16)`` tuple Gradio renders best."""
    mono = np.clip(audio.to_mono().samples, -1.0, 1.0)
    return audio.sample_rate, (mono * 32767.0).astype(np.int16)


def probe(path: str | Path) -> dict[str, Any]:
    """Cheap metadata read without decoding the whole file."""
    file_path = Path(str(path)).expanduser()
    try:
        import soundfile as sf

        info = sf.info(str(file_path))
        return {
            "duration": float(info.duration),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "format": f"{info.format}/{info.subtype}",
            "size_mb": file_path.stat().st_size / 1e6,
        }
    except Exception:
        return {
            "size_mb": file_path.stat().st_size / 1e6 if file_path.is_file() else 0.0,
            "format": file_path.suffix.lstrip(".").upper() or "unknown",
        }


__all__ = [
    "AUDIO_EXTENSIONS",
    "Audio",
    "is_audio_file",
    "load_audio",
    "probe",
    "resample_array",
    "save_audio",
    "to_gradio",
]
