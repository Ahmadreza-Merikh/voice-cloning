"""Audio I/O, analysis, enhancement and assembly."""

from .analysis import QualityReport, analyze
from .denoise import butter_filter, denoise, separate_vocals
from .io import Audio, load_audio, save_audio, to_gradio
from .loudness import fade_edges, measure_lufs, normalize_loudness, peak_limit
from .postprocess import join_chunks, master, trim_silence_edges
from .preprocess import PreprocessResult, preprocess_reference, preprocess_source_audio
from .segment import WindowChoice, select_reference_window
from .vad import detect_speech, trim_and_compact

__all__ = [
    "Audio",
    "PreprocessResult",
    "QualityReport",
    "WindowChoice",
    "analyze",
    "butter_filter",
    "denoise",
    "detect_speech",
    "fade_edges",
    "join_chunks",
    "load_audio",
    "master",
    "measure_lufs",
    "normalize_loudness",
    "peak_limit",
    "preprocess_reference",
    "preprocess_source_audio",
    "save_audio",
    "select_reference_window",
    "separate_vocals",
    "to_gradio",
    "trim_and_compact",
    "trim_silence_edges",
]
