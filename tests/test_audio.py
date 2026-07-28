"""Audio I/O, analysis, loudness, VAD and assembly."""

from __future__ import annotations

import numpy as np
import pytest

from voiceclone.audio.analysis import analyze, effective_bandwidth, estimate_snr
from voiceclone.audio.io import Audio, load_audio, resample_array, save_audio, to_gradio
from voiceclone.audio.loudness import (
    fade_edges,
    normalize_loudness,
    peak_limit,
    remove_dc_offset,
)
from voiceclone.audio.postprocess import equal_power_crossfade, join_chunks, trim_silence_edges
from voiceclone.audio.preprocess import preprocess_reference
from voiceclone.audio.segment import select_reference_window
from voiceclone.audio.vad import detect_speech, trim_and_compact
from voiceclone.errors import AudioLoadError, ReferenceQualityError


class TestAudio:
    def test_basic_properties(self, speech_audio: Audio) -> None:
        assert speech_audio.is_mono
        assert speech_audio.channels == 1
        assert speech_audio.duration == pytest.approx(6.0, abs=0.05)
        assert 0.0 < speech_audio.peak <= 1.5

    def test_dtype_is_coerced(self) -> None:
        audio = Audio(np.zeros(100, dtype=np.float64), 16000)
        assert audio.samples.dtype == np.float32

    def test_stereo_is_transposed_to_channels_first(self) -> None:
        audio = Audio(np.zeros((1000, 2), dtype=np.float32), 16000)
        assert audio.channels == 2
        assert audio.num_frames == 1000

    def test_to_mono(self) -> None:
        stereo = Audio(np.ones((2, 500), dtype=np.float32), 16000)
        assert stereo.to_mono().is_mono

    def test_slice_seconds(self, speech_audio: Audio) -> None:
        piece = speech_audio.slice_seconds(1.0, 2.0)
        assert piece.duration == pytest.approx(1.0, abs=0.02)

    def test_slice_past_the_end_is_empty(self, speech_audio: Audio) -> None:
        assert speech_audio.slice_seconds(100.0, 101.0).num_frames == 0

    def test_pad(self, speech_audio: Audio) -> None:
        padded = speech_audio.pad(0.5, 0.5)
        assert padded.duration == pytest.approx(speech_audio.duration + 1.0, abs=0.02)


class TestResampling:
    @pytest.mark.parametrize(("source", "target"), [(48000, 24000), (16000, 24000), (44100, 24000)])
    def test_length_scales_with_rate(self, source: int, target: int) -> None:
        samples = np.random.default_rng(0).normal(0, 0.1, source).astype(np.float32)
        out = resample_array(samples, source, target)
        assert abs(len(out) - target) < target * 0.02

    def test_identity(self) -> None:
        samples = np.zeros(100, dtype=np.float32)
        assert resample_array(samples, 24000, 24000) is samples

    def test_audio_resample(self, speech_audio: Audio) -> None:
        out = speech_audio.resample(16000)
        assert out.sample_rate == 16000
        assert out.duration == pytest.approx(speech_audio.duration, abs=0.02)


class TestIO:
    def test_roundtrip(self, speech_audio: Audio, tmp_path) -> None:
        path = save_audio(speech_audio, tmp_path / "test.wav")
        assert path.is_file()
        loaded = load_audio(path)
        assert loaded.sample_rate == speech_audio.sample_rate
        assert loaded.num_frames == speech_audio.num_frames

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(AudioLoadError):
            load_audio(tmp_path / "nope.wav")

    def test_gradio_tuple_input(self) -> None:
        samples = (np.random.default_rng(0).normal(0, 0.1, 2400) * 32767).astype(np.int16)
        audio = load_audio((24000, samples))
        assert audio.sample_rate == 24000
        assert audio.samples.dtype == np.float32
        assert audio.peak <= 1.0

    def test_to_gradio(self, speech_audio: Audio) -> None:
        rate, samples = to_gradio(speech_audio)
        assert rate == speech_audio.sample_rate
        assert samples.dtype == np.int16

    def test_non_finite_samples_are_scrubbed(self, tmp_path) -> None:
        broken = np.array([0.1, np.nan, np.inf, -np.inf, 0.2], dtype=np.float32)
        audio = load_audio(Audio(broken, 24000))
        assert np.all(np.isfinite(audio.samples))


class TestLoudness:
    def test_dc_offset_removal(self) -> None:
        audio = Audio(np.full(24000, 0.3, dtype=np.float32), 24000)
        assert abs(float(np.mean(remove_dc_offset(audio).samples))) < 1e-6

    def test_peak_limit(self) -> None:
        audio = Audio(np.full(2400, 0.99, dtype=np.float32), 24000)
        assert peak_limit(audio, -6.0).peak <= 0.51

    def test_peak_limit_leaves_quiet_audio_alone(self) -> None:
        audio = Audio(np.full(2400, 0.1, dtype=np.float32), 24000)
        assert peak_limit(audio, -1.0).peak == pytest.approx(0.1)

    def test_normalisation_moves_toward_the_target(self, speech_audio: Audio) -> None:
        quiet = speech_audio.with_samples(speech_audio.samples * 0.02)
        normalised = normalize_loudness(quiet, -20.0)
        assert normalised.rms > quiet.rms

    def test_silence_survives_normalisation(self) -> None:
        silence = Audio(np.zeros(24000, dtype=np.float32), 24000)
        assert normalize_loudness(silence, -20.0).num_frames == 24000

    def test_fades_zero_the_edges(self, speech_audio: Audio) -> None:
        faded = fade_edges(speech_audio, 20)
        assert abs(float(faded.samples[0])) < 1e-4
        assert abs(float(faded.samples[-1])) < 1e-4


class TestAnalysis:
    def test_report_fields(self, speech_audio: Audio) -> None:
        report = analyze(speech_audio)
        assert report.duration > 0
        assert report.sample_rate == 24000
        assert 0.0 <= report.score <= 100.0
        assert report.verdict in {"excellent", "good", "usable", "poor", "unusable"}

    def test_noise_lowers_the_score(self, speech_audio: Audio, noisy_audio: Audio) -> None:
        assert analyze(noisy_audio).score < analyze(speech_audio).score

    def test_snr_ordering(self, speech_audio: Audio, noisy_audio: Audio) -> None:
        clean_snr, _, _ = estimate_snr(speech_audio.samples, 24000)
        noisy_snr, _, _ = estimate_snr(noisy_audio.samples, 24000)
        assert clean_snr > noisy_snr

    def test_short_clip_is_flagged(self, short_audio: Audio) -> None:
        report = analyze(short_audio, min_seconds=3.0)
        assert report.issues
        assert report.score < 80

    def test_clipping_is_detected(self, speech_audio: Audio) -> None:
        clipped = speech_audio.with_samples(np.clip(speech_audio.samples * 12.0, -1.0, 1.0))
        assert analyze(clipped).clipping_ratio > 0.001

    def test_bandwidth_detects_lowpass(self, speech_audio: Audio) -> None:
        from voiceclone.audio.denoise import butter_filter

        narrow = butter_filter(speech_audio, lowpass_hz=1500.0)
        assert effective_bandwidth(narrow.samples, 24000) < effective_bandwidth(
            speech_audio.samples, 24000
        )

    def test_report_serialises(self, speech_audio: Audio) -> None:
        data = analyze(speech_audio).to_dict()
        assert "score" in data and "verdict" in data


class TestVAD:
    def test_speech_is_found(self, speech_audio: Audio) -> None:
        segments = detect_speech(speech_audio, backend="energy")
        assert segments
        assert all(end > start for start, end in segments)

    def test_trimming_shortens_padded_audio(self, speech_audio: Audio) -> None:
        padded = speech_audio.pad(2.0, 2.0)
        result = trim_and_compact(padded, backend="energy")
        assert result.audio.duration < padded.duration

    def test_empty_audio(self) -> None:
        assert detect_speech(Audio(np.zeros(0, dtype=np.float32), 24000)) == []


class TestWindowSelection:
    def test_short_clip_is_returned_whole(self, speech_audio: Audio) -> None:
        choice = select_reference_window(speech_audio, target_seconds=10.0)
        assert choice.audio.duration == pytest.approx(speech_audio.duration, abs=0.1)

    def test_long_clip_is_trimmed_to_target(self, long_audio: Audio) -> None:
        choice = select_reference_window(long_audio, target_seconds=8.0, max_seconds=12.0)
        assert choice.audio.duration <= 12.5
        assert choice.reason


class TestPostprocess:
    def test_crossfade_length(self) -> None:
        left = np.ones(1000, dtype=np.float32)
        right = np.ones(1000, dtype=np.float32)
        joined = equal_power_crossfade(left, right, 100)
        assert len(joined) == 1900

    def test_crossfade_has_no_discontinuity(self) -> None:
        left = np.ones(500, dtype=np.float32)
        right = -np.ones(500, dtype=np.float32)
        joined = equal_power_crossfade(left, right, 200)
        assert float(np.max(np.abs(np.diff(joined)))) < 0.05

    def test_join_with_gaps(self, speech_audio: Audio) -> None:
        pieces = [speech_audio.slice_seconds(0, 1), speech_audio.slice_seconds(1, 2)]
        joined = join_chunks(pieces, sample_rate=24000, gaps_ms=[0, 500])
        assert joined.duration > 1.4

    def test_join_single_chunk(self, speech_audio: Audio) -> None:
        assert join_chunks([speech_audio], sample_rate=24000).num_frames > 0

    def test_join_empty_list(self) -> None:
        assert join_chunks([], sample_rate=24000).num_frames == 0

    def test_trim_edges(self, speech_audio: Audio) -> None:
        padded = speech_audio.pad(1.0, 1.0)
        assert trim_silence_edges(padded).duration < padded.duration


class TestPreprocessPipeline:
    def test_pipeline_runs(self, speech_audio: Audio, config) -> None:
        result = preprocess_reference(speech_audio, config.audio)
        assert result.audio.num_frames > 0
        assert result.audio.sample_rate == config.audio.sample_rate
        assert result.steps
        assert result.summary()

    def test_denoise_can_be_disabled(self, noisy_audio: Audio, config) -> None:
        result = preprocess_reference(noisy_audio, config.audio, override={"denoise": "none"})
        assert result.denoiser == "none"

    def test_too_short_is_rejected(self, config) -> None:
        tiny = Audio(np.zeros(2400, dtype=np.float32), 24000)
        with pytest.raises(ReferenceQualityError):
            preprocess_reference(tiny, config.audio)

    def test_long_input_is_windowed(self, long_audio: Audio, config) -> None:
        result = preprocess_reference(long_audio, config.audio, override={"denoise": "none"})
        assert result.audio.duration <= config.audio.reference.max_seconds + 1.0

    def test_report_serialises(self, speech_audio: Audio, config) -> None:
        data = preprocess_reference(speech_audio, config.audio).to_dict()
        assert "before" in data and "after" in data and "steps" in data
