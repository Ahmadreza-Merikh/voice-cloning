"""The Studio - one object that owns models and runs both pipelines.

Loading a TTS model takes seconds and gigabytes, so nothing is constructed per
request: the Studio caches engines, the ASR model and preprocessed references
for the lifetime of the process. The UI and the CLI both drive this class,
which keeps behaviour identical between them.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.io import Audio, load_audio, save_audio
from ..audio.postprocess import join_chunks, master, strip_reference_leakage
from ..audio.preprocess import PreprocessResult, preprocess_reference, preprocess_source_audio
from ..config import AppConfig
from ..device import DeviceSpec, configure_threads, empty_cache, get_device_spec
from ..engines import ASREngine, TTSEngine, VCEngine, get_tts_engine, get_vc_engine
from ..errors import SynthesisError
from ..logging_utils import ProgressReporter, get_logger
from ..text.pipeline import TextPipeline
from .results import CloneResult, QualityScores

log = get_logger("studio")

ReferenceInput = str | Path | Audio | tuple[int, np.ndarray]


@contextmanager
def _stopwatch(timings: dict[str, float], label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[label] = timings.get(label, 0.0) + (time.perf_counter() - start)


def _candidate_rank(
    wer: float | None,
    similarity: float | None,
    wer_floor: float,
    similarity_step: float,
) -> tuple[float, float]:
    """Sort key for one best-of-N take. Lower is better.

    Both metrics are quantised to their own measurement noise before they are
    compared, because neither resolves differences below it and sorting on
    what a metric cannot resolve is sorting on nothing.

    *WER* is clamped at ``wer_floor``. Transcribing genuine human speech from
    the reference corpus with the same Whisper model scores 4-18% per clip
    (mean 11.3%), so a take at 0% and a take at 7% differ only in the
    recogniser. Above the floor the difference is real, and WER dominates -
    similarity cannot buy its way past a dropped word.

    *Similarity* is bucketed to ``similarity_step``. Perturbing one render in
    ways that cannot change who is speaking - cropping 85-95% of it, +/-1 dB
    of gain, dither at -60 dBFS - moves the ECAPA score by up to 0.0205
    (sd 0.0055), and cropping accounts for nearly all of it. For scale, the
    whole temperature sweep moved similarity by 0.016, so that comparison was
    inside the noise too.

    A hard WER threshold was tried first and is the wrong shape: near the
    floor almost nothing passes and selection silently becomes "lowest WER";
    high enough for ordinary takes to pass and similarity starts outranking
    real mistakes. Quantising degrades smoothly instead of flipping between
    those failures.

    ``wer`` of ``None`` ranks at the floor: a take nobody transcribed is not
    thereby a bad take. ``similarity`` of ``None`` ranks last on that axis
    instead, because the two absences are not symmetric - the WER floor is a
    real "no worse than this" bound, whereas an unscored take simply cannot
    be *shown* to resemble the reference. In practice the speaker encoder is
    all-or-nothing, so every take ties there and the WER clamp decides;
    `_render_best_of` warns when that happens rather than letting it pass
    silently.
    """
    ranked_wer = max(wer, wer_floor) if wer is not None else wer_floor
    if similarity is None or similarity_step <= 0:
        ranked_similarity = 0.0
    else:
        ranked_similarity = -round(similarity / similarity_step)
    return (ranked_wer, ranked_similarity)


class Studio:
    """Model registry, caches and the two synthesis pipelines."""

    def __init__(self, config: AppConfig | None = None, device: DeviceSpec | None = None) -> None:
        self.config = config or AppConfig.load()
        self.device = device or get_device_spec(
            self.config.runtime.device,
            self.config.runtime.dtype,
            mps_fallback=self.config.runtime.mps_fallback,
        )
        configure_threads(self.config.runtime.num_threads)

        self.text_pipeline = TextPipeline(self.config.text)
        self._tts_engines: dict[str, TTSEngine] = {}
        self._vc_engines: dict[str, VCEngine] = {}
        self._asr: ASREngine | None = None
        self._reference_cache: dict[str, PreprocessResult] = {}
        self._transcript_cache: dict[str, str] = {}

        log.info("Studio ready on %s", self.device)

    # ------------------------------------------------------------------
    # Model access
    # ------------------------------------------------------------------
    def tts_engine(self, name: str | None = None) -> TTSEngine:
        key = (name or self.config.engines.tts).lower()
        if key not in self._tts_engines:
            self._tts_engines[key] = get_tts_engine(key, self.config, self.device)
        return self._tts_engines[key]

    def vc_engine(self, name: str | None = None) -> VCEngine:
        key = (name or self.config.engines.vc).lower()
        if key not in self._vc_engines:
            self._vc_engines[key] = get_vc_engine(key, self.config, self.device)
        return self._vc_engines[key]

    @property
    def asr(self) -> ASREngine:
        if self._asr is None:
            self._asr = ASREngine(self.config, self.device)
        return self._asr

    def release(self, keep: str | None = None) -> None:
        """Unload engines to free memory, optionally keeping one loaded."""
        for key, engine in list(self._tts_engines.items()):
            if key != keep:
                engine.unload()
                self._tts_engines.pop(key, None)
        for key, engine in list(self._vc_engines.items()):
            if key != keep:
                engine.unload()
                self._vc_engines.pop(key, None)
        empty_cache(self.device)

    # ------------------------------------------------------------------
    # Reference handling
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_key(source: ReferenceInput, extra: str = "") -> str:
        if isinstance(source, Audio):
            payload = source.samples.tobytes()
        elif isinstance(source, tuple):
            payload = np.asarray(source[1]).tobytes()
        else:
            path = Path(str(source))
            try:
                stat = path.stat()
                payload = f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            except OSError:
                payload = str(path).encode()
        return hashlib.blake2b(payload + extra.encode(), digest_size=16).hexdigest()

    def prepare_reference(
        self,
        source: ReferenceInput,
        *,
        overrides: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> PreprocessResult:
        """Preprocess (and cache) a reference recording."""
        key = self._cache_key(source, repr(sorted((overrides or {}).items())))
        if use_cache and key in self._reference_cache:
            log.debug("Reference cache hit")
            return self._reference_cache[key]

        result = preprocess_reference(source, self.config.audio, override=overrides)
        if use_cache:
            self._reference_cache[key] = result
        return result

    def transcribe_reference(self, reference: Audio, *, language: str | None = None) -> str:
        """Transcribe the reference clip, caching per audio content."""
        key = self._cache_key(reference, f"asr:{language}")
        if key in self._transcript_cache:
            return self._transcript_cache[key]

        transcript = self.asr.transcribe(reference, language=language)
        text = transcript.text.strip()
        if not transcript.is_confident:
            log.warning("Reference transcription confidence is low; check it before synthesising.")
        self._transcript_cache[key] = text
        return text

    # ------------------------------------------------------------------
    # Text to speech
    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        reference: ReferenceInput,
        *,
        engine: str | None = None,
        reference_text: str | None = None,
        speed: float = 1.0,
        seed: int | None = None,
        output_path: str | Path | None = None,
        progress: Callable[[float, str], None] | None = None,
        preprocess_overrides: dict[str, Any] | None = None,
        engine_options: dict[str, Any] | None = None,
        use_phonemes: bool | None = None,
    ) -> CloneResult:
        """Clone ``reference`` and make it say ``text``.

        ``engine_options`` is forwarded verbatim to the engine (``nfe_step``,
        ``cfg_strength``, ...), which lets the UI apply quality presets
        without mutating shared configuration.

        ``use_phonemes`` requests IPA input; it is always intersected with the
        engine's declared capability, because handing IPA to a grapheme
        tokenizer produces confident nonsense rather than an error.
        """
        report = progress if isinstance(progress, ProgressReporter) else ProgressReporter(progress)
        timings: dict[str, float] = {}
        warnings: list[str] = []

        engine_name = (engine or self.config.engines.tts).lower()
        tts = self.tts_engine(engine_name)

        # -- reference -----------------------------------------------------
        report(0.02, "Analysing reference audio")
        with _stopwatch(timings, "reference"):
            reference_result = self.prepare_reference(reference, overrides=preprocess_overrides)
        warnings.extend(reference_result.report_after.issues)

        # -- reference transcript (F5-TTS needs one) -----------------------
        resolved_reference_text = (reference_text or "").strip()
        if tts.capabilities.requires_reference_text and not resolved_reference_text:
            report(0.08, "Transcribing reference audio")
            with _stopwatch(timings, "asr"):
                try:
                    resolved_reference_text = self.transcribe_reference(reference_result.audio)
                except Exception as exc:
                    raise SynthesisError(
                        f"Could not transcribe the reference clip: {exc}",
                        hint="Type the reference transcript manually, or install faster-whisper.",
                    ) from exc
            if not resolved_reference_text:
                raise SynthesisError(
                    "The reference transcript came back empty.",
                    hint="The clip may contain no clear speech. Try another recording.",
                )
            log.info("Reference transcript: %s", resolved_reference_text)

        # -- text ----------------------------------------------------------
        report(0.12, "Preparing text")
        wants_phonemes = (
            self.text_pipeline.should_use_phonemes(engine_name)
            if use_phonemes is None
            else bool(use_phonemes)
        )
        if wants_phonemes and not tts.capabilities.accepts_phonemes:
            log.info(
                "%s reads graphemes, not phonemes; sending Persian script instead.",
                tts.display_name,
            )
            wants_phonemes = False

        with _stopwatch(timings, "text"):
            prepared = self.text_pipeline.prepare(
                text, engine_name=engine_name, use_phonemes=wants_phonemes
            )

        if not prepared.chunks:
            raise SynthesisError("There is nothing to synthesise after normalisation.")

        # -- synthesis -----------------------------------------------------
        report(0.15, f"Loading {tts.display_name}")
        with _stopwatch(timings, "load"):
            tts.load()

        takes = max(1, int(self.config.evaluation.candidates))

        def render(take_seed: int | None, base: float, span: float) -> tuple[Audio, int | None]:
            rendered: list[Audio] = []
            gaps: list[int] = []
            used = take_seed

            for position, chunk in enumerate(prepared.chunks):
                fraction = base + span * 0.8 * (position / max(len(prepared.chunks), 1))
                report(fraction, f"Synthesising chunk {position + 1}/{len(prepared.chunks)}")

                result = tts.synthesize(
                    chunk.synth_text,
                    reference_result.audio,
                    reference_text=resolved_reference_text or None,
                    speed=speed,
                    seed=used,
                    is_phonemes=bool(chunk.phonemes),
                    **(engine_options or {}),
                )
                # Keep the seed stable across chunks so timbre does not drift.
                if used is None:
                    used = result.seed

                rendered.append(strip_reference_leakage(result.audio))
                gaps.append(chunk.gap_before_ms)

            if not rendered:
                raise SynthesisError("The engine produced no audio.")

            report(base + span * 0.9, "Joining and mastering")
            output_settings = self.config.audio.output
            joined = join_chunks(
                rendered,
                sample_rate=rendered[0].sample_rate,
                crossfade_ms=output_settings.join_crossfade_ms,
                gaps_ms=gaps,
            )
            return (
                master(
                    joined,
                    target_lufs=output_settings.loudness_lufs,
                    peak_dbfs=output_settings.peak_dbfs,
                    fade_ms=output_settings.fade_ms,
                ),
                used,
            )

        with _stopwatch(timings, "synthesis"):
            if takes == 1:
                final, used_seed = render(seed, 0.2, 0.7)
            else:
                final, used_seed = self._render_best_of(
                    render,
                    takes=takes,
                    seed=seed,
                    reference=reference_result.audio,
                    expected_text=prepared.normalized,
                    warnings=warnings,
                    report=report,
                )

        clone = CloneResult(
            audio=final,
            engine=engine_name,
            mode="tts",
            seed=used_seed,
            reference=reference_result,
            text=prepared,
            reference_transcript=resolved_reference_text or None,
            timings=timings,
            warnings=warnings,
        )

        self._finalise(clone, report, output_path, expected_text=prepared.normalized)
        return clone

    def _render_best_of(
        self,
        render: Callable[[int | None, float, float], tuple[Audio, int | None]],
        *,
        takes: int,
        seed: int | None,
        reference: Audio,
        expected_text: str,
        warnings: list[str],
        report: ProgressReporter,
    ) -> tuple[Audio, int | None]:
        """Render ``takes`` candidates and return the best one.

        Both metrics are quantised to their own measurement noise and then
        compared lexicographically - WER clamped at the transcription floor
        first, bucketed speaker similarity second. `_candidate_rank` carries
        the reasoning and the measurements.

        Two consequences worth knowing about. Takes whose WER is inside the
        floor are ordered purely on similarity, which is the case this exists
        for. And when several takes land in the same bucket on *both* axes
        they are genuinely indistinguishable by anything measured here; the
        pick is then arbitrary but deterministic, and the result says so
        rather than implying the winner was better.
        """
        from ..evaluation.similarity import speaker_similarity, word_error_rate

        settings = self.config.evaluation
        floor = float(settings.candidate_wer_floor)
        step = float(settings.candidate_similarity_step)
        span = 0.7 / takes
        scored: list[tuple[float | None, float | None, Audio, int | None]] = []

        for index in range(takes):
            base = 0.2 + span * index
            report(base, f"Take {index + 1}/{takes}")
            # Derive each take from the caller's seed so a run stays
            # reproducible; None means the engine picks its own.
            take_seed = None if seed is None else seed + index
            audio, used = render(take_seed, base, span)

            wer: float | None = None
            try:
                heard = self.asr.transcribe(audio).text.strip()
                wer = word_error_rate(expected_text, heard)
            except Exception:
                log.debug("Round-trip transcription failed for take %d", index + 1, exc_info=True)

            # None, not 0.0: an unscored take and a dissimilar one are
            # different facts, and only one of them is about the audio.
            similarity: float | None = None
            try:
                result = speaker_similarity(reference, audio)
                if result is not None:
                    similarity = result.score
            except Exception:
                log.debug("Similarity scoring failed for take %d", index + 1, exc_info=True)

            log.info(
                "Take %d/%d: similarity %s, WER %s",
                index + 1,
                takes,
                "n/a" if similarity is None else f"{similarity:.3f}",
                "n/a" if wer is None else f"{wer * 100:.1f}%",
            )
            scored.append((wer, similarity, audio, used))

        def rank(row: tuple[float | None, float | None, Audio, int | None]) -> tuple[float, float]:
            return _candidate_rank(row[0], row[1], floor, step)

        best = min(scored, key=rank)
        tied = sum(1 for row in scored if rank(row) == rank(best))

        if all(row[1] is None for row in scored):
            warnings.append(
                "Speaker similarity was unavailable, so takes could only be ranked on "
                "transcription. Install speechbrain to let best-of-N compare voices."
            )
        elif tied > 1:
            # Say it plainly: on everything measured here these takes are the
            # same, so N renders bought nothing this time.
            warnings.append(
                f"{tied} of {takes} takes scored identically once WER and similarity "
                "were quantised to their measurement noise; the pick between them is "
                "arbitrary. Best-of-N is not helping on this text."
            )

        if best[0] is not None and best[0] > floor:
            warnings.append(
                f"Every take transcribed back above the {floor * 100:.0f}% noise floor "
                f"(best {best[0] * 100:.1f}%), so all of them lost words. "
                "A cleaner reference or a lower temperature usually helps."
            )

        log.info(
            "Best of %d: similarity %s, WER %s%s",
            takes,
            "n/a" if best[1] is None else f"{best[1]:.3f}",
            "n/a" if best[0] is None else f"{best[0] * 100:.1f}%",
            f" ({tied} tied)" if tied > 1 else "",
        )
        return best[2], best[3]

    # ------------------------------------------------------------------
    # Speech to speech
    # ------------------------------------------------------------------
    def convert(
        self,
        source: ReferenceInput,
        reference: ReferenceInput,
        *,
        mode: str | None = None,
        engine: str | None = None,
        speed: float = 1.0,
        seed: int | None = None,
        source_text: str | None = None,
        output_path: str | Path | None = None,
        progress: Callable[[float, str], None] | None = None,
        preprocess_overrides: dict[str, Any] | None = None,
        engine_options: dict[str, Any] | None = None,
        use_phonemes: bool | None = None,
    ) -> CloneResult:
        """Re-voice ``source`` audio with the timbre of ``reference``.

        Two strategies:

        * ``cascade`` - recognise the speech, run it through the Persian text
          pipeline, and re-synthesise. Best pronunciation, and it cleans up a
          poor-quality source, but the original delivery is replaced.
        * ``seedvc``  - acoustic conversion. Keeps the source's rhythm,
          emphasis and emotion; only the timbre changes.
        """
        strategy = (mode or self.config.engines.vc).lower()
        report = progress if isinstance(progress, ProgressReporter) else ProgressReporter(progress)

        if strategy in {"cascade", "asr", "tts"}:
            return self._convert_cascade(
                source,
                reference,
                engine=engine,
                speed=speed,
                seed=seed,
                source_text=source_text,
                output_path=output_path,
                progress=report,
                preprocess_overrides=preprocess_overrides,
                engine_options=engine_options,
                use_phonemes=use_phonemes,
            )
        return self._convert_direct(
            source,
            reference,
            engine=strategy,
            seed=seed,
            output_path=output_path,
            progress=report,
            preprocess_overrides=preprocess_overrides,
            engine_options=engine_options,
        )

    def _convert_cascade(
        self,
        source: ReferenceInput,
        reference: ReferenceInput,
        *,
        engine: str | None,
        speed: float,
        seed: int | None,
        source_text: str | None,
        output_path: str | Path | None,
        progress: ProgressReporter,
        preprocess_overrides: dict[str, Any] | None,
        engine_options: dict[str, Any] | None = None,
        use_phonemes: bool | None = None,
    ) -> CloneResult:
        timings: dict[str, float] = {}

        transcript = (source_text or "").strip()
        if not transcript:
            progress(0.05, "Transcribing source audio")
            with _stopwatch(timings, "asr"):
                source_audio = preprocess_source_audio(source, self.config.audio)
                result = self.asr.transcribe(source_audio)
                transcript = result.text.strip()

            if not transcript:
                raise SynthesisError(
                    "No speech was recognised in the source audio.",
                    hint="Check that the file contains clear speech, or type the text manually.",
                )
            log.info("Source transcript: %s", transcript)

        clone = self.synthesize(
            transcript,
            reference,
            engine=engine,
            speed=speed,
            seed=seed,
            output_path=output_path,
            progress=progress,
            preprocess_overrides=preprocess_overrides,
            engine_options=engine_options,
            use_phonemes=use_phonemes,
        )
        clone.mode = "s2s-cascade"
        clone.source_transcript = transcript
        for key, value in timings.items():
            clone.timings[key] = clone.timings.get(key, 0.0) + value
        return clone

    def _convert_direct(
        self,
        source: ReferenceInput,
        reference: ReferenceInput,
        *,
        engine: str,
        seed: int | None,
        output_path: str | Path | None,
        progress: ProgressReporter,
        preprocess_overrides: dict[str, Any] | None,
        engine_options: dict[str, Any] | None = None,
    ) -> CloneResult:
        timings: dict[str, float] = {}

        progress(0.05, "Preparing audio")
        with _stopwatch(timings, "reference"):
            reference_result = self.prepare_reference(reference, overrides=preprocess_overrides)
            source_audio = preprocess_source_audio(source, self.config.audio)

        vc = self.vc_engine(engine)
        progress(0.15, f"Loading {vc.display_name}")
        with _stopwatch(timings, "load"):
            vc.load()

        progress(0.3, "Converting voice")
        with _stopwatch(timings, "synthesis"):
            result = vc.convert(
                source_audio, reference_result.audio, seed=seed, **(engine_options or {})
            )

        progress(0.9, "Mastering")
        with _stopwatch(timings, "postprocess"):
            output_settings = self.config.audio.output
            final = master(
                result.audio,
                target_lufs=output_settings.loudness_lufs,
                peak_dbfs=output_settings.peak_dbfs,
                fade_ms=output_settings.fade_ms,
            )

        clone = CloneResult(
            audio=final,
            engine=engine,
            mode="s2s-vc",
            seed=result.seed,
            reference=reference_result,
            timings=timings,
        )
        self._finalise(clone, progress, output_path, expected_text=None)
        return clone

    # ------------------------------------------------------------------
    # Shared tail: evaluation + saving
    # ------------------------------------------------------------------
    def _finalise(
        self,
        clone: CloneResult,
        progress: ProgressReporter,
        output_path: str | Path | None,
        *,
        expected_text: str | None,
    ) -> None:
        evaluation = self.config.evaluation
        if evaluation.enabled and clone.reference is not None:
            progress(0.95, "Measuring quality")
            timings = clone.timings
            with _stopwatch(timings, "evaluation"):
                clone.scores = self._evaluate(clone, expected_text)

        progress(0.98, "Saving")
        target = Path(output_path) if output_path else self._default_output_path(clone)
        try:
            clone.output_path = save_audio(
                clone.audio,
                target,
                audio_format=self.config.audio.output.format,
                bit_depth=self.config.audio.output.bit_depth,
            )
            log.info("Wrote %s (%.1fs)", clone.output_path, clone.duration)
        except Exception as exc:
            clone.warnings.append(f"Could not write the output file: {exc}")
            log.warning("Saving failed", exc_info=True)

        progress(1.0, "Done")

    def _evaluate(self, clone: CloneResult, expected_text: str | None) -> QualityScores:
        scores = QualityScores()
        evaluation = self.config.evaluation

        if evaluation.speaker_similarity and clone.reference is not None:
            try:
                from ..evaluation.similarity import speaker_similarity

                result = speaker_similarity(clone.reference.audio, clone.audio)
                if result is not None:
                    scores.speaker_similarity = round(result.score, 4)
                    if not result.reliable:
                        clone.warnings.append(
                            "Speaker similarity is approximate - install speechbrain for a real score."
                        )
            except Exception:
                log.debug("Similarity scoring failed", exc_info=True)

        if evaluation.transcribe_check and expected_text:
            try:
                from ..evaluation.similarity import word_error_rate

                heard = self.asr.transcribe(clone.audio).text.strip()
                scores.transcription = heard
                scores.word_error_rate = word_error_rate(expected_text, heard)
            except Exception:
                log.debug("Round-trip transcription failed", exc_info=True)

        return scores

    def _default_output_path(self, clone: CloneResult) -> Path:
        directory = self.config.paths.ensure("output_dir")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        extension = self.config.audio.output.format
        return directory / f"{clone.mode}_{clone.engine}_{stamp}.{extension}"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, Any]:
        """Everything the Diagnostics tab shows."""
        from ..audio.denoise import available_backends as denoise_backends
        from ..device import describe_hardware
        from ..engines import engine_status
        from ..evaluation.similarity import available_backends as similarity_backends
        from ..models import free_disk_gb

        return {
            "hardware": describe_hardware(),
            "device": str(self.device),
            "engines": engine_status(),
            "asr": ASREngine.available_backends(),
            "denoise": denoise_backends(),
            "similarity": similarity_backends(),
            "g2p": self.text_pipeline.g2p_status(),
            "cache_dir": str(self.config.paths.resolved("cache_dir")),
            "output_dir": str(self.config.paths.resolved("output_dir")),
            "free_disk_gb": round(free_disk_gb(self.config.paths.resolved("cache_dir")), 1),
        }


def load_reference(source: ReferenceInput) -> Audio:
    """Public helper used by the UI to preview a file before processing."""
    if isinstance(source, Audio):
        return source
    return load_audio(source, mono=True)


__all__ = ["ReferenceInput", "Studio", "load_reference"]
