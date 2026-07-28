"""The Gradio interface.

Four tabs:

* **Text to Speech**   - text + reference voice -> cloned speech
* **Speech to Speech** - source audio + reference voice -> re-voiced speech
* **Voice Library**    - save, preview and delete reference voices
* **Diagnostics**      - what is installed, what is missing, and how to fix it

The UI layer only marshals values; every decision of substance lives in
:class:`voiceclone.pipelines.Studio`, so the CLI behaves identically.
"""

from __future__ import annotations

import html
from typing import Any

import gradio as gr

from ..config import AppConfig
from ..errors import VoiceCloneError
from ..logging_utils import ProgressReporter, get_logger
from ..pipelines import CloneResult, Studio, VoiceLibrary
from .strings import DEFAULT_PRESET, QUALITY_PRESETS, SAMPLE_TEXTS, get_strings
from .styles import CSS, build_theme, quality_badge, similarity_badge

log = get_logger("ui")


def _gradio_major() -> int:
    try:
        return int(str(gr.__version__).split(".", 1)[0])
    except (ValueError, AttributeError):  # pragma: no cover - exotic builds
        return 5


#: Gradio 6 moved `theme` and `css` from the Blocks constructor to `launch()`.
_STYLE_ON_LAUNCH = _gradio_major() >= 6

REFERENCE_UPLOAD = "Upload / record"
REFERENCE_SAVED = "Saved voice"

CONVERSION_MODES = [
    ("Re-speak (ASR -> Persian G2P -> TTS) - best pronunciation", "cascade"),
    ("Voice conversion (Seed-VC) - keeps original delivery", "seedvc"),
]


class VoiceCloneUI:
    """Builds and serves the Gradio app around a single :class:`Studio`."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig.load()
        self.strings = get_strings(self.config.ui.language)
        self.library = VoiceLibrary(self.config.paths.ensure("voices_dir"))
        self._studio: Studio | None = None

    # ------------------------------------------------------------------
    # Lazy studio: never build models at import/launch time
    # ------------------------------------------------------------------
    @property
    def studio(self) -> Studio:
        if self._studio is None:
            self._studio = Studio(self.config)
        return self._studio

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_reference(self, mode: str, uploaded: str | None, saved_slug: str | None) -> Any:
        """Turn the two reference inputs into something the Studio accepts."""
        if mode == REFERENCE_SAVED:
            if not saved_slug:
                raise gr.Error("Choose a saved voice, or switch to upload.")
            try:
                _, audio = self.library.load(saved_slug)
            except VoiceCloneError as exc:
                raise gr.Error(str(exc)) from exc
            return audio

        if not uploaded:
            raise gr.Error(self.strings["no_reference"])
        return uploaded

    @staticmethod
    def _preprocess_overrides(
        denoise: str, separate_vocals: bool, trim_silence: bool, auto_window: bool
    ) -> dict[str, Any]:
        return {
            "denoise": denoise,
            "separate_vocals": bool(separate_vocals),
            "trim_silence": bool(trim_silence),
            "reference.auto_select_window": bool(auto_window),
        }

    @staticmethod
    def _engine_options(preset_name: str, engine: str) -> dict[str, Any]:
        preset = QUALITY_PRESETS.get(preset_name, {})
        if engine == "f5":
            return {
                key: preset[key] for key in ("nfe_step", "cfg_strength") if key in preset
            }
        if engine == "seedvc":
            return {"diffusion_steps": preset["diffusion_steps"]} if "diffusion_steps" in preset else {}
        return {}

    @staticmethod
    def _badges(result: CloneResult) -> str:
        parts = [similarity_badge(result.scores.speaker_similarity)]
        if result.reference is not None:
            report = result.reference.report_after
            parts.append(quality_badge(report.score, f"reference {report.verdict}"))
        if result.text is not None and result.text.used_phonemes:
            parts.append('<span class="badge badge-good">IPA PHONEMES</span>')
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def on_analyse(
        self,
        mode: str,
        uploaded: str | None,
        saved_slug: str | None,
        denoise: str,
        separate_vocals: bool,
        trim_silence: bool,
        auto_window: bool,
        progress: gr.Progress = gr.Progress(),  # noqa: B008 - Gradio idiom
    ) -> tuple[Any, str, str, str]:
        """Preprocess the reference, show its quality report, fill the transcript."""
        reference = self._resolve_reference(mode, uploaded, saved_slug)
        report = ProgressReporter.from_gradio(progress)

        report(0.2, "Processing reference audio")
        try:
            result = self.studio.prepare_reference(
                reference,
                overrides=self._preprocess_overrides(denoise, separate_vocals, trim_silence, auto_window),
            )
        except VoiceCloneError as exc:
            raise gr.Error(str(exc)) from exc

        report(0.7, "Transcribing")
        transcript = ""
        try:
            transcript = self.studio.transcribe_reference(result.audio)
        except Exception as exc:  # ASR is optional at this stage
            log.info("Reference transcription unavailable: %s", exc)

        report(1.0, "Done")
        from ..audio.io import to_gradio

        after = result.report_after
        summary = result.summary()
        if transcript:
            summary += f"\n\nTranscript: {transcript}"

        return (
            to_gradio(result.audio),
            summary,
            quality_badge(after.score, after.verdict),
            transcript,
        )

    def on_generate_tts(
        self,
        text: str,
        mode: str,
        uploaded: str | None,
        saved_slug: str | None,
        reference_text: str,
        engine: str,
        preset: str,
        speed: float,
        seed: float,
        use_phonemes: bool,
        denoise: str,
        separate_vocals: bool,
        trim_silence: bool,
        auto_window: bool,
        progress: gr.Progress = gr.Progress(),  # noqa: B008
    ) -> tuple[Any, str, str]:
        if not text or not text.strip():
            raise gr.Error(self.strings["no_text"])

        reference = self._resolve_reference(mode, uploaded, saved_slug)

        # The phoneme toggle is a per-request override of the config default.
        studio = self.studio
        studio.text_pipeline.config.g2p.apply_to_engines = (
            [engine] if use_phonemes else []
        )

        try:
            result = studio.synthesize(
                text,
                reference,
                engine=engine,
                reference_text=(reference_text or "").strip() or None,
                speed=float(speed),
                seed=int(seed) or None,
                progress=ProgressReporter.from_gradio(progress),
                preprocess_overrides=self._preprocess_overrides(
                    denoise, separate_vocals, trim_silence, auto_window
                ),
                engine_options=self._engine_options(preset, engine),
            )
        except VoiceCloneError as exc:
            raise gr.Error(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - unexpected engine failure
            log.exception("Synthesis failed")
            raise gr.Error(f"Synthesis failed: {exc}") from exc

        from ..audio.io import to_gradio

        return to_gradio(result.audio), result.report(), self._badges(result)

    def on_generate_s2s(
        self,
        source: str | None,
        conversion_mode: str,
        mode: str,
        uploaded: str | None,
        saved_slug: str | None,
        engine: str,
        preset: str,
        speed: float,
        seed: float,
        source_text: str,
        use_phonemes: bool,
        denoise: str,
        separate_vocals: bool,
        trim_silence: bool,
        auto_window: bool,
        progress: gr.Progress = gr.Progress(),  # noqa: B008
    ) -> tuple[Any, str, str]:
        if not source:
            raise gr.Error(self.strings["no_source"])

        reference = self._resolve_reference(mode, uploaded, saved_slug)

        studio = self.studio
        studio.text_pipeline.config.g2p.apply_to_engines = [engine] if use_phonemes else []

        target_engine = engine if conversion_mode == "cascade" else conversion_mode

        try:
            result = studio.convert(
                source,
                reference,
                mode=conversion_mode,
                engine=engine,
                speed=float(speed),
                seed=int(seed) or None,
                source_text=(source_text or "").strip() or None,
                progress=ProgressReporter.from_gradio(progress),
                preprocess_overrides=self._preprocess_overrides(
                    denoise, separate_vocals, trim_silence, auto_window
                ),
                engine_options=self._engine_options(preset, target_engine),
            )
        except VoiceCloneError as exc:
            raise gr.Error(str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            log.exception("Conversion failed")
            raise gr.Error(f"Conversion failed: {exc}") from exc

        from ..audio.io import to_gradio

        return to_gradio(result.audio), result.report(), self._badges(result)

    # -- voice library -----------------------------------------------------
    def on_save_voice(
        self,
        name: str,
        mode: str,
        uploaded: str | None,
        saved_slug: str | None,
        transcript: str,
        notes: str,
        denoise: str,
        separate_vocals: bool,
        trim_silence: bool,
        auto_window: bool,
        progress: gr.Progress = gr.Progress(),  # noqa: B008
    ) -> tuple[str, Any]:
        if not name or not name.strip():
            raise gr.Error("Give the voice a name.")

        reference = self._resolve_reference(mode, uploaded, saved_slug)
        report = ProgressReporter.from_gradio(progress)

        report(0.2, "Processing reference")
        try:
            processed = self.studio.prepare_reference(
                reference,
                overrides=self._preprocess_overrides(denoise, separate_vocals, trim_silence, auto_window),
            )
        except VoiceCloneError as exc:
            raise gr.Error(str(exc)) from exc

        resolved_transcript = (transcript or "").strip()
        if not resolved_transcript:
            report(0.6, "Transcribing")
            try:
                resolved_transcript = self.studio.transcribe_reference(processed.audio)
            except Exception as exc:
                log.info("Could not transcribe while saving: %s", exc)

        report(0.9, "Saving")
        profile = self.library.save(
            name,
            processed.audio,
            transcript=resolved_transcript,
            notes=notes,
            quality_score=processed.report_after.score,
            quality_verdict=processed.report_after.verdict,
            preprocessing=processed.steps,
        )
        gr.Info(f"Saved '{profile.name}'")
        return self._voice_table(), gr.update(choices=self.library.choices(), value=profile.slug)

    def on_delete_voice(self, slug: str | None) -> tuple[str, Any]:
        if not slug:
            raise gr.Error("Select a voice to delete.")
        if self.library.delete(slug):
            gr.Info(f"Deleted '{slug}'")
        else:
            gr.Warning(f"Voice '{slug}' was not found.")
        return self._voice_table(), gr.update(choices=self.library.choices(), value=None)

    def on_refresh_voices(self) -> tuple[str, Any]:
        return self._voice_table(), gr.update(choices=self.library.choices())

    def _voice_table(self) -> str:
        profiles = self.library.list()
        if not profiles:
            return "<p><em>No saved voices yet. Process a reference and save it here.</em></p>"

        rows = [
            "<div class='diagnostics'><table><thead><tr>"
            "<th>Name</th><th>Length</th><th>Quality</th><th>Language</th>"
            "<th>Created</th><th>Transcript</th></tr></thead><tbody>"
        ]
        for profile in profiles:
            snippet = profile.transcript[:70] + ("..." if len(profile.transcript) > 70 else "")
            rows.append(
                "<tr>"
                f"<td><strong>{html.escape(profile.name)}</strong><br>"
                f"<code>{html.escape(profile.slug)}</code></td>"
                f"<td>{profile.duration:.1f}s</td>"
                f"<td>{quality_badge(profile.quality_score, profile.quality_verdict or 'n/a')}</td>"
                f"<td>{html.escape(profile.language)}</td>"
                f"<td>{html.escape(profile.created_at)}</td>"
                f"<td dir='auto'>{html.escape(snippet)}</td>"
                "</tr>"
            )
        rows.append("</tbody></table></div>")
        return "".join(rows)

    # -- diagnostics -------------------------------------------------------
    def on_diagnostics(self) -> str:
        try:
            info = self.studio.diagnostics()
        except Exception as exc:  # pragma: no cover
            return f"<p>Could not gather diagnostics: {html.escape(str(exc))}</p>"

        def yes_no(value: bool) -> str:
            css = "badge-good" if value else "badge-bad"
            return f"<span class='badge {css}'>{'available' if value else 'missing'}</span>"

        sections: list[str] = ["<div class='diagnostics'>"]

        sections.append("<h3>System</h3><table><tbody>")
        for key, value in info["hardware"].items():
            sections.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>")
        sections.append(
            f"<tr><th>compute</th><td>{html.escape(info['device'])}</td></tr>"
            f"<tr><th>free disk</th><td>{info['free_disk_gb']} GB</td></tr>"
            f"<tr><th>cache</th><td><code>{html.escape(info['cache_dir'])}</code></td></tr>"
            f"<tr><th>outputs</th><td><code>{html.escape(info['output_dir'])}</code></td></tr>"
            "</tbody></table>"
        )

        sections.append("<h3>Engines</h3><table><thead><tr><th>Engine</th><th>Type</th>"
                        "<th>Status</th><th>Persian</th><th>Phonemes</th><th>Download</th>"
                        "</tr></thead><tbody>")
        for row in info["engines"]:
            status = yes_no(row["available"])
            if not row["available"] and row["reason"]:
                status += f"<br><small>{html.escape(row['reason'])}</small>"
            sections.append(
                f"<tr><td><strong>{html.escape(row['display_name'])}</strong><br>"
                f"<code>{html.escape(row['name'])}</code></td>"
                f"<td>{html.escape(row['kind'].upper())}</td>"
                f"<td>{status}</td>"
                f"<td>{'yes' if row['native_persian'] else 'no'}</td>"
                f"<td>{'yes' if row['accepts_phonemes'] else 'no'}</td>"
                f"<td>{row['download_gb'] or '-'} GB</td></tr>"
            )
        sections.append("</tbody></table>")

        sections.append(
            "<h3>Persian G2P</h3><p>"
            f"{html.escape(info['g2p'])}</p>"
            "<p class='hint'>Phoneme input resolves short vowels, ezafe and homographs "
            "before synthesis. Without it, engines read raw Persian script and guess.</p>"
        )

        sections.append("<h3>Optional components</h3><table><tbody>")
        for label, mapping in (
            ("Speech recognition", info["asr"]),
            ("Noise reduction", info["denoise"]),
            ("Similarity scoring", info["similarity"]),
        ):
            entries = " ".join(
                f"{html.escape(key)} {yes_no(bool(value))}" for key, value in mapping.items()
            )
            sections.append(f"<tr><th>{html.escape(label)}</th><td>{entries}</td></tr>")
        sections.append("</tbody></table></div>")

        return "".join(sections)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _reference_block(self, prefix: str) -> dict[str, Any]:
        """Shared reference-voice controls. Returns the components by name."""
        strings = self.strings
        components: dict[str, Any] = {}

        components["mode"] = gr.Radio(
            choices=[REFERENCE_UPLOAD, REFERENCE_SAVED],
            value=REFERENCE_UPLOAD,
            label=strings["reference"],
            elem_id=f"{prefix}-ref-mode",
        )
        components["upload"] = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label=strings["reference"],
        )
        gr.Markdown(f"<div class='hint'>{strings['reference_help']}</div>")
        components["saved"] = gr.Dropdown(
            choices=self.library.choices(),
            label=strings["use_saved"],
            visible=False,
            interactive=True,
        )

        components["mode"].change(
            fn=lambda choice: (
                gr.update(visible=choice == REFERENCE_UPLOAD),
                gr.update(visible=choice == REFERENCE_SAVED),
            ),
            inputs=[components["mode"]],
            outputs=[components["upload"], components["saved"]],
        )
        return components

    def _preprocessing_block(self) -> dict[str, Any]:
        strings = self.strings
        components: dict[str, Any] = {}
        with gr.Accordion(strings["preprocessing"], open=False):
            with gr.Row():
                components["denoise"] = gr.Dropdown(
                    choices=[
                        ("Automatic (skip when already clean)", "auto"),
                        ("DeepFilterNet (best)", "deepfilternet"),
                        ("Spectral gating", "noisereduce"),
                        ("Off", "none"),
                    ],
                    value="auto",
                    label=strings["denoise"],
                )
                components["separate_vocals"] = gr.Checkbox(
                    value=False, label=strings["separate_vocals"]
                )
            with gr.Row():
                components["trim_silence"] = gr.Checkbox(value=True, label=strings["trim_silence"])
                components["auto_window"] = gr.Checkbox(value=True, label=strings["auto_window"])
        return components

    def _engine_choices(self) -> list[tuple[str, str]]:
        from ..engines import list_tts_engines

        choices: list[tuple[str, str]] = []
        for key, cls in list_tts_engines().items():
            available, reason = cls.is_available()
            label = cls.display_name if available else f"{cls.display_name} - unavailable ({reason})"
            choices.append((label, key))
        return choices or [("F5-TTS", "f5")]

    def build(self) -> gr.Blocks:
        strings = self.strings

        block_kwargs: dict[str, Any] = {"title": strings["title"], "analytics_enabled": False}
        if not _STYLE_ON_LAUNCH:
            block_kwargs["theme"] = build_theme(self.config.ui.theme)
            block_kwargs["css"] = CSS

        with gr.Blocks(**block_kwargs) as demo:
            gr.HTML(
                f"<div id='app-header'><h1>{html.escape(strings['title'])}</h1>"
                f"<p>{html.escape(strings['subtitle'])}</p></div>"
            )

            with gr.Tabs():
                # ---------------------------------------------------------
                # Text to speech
                # ---------------------------------------------------------
                with gr.Tab(strings["tab_tts"]):
                    with gr.Row():
                        with gr.Column(scale=3):
                            tts_text = gr.Textbox(
                                label=strings["text"],
                                placeholder=strings["text_placeholder"],
                                lines=8,
                                elem_classes=["rtl"],
                            )
                            gr.Examples(examples=[[text] for text in SAMPLE_TEXTS], inputs=[tts_text])

                        with gr.Column(scale=2):
                            tts_ref = self._reference_block("tts")
                            tts_ref_text = gr.Textbox(
                                label=strings["reference_text"],
                                placeholder=strings["reference_text_help"],
                                lines=2,
                                elem_classes=["rtl"],
                            )
                            tts_analyse = gr.Button(strings["analyse"], size="sm")

                    with gr.Row():
                        tts_engine = gr.Dropdown(
                            choices=self._engine_choices(),
                            value=self.config.engines.tts,
                            label=strings["engine"],
                        )
                        tts_preset = gr.Dropdown(
                            choices=list(QUALITY_PRESETS),
                            value=DEFAULT_PRESET,
                            label=strings["quality_preset"],
                        )
                        tts_speed = gr.Slider(0.5, 1.6, value=1.0, step=0.05, label=strings["speed"])
                        tts_seed = gr.Number(value=0, precision=0, label=strings["seed"])

                    tts_phonemes = gr.Checkbox(
                        value=True,
                        label=strings["use_phonemes"],
                        info=strings["use_phonemes_help"],
                    )
                    tts_pre = self._preprocessing_block()

                    tts_button = gr.Button(
                        strings["generate"], variant="primary", elem_classes=["primary-action"]
                    )

                    with gr.Row():
                        tts_out = gr.Audio(label=strings["output"], type="numpy", autoplay=False)
                    tts_badges = gr.HTML()
                    tts_report = gr.Textbox(
                        label=strings["details"], lines=16, elem_classes=["report"]
                    )

                # ---------------------------------------------------------
                # Speech to speech
                # ---------------------------------------------------------
                with gr.Tab(strings["tab_s2s"]):
                    with gr.Row():
                        with gr.Column(scale=3):
                            s2s_source = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label=strings["source_audio"],
                            )
                            gr.Markdown(f"<div class='hint'>{strings['source_audio_help']}</div>")
                            s2s_mode = gr.Radio(
                                choices=CONVERSION_MODES,
                                value="cascade",
                                label=strings["mode"],
                            )
                            s2s_text = gr.Textbox(
                                label="Override source transcript (optional)",
                                lines=2,
                                elem_classes=["rtl"],
                            )
                        with gr.Column(scale=2):
                            s2s_ref = self._reference_block("s2s")

                    with gr.Row():
                        s2s_engine = gr.Dropdown(
                            choices=self._engine_choices(),
                            value=self.config.engines.tts,
                            label=strings["engine"],
                        )
                        s2s_preset = gr.Dropdown(
                            choices=list(QUALITY_PRESETS), value=DEFAULT_PRESET, label=strings["quality_preset"]
                        )
                        s2s_speed = gr.Slider(0.5, 1.6, value=1.0, step=0.05, label=strings["speed"])
                        s2s_seed = gr.Number(value=0, precision=0, label=strings["seed"])

                    s2s_phonemes = gr.Checkbox(value=True, label=strings["use_phonemes"])
                    s2s_pre = self._preprocessing_block()

                    s2s_button = gr.Button(
                        strings["generate"], variant="primary", elem_classes=["primary-action"]
                    )
                    s2s_out = gr.Audio(label=strings["output"], type="numpy")
                    s2s_badges = gr.HTML()
                    s2s_report = gr.Textbox(
                        label=strings["details"], lines=16, elem_classes=["report"]
                    )

                # ---------------------------------------------------------
                # Voice library
                # ---------------------------------------------------------
                with gr.Tab(strings["tab_voices"]):
                    gr.Markdown(
                        "Saving a voice stores the *processed* reference plus its transcript, "
                        "so later renders skip preprocessing and stay consistent."
                    )
                    with gr.Row():
                        with gr.Column(scale=2):
                            lib_ref = self._reference_block("lib")
                            lib_name = gr.Textbox(label=strings["voice_name"], placeholder="e.g. Narrator")
                            lib_transcript = gr.Textbox(
                                label=strings["reference_text"], lines=2, elem_classes=["rtl"]
                            )
                            lib_notes = gr.Textbox(label="Notes", lines=2)
                            lib_pre = self._preprocessing_block()
                            lib_save = gr.Button(strings["save_voice"], variant="primary")
                        with gr.Column(scale=3):
                            lib_table = gr.HTML(self._voice_table())
                            with gr.Row():
                                lib_pick = gr.Dropdown(
                                    choices=self.library.choices(), label=strings["saved_voice"]
                                )
                                lib_refresh = gr.Button(strings["refresh"], size="sm")
                                lib_delete = gr.Button(strings["delete"], variant="stop", size="sm")

                # ---------------------------------------------------------
                # Diagnostics
                # ---------------------------------------------------------
                with gr.Tab(strings["tab_diagnostics"]):
                    diag_button = gr.Button("Run diagnostics", variant="primary")
                    diag_out = gr.HTML()
                    diag_button.click(fn=self.on_diagnostics, outputs=[diag_out])
                    demo.load(fn=self.on_diagnostics, outputs=[diag_out])

            # -- wiring --------------------------------------------------
            def preprocessing_inputs(block: dict[str, Any]) -> list[Any]:
                return [
                    block["denoise"],
                    block["separate_vocals"],
                    block["trim_silence"],
                    block["auto_window"],
                ]

            tts_analyse.click(
                fn=self.on_analyse,
                inputs=[tts_ref["mode"], tts_ref["upload"], tts_ref["saved"], *preprocessing_inputs(tts_pre)],
                outputs=[tts_out, tts_report, tts_badges, tts_ref_text],
            )

            tts_button.click(
                fn=self.on_generate_tts,
                inputs=[
                    tts_text,
                    tts_ref["mode"], tts_ref["upload"], tts_ref["saved"],
                    tts_ref_text,
                    tts_engine, tts_preset, tts_speed, tts_seed, tts_phonemes,
                    *preprocessing_inputs(tts_pre),
                ],
                outputs=[tts_out, tts_report, tts_badges],
            )

            s2s_button.click(
                fn=self.on_generate_s2s,
                inputs=[
                    s2s_source, s2s_mode,
                    s2s_ref["mode"], s2s_ref["upload"], s2s_ref["saved"],
                    s2s_engine, s2s_preset, s2s_speed, s2s_seed, s2s_text, s2s_phonemes,
                    *preprocessing_inputs(s2s_pre),
                ],
                outputs=[s2s_out, s2s_report, s2s_badges],
            )

            lib_save.click(
                fn=self.on_save_voice,
                inputs=[
                    lib_name,
                    lib_ref["mode"], lib_ref["upload"], lib_ref["saved"],
                    lib_transcript, lib_notes,
                    *preprocessing_inputs(lib_pre),
                ],
                outputs=[lib_table, lib_pick],
            )
            lib_delete.click(fn=self.on_delete_voice, inputs=[lib_pick], outputs=[lib_table, lib_pick])
            lib_refresh.click(fn=self.on_refresh_voices, outputs=[lib_table, lib_pick])

            gr.Markdown(
                "<div class='hint'>Only clone voices you have permission to use. "
                "Generated audio is synthetic - label it as such.</div>"
            )

        return demo

    # ------------------------------------------------------------------
    def launch(self, **kwargs: Any) -> None:
        settings = self.config.ui
        options: dict[str, Any] = {
            "server_name": settings.server_name,
            "server_port": settings.server_port,
            "share": settings.share,
            "show_error": True,
            "max_file_size": f"{settings.max_upload_mb}mb",
        }
        if _STYLE_ON_LAUNCH:
            options["theme"] = build_theme(settings.theme)
            options["css"] = CSS
        options.update(kwargs)

        demo = self.build()
        # Model loads are serialised: two concurrent renders would fight over
        # accelerator memory and thrash.
        demo.queue(default_concurrency_limit=1)
        demo.launch(**options)


def build_app(config: AppConfig | None = None) -> gr.Blocks:
    """Build the Blocks app without launching it (useful for tests/mounting)."""
    return VoiceCloneUI(config).build()


def launch(config: AppConfig | None = None, **kwargs: Any) -> None:
    VoiceCloneUI(config).launch(**kwargs)


__all__ = ["VoiceCloneUI", "build_app", "launch"]
