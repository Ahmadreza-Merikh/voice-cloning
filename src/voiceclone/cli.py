"""Command-line interface.

    voiceclone tts   "سلام دنیا" --reference ref.wav --output out.wav
    voiceclone s2s   source.wav  --reference ref.wav --mode seedvc
    voiceclone text  "قیمت ۲۵۰۰ تومان است"          # inspect the text pipeline
    voiceclone check ref.wav                        # reference quality report
    voiceclone doctor                               # environment diagnostics
    voiceclone voices list

Everything runs through the same :class:`~voiceclone.pipelines.Studio` as the
web UI, so results are identical between the two.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import AppConfig
from .errors import VoiceCloneError
from .logging_utils import get_logger, setup_logging

log = get_logger("cli")


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voiceclone",
        description="Persian voice cloning: text-to-speech and speech-to-speech",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser, *, reference_required: bool = True) -> None:
        sub.add_argument("-r", "--reference", required=reference_required, help="Reference voice audio")
        sub.add_argument("-o", "--output", default=None, help="Output audio path")
        sub.add_argument("-e", "--engine", default=None, help="TTS engine (f5 / moss)")
        sub.add_argument("--speed", type=float, default=1.0)
        sub.add_argument("--seed", type=int, default=0, help="0 = random")
        sub.add_argument(
            "--held-out",
            nargs="+",
            default=None,
            metavar="CLIP",
            help=(
                "Other recordings of the same speaker, not used as the prompt. "
                "Scoring against these is the only similarity figure that measures "
                "identity rather than channel."
            ),
        )
        sub.add_argument(
            "--best-of",
            type=int,
            default=None,
            metavar="N",
            help="Render N takes and keep the best (costs N x the time)",
        )
        sub.add_argument("--no-denoise", action="store_true")
        sub.add_argument("--separate-vocals", action="store_true", help="Strip music with Demucs")
        sub.add_argument("--no-phonemes", action="store_true", help="Disable Persian G2P")

    tts = subparsers.add_parser("tts", help="Synthesise text in a cloned voice")
    tts.add_argument("text", help="Text to speak, or @path/to/file.txt")
    tts.add_argument("--reference-text", default=None, help="Transcript of the reference clip")
    add_common(tts)

    s2s = subparsers.add_parser("s2s", help="Re-voice an audio file")
    s2s.add_argument("source", help="Source audio file")
    s2s.add_argument(
        "--mode", default="cascade", choices=["cascade", "seedvc"], help="Conversion strategy"
    )
    s2s.add_argument("--source-text", default=None, help="Skip ASR and use this transcript")
    add_common(s2s)

    text_cmd = subparsers.add_parser("text", help="Inspect the Persian text pipeline")
    text_cmd.add_argument("text", help="Text to analyse, or @path/to/file.txt")
    text_cmd.add_argument("-e", "--engine", default="moss", help="Engine the text is destined for")

    check = subparsers.add_parser("check", help="Analyse a reference recording")
    check.add_argument("audio", help="Audio file to inspect")
    check.add_argument("--save", default=None, help="Write the processed reference here")

    subparsers.add_parser("doctor", help="Report installed engines and backends")

    voices = subparsers.add_parser("voices", help="Manage the voice library")
    voices.add_argument("action", choices=["list", "add", "delete"])
    voices.add_argument("name", nargs="?", help="Voice name (add) or slug (delete)")
    voices.add_argument("-r", "--reference", default=None, help="Audio file for `add`")
    voices.add_argument("--transcript", default=None)

    return parser


def _read_text_argument(value: str) -> str:
    """Support ``@file.txt`` to read long input from disk."""
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        if not path.is_file():
            raise VoiceCloneError(f"Text file not found: {path}")
        return path.read_text(encoding="utf-8")
    return value


def _build_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.load(args.config)
    overrides: dict[str, object] = {}
    if args.device:
        overrides["runtime.device"] = args.device
    return config.copy_with(**overrides) if overrides else config


def _preprocess_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if getattr(args, "no_denoise", False):
        overrides["denoise"] = "none"
    if getattr(args, "separate_vocals", False):
        overrides["separate_vocals"] = True
    return overrides


def _phoneme_preference(args: argparse.Namespace) -> bool | None:
    """``False`` when --no-phonemes was passed, otherwise the config default.

    Never forces phonemes *on*: whether an engine can accept IPA is a property
    of the engine, resolved inside the Studio.
    """
    return False if getattr(args, "no_phonemes", False) else None


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def command_tts(args: argparse.Namespace) -> int:
    from .pipelines import Studio

    config = _build_config(args)
    studio = Studio(config)
    engine = (args.engine or config.engines.tts).lower()

    result = studio.synthesize(
        _read_text_argument(args.text),
        args.reference,
        engine=engine,
        reference_text=args.reference_text,
        speed=args.speed,
        seed=args.seed or None,
        output_path=args.output,
        preprocess_overrides=_preprocess_overrides(args),
        use_phonemes=_phoneme_preference(args),
        candidates=max(1, int(args.best_of)) if getattr(args, "best_of", None) else None,
        held_out=getattr(args, "held_out", None),
    )

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.report())
        print(f"\nSaved: {result.output_path}")
    return 0


def command_s2s(args: argparse.Namespace) -> int:
    from .pipelines import Studio

    config = _build_config(args)
    studio = Studio(config)
    engine = (args.engine or config.engines.tts).lower()

    result = studio.convert(
        args.source,
        args.reference,
        mode=args.mode,
        engine=engine,
        speed=args.speed,
        seed=args.seed or None,
        source_text=args.source_text,
        output_path=args.output,
        preprocess_overrides=_preprocess_overrides(args),
        use_phonemes=_phoneme_preference(args),
        candidates=max(1, int(args.best_of)) if getattr(args, "best_of", None) else None,
        held_out=getattr(args, "held_out", None),
    )

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.report())
        print(f"\nSaved: {result.output_path}")
    return 0


def command_text(args: argparse.Namespace) -> int:
    from .text import TextPipeline

    config = _build_config(args)
    pipeline = TextPipeline(config.text)
    prepared = pipeline.prepare(_read_text_argument(args.text), engine_name=args.engine)

    if args.json:
        print(json.dumps(prepared.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(prepared.summary())
    print(f"\nNormalised:\n  {prepared.normalized}")
    print(f"\nChunks ({len(prepared.chunks)}):")
    for chunk in prepared.chunks:
        print(f"  [{chunk.index}] gap={chunk.gap_before_ms}ms ~{chunk.estimated_seconds:.1f}s")
        print(f"      {chunk.text}")
        if chunk.phonemes:
            print(f"      IPA: {chunk.phonemes}")
    return 0


def command_check(args: argparse.Namespace) -> int:
    from .audio import load_audio, preprocess_reference, save_audio

    config = _build_config(args)
    result = preprocess_reference(load_audio(args.audio), config.audio)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.summary())

    if args.save:
        path = save_audio(result.audio, args.save)
        print(f"\nProcessed reference written to {path}")
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    from .pipelines import Studio

    info = Studio(_build_config(args)).diagnostics()

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    print("=== System ===")
    for key, value in info["hardware"].items():
        print(f"  {key:14s} {value}")
    print(f"  {'compute':14s} {info['device']}")
    print(f"  {'free disk':14s} {info['free_disk_gb']} GB")

    print("\n=== Engines ===")
    for row in info["engines"]:
        mark = "OK " if row["available"] else "-- "
        detail = "" if row["available"] else f"  ({row['reason']})"
        print(f"  {mark}[{row['kind']}] {row['name']:8s} {row['display_name']}{detail}")

    print(f"\n=== Persian G2P ===\n  {info['g2p']}")

    for label, key in (
        ("Speech recognition", "asr"),
        ("Noise reduction", "denoise"),
        ("Similarity scoring", "similarity"),
    ):
        print(f"\n=== {label} ===")
        for name, available in info[key].items():
            print(f"  {'OK ' if available else '-- '}{name}")
    return 0


def command_voices(args: argparse.Namespace) -> int:
    from .audio import load_audio, preprocess_reference
    from .pipelines import VoiceLibrary

    config = _build_config(args)
    library = VoiceLibrary(config.paths.ensure("voices_dir"))

    if args.action == "list":
        profiles = library.list()
        if not profiles:
            print("No saved voices.")
            return 0
        for profile in profiles:
            print(f"  {profile.slug:24s} {profile.duration:6.1f}s  "
                  f"{profile.quality_verdict:10s} {profile.name}")
        return 0

    if args.action == "add":
        if not args.name or not args.reference:
            print("Usage: voiceclone voices add <name> --reference <audio>", file=sys.stderr)
            return 2
        processed = preprocess_reference(load_audio(args.reference), config.audio)
        profile = library.save(
            args.name,
            processed.audio,
            transcript=args.transcript or "",
            quality_score=processed.report_after.score,
            quality_verdict=processed.report_after.verdict,
            preprocessing=processed.steps,
        )
        print(f"Saved '{profile.name}' as '{profile.slug}' ({profile.duration:.1f}s)")
        return 0

    if args.action == "delete":
        if not args.name:
            print("Usage: voiceclone voices delete <slug>", file=sys.stderr)
            return 2
        print("Deleted." if library.delete(args.name) else f"No voice named '{args.name}'.")
        return 0

    return 2


COMMANDS = {
    "tts": command_tts,
    "s2s": command_s2s,
    "text": command_text,
    "check": command_check,
    "doctor": command_doctor,
    "voices": command_voices,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    handler = COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse enforces this
        return 2

    try:
        return handler(args)
    except VoiceCloneError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
