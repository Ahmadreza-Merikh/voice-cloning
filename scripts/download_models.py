#!/usr/bin/env python3
"""Pre-download model weights so the first synthesis is not a surprise.

    python scripts/download_models.py                 # default engine + ASR
    python scripts/download_models.py --engine moss    # add MOSS-TTS (6.2 GB)
    python scripts/download_models.py --all            # everything
    python scripts/download_models.py --list           # sizes, no download

Downloads land in ``paths.cache_dir`` (``~/.cache/persian-voice-cloning`` by
default), so removing that directory reclaims every byte.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from voiceclone.config import AppConfig  # noqa: E402
from voiceclone.errors import VoiceCloneError  # noqa: E402
from voiceclone.logging_utils import get_logger, setup_logging  # noqa: E402
from voiceclone.models import (  # noqa: E402
    free_disk_gb,
    is_cached,
    resolve_model,
    slim_f5_checkpoint,
)

log = get_logger("download")


def _f5_targets(config: AppConfig) -> list[tuple[str, str, float]]:
    settings = config.engines.f5
    return [
        (settings.vocab, "F5-TTS Persian vocabulary", 0.01),
        (settings.checkpoint, "F5-TTS Persian checkpoint", 5.4),
    ]


def download_f5(config: AppConfig, *, slim: bool = True) -> None:
    cache_dir = config.paths.ensure("cache_dir")
    checkpoint_path: Path | None = None

    for uri, description, _size in _f5_targets(config):
        resolved = resolve_model(uri, cache_dir=cache_dir, description=description)
        print(f"  OK  {description}: {resolved.path} ({resolved.size_gb:.2f} GB)")
        if "checkpoint" in description:
            checkpoint_path = resolved.path

    if slim and checkpoint_path is not None:
        print("  ..  Converting the training checkpoint to EMA-only safetensors")
        slim_path = slim_f5_checkpoint(checkpoint_path)
        if slim_path != checkpoint_path:
            print(f"  OK  Slim checkpoint: {slim_path} ({slim_path.stat().st_size / 1e9:.2f} GB)")
            print("      The original .pt can be deleted if you want the space back.")


def download_moss(config: AppConfig) -> None:
    """Fetch the MOSS-TTS repository snapshot (weights + custom code)."""
    from huggingface_hub import snapshot_download  # type: ignore

    model_id = config.engines.moss.model_id
    print(f"  ..  Downloading {model_id} (~6.2 GB)")
    path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(config.paths.ensure("cache_dir")),
    )
    print(f"  OK  MOSS-TTS: {path}")


def download_asr(config: AppConfig) -> None:
    settings = config.engines.asr
    try:
        from faster_whisper import WhisperModel  # type: ignore

        print(f"  ..  Downloading faster-whisper '{settings.model}'")
        WhisperModel(
            settings.model,
            device="cpu",
            compute_type="int8",
            download_root=str(config.paths.ensure("cache_dir") / "whisper"),
        )
        print(f"  OK  ASR: faster-whisper/{settings.model}")
    except ImportError:
        print("  --  faster-whisper is not installed; skipping ASR weights.")
    except Exception as exc:
        print(f"  !!  Could not fetch ASR weights: {exc}")


def download_vocoder() -> None:
    """F5-TTS decodes mel spectrograms with Vocos."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        print("  ..  Downloading the Vocos vocoder")
        path = snapshot_download(repo_id="charactr/vocos-mel-24khz")
        print(f"  OK  Vocoder: {path}")
    except Exception as exc:
        print(f"  !!  Could not fetch the vocoder: {exc}")


def download_similarity() -> None:
    try:
        import importlib.util

        if importlib.util.find_spec("speechbrain") is None:
            print("  --  speechbrain is not installed; similarity scoring will be approximate.")
            return
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

        print("  ..  Downloading the ECAPA-TDNN speaker encoder")
        EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=None)
        print("  OK  Speaker similarity: ECAPA-TDNN")
    except Exception as exc:
        print(f"  !!  Could not fetch the speaker encoder: {exc}")


def show_status(config: AppConfig) -> None:
    cache_dir = config.paths.resolved("cache_dir")
    print("Model status\n")
    print(f"  cache directory : {cache_dir}")
    print(f"  free disk       : {free_disk_gb(cache_dir):.1f} GB\n")

    print(f"  {'component':34s} {'size':>8s}  cached")
    print(f"  {'-' * 34} {'-' * 8}  ------")
    for uri, description, size in _f5_targets(config):
        cached = is_cached(uri, cache_dir)
        print(f"  {description:34s} {size:7.2f}G  {'yes' if cached else 'no'}")
    print(f"  {'MOSS-TTS (' + config.engines.moss.model_id.split('/')[-1] + ')':34s} {6.2:7.2f}G  see HF cache")
    print(f"  {'faster-whisper ' + config.engines.asr.model:34s} {1.6:7.2f}G  see HF cache")
    print(f"  {'Vocos vocoder':34s} {0.08:7.2f}G  see HF cache")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    parser.add_argument(
        "--engine",
        action="append",
        choices=["f5", "moss"],
        help="Engine weights to fetch (repeatable). Defaults to the configured engine.",
    )
    parser.add_argument("--asr", action="store_true", help="Also fetch speech-recognition weights")
    parser.add_argument("--similarity", action="store_true", help="Also fetch the speaker encoder")
    parser.add_argument("--all", action="store_true", help="Fetch everything")
    parser.add_argument("--list", action="store_true", help="Show status and exit")
    parser.add_argument("--no-slim", action="store_true", help="Keep the full F5 training checkpoint")
    args = parser.parse_args(argv)

    setup_logging("INFO")
    config = AppConfig.load(args.config)

    if args.list:
        show_status(config)
        return 0

    engines = set(args.engine or [])
    if args.all:
        engines |= {"f5", "moss"}
    if not engines:
        engines = {config.engines.tts}

    free = free_disk_gb(config.paths.resolved("cache_dir"))
    needed = (5.5 if "f5" in engines else 0) + (6.3 if "moss" in engines else 0)
    print(f"Downloading: {', '.join(sorted(engines))}")
    print(f"Estimated {needed:.1f} GB required, {free:.1f} GB free.\n")
    if needed > free:
        print("Not enough free disk space. Free some room and try again.", file=sys.stderr)
        return 1

    try:
        if "f5" in engines:
            print("F5-TTS")
            download_f5(config, slim=not args.no_slim)
            download_vocoder()
        if "moss" in engines:
            print("\nMOSS-TTS")
            download_moss(config)
        if args.asr or args.all:
            print("\nSpeech recognition")
            download_asr(config)
        if args.similarity or args.all:
            print("\nEvaluation")
            download_similarity()
    except VoiceCloneError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run to resume - downloads are cached.", file=sys.stderr)
        return 130

    print("\nDone. Launch the studio with:  python app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
