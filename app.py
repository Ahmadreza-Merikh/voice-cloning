#!/usr/bin/env python3
"""Launch the Persian Voice Cloning Studio web interface.

    python app.py
    python app.py --port 7861 --share
    python app.py --config configs/my-setup.yaml --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support running straight from a checkout, without `pip install -e .`.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from voiceclone.config import AppConfig  # noqa: E402
from voiceclone.logging_utils import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persian Voice Cloning Studio - Gradio interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    parser.add_argument("--host", type=str, default=None, help="Bind address")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument(
        "--device", type=str, default=None, choices=["auto", "cuda", "mps", "cpu"], help="Compute device"
    )
    parser.add_argument(
        "--language", type=str, default=None, choices=["fa", "en"], help="Interface language"
    )
    parser.add_argument("--engine", type=str, default=None, help="Default TTS engine (f5 / moss)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging verbosity")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    overrides: dict[str, object] = {}
    if args.host:
        overrides["ui.server_name"] = args.host
    if args.port:
        overrides["ui.server_port"] = args.port
    if args.share:
        overrides["ui.share"] = True
    if args.device:
        overrides["runtime.device"] = args.device
    if args.language:
        overrides["ui.language"] = args.language
    if args.engine:
        overrides["engines.tts"] = args.engine

    config = AppConfig.load(args.config)
    if overrides:
        config = config.copy_with(**overrides)

    from voiceclone.ui import VoiceCloneUI

    print(
        f"\n  Persian Voice Cloning Studio\n"
        f"  http://{config.ui.server_name}:{config.ui.server_port}\n"
    )
    VoiceCloneUI(config).launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
