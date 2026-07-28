"""Seed-VC engine - true acoustic voice conversion.

Where the cascade backend re-*speaks* the source text, Seed-VC re-*voices* the
source waveform: rhythm, emphasis, emotion and timing all survive, only the
timbre changes. It is acoustic and therefore language-agnostic, so Persian
works without any Persian-specific model.

Seed-VC pins ``transformers==4.46`` and ``numpy==1.26``, which cannot coexist
with the versions F5-TTS needs. Rather than force a downgrade on the whole
project, it is driven out-of-process in its own interpreter. Point
``engines.seedvc.python`` at that environment's binary.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..audio.io import Audio, load_audio, save_audio
from ..config import AppConfig
from ..device import DeviceSpec
from ..errors import EngineNotAvailableError, SynthesisError
from ..logging_utils import get_logger
from .base import EngineCapabilities, SynthesisResult, VCEngine, register_vc

log = get_logger("engines.seedvc")

#: Where to look for a Seed-VC checkout.
_REPO_CANDIDATES = ("~/seed-vc", "~/Seed-VC", "~/projects/seed-vc", "./third_party/seed-vc")


def find_seedvc_repo(configured: str | None = None) -> Path | None:
    """Locate a Seed-VC checkout containing ``inference.py``."""
    candidates: list[str] = []
    if configured and configured != "auto":
        candidates.append(configured)
    env = os.environ.get("VOICECLONE_SEEDVC_REPO")
    if env:
        candidates.append(env)
    candidates.extend(_REPO_CANDIDATES)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if (path / "inference.py").is_file():
            return path
    return None


def find_seedvc_python(repo: Path, configured: str | None = None) -> str | None:
    """Find an interpreter able to run Seed-VC."""
    if configured and configured != "auto":
        path = Path(configured).expanduser()
        return str(path) if path.exists() else None

    for relative in (".venv/bin/python", "venv/bin/python", ".venv/Scripts/python.exe"):
        candidate = repo / relative
        if candidate.exists():
            return str(candidate)

    # Last resort: this interpreter. Works only if it happens to satisfy
    # Seed-VC's pins, which the default install does not.
    return sys.executable


@register_vc
class SeedVCEngine(VCEngine):
    name = "seedvc"
    display_name = "Seed-VC (acoustic voice conversion)"
    capabilities = EngineCapabilities(
        accepts_phonemes=False,
        requires_reference_text=False,
        native_persian=True,  # language-agnostic: operates on acoustics
        supports_speed=True,
        supports_seed=False,
        approx_download_gb=1.0,
    )

    def __init__(self, config: AppConfig, device: DeviceSpec) -> None:
        super().__init__(config, device)
        self.repo: Path | None = None
        self.python: str | None = None

    # -- availability ------------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        repo = find_seedvc_repo()
        if repo is None:
            return False, (
                "Seed-VC checkout not found. "
                "git clone https://github.com/Plachtaa/seed-vc ~/seed-vc"
            )
        return True, ""

    def _load(self) -> None:
        settings = self.config.engines.seedvc
        self.repo = find_seedvc_repo(settings.repo_path)
        if self.repo is None:
            raise EngineNotAvailableError(
                "Seed-VC checkout not found.",
                hint="git clone https://github.com/Plachtaa/seed-vc ~/seed-vc",
            )
        self.python = find_seedvc_python(self.repo, settings.python)
        if not self.python:
            raise EngineNotAvailableError(
                "No interpreter configured for Seed-VC.",
                hint=(
                    f"Create one:  python3.10 -m venv {self.repo}/.venv && "
                    f"{self.repo}/.venv/bin/pip install -r {self.repo}/requirements.txt"
                ),
            )
        log.info("Seed-VC: %s (python: %s)", self.repo, self.python)

    # -- inference ---------------------------------------------------------
    def convert(
        self,
        source: Audio,
        reference: Audio,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SynthesisResult:
        self.load()
        assert self.repo is not None and self.python is not None

        settings = self.config.engines.seedvc

        with tempfile.TemporaryDirectory(prefix="voiceclone_seedvc_") as workdir:
            work = Path(workdir)
            source_path = save_audio(source.to_mono(), work / "source.wav", bit_depth=32)
            target_path = save_audio(reference.to_mono(), work / "target.wav", bit_depth=32)
            out_dir = work / "out"
            out_dir.mkdir()

            command = [
                self.python,
                str(self.repo / "inference.py"),
                "--source", str(source_path),
                "--target", str(target_path),
                "--output", str(out_dir),
                "--diffusion-steps", str(int(kwargs.get("diffusion_steps", settings.diffusion_steps))),
                "--length-adjust", str(float(kwargs.get("length_adjust", settings.length_adjust))),
                "--inference-cfg-rate", str(float(settings.inference_cfg_rate)),
                "--f0-condition", str(bool(kwargs.get("f0_condition", settings.f0_condition))),
                "--auto-f0-adjust", str(bool(settings.auto_f0_adjust)),
                "--semi-tone-shift", str(int(kwargs.get("semi_tone_shift", settings.semi_tone_shift))),
                # fp16 is a CUDA optimisation; it destabilises CPU/MPS runs.
                "--fp16", str(bool(self.device.is_cuda)),
            ]

            log.info("Running Seed-VC (%d diffusion steps) ...", settings.diffusion_steps)
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(self.repo),
                    capture_output=True,
                    text=True,
                    timeout=int(settings.timeout_seconds),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise SynthesisError(
                    f"Seed-VC timed out after {settings.timeout_seconds}s.",
                    hint="Shorten the source clip or lower engines.seedvc.diffusion_steps.",
                ) from exc
            except OSError as exc:
                raise SynthesisError(f"Could not start Seed-VC: {exc}") from exc

            if completed.returncode != 0:
                tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-12:]
                raise SynthesisError(
                    f"Seed-VC exited with code {completed.returncode}.",
                    hint=(
                        "Seed-VC needs its own environment (transformers 4.46, numpy 1.26).\n"
                        + "\n".join(f"    {line}" for line in tail)
                    ),
                )

            produced = sorted(out_dir.glob("*.wav"))
            if not produced:
                raise SynthesisError(
                    "Seed-VC finished but produced no audio.",
                    hint=(completed.stdout or "")[-500:],
                )

            converted = load_audio(produced[0], mono=True)

        return SynthesisResult(
            audio=converted,
            engine=self.name,
            seed=None,
            metadata={
                "diffusion_steps": settings.diffusion_steps,
                "inference_cfg_rate": settings.inference_cfg_rate,
                "source_seconds": round(source.duration, 2),
            },
        )


__all__ = ["SeedVCEngine", "find_seedvc_python", "find_seedvc_repo"]
