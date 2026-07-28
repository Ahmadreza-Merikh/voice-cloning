"""Model weight resolution, download and caching.

Checkpoints are referenced with a small URI scheme so config files stay
readable and portable:

    hf://<repo_id>/<path/inside/repo>    -> downloaded from the Hugging Face Hub
    /absolute/path or ./relative/path    -> used as-is

Nothing is downloaded at import time; resolution happens when an engine
actually loads.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .errors import DependencyMissingError, VoiceCloneError
from .logging_utils import get_logger

log = get_logger("models")

HF_PREFIX = "hf://"


@dataclass
class ResolvedModel:
    """A checkpoint that exists on disk."""

    path: Path
    source: str
    downloaded: bool = False

    @property
    def size_gb(self) -> float:
        try:
            return self.path.stat().st_size / 1e9
        except OSError:
            return 0.0


def parse_hf_uri(uri: str) -> tuple[str, str]:
    """``hf://owner/repo/some/file.pt`` -> ``('owner/repo', 'some/file.pt')``."""
    if not uri.startswith(HF_PREFIX):
        raise VoiceCloneError(f"Not a Hugging Face URI: {uri}")
    remainder = uri[len(HF_PREFIX) :].strip("/")
    parts = remainder.split("/")
    if len(parts) < 3:
        raise VoiceCloneError(
            f"Malformed Hugging Face URI: {uri}",
            hint="Expected hf://<owner>/<repo>/<file-path-inside-repo>",
        )
    return "/".join(parts[:2]), "/".join(parts[2:])


def is_cached(uri: str, cache_dir: str | Path | None = None) -> bool:
    """True when a URI can be satisfied without network access."""
    if not uri.startswith(HF_PREFIX):
        return Path(uri).expanduser().exists()
    try:
        repo_id, filename = parse_hf_uri(uri)
        from huggingface_hub import try_to_load_from_cache  # type: ignore

        result = try_to_load_from_cache(
            repo_id=repo_id, filename=filename, cache_dir=str(cache_dir) if cache_dir else None
        )
        return isinstance(result, str) and Path(result).is_file()
    except Exception:
        return False


def resolve_model(
    uri: str,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
    description: str = "model",
) -> ResolvedModel:
    """Turn a URI into a local file path, downloading if permitted."""
    if not uri:
        raise VoiceCloneError(f"No {description} specified.")

    if not uri.startswith(HF_PREFIX):
        path = Path(os.path.expandvars(uri)).expanduser()
        if not path.exists():
            raise VoiceCloneError(
                f"{description.capitalize()} not found: {path}",
                hint="Check the path in configs/default.yaml.",
            )
        return ResolvedModel(path, uri)

    repo_id, filename = parse_hf_uri(uri)

    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache  # type: ignore
    except ImportError as exc:  # pragma: no cover - core dependency
        raise DependencyMissingError("huggingface-hub", "model download") from exc

    cached = try_to_load_from_cache(
        repo_id=repo_id, filename=filename, cache_dir=str(cache_dir) if cache_dir else None
    )
    if isinstance(cached, str) and Path(cached).is_file():
        return ResolvedModel(Path(cached), uri)

    if not allow_download:
        raise VoiceCloneError(
            f"{description.capitalize()} is not cached and downloads are disabled: {uri}",
            hint="Run  python scripts/download_models.py  while online.",
        )

    log.info("Downloading %s from %s (%s) ...", description, repo_id, filename)
    try:
        path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=str(cache_dir) if cache_dir else None
        )
    except Exception as exc:
        raise VoiceCloneError(
            f"Could not download {description} from {uri}: {exc}",
            hint="Check your network connection, or set HF_ENDPOINT if the Hub is blocked.",
        ) from exc

    resolved = ResolvedModel(Path(path), uri, downloaded=True)
    log.info("Downloaded %s (%.2f GB) -> %s", description, resolved.size_gb, resolved.path)
    return resolved


# --------------------------------------------------------------------------
# F5-TTS checkpoint slimming
# --------------------------------------------------------------------------
def slim_f5_checkpoint(checkpoint: Path, output: Path | None = None) -> Path:
    """Strip optimizer state from an F5-TTS training checkpoint.

    Community finetunes are usually published as raw training checkpoints:
    a Persian F5-TTS ``model_last.pt`` is ~5.4 GB, of which only ~1.3 GB is
    the EMA weights that inference needs. Converting once saves the rest of
    that on every load, and lets the file be memory-mapped as safetensors.

    Returns the slim file, or the original when no saving is possible.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.suffix == ".safetensors":
        return checkpoint

    output = output or checkpoint.with_name(f"{checkpoint.stem}_ema.safetensors")
    if output.is_file() and output.stat().st_size > 0:
        log.debug("Slim checkpoint already present: %s", output)
        return output

    try:
        import torch
        from safetensors.torch import save_file  # type: ignore
    except ImportError:  # pragma: no cover - core dependency
        log.warning("safetensors/torch unavailable; using the full checkpoint.")
        return checkpoint

    log.info("Converting %s (%.2f GB) to EMA-only safetensors ...", checkpoint.name, checkpoint.stat().st_size / 1e9)
    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except Exception:
        # Older checkpoints may contain pickled objects that weights_only rejects.
        log.debug("weights_only load failed; retrying without it", exc_info=True)
        try:
            state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        except Exception:
            log.warning("Could not read %s for slimming; using it as-is.", checkpoint.name, exc_info=True)
            return checkpoint

    weights = None
    for key in ("ema_model_state_dict", "model_state_dict", "state_dict"):
        if isinstance(state, dict) and key in state:
            weights = state[key]
            break
    if weights is None:
        weights = state if isinstance(state, dict) else None
    if not isinstance(weights, dict):
        log.warning("Unexpected checkpoint layout in %s; using it as-is.", checkpoint.name)
        return checkpoint

    # The EMA wrapper prefixes tensors and stores bookkeeping entries that
    # `load_model` does not want.
    cleaned = {}
    for key, value in weights.items():
        if key in {"initted", "step"}:
            continue
        name = key[len("ema_model.") :] if key.startswith("ema_model.") else key
        if hasattr(value, "contiguous"):
            cleaned[name] = value.contiguous()

    if not cleaned:
        log.warning("No tensors extracted from %s; using it as-is.", checkpoint.name)
        return checkpoint

    tmp = output.with_suffix(".tmp")
    try:
        save_file(cleaned, str(tmp))
        tmp.replace(output)
    except Exception:
        log.warning("Could not write slim checkpoint; using the original.", exc_info=True)
        tmp.unlink(missing_ok=True)
        return checkpoint

    log.info(
        "Slim checkpoint written: %s (%.2f GB, was %.2f GB)",
        output.name,
        output.stat().st_size / 1e9,
        checkpoint.stat().st_size / 1e9,
    )
    return output


def cache_report(cache_dir: str | Path | None = None) -> dict[str, object]:
    """Summarise on-disk model cache usage for the Diagnostics tab."""
    try:
        from huggingface_hub import scan_cache_dir  # type: ignore

        info = scan_cache_dir(cache_dir)
        return {
            "size_gb": round(info.size_on_disk / 1e9, 2),
            "repos": sorted(repo.repo_id for repo in info.repos),
            "path": str(info.cache_dir),
        }
    except Exception:
        path = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "huggingface"
        total, _, free = shutil.disk_usage(path if path.exists() else Path.home())
        return {"size_gb": None, "repos": [], "path": str(path), "disk_free_gb": round(free / 1e9, 1)}


def free_disk_gb(path: str | Path | None = None) -> float:
    target = Path(path).expanduser() if path else Path.home()
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / 1e9


__all__ = [
    "HF_PREFIX",
    "ResolvedModel",
    "cache_report",
    "free_disk_gb",
    "is_cached",
    "parse_hf_uri",
    "resolve_model",
    "slim_f5_checkpoint",
]
