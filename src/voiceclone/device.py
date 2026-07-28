"""Device and dtype resolution across CUDA, Apple MPS and CPU."""

from __future__ import annotations

import gc
import os
import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from .logging_utils import get_logger

log = get_logger("device")


@dataclass(frozen=True)
class DeviceSpec:
    """Everything the engines need to know about where they run."""

    device: torch.device
    dtype: torch.dtype
    #: True when half-precision autocast is worth using.
    supports_autocast: bool

    @property
    def type(self) -> str:
        return self.device.type

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def is_mps(self) -> bool:
        return self.device.type == "mps"

    @property
    def is_cpu(self) -> bool:
        return self.device.type == "cpu"

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.device}/{str(self.dtype).replace('torch.', '')}"


def _mps_is_usable() -> bool:
    """MPS can be *built* but unusable; probe with a real allocation."""
    backend = getattr(torch.backends, "mps", None)
    if backend is None or not backend.is_available():
        return False
    try:
        probe = torch.zeros(8, device="mps")
        del probe
        empty_cache(torch.device("mps"))
        return True
    except Exception:  # pragma: no cover - depends on host
        log.warning("MPS reported available but failed a probe allocation; using CPU.")
        return False


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a compute device.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. An explicit preference is
    honoured when usable and downgraded with a warning when not.
    """
    preference = (preference or "auto").lower()

    if preference not in {"auto", "cuda", "mps", "cpu"}:
        log.warning("Unknown device %r; falling back to auto.", preference)
        preference = "auto"

    if preference == "cpu":
        return torch.device("cpu")

    if preference in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "cuda":
        log.warning("CUDA requested but unavailable; falling back.")

    if preference in {"auto", "mps"} and _mps_is_usable():
        return torch.device("mps")
    if preference == "mps":
        log.warning("MPS requested but unavailable; falling back to CPU.")

    return torch.device("cpu")


def resolve_dtype(device: torch.device, preference: str = "auto") -> torch.dtype:
    """Pick a compute dtype that the device actually handles well.

    CPU stays in float32: float16 matmuls are emulated and slower, and
    bfloat16 support is uneven. MPS keeps float32 for the numerically
    sensitive flow-matching / vocoder path; large LLM-style engines opt into
    half precision explicitly via their own config.
    """
    preference = (preference or "auto").lower()
    explicit = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    if preference in explicit:
        dtype = explicit[preference]
        if dtype is torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            log.warning("bfloat16 unsupported on this GPU; using float16.")
            return torch.float16
        return dtype

    if preference != "auto":
        log.warning("Unknown dtype %r; using auto.", preference)

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def get_device_spec(device: str = "auto", dtype: str = "auto", *, mps_fallback: bool = True) -> DeviceSpec:
    """Resolve device + dtype together and apply platform workarounds."""
    if mps_fallback:
        # Ops without an MPS kernel silently run on CPU instead of raising.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(resolved_device, dtype)
    supports_autocast = resolved_device.type == "cuda" and resolved_dtype != torch.float32

    return DeviceSpec(resolved_device, resolved_dtype, supports_autocast)


def empty_cache(device: torch.device | DeviceSpec | None = None) -> None:
    """Release cached accelerator memory. Safe to call unconditionally."""
    if isinstance(device, DeviceSpec):
        device = device.device

    gc.collect()
    try:
        if (device is None or device.type == "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if device is None or device.type == "mps":
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()
    except Exception:  # pragma: no cover - best effort
        log.debug("empty_cache failed", exc_info=True)


def seed_everything(seed: int | None) -> int:
    """Seed Python, NumPy and Torch. ``0``/``None`` draws a fresh seed.

    Returns the seed actually used so it can be reported and replayed.
    """
    if not seed:
        seed = random.randint(1, 2**31 - 1)
    seed = int(seed) % (2**31 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def configure_threads(num_threads: int = 0) -> None:
    """Cap PyTorch intra-op threads (0 keeps PyTorch's default)."""
    if num_threads and num_threads > 0:
        torch.set_num_threads(int(num_threads))
        log.debug("torch intra-op threads set to %d", num_threads)


@lru_cache(maxsize=1)
def describe_hardware() -> dict[str, str]:
    """Human-readable hardware summary for the diagnostics panel."""
    import platform

    info: dict[str, str] = {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "torch": torch.__version__,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["accelerator"] = f"CUDA - {props.name} ({props.total_memory / 1024**3:.1f} GB)"
    elif _mps_is_usable():
        info["accelerator"] = "Apple Metal (MPS)"
    else:
        info["accelerator"] = "CPU only"

    try:
        import psutil  # type: ignore

        info["ram"] = f"{psutil.virtual_memory().total / 1024**3:.1f} GB"
    except Exception:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            try:
                total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                info["ram"] = f"{total / 1024**3:.1f} GB"
            except (ValueError, OSError):
                pass

    return info
