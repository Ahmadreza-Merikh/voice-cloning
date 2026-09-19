"""The Mac's own memory-pressure signal, for stopping work before the machine thrashes.

macOS publishes one system-wide pressure level - 1 normal, 2 warning, 4
critical - as the sysctl ``kern.memorystatus_vm_pressure_level``; it is the
signal behind Activity Monitor's pressure graph. It is read here through
``sysctlbyname``, which costs microseconds, so it can be polled while a model
loads. This laptop has 16 GB and no swap: on 2026-09-19 a guarded OmniVoice
load took it to warning with 5.1 GB already in the compressor.

Anywhere without that key - Linux, a Kaggle GPU box - every reading is None
and nothing is ever refused or stopped.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
import threading
from collections.abc import Callable

#: The level at which work is refused or stopped.
WARNING = 2

_KEY = b"kern.memorystatus_vm_pressure_level"
_libc: ctypes.CDLL | None = None


def pressure_level() -> int | None:
    """1 normal, 2 warning, 4 critical; None where the system does not publish it."""
    global _libc
    if sys.platform != "darwin":
        return None
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if _libc.sysctlbyname(_KEY, ctypes.byref(value), ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
        return None
    return int(value.value)


def available_gb() -> float | None:
    """Memory the system can hand out without pressure, where psutil can say."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.virtual_memory().available / 2**30


def describe(level: int | None) -> str:
    return {1: "normal", 2: "warning", 4: "critical"}.get(level or 0, f"level {level}")


class PressureWatch:
    """Poll the pressure level on a thread and call ``on_warning`` once if it reaches WARNING.

    Used as a context manager around the stretch of work it guards. Where no
    level is published it starts no thread at all.
    """

    def __init__(self, on_warning: Callable[[int], None], interval: float = 0.25,
                 read: Callable[[], int | None] | None = None) -> None:
        self._on_warning = on_warning
        self._interval = interval
        self._read = read or pressure_level
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        #: The level that set it off, if it went off.
        self.triggered: int | None = None

    def _watch(self) -> None:
        while not self._stop.is_set():
            level = self._read()
            if level is not None and level >= WARNING:
                self.triggered = level
                self._on_warning(level)
                return
            self._stop.wait(self._interval)

    def __enter__(self) -> PressureWatch:
        if self._read() is not None:
            self._thread = threading.Thread(target=self._watch, name="memory-pressure-watch", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


__all__ = ["WARNING", "PressureWatch", "available_gb", "describe", "pressure_level"]
