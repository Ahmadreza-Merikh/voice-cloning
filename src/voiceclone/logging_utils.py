"""Logging setup and lightweight progress/timing helpers."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

_CONFIGURED = False
_LOGGER_NAME = "voiceclone"


def setup_logging(level: str | int = "INFO", *, force: bool = False) -> logging.Logger:
    """Configure the package logger once, preferring `rich` when available."""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)

    if _CONFIGURED and not force:
        return logger

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    env_level = os.environ.get("VOICECLONE_LOG_LEVEL")
    if env_level:
        level = getattr(logging, env_level.upper(), level)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler: logging.Handler
    try:
        from rich.logging import RichHandler

        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=False,
            log_time_format="%H:%M:%S",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    except Exception:  # pragma: no cover - rich is optional
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )

    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    # These libraries are chatty at INFO and drown out our own output.
    for noisy in ("numba", "matplotlib", "urllib3", "httpx", "filelock", "asyncio", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the package logger, configuring it on first use."""
    setup_logging()
    return logging.getLogger(_LOGGER_NAME if not name else f"{_LOGGER_NAME}.{name}")


@contextmanager
def timed(label: str, logger: logging.Logger | None = None, level: int = logging.INFO) -> Iterator[None]:
    """Log how long a block took.

    >>> with timed("synthesis"):
    ...     ...
    """
    log = logger or get_logger()
    start = time.perf_counter()
    try:
        yield
    finally:
        log.log(level, "%s finished in %.2fs", label, time.perf_counter() - start)


class ProgressReporter:
    """Adapter that forwards progress to a Gradio callback and to the log.

    Gradio hands synthesis functions a ``gr.Progress`` object; the CLI has no
    such thing. Wrapping both behind one interface keeps the pipelines free of
    UI conditionals.
    """

    def __init__(self, sink: Callable[[float, str], None] | None = None, logger: logging.Logger | None = None) -> None:
        self._sink = sink
        self._log = logger or get_logger("progress")

    def __call__(self, fraction: float, message: str = "") -> None:
        fraction = min(max(float(fraction), 0.0), 1.0)
        if message:
            self._log.info("[%3d%%] %s", round(fraction * 100), message)
        if self._sink is not None:
            try:
                self._sink(fraction, message)
            except Exception:  # pragma: no cover - never let UI break synthesis
                self._log.debug("progress sink raised", exc_info=True)

    @classmethod
    def from_gradio(cls, progress: object | None) -> ProgressReporter:
        """Build a reporter from a ``gr.Progress`` instance (or ``None``)."""
        if progress is None:
            return cls()

        def sink(fraction: float, message: str) -> None:
            progress(fraction, desc=message)  # type: ignore[operator]

        return cls(sink)
