"""Stage gates and a bounded runner for the GPU notebooks.

A notebook cell runs whether or not the cell before it worked, and a
subprocess killed on timeout can leave its own children holding a GPU. Both
happened in review rather than in production, and both are fixed here, in a
module the notebooks import and the tests can exercise without a GPU:

* ``Gates`` records each stage's verdict with a reason, and a stage asks it
  before starting. A stage whose dependencies failed, timed out or never ran is
  skipped - and the skip is recorded as a failure of that stage, so everything
  downstream of it is skipped too
* each ``*_verdict`` function is the definition of "that stage worked": not
  "the command exited" but the output the next stage relies on, checked
* ``run_bounded`` starts a command in its own process group and, on timeout,
  terminates and then kills that group and waits for it - its own processes and
  nothing else

Passing every gate means the path works. It says nothing about quality.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Gates:
    status: dict[str, tuple[bool, str]] = field(default_factory=dict)

    def record(self, stage: str, passed: bool, reason: str) -> None:
        self.status[stage] = (bool(passed), str(reason))
        print(f"GATE {stage}: {'PASS' if passed else 'FAIL'} - {reason}", flush=True)

    def allowed(self, needs: tuple[str, ...]) -> tuple[bool, str]:
        for stage in needs:
            if stage not in self.status:
                return False, f"{stage} never ran"
            passed, reason = self.status[stage]
            if not passed:
                return False, f"{stage} failed: {reason}"
        return True, "all prerequisites passed"

    def skip(self, stage: str, needs: tuple[str, ...]) -> bool:
        """Record and announce a skip. Returns True when the stage must not run."""
        ok, why = self.allowed(needs)
        if not ok:
            self.record(stage, False, f"skipped - {why}")
        return not ok

    def summary(self) -> dict[str, Any]:
        return {stage: {"passed": p, "reason": r} for stage, (p, r) in self.status.items()}


def run_bounded(command: list[str], timeout: float | None = None, cwd: str | None = None,
                grace: float = 30.0) -> tuple[int, float, bool]:
    """Run a command in its own process group; on timeout, end the whole group.

    Returns (exit code, seconds, timed_out). A timed-out command reports -1.
    """
    print("$", " ".join(str(part) for part in command), flush=True)
    start = time.time()
    process = subprocess.Popen(command, cwd=cwd, text=True, start_new_session=True)
    timed_out = False
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _end_group(process, grace)
        code = -1
    elapsed = time.time() - start
    state = f"TIMED OUT after {timeout}s - process group ended" if timed_out else f"exit {code}"
    print(f"{state} in {elapsed / 60:.1f} min", flush=True)
    return code, elapsed, timed_out


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _end_group(process: subprocess.Popen, grace: float) -> None:
    """TERM the group, give it ``grace`` seconds, then KILL what is left; reap the leader.

    start_new_session makes the child the leader of a group containing only it
    and whatever it spawned, so this reaches the worker a driver started and
    nothing outside the job.
    """
    group = process.pid
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            break
        deadline = time.time() + grace
        while time.time() < deadline:
            if process.poll() is not None and not _group_alive(group):
                return
            time.sleep(0.1)
    process.wait()


def worker_verdict(code: int, timed_out: bool, render: Path, smoke: Path,
                   expect_rate: int, expect_channels: int) -> tuple[bool, str, dict[str, Any]]:
    """The worker stage worked if it produced audio of the documented format."""
    import numpy as np
    import soundfile as sf

    if timed_out:
        return False, "timed out", {}
    if code != 0:
        return False, f"driver exited {code}", {}
    if not render.is_file():
        return False, "no render written", {}
    facts = json.loads(smoke.read_text()) if smoke.is_file() else {}
    samples, rate = sf.read(render, always_2d=True, dtype="float32")
    flat = samples.reshape(-1)
    if rate != expect_rate or samples.shape[1] != expect_channels:
        return False, f"{rate} Hz x {samples.shape[1]} channels, expected {expect_rate} x {expect_channels}", facts
    if not np.isfinite(flat).all():
        return False, "non-finite samples", facts
    if np.unique(flat).size < 1000:
        return False, f"only {np.unique(flat).size} distinct values", facts
    if not facts.get("result", {}).get("ok"):
        return False, "the smoke record does not show a successful render", facts
    return True, f"{samples.shape[0] / rate:.2f} s of {rate} Hz x {samples.shape[1]} audio", facts


def bench_verdict(code: int, timed_out: bool, results: Path) -> tuple[bool, str]:
    """The scored smoke cell worked if it rendered, was scored and was transcribed."""
    if timed_out:
        return False, "timed out"
    if code != 0:
        return False, f"benchmark exited {code}"
    if not results.is_file():
        return False, "no results file"
    data = json.loads(results.read_text())
    summary = data.get("summary", {})
    rows = data.get("rows") or []
    if summary.get("renders") != 1 or summary.get("failures"):
        return False, f"renders {summary.get('renders')}, failures {summary.get('failures')}"
    if summary.get("untranscribed"):
        return False, "the render was never transcribed, so there is no CER"
    row = rows[0] if rows else {}
    held, cer = row.get("held_out_similarity"), row.get("character_error_rate")
    if held is None or not math.isfinite(held):
        return False, "no finite held-out similarity"
    if cer is None or not math.isfinite(cer):
        return False, "no finite CER"
    return True, f"held-out {held:.4f}, CER {cer:.3f}"


def control_verdict(code: int, timed_out: bool, check: dict[str, Any] | None) -> tuple[bool, str]:
    """The scale control worked if every baseline cell re-scored identically at 4 dp."""
    if timed_out:
        return False, "timed out"
    if code != 0 or check is None:
        return False, f"control exited {code}"
    if check["only_reference"] or check["only_control"] or check["unusable"]:
        return False, (f"keys differ: {len(check['only_reference'])} only in baseline, "
                       f"{len(check['only_control'])} only here, {check['unusable']} unusable")
    if check["unequal"]:
        return False, f"{len(check['unequal'])} of {check['shared']} cells differ (max {check['max_abs']:.4f})"
    return True, f"all {check['shared']} cells equal at 4 dp"


def budget_verdict(load_seconds: float, cell_seconds: float, cells: int, timeout: float,
                   peaks_gib: dict[str, float], capacity_gib: dict[str, float],
                   time_share: float = 0.8, memory_share: float = 0.95) -> tuple[bool, str]:
    """The full run is allowed only if its measured cost fits, with margin."""
    estimate = load_seconds + cells * cell_seconds
    if estimate > time_share * timeout:
        return False, (f"estimated {estimate / 60:.0f} min for {cells} cells exceeds "
                       f"{time_share:.0%} of the {timeout / 60:.0f} min budget")
    for card, peak in peaks_gib.items():
        capacity = capacity_gib.get(card)
        if capacity is None:
            return False, f"no capacity known for {card}"
        if peak > memory_share * capacity:
            return False, f"{card} peaked at {peak:.1f} GiB of {capacity:.1f}"
    return True, (f"estimated {estimate / 60:.0f} min; peaks "
                  + ", ".join(f"{c} {p:.1f}/{capacity_gib[c]:.1f} GiB" for c, p in sorted(peaks_gib.items())))
