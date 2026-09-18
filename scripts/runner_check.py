"""A model-free check of the runner, on the machine and display about to run the real job.

The T4 probe of 2026-09-18 stalled because the runner's reader waited on the
notebook display (eval/runs/persian-voice-moss15-probe-v2.json). The tests
reproduce that on a laptop with a pipe nobody reads; a laptop is not a Kaggle
notebook, so the notebook runs the same job first, on its own display, before
it downloads or loads anything:

* a fake worker that loads the real ``moss15_worker.py`` module - so the real
  ``_claim_stdout`` runs - prints what libraries print, through every route a
  library might use, then 3,000 lines and 20,000 carriage-return progress
  updates, about what the stalled codec load was writing, then its ready line
  through the real ``_emit``
* the probe's strict ready-then-quit driver, which fails on anything but the
  protocol's JSON on the worker's stdout

It passes only if the run exits 0 and the log holds every line and every
update. ``diagnostics`` separately checks that the group report can say, on
this machine, which descriptor a blocked process is waiting on.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

LINES, UPDATES = 3000, 20000

FAKE_WORKER = """\
import importlib.util, json, logging, os, sys
early = logging.StreamHandler(sys.stdout)
spec = importlib.util.spec_from_file_location("moss15_worker", sys.argv[1])
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
lines, updates = int(sys.argv[2]), int(sys.argv[3])
early.stream.write("early handler chatter\\n"); early.stream.flush()
print("print() chatter", flush=True)
os.write(1, b"raw fd 1 chatter\\n")
sys.__stdout__.write("original stdout object chatter\\n"); sys.__stdout__.flush()
for i in range(lines):
    sys.stderr.write(f"line {i} of {lines} " + "x" * 100 + "\\n")
for i in range(updates):
    sys.stderr.write(f"\\rLoading weights: {i}/{updates} [Materializing param=decoder.{i}.norm1.bias]")
    sys.stderr.flush()
sys.stderr.write("\\n")
worker._emit({"ready": True})
for line in sys.stdin:
    if json.loads(line).get("op") == "quit":
        break
"""

DRIVER = """\
import json, subprocess, sys
worker = subprocess.Popen([sys.executable] + sys.argv[1:], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True)
line = worker.stdout.readline()
print("WORKER SAID", line.strip(), flush=True)
reply = json.loads(line)
worker.stdin.write(json.dumps({"op": "quit"}) + "\\n")
worker.stdin.flush()
worker.wait(timeout=60)
sys.exit(0 if reply.get("ready") else 1)
"""

CHATTER = (b"early handler chatter", b"print() chatter", b"raw fd 1 chatter", b"original stdout object chatter")


def write_scripts(folder: Path) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "driver.py").write_text(DRIVER)
    (folder / "fake_worker.py").write_text(FAKE_WORKER)
    return folder / "driver.py", folder / "fake_worker.py"


def command(folder: Path, worker: Path, lines: int = LINES, updates: int = UPDATES) -> list[str]:
    driver, fake = write_scripts(folder)
    return [sys.executable, str(driver), str(fake), str(worker), str(lines), str(updates)]


def check(run_bounded: Callable[..., Any], folder: Path, worker: Path, timeout: float = 120,
          display: Any = None) -> tuple[bool, str, Any]:
    run = run_bounded(command(folder, worker), timeout=timeout, grace=5, log=folder / "runner_check.log",
                      display=display)
    if not run.ok:
        return False, f"the fake worker's run {run.outcome}", run
    kept = Path(run.log).read_bytes()
    if b'WORKER SAID {"ready": true}' not in kept:
        return False, "no ready line came through the protocol", run
    missing = [c.decode() for c in CHATTER if c not in kept]
    if missing:
        return False, f"missing from the log: {missing}", run
    if kept.count(b"\rLoading weights") != UPDATES or f"line {LINES - 1} of {LINES}".encode() not in kept:
        return False, "the log is missing output", run
    return True, (f"{run.output_bytes} bytes through the protocol path in {run.seconds:.1f}s; "
                  f"{run.shown_bytes} shown"), run


def diagnostics(report_group: Callable[..., list[str]]) -> tuple[bool, str]:
    """Whether the group report names the descriptor a blocked process waits on."""
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                             stdin=subprocess.PIPE, start_new_session=True)
    try:
        time.sleep(1.0)
        lines = report_group(child.pid)
    finally:
        child.kill()
        child.wait()
    mine = [line for line in lines if line.startswith(f"pid {child.pid}:")]
    if not mine:
        return False, "no per-process line (no /proc here)"
    return "read(fd 0 = pipe:" in mine[0], mine[0]
