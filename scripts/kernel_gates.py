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
* ``run_bounded`` runs a command in its own process group, keeps everything it
  prints in a log file, shows a bounded view of that without ever letting the
  view hold the command up, and on timeout ends the group - its own processes
  and nothing else - and reports whether that worked

Passing every gate means the path works. It says nothing about quality.
"""

from __future__ import annotations

import codecs
import collections
import contextlib
import contextvars
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

#: Bytes of a command's own output shown in the notebook per run. The log
#: file keeps all of it.
DISPLAY_LIMIT = 100_000
#: A progress bar redraws itself thousands of times; the view shows its newest
#: state at most this often, and its last state when it ends.
PROGRESS_EVERY = 15.0
#: Once the limit is reached, the last lines are still worth seeing - an error,
#: a result - so this many are kept and shown when the command ends.
TAIL_LINES = 40
#: Chunks waiting for the display. When the display falls this far behind, new
#: chunks are not shown - they are still in the log file.
DISPLAY_BACKLOG = 256
#: After the process group has ended, how long to wait for the last output and
#: for the display to catch up. Neither wait can hold up a result.
PUMP_WAIT = 10.0
DISPLAY_WAIT = 5.0


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


def describe_exit(code: int | None) -> str:
    if code is None:
        return "no exit status"
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    return f"exited {code}"


@dataclass
class Run:
    """What happened to one command. A timeout is not an exit code, so it has none."""

    command: list[str]
    exit_code: int | None
    timed_out: bool
    seconds: float
    group: str
    log: Path
    timeout: float | None = None
    output_bytes: int = 0
    shown_bytes: int = 0
    display_blocked: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    @property
    def outcome(self) -> str:
        if self.timed_out:
            return f"timed out after {self.timeout:.0f}s"
        return describe_exit(self.exit_code)

    def summary(self, with_view: bool = True) -> str:
        view = ""
        if with_view:
            view = f"{self.shown_bytes} of {self.output_bytes} output bytes shown; "
            if self.display_blocked:
                view = view[:-2] + ", the display stalled; "
        extra = "".join(f"; {problem}" for problem in self.problems)
        return (f"{self.outcome} in {self.seconds / 60:.1f} min; process group {self.group}; "
                f"{view}full log {self.log}{extra}")

    def record(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "exit_code": self.exit_code, "timed_out": self.timed_out,
                "timeout": self.timeout, "seconds": round(self.seconds, 1), "process_group": self.group,
                "log": str(self.log),
                "output_bytes": self.output_bytes, "shown_bytes": self.shown_bytes,
                "display_blocked": self.display_blocked, "problems": self.problems}


_RECORD = re.compile(r"([^\r\n]*)(\r\n|\n|\r)")


class _Display:
    """A bounded view of a run's output, drawn by its own thread.

    ``offer`` and ``note`` never wait: if this thread is stuck inside a write,
    what it has not taken is simply not shown. Nothing that must happen - the
    log, the timeout, the result - goes through here.
    """

    def __init__(self, stream: TextIO, limit: int, progress_every: float) -> None:
        self.stream, self.limit, self.progress_every = stream, limit, progress_every
        self.shown = 0
        self.last_line = ""
        self.native_id: int | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=DISPLAY_BACKLOG)
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._partial = ""
        self._progress: str | None = None
        self._progress_at = -math.inf
        self._limit_hit = False
        self._tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
        self._hidden = 0
        self._broken = False
        self._stop = threading.Event()
        # A notebook routes a thread's output by the context it was started in.
        context = contextvars.copy_context()
        self.thread = threading.Thread(target=context.run, args=(self._loop,), daemon=True)
        self.thread.start()

    def offer(self, chunk: bytes) -> None:
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(("output", chunk))

    def note(self, text: str) -> None:
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(("note", text))

    def close(self, wait: float) -> bool:
        """Ask the thread to finish; True if it did within ``wait`` seconds."""
        self._stop.set()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(("stop", None))
        self.thread.join(timeout=wait)
        return not self.thread.is_alive()

    def _loop(self) -> None:
        self.native_id = threading.get_native_id()
        while True:
            try:
                kind, item = self._queue.get(timeout=1.0)
            except queue.Empty:
                if self._stop.is_set():
                    break
                self._flush_progress(force=False)
                continue
            if kind == "stop":
                break
            if kind == "note":
                self._write(item.rstrip("\n") + "\n")
            else:
                self._take(self._decoder.decode(item))
            self._flush_progress(force=False)
        self._take(self._decoder.decode(b"", final=True))
        if self._partial.strip():
            self._line(self._partial)
        self._flush_progress(force=True)
        if self._tail:
            hidden = self._hidden - sum(len(line.encode("utf-8", "replace")) for line in self._tail)
            self._write(f"[... {hidden} bytes not shown; the last {len(self._tail)} lines:]\n")
            for line in self._tail:
                self.shown += len(line.encode("utf-8", "replace"))
                self._write(line)

    def _take(self, text: str) -> None:
        text = self._partial + text
        cut = max(text.rfind("\n"), text.rfind("\r"))
        if cut < 0:
            self._partial = text[-65536:]
            return
        complete, self._partial = text[:cut + 1], text[cut + 1:]
        for match in _RECORD.finditer(complete):
            body, end = match.groups()
            if end == "\r":
                if body.strip():
                    self._progress = body
                    self.last_line = body
                continue
            self._line(body)

    def _line(self, body: str) -> None:
        self._progress = None
        if body.strip():
            self.last_line = body
        self._show(body + "\n")

    def _flush_progress(self, force: bool) -> None:
        if self._progress is None:
            return
        now = time.monotonic()
        if force or now - self._progress_at >= self.progress_every:
            text, self._progress, self._progress_at = self._progress, None, now
            self._show(text + "\n")

    def _show(self, text: str) -> None:
        size = len(text.encode("utf-8", "replace"))
        if self._limit_hit or self.shown + size > self.limit:
            if not self._limit_hit:
                self._limit_hit = True
                self._write(f"[display limit of {self.limit} bytes reached - the rest is in the log "
                            f"file; the last {TAIL_LINES} lines follow when the command ends]\n")
            self._hidden += size
            self._tail.append(text[:500] + ("\n" if len(text) > 500 else ""))
            return
        self.shown += size
        self._write(text)

    def _write(self, text: str) -> None:
        if self._broken:
            return
        try:
            self.stream.write(text)
            self.stream.flush()
        except (OSError, ValueError):
            self._broken = True


def _write_all(sink: Any, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = sink.write(view)
        view = view[written or 0:]


def run_bounded(command: list[str], timeout: float | None = None, cwd: str | None = None,
                grace: float = 30.0, monitor_every: float | None = None, log: Path | None = None,
                display: TextIO | None = None, display_limit: int = DISPLAY_LIMIT,
                progress_every: float = PROGRESS_EVERY) -> Run:
    """Run a command in its own process group; keep its output; on timeout, end the group.

    The command's stdout and stderr go to one pipe. A reader thread empties it
    continuously into ``log`` - unbuffered, so the file is complete whatever
    happens next - and offers each chunk to the display. That order is the
    fix for what the T4 probe of 2026-09-18 measured
    (eval/runs/persian-voice-moss15-probe-v2.json): the old reader wrote every
    chunk to the notebook's raw stdout before reading the next. That write
    stopped returning after 123 KB, so the reader stopped reading, the pipe
    filled, and the worker sat in pipe_write for the whole 20-minute budget
    while its driver waited in pipe_read for a line that was never written.
    The old log file held one chunk - 64 KB - more than the notebook ever showed.

    Now nothing that must happen waits for the display. The view is drawn by
    its own thread from a bounded queue that drops instead of waiting; progress
    bars are collapsed to one update every ``progress_every`` seconds and the
    command's output stops being shown after ``display_limit`` bytes. The
    waiting thread never writes to the display either: its monitor reports go
    to the log and the queue. A stalled display can cost the view, never the
    timeout or the result.

    ``monitor_every`` records, at that interval, each group member's state, the
    kernel function it is waiting in and - on Linux - which descriptor it is
    reading or writing and what that descriptor is, plus this runner's own
    reader and display threads, so the next silent process says which pipe it
    is stuck on.

    After the command exits, anything it left running in its group is ended
    too. The returned ``Run`` separates a timeout from an exit code, and its
    ``group`` says what actually became of the group - never "ended" when a
    member survived.
    """
    log = Path(log) if log is not None else Path(tempfile.mkstemp(prefix="run-", suffix=".log")[1])
    log.parent.mkdir(parents=True, exist_ok=True)
    # Unbuffered and closed by hand: the reader thread writes to it and may, in
    # the worst case, outlive this function.
    sink = open(log, "ab", buffering=0)  # noqa: SIM115
    lock = threading.Lock()
    view = _Display(display if display is not None else sys.stdout, display_limit, progress_every)
    problems: list[str] = []
    log_failed: list[bool] = []

    def keep(data: bytes) -> None:
        try:
            with lock:
                _write_all(sink, data)
        except (OSError, ValueError) as exc:
            if not log_failed:
                log_failed.append(True)
                problems.append(f"the log file stopped accepting writes: {exc}")

    def note(text: str) -> None:
        stamped = f"[runner {time.strftime('%H:%M:%S')}] {text}"
        keep(("\n" + stamped + "\n").encode("utf-8", "replace"))
        view.note(stamped)

    command = [str(part) for part in command]
    note("$ " + " ".join(command))
    start = time.time()
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = {"bytes": 0, "reader": None}

    def pump() -> None:
        output["reader"] = threading.get_native_id()
        source = process.stdout.fileno()
        while True:
            try:
                chunk = os.read(source, 65536)
            except OSError as exc:
                problems.append(f"reading the command's output failed: {exc}")
                return
            if not chunk:
                return
            keep(chunk)
            output["bytes"] += len(chunk)
            view.offer(chunk)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    def monitor() -> None:
        threads = {"reader": output["reader"], "display": view.native_id}
        for line in report_group(process.pid, threads, view.last_line):
            note(line)

    timed_out = False
    code: int | None = None
    deadline = None if timeout is None else start + timeout
    next_report = None if monitor_every is None else start + monitor_every
    while True:
        step = 5.0
        for moment in (deadline, next_report):
            if moment is not None:
                step = min(step, max(0.0, moment - time.time()))
        try:
            code = process.wait(timeout=step)
            break
        except subprocess.TimeoutExpired:
            pass
        if deadline is not None and time.time() >= deadline:
            timed_out = True
            if monitor_every is not None:
                monitor()
            ended, detail = _end_group(process, grace)
            group = f"ended by this runner after the timeout ({detail})" if ended else f"NOT ended: {detail}"
            break
        if next_report is not None and time.time() >= next_report:
            monitor()
            next_report += monitor_every

    if not timed_out:
        if _group_alive(process.pid):
            ended, detail = _end_group(process, grace)
            group = (f"ended: members outlived the command and this runner stopped them ({detail})" if ended
                     else f"NOT ended: members outlived the command; {detail}")
        else:
            group = "ended with the command"

    reader.join(timeout=PUMP_WAIT)
    if reader.is_alive():
        problems.append("the output pipe is still open after the group ended - "
                        "a process outside the group holds it")
    run = Run(command=command, exit_code=None if timed_out else code, timed_out=timed_out,
              seconds=time.time() - start, group=group, log=log, timeout=timeout,
              output_bytes=output["bytes"], problems=problems)
    note(run.summary(with_view=False))
    run.display_blocked = not view.close(DISPLAY_WAIT)
    run.shown_bytes = view.shown
    keep(f"\n[runner] {run.summary()}\n".encode("utf-8", "replace"))
    sink.close()
    return run


#: x86_64 numbers for the calls a stuck process is usually in.
_SYSCALLS = {0: "read", 1: "write", 7: "poll", 23: "select", 35: "nanosleep", 61: "wait4",
             202: "futex", 230: "clock_nanosleep", 232: "epoll_wait", 270: "pselect6", 271: "ppoll",
             281: "epoll_pwait"}


def _fd_target(pid: int | str, fd: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/fd/{fd}")
    except OSError:
        return "?"


def _where(task: str, pid: int | str) -> str:
    """A Linux task's wait channel and system call - for read or write, on which descriptor."""
    try:
        wchan = Path(task, "wchan").read_text().strip() or "-"
        fields = Path(task, "syscall").read_text().split()
    except OSError:
        return "not readable"
    if not fields or not fields[0].isdigit():
        return f"wchan {wchan}, {' '.join(fields[:1]) or '?'}"
    number = int(fields[0])
    call = _SYSCALLS.get(number, f"syscall {number}")
    if number in (0, 1) and len(fields) > 1:
        fd = int(fields[1], 16)
        call += f"(fd {fd} = {_fd_target(pid, fd)})"
    return f"wchan {wchan}, {call}"


def _descriptors(pid: int) -> str:
    try:
        numbers = sorted(int(name) for name in os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        return "?"
    shown = []
    for fd in numbers:
        target = _fd_target(pid, fd)
        if fd <= 2 or target.startswith(("pipe:", "socket:")):
            shown.append(f"{fd}={target}")
    return " ".join(shown[:12])


def _members(group: int) -> list[tuple[int, str]] | None:
    """(pid, state) of each process in the group, or None if ``ps`` cannot say."""
    try:
        listing = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,stat="], capture_output=True, text=True)
    except OSError:
        return None
    if listing.returncode != 0:
        return None
    members = []
    for line in listing.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == str(group):
            members.append((int(parts[0]), parts[2]))
    return members


def report_group(group: int, threads: dict[str, int | None] | None = None,
                 last_output: str | None = None) -> list[str]:
    """Each member's state and wait channel, where it is blocked, and free memory.

    Linux gives all of it; elsewhere this is best effort.
    """
    lines = [f"--- group {group} at {time.strftime('%H:%M:%S')}"]
    members = [pid for pid, state in (_members(group) or []) if not state.startswith("Z")]
    if members:
        columns = "pid,stat,wchan:32,rss,etime,args" if sys.platform.startswith("linux") else "pid,stat,wchan,rss,etime,args"
        listing = subprocess.run(["ps", "-o", columns, "-p", ",".join(map(str, members))],
                                 capture_output=True, text=True)
        lines += [line[:200] for line in (listing.stdout or listing.stderr).splitlines()]
        for pid in members:
            if Path(f"/proc/{pid}").is_dir():
                lines.append(f"pid {pid}: {_where(f'/proc/{pid}', pid)}; descriptors {_descriptors(pid)}")
    else:
        lines.append("no live members")
    for name, tid in (threads or {}).items():
        if tid is not None and Path(f"/proc/self/task/{tid}").is_dir():
            lines.append(f"runner {name} thread: {_where(f'/proc/self/task/{tid}', 'self')}")
    if last_output:
        lines.append(f"last output: {last_output[-200:]}")
    if shutil.which("free"):
        memory = subprocess.run(["free", "-m"], capture_output=True, text=True)
        lines += memory.stdout.splitlines()
    return lines


def _group_alive(group: int) -> bool:
    members = _members(group)
    if members is not None:
        return any(not state.startswith("Z") for _, state in members)
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _end_group(process: subprocess.Popen, grace: float) -> tuple[bool, str]:
    """TERM the group, give it ``grace`` seconds, KILL what is left; never wait without a limit.

    Returns whether the whole group is gone, and how it went.

    start_new_session makes the command the leader of a group containing only
    it and whatever it spawned, so this reaches the worker a driver started and
    nothing outside the job. Zombies - dead, waiting to be reaped - do not count
    as alive.

    Every wait here is bounded. A process in uninterruptible sleep - stuck in a
    driver or on I/O - does not die even on SIGKILL, and an unbounded wait for
    it would hang the notebook instead of letting the next cell record what
    happened. So after the last grace period this reports and returns.
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
                return True, f"gone after {sig.name}"
            time.sleep(0.1)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        return False, "the command is still alive after SIGKILL - probably in uninterruptible sleep; left behind"
    if _group_alive(group):
        return False, "members of its group are still alive after SIGKILL; left behind"
    return True, "gone"


def worker_verdict(run: Run, render: Path, smoke: Path,
                   expect_rate: int, expect_channels: int) -> tuple[bool, str, dict[str, Any]]:
    """The worker stage worked if it produced audio of the documented format."""
    import numpy as np
    import soundfile as sf

    if not run.ok:
        return False, f"driver {run.outcome}", {}
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


def bench_verdict(run: Run, results: Path) -> tuple[bool, str]:
    """The scored smoke cell worked if it rendered, was scored and was transcribed."""
    if not run.ok:
        return False, f"benchmark {run.outcome}"
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


def control_verdict(run: Run, check: dict[str, Any] | None) -> tuple[bool, str]:
    """The scale control worked if every baseline cell re-scored identically at 4 dp."""
    if not run.ok:
        return False, f"control {run.outcome}"
    if check is None:
        return False, "control produced no comparison"
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
