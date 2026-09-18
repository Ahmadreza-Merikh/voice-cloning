"""The GPU notebook stops where it should, and its runner cannot be stalled by its display.

A review ran two cells of the MOSS v1.5 notebook against a fake run that failed
- exit 1, no audio - and the benchmark cell started anyway. These tests do the
same against the generated notebook, and check the pieces it is built from:
each stage's definition of success, the gate chain, and a runner that ends a
driver's own children rather than only the driver.

The T4 probe of 2026-09-18 then showed the runner itself stalling: its reader
wrote each chunk to the notebook display before reading the next, the display
stopped taking data, and the worker blocked on a full pipe for 20 minutes
(eval/runs/persian-voice-moss15-probe-v2.json). The tests below reproduce that
without a model - a display nobody reads, a fake worker that goes through the
real worker's stdout claim and protocol - first against a copy of the old
reader, where it deadlocks, then against the runner, where it must not.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kernel_gates", ROOT / "scripts" / "kernel_gates.py")
gates = importlib.util.module_from_spec(spec)
sys.modules["kernel_gates"] = gates
spec.loader.exec_module(gates)

NOTEBOOK = ROOT / "kaggle" / "moss15" / "persian-voice-moss15.ipynb"
WORKER = ROOT / "scripts" / "moss15_worker.py"

_check_spec = importlib.util.spec_from_file_location("runner_check", ROOT / "scripts" / "runner_check.py")
runner_check = importlib.util.module_from_spec(_check_spec)
sys.modules["runner_check"] = runner_check
_check_spec.loader.exec_module(runner_check)


class StalledDisplay:
    """A display whose reader has stopped: writes block once the pipe is full."""

    def __init__(self) -> None:
        self.read_end, self.write_end = os.pipe()

    def write(self, text: str) -> None:
        os.write(self.write_end, text.encode())

    def flush(self) -> None:
        pass

    def held(self) -> int:
        os.set_blocking(self.read_end, False)
        total = 0
        try:
            while chunk := os.read(self.read_end, 1 << 20):
                total += len(chunk)
        except BlockingIOError:
            pass
        return total

    def close(self) -> None:
        os.close(self.read_end)
        time.sleep(1.5)
        os.close(self.write_end)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def notebook_cell(marker: str) -> str:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    matches = ["".join(c["source"]) for c in cells if marker in "".join(c["source"])]
    assert len(matches) == 1, marker
    return matches[0]


def speech(rate: int = 48000, channels: int = 2, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    tone = 0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    return np.stack([tone] * channels, axis=1).astype(np.float32)


def finished(code: int | None = 0, timed_out: bool = False) -> gates.Run:
    return gates.Run(command=["x"], exit_code=None if timed_out else code, timed_out=timed_out,
                     seconds=1.0, group="ended with the command", log=Path("x.log"),
                     timeout=1.0 if timed_out else None)


class RecordingRun:
    def __init__(self, code: int = 0, timed_out: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.code, self.timed_out = code, timed_out

    def __call__(self, command, **options):
        self.calls.append(list(map(str, command)))
        return finished(self.code, self.timed_out)


def namespace(state: gates.Gates, run: RecordingRun, tmp_path: Path) -> dict:
    return {"GATES": state, "run_bounded": run, "sys": sys, "REPO": tmp_path, "LOGS": tmp_path,
            "RUNS": {}, "bench_verdict": gates.bench_verdict}


class TestTheNotebookStopsWhereItShould:
    def test_a_failed_worker_means_no_benchmark(self, tmp_path: Path) -> None:
        # The reviewer's reproduction: worker exit 1, no render.
        state = gates.Gates()
        state.record("download", True, "fake")
        state.record("worker", False, "driver exited 1")
        run = RecordingRun()
        exec(notebook_cell('GATES.skip("bench"'), namespace(state, run, tmp_path))
        assert run.calls == []
        assert state.status["bench"][0] is False and "worker failed" in state.status["bench"][1]

    def test_everything_downstream_of_a_failure_is_skipped(self, tmp_path: Path) -> None:
        state = gates.Gates()
        state.record("download", True, "fake")
        state.record("worker", False, "driver timed out after 2700s")
        run = RecordingRun()
        cells = namespace(state, run, tmp_path)
        for marker in ('GATES.skip("bench"', 'GATES.skip("control"', 'GATES.skip("budget"'):
            exec(notebook_cell(marker), cells)
        assert run.calls == []
        for stage in ("bench", "control", "budget", "full"):
            assert state.status[stage][0] is False, stage

    def test_a_benchmark_that_exits_badly_fails_its_gate(self, tmp_path: Path) -> None:
        state = gates.Gates()
        state.record("worker", True, "fake")
        run = RecordingRun(code=1)
        cells = namespace(state, run, tmp_path)
        exec(notebook_cell('GATES.skip("bench"'), cells)
        assert len(run.calls) == 1
        assert state.status["bench"] == (False, "benchmark exited 1")
        assert cells["RUNS"]["bench"]["exit_code"] == 1 and cells["RUNS"]["bench"]["timed_out"] is False

    def test_no_cell_lets_a_subprocess_write_to_the_raw_stdout(self) -> None:
        # What stalled was the notebook's raw stdout. Every subprocess a cell
        # starts either runs under run_bounded or has its output captured.
        cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        calls = 0
        for cell in cells:
            source = "".join(line for line in cell["source"] if not line.lstrip().startswith("!"))
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                        isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                    calls += 1
                    keywords = {k.arg: k.value for k in node.keywords}
                    captured = keywords.get("capture_output")
                    assert isinstance(captured, ast.Constant) and captured.value is True, ast.unparse(node)
        assert calls >= 3


class TestVerdicts:
    def test_worker_needs_exit_zero_a_file_the_documented_format_and_a_good_record(self, tmp_path: Path) -> None:
        render, smoke = tmp_path / "r.wav", tmp_path / "s.json"
        assert gates.worker_verdict(finished(1), render, smoke, 48000, 2) == (False, "driver exited 1", {})
        assert gates.worker_verdict(finished(timed_out=True), render, smoke, 48000, 2)[1] == \
            "driver timed out after 1s"
        assert gates.worker_verdict(finished(-9), render, smoke, 48000, 2)[1] == "driver killed by SIGKILL"
        assert gates.worker_verdict(finished(), render, smoke, 48000, 2)[1] == "no render written"
        sf.write(render, speech(24000, 1), 24000, subtype="FLOAT")
        smoke.write_text(json.dumps({"result": {"ok": True}}))
        assert "expected 48000 x 2" in gates.worker_verdict(finished(), render, smoke, 48000, 2)[1]
        sf.write(render, speech(), 48000, subtype="FLOAT")
        assert gates.worker_verdict(finished(), render, smoke, 48000, 2)[0] is True
        sf.write(render, np.full((48000, 2), -1.0, dtype=np.float32), 48000, subtype="FLOAT")
        assert "distinct" in gates.worker_verdict(finished(), render, smoke, 48000, 2)[1]

    def test_bench_needs_a_transcribed_finite_cell(self, tmp_path: Path) -> None:
        path = tmp_path / "x.json"
        base = {"summary": {"renders": 1, "failures": 0, "untranscribed": 0},
                "rows": [{"held_out_similarity": 0.55, "character_error_rate": 0.03}]}
        path.write_text(json.dumps(base))
        assert gates.bench_verdict(finished(), path)[0] is True
        assert gates.bench_verdict(finished(timed_out=True), path) == (False, "benchmark timed out after 1s")
        path.write_text(json.dumps({**base, "summary": {**base["summary"], "untranscribed": 1}}))
        assert "never transcribed" in gates.bench_verdict(finished(), path)[1]
        path.write_text(json.dumps({**base, "rows": [{"held_out_similarity": None, "character_error_rate": 0.03}]}))
        assert gates.bench_verdict(finished(), path)[0] is False

    def test_control_needs_every_cell_equal(self) -> None:
        clean = {"only_reference": [], "only_control": [], "unusable": 0, "unequal": {}, "shared": 27, "max_abs": 0.0}
        assert gates.control_verdict(finished(), clean)[0] is True
        assert gates.control_verdict(finished(), {**clean, "unequal": {("a",): 0.001}, "max_abs": 0.001})[0] is False
        assert gates.control_verdict(finished(), {**clean, "only_control": [("b",)]})[0] is False
        assert gates.control_verdict(finished(), None) == (False, "control produced no comparison")
        assert gates.control_verdict(finished(2), clean) == (False, "control exited 2")

    def test_budget_refuses_a_run_that_would_not_fit(self) -> None:
        capacity = {"cuda:0": 14.6, "cuda:1": 14.6}
        fits = gates.budget_verdict(120, 60, 27, 4 * 3600, {"cuda:0": 11.0, "cuda:1": 5.0}, capacity)
        assert fits[0] is True
        too_slow = gates.budget_verdict(120, 600, 27, 4 * 3600, {"cuda:0": 11.0}, capacity)
        assert too_slow[0] is False and "exceeds" in too_slow[1]
        too_big = gates.budget_verdict(120, 60, 27, 4 * 3600, {"cuda:0": 14.2}, capacity)
        assert too_big[0] is False and "peaked" in too_big[1]


def legacy_run(command: list[str], display_fd: int, timeout: float, log: Path) -> bool:
    """The reader as it was when the T4 probe ran (fa7578f): it wrote each chunk
    to the display before reading the next. Returns whether it timed out."""
    process = subprocess.Popen(command, start_new_session=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)

    def pump() -> None:
        with open(log, "ab") as sink:
            while chunk := os.read(process.stdout.fileno(), 65536):
                sink.write(chunk)
                sink.flush()
                try:
                    os.write(display_fd, chunk.replace(b"\r", b"\n"))
                except OSError:
                    return

    threading.Thread(target=pump, daemon=True).start()
    try:
        process.wait(timeout=timeout)
        return False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return True


class TestAStalledDisplay:
    def test_deadlocked_the_old_reader(self, tmp_path: Path) -> None:
        # The cause, reproduced: once the display stops taking data, the old
        # reader stops reading, the worker blocks writing progress, and its
        # ready line never comes - a job that takes a second here.
        display, log = StalledDisplay(), tmp_path / "legacy.log"
        try:
            timed_out = legacy_run(runner_check.command(tmp_path, WORKER, lines=0),
                                   display.write_end, timeout=4, log=log)
            held = display.held()
        finally:
            display.close()
        kept = log.read_bytes()
        assert timed_out
        assert b"WORKER SAID" not in kept
        assert kept.count(b"\rLoading weights") < 20000
        # The reader read more than the display ever took: the chunk it was stuck on.
        assert held < len(kept) <= held + 2 * 65536

    def test_cannot_stall_the_runner(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The same job, a display that also stops taking data - and 3,000 plain
        # lines so the display really fills. The runner must finish, keep every
        # byte, and carry the protocol through clean.
        monkeypatch.setattr(gates, "DISPLAY_WAIT", 0.5)
        display = StalledDisplay()
        try:
            ok, reason, run = runner_check.check(gates.run_bounded, tmp_path, WORKER, timeout=60,
                                                 display=display)
        finally:
            display.close()
        assert ok, reason
        assert run.display_blocked and run.shown_bytes < run.output_bytes

    def test_cannot_delay_a_timeout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The waiting thread writes its monitor reports to the log, never to the
        # display, so a stalled display cannot push the timeout back.
        monkeypatch.setattr(gates, "DISPLAY_WAIT", 0.5)
        chatty = "import sys, time\nfor i in range(5000): print('line', i, 'x' * 100, flush=True)\ntime.sleep(60)\n"
        display, log = StalledDisplay(), tmp_path / "t.log"
        try:
            run = gates.run_bounded([sys.executable, "-c", chatty], timeout=3, grace=2, monitor_every=0.5,
                                    log=log, display=display)
        finally:
            display.close()
        assert run.timed_out and run.seconds < 3 + 6
        assert log.read_text(errors="replace").count("--- group") >= 3


    def test_the_notebooks_own_check_passes_on_a_working_display(self, tmp_path: Path) -> None:
        shown = io.StringIO()
        ok, reason, run = runner_check.check(gates.run_bounded, tmp_path, WORKER, timeout=60, display=shown)
        assert ok, reason
        assert not run.display_blocked and run.shown_bytes <= gates.DISPLAY_LIMIT + gates.TAIL_LINES * 501
        assert 'WORKER SAID {"ready": true}' in shown.getvalue()


@pytest.mark.skipif(not Path("/proc/self/syscall").exists(), reason="the descriptor report needs Linux /proc")
def test_the_group_report_names_the_descriptor_a_process_waits_on() -> None:
    seen, line = runner_check.diagnostics(gates.report_group)
    assert seen, line


class TestOutcomes:
    def test_a_timeout_is_not_an_exit_code(self, tmp_path: Path) -> None:
        run = gates.run_bounded([sys.executable, "-c", "import time; time.sleep(60)"], timeout=1, grace=2,
                                log=tmp_path / "t.log", display=io.StringIO())
        assert run.timed_out and run.exit_code is None and not run.ok
        assert run.outcome == "timed out after 1s"
        assert run.group.startswith("ended by this runner after the timeout")
        assert "exited" not in run.summary()

    def test_a_nonzero_exit_is_its_own_outcome(self, tmp_path: Path) -> None:
        run = gates.run_bounded([sys.executable, "-c", "import sys; sys.exit(3)"], timeout=30,
                                log=tmp_path / "e.log", display=io.StringIO())
        assert (run.exit_code, run.timed_out, run.ok, run.outcome) == (3, False, False, "exited 3")
        assert run.group == "ended with the command"

    def test_death_by_a_signal_is_named(self, tmp_path: Path) -> None:
        run = gates.run_bounded([sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"],
                                timeout=30, log=tmp_path / "k.log", display=io.StringIO())
        assert (run.exit_code, run.timed_out, run.outcome) == (-9, False, "killed by SIGKILL")

    def test_the_group_is_never_called_ended_when_a_member_survived(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The probe's last line said "process group ended" whatever had happened.
        def survivor(process, grace):
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return False, "members of its group are still alive after SIGKILL; left behind"

        monkeypatch.setattr(gates, "_end_group", survivor)
        run = gates.run_bounded([sys.executable, "-c", "import time; time.sleep(60)"], timeout=1,
                                log=tmp_path / "s.log", display=io.StringIO())
        assert run.group.startswith("NOT ended") and "left behind" in run.summary()
        assert "ended by" not in run.summary() and "ended with" not in run.summary()


def test_a_timeout_ends_the_drivers_children_too(tmp_path: Path) -> None:
    # A driver that starts a long-lived child - as the notebook's driver starts
    # the worker - and then hangs. Killing only the driver would orphan it.
    pid_file = tmp_path / "child.pid"
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(120)\n"
    )
    run = gates.run_bounded([sys.executable, str(driver)], timeout=2, grace=5, log=tmp_path / "d.log",
                            display=io.StringIO())
    assert run.timed_out and run.exit_code is None
    child = int(pid_file.read_text())
    deadline = time.time() + 5
    while alive(child) and time.time() < deadline:
        time.sleep(0.1)
    assert not alive(child), f"child {child} of the timed-out driver is still alive"


def test_children_a_finished_command_left_behind_are_ended(tmp_path: Path) -> None:
    # A driver that exits 0 but leaves its worker running would leave it holding
    # a GPU - and the output pipe open.
    pid_file = tmp_path / "child.pid"
    driver = tmp_path / "leaves.py"
    driver.write_text(
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
    )
    run = gates.run_bounded([sys.executable, str(driver)], timeout=30, grace=2, log=tmp_path / "l.log",
                            display=io.StringIO())
    assert run.ok and run.group.startswith("ended: members outlived the command")
    child = int(pid_file.read_text())
    deadline = time.time() + 5
    while alive(child) and time.time() < deadline:
        time.sleep(0.1)
    assert not alive(child)


def test_a_progress_bar_is_kept_whole_and_shown_briefly(tmp_path: Path) -> None:
    # Two megabytes of carriage-return progress: what a long weight-loading
    # bar writes. The log keeps every byte; the view shows its first and last state.
    script = tmp_path / "chatty.py"
    script.write_text(
        "import sys\n"
        "for i in range(20000):\n"
        "    sys.stderr.write(f'\\rLoading weights: {i}/20000 ' + 'x' * 80)\n"
        "sys.stderr.write('\\ndone\\n')\n"
    )
    expected = b"".join(f"\rLoading weights: {i}/20000 ".encode() + b"x" * 80 for i in range(20000))
    expected += b"\ndone\n"
    shown, log = io.StringIO(), tmp_path / "p.log"
    run = gates.run_bounded([sys.executable, str(script)], timeout=60, log=log, display=shown)
    assert run.ok and run.output_bytes == len(expected)
    assert expected in log.read_bytes()
    view = shown.getvalue()
    assert "done" in view and "19999/20000" in view and view.count("Loading weights") <= 3


def test_the_display_stops_at_its_limit_and_says_so(tmp_path: Path) -> None:
    chatty = "for i in range(5000): print('line', i, 'x' * 100)\n"
    shown, log = io.StringIO(), tmp_path / "limit.log"
    run = gates.run_bounded([sys.executable, "-c", chatty], timeout=60, log=log, display=shown,
                            display_limit=10_000)
    assert run.ok and run.shown_bytes <= 10_000 + 40 * 150 < run.output_bytes
    view = shown.getvalue()
    assert "display limit of 10000 bytes reached" in view
    # The end of the output is shown anyway - an error or a result is usually there.
    assert "line 4999 " in view and "line 4959 " not in view and "line 4960 " in view
    assert b"line 4999 " in log.read_bytes()


def test_the_group_report_names_its_members(tmp_path: Path) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        lines = gates.report_group(child.pid)
    finally:
        child.kill()
        child.wait()
    assert lines[0].startswith(f"--- group {child.pid}")
    assert any(str(child.pid) in line for line in lines[1:])


def test_ending_a_group_never_waits_without_a_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # A process in uninterruptible sleep ignores even SIGKILL. The old code then
    # waited on it forever, which would have hung the notebook before its
    # record cell ran.
    class Unkillable:
        pid = 999_999_999

        def poll(self):
            return None

        def wait(self, timeout=None):
            if timeout is None:
                raise AssertionError("waited without a limit")
            raise subprocess.TimeoutExpired("x", timeout)

    monkeypatch.setattr(gates.os, "killpg", lambda group, sig: None)
    started = time.time()
    ended, detail = gates._end_group(Unkillable(), grace=0.2)
    assert not ended and "still alive" in detail
    assert time.time() - started < 5
