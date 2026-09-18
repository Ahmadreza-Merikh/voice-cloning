"""The GPU notebook stops where it should, and cleans up after itself.

A review ran two cells of the MOSS v1.5 notebook against a fake run that failed
- exit 1, no audio - and the benchmark cell started anyway. These tests do the
same against the generated notebook, and check the pieces it is built from:
each stage's definition of success, the gate chain, and a timeout that ends a
driver's own children rather than only the driver.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
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


def notebook_cell(marker: str) -> str:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    matches = ["".join(c["source"]) for c in cells if marker in "".join(c["source"])]
    assert len(matches) == 1, marker
    return matches[0]


def speech(rate: int = 48000, channels: int = 2, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    tone = 0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    return np.stack([tone] * channels, axis=1).astype(np.float32)


class RecordingRun:
    def __init__(self, code: int = 0, timed_out: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.code, self.timed_out = code, timed_out

    def __call__(self, command, timeout=None, cwd=None):
        self.calls.append(list(map(str, command)))
        return self.code, 1.0, self.timed_out


class TestTheNotebookStopsWhereItShould:
    def test_a_failed_worker_means_no_benchmark(self) -> None:
        # The reviewer's reproduction: worker exit 1, no render.
        state = gates.Gates()
        state.record("download", True, "fake")
        state.record("worker", False, "driver exited 1")
        run = RecordingRun()
        namespace = {"GATES": state, "run_bounded": run}
        exec(notebook_cell('GATES.skip("bench"'), namespace)
        assert run.calls == []
        assert state.status["bench"][0] is False and "worker failed" in state.status["bench"][1]

    def test_everything_downstream_of_a_failure_is_skipped(self) -> None:
        state = gates.Gates()
        state.record("download", True, "fake")
        state.record("worker", False, "timed out")
        run = RecordingRun()
        namespace = {"GATES": state, "run_bounded": run}
        for marker in ('GATES.skip("bench"', 'GATES.skip("control"', 'GATES.skip("budget"'):
            exec(notebook_cell(marker), namespace)
        assert run.calls == []
        for stage in ("bench", "control", "budget", "full"):
            assert state.status[stage][0] is False, stage

    def test_a_benchmark_that_exits_badly_fails_its_gate(self, tmp_path: Path) -> None:
        state = gates.Gates()
        state.record("worker", True, "fake")
        run = RecordingRun(code=1)
        namespace = {"GATES": state, "run_bounded": run, "sys": sys, "REPO": tmp_path,
                     "bench_verdict": gates.bench_verdict}
        exec(notebook_cell('GATES.skip("bench"'), namespace)
        assert len(run.calls) == 1
        assert state.status["bench"] == (False, "benchmark exited 1")


class TestVerdicts:
    def test_worker_needs_exit_zero_a_file_the_documented_format_and_a_good_record(self, tmp_path: Path) -> None:
        render, smoke = tmp_path / "r.wav", tmp_path / "s.json"
        assert gates.worker_verdict(1, False, render, smoke, 48000, 2)[0] is False
        assert gates.worker_verdict(0, True, render, smoke, 48000, 2)[0] is False
        assert gates.worker_verdict(0, False, render, smoke, 48000, 2)[1] == "no render written"
        sf.write(render, speech(24000, 1), 24000, subtype="FLOAT")
        smoke.write_text(json.dumps({"result": {"ok": True}}))
        assert "expected 48000 x 2" in gates.worker_verdict(0, False, render, smoke, 48000, 2)[1]
        sf.write(render, speech(), 48000, subtype="FLOAT")
        assert gates.worker_verdict(0, False, render, smoke, 48000, 2)[0] is True
        sf.write(render, np.full((48000, 2), -1.0, dtype=np.float32), 48000, subtype="FLOAT")
        assert "distinct" in gates.worker_verdict(0, False, render, smoke, 48000, 2)[1]

    def test_bench_needs_a_transcribed_finite_cell(self, tmp_path: Path) -> None:
        path = tmp_path / "x.json"
        base = {"summary": {"renders": 1, "failures": 0, "untranscribed": 0},
                "rows": [{"held_out_similarity": 0.55, "character_error_rate": 0.03}]}
        path.write_text(json.dumps(base))
        assert gates.bench_verdict(0, False, path)[0] is True
        path.write_text(json.dumps({**base, "summary": {**base["summary"], "untranscribed": 1}}))
        assert "never transcribed" in gates.bench_verdict(0, False, path)[1]
        path.write_text(json.dumps({**base, "rows": [{"held_out_similarity": None, "character_error_rate": 0.03}]}))
        assert gates.bench_verdict(0, False, path)[0] is False

    def test_control_needs_every_cell_equal(self) -> None:
        clean = {"only_reference": [], "only_control": [], "unusable": 0, "unequal": {}, "shared": 27, "max_abs": 0.0}
        assert gates.control_verdict(0, False, clean)[0] is True
        assert gates.control_verdict(0, False, {**clean, "unequal": {("a",): 0.001}, "max_abs": 0.001})[0] is False
        assert gates.control_verdict(0, False, {**clean, "only_control": [("b",)]})[0] is False
        assert gates.control_verdict(0, False, None)[0] is False

    def test_budget_refuses_a_run_that_would_not_fit(self) -> None:
        capacity = {"cuda:0": 14.6, "cuda:1": 14.6}
        fits = gates.budget_verdict(120, 60, 27, 4 * 3600, {"cuda:0": 11.0, "cuda:1": 5.0}, capacity)
        assert fits[0] is True
        too_slow = gates.budget_verdict(120, 600, 27, 4 * 3600, {"cuda:0": 11.0}, capacity)
        assert too_slow[0] is False and "exceeds" in too_slow[1]
        too_big = gates.budget_verdict(120, 60, 27, 4 * 3600, {"cuda:0": 14.2}, capacity)
        assert too_big[0] is False and "peaked" in too_big[1]


def test_a_timeout_ends_the_drivers_children_too(tmp_path: Path) -> None:
    # A driver that starts a long-lived child - as the notebook's driver starts
    # the worker - and then hangs. Killing only the driver would orphan it.
    pid_file = tmp_path / "child.pid"
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(120)\n"
    )
    code, seconds, timed_out = gates.run_bounded([sys.executable, str(driver)], timeout=2, grace=5)
    assert timed_out and code == -1
    child = int(pid_file.read_text())
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"child {child} of the timed-out driver is still alive")


def test_a_chatty_child_is_drained_and_its_output_kept(tmp_path: Path) -> None:
    # Two megabytes of carriage-return progress and no newline: what a long
    # weight-loading bar writes. A drained pipe cannot back up, so this ends.
    script = tmp_path / "chatty.py"
    script.write_text(
        "import sys\n"
        "for i in range(20000):\n"
        "    sys.stderr.write(f'\\rLoading weights: {i}/20000 ' + 'x' * 80)\n"
        "sys.stderr.write('\\ndone\\n')\n"
    )
    tee = tmp_path / "tee.log"
    code, seconds, timed_out = gates.run_bounded([sys.executable, str(script)], timeout=60, tee=tee)
    assert (code, timed_out) == (0, False)
    kept = tee.read_bytes()
    assert kept.endswith(b"done\n") and len(kept) > 2_000_000


def test_the_undrained_mode_still_runs(tmp_path: Path) -> None:
    code, seconds, timed_out = gates.run_bounded([sys.executable, "-c", "print('ok')"], timeout=30, drain=False)
    assert (code, timed_out) == (0, False)
