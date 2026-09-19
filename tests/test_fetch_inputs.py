"""A GPU session's downloads: bounded, retried, checked, and reported as they go.

Session a v1 of the OmniVoice adaptation stalled in a download cell that had no
time limit and printed nothing more for the rest of the session. These tests
pin the replacement without touching the network: a request always carries a
timeout, a failure is retried a bounded number of times, the checks run, and
the report exists after every step, so a run killed at its budget still says
how far it got.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_inputs", ROOT / "scripts" / "fetch_inputs.py")
fetch = importlib.util.module_from_spec(spec)
sys.modules["fetch_inputs"] = fetch
spec.loader.exec_module(fetch)


def test_a_request_always_carries_a_timeout(monkeypatch) -> None:
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    def urlopen(url, timeout=None):
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    assert fetch.fetch_url("https://example.invalid/x") == b"{}"
    assert seen["timeout"] == fetch.REQUEST_TIMEOUT


def test_a_failure_is_retried_a_bounded_number_of_times() -> None:
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise TimeoutError("read timed out")
        return "done"

    result, tries = fetch.attempt("step", flaky, attempts=3, pause=0, log=lambda line: None)
    assert result == "done" and len(tries) == 2 and "TimeoutError" in tries[0]["error"]

    with pytest.raises(TimeoutError):
        fetch.attempt("step", lambda: (_ for _ in ()).throw(TimeoutError("again")), attempts=2, pause=0,
                      log=lambda line: None)


@pytest.fixture
def hub(monkeypatch, tmp_path: Path):
    import huggingface_hub

    made = []

    def snapshot_download(repo, repo_type="model", revision=None, allow_patterns=None, local_dir=None):
        folder = Path(local_dir)
        (folder / "data").mkdir(parents=True, exist_ok=True)
        (folder / "data" / "train-00000.parquet").write_bytes(b"parquet")
        (folder / "model.safetensors").write_bytes(b"weights")
        made.append((repo, revision, allow_patterns))
        return str(folder)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return made


def plan(tmp_path: Path, **extra) -> dict:
    step = {"name": "base", "repo": "k2-fsa/OmniVoice", "revision": "c5fdb5cc", "local_dir": str(tmp_path / "base"),
            "sha256": {"model.safetensors": fetch.hashlib.sha256(b"weights").hexdigest()}}
    step.update(extra)
    return {"snapshots": [step]}


def test_a_snapshot_is_fetched_checked_and_reported(hub, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report = fetch.run(plan(tmp_path), report_path, log=lambda line: None)
    assert report["ok"] is True
    assert hub == [("k2-fsa/OmniVoice", "c5fdb5cc", None)]
    saved = json.loads(report_path.read_text())
    assert saved["ok"] is True and saved["steps"][0]["bytes"] == len(b"parquet") + len(b"weights")


def test_a_wrong_hash_stops_the_stage_and_the_report_says_why(hub, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    with pytest.raises(SystemExit, match="sha256"):
        fetch.run(plan(tmp_path, sha256={"model.safetensors": "0" * 64}), report_path, log=lambda line: None)
    saved = json.loads(report_path.read_text())
    assert saved["ok"] is False and "expected" in saved["steps"][0]["failed_check"]


def test_a_sealed_file_that_arrives_stops_the_stage(hub, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="forbidden"):
        fetch.run(plan(tmp_path, forbid=["train-*.parquet"]), tmp_path / "report.json", log=lambda line: None)


def test_a_url_is_checked_against_its_git_blob(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fetch, "fetch_url", lambda url, timeout=60: b"config")
    good = {"urls": [{"name": "config", "url": "u", "path": str(tmp_path / "c.json"),
                      "git_blob_prefix": fetch.git_blob(b"config")[:12]}]}
    assert fetch.run(good, tmp_path / "r.json", log=lambda line: None)["ok"] is True
    assert (tmp_path / "c.json").read_bytes() == b"config"
    bad = {"urls": [{**good["urls"][0], "git_blob_prefix": "000000000000"}]}
    with pytest.raises(SystemExit, match="git blob"):
        fetch.run(bad, tmp_path / "r.json", log=lambda line: None)
