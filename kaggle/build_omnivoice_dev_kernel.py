#!/usr/bin/env python3
"""Generate the development-set session that chooses among the adaptation's exports.

The reading was fixed before the run (SOTA_ROADMAP.md section 8): every
float16 export the training sessions kept (steps 1,000-5,000) is rendered and
scored on the development set - nine speakers, three sentences, seed 42, the
first reference clip, as every engine was - beside the published model scored
in the same session, and the export with the highest held-out similarity whose
CER interval against the published model does not lie above zero is the
candidate. Choosing among five on the development set is what it is for; the
test sets stay sealed.

Everything renders through this repository's OmniVoice engine and worker at
its configured settings (float16 on CUDA, 32 steps, guidance 2.0 ...), so the
only thing that differs between the runs is the weights. The published model
is loaded from a local snapshot of the revision every earlier OmniVoice number
was measured on. The baseline's F5 renders are re-scored here too, as the
scale control every earlier kernel carried.

    .venv/bin/python kaggle/build_omnivoice_dev_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/omnivoice-dev
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("probe", HERE / "build_omnivoice_adapt_kernel.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

SLUG = "persian-voice-omnivoice-dev"
SESSION = f"kaggle:{SLUG}-v1"
SOURCES = [f"{probe.USER}/persian-voice-omnivoice-train-{s}" for s in ("a", "b", "c")]
CONTROL_LABEL = "omnivoice-published-devctl"
BASELINE_CONTROL = "baseline-omnidev-env"
EXPORT_LABEL = "omnivoice-fa-{step}"

FIND_REPO = 'found = sorted(INPUT.rglob("repo/scripts/omnivoice_adapt.py"))'
# Every training session's output carries the copy of the repository it ran
# with, and those are mounted here too; the code must be the dataset's.
FIND_REPO_IN_DATASET = ('found = sorted(p for p in INPUT.rglob("repo/scripts/omnivoice_adapt.py") '
                        f'if "{probe.DATASET}" in p.parts)')


def environment(session: str) -> str:
    source = probe.ENVIRONMENT.replace(probe.SESSION, session)
    assert source.count(FIND_REPO) == 1, "the probe's repository lookup changed"
    return source.replace(FIND_REPO, FIND_REPO_IN_DATASET)

FETCH = '''\
def fetch(plan, minutes, flags=()):
    """scripts/fetch_inputs.py under run_bounded: every request timed out, the stage bounded."""
    plan_path, report_path = RECORDS / "fetch_plan.json", RECORDS / "fetch.json"
    plan_path.write_text(json.dumps(plan))
    run = run_bounded([sys.executable, str(REPO / "scripts/fetch_inputs.py"), "--plan", str(plan_path),
                       "--report", str(report_path), *flags], timeout=minutes * 60, log=LOGS / "fetch.log",
                      monitor_every=300)
    RUNS["fetch"] = run.record()
    print(run.summary())
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    fetched = {step["name"]: {k: step.get(k) for k in ("gib", "seconds", "tries", "failed_check")}
               for step in report.get("steps", [])}
    return run.ok and report.get("ok") is True, fetched
'''

#: The base alone, about 3 GiB; decided flags come from the CPU-only fetch probe.
FETCH_MINUTES = 20
#: Plain HTTP: the CPU-only probe (eval/runs/persian-voice-fetch-probe-v1.json) fetched everything
#: both ways, Xet in 2.6 min and HTTP in 4.5; HTTP's per-request timeout is the documented one.
FETCH_FLAGS: list[str] = ["--no-xet"]
BASE_PLAN = {"snapshots": [{"name": "base", "repo": probe.BASE, "revision": probe.BASE_REVISION,
                            "local_dir": "/tmp/omnivoice-base",
                            "sha256": {"audio_tokenizer/model.safetensors": probe.TOKENIZER_SHA256}}]}

SETUP = probe.SETUP.rstrip("\n") + '''
!pip -q install speechbrain faster-whisper jiwer pyloudnorm noisereduce num2words silero-vad onnxruntime 2>&1 | tail -3
'''

RUN = FETCH + f'''
os.environ["VOICECLONE_OMNIVOICE_PYTHON"] = sys.executable
provenance = found[0].parents[2] / "PROVENANCE.md"
os.environ["VOICECLONE_CODE_COMMIT"] = (provenance.read_text().split("commit ")[-1].strip(" .\\n")
                                        if provenance.is_file() else "the dataset's PROVENANCE.md was not found")
BASE = Path("/tmp/omnivoice-base")
if not GATES.skip("download", ("runner",)):
    fetched_ok, fetched = fetch({BASE_PLAN!r}, {FETCH_MINUTES}, {FETCH_FLAGS!r})
    GATES.record("download", fetched_ok, json.dumps(fetched))
EXPORTS = sorted({{p.name: p for p in INPUT.rglob("export-*-float16") if p.is_dir()}}.values(),
                 key=lambda p: int(p.name.split("-")[1]))
print("exports:", [str(p) for p in EXPORTS])
MODEL_KEY = "VOICECLONE__ENGINES__OMNIVOICE__MODEL_ID"


def bench(label, *arguments, model=None):
    # The engine reads its weights from this override; nothing else differs.
    saved = os.environ.get(MODEL_KEY)
    if model is not None:
        os.environ[MODEL_KEY] = str(model)
    try:
        run = run_bounded([sys.executable, "scripts/benchmark.py", "--label", label, *arguments,
                           "--device", "cuda"], cwd=REPO, timeout=5400, log=LOGS / f"{{label}}.log",
                          monitor_every=600)
    finally:
        if saved is None:
            os.environ.pop(MODEL_KEY, None)
        else:
            os.environ[MODEL_KEY] = saved
    RUNS[label] = run.record()
    print(run.summary())
    results = REPO / "eval/results" / f"{{label}}.json"
    summary = json.loads(results.read_text())["summary"] if results.is_file() else {{}}
    return run.ok and summary.get("failures", 1) == 0 and summary.get("renders", 0) == 27, summary


SCORES = {{}}
if not GATES.skip("control", ("runner", "download")):
    ok, summary = bench("{BASELINE_CONTROL}", "--score-only", "--replay-from", "baseline")
    GATES.record("scale-control", ok, json.dumps(summary))
    ok, summary = bench("{CONTROL_LABEL}", "--engine", "omnivoice", model=BASE)
    SCORES["published"] = summary
    GATES.record("control", ok, json.dumps(summary))

if not GATES.skip("exports", ("control",)):
    ok = bool(EXPORTS)
    for export in EXPORTS:
        step = int(export.name.split("-")[1])
        passed, summary = bench("{EXPORT_LABEL}".format(step=step), "--engine", "omnivoice", model=export)
        SCORES[step] = summary
        ok = ok and passed
    GATES.record("exports", ok, json.dumps({{str(k): v.get("held_out_mean") for k, v in SCORES.items()}}))
'''

CHOOSE = f'''\
sys.path.insert(0, str(REPO / "scripts"))
import compare_runs  # noqa: E402

CHOICE = {{}}
if not GATES.skip("choice", ("exports",)):
    control = compare_runs.load("{CONTROL_LABEL}")
    rows = []
    for export in EXPORTS:
        step = int(export.name.split("-")[1])
        label = "{EXPORT_LABEL}".format(step=step)
        data = compare_runs.load(label)
        similarity = compare_runs.compare(label, data, "{CONTROL_LABEL}", control, "held_out_similarity")
        cer = compare_runs.compare(label, data, "{CONTROL_LABEL}", control, "character_error_rate")
        rows.append({{"step": step, "label": label, "held_out": data["summary"]["held_out_mean"],
                     "similarity_minus_published": [similarity.mean, similarity.low, similarity.high],
                     "cer_minus_published": [cer.mean, cer.low, cer.high],
                     "cer_interval_above_zero": cer.low is not None and cer.low > 0}})
    eligible = [r for r in rows if not r["cer_interval_above_zero"]]
    best = max(eligible, key=lambda r: r["held_out"]) if eligible else None
    CHOICE = {{"rule": "highest held-out similarity among exports whose CER interval against the published model "
                       "does not lie above zero (SOTA_ROADMAP section 8, fixed before the run)",
              "rows": rows, "candidate": best}}
    GATES.record("choice", best is not None, json.dumps(CHOICE))
'''

COLLECT = '''\
destination = Path("/kaggle/working/dev")
destination.mkdir(exist_ok=True)
for path in (REPO / "eval/results").glob("*.json"):
    shutil.copy2(path, destination / path.name)
(destination / "choice.json").write_text(json.dumps(CHOICE, indent=2))
(destination / "gates.json").write_text(json.dumps(GATES.summary(), indent=2))
(destination / "runs.json").write_text(json.dumps(RUNS, indent=2))
shutil.copy2("/kaggle/working/environment.json", destination / "environment.json")
renders = Path("/kaggle/working/renders")
renders.mkdir(exist_ok=True)
for folder in (REPO / "outputs/benchmark").iterdir():
    if folder.is_dir() and folder.name.startswith("omnivoice"):
        for path in folder.glob("*.wav"):
            shutil.copy2(path, renders / f"{folder.name}__{path.name}")
print("gates:", json.dumps(GATES.summary(), indent=1))
'''


def main() -> int:
    cells = [SETUP, environment(SESSION), probe.RUNNER, RUN, CHOOSE, COLLECT]
    for index, source in enumerate(cells):
        compile("".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
                f"<cell {index}>", "exec")
    out = HERE / "omnivoice-dev"
    out.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": [probe.cell(s) for s in cells],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = out / f"{SLUG}.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    metadata = {"id": f"{probe.USER}/{SLUG}", "title": SLUG, "code_file": path.name, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                "machine_shape": probe.MACHINE_SHAPE, "enable_internet": True,
                "dataset_sources": [f"{probe.USER}/{probe.DATASET}"], "competition_sources": [],
                "kernel_sources": SOURCES, "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells, all compile)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
