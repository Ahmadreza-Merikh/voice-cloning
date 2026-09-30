#!/usr/bin/env python3
"""Generate the one sealed test session: the chosen export and the published model, on both test sets.

The reading was fixed before any test render (SOTA_ROADMAP.md section 3.1)
and is code: ``scripts/read_sealed_test.py``. This session

1. writes the two test sets as benchmark corpora with
   ``scripts/prepare_test_corpus.py`` - Neyshekar's test shards, the primary
   set, and the Common Voice test speakers, the secondary one - listing anyone
   replaced from the reserve or left out
2. takes the candidate the development-set session chose (its
   ``choice.json``) and the float16 export of that step
3. renders the twelve sentences of ``eval/manifests/test_texts_v1.json`` for
   every test speaker, seeds 42 and 43, with the published model and then
   with the candidate, through this repository's OmniVoice engine at its
   configured settings - the weights are the only thing that differs - and
   scores both in this one session
4. applies the reading: primary, then secondary

Nothing is chosen here, and it is meant to run once.

``--primary-rerun`` builds the primary set alone as a notebook of its own,
``persian-voice-omnivoice-test-primary``: version 1 of this session scored the
secondary set, and its primary runs stopped on a harness error before anything
of theirs was scored (eval/runs/persian-voice-omnivoice-test.json). A separate
notebook keeps version 1's output as it is and gives the rerun its own scoring
environment.

    .venv/bin/python kaggle/build_omnivoice_test_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/omnivoice-test
    .venv/bin/python kaggle/build_omnivoice_test_kernel.py --primary-rerun
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


probe = module("probe", "build_omnivoice_adapt_kernel.py")
dev = module("dev", "build_omnivoice_dev_kernel.py")

SLUG = "persian-voice-omnivoice-test"
SESSION = f"kaggle:{SLUG}-v1"
PRIMARY_RERUN = "persian-voice-omnivoice-test-primary"
SOURCES = [f"{probe.USER}/{dev.SLUG}", *dev.SOURCES]
SEEDS = (42, 43)
#: source, role, planned speakers
SETS = {"ney": ("neyshekar", "primary", 28), "cv": ("cv26", "secondary", 30)}
#: The session must end before Kaggle's 12 hours or its output is lost.
SESSION_BUDGET_HOURS = 11.25
RUN_TIMEOUT_HOURS = 3.0

#: Base, Neyshekar's test shards and Common Voice: about 11 GiB.
FETCH_MINUTES = 40
TEST_PLAN = {"snapshots": [
    {"name": "base", "repo": probe.BASE, "revision": probe.BASE_REVISION, "local_dir": "/tmp/omnivoice-base",
     "sha256": {"audio_tokenizer/model.safetensors": probe.TOKENIZER_SHA256}},
    # Neyshekar's test split is the primary test set; only its test shards are fetched.
    {"name": "neyshekar test", "repo": probe.NEY_REPO, "repo_type": "dataset", "revision": probe.NEY_REVISION,
     "allow_patterns": ["data/test-*"], "local_dir": "/tmp/neyshekar-test", "forbid": ["train-*", "validation-*"]},
    {"name": "common voice 26", "repo": probe.CV_REPO, "repo_type": "dataset", "revision": probe.CV_REVISION,
     "allow_patterns": ["data/*"], "local_dir": "/tmp/cv26"},
]}

DOWNLOAD = dev.FETCH + f'''
os.environ["VOICECLONE_OMNIVOICE_PYTHON"] = sys.executable
provenance = found[0].parents[2] / "PROVENANCE.md"
os.environ["VOICECLONE_CODE_COMMIT"] = (provenance.read_text().split("commit ")[-1].strip(" .\\n")
                                        if provenance.is_file() else "the dataset's PROVENANCE.md was not found")
PARQUET = {{"neyshekar": Path("/tmp/neyshekar-test"), "cv26": Path("/tmp/cv26")}}
TEXTS = REPO / "eval/manifests/test_texts_v1.json"
BASE = None
# A dataset staged before this code existed would fail only after the downloads.
NEEDED = ["scripts/prepare_test_corpus.py", "scripts/read_sealed_test.py", "scripts/build_adaptation_splits.py",
          "eval/manifests/test_texts_v1.json"]
missing = [name for name in NEEDED if not (REPO / name).is_file()]
if "--sentences-file" not in (REPO / "scripts/benchmark.py").read_text():
    missing.append("benchmark.py --sentences-file")
GATES.record("code", not missing, f"missing from the dataset: {{missing}}" if missing else "the test code is present")
if not GATES.skip("download", ("runner", "code")):
    fetched_ok, fetched = fetch({TEST_PLAN!r}, {FETCH_MINUTES}, {dev.FETCH_FLAGS!r})
    if fetched_ok:
        BASE = Path("/tmp/omnivoice-base")
    GATES.record("download", fetched_ok, json.dumps(fetched))
'''

def corpora_cell(sets: dict) -> str:
    return f'''\
SETS = {sets!r}
CORPORA = {{}}
if not GATES.skip("corpora", ("download",)):
    ok = True
    for name, (source, role, planned) in SETS.items():
        out = Path(f"/tmp/test-{{name}}")
        run = run_bounded([sys.executable, str(REPO / "scripts/prepare_test_corpus.py"), "--splits", str(SPLITS),
                           "--source", source, "--parquet-dir", str(PARQUET[source]), "--out", str(out)],
                          timeout=2 * 3600, log=LOGS / f"corpus_{{name}}.log", monitor_every=600)
        RUNS[f"corpus_{{name}}"] = run.record()
        print(run.summary())
        if not run.ok or not (out / "roster.json").is_file():
            ok = False
            continue
        roster = json.loads((out / "roster.json").read_text())
        CORPORA[name] = {{"dir": out, "roster": roster, "speakers": len(roster["speakers"]), "planned": planned}}
        # Fewer than planned is reported, never topped up: no speaker is added after looking.
        print(name, len(roster["speakers"]), "speakers of", planned, "| replaced", roster["replaced"],
              "| left out", roster["left_out"])
        ok = ok and len(roster["speakers"]) >= 5
    GATES.record("corpora", ok, json.dumps({{name: {{k: v for k, v in c.items() if k != "dir"}}
                                             for name, c in CORPORA.items()}}, default=str))
'''


CORPORA = corpora_cell(SETS)

CANDIDATE = '''\
CANDIDATE = {}
EXPORT = None
if not GATES.skip("candidate", ("corpora",)):
    choices = sorted(INPUT.rglob("choice.json"))
    choice = json.loads(choices[0].read_text()) if len(choices) == 1 else {}
    best = choice.get("candidate")
    if best:
        matches = sorted(p for p in INPUT.rglob(f"export-{best['step']}-float16") if p.is_dir())
        EXPORT = matches[0] if len(matches) == 1 else None
    weights = sorted(EXPORT.glob("*.safetensors")) if EXPORT else []
    CANDIDATE = {"choice_files": [str(p) for p in choices], "rule": choice.get("rule"), "candidate": best,
                 "export": str(EXPORT) if EXPORT else None,
                 "weights_sha256": {p.name: sha256(p) for p in weights}}
    GATES.record("candidate", bool(best) and EXPORT is not None and bool(weights), json.dumps(CANDIDATE))
'''

RENDER = f'''\
MODEL_KEY = "VOICECLONE__ENGINES__OMNIVOICE__MODEL_ID"
SEEDS = {list(SEEDS)!r}
LABELS = {{}}


def remaining_hours():
    return {SESSION_BUDGET_HOURS} - (time.time() - NOTEBOOK_START) / 3600


def bench(label, model, corpus_dir, speakers, *extra):
    timeout = min({RUN_TIMEOUT_HOURS}, remaining_hours() - 0.5) * 3600
    if timeout < 1800:
        print(f"{{label}}: {{remaining_hours():.2f}} h left in the session - not started")
        return False, {{}}
    saved = os.environ.get(MODEL_KEY)
    os.environ[MODEL_KEY] = str(model)
    try:
        run = run_bounded([sys.executable, "scripts/benchmark.py", "--label", label, "--engine", "omnivoice",
                           "--corpus-dir", str(corpus_dir), "--sentences-file", str(TEXTS),
                           "--seeds", *map(str, SEEDS), "--device", "cuda", *extra],
                          cwd=REPO, timeout=timeout, log=LOGS / f"{{label}}.log", monitor_every=600)
    finally:
        if saved is None:
            os.environ.pop(MODEL_KEY, None)
        else:
            os.environ[MODEL_KEY] = saved
    RUNS[label] = run.record()
    print(run.summary())
    results = REPO / "eval/results" / f"{{label}}.json"
    summary = json.loads(results.read_text())["summary"] if results.is_file() else {{}}
    expected = speakers * 12 * len(SEEDS)
    return run.ok and summary.get("failures", 1) == 0 and summary.get("renders", 0) == expected, summary


if not GATES.skip("renders", ("candidate",)):
    ok = True
    facts = {{}}
    for name, (source, role, planned) in SETS.items():
        corpus = CORPORA[name]
        published = f"omnivoice-published-test-{{name}}"
        candidate = f"omnivoice-fa-{{CANDIDATE['candidate']['step']}}-test-{{name}}"
        LABELS[name] = {{"published": published, "candidate": candidate, "role": role}}
        # The recogniser's floor is measured on this set's real clips, by the first run.
        passed_published, summary_published = bench(published, BASE, corpus["dir"], corpus["speakers"], "--asr-floor")
        passed_candidate, summary_candidate = bench(candidate, EXPORT, corpus["dir"], corpus["speakers"])
        facts[name] = {{"published": summary_published.get("held_out_mean"),
                       "candidate": summary_candidate.get("held_out_mean"),
                       "complete": [passed_published, passed_candidate]}}
        ok = ok and passed_published and passed_candidate
    GATES.record("renders", ok, json.dumps(facts))
'''

READ = '''\
import compare_runs  # noqa: E402
import read_sealed_test  # noqa: E402

READING = {}
if not GATES.skip("reading", ("candidate",)):
    for name, labels in LABELS.items():
        try:
            READING[name] = read_sealed_test.reading(
                labels["candidate"], compare_runs.load(labels["candidate"]),
                labels["published"], compare_runs.load(labels["published"]),
                labels["role"], read_sealed_test.categories_of(TEXTS))
        except SystemExit as exc:
            READING[name] = {"error": str(exc)}
        print(name, json.dumps(READING[name].get("verdict", READING[name]), ensure_ascii=False))
    GATES.record("reading", bool(READING) and all("verdict" in r for r in READING.values()),
                 json.dumps({name: r.get("verdict") for name, r in READING.items()}))
'''

COLLECT = '''\
destination = Path("/kaggle/working/test")
destination.mkdir(exist_ok=True)
for path in (REPO / "eval/results").glob("*.json"):
    shutil.copy2(path, destination / path.name)
(destination / "reading.json").write_text(json.dumps(READING, indent=2, ensure_ascii=False))
(destination / "candidate.json").write_text(json.dumps(CANDIDATE, indent=2))
(destination / "gates.json").write_text(json.dumps(GATES.summary(), indent=2))
(destination / "runs.json").write_text(json.dumps(RUNS, indent=2))
shutil.copy2("/kaggle/working/environment.json", destination / "environment.json")
for name, corpus in CORPORA.items():
    # The real clips stay with the renders: a listening test needs both.
    shutil.copytree(corpus["dir"], Path("/kaggle/working/corpora") / name, dirs_exist_ok=True)
renders = Path("/kaggle/working/renders")
renders.mkdir(exist_ok=True)
for labels in LABELS.values():
    for label in (labels["published"], labels["candidate"]):
        folder = REPO / "outputs/benchmark" / label
        for path in folder.glob("*.wav") if folder.is_dir() else []:
            shutil.copy2(path, renders / f"{label}__{path.name}")
print("gates:", json.dumps(GATES.summary(), indent=1))
'''


def main(argv: list[str] | None = None) -> int:
    rerun = "--primary-rerun" in (argv or [])
    slug = PRIMARY_RERUN if rerun else SLUG
    sets = {"ney": SETS["ney"]} if rerun else SETS
    environment = dev.environment(f"kaggle:{slug}-v1" if rerun else SESSION).replace(
        "import hashlib\n", "import hashlib\nimport time as _clock\nNOTEBOOK_START = _clock.time()\n", 1)
    assert "NOTEBOOK_START" in environment
    helpers = '''\
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
'''
    cells = [dev.SETUP, environment, probe.RUNNER, helpers + "\n\n" + DOWNLOAD, corpora_cell(sets), CANDIDATE, RENDER,
             READ, COLLECT]
    for index, source in enumerate(cells):
        compile("".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
                f"<cell {index}>", "exec")
    out = HERE / ("omnivoice-test-primary" if rerun else "omnivoice-test")
    out.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": [probe.cell(s) for s in cells],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = out / f"{slug}.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    metadata = {"id": f"{probe.USER}/{slug}", "title": slug, "code_file": path.name, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                "machine_shape": probe.MACHINE_SHAPE, "enable_internet": True,
                "dataset_sources": [f"{probe.USER}/{probe.DATASET}"], "competition_sources": [],
                "kernel_sources": SOURCES, "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells, all compile); sources {SOURCES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
