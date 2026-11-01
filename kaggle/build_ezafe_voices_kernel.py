#!/usr/bin/env python3
"""Generate the session that reads the ezafe check's three forms with many voices.

Registered before it ran (``eval/runs/persian-voice-ezafe-voices-v1.json``). The
ezafe check heard E5 read best, but on its one voice E5's held-out similarity
was below E1's and A1's. This asks the same of 25 voices: Neyshekar's
validation speakers, written as a corpus here by
``scripts/prepare_valid_corpus.py`` - the corpus this Mac wrote, file for file.

One T4 x2 session on step 3's code dataset (e1d96e4), as the listening checks:

1. OmniVoice's pinned stack and the scoring packages, the environment cell
2. the runner check; the code, the registered texts and the corpus script
   (written in from this notebook: the dataset predates it) checked; the base
   and Neyshekar's validation shards fetched; A1, E1 and E5 found under the
   sessions that made them, each checked against the sha256 recorded before
3. the corpus written and checked against this Mac's manifest and files
4. ``scripts/benchmark.py`` once per form: the twelve texts as the app writes
   them, with the tagger off so the dataset's code adds no ezafe of its own,
   seed 42, each speaker's reference clip and transcript; held-out similarity
   and CER scored in the same session, the recogniser's floor by the first run.
   Every render's metadata must name the weights its form was given
5. the results, the corpus record and the renders copied out

    .venv/bin/python kaggle/build_ezafe_voices_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/ezafe-voices
    .venv/bin/python kaggle/fetch_output.py marsmanam/persian-voice-ezafe-voices-run remote/ezafe-voices-v1 \\
        ezafe-voices-v1/results/ ezafe-voices-v1/corpus/ ezafe-voices-v1/gates.json ezafe-voices-v1/runs.json \\
        ezafe-voices-v1/environment.json logs/
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


probe = module("probe", "build_omnivoice_adapt_kernel.py")
dev = module("dev", "build_omnivoice_dev_kernel.py")
listening = module("listening", "build_frontend_listening_kernel.py")

SLUG = "persian-voice-ezafe-voices-run"
NAME = "ezafe-voices-v1"
FOLDER = "ezafe-voices"
TEXTS = "eval/runs/ezafe-voices-v1/texts.json"
CORPUS_SCRIPT = "scripts/prepare_valid_corpus.py"
STEP = {"slug": SLUG, "dataset": "persian-voice-frontend-listening-v2",
        "models": {"A1": "export-1000-float16", "E1": "export-1000-float16", "E5": "export-5000-float16"},
        "sessions": {"A1": "persian-voice-omnivoice-train-a", "E1": "persian-voice-omnivoice-ezafe-a",
                     "E5": "persian-voice-omnivoice-ezafe-c"},
        "known_sha256": {"A1": "9c453a74e9eab8324c27e039b9bdc052cd8d6721cecbb20df53c919a0e6ba39a",
                         "E1": "136aac641d2e04c24c8c081bcefd8ceccf310969b065ef510a5fc2903d720e2b",
                         "E5": "1ecbaa58e2cf3efb753714818c974f1027b61c1564e54b489fc5f23af3595a4e"}}
KERNEL_SOURCES = [f"{probe.USER}/{session}" for session in STEP["sessions"].values()]
#: The corpus this Mac wrote from the same split and shards (eval/runs/persian-voice-ezafe-voices-v1.json).
CORPUS_DIGESTS = {"manifest_sha256": "47b8b569d8b130fce7e89b3ae230bac003dd25d13ac46aee939922456101c944", "files_sha256": "9131dc933ea66d4853160d1e7187d9c4f9414227be2040a73b055d50e58ffcc5"}
SPEAKERS = 25
SEEDS = (42,)
RUN_HOURS = 2.5
VALIDATION = {"name": "neyshekar validation", "repo": probe.NEY_REPO, "repo_type": "dataset",
              "revision": probe.NEY_REVISION, "allow_patterns": ["data/validation-*"],
              "local_dir": "/tmp/neyshekar-validation", "forbid": ["train-*", "test-*"]}
PLAN = {"snapshots": [*dev.BASE_PLAN["snapshots"], VALIDATION]}
FETCH_MINUTES = 30


def download() -> str:
    embedded = ""
    for name in (TEXTS, CORPUS_SCRIPT):
        content = (ROOT / name).read_text(encoding="utf-8")
        embedded += (f'\n(REPO / "{name}").parent.mkdir(parents=True, exist_ok=True)\n'
                     f'(REPO / "{name}").write_text({content!r}, encoding="utf-8")')
    needed = ["scripts/benchmark.py", "scripts/prepare_test_corpus.py", "scripts/build_adaptation_splits.py",
              "scripts/prepare_adaptation_data.py", "scripts/omnivoice_worker.py", TEXTS, CORPUS_SCRIPT]
    return dev.FETCH + embedded + f'''
os.environ["VOICECLONE_OMNIVOICE_PYTHON"] = sys.executable
provenance = found[0].parents[2] / "PROVENANCE.md"
os.environ["VOICECLONE_CODE_COMMIT"] = (provenance.read_text().split("commit ")[-1].strip(" .\\n")
                                        if provenance.is_file() else "the dataset's PROVENANCE.md was not found")
NEEDED = {needed!r}
missing = [name for name in NEEDED if not (REPO / name).is_file()]
if "--sentences-file" not in (REPO / "scripts/benchmark.py").read_text():
    missing.append("benchmark.py --sentences-file")
{listening.registered_check([TEXTS, CORPUS_SCRIPT])}
GATES.record("code", not missing, f"missing or changed: {{missing}}" if missing else
             "the benchmark, the corpus code and the registered texts are present; code from " +
             os.environ["VOICECLONE_CODE_COMMIT"])
if not GATES.skip("download", ("runner", "code")):
    fetched_ok, fetched = fetch({PLAN!r}, {FETCH_MINUTES}, {dev.FETCH_FLAGS!r})
    GATES.record("download", fetched_ok, json.dumps(fetched))
# The texts carry the app's marks; the dataset's tagger must not add its own.
os.environ["VOICECLONE__TEXT__EZAFE__ENABLED"] = "false"
os.environ["VOICECLONE__ENGINES__OMNIVOICE__MODEL_ID"] = "{listening.BASE_DIR}"
''' + listening.weights(STEP)


CORPUS = f'''\
CORPUS = Path("/tmp/valid-ney")
EXPECTED_CORPUS = {CORPUS_DIGESTS!r}
ROSTER = {{}}
if not GATES.skip("corpus", ("download",)):
    run = run_bounded([sys.executable, str(REPO / "{CORPUS_SCRIPT}"), "--splits", str(SPLITS),
                       "--parquet-dir", "{VALIDATION["local_dir"]}", "--out", str(CORPUS)],
                      timeout=30 * 60, log=LOGS / "corpus.log", monitor_every=300)
    RUNS["corpus"] = run.record()
    print(run.summary())
    ROSTER = json.loads((CORPUS / "roster.json").read_text()) if (CORPUS / "roster.json").is_file() else {{}}
    got = {{"manifest_sha256": ROSTER.get("manifest_sha256"),
           "files_sha256": hashlib.sha256(json.dumps(ROSTER.get("files", {{}}), sort_keys=True).encode()).hexdigest()}}
    GATES.record("corpus", run.ok and got == EXPECTED_CORPUS and len(ROSTER.get("speakers", [])) == {SPEAKERS},
                 json.dumps({{"speakers": len(ROSTER.get("speakers", [])), "left_out": ROSTER.get("left_out"), **got}}))
'''

BENCH = f'''\
LABELS = {{form: f"ezafe-voices-{{form}}" for form in MODELS}}
SEEDS = {list(SEEDS)!r}
RESULT = {{}}
for index, form in enumerate(MODELS):
    gate = f"bench-{{form}}"
    if GATES.skip(gate, ("weights", "corpus")):
        continue
    os.environ["VOICECLONE__ENGINES__OMNIVOICE__MODEL_ID"] = WEIGHTS[form]["path"]
    run = run_bounded([sys.executable, "scripts/benchmark.py", "--label", LABELS[form], "--engine", "omnivoice",
                       "--corpus-dir", str(CORPUS), "--sentences-file", str(REPO / "{TEXTS}"),
                       "--seeds", *map(str, SEEDS), "--device", "cuda", *(["--asr-floor"] if index == 0 else [])],
                      cwd=REPO, timeout={RUN_HOURS} * 3600, log=LOGS / f"{{LABELS[form]}}.log", monitor_every=600)
    RUNS[gate] = run.record()
    print(run.summary())
    path = REPO / "eval/results" / f"{{LABELS[form]}}.json"
    data = json.loads(path.read_text()) if path.is_file() else {{}}
    summary = data.get("summary", {{}})
    models = sorted({{str((row.get("metadata") or {{}}).get("model")) for row in data.get("rows", [])}})
    RESULT[form] = {{**{{k: summary.get(k) for k in ("renders", "failures", "held_out_mean", "cer_mean")}},
                    "models_in_the_renders": models}}
    GATES.record(gate, run.ok and summary.get("failures", 1) == 0
                 and summary.get("renders", 0) == {SPEAKERS} * 12 * len(SEEDS)
                 and models == [WEIGHTS[form]["path"]], json.dumps(RESULT[form], ensure_ascii=False))
os.environ["VOICECLONE__ENGINES__OMNIVOICE__MODEL_ID"] = "{listening.BASE_DIR}"
'''

COLLECT = f'''\
destination = Path("/kaggle/working/{NAME}")
(destination / "results").mkdir(parents=True, exist_ok=True)
(destination / "corpus").mkdir(exist_ok=True)
for form, label in LABELS.items():
    if (REPO / "eval/results" / f"{{label}}.json").is_file():
        shutil.copy2(REPO / "eval/results" / f"{{label}}.json", destination / "results" / f"{{label}}.json")
    if (REPO / "outputs/benchmark" / label).is_dir():
        shutil.copytree(REPO / "outputs/benchmark" / label, destination / "renders" / label, dirs_exist_ok=True)
for name in ("manifest.json", "roster.json"):
    if (CORPUS / name).is_file():
        shutil.copy2(CORPUS / name, destination / "corpus" / name)
(destination / "gates.json").write_text(json.dumps(GATES.summary(), indent=2))
(destination / "runs.json").write_text(json.dumps(RUNS, indent=2))
(destination / "result.json").write_text(json.dumps({{"bench": RESULT, "weights": WEIGHTS}}, indent=2,
                                                    ensure_ascii=False))
shutil.copy2("/kaggle/working/environment.json", destination / "environment.json")
print("gates:", json.dumps(GATES.summary(), indent=1))
'''


def main() -> int:
    if "__" in "".join(CORPUS_DIGESTS.values()):
        raise SystemExit("write the Mac's corpus digests into CORPUS_DIGESTS first")
    cells = [dev.SETUP, listening.environment(STEP), probe.RUNNER, download(), CORPUS, BENCH, COLLECT]
    for index, source in enumerate(cells):
        compile("".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
                f"<cell {index}>", "exec")
    out = HERE / FOLDER
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
                "dataset_sources": [f"{probe.USER}/{STEP['dataset']}"], "competition_sources": [],
                "kernel_sources": KERNEL_SOURCES, "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells, all compile); texts {TEXTS}, corpus {CORPUS_DIGESTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
