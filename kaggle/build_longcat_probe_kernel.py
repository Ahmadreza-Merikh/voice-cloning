#!/usr/bin/env python3
"""Generate the LongCat probe: LongCat-AudioDiT against OmniVoice on Persian speakers' voices, English text.

Pre-registered before any render (``eval/runs/persian-voice-longcat-probe.json``):
does LongCat carry a Persian speaker's voice better than OmniVoice when both
read text they know? Only a clear yes makes an adaptation of it worth costing.

One T4 x2 session, one interpreter for every arm:

1. OmniVoice's pinned stack (torch 2.8.0+cu128, transformers 5.3.0 - the
   version LongCat's config names) plus the scoring packages and ``einops``,
   checked by the same environment cell as every OmniVoice session
2. the runner check, then bounded downloads with every artifact pinned: the
   OmniVoice base at c5fdb5cc, both LongCat checkpoints (sha256), LongCat's
   code at 12c76b51 (git blob ids), google/umt5-base's tokenizer, and
   Neyshekar's validation shards only
3. the probe corpus: Neyshekar's valid speakers with the sealed test's roles
   (``scripts/prepare_probe_corpus.py``)
4. the renders, the primary pair first so a short session still answers the
   question: OmniVoice and LongCat-1B on the six English sentences, then
   LongCat-3.5B with its transformer on the second GPU, then both reading two
   Persian sentences (exploratory)
5. the reading fixed before the run (``scripts/read_longcat_probe.py``)

It mounts its own dataset, ``persian-voice-longcat-probe``, so
``persian-voice-eval`` - which the adaptation's pending sessions read - is not
re-staged for it.

    .venv/bin/python kaggle/build_dataset.py --dataset persian-voice-longcat-probe
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle datasets create \\
        -p kaggle/dataset-persian-voice-longcat-probe --dir-mode zip
    .venv/bin/python kaggle/build_longcat_probe_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/longcat-probe
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


probe = module("probe", "build_omnivoice_adapt_kernel.py")
dev = module("dev", "build_omnivoice_dev_kernel.py")

SLUG = "persian-voice-longcat-probe"
DATASET = "persian-voice-longcat-probe"
SESSION = f"kaggle:{SLUG}-v1"
#: Amended from 2.5 before the push: 2.51 GPU-hours were left this week (the plan's amendment).
SESSION_BUDGET_HOURS = 2.0
MIN_SPEAKERS = 20
DISK_GIB = 40

LONGCAT_CODE_REPO = "meituan-longcat/LongCat-AudioDiT"
LONGCAT_CODE_COMMIT = "12c76b51d2a8aa6b6c9af5b25cd5ff8f7aa8178a"
#: Git blob ids at that commit, from its tree; the fetch refuses anything else.
LONGCAT_CODE = {
    "audiodit/__init__.py": "3e02bb188c3729e3cb18f45226e73a7d43ec6a1f",
    "audiodit/configuration_audiodit.py": "36f7aacb0d535fd9bf83d7c251626544a3e17ac0",
    "audiodit/modeling_audiodit.py": "1333cc79ee449adbebe767772fe8fd16bb2d9c9f",
    "utils.py": "1e208da4d160de40f90adfeff5d82149618ac2b0",
}
LONGCAT_1B = ("meituan-longcat/LongCat-AudioDiT-1B", "4f4c19434f2e3b991998d0ead6ab2cd056d947b7",
              "7f41b20933e4466400b8487fd20ca195efa65c5ca7c61f8e9bba6316aa3edcde")
LONGCAT_3_5B = ("meituan-longcat/LongCat-AudioDiT-3.5B", "5d3e6dc007872d756c3a69b44951598d7d9bdf6d",
                "708cdbf0bdf71cd485bf10c70a6043b22729730ef55a6b4bc2de2ce552e52fd8")
UMT5 = ("google/umt5-base", "0de9394d54f8975e71838d309de1cb496c894ab9")
UMT5_SHA256 = {"spiece.model": "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",
               "tokenizer.json": "af904105ce1071b1202bba0059a841f4a7b85b48b6ec179c4948e3483476e0dd"}

#: About 25 GB; the CPU-only fetch probe moved 21 GiB over plain HTTP in 4.5 minutes.
FETCH_MINUTES = 45
PLAN = {
    "urls": [{"name": f"longcat code {path}",
              "url": f"https://raw.githubusercontent.com/{LONGCAT_CODE_REPO}/{LONGCAT_CODE_COMMIT}/{path}",
              "path": f"/tmp/longcat-code/{path}", "git_blob_prefix": blob} for path, blob in LONGCAT_CODE.items()],
    "snapshots": [
        {"name": "omnivoice base", "repo": probe.BASE, "revision": probe.BASE_REVISION,
         "local_dir": "/tmp/omnivoice-base", "sha256": {"audio_tokenizer/model.safetensors": probe.TOKENIZER_SHA256}},
        {"name": "longcat 1b", "repo": LONGCAT_1B[0], "revision": LONGCAT_1B[1],
         "allow_patterns": ["config.json", "model.safetensors", "LICENSE"], "local_dir": "/tmp/longcat-1b",
         "sha256": {"model.safetensors": LONGCAT_1B[2]}},
        {"name": "longcat 3.5b", "repo": LONGCAT_3_5B[0], "revision": LONGCAT_3_5B[1],
         "allow_patterns": ["config.json", "model.safetensors", "LICENSE"], "local_dir": "/tmp/longcat-3.5b",
         "sha256": {"model.safetensors": LONGCAT_3_5B[2]}},
        {"name": "umt5 tokenizer", "repo": UMT5[0], "revision": UMT5[1],
         "allow_patterns": ["spiece.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                            "config.json"], "local_dir": "/tmp/umt5-base", "sha256": UMT5_SHA256},
        {"name": "neyshekar valid", "repo": probe.NEY_REPO, "repo_type": "dataset", "revision": probe.NEY_REVISION,
         "allow_patterns": ["data/validation-*"], "local_dir": "/tmp/neyshekar-valid",
         "forbid": ["train-*", "test-*"]},
    ],
}

ENGLISH = "eval/manifests/probe_texts_en_v1.json"
#: label, engine, environment, extra arguments, cells per speaker, hours at most.
ARMS = (
    ("omnivoice-probe-en", "omnivoice",
     {"VOICECLONE__ENGINES__OMNIVOICE__MODEL_ID": "/tmp/omnivoice-base", "VOICECLONE__ENGINES__ASR__LANGUAGE": "en"},
     ["--sentences-file", ENGLISH, "--engine-option", "language=en"], 6, 0.85, "primary"),
    ("longcat-1b-probe-en", "longcat",
     {"VOICECLONE__ENGINES__LONGCAT__MODEL_DIR": "/tmp/longcat-1b", "VOICECLONE__ENGINES__ASR__LANGUAGE": "en"},
     ["--sentences-file", ENGLISH], 6, 0.85, "primary"),
    ("longcat-3.5b-probe-en", "longcat",
     {"VOICECLONE__ENGINES__LONGCAT__MODEL_DIR": "/tmp/longcat-3.5b",
      "VOICECLONE__ENGINES__LONGCAT__TRANSFORMER_DEVICE": "cuda:1", "VOICECLONE__ENGINES__ASR__LANGUAGE": "en"},
     ["--sentences-file", ENGLISH], 6, 1.2, "secondary"),
    ("omnivoice-probe-fa", "omnivoice",
     {"VOICECLONE__ENGINES__OMNIVOICE__MODEL_ID": "/tmp/omnivoice-base", "VOICECLONE__ENGINES__ASR__LANGUAGE": "fa"},
     ["--sentences", "s1", "s2"], 2, 0.35, "persian"),
    ("longcat-1b-probe-fa", "longcat",
     {"VOICECLONE__ENGINES__LONGCAT__MODEL_DIR": "/tmp/longcat-1b", "VOICECLONE__ENGINES__ASR__LANGUAGE": "fa"},
     ["--sentences", "s1", "s2"], 2, 0.35, "persian"),
)

SETUP = dev.SETUP.rstrip("\n") + '''
!pip -q install "einops>=0.8.0" 2>&1 | tail -3
'''

FIND_REPO_IN_PROBE_DATASET = ('found = sorted(p for p in INPUT.rglob("repo/scripts/omnivoice_adapt.py") '
                              f'if "{DATASET}" in p.parts)')


def environment() -> str:
    source = probe.ENVIRONMENT.replace(probe.SESSION, SESSION)
    assert source.count(dev.FIND_REPO) == 1, "the probe's repository lookup changed"
    source = source.replace(dev.FIND_REPO, FIND_REPO_IN_PROBE_DATASET)
    return source.replace("import hashlib\n", "import hashlib\nimport time as _clock\nNOTEBOOK_START = _clock.time()\n", 1)


DOWNLOAD = dev.FETCH + f'''
os.environ["VOICECLONE_OMNIVOICE_PYTHON"] = sys.executable
os.environ["VOICECLONE_LONGCAT_PYTHON"] = sys.executable
provenance = found[0].parents[2] / "PROVENANCE.md"
os.environ["VOICECLONE_CODE_COMMIT"] = (provenance.read_text().split("commit ")[-1].strip(" .\\n")
                                        if provenance.is_file() else "the dataset's PROVENANCE.md was not found")
NEEDED = ["scripts/prepare_probe_corpus.py", "scripts/read_longcat_probe.py", "scripts/longcat_worker.py",
          "{ENGLISH}", "src/voiceclone/engines/longcat_tts.py"]
missing = [name for name in NEEDED if not (REPO / name).is_file()]
GATES.record("code", not missing, f"missing from the dataset: {{missing}}" if missing else "the probe's code is present")
free = shutil.disk_usage("/tmp").free / 2**30
if not GATES.skip("download", ("runner", "code")):
    if free < {DISK_GIB}:
        GATES.record("download", False, f"{{free:.0f}} GiB free under /tmp, {DISK_GIB} needed")
    else:
        fetched_ok, fetched = fetch({PLAN!r}, {FETCH_MINUTES}, {dev.FETCH_FLAGS!r})
        GATES.record("download", fetched_ok, json.dumps(fetched))
'''

CORPUS = f'''\
CORPUS = Path("/tmp/probe-ney")
SPEAKERS = 0
if not GATES.skip("corpus", ("download",)):
    run = run_bounded([sys.executable, "scripts/prepare_probe_corpus.py", "--splits", str(SPLITS),
                       "--parquet-dir", "/tmp/neyshekar-valid", "--out", str(CORPUS)],
                      cwd=REPO, timeout=1200, log=LOGS / "corpus.log")
    RUNS["corpus"] = run.record()
    print(run.summary())
    roster = json.loads((CORPUS / "roster.json").read_text()) if (CORPUS / "roster.json").is_file() else {{}}
    SPEAKERS = len(roster.get("speakers", []))
    GATES.record("corpus", run.ok and SPEAKERS >= {MIN_SPEAKERS},
                 f"{{SPEAKERS}} speakers; left out: {{roster.get('left_out')}}")
'''

CONFIGURE = '''\
if not GATES.skip("configure", ("corpus",)):
    os.environ["VOICECLONE__ENGINES__LONGCAT__CODE_DIR"] = "/tmp/longcat-code"
    os.environ["VOICECLONE__ENGINES__LONGCAT__TOKENIZER_DIR"] = "/tmp/umt5-base"
    present = {path: Path(path).exists() for path in (
        "/tmp/omnivoice-base/config.json", "/tmp/longcat-1b/model.safetensors", "/tmp/longcat-3.5b/model.safetensors",
        "/tmp/longcat-code/audiodit/modeling_audiodit.py", "/tmp/longcat-code/utils.py", "/tmp/umt5-base/spiece.model")}
    GATES.record("configure", all(present.values()), json.dumps(present))
'''

RENDER = f'''\
ARMS = {[list(arm) for arm in ARMS]!r}
FACTS = {{}}


def remaining_hours():
    return {SESSION_BUDGET_HOURS} - (time.time() - NOTEBOOK_START) / 3600


def bench(label, engine, env, extra, per_speaker, hours):
    left = remaining_hours() - 0.1
    if left < 0.15:
        print(f"{{label}}: {{remaining_hours():.2f}} h left in the session - not started")
        return False, {{}}
    saved = {{key: os.environ.get(key) for key in env}}
    os.environ.update(env)
    try:
        run = run_bounded([sys.executable, "scripts/benchmark.py", "--label", label, "--engine", engine,
                           "--corpus-dir", str(CORPUS), "--seeds", "42", "--device", "cuda", *extra],
                          cwd=REPO, timeout=min(hours, left) * 3600, log=LOGS / f"{{label}}.log", monitor_every=600)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    RUNS[label] = run.record()
    print(run.summary())
    results = REPO / "eval/results" / f"{{label}}.json"
    summary = json.loads(results.read_text())["summary"] if results.is_file() else {{}}
    return run.ok and summary.get("renders", 0) >= 0.9 * SPEAKERS * per_speaker, summary


if not GATES.skip("renders", ("configure",)):
    for label, engine, env, extra, per_speaker, hours, role in ARMS:
        passed, summary = bench(label, engine, env, extra, per_speaker, hours)
        FACTS[label] = {{"role": role, "complete": passed, "held_out": summary.get("held_out_mean"),
                        "reference": summary.get("reference_mean"), "cer": summary.get("cer_mean"),
                        "renders": summary.get("renders"), "failures": summary.get("failures")}}
        print(label, json.dumps(FACTS[label]))
    for role in ("primary", "secondary", "persian"):
        rows = {{label: facts for label, facts in FACTS.items() if facts["role"] == role}}
        GATES.record(role, bool(rows) and all(f["complete"] for f in rows.values()), json.dumps(rows))
'''

READ = '''\
READING = {}
if not GATES.skip("reading", ("primary",)):
    have = {label for label, *_ in ARMS if (REPO / "eval/results" / f"{label}.json").is_file()}
    arguments = ["--omnivoice", "omnivoice-probe-en", "--longcat", "longcat-1b-probe-en",
                 "--out", str(RECORDS / "reading.json")]
    if "longcat-3.5b-probe-en" in have:
        arguments += ["--secondary", "longcat-3.5b-probe-en"]
    if {"omnivoice-probe-fa", "longcat-1b-probe-fa"} <= have:
        arguments += ["--omnivoice-fa", "omnivoice-probe-fa", "--longcat-fa", "longcat-1b-probe-fa"]
    run = run_bounded([sys.executable, "scripts/read_longcat_probe.py", *arguments], cwd=REPO, timeout=600,
                      log=LOGS / "reading.log")
    RUNS["reading"] = run.record()
    READING = json.loads((RECORDS / "reading.json").read_text()) if (RECORDS / "reading.json").is_file() else {}
    GATES.record("reading", run.ok and bool(READING), json.dumps(READING.get("decision")))
    print(json.dumps(READING.get("primary", {}).get("verdict")))
'''

COLLECT = '''\
destination = Path("/kaggle/working/longcat-probe")
destination.mkdir(exist_ok=True)
for label, *_ in ARMS:
    for suffix in (".json", ".recipe.json"):
        path = REPO / "eval/results" / f"{label}{suffix}"
        if path.is_file():
            shutil.copy2(path, destination / path.name)
for name in ("roster.json", "manifest.json"):
    if (CORPUS / name).is_file():
        shutil.copy2(CORPUS / name, destination / f"corpus-{name}")
for path in RECORDS.glob("*.json"):
    shutil.copy2(path, destination / path.name)
(destination / "gates.json").write_text(json.dumps(GATES.summary(), indent=2))
(destination / "runs.json").write_text(json.dumps(RUNS, indent=2))
(destination / "facts.json").write_text(json.dumps(FACTS, indent=2))
shutil.copy2("/kaggle/working/environment.json", destination / "environment.json")
renders = Path("/kaggle/working/renders")
renders.mkdir(exist_ok=True)
for label, *_ in ARMS:
    folder = REPO / "outputs/benchmark" / label
    for path in (folder.glob("*.wav") if folder.is_dir() else []):
        shutil.copy2(path, renders / f"{label}__{path.name}")
print("gates:", json.dumps(GATES.summary(), indent=1))
'''


def cells() -> list[str]:
    return [SETUP, environment(), probe.RUNNER, DOWNLOAD, CORPUS, CONFIGURE, RENDER, READ, COLLECT]


def main() -> int:
    sources = cells()
    for index, source in enumerate(sources):
        compile("".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
                f"<cell {index}>", "exec")
    out = HERE / "longcat-probe"
    out.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": [probe.cell(s) for s in sources],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = out / f"{SLUG}.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    metadata = {"id": f"{probe.USER}/{SLUG}", "title": SLUG, "code_file": path.name, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                "machine_shape": probe.MACHINE_SHAPE, "enable_internet": True,
                "dataset_sources": [f"{probe.USER}/{DATASET}"], "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(sources)} cells, all compile); arms {[arm[0] for arm in ARMS]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
