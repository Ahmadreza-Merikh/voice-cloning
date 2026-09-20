#!/usr/bin/env python3
"""Generate the arms session: F5-TTS and Pocket-TTS on the sealed test's own corpora.

Registered on 2026-09-20, before anything had rendered a test clip
(``eval/runs/persian-voice-omnivoice-test.json`` ->
``amendment_2026_09_20_other_engines_on_the_same_test``). The sealed test as
first written compares the adaptation against the model it started from and
answers nothing else - and the development set that chose that base is very
probably contaminated, eight of its nine speakers being Common Voice. So two
other engines render the same cells on the same unseen speakers:

* **F5-TTS Persian**, the best-measured alternative on the development set
  (0.546 against OmniVoice's 0.583) and the engine this project started from
* **Pocket-TTS Farsi v2**, trained on 973 h of Persian from 2,978 speakers,
  measured on the development set at 0.333
  (``eval/runs/persian-voice-pocket-tts-v1.json``)

They run here rather than inside the test session because that session's
interpreter carries OmniVoice's pinned stack from ``uv.lock``; installing
``f5-tts`` or ``pocket-tts`` beside it could change the very environment the
primary comparison is measured in. Both arms install **before** either renders,
so they share one environment and their CER is comparable with each other -
never with the test session's, per FINDINGS section 9.

The corpora are rebuilt here by the same deterministic
``scripts/prepare_test_corpus.py`` from the same manifest, so these are the
same cells: same speakers, same reference and held-out clips, same twelve
texts, same seeds.

Nothing here can promote or demote the primary result. It is read beside it.

    .venv/bin/python kaggle/build_omnivoice_arms_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/omnivoice-arms
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
test = module("test", "build_omnivoice_test_kernel.py")

SLUG = "persian-voice-omnivoice-arms"
SESSION = f"kaggle:{SLUG}-v1"
SEEDS = test.SEEDS
SETS = test.SETS
SESSION_BUDGET_HOURS = test.SESSION_BUDGET_HOURS
RUN_TIMEOUT_HOURS = test.RUN_TIMEOUT_HOURS

#: The engines, in the order they render: the better-measured one first, so a
#: session cut short still answers the question it was added for.
ARMS = ("f5", "pocket")

#: Revisions as measured here. F5's Persian checkpoint is what FINDINGS scores;
#: the Pocket pins are the ones its development-set run used.
F5_REPO, F5_REVISION = "Lumos675/F5_TTS_Persian", "cc322511e417"
F5_BASE_REPO, F5_BASE_REVISION = "SWivid/F5-TTS", "84e5a410d9ce"
POCKET_REPO = "mehdi-hf/pocket-tts-farsi-v2"
POCKET_REVISION = "e87097ed98edb8bd9c9bcabe9b575a51a466d0e4"
G2P_REPO = "mehdi-hf/Homo-GE2PE-Persian-HF"
G2P_REVISION = "e314087bc761e323d127ee7899cb34f7c7d697cd"

#: Test shards, Common Voice, both F5 checkpoints and Pocket: about 13 GiB.
FETCH_MINUTES = 45
ARMS_PLAN = {"snapshots": [
    # Neyshekar's test split is the primary set; only its test shards are fetched.
    {"name": "neyshekar test", "repo": probe.NEY_REPO, "repo_type": "dataset", "revision": probe.NEY_REVISION,
     "allow_patterns": ["data/test-*"], "local_dir": "/tmp/neyshekar-test", "forbid": ["train-*", "validation-*"]},
    {"name": "common voice 26", "repo": probe.CV_REPO, "repo_type": "dataset", "revision": probe.CV_REVISION,
     "allow_patterns": ["data/*"], "local_dir": "/tmp/cv26"},
    {"name": "f5 persian", "repo": F5_REPO, "revision": F5_REVISION,
     "allow_patterns": ["model_last.pt", "vocab.txt"], "local_dir": "/tmp/f5-persian"},
    {"name": "f5 base", "repo": F5_BASE_REPO, "revision": F5_BASE_REVISION,
     "allow_patterns": ["F5TTS_v1_Base/*"], "local_dir": "/tmp/f5-base"},
    {"name": "pocket", "repo": POCKET_REPO, "revision": POCKET_REVISION,
     "allow_patterns": ["*.yaml", "*.safetensors", "*.py", "*.model", "README.md"],
     "local_dir": "/tmp/pocket/model"},
    {"name": "pocket g2p", "repo": G2P_REPO, "revision": G2P_REVISION, "local_dir": "/tmp/pocket/g2p"},
]}

#: Both arms are installed before either renders, so one environment scores both.
SETUP = '''\
!pip -q install f5-tts "pocket-tts @ git+https://github.com/mallahyari/pocket-tts@main" speechbrain faster-whisper jiwer pyloudnorm noisereduce num2words silero-vad onnxruntime 2>&1 | tail -5
'''

DOWNLOAD = dev.FETCH + f'''
# Both engines run in this interpreter: nothing here is pinned to protect.
os.environ["VOICECLONE_POCKET_PYTHON"] = sys.executable
provenance = found[0].parents[2] / "PROVENANCE.md"
os.environ["VOICECLONE_CODE_COMMIT"] = (provenance.read_text().split("commit ")[-1].strip(" .\\n")
                                        if provenance.is_file() else "the dataset's PROVENANCE.md was not found")
PARQUET = {{"neyshekar": Path("/tmp/neyshekar-test"), "cv26": Path("/tmp/cv26")}}
TEXTS = REPO / "eval/manifests/test_texts_v1.json"
NEEDED = ["scripts/prepare_test_corpus.py", "scripts/pocket_worker.py", "eval/manifests/test_texts_v1.json"]
missing = [name for name in NEEDED if not (REPO / name).is_file()]
if "pocket" not in (REPO / "src/voiceclone/engines/__init__.py").read_text():
    missing.append("the pocket engine")
GATES.record("code", not missing, f"missing from the dataset: {{missing}}" if missing else "the arms code is present")
if not GATES.skip("download", ("runner", "code")):
    fetched_ok, fetched = fetch({ARMS_PLAN!r}, {FETCH_MINUTES}, {dev.FETCH_FLAGS!r})
    GATES.record("download", fetched_ok, json.dumps(fetched))
'''

#: Point each engine at what was fetched, so nothing reaches the network again.
CONFIGURE = '''\
if not GATES.skip("configure", ("download",)):
    os.environ["VOICECLONE__ENGINES__F5__CHECKPOINT"] = "/tmp/f5-persian/model_last.pt"
    os.environ["VOICECLONE__ENGINES__F5__VOCAB"] = "/tmp/f5-persian/vocab.txt"
    os.environ["VOICECLONE__ENGINES__F5__FALLBACK_CHECKPOINT"] = (
        "/tmp/f5-base/F5TTS_v1_Base/model_1250000.safetensors")
    os.environ["VOICECLONE__ENGINES__POCKET__MODEL_DIR"] = "/tmp/pocket/model"
    os.environ["VOICECLONE__ENGINES__POCKET__G2P_DIR"] = "/tmp/pocket/g2p"
    present = {name: Path(path).exists() for name, path in (
        ("f5", "/tmp/f5-persian/model_last.pt"),
        ("f5 base", "/tmp/f5-base/F5TTS_v1_Base/model_1250000.safetensors"),
        ("pocket", "/tmp/pocket/model/model.safetensors"),
        ("g2p", "/tmp/pocket/g2p/config.json"))}
    GATES.record("configure", all(present.values()), json.dumps(present))
'''

RENDER = f'''\
SEEDS = {list(SEEDS)!r}
ARMS = {list(ARMS)!r}
LABELS = {{}}


def remaining_hours():
    return {SESSION_BUDGET_HOURS} - (time.time() - NOTEBOOK_START) / 3600


def bench(label, engine, corpus_dir, speakers, *extra):
    timeout = min({RUN_TIMEOUT_HOURS}, remaining_hours() - 0.5) * 3600
    if timeout < 1800:
        print(f"{{label}}: {{remaining_hours():.2f}} h left in the session - not started")
        return False, {{}}
    run = run_bounded([sys.executable, "scripts/benchmark.py", "--label", label, "--engine", engine,
                       "--corpus-dir", str(corpus_dir), "--sentences-file", str(TEXTS),
                       "--seeds", *map(str, SEEDS), "--device", "cuda", *extra],
                      cwd=REPO, timeout=timeout, log=LOGS / f"{{label}}.log", monitor_every=600)
    RUNS[label] = run.record()
    print(run.summary())
    results = REPO / "eval/results" / f"{{label}}.json"
    summary = json.loads(results.read_text())["summary"] if results.is_file() else {{}}
    expected = speakers * 12 * len(SEEDS)
    return run.ok and summary.get("failures", 1) == 0 and summary.get("renders", 0) == expected, summary


if not GATES.skip("renders", ("corpora", "configure")):
    facts = {{}}
    for name, (source, role, planned) in SETS.items():
        corpus = CORPORA[name]
        LABELS[name] = {{"role": role}}
        for arm in ARMS:
            label = f"{{arm}}-test-{{name}}"
            LABELS[name][arm] = label
            # The recogniser's floor on this set's real clips, measured once here.
            extra = ["--asr-floor"] if arm == ARMS[0] else []
            passed, summary = bench(label, arm, corpus["dir"], corpus["speakers"], *extra)
            facts[label] = {{"held_out": summary.get("held_out_mean"), "cer": summary.get("cer_mean"),
                            "renders": summary.get("renders"), "complete": passed}}
            print(label, json.dumps(facts[label]))
    # Every arm is secondary: one that fails is reported, and the session still
    # keeps what the others produced.
    GATES.record("renders", any(f["complete"] for f in facts.values()), json.dumps(facts))
'''

COLLECT = '''\
destination = Path("/kaggle/working/arms")
destination.mkdir(exist_ok=True)
for path in (REPO / "eval/results").glob("*.json"):
    shutil.copy2(path, destination / path.name)
(destination / "labels.json").write_text(json.dumps(LABELS, indent=2))
(destination / "gates.json").write_text(json.dumps(GATES.summary(), indent=2))
(destination / "runs.json").write_text(json.dumps(RUNS, indent=2))
shutil.copy2("/kaggle/working/environment.json", destination / "environment.json")
renders = Path("/kaggle/working/renders")
renders.mkdir(exist_ok=True)
for name, labels in LABELS.items():
    for key, label in labels.items():
        folder = REPO / "outputs/benchmark" / label if key != "role" else None
        for path in (folder.glob("*.wav") if folder and folder.is_dir() else []):
            shutil.copy2(path, renders / f"{label}__{path.name}")
print("gates:", json.dumps(GATES.summary(), indent=1))
'''


def main() -> int:
    environment = dev.environment(SESSION).replace(
        "import hashlib\n", "import hashlib\nimport time as _clock\nNOTEBOOK_START = _clock.time()\n", 1)
    assert "NOTEBOOK_START" in environment
    cells = [SETUP, environment, probe.RUNNER, DOWNLOAD, test.CORPORA, CONFIGURE, RENDER, COLLECT]
    for index, source in enumerate(cells):
        compile("".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
                f"<cell {index}>", "exec")
    out = HERE / "omnivoice-arms"
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
                "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells, all compile); arms {ARMS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
