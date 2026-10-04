#!/usr/bin/env python3
"""Generate the session that renders the front-end listening test on a T4.

Registered before it ran (``eval/runs/persian-voice-frontend-listening-v1.json``
-> ``amendment_2026_10_03_render_on_kaggle``). The renders were to be made on
the Mac, whose memory guard refused OmniVoice (5.1 GB free, 6 needed); the
user chose Kaggle. Only the machine changes: the same app code
(``scripts/render_frontend_conditions.py`` through the app's Studio and
OmniVoice worker), the same texts, voice, seed and settings, the weights at the
revision the Mac has, float16 as on the Mac - on CUDA instead of Metal. All 48
render in one session, so the four forms of a sentence still differ only in
their text.

One T4 x2 session, built from the OmniVoice sessions' verified cells:

1. OmniVoice's pinned stack and the scoring packages, checked by the same
   environment cell as every OmniVoice session
2. the runner check; the code and the registered texts checked; the base
   weights fetched and their tokenizer's sha256 checked
3. ``render_frontend_conditions.py --cer``: the marks checked through the app's
   text pipeline, 48 renders, then the recogniser alone for the CER harm check
4. the manifest and the wavs copied out

It mounts its own dataset, ``persian-voice-frontend-listening``, so
``persian-voice-eval`` stays the copy the sealed test ran.

    .venv/bin/python kaggle/build_dataset.py --dataset persian-voice-frontend-listening
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle datasets create \\
        -p kaggle/dataset-persian-voice-frontend-listening --dir-mode zip
    .venv/bin/python kaggle/build_frontend_listening_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/frontend-listening
    .venv/bin/python kaggle/fetch_output.py marsmanam/persian-voice-frontend-listening-run \\
        remote/frontend-listening-v1 frontend-listening-v1/ logs/

Step 3 (eval/runs/persian-voice-frontend-listening-v2.json) is the same session
on its own texts and dataset, with the app's normaliser for OmniVoice:

    .venv/bin/python kaggle/build_dataset.py --dataset persian-voice-frontend-listening-v2
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle datasets create \\
        -p kaggle/dataset-persian-voice-frontend-listening-v2 --dir-mode zip
    .venv/bin/python kaggle/build_frontend_listening_kernel.py --step 3
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/frontend-listening-v2
"""

from __future__ import annotations

import argparse
import hashlib
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

#: Kaggle refuses a notebook whose title a dataset already uses, so the session and its dataset differ.
#: Step 2's names are kept as they ran; step 3 runs on its own texts and dataset.
STEPS = {
    2: {"slug": "persian-voice-frontend-listening-run", "dataset": "persian-voice-frontend-listening",
        "name": "frontend-listening-v1", "folder": "frontend-listening", "extra": []},
    3: {"slug": "persian-voice-frontend-listening-v2-run", "dataset": "persian-voice-frontend-listening-v2",
        "name": "frontend-listening-v2", "folder": "frontend-listening-v2",
        # The texts path goes in absolute: the script on the dataset (e1d96e4) stops on a relative one.
        "extra": ["--texts", {"repo": "eval/runs/frontend-listening-v2/texts.json"}, "--app-normaliser"]},
}
EXPECTED = 48
RENDER_MINUTES = 60
BASE_DIR = dev.BASE_PLAN["snapshots"][0]["local_dir"]
NOT_FOUND = "no repo with scripts/omnivoice_adapt.py under /kaggle/input"


def texts_of(step: dict) -> str:
    return f"eval/runs/{step['name']}/texts.json"


def environment(step: dict) -> str:
    find = ('found = sorted(p for p in INPUT.rglob("repo/scripts/render_frontend_conditions.py") '
            f'if "{step["dataset"]}" in p.parts)')
    source = probe.ENVIRONMENT.replace(probe.SESSION, f"kaggle:{step['slug']}-v1")
    for old, new in ((dev.FIND_REPO, find),
                     (NOT_FOUND, NOT_FOUND.replace("omnivoice_adapt.py", "render_frontend_conditions.py"))):
        assert source.count(old) == 1, f"the environment cell changed: {old!r}"
        source = source.replace(old, new)
    return source


def download(step: dict) -> str:
    texts = texts_of(step)
    texts_sha256 = hashlib.sha256((ROOT / texts).read_bytes()).hexdigest()
    needed = ("scripts/render_frontend_conditions.py", "scripts/ab_omnivoice_ipa.py", "scripts/omnivoice_worker.py",
              "src/voiceclone/text/marks_fa.py", "eval/corpus/manifest.json", "eval/corpus/mana_000.wav", texts)
    return dev.FETCH + f'''
os.environ["VOICECLONE_OMNIVOICE_PYTHON"] = sys.executable
provenance = found[0].parents[2] / "PROVENANCE.md"
os.environ["VOICECLONE_CODE_COMMIT"] = (provenance.read_text().split("commit ")[-1].strip(" .\\n")
                                        if provenance.is_file() else "the dataset's PROVENANCE.md was not found")
NEEDED = {list(needed)!r}
missing = [name for name in NEEDED if not (REPO / name).is_file()]
texts = REPO / "{texts}"
if texts.is_file() and hashlib.sha256(texts.read_bytes()).hexdigest() != "{texts_sha256}":
    missing.append("texts.json is not the registered one")
GATES.record("code", not missing, f"missing or changed: {{missing}}" if missing else
             "the render script, the marks and the registered texts are present; code from " +
             os.environ["VOICECLONE_CODE_COMMIT"])
if not GATES.skip("download", ("runner", "code")):
    fetched_ok, fetched = fetch({dev.BASE_PLAN!r}, {dev.FETCH_MINUTES}, {dev.FETCH_FLAGS!r})
    GATES.record("download", fetched_ok, json.dumps(fetched))
# The engine reads its weights from this override; nothing else differs from the Mac's settings.
os.environ["VOICECLONE__ENGINES__OMNIVOICE__MODEL_ID"] = "{BASE_DIR}"
'''

def render(step: dict) -> str:
    command = ", ".join(f"str(REPO / {json.dumps(a['repo'])})" if isinstance(a, dict) else json.dumps(a)
                        for a in ["scripts/render_frontend_conditions.py", "--cer", *step["extra"]])
    return f'''\
OUT = REPO / "eval/runs/{step['name']}"
RESULT = {{}}
if not GATES.skip("renders", ("download",)):
    run = run_bounded([sys.executable, {command}], cwd=REPO,
                      timeout={RENDER_MINUTES} * 60, log=LOGS / "render.log", monitor_every=300)
    RUNS["render"] = run.record()
    print(run.summary())
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {{"renders": [], "events": []}}
    rendered = [r for r in manifest["renders"] if r.get("wav") and Path(r["wav"]).is_file()]
    scored = [r for r in rendered if r.get("cer") is not None]
    text = run.log.read_text(errors="replace") if run.log.is_file() else ""
    by_condition = {{}}
    for r in scored:
        by_condition.setdefault(r["condition"], []).append(r["cer"])
    checked = [e for e in manifest["events"] if e.get("event") == "marks_checked_through_the_text_pipeline"]
    RESULT = {{"rendered": len(rendered), "scored": len(scored),
              "worker": [line.strip() for line in text.splitlines() if "OmniVoice ready" in line][:1],
              "mean_cer": {{c: round(sum(v) / len(v), 4) for c, v in sorted(by_condition.items())}},
              "other_changes_by_the_app": checked[0]["other_changes_by_the_app"] if checked else None}}
    GATES.record("renders", run.ok and len(rendered) == {EXPECTED} and len(scored) == {EXPECTED} and bool(checked),
                 json.dumps(RESULT, ensure_ascii=False))
'''

def collect(step: dict) -> str:
    return '''\
destination = Path("/kaggle/working/__NAME__")
destination.mkdir(exist_ok=True)
for name in ("manifest.json", "texts.json"):
    if (OUT / name).is_file():
        shutil.copy2(OUT / name, destination / name)
if (OUT / "wavs").is_dir():
    shutil.copytree(OUT / "wavs", destination / "wavs", dirs_exist_ok=True)
(destination / "gates.json").write_text(json.dumps(GATES.summary(), indent=2))
(destination / "runs.json").write_text(json.dumps(RUNS, indent=2))
(destination / "result.json").write_text(json.dumps(RESULT, indent=2, ensure_ascii=False))
shutil.copy2("/kaggle/working/environment.json", destination / "environment.json")
print("gates:", json.dumps(GATES.summary(), indent=1))
'''.replace("__NAME__", step["name"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--step", type=int, choices=sorted(STEPS), default=2)
    step = STEPS[parser.parse_args(argv).step]
    cells = [dev.SETUP, environment(step), probe.RUNNER, download(step), render(step), collect(step)]
    for index, source in enumerate(cells):
        compile("".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
                f"<cell {index}>", "exec")
    out = HERE / step["folder"]
    out.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": [probe.cell(s) for s in cells],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = out / f"{step['slug']}.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    metadata = {"id": f"{probe.USER}/{step['slug']}", "title": step["slug"], "code_file": path.name, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                "machine_shape": probe.MACHINE_SHAPE, "enable_internet": True,
                "dataset_sources": [f"{probe.USER}/{step['dataset']}"], "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells, all compile); texts {texts_of(step)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
