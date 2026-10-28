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
    # The روی check (eval/runs/persian-voice-rui-check-v1.json) runs on step 3's code dataset with its
    # texts written by the notebook itself, so nothing has to be uploaded from the Mac but the notebook.
    4: {"slug": "persian-voice-rui-check-run", "dataset": "persian-voice-frontend-listening-v2",
        "name": "rui-check-v1", "folder": "rui-check", "expected": 20, "embed": True,
        "extra": ["--texts", {"repo": "eval/runs/rui-check-v1/texts.json"}, "--app-normaliser"]},
    # The adapted reading check (eval/runs/persian-voice-adapted-reading-v1.json): the same text read by
    # three sets of weights, one render run each, the exports mounted from their training sessions.
    5: {"slug": "persian-voice-adapted-reading-run", "dataset": "persian-voice-frontend-listening-v2",
        "name": "adapted-reading-v1", "folder": "adapted-reading", "expected": 12, "embed": True,
        "texts": [f"eval/runs/adapted-reading-v1/{form}/texts.json" for form in ("P", "A1", "A5")],
        "models": {"P": None, "A1": "export-1000-float16", "A5": "export-5000-float16"},
        "known_sha256": {"A1": "9c453a74e9eab8324c27e039b9bdc052cd8d6721cecbb20df53c919a0e6ba39a"},
        "kernel_sources": [f"{probe.USER}/persian-voice-omnivoice-train-a",
                           f"{probe.USER}/persian-voice-omnivoice-train-c"]},
    # The news vowels check (eval/runs/persian-voice-news-vowels-v1.json): two texts under each of two sets
    # of weights, one render run per set.
    6: {"slug": "persian-voice-news-vowels-run", "dataset": "persian-voice-frontend-listening-v2",
        "name": "news-vowels-v1", "folder": "news-vowels", "expected": 16, "embed": True,
        "texts": [f"eval/runs/news-vowels-v1/{model}/texts.json" for model in ("P", "A")],
        "models": {"P": None, "A": "export-1000-float16"},
        "known_sha256": {"A": "9c453a74e9eab8324c27e039b9bdc052cd8d6721cecbb20df53c919a0e6ba39a"},
        "kernel_sources": [f"{probe.USER}/persian-voice-omnivoice-train-a"]},
    # The XTTS check (eval/runs/persian-voice-xtts-check-v1.json): the app's form only; XTTS renders on the Mac.
    7: {"slug": "persian-voice-xtts-check-run", "dataset": "persian-voice-frontend-listening-v2",
        "name": "xtts-check-v1", "folder": "xtts-check", "expected": 10, "embed": True,
        "texts": ["eval/runs/xtts-check-v1/A/texts.json"],
        "models": {"A": "export-1000-float16"},
        "known_sha256": {"A": "9c453a74e9eab8324c27e039b9bdc052cd8d6721cecbb20df53c919a0e6ba39a"},
        "kernel_sources": [f"{probe.USER}/persian-voice-omnivoice-train-a"]},
    # The ezafe check (eval/runs/persian-voice-omnivoice-ezafe-train.json): the same text read by the app's weights
    # and two exports of the ezafe-marked run. A1 and E1 share an export's name, so each is looked for only
    # under the session that made it.
    8: {"slug": "persian-voice-ezafe-check-run", "dataset": "persian-voice-frontend-listening-v2",
        "name": "ezafe-check-v1", "folder": "ezafe-check", "expected": 12, "embed": True,
        "texts": [f"eval/runs/ezafe-check-v1/{form}/texts.json" for form in ("A1", "E1", "E5")],
        "models": {"A1": "export-1000-float16", "E1": "export-1000-float16", "E5": "export-5000-float16"},
        "sessions": {"A1": "persian-voice-omnivoice-train-a", "E1": "persian-voice-omnivoice-ezafe-a",
                     "E5": "persian-voice-omnivoice-ezafe-c"},
        "known_sha256": {"A1": "9c453a74e9eab8324c27e039b9bdc052cd8d6721cecbb20df53c919a0e6ba39a"},
        "kernel_sources": [f"{probe.USER}/persian-voice-omnivoice-train-a",
                           f"{probe.USER}/persian-voice-omnivoice-ezafe-a",
                           f"{probe.USER}/persian-voice-omnivoice-ezafe-c"]},
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


def texts_files(step: dict) -> list[str]:
    return step.get("texts") or [texts_of(step)]


def registered_check(files: list[str]) -> str:
    if len(files) == 1:
        texts_sha256 = hashlib.sha256((ROOT / files[0]).read_bytes()).hexdigest()
        return f'''texts = REPO / "{files[0]}"
if texts.is_file() and hashlib.sha256(texts.read_bytes()).hexdigest() != "{texts_sha256}":
    missing.append("texts.json is not the registered one")'''
    registered = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    return f'''REGISTERED = {registered!r}
for name, digest in REGISTERED.items():
    if (REPO / name).is_file() and hashlib.sha256((REPO / name).read_bytes()).hexdigest() != digest:
        missing.append(f"{{name}} is not the registered one")'''


def weights(step: dict) -> str:
    """Each form's weights: the base as fetched, each export found once under the mounted sessions and hashed."""
    sessions, within = "", ""
    if step.get("sessions"):
        sessions = f'\nSESSIONS = {step["sessions"]!r}'
        within = " and SESSIONS[form] in p.parts"
    return f'''
MODELS = {step["models"]!r}{sessions}
KNOWN_SHA256 = {step["known_sha256"]!r}
WEIGHTS = {{}}
if not GATES.skip("weights", ("download",)):
    problems = []
    for form, name in MODELS.items():
        if name is None:
            WEIGHTS[form] = {{"path": "{BASE_DIR}"}}
            continue
        hits = sorted(p for p in INPUT.rglob(name) if p.is_dir() and (p / "model.safetensors").is_file(){within})
        if not hits:
            problems.append(f"{{form}}: no {{name}} with a model.safetensors under /kaggle/input")
            continue
        digest = hashlib.sha256()
        with open(hits[0] / "model.safetensors", "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        WEIGHTS[form] = {{"path": str(hits[0]), "found": [str(p) for p in hits], "model_sha256": digest.hexdigest()}}
        if form in KNOWN_SHA256 and digest.hexdigest() != KNOWN_SHA256[form]:
            problems.append(f"{{form}}: model.safetensors is not the one recorded before")
    GATES.record("weights", not problems, json.dumps({{"weights": WEIGHTS, "problems": problems}}))
'''


def download(step: dict) -> str:
    files = texts_files(step)
    needed = ("scripts/render_frontend_conditions.py", "scripts/ab_omnivoice_ipa.py", "scripts/omnivoice_worker.py",
              "src/voiceclone/text/marks_fa.py", "eval/corpus/manifest.json", "eval/corpus/mana_000.wav", *files)
    embed = ""
    if step.get("embed"):
        for texts in files:
            content = (ROOT / texts).read_text(encoding="utf-8")
            embed += (f'\n(REPO / "{texts}").parent.mkdir(parents=True, exist_ok=True)\n'
                      f'(REPO / "{texts}").write_text({content!r}, encoding="utf-8")')
    return dev.FETCH + embed + f'''
os.environ["VOICECLONE_OMNIVOICE_PYTHON"] = sys.executable
provenance = found[0].parents[2] / "PROVENANCE.md"
os.environ["VOICECLONE_CODE_COMMIT"] = (provenance.read_text().split("commit ")[-1].strip(" .\\n")
                                        if provenance.is_file() else "the dataset's PROVENANCE.md was not found")
NEEDED = {list(needed)!r}
missing = [name for name in NEEDED if not (REPO / name).is_file()]
{registered_check(files)}
GATES.record("code", not missing, f"missing or changed: {{missing}}" if missing else
             "the render script, the marks and the registered texts are present; code from " +
             os.environ["VOICECLONE_CODE_COMMIT"])
if not GATES.skip("download", ("runner", "code")):
    fetched_ok, fetched = fetch({dev.BASE_PLAN!r}, {dev.FETCH_MINUTES}, {dev.FETCH_FLAGS!r})
    GATES.record("download", fetched_ok, json.dumps(fetched))
# The engine reads its weights from this override; nothing else differs from the Mac's settings.
os.environ["VOICECLONE__ENGINES__OMNIVOICE__MODEL_ID"] = "{BASE_DIR}"
''' + (weights(step) if step.get("models") else "")

def render_models(step: dict) -> str:
    """One render run per form with its own weights; each gate checks the weights it rendered with."""
    expected = step["expected"]
    return f'''\
OUT = REPO / "eval/runs/{step['name']}"
RESULT = {{}}
for form in MODELS:
    gate = f"renders-{{form}}"
    if GATES.skip(gate, ("weights",)):
        continue
    os.environ["VOICECLONE__ENGINES__OMNIVOICE__MODEL_ID"] = WEIGHTS[form]["path"]
    run = run_bounded([sys.executable, str(REPO / "scripts/render_frontend_conditions.py"), "--cer", "--texts",
                       str(OUT / form / "texts.json"), "--app-normaliser"], cwd=REPO,
                      timeout={RENDER_MINUTES} * 60, log=LOGS / f"render-{{form}}.log", monitor_every=300)
    RUNS[gate] = run.record()
    print(run.summary())
    manifest_path = OUT / form / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {{"renders": [], "events": []}}
    rendered = [r for r in manifest["renders"] if r.get("wav") and Path(r["wav"]).is_file()]
    scored = [r for r in rendered if r.get("cer") is not None]
    text = run.log.read_text(errors="replace") if run.log.is_file() else ""
    checked = [e for e in manifest["events"] if e.get("event") == "marks_checked_through_the_text_pipeline"]
    used = manifest.get("settings", {{}}).get("omnivoice_model_id")
    RESULT[form] = {{"rendered": len(rendered), "scored": len(scored), "weights": used,
                    "worker": [line.strip() for line in text.splitlines() if "OmniVoice ready" in line][:1],
                    "mean_cer": round(sum(r["cer"] for r in scored) / len(scored), 4) if scored else None,
                    "other_changes_by_the_app": checked[0]["other_changes_by_the_app"] if checked else None}}
    GATES.record(gate, run.ok and len(rendered) == {expected} and len(scored) == {expected} and bool(checked)
                 and used == WEIGHTS[form]["path"], json.dumps(RESULT[form], ensure_ascii=False))
os.environ["VOICECLONE__ENGINES__OMNIVOICE__MODEL_ID"] = "{BASE_DIR}"
'''


def render(step: dict) -> str:
    if step.get("models"):
        return render_models(step)
    command = ", ".join(f"str(REPO / {json.dumps(a['repo'])})" if isinstance(a, dict) else json.dumps(a)
                        for a in ["scripts/render_frontend_conditions.py", "--cer", *step["extra"]])
    expected = step.get("expected", EXPECTED)
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
    GATES.record("renders", run.ok and len(rendered) == {expected} and len(scored) == {expected} and bool(checked),
                 json.dumps(RESULT, ensure_ascii=False))
'''

def collect_models(step: dict) -> str:
    return '''\
destination = Path("/kaggle/working/__NAME__")
destination.mkdir(exist_ok=True)
for form in MODELS:
    (destination / form).mkdir(exist_ok=True)
    for name in ("manifest.json", "texts.json"):
        if (OUT / form / name).is_file():
            shutil.copy2(OUT / form / name, destination / form / name)
    if (OUT / form / "wavs").is_dir():
        shutil.copytree(OUT / form / "wavs", destination / form / "wavs", dirs_exist_ok=True)
(destination / "gates.json").write_text(json.dumps(GATES.summary(), indent=2))
(destination / "runs.json").write_text(json.dumps(RUNS, indent=2))
(destination / "result.json").write_text(json.dumps({"renders": RESULT, "weights": WEIGHTS}, indent=2,
                                                    ensure_ascii=False))
shutil.copy2("/kaggle/working/environment.json", destination / "environment.json")
print("gates:", json.dumps(GATES.summary(), indent=1))
'''.replace("__NAME__", step["name"])


def collect(step: dict) -> str:
    if step.get("models"):
        return collect_models(step)
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
                "kernel_sources": step.get("kernel_sources", []), "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells, all compile); texts {', '.join(texts_files(step))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
