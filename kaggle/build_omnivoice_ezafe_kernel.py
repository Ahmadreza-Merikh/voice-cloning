#!/usr/bin/env python3
"""Generate the sessions of the ezafe-marked adaptation on Kaggle T4 pairs.

Registered before any session (eval/runs/persian-voice-omnivoice-ezafe-train.json).
It is the adaptation run (build_omnivoice_train_kernel.py) with one thing
changed: the text of every training and validation row. The audio tokens are
session train-a's, mounted and read as they are; each shard's transcript file
is rewritten through the registered map of original transcript to transcript
with the app's ezafe written in (eval/manifests/adaptation_ezafe_texts_v1.jsonl.gz,
uploaded as the dataset persian-voice-ezafe-texts), and the session stops if
any row's text is missing from the map or the map is not the registered one.
The recipe, code dataset, memory mode, budget, checkpoints and exports are the
adaptation's own cells.

``a``  trains from the published weights
``b``  resumes from ``a``'s latest checkpoint
``c``  resumes from ``b``'s, only if ``b`` cannot finish the 5,000 steps

    .venv/bin/python kaggle/build_omnivoice_ezafe_kernel.py a
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/omnivoice-ezafe-a
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


train = module("train", "build_omnivoice_train_kernel.py")
dev = module("dev", "build_omnivoice_dev_kernel.py")
probe = train.probe

SLUG = "persian-voice-omnivoice-ezafe-{}"
SESSIONS = {"a": None, "b": "a", "c": "b"}
TEXTS_DATASET = "persian-voice-ezafe-texts"
TEXTS_MAP = ROOT / "eval" / "manifests" / "adaptation_ezafe_texts_v1.jsonl.gz"
TOKENS_FROM = "persian-voice-omnivoice-train-a"


def mounted(session: str, map_sha256: str) -> str:
    previous = SESSIONS[session]
    resume = "" if previous is None else f'''
        checkpoints = sorted((p for p in INPUT.rglob("run/checkpoint-*")
                              if "{SLUG.format(previous)}" in str(p) and p.is_dir() and (p / "optimizer.bin").is_file()),
                             key=lambda p: int(p.name.split("-")[-1]))
        assert checkpoints, "no checkpoint with optimizer state from {SLUG.format(previous)} under /kaggle/input"
        RESUME = checkpoints[-1]'''
    return f'''\
# The tokens are session train-a's, read unchanged. Only each row's text is
# replaced, through the registered map of original transcript to transcript
# with the app's ezafe written in; a row whose text is not in the map stops the session.
TOKENS = WORK / "tokens"
RESUME = None
if not GATES.skip("mounted", ("download",)):
    try:
        import gzip
        maps = sorted(INPUT.rglob("adaptation_ezafe_texts_v1.jsonl.gz"))
        assert maps, "no adaptation_ezafe_texts_v1.jsonl.gz under /kaggle/input"
        assert hashlib.sha256(maps[0].read_bytes()).hexdigest() == "{map_sha256}", "the transcript map is not the registered one"
        with gzip.open(maps[0], "rt", encoding="utf-8") as handle:
            EZAFE = {{row["text"]: row["ezafe"] for row in map(json.loads, handle)}}
        sources = {{}}
        for split in ("train", "valid"):
            found = sorted(p for p in INPUT.rglob(f"tokens/{{split}}/data.lst") if "{TOKENS_FROM}" in str(p))
            assert found, f"no {TOKENS_FROM} tokens/{{split}}/data.lst under /kaggle/input"
            source = found[0].parent
            (TOKENS / split / "txts").mkdir(parents=True, exist_ok=True)
            lines, rows, marked = [], 0, 0
            for line in found[0].read_text().splitlines():
                tar, jsonl, count, seconds = line.split()
                tar, original = source / "audios" / Path(tar).name, source / "txts" / Path(jsonl).name
                assert tar.is_file() and original.is_file(), (tar, original)
                out = []
                for raw in original.read_text(encoding="utf-8").splitlines():
                    row = json.loads(raw)
                    text = EZAFE[row["text"]]
                    marked += text != row["text"]
                    rows += 1
                    row["text"] = text
                    out.append(json.dumps(row, ensure_ascii=False))
                rewritten = TOKENS / split / "txts" / original.name
                rewritten.write_text("\\n".join(out) + "\\n", encoding="utf-8")
                lines.append(f"{{tar}} {{rewritten}} {{count}} {{seconds}}")
            (TOKENS / split / "data.lst").write_text("\\n".join(lines) + "\\n")
            sources[split] = {{"shards": len(lines), "rows": rows, "rows_with_an_ezafe": marked, "tokens_from": str(source)}}{resume}
        GATES.record("mounted", True, json.dumps({{**sources, "resume_from": str(RESUME)}}))
    except Exception as exc:  # noqa: BLE001 - the gate records it
        GATES.record("mounted", False, f"{{type(exc).__name__}}: {{exc}}")
'''


def train_cell(session: str, hours: float) -> str:
    if SESSIONS[session] is not None:
        return train.train("b", hours)
    source = train.train("a", hours)
    assert source.count('("tokenize",)') == 1, "the adaptation's train cell changed"
    return source.replace('("tokenize",)', '("mounted",)')


def build(session: str, hours: float = train.SESSION_BUDGET_HOURS) -> Path:
    import hashlib

    slug = SLUG.format(session)
    map_sha256 = hashlib.sha256(TEXTS_MAP.read_bytes()).hexdigest()
    environment = probe.ENVIRONMENT.replace(probe.SESSION, f"kaggle:{slug}-v1")
    assert environment.count(dev.FIND_REPO) == 1, "the probe's repository lookup changed"
    environment = environment.replace(dev.FIND_REPO, dev.FIND_REPO_IN_DATASET)
    environment = environment.replace("import hashlib\n", "import hashlib\nimport time as _clock\nNOTEBOOK_START = _clock.time()\n", 1)
    cells = [probe.SETUP, environment, probe.RUNNER, train.download("b"), mounted(session, map_sha256),
             train_cell(session, hours), train.COLLECT]
    for index, source in enumerate(cells):
        compile("".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
                f"<{slug} cell {index}>", "exec")
    out = HERE / f"omnivoice-ezafe-{session}"
    out.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": [probe.cell(s) for s in cells],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = out / f"{slug}.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    sources = [f"{probe.USER}/{TOKENS_FROM}"] + ([f"{probe.USER}/{SLUG.format(SESSIONS[session])}"]
                                                if SESSIONS[session] else [])
    metadata = {"id": f"{probe.USER}/{slug}", "title": slug, "code_file": path.name, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                "machine_shape": probe.MACHINE_SHAPE, "enable_internet": True,
                "dataset_sources": [f"{probe.USER}/{probe.DATASET}", f"{probe.USER}/{TEXTS_DATASET}"],
                "competition_sources": [], "kernel_sources": sources, "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells, all compile); {hours} h budget; sources {sources}; map {map_sha256[:12]}")
    return path


def main(argv: list[str]) -> int:
    hours = train.SESSION_BUDGET_HOURS
    rest = [arg for arg in argv if not arg.startswith("--hours=")]
    for arg in argv:
        if arg.startswith("--hours="):
            hours = float(arg.split("=", 1)[1])
    if hours > train.SESSION_BUDGET_HOURS:
        raise SystemExit(f"{hours} h is over a Kaggle session's own limit")
    for session in rest or ["a"]:
        build(session, hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
