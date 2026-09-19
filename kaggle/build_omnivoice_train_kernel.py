#!/usr/bin/env python3
"""Generate the sessions of the full OmniVoice adaptation run on Kaggle T4 pairs.

The recipe is OmniVoice's official one (SOTA_ROADMAP.md section 8): the SDPA
config at k2-fsa/OmniVoice 08be0b4c, 5,000 steps at 8,192 tokens per GPU on
two GPUs, lr 1e-5, bf16. On a T4 pair it needs gradient checkpointing and DDP
bucket views, which probe version 2 showed leave the first loss and the first
step's gradients unchanged (eval/runs/persian-voice-omnivoice-probe-v2.json).
At 13.98 s a step it takes about 19.4 hours, so it runs in sessions of at most
12 hours, each stopping cleanly after an official checkpoint:

``a``  writes and tokenises the whole train and valid split with the official
       script - never a test clip - keeps the tokens as output, then trains
       from the published weights for as long as the session allows
``b``  mounts ``a``'s tokens and latest checkpoint and resumes through the
       official ``resume_from_checkpoint``
``c``  mounts ``a``'s tokens and ``b``'s checkpoint and finishes

Each session keeps its latest full checkpoint (for the next session) and a
float16 export every 1,000 steps (for choosing on the development set), and a
record of every step's time, memory and loss. Nothing here scores anything.

    .venv/bin/python kaggle/build_omnivoice_train_kernel.py a
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/omnivoice-train-a
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("probe", HERE / "build_omnivoice_adapt_kernel.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

USER = probe.USER
DATASET = probe.DATASET
FT = probe.FT
MACHINE_SHAPE = probe.MACHINE_SHAPE
SESSIONS = {"a": None, "b": "a", "c": "b"}
SLUG = "persian-voice-omnivoice-train-{}"

MEMORY_MODE = "checkpointing+bucket-view"
#: A Kaggle session may run 12 hours; the run stops by itself well before.
SESSION_BUDGET_HOURS = 11.25
#: Below this there is no room for even one official checkpoint (500 steps at
#: the probe's 13.98 s, with a margin), so the session keeps what it has.
MINIMUM_TRAINING_HOURS = 2.3
EXPORT_EVERY = 1000

CV_FILES = "data/*"
NEY_FILES = ("data/train-*", "data/validation-*")
#: The download stage's budget. Session a fetches about 21 GiB (base 3.04,
#: tokenizer 0.75, Common Voice 6.94, Neyshekar train and valid 10.22): at even
#: 20 MiB/s that is 18 minutes, so 60 bounds it with room for retries.
FETCH_MINUTES = {"a": 60, "resume": 20}
#: Extra flags for scripts/fetch_inputs.py; decided by the CPU-only fetch probe.
#: Plain HTTP: the CPU-only probe (eval/runs/persian-voice-fetch-probe-v1.json) fetched everything
#: both ways, Xet in 2.6 min and HTTP in 4.5; HTTP's per-request timeout is the documented one.
FETCH_FLAGS: list[str] = ["--no-xet"]


def fetch_plan(session: str) -> dict:
    """What a session downloads: the official SDPA config, the base, and for ``a`` the data."""
    configs = {name: blob for name, blob in probe.CONFIGS.items() if name.endswith("_sdpa.json")}
    plan = {
        "urls": [{"name": Path(name).name, "git_blob_prefix": blob, "path": f"/tmp/adapt/official/{Path(name).name}",
                  "url": f"https://raw.githubusercontent.com/{probe.OMNIVOICE_REPO}/{probe.OMNIVOICE_COMMIT}/{name}"}
                 for name, blob in configs.items()],
        "snapshots": [{"name": "base", "repo": probe.BASE, "revision": probe.BASE_REVISION,
                       "local_dir": "/tmp/omnivoice-base",
                       "sha256": {"audio_tokenizer/model.safetensors": probe.TOKENIZER_SHA256}}],
    }
    if session == "a":
        plan["snapshots"] += [
            {"name": "audio tokenizer", "repo": probe.TOKENIZER, "revision": probe.TOKENIZER_REVISION,
             "local_dir": "/tmp/higgs-audio-v2-tokenizer", "sha256": {"model.safetensors": probe.TOKENIZER_SHA256}},
            {"name": "common voice 26", "repo": probe.CV_REPO, "repo_type": "dataset", "revision": probe.CV_REVISION,
             "allow_patterns": [CV_FILES], "local_dir": "/tmp/cv26"},
            # Neyshekar's test shards are never fetched: its test speakers are the
            # sealed primary test set. Common Voice test clips share shards with
            # training clips, and the data writer refuses them by name.
            {"name": "neyshekar train and valid", "repo": probe.NEY_REPO, "repo_type": "dataset",
             "revision": probe.NEY_REVISION, "allow_patterns": list(NEY_FILES), "local_dir": "/tmp/neyshekar",
             "forbid": ["test-*.parquet"]},
        ]
    return plan


def download(session: str) -> str:
    """The download cell: scripts/fetch_inputs.py under run_bounded, so a stall ends the stage.

    Session a v1 ran these downloads in the notebook's own process with no
    time limit, stalled 138 s in, and held its T4 pair for the rest of the
    session (eval/runs/persian-voice-omnivoice-adapt.json).
    """
    plan = fetch_plan(session)
    config_name = plan["urls"][0]["path"]
    minutes = FETCH_MINUTES["a" if session == "a" else "resume"]
    return f'''\
CV_DIR, NEY_DIR = Path("/tmp/cv26"), Path("/tmp/neyshekar")
ARTIFACTS = {{}}
if not GATES.skip("download", ("runner",)):
    start = time.time()
    plan_path, report_path = RECORDS / "fetch_plan.json", RECORDS / "fetch.json"
    plan_path.write_text({json.dumps(json.dumps(plan))})
    run = run_bounded([sys.executable, str(REPO / "scripts/fetch_inputs.py"), "--plan", str(plan_path),
                       "--report", str(report_path), *{FETCH_FLAGS!r}],
                      timeout={minutes} * 60, log=LOGS / "fetch.log", monitor_every=300)
    RUNS["fetch"] = run.record()
    print(run.summary())
    report = json.loads(report_path.read_text()) if report_path.is_file() else {{}}
    fetched = {{step["name"]: {{k: step.get(k) for k in ("gib", "seconds", "tries", "failed_check")}}
               for step in report.get("steps", [])}}
    if run.ok and report.get("ok") is True:
        ARTIFACTS = {{"base": "/tmp/omnivoice-base", "config": "{config_name}", "fetched": fetched}}
        if Path("/tmp/higgs-audio-v2-tokenizer").is_dir():
            ARTIFACTS["tokenizer"] = "/tmp/higgs-audio-v2-tokenizer"
        GATES.record("download", True, f"{{(time.time() - start) / 60:.1f}} min; {{json.dumps(ARTIFACTS)}}")
    else:
        GATES.record("download", False, f"{{run.summary()}}; fetched so far: {{json.dumps(fetched)}}")
'''


DATA = '''\
EXPECTED = json.loads((MANIFESTS / "adaptation_splits_v1.summary.json").read_text())["splits"]
DATA_FACTS = {}
if not GATES.skip("data", ("download",)):
    ok = True
    for split in ("train", "valid"):
        summary_path = RECORDS / f"data_{split}.json"
        run = run_bounded([sys.executable, str(REPO / "scripts/prepare_adaptation_data.py"),
                           "--splits", str(SPLITS), "--split", split, "--cv-dir", str(CV_DIR),
                           "--neyshekar-dir", str(NEY_DIR), "--out", str(WORK / "data" / split),
                           "--summary", str(summary_path)],
                          timeout=4 * 3600, log=LOGS / f"data_{split}.log", monitor_every=600)
        RUNS[f"data_{split}"] = run.record()
        print(run.summary())
        if not run.ok or not summary_path.is_file():
            ok = False
            continue
        facts = json.loads(summary_path.read_text())
        DATA_FACTS[split] = facts
        want = sum(EXPECTED[source][split]["clips"] for source in EXPECTED if split in EXPECTED[source])
        written = sum(facts["written"].values())
        left_out = sum(facts["left_out"].values())
        # Every clip of the split is either written or left out for a stated
        # reason; more than one in a thousand left out stops the run.
        ok = ok and written + left_out == want and left_out <= want / 1000
        print(f"{split}: {written} written, {left_out} left out, {want} in the split")
    GATES.record("data", ok, json.dumps({k: {"written": v["written"], "hours": v["hours"],
                                              "left_out": v["left_out"]} for k, v in DATA_FACTS.items()}))
'''

TOKENIZE = '''\
TOKENS = Path("/kaggle/working/tokens")
TOKEN_FACTS = {}
if not GATES.skip("tokenize", ("data",)):
    ok = True
    for split in ("train", "valid"):
        start = time.time()
        run = run_bounded([sys.executable, "-m", "omnivoice.scripts.extract_audio_tokens",
                           "--input_jsonl", DATA_FACTS[split]["jsonl"][split],
                           "--tar_output_pattern", str(TOKENS / split / "audios/shard-%06d.tar"),
                           "--jsonl_output_pattern", str(TOKENS / split / "txts/shard-%06d.jsonl"),
                           "--tokenizer_path", ARTIFACTS["tokenizer"], "--nj_per_gpu", "3", "--shuffle", "True"],
                          timeout=8 * 3600, log=LOGS / f"tokenize_{split}.log", monitor_every=900)
        RUNS[f"tokenize_{split}"] = run.record()
        print(run.summary())
        manifest = TOKENS / split / "data.lst"
        errors = TOKENS / split / "errors.jsonl"
        lines = manifest.read_text().splitlines() if manifest.is_file() else []
        samples = sum(int(line.split()[2]) for line in lines)
        seconds = sum(float(line.split()[3]) for line in lines)
        failed = len(errors.read_text().splitlines()) if errors.is_file() else 0
        written = sum(DATA_FACTS[split]["written"].values())
        TOKEN_FACTS[split] = {"shards": len(lines), "samples": samples, "audio_hours": round(seconds / 3600, 3),
                              "failed": failed, "wall_seconds": round(time.time() - start, 1),
                              "realtime_factor": round(seconds / max(time.time() - start, 1e-6), 1)}
        ok = ok and run.ok and samples + failed == written and failed <= written / 1000
    (RECORDS / "tokens.json").write_text(json.dumps(TOKEN_FACTS, indent=2))
    GATES.record("tokenize", ok, json.dumps(TOKEN_FACTS))
'''

MOUNTED = '''\
# The tokens come from session a's output, the checkpoint from the previous
# session's. data.lst names absolute paths from session a, so it is rewritten
# to where the shards are mounted here - the shards themselves are read as they are.
TOKENS = WORK / "tokens"
RESUME = None
if not GATES.skip("mounted", ("download",)):
    try:
        sources = {}
        for split in ("train", "valid"):
            found = sorted(INPUT.rglob(f"tokens/{split}/data.lst"))
            assert found, f"no tokens/{split}/data.lst under /kaggle/input"
            source = found[0].parent
            lines = []
            for line in found[0].read_text().splitlines():
                tar, jsonl, count, seconds = line.split()
                tar = source / "audios" / Path(tar).name
                jsonl = source / "txts" / Path(jsonl).name
                assert tar.is_file() and jsonl.is_file(), (tar, jsonl)
                lines.append(f"{tar} {jsonl} {count} {seconds}")
            (TOKENS / split).mkdir(parents=True, exist_ok=True)
            (TOKENS / split / "data.lst").write_text("\\n".join(lines) + "\\n")
            sources[split] = {"shards": len(lines), "from": str(source)}
        checkpoints = sorted(INPUT.rglob("run/checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        checkpoints = [p for p in checkpoints if p.is_dir() and (p / "optimizer.bin").is_file()]
        assert checkpoints, "no checkpoint with optimizer state under /kaggle/input"
        RESUME = checkpoints[-1]
        GATES.record("mounted", True, json.dumps({**sources, "resume_from": str(RESUME)}))
    except Exception as exc:  # noqa: BLE001 - the gate records it
        GATES.record("mounted", False, f"{type(exc).__name__}: {exc}")
'''


def train(session: str) -> str:
    needs = '("tokenize",)' if session == "a" else '("mounted",)'
    resume = "" if session == "a" else '\n        "--set", f"resume_from_checkpoint={RESUME}",'
    return f'''\
RUN_DIR = Path("/tmp/run")
TRAIN_FACTS = {{}}
if not GATES.skip("train", {needs}):
    remaining = NOTEBOOK_START + {SESSION_BUDGET_HOURS} * 3600 - time.time()
    if remaining < {MINIMUM_TRAINING_HOURS} * 3600:
        GATES.record("train", False, f"not started: {{remaining / 3600:.2f}} h left, less than one checkpoint needs")
    else:
        DATA_CONFIG = WORK / "data_config.json"
        DATA_CONFIG.write_text(json.dumps({{"train": [{{"manifest_path": [str(TOKENS / "train/data.lst")]}}],
                                            "dev": [{{"manifest_path": [str(TOKENS / "valid/data.lst")]}}]}}, indent=2))
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        command = ["accelerate", "launch", "--gpu_ids", "0,1", "--num_processes", "2",
            str(REPO / "scripts/omnivoice_adapt.py"),
            "--train_config", ARTIFACTS["config"], "--data_config", str(DATA_CONFIG), "--output_dir", str(RUN_DIR),
            "--set", f"init_from_checkpoint={{ARTIFACTS['base']}}", "--set", "keep_last_n_checkpoints=1",{resume}
            "--memory", "{MEMORY_MODE}", "--session-seconds", str(int(remaining - 600)),
            "--export-every", "{EXPORT_EVERY}", "--audio-tokenizer", str(Path(ARTIFACTS["base"]) / "audio_tokenizer"),
            "--record", str(RECORDS / "train.json")]
        run = run_bounded(command, timeout=remaining, log=LOGS / "train.log", monitor_every=1800)
        RUNS["train"] = run.record()
        print(run.summary())
        ranks = [json.loads(p.read_text()) for p in sorted(RECORDS.glob("train.rank*.json"))]
        saved = sorted(RUN_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        TRAIN_FACTS = {{"outcome": [r["outcome"] for r in ranks], "global_step": [r["global_step"] for r in ranks],
                       "checkpoints": [p.name for p in saved], "exports": sorted(p.name for p in RUN_DIR.glob("export-*")),
                       "mean_step_seconds": [r["mean_step_seconds_after_two"] for r in ranks],
                       "peak_reserved_gib": [r["peak_reserved_gib"] for r in ranks]}}
        GATES.record("train", run.ok and bool(saved), json.dumps(TRAIN_FACTS))
'''


COLLECT = '''\
# The latest full checkpoint (for the next session) and every float16 export
# (for choosing on the development set) are kept; older checkpoints were
# rotated away by the trainer.
KEEP = Path("/kaggle/working/run")
KEEP.mkdir(exist_ok=True)
saved = sorted(Path("/tmp/run").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])) if Path("/tmp/run").is_dir() else []
if saved:
    shutil.copytree(saved[-1], KEEP / saved[-1].name, dirs_exist_ok=True)
for export in sorted(Path("/tmp/run").glob("export-*")) if Path("/tmp/run").is_dir() else []:
    shutil.copytree(export, KEEP / export.name, dirs_exist_ok=True)
for name in ("train.log", "initial_config.json"):
    if (Path("/tmp/run") / name).is_file():
        shutil.copy2(Path("/tmp/run") / name, KEEP / name)
if (Path("/tmp/run") / "tensorboard").is_dir():
    shutil.copytree(Path("/tmp/run") / "tensorboard", KEEP / "tensorboard", dirs_exist_ok=True)
destination = Path("/kaggle/working/session")
destination.mkdir(exist_ok=True)
(destination / "gates.json").write_text(json.dumps(GATES.summary(), indent=2))
(destination / "runs.json").write_text(json.dumps(RUNS, indent=2))
shutil.copy2("/kaggle/working/environment.json", destination / "environment.json")
for path in RECORDS.glob("*.json"):
    shutil.copy2(path, destination / path.name)
total = sum(p.stat().st_size for p in Path("/kaggle/working").rglob("*") if p.is_file())
print("kept:", [p.name for p in KEEP.iterdir()], f"- {total / 2**30:.1f} GiB in /kaggle/working")
print("gates:", json.dumps(GATES.summary(), indent=1))
'''


def build(session: str) -> Path:
    slug = SLUG.format(session)
    environment = probe.ENVIRONMENT.replace(probe.SESSION, f"kaggle:{slug}-v1")
    environment = environment.replace("import hashlib\n", "import hashlib\nimport time as _clock\nNOTEBOOK_START = _clock.time()\n", 1)
    cells = [probe.SETUP, environment, probe.RUNNER, download(session)]
    cells += [DATA, TOKENIZE] if session == "a" else [MOUNTED]
    cells += [train(session), COLLECT]
    for index, source in enumerate(cells):
        compile("".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
                f"<{slug} cell {index}>", "exec")
    out = HERE / f"omnivoice-train-{session}"
    out.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": [probe.cell(s) for s in cells],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = out / f"{slug}.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    sources = [f"{USER}/{SLUG.format(s)}" for s in ("a", SESSIONS[session]) if s and session != "a"]
    metadata = {"id": f"{USER}/{slug}", "title": slug, "code_file": path.name, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                "machine_shape": MACHINE_SHAPE, "enable_internet": True,
                "dataset_sources": [f"{USER}/{DATASET}"], "competition_sources": [],
                "kernel_sources": sorted(set(sources)), "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells, all compile); sources {metadata['kernel_sources']}")
    return path


def main(argv: list[str]) -> int:
    wanted = argv or list(SESSIONS)
    for session in wanted:
        if session not in SESSIONS:
            raise SystemExit(f"unknown session {session!r}; one of {sorted(SESSIONS)}")
        build(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
