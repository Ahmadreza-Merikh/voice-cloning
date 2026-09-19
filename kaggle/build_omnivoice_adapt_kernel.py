#!/usr/bin/env python3
"""Generate the probe for adapting OmniVoice to Persian with its official recipe.

This notebook trains nothing that is kept. It answers, on the hardware that
would run the full recipe, the questions the recipe's cost depends on - before
any long run is sent:

1. the runner check: the model-free job that once stalled a worker, on this
   notebook's own display, before anything is downloaded
2. the official runtime (OmniVoice's ``uv.lock`` at 08be0b4c: torch 2.8.0 cu128,
   transformers 5.3.0, accelerate 1.13.0, webdataset 1.0.2, ...), installed and
   checked; ``omnivoice`` itself installed from that commit
3. the pinned artifacts: base weights ``k2-fsa/OmniVoice`` @ c5fdb5cc, the audio
   tokenizer ``eustlb/higgs-audio-v2-tokenizer`` @ 528e871c (sha256 checked
   against the copy bundled with the weights), the official configs checked
   against their git blob ids, and one parquet shard of each data source
4. a slice of the training and valid splits written to disk - never a test
   clip; ``prepare_adaptation_data.py`` refuses them by name
5. the official tokeniser (``omnivoice.scripts.extract_audio_tokens``) on that
   slice, with its throughput measured
6. a few optimizer steps at the full recipe, 8,192 tokens per GPU on both T4s:
   first the default ``flex_attention`` config, then the official SDPA config
   for GPUs older than Ampere. Peak memory per GPU, seconds per step, losses
7. only if the official SDPA config does not fit: the same with gradient
   checkpointing and DDP bucket views - memory techniques, not arithmetic -
   plus a check at a small test-only batch that they leave the first two
   losses unchanged
8. the budget: what 5,000 steps and 128 hours of tokenising would cost, against
   the free quota and the 12-hour session. Nothing long is started here

Every command runs through ``kernel_gates.run_bounded``: its own process group,
its full output in a file, a bounded view in the notebook.

    .venv/bin/python kaggle/build_omnivoice_adapt_kernel.py
    KAGGLE_CONFIG_DIR=~/.kaggle-oauth .venv/bin/kaggle kernels push -p kaggle/omnivoice-probe
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "omnivoice-probe"

USER = "marsmanam"
DATASET = "persian-voice-eval"
SLUG = "persian-voice-omnivoice-probe"
MACHINE_SHAPE = "NvidiaTeslaT4"
SESSION = "kaggle:persian-voice-omnivoice-probe-v2"

FT = "fine" + "tune"
OMNIVOICE_REPO = "k2-fsa/OmniVoice"
OMNIVOICE_COMMIT = "08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
BASE = "k2-fsa/OmniVoice"
BASE_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
TOKENIZER = "eustlb/higgs-audio-v2-tokenizer"
TOKENIZER_REVISION = "528e871c2a26c4f0f7773b9754e2e1acae20899d"
TOKENIZER_SHA256 = "fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d"

#: Git blob ids of the official files at OMNIVOICE_COMMIT, from the tree API.
CONFIGS = {
    f"examples/config/train_config_{FT}.json": "f80f573b158b",
    f"examples/config/train_config_{FT}_sdpa.json": "4b0a543e23be",
    f"examples/config/data_config_{FT}.json": "7b106dd96155",
}

CV_REPO = "pymmdrza/Common-Voice-Speech-26.0-Persian-Clean"
CV_REVISION = "142e9bb40b56d9491faa0ec0987ee8d408bd9c05"
CV_FILES = ("data/train-00000-of-00034.parquet", "data/validation-00000-of-00001.parquet")
NEY_REPO = "shekar-ai/neyshekar"
NEY_REVISION = "30696ada746a66bb3047c984744b5ecfd60fe4e9"
NEY_FILES = ("data/train-00000-of-00022.parquet", "data/validation-00000-of-00004.parquet")

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
#: OmniVoice's uv.lock at OMNIVOICE_COMMIT.
OFFICIAL = {
    "torch": "2.8.0+cu128", "torchaudio": "2.8.0+cu128", "transformers": "5.3.0",
    "accelerate": "1.13.0", "webdataset": "1.0.2", "safetensors": "0.7.0", "tokenizers": "0.22.2",
    "huggingface-hub": "1.7.2", "tensorboardX": "2.6.4", "soundfile": "0.13.1", "librosa": "0.11.0",
    "pydub": "0.25.1", "numpy": "2.4.3", "triton": "3.4.0",
}
#: What the training and tokenising code imports; the run stops if any differs.
PINNED = ("torch", "torchaudio", "transformers", "accelerate", "webdataset", "safetensors",
          "tokenizers", "huggingface-hub")

#: Probe sizes. The slice is enough for batches of the recipe's size; the steps
#: are enough to time a step after two warm-up steps.
SLICE_TRAIN = 700
SLICE_VALID = 80
FLEX_STEPS = 4
SDPA_STEPS = 8
EQUIVALENCE_BATCH_TOKENS = 512
#: Version 1 ran the default flex_attention config out of memory in its first
#: step, and the official docs send GPUs older than Ampere to SDPA; version 2
#: does not repeat it.
RUN_FLEX = False
FULL_STEPS = 5000
TRAIN_HOURS = 127.85
QUOTA_HOURS_PER_WEEK = 30.0
SESSION_HOURS = 12.0

SETUP = f'''\
# OmniVoice's own lockfile, from PyTorch's cu128 index for torch; torchvision at
# the version that belongs to that torch so transformers can import beside it.
!pip -q install "torch==2.8.0" "torchaudio==2.8.0" "torchvision==0.23.0" --index-url {TORCH_INDEX} 2>&1 | tail -3
!pip -q install "transformers==5.3.0" "accelerate==1.13.0" "webdataset==1.0.2" "safetensors==0.7.0" "tokenizers==0.22.2" "huggingface-hub==1.7.2" "tensorboardX==2.6.4" "soundfile==0.13.1" "librosa==0.11.0" "pydub==0.25.1" 2>&1 | tail -3
!pip -q install --no-deps "git+https://github.com/{OMNIVOICE_REPO}@{OMNIVOICE_COMMIT}" 2>&1 | tail -3
'''

ENVIRONMENT = '''\
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def captured(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr).strip()


probe = subprocess.run([sys.executable, "-c", "from transformers import PreTrainedModel"],
                       text=True, capture_output=True)
if probe.returncode != 0 and "torchvision" in probe.stderr:
    print("transformers cannot import beside this torchvision - removing torchvision")
    print(captured([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision"])[1])
    probe = subprocess.run([sys.executable, "-c", "from transformers import PreTrainedModel"],
                           text=True, capture_output=True)
assert probe.returncode == 0, probe.stderr[-2000:]

import torch

OFFICIAL = __OFFICIAL__
PINNED = __PINNED__


def installed(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def matches(have, want):
    if have is None:
        return False
    return have == want or (have.split("+")[0] == want.split("+")[0] and "+" not in want)


stack = {name: {"official": want, "installed": installed(name),
                "matches": matches(installed(name), want), "pinned_here": name in PINNED}
         for name, want in OFFICIAL.items()}
direct = json.loads(importlib.metadata.distribution("omnivoice").read_text("direct_url.json") or "{}")
code, driver = captured(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
ENVIRONMENT_RECORD = {
    "python": platform.python_version(),
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "bf16_supported_including_emulation": torch.cuda.is_bf16_supported(),
    "gpus": [{"name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i))}
             for i in range(torch.cuda.device_count())],
    "nvidia_smi": driver,
    "official_stack": stack,
    "omnivoice": {"version": installed("omnivoice"),
                  "commit": (direct.get("vcs_info") or {}).get("commit_id"), "url": direct.get("url")},
    "other_packages": {name: installed(name) for name in ("torchvision", "ipykernel", "papermill", "scipy")},
}
Path("/kaggle/working/environment.json").write_text(json.dumps(ENVIRONMENT_RECORD, indent=2))
for name, row in stack.items():
    mark = "ok" if row["matches"] else ("DIFFERS" if row["pinned_here"] else "differs - image's version kept")
    print(f"{name:16} official {row['official']:12} installed {str(row['installed']):16} {mark}")
print("omnivoice", ENVIRONMENT_RECORD["omnivoice"])
print("cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version(), "|", driver)
for name in PINNED:
    assert stack[name]["matches"], f"{name} is not the official {OFFICIAL[name]}: {stack[name]['installed']}"
assert ENVIRONMENT_RECORD["omnivoice"]["commit"] == "__COMMIT__", ENVIRONMENT_RECORD["omnivoice"]
assert torch.cuda.device_count() >= 2, "the official recipe trains on two GPUs - use T4 x2"
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"cuda:{i}: {torch.cuda.get_device_name(i)}, capability {torch.cuda.get_device_capability(i)}, "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
print("disk:", f"{shutil.disk_usage('/tmp').free / 2**30:.0f} GiB free under /tmp")

INPUT = Path("/kaggle/input")
found = sorted(INPUT.rglob("repo/scripts/omnivoice_adapt.py"))
if not found:
    for path in sorted(p for p in INPUT.rglob("*") if p.is_dir())[:40]:
        print("   ", path)
    raise SystemExit("no repo with scripts/omnivoice_adapt.py under /kaggle/input")
REPO = Path("/kaggle/working/repo")
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(found[0].parent.parent, REPO)
# Kaggle unpacks the .gz when it builds the dataset; either form is checked
# against the hash the split builder recorded for its decompressed content.
MANIFESTS = REPO / "eval/manifests"
SPLITS = next((p for p in (MANIFESTS / "adaptation_splits_v1.jsonl.gz", MANIFESTS / "adaptation_splits_v1.jsonl")
               if p.is_file()), None)
assert SPLITS is not None, "the dataset predates the adaptation splits"
import gzip  # noqa: E402

content = gzip.open(SPLITS).read() if SPLITS.suffix == ".gz" else SPLITS.read_bytes()
expected = json.loads((MANIFESTS / "adaptation_splits_v1.summary.json").read_text())["records_sha256"]
assert hashlib.sha256(content).hexdigest() == expected, "the split file is not the one the builder wrote"
print("splits:", SPLITS.name, "sha256 matches the summary")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONPATH"] = str(REPO / "src")
os.environ["VOICECLONE_SCORING_ENV"] = "__SESSION__"
sys.path.insert(0, str(REPO / "scripts"))
from kernel_gates import Gates, report_group, run_bounded  # noqa: E402

GATES = Gates()
RUNS = {}
LOGS = Path("/kaggle/working/logs")
LOGS.mkdir(exist_ok=True)
WORK = Path("/tmp/adapt")
WORK.mkdir(parents=True, exist_ok=True)
RECORDS = Path("/kaggle/working/records")
RECORDS.mkdir(exist_ok=True)
'''
ENVIRONMENT = (ENVIRONMENT.replace("__OFFICIAL__", repr(OFFICIAL)).replace("__PINNED__", repr(PINNED))
               .replace("__COMMIT__", OMNIVOICE_COMMIT).replace("__SESSION__", SESSION))

RUNNER = '''\
from runner_check import check, diagnostics  # noqa: E402

ok, reason, run = check(run_bounded, Path("/kaggle/working/runner_check"), REPO / "scripts/moss15_worker.py")
RUNS["runner_check"] = run.record()
print(run.summary())
seen, line = diagnostics(report_group)
print("descriptor report:", "works" if seen else "DOES NOT WORK", "-", line)
GATES.record("runner", ok, reason)
'''

DOWNLOAD = '''\
from huggingface_hub import hf_hub_download, snapshot_download

BLOBS = __CONFIGS__


def blob_id(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\\0" + data).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


ARTIFACTS = {}
if not GATES.skip("download", ("runner",)):
    try:
        start = time.time()
        BASE = Path(snapshot_download("__BASE__", revision="__BASE_REVISION__", local_dir="/tmp/omnivoice-base"))
        TOKENIZER = Path(snapshot_download("__TOKENIZER__", revision="__TOKENIZER_REVISION__",
                                           local_dir="/tmp/higgs-audio-v2-tokenizer"))
        hashes = {"tokenizer": sha256(TOKENIZER / "model.safetensors"),
                  "bundled_with_base": sha256(BASE / "audio_tokenizer/model.safetensors")}
        assert set(hashes.values()) == {"__TOKENIZER_SHA256__"}, hashes
        OFFICIAL_CONFIGS = {}
        for name, blob in BLOBS.items():
            with urllib.request.urlopen(f"https://raw.githubusercontent.com/__REPO__/__COMMIT__/{name}") as response:
                data = response.read()
            assert blob_id(data).startswith(blob), (name, blob_id(data))
            target = WORK / "official" / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            OFFICIAL_CONFIGS[Path(name).name] = target
        CV_DIR, NEY_DIR = Path("/tmp/cv26"), Path("/tmp/neyshekar")
        for repo, revision, files, folder in (("__CV_REPO__", "__CV_REVISION__", __CV_FILES__, CV_DIR),
                                              ("__NEY_REPO__", "__NEY_REVISION__", __NEY_FILES__, NEY_DIR)):
            for name in files:
                hf_hub_download(repo, name, repo_type="dataset", revision=revision, local_dir=str(folder))
        ARTIFACTS = {"base": str(BASE), "tokenizer": str(TOKENIZER), "tokenizer_sha256": hashes,
                     "configs": {k: str(v) for k, v in OFFICIAL_CONFIGS.items()}}
        GATES.record("download", True, f"{(time.time() - start) / 60:.1f} min; tokenizer hashes agree; "
                     f"{len(OFFICIAL_CONFIGS)} official configs match their blob ids")
    except Exception as exc:  # noqa: BLE001 - the gate records it
        GATES.record("download", False, f"{type(exc).__name__}: {exc}")
'''
for key, value in {
    "__CONFIGS__": repr(CONFIGS), "__BASE__": BASE, "__BASE_REVISION__": BASE_REVISION,
    "__TOKENIZER__": TOKENIZER, "__TOKENIZER_REVISION__": TOKENIZER_REVISION,
    "__TOKENIZER_SHA256__": TOKENIZER_SHA256, "__REPO__": OMNIVOICE_REPO, "__COMMIT__": OMNIVOICE_COMMIT,
    "__CV_REPO__": CV_REPO, "__CV_REVISION__": CV_REVISION, "__CV_FILES__": repr(CV_FILES),
    "__NEY_REPO__": NEY_REPO, "__NEY_REVISION__": NEY_REVISION, "__NEY_FILES__": repr(NEY_FILES),
}.items():
    DOWNLOAD = DOWNLOAD.replace(key, value)

DATA = f'''\
SLICE = {{}}
if not GATES.skip("data", ("download",)):
    ok = True
    for split, limit in (("train", {SLICE_TRAIN}), ("valid", {SLICE_VALID})):
        summary_path = RECORDS / f"data_{{split}}.json"
        run = run_bounded([sys.executable, str(REPO / "scripts/prepare_adaptation_data.py"),
                           "--splits", str(SPLITS), "--split", split, "--cv-dir", str(CV_DIR),
                           "--neyshekar-dir", str(NEY_DIR), "--out", str(WORK / "data" / split),
                           "--limit", str(limit), "--summary", str(summary_path)],
                          timeout=1800, log=LOGS / f"data_{{split}}.log")
        RUNS[f"data_{{split}}"] = run.record()
        print(run.summary())
        if not run.ok or not summary_path.is_file():
            ok = False
            continue
        SLICE[split] = json.loads(summary_path.read_text())
        ok = ok and sum(SLICE[split]["written"].values()) > 0 and not any(SLICE[split]["left_out"].values())
    GATES.record("data", ok, json.dumps({{k: v["written"] for k, v in SLICE.items()}}))
'''

TOKENIZE = '''\
TOKENS = WORK / "tokens"
TOKEN_FACTS = {}
if not GATES.skip("tokenize", ("data",)):
    ok = True
    for split in ("train", "valid"):
        source = SLICE[split]["jsonl"][split]
        start = time.time()
        run = run_bounded([sys.executable, "-m", "omnivoice.scripts.extract_audio_tokens",
                           "--input_jsonl", source,
                           "--tar_output_pattern", str(TOKENS / split / "audios/shard-%06d.tar"),
                           "--jsonl_output_pattern", str(TOKENS / split / "txts/shard-%06d.jsonl"),
                           "--tokenizer_path", ARTIFACTS["tokenizer"], "--nj_per_gpu", "3", "--shuffle", "True"],
                          timeout=3600, log=LOGS / f"tokenize_{split}.log")
        RUNS[f"tokenize_{split}"] = run.record()
        print(run.summary())
        manifest = TOKENS / split / "data.lst"
        errors = TOKENS / split / "errors.jsonl"
        lines = manifest.read_text().splitlines() if manifest.is_file() else []
        samples = sum(int(line.split()[2]) for line in lines)
        seconds = sum(float(line.split()[3]) for line in lines)
        failed = len(errors.read_text().splitlines()) if errors.is_file() else 0
        TOKEN_FACTS[split] = {"shards": len(lines), "samples": samples, "audio_seconds": round(seconds, 1),
                              "failed": failed, "wall_seconds": round(time.time() - start, 1),
                              "realtime_factor": round(seconds / max(time.time() - start, 1e-6), 1)}
        ok = ok and run.ok and samples > 0 and failed == 0
    if ok:
        import io
        import tarfile

        import numpy as np
        first = sorted((TOKENS / "train" / "audios").glob("*.tar"))[0]
        with tarfile.open(first) as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith(".npy"))
            tokens = np.load(io.BytesIO(archive.extractfile(member).read()))
        TOKEN_FACTS["example"] = {"member": member.name, "shape": list(tokens.shape), "dtype": str(tokens.dtype),
                                  "max": int(tokens.max()), "min": int(tokens.min())}
        ok = tokens.ndim == 2 and tokens.shape[0] == 8 and tokens.dtype == np.int16 and 0 <= tokens.min() and tokens.max() < 1024
    (RECORDS / "tokens.json").write_text(json.dumps(TOKEN_FACTS, indent=2))
    GATES.record("tokenize", ok, json.dumps(TOKEN_FACTS))
'''

STEPS = f'''\
DATA_CONFIG = WORK / "data_config.json"
STEP_RECORDS = {{}}


def attempt(name, config_path, steps, memory="official", timeout=2400, evaluate=False, extra=(), gpus=2):
    # The allocator setting changes fragmentation, not arithmetic; it goes only
    # with the memory techniques, never with the official attempt.
    saved = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if memory != "official":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    record = RECORDS / f"{{name}}.json"
    # The official launch line from run_{FT}.sh, with the wrapper in place of -m omnivoice.cli.train.
    command = ["accelerate", "launch", "--gpu_ids", "0,1" if gpus == 2 else "0", "--num_processes", str(gpus),
               str(REPO / "scripts/omnivoice_adapt.py"),
               "--train_config", str(config_path), "--data_config", str(DATA_CONFIG),
               "--output_dir", str(WORK / "runs" / name), "--set", f"init_from_checkpoint={{ARTIFACTS['base']}}",
               "--stop-after-steps", str(steps), "--memory", memory, "--record", str(record), *extra]
    if evaluate:
        command.append("--evaluate-once")
    run = run_bounded(command, timeout=timeout, log=LOGS / f"{{name}}.log", monitor_every=300)
    if saved is None:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = saved
    RUNS[name] = run.record()
    print(run.summary())
    ranks = [json.loads(p.read_text()) for p in sorted(RECORDS.glob(f"{{name}}.rank*.json"))]
    text = run.log.read_text(errors="replace") if run.log.is_file() else ""
    tail = text.splitlines()[-25:]
    # Version 1 read only the tail, which holds torchrun's summary, and missed
    # the out-of-memory both processes had recorded.
    oom = (any("out of memory" in (r.get("outcome") or "") for r in ranks)
           or "OutOfMemoryError" in text or "CUDA out of memory" in text)
    STEP_RECORDS[name] = {{"ok": run.ok, "outcome": run.outcome, "oom": oom, "ranks": ranks, "log_tail": tail}}
    return STEP_RECORDS[name]


def summarise(result):
    if not result["ranks"]:
        return f"{{result['outcome']}}; out of memory: {{result['oom']}}; last line: {{(result['log_tail'] or [''])[-1][:300]}}"
    return "; ".join(f"rank {{r['rank']}}: {{r['optimizer_steps']}} steps, {{r['mean_step_seconds_after_two']}} s/step, "
                     f"peak {{r['peak_allocated_gib']}} GiB allocated / {{r['peak_reserved_gib']}} reserved of "
                     f"{{r['capacity_gib']}}, losses {{r['train_losses'][:4]}}" for r in result["ranks"])


if not GATES.skip("flex", ("tokenize",)):
    DATA_CONFIG.write_text(json.dumps({{"train": [{{"manifest_path": [str(TOKENS / "train/data.lst")]}}],
                                        "dev": [{{"manifest_path": [str(TOKENS / "valid/data.lst")]}}]}}, indent=2))
    if {RUN_FLEX}:
        result = attempt("flex", ARTIFACTS["configs"]["train_config_{FT}.json"], {FLEX_STEPS}, timeout=1800)
        GATES.record("flex", result["ok"], summarise(result))
    else:
        GATES.record("flex", False, "not repeated: version 1 ran the default flex_attention config out of "
                     "memory in its first step, and the official docs send GPUs older than Ampere to SDPA")

if not GATES.skip("sdpa", ("tokenize",)):
    result = attempt("sdpa", ARTIFACTS["configs"]["train_config_{FT}_sdpa.json"], {SDPA_STEPS}, evaluate=True)
    GATES.record("sdpa", result["ok"], summarise(result))

MEMORY_MODE = None
MODES = ("checkpointing+bucket-view", "checkpointing+bucket-view+single-tensor")
if GATES.status.get("sdpa", (False, ""))[0]:
    MEMORY_MODE = "official"
elif STEP_RECORDS.get("sdpa", {{}}).get("oom"):
    for mode in MODES:
        name = "sdpa_" + mode.replace("+", "_").replace("-", "_")
        result = attempt(name, ARTIFACTS["configs"]["train_config_finetune_sdpa.json"], {SDPA_STEPS},
                         memory=mode, evaluate=True)
        GATES.record(f"sdpa [{{mode}}]", result["ok"], summarise(result))
        if result["ok"]:
            MEMORY_MODE, MEMORY_RUN = mode, name
            break
        if not result["oom"]:
            break
else:
    GATES.record("sdpa-memory", False, "not tried: the official SDPA attempt failed for another reason than memory")

if MEMORY_MODE not in (None, "official"):
    # One GPU, a test-only batch small enough for the official path: the same
    # data and seed with and without the techniques must give the same first
    # loss and the same gradients before the first update. DDP bucket views
    # do not exist on one process, so they are covered by DDP's own contract.
    small = json.loads(Path(ARTIFACTS["configs"]["train_config_finetune_sdpa.json"]).read_text())
    small["batch_tokens"] = {EQUIVALENCE_BATCH_TOKENS}
    small_path = WORK / "official" / "equivalence_test_only.json"
    small_path.write_text(json.dumps(small, indent=2))
    plain = attempt("equivalence_official", small_path, 1, gpus=1)
    tested = "+".join(t for t in MEMORY_MODE.split("+") if t != "bucket-view")
    tricks = attempt("equivalence_memory", small_path, 1, memory=tested, gpus=1)
    facts = [x["ranks"][0] if x["ranks"] else {{}} for x in (plain, tricks)]
    fingerprints = [f.get("first_step_gradients") or {{}} for f in facts]
    losses = [f.get("train_losses", [None])[:1] for f in facts]
    # The fingerprint is taken before the first update, so an optimizer that
    # then runs out of memory allocating its state has still shown its gradients.
    if not (fingerprints[0] and fingerprints[1]):
        GATES.record("equivalence", False, "not testable on this card: " + "; ".join(
            f"{{label}} {{x['outcome']}} (out of memory: {{x['oom']}})"
            for label, x in (("official", plain), (tested, tricks))))
    else:
        norms = [f.get("norm_after_clipping") for f in fingerprints]
        same = (None not in norms and abs(norms[0] - norms[1]) <= 1e-2 * max(abs(norms[0]), 1e-6)
                and losses[0] and losses[1] and abs(losses[0][0] - losses[1][0]) <= 1e-3 * max(abs(losses[0][0]), 1.0))
        GATES.record("equivalence", bool(same), json.dumps({{"official": {{"loss": losses[0], **fingerprints[0]}},
                                                              tested: {{"loss": losses[1], **fingerprints[1]}}}}))
        if not same:
            MEMORY_MODE = None
'''

BUDGET = f'''\
BUDGET = {{}}
if MEMORY_MODE is None:
    GATES.record("budget", False, "no configuration of the full recipe ran on this hardware")
else:
    name = "sdpa" if MEMORY_MODE == "official" else MEMORY_RUN
    ranks = STEP_RECORDS[name]["ranks"]
    per_step = max(r["mean_step_seconds_after_two"] for r in ranks)
    train_rate = TOKEN_FACTS["train"]["realtime_factor"]
    evaluation = [r.get("evaluation_once") or {{}} for r in ranks]
    eval_seconds = max((e.get("seconds") or 0) for e in evaluation)
    valid_hours_slice = sum(SLICE["valid"]["hours"].values())
    eval_full = eval_seconds * (3.675 / max(valid_hours_slice, 1e-6))
    training_hours = {FULL_STEPS} * per_step / 3600
    tokenise_hours = {TRAIN_HOURS} * 3600 / max(train_rate, 1e-6) / 3600
    evals_hours = 10 * eval_full / 3600
    total = training_hours + tokenise_hours + evals_hours
    BUDGET = {{"memory_mode": MEMORY_MODE, "seconds_per_step": per_step,
              "training_hours": round(training_hours, 2), "tokenising_hours": round(tokenise_hours, 2),
              "evaluation_hours": round(evals_hours, 2), "total_hours": round(total, 2),
              "sessions_of_{int(SESSION_HOURS)}h": int(-(-total // ({SESSION_HOURS} - 1))),
              "weekly_quota_hours": {QUOTA_HOURS_PER_WEEK},
              "peaks_gib": {{f"rank {{r['rank']}}": r["peak_reserved_gib"] for r in ranks}},
              "capacity_gib": ranks[0]["capacity_gib"]}}
    fits = total <= 0.85 * {QUOTA_HOURS_PER_WEEK}
    GATES.record("budget", fits, json.dumps(BUDGET))
(RECORDS / "budget.json").write_text(json.dumps(BUDGET, indent=2))
'''

COLLECT = '''\
destination = Path("/kaggle/working/probe")
destination.mkdir(exist_ok=True)
(destination / "gates.json").write_text(json.dumps(GATES.summary(), indent=2))
(destination / "runs.json").write_text(json.dumps(RUNS, indent=2))
(destination / "steps.json").write_text(json.dumps(STEP_RECORDS, indent=2))
(destination / "artifacts.json").write_text(json.dumps(ARTIFACTS, indent=2))
shutil.copy2("/kaggle/working/environment.json", destination / "environment.json")
for path in RECORDS.glob("*.json"):
    shutil.copy2(path, destination / path.name)
print("gates:", json.dumps(GATES.summary(), indent=1))
'''


def cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def main() -> int:
    sources = [SETUP, ENVIRONMENT, RUNNER, DOWNLOAD, DATA, TOKENIZE, STEPS, BUDGET, COLLECT]
    for index, source in enumerate(sources):
        compile("".join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith("!")),
                f"<cell {index}>", "exec")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": [cell(s) for s in sources],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = OUT_DIR / f"{SLUG}.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    metadata = {"id": f"{USER}/{SLUG}", "title": SLUG, "code_file": path.name, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                "machine_shape": MACHINE_SHAPE, "enable_internet": True,
                "dataset_sources": [f"{USER}/{DATASET}"], "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(sources)} cells, all compile)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
