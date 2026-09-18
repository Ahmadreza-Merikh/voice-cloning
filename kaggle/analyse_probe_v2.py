#!/usr/bin/env python3
"""Recompute, from the files the T4 probe of 2026-09-18 kept, where its worker stage stopped.

The run left two independent records of the worker stage:

* ``worker_load.log`` - the runner's tee: every byte its reader thread took
  from the worker's pipe, written *before* the reader passed the chunk on
* the service log - everything the notebook displayed, from Kaggle

If the reader had been blocked reading, the two would end at the same place.
They do not: the tee runs on past what the notebook ever showed, by less
than one 64 KiB read. That chunk is the one the reader was stuck writing to
the display. The numbers in eval/runs/persian-voice-moss15-probe-v2.json come
from this script:

    .venv/bin/python kaggle/analyse_probe_v2.py
"""

from __future__ import annotations

import collections
import gzip
import json
import re
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1] / "eval/runs/evidence/moss15-probe-v2"
PROGRESS = re.compile(rb"(\d+)/2094")


def last_progress(data: bytes) -> int:
    found = PROGRESS.findall(data)
    return int(found[-1]) if found else 0


def main() -> int:
    tee = gzip.decompress((EVIDENCE / "worker_load.log.gz").read_bytes())
    log = json.loads(gzip.decompress((EVIDENCE / "service-log.json.gz").read_bytes()))
    start = next(i for i, e in enumerate(log) if "moss15 [14:45:12]" in e["data"])
    first_report = next(i for i, e in enumerate(log) if "--- group 149" in e["data"])
    stage = log[start:first_report]

    # The service log holds every raw-stdout write twice, about 0.2 s apart,
    # while print() output appears once - presumably ipykernel's descriptor
    # capture echoing to the kernel's own stdout besides feeding the notebook.
    # Count one copy.
    copies = sum(1 for e in stage if e["data"].startswith("Loading weights:   0%|          | 0/2094"))
    # The last update had no newline yet; it reached the log glued to the front
    # of the first monitor report, 48 s later.
    tail = log[first_report]["data"].split("--- group")[0].encode()
    shown = (sum(len(e["data"].encode()) for e in stage) + len(tail)) // copies
    shown_text = b"".join(e["data"].encode() for e in stage) + tail
    shown_up_to = last_progress(shown_text)
    marker = f"{shown_up_to}/2094".encode()
    cut = tee.find(b"]", tee.find(marker)) + 1

    reports = [e for e in log if re.search(r"^\s+151 ", e["data"])]
    states = collections.Counter(tuple(e["data"].split()[1:4]) for e in reports)

    drained_start = next(i for i, e in enumerate(log) if e["data"].startswith("$ ") and "codec_probe.py" in e["data"])
    drained_end = next(i for i, e in enumerate(log) if e["data"].startswith("exit 0 in"))
    drained_raw = gzip.decompress((EVIDENCE / "drained.log.gz").read_bytes())
    per_second = collections.Counter()
    for e in log[drained_start + 1:drained_end]:
        per_second[int(e["time"])] += len(e["data"].encode())

    print(f"worker stage in the service log: {len(stage)} entries, {copies} copies of each raw write")
    print(f"the reader took {len(tee)} bytes (progress {last_progress(tee)} of 2094)")
    print(f"the notebook showed about {shown} bytes per copy (progress {shown_up_to} of 2094); "
          f"that point is byte {cut} of the tee")
    print(f"taken but never shown: {len(tee) - cut} bytes = {(len(tee) - cut) / 65536:.2f} of a 64 KiB read")
    print(f"worker reports: {dict(states)}  (pid, state, wchan, rss KB)")
    print(f"drained codec-only stage: {len(drained_raw)} bytes taken, all shown; "
          f"peak display rate {max(per_second.values()) / 1000:.0f} KB/s")
    print("run length:", round(log[-1]["time"]), "s;", "probe.json:",
          json.loads((EVIDENCE / "probe.json").read_text())["worker_path_drained"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
