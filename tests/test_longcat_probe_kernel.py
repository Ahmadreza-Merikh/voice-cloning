"""The LongCat probe's notebook runs what was pre-registered, on its own dataset.

A push starts a GPU run, so what the notebook would do is checked here instead:
the pins match the pre-registration, the primary pair renders first, only the
3.5B moves its transformer, the English runs use the English recogniser, and
the committed notebook is what the builder writes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_longcat_probe_kernel",
                                              ROOT / "kaggle" / "build_longcat_probe_kernel.py")
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)

PLAN = json.loads((ROOT / "eval" / "runs" / "persian-voice-longcat-probe.json").read_text(encoding="utf-8"))
NOTEBOOK = ROOT / "kaggle" / "longcat-probe" / "persian-voice-longcat-probe-run.ipynb"


def test_the_pins_are_the_pre_registered_ones() -> None:
    arms = PLAN["arms"]
    for repo, revision, digest in (kernel.LONGCAT_1B, kernel.LONGCAT_3_5B):
        arm = arms["longcat-1b" if repo.endswith("-1B") else "longcat-3.5b"]
        assert revision.startswith(arm["weights"].split(" at ")[1].split(",")[0])
        assert digest in arm["weights"]
    assert kernel.LONGCAT_CODE_COMMIT.startswith("12c76b51")
    assert kernel.UMT5[1].startswith("0de9394d")
    assert kernel.probe.BASE_REVISION.startswith("c5fdb5cc")


def test_the_texts_are_the_pre_registered_ones() -> None:
    texts = json.loads((ROOT / kernel.ENGLISH).read_text(encoding="utf-8"))
    assert PLAN["texts"]["file"] == kernel.ENGLISH
    assert texts["sha256_of_texts"] == PLAN["texts"]["sha256_of_texts"]
    english = [arm for arm in kernel.ARMS if arm[0].endswith("-en")]
    assert all(arm[4] == len(texts["texts"]) for arm in english)


def test_the_primary_pair_renders_first_and_labels_match_the_plan() -> None:
    labels = [arm[0] for arm in kernel.ARMS]
    assert labels[:2] == [PLAN["arms"]["omnivoice"]["label"], PLAN["arms"]["longcat-1b"]["label"]]
    assert labels[2] == PLAN["arms"]["longcat-3.5b"]["label"]
    assert [arm[6] for arm in kernel.ARMS] == ["primary", "primary", "secondary", "persian", "persian"]


def test_only_the_3_5b_moves_its_transformer_and_every_run_sets_its_recogniser() -> None:
    for label, engine, env, extra, *_ in kernel.ARMS:
        moved = env.get("VOICECLONE__ENGINES__LONGCAT__TRANSFORMER_DEVICE")
        assert moved == ("cuda:1" if label == "longcat-3.5b-probe-en" else None)
        assert env["VOICECLONE__ENGINES__ASR__LANGUAGE"] == ("en" if label.endswith("-en") else "fa")
        if engine == "omnivoice" and label.endswith("-en"):
            assert extra[-2:] == ["--engine-option", "language=en"]


def test_the_session_stays_inside_its_declared_bound() -> None:
    assert kernel.SESSION_BUDGET_HOURS == 2.0
    assert "2.0 hours" in PLAN["budget_declared_before_the_push"]["bounds"]
    assert "2.51 left" in PLAN["amendment_2026_09_25_quota"]["why"]


def test_it_mounts_its_own_dataset_and_not_the_adaptations() -> None:
    metadata = json.loads((NOTEBOOK.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_sources"] == ["marsmanam/persian-voice-longcat-probe"]
    assert metadata["machine_shape"] == "NvidiaTeslaT4" and metadata["is_private"] is True


def test_the_committed_notebook_is_what_the_builder_writes() -> None:
    committed = ["".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]]
    built = ["".join(kernel.probe.cell(source)["source"]) for source in kernel.cells()]
    assert committed == built
