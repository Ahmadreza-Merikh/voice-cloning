"""Everything an OmniVoice session runs must be in the dataset it mounts.

The kernels cannot clone this repository; they run what ``kaggle/build_dataset.py``
stages. A script a notebook calls, or a script that one imports, missing from
that list fails on Kaggle - for the sealed test session, only after its
downloads have spent part of the session.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
spec = importlib.util.spec_from_file_location("build_dataset_under_test", ROOT / "kaggle" / "build_dataset.py")
build_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_dataset)

NOTEBOOKS = sorted(p for pattern in ("omnivoice-*", "longcat-*") for p in (ROOT / "kaggle").glob(f"{pattern}/*.ipynb"))


def local(name: str) -> bool:
    return (SCRIPTS / f"{name}.py").is_file()


def called(source: str) -> set[str]:
    names = set(re.findall(r"scripts/(\w+)\.py", source))
    names |= {name for name in re.findall(r"^\s*(?:import|from)\s+(\w+)", source, re.MULTILINE) if local(name)}
    return names


def imports_of(name: str) -> set[str]:
    tree = ast.parse((SCRIPTS / f"{name}.py").read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return {name for name in found if local(name)}


def closure(names: set[str]) -> set[str]:
    seen, queue = set(), list(names)
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        queue.extend(imports_of(name) - seen)
    return seen


def test_the_omnivoice_notebooks_are_all_checked() -> None:
    assert {p.parent.name for p in NOTEBOOKS} >= {"omnivoice-probe", "omnivoice-train-a", "omnivoice-dev",
                                                   "omnivoice-test"}


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.parent.name)
def test_every_script_a_session_runs_is_staged(notebook: Path) -> None:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    source = "".join("".join(cell["source"]) for cell in cells)
    needed = closure(called(source))
    staged = {Path(name).stem for name in build_dataset.SCRIPTS}
    assert needed, "no script found - the pattern no longer matches the notebooks"
    assert needed <= staged, f"not in kaggle/build_dataset.py SCRIPTS: {sorted(needed - staged)}"
