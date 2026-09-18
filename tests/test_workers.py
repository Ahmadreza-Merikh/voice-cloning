"""Every out-of-process engine ships a worker that parses and speaks the protocol.

A worker runs in a different interpreter with different dependencies, so nothing
in this package ever imports one. That means a syntax error in a worker is
invisible here and surfaces on a GPU session minutes in, after an install and a
download - which is exactly how one Kaggle run was spent.

The workers cannot be imported to check them: each takes ownership of file
descriptor 1 at import time, which would swallow pytest's own output. So they
are compiled and read as source instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from voiceclone.engines import list_tts_engines

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

#: Engine name -> the worker it starts.
WORKERS = {
    "chatterbox": "chatterbox_worker.py",
    "omnivoice": "omnivoice_worker.py",
    "fish": "fish_worker.py",
    "higgs": "higgs_worker.py",
    "moss15": "moss15_worker.py",
}


@pytest.mark.parametrize("script", sorted(WORKERS.values()))
class TestWorkerScripts:
    def test_it_parses(self, script: str) -> None:
        source = (SCRIPTS / script).read_text(encoding="utf-8")
        compile(source, script, "exec")

    def test_it_speaks_the_protocol(self, script: str) -> None:
        source = (SCRIPTS / script).read_text(encoding="utf-8")
        # One ready line, one result line per job, and a way to be shut down.
        assert '"ready"' in source
        assert '"ok"' in source
        assert '"quit"' in source

    def test_it_takes_the_arguments_its_engine_passes(self, script: str) -> None:
        """The engine builds a command line; argparse has to accept every flag.

        A flag renamed on one side and not the other is a failure that only
        appears when the worker is started for real.
        """
        tree = ast.parse((SCRIPTS / script).read_text(encoding="utf-8"))
        declared = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        }
        engine_name = next(name for name, file in WORKERS.items() if file == script)
        engine = list_tts_engines().get(engine_name)
        if engine is None:  # an engine that is not registered has no command line
            pytest.skip(f"{engine_name} is not registered")
        module = __import__(engine.__module__, fromlist=["_"])
        source = Path(module.__file__).read_text(encoding="utf-8")
        used = {
            line.strip().strip('",')
            for line in source.splitlines()
            if line.strip().startswith('"--')
        }
        assert used, f"{engine_name} passes no flags - has the command line moved?"
        assert used <= declared, f"{engine_name} passes {sorted(used - declared)}"


def test_every_out_of_process_engine_has_its_worker() -> None:
    for name, script in WORKERS.items():
        assert (SCRIPTS / script).is_file(), f"{name} has no worker at {script}"
