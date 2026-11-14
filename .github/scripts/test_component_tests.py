"""CI-glue negative controls; these do not change any product test oracle."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "component_tests", Path(__file__).with_name("component_tests.py"),
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def evidence():
    ids = ["tests/example.py::test_one", "tests/example.py::test_two"]
    collection = {
        "collected": ids, "testscollected": 2, "collectonly": True,
        "exitstatus": 0, "collection_errors": [], "reports": [],
    }
    execution = {
        **deepcopy(collection), "collectonly": False,
        "reports": [
            {"nodeid": nodeid, "when": phase, "outcome": "passed", "wasxfail": None}
            for nodeid in ids for phase in ("setup", "call", "teardown")
        ],
    }
    return collection, execution


def test_normal_pass():
    assert runner.validate_typed(*evidence(), required=False) == []


@pytest.mark.parametrize("mutation", [
    "missing", "extra", "duplicate", "rename", "skip", "error", "teardown",
    "collectonly", "xpass", "collection_error", "collection_duplicate", "no_call",
])
def test_reject_invalid_evidence(mutation):
    collection, execution = evidence()
    if mutation == "missing":
        execution["reports"] = execution["reports"][3:]
    elif mutation == "extra":
        execution["reports"].append({**execution["reports"][0], "nodeid": "extra"})
    elif mutation == "duplicate":
        execution["reports"].append(deepcopy(execution["reports"][0]))
    elif mutation == "rename":
        execution["reports"][0]["nodeid"] = "renamed"
    elif mutation == "skip":
        execution["reports"][1]["outcome"] = "skipped"
    elif mutation == "error":
        execution["reports"][0]["outcome"] = "failed"
    elif mutation == "teardown":
        execution["reports"][2]["outcome"] = "failed"
    elif mutation == "collectonly":
        execution["collectonly"] = True
    elif mutation == "xpass":
        execution["reports"][1]["wasxfail"] = "non-strict unexpected pass"
    elif mutation == "collection_error":
        collection["collection_errors"] = ["collection failed"]
    elif mutation == "collection_duplicate":
        collection["collected"] = [collection["collected"][0]] * 2
    elif mutation == "no_call":
        execution["reports"].pop(1)
    assert runner.validate_typed(collection, execution, required=False)


def test_required_two_instead_of_54_rejected():
    assert runner.validate_typed(*evidence(), required=True)


def test_required_54_renamed_duplicates_rejected():
    collection, execution = evidence()
    collection["collected"] = ["tests/example.py::test_one"] * 54
    collection["testscollected"] = 54
    execution.update(deepcopy(collection), collectonly=False)
    assert runner.validate_typed(collection, execution, required=True)


def test_actual_non_strict_xpass_is_retained_and_rejected(tmp_path):
    import json
    import subprocess
    import sys

    case = tmp_path / "test_xpass.py"
    case.write_text(
        "import pytest\n@pytest.mark.xfail(strict=False)\ndef test_xpass():\n    assert True\n"
    )
    report = tmp_path / "outcomes.json"
    result = subprocess.run([
        sys.executable, str(Path(runner.__file__).resolve()), "--pytest-child",
        str(report), str(case), "-q",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    execution = json.loads(report.read_text())
    assert any(row["wasxfail"] is not None for row in execution["reports"])
    collection = {**deepcopy(execution), "reports": [], "collectonly": True}
    assert runner.validate_typed(collection, execution, required=False)