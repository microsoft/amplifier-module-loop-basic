"""CI-glue negative controls; these do not change any product test oracle."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

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


@pytest.mark.parametrize("test_class", ["", "TestBudget::"])
@pytest.mark.parametrize("swap_skips", [False, True], ids=["consistent", "skip-swap"])
def test_run_suite_joins_junit_outcome_per_node(tmp_path, monkeypatch, test_class, swap_skips):
    """A count-preserving skip swap must not hide a skipped required case."""
    passed = (
        "tests/test_budget_lifecycle_corrections.py::"
        f"{test_class}test_required_provider_context_estimate_is_message_scale[openai--1]"
    )
    skipped = (
        "tests/test_media_f1_pairing.py::"
        f"{test_class}test_actual_f1_store_checkpoint_and_private_request[preserve--1]"
    )
    collection, execution = evidence()
    collection.update(collected=[passed, skipped])
    execution.update(
        collected=[passed, skipped],
        reports=[
            {"nodeid": nodeid, "when": phase, "outcome": outcome, "wasxfail": None}
            for nodeid, phases in [
                (passed, [("setup", "passed"), ("call", "passed"), ("teardown", "passed")]),
                (skipped, [("setup", "skipped"), ("teardown", "passed")]),
            ]
            for phase, outcome in phases
        ],
    )
    # The typed stream stays valid and unchanged even in the swapped JUnit case.
    assert runner.validate_typed(collection, execution, required=False) == []
    (tmp_path / "full-collection.json").write_text(json.dumps(collection))
    (tmp_path / "full-execution.json").write_text(json.dumps(execution))
    suite = ET.Element("testsuite", tests="2", skipped="1", failures="0", errors="0")
    # JUnit order need not match collection order.
    for nodeid in reversed(collection["collected"]):
        parts = nodeid.split("::")
        classname = parts[0].removesuffix(".py").replace("/", ".")
        if test_class:
            classname += "." + parts[1]
        case = ET.SubElement(suite, "testcase", classname=classname, name=parts[-1])
        if nodeid == (passed if swap_skips else skipped):
            ET.SubElement(case, "skipped", message="optional pairing not selected")
    ET.ElementTree(suite).write(tmp_path / "full.xml")

    class CompletedPytest:
        stdout = ()

        def __init__(self, *args, **kwargs):
            pass

        def wait(self):
            return 0

    # Replay artifacts through the real run_suite validator, without invoking
    # maintained media oracles or pretending that these synthetic cases ran.
    monkeypatch.setattr(runner.subprocess, "Popen", CompletedPytest)
    result = runner.run_suite("full", ("tests",), tmp_path)
    assert result["counts"] == {
        "tests": 2, "failures": 0, "errors": 0, "skipped": 1, "passed": 1,
    }
    assert result["valid"] is not swap_skips, result
    if swap_skips:
        assert "JUnit outcomes do not match typed execution per node" in result["errors"]
    else:
        assert result["errors"] == []