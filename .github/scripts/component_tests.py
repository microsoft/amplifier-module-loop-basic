"""Run component checks; never claim Foundation/host/media-graph qualification."""

import hashlib
from collections import Counter, defaultdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


REQUIRED = (
    "tests/test_budget_lifecycle_corrections.py",
    "tests/test_final_request_budget.py",
)
# Exact 54-node inventory at reviewed 9b43be50, sorted and joined with LF, no
# trailing LF. Updating this guard requires reviewing the changed test inventory.
REQUIRED_COUNT = 54
REQUIRED_SHA256 = "3c55683ecf677ec32817366c0aff75c210434c9f1d59b49841bec0d738311589"
OPTIONAL = {
    "tests/test_media_f1_pairing.py",
    "tests/test_media_positive_graph.py",
}


class Recorder:
    """Retain typed pytest outcomes, including non-strict XPASS hidden by JUnit."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = {"collected": [], "collection_errors": [], "reports": []}

    def pytest_collection_finish(self, session):
        self.data["collected"] = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.data["collection_errors"].append({
                "nodeid": report.nodeid, "outcome": report.outcome,
                "detail": str(report.longrepr),
            })

    def pytest_runtest_logreport(self, report):
        self.data["reports"].append({
            "nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
            "wasxfail": getattr(report, "wasxfail", None),
            "detail": str(report.longrepr) if report.longrepr else None,
        })

    def pytest_sessionfinish(self, session, exitstatus):
        self.data.update({
            "exitstatus": int(exitstatus), "collectonly": session.config.option.collectonly,
            "testscollected": session.testscollected,
        })
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")


def validate_typed(collection, execution, required):
    """Require exact identities and complete per-node setup/call/teardown."""
    expected = collection["collected"]
    errors = []
    if (
        collection["exitstatus"] != 0 or not collection["collectonly"]
        or collection["collection_errors"] or not expected
        or len(set(expected)) != len(expected)
        or collection["testscollected"] != len(expected)
    ):
        errors.append("invalid collection inventory")
    digest = hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest()
    if required and (len(expected) != REQUIRED_COUNT or digest != REQUIRED_SHA256):
        errors.append("required inventory changed from reviewed 54-node contract")
    if (
        execution["exitstatus"] != 0 or execution["collectonly"]
        or execution["collection_errors"]
        or execution["collected"] != expected
        or execution["testscollected"] != len(expected)
    ):
        errors.append("execution/collection mismatch or pytest failure")
    phases = defaultdict(list)
    for report in execution["reports"]:
        phases[report["nodeid"]].append(report)
    if set(phases) != set(expected):
        errors.append("missing/extra execution node identities")
    for nodeid, reports in phases.items():
        if any(report["wasxfail"] is not None for report in reports):
            errors.append(f"xfail/XPASS is not component pass evidence: {nodeid}")
        outcomes = [(row["when"], row["outcome"]) for row in reports]
        passed = [("setup", "passed"), ("call", "passed"), ("teardown", "passed")]
        optional_skip = (
            not required and nodeid.split("::", 1)[0] in OPTIONAL
            and outcomes == [("setup", "skipped"), ("teardown", "passed")]
        )
        if outcomes != passed and not optional_skip:
            errors.append(f"incomplete/failed/duplicate/unexpected skipped phases: {nodeid}")
    return errors


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def run_suite(name, selectors, destination):
    """Preserve pytest's actual exit code, console log and per-case JUnit."""
    junit = destination / f"{name}.xml"
    def command_for(report, *extra):
        return [
            sys.executable, str(Path(__file__).resolve()), "--pytest-child",
            str(report), *selectors, *extra,
        ]

    collection_path = destination / f"{name}-collection.json"
    execution_path = destination / f"{name}-execution.json"
    collect_command = command_for(collection_path, "--collect-only", "-q")
    command = command_for(
        execution_path, "-ra",
        f"--junitxml={junit}",
    )
    returncodes = []
    for label, invocation in (("collection", collect_command), ("execution", command)):
        with (destination / f"{name}-{label}.log").open("w") as log:
            log.write(f"command: {json.dumps(invocation)}\n")
            log.flush()
            process = subprocess.Popen(
                invocation, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            returncodes.append(process.wait())
    result = {"command": command, "returncode": returncodes[-1], "returncodes": returncodes}
    if not all(path.is_file() for path in (junit, collection_path, execution_path)):
        return {**result, "valid": False, "reason": "JUnit/typed report missing"}
    collection = json.loads(collection_path.read_text())
    execution = json.loads(execution_path.read_text())
    errors = validate_typed(collection, execution, name == "required")
    if any(returncodes):
        errors.append("pytest subprocess failed")
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    counts = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
    skips = []
    for case in cases:
        for outcome in ("failure", "error", "skipped"):
            node = case.find(outcome)
            if node is not None:
                counts[{"failure": "failures", "error": "errors"}.get(outcome, outcome)] += 1
                if outcome == "skipped":
                    skips.append({
                        "class": case.get("classname"), "name": case.get("name"),
                        "reason": node.get("message"),
                    })
    counts["passed"] = (
        counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    )
    expected = collection["collected"]
    # JUnit must describe exactly the same cases as typed collection/execution.
    junit_ids = [
        row.get("classname", "").replace(".", "/") + "::" + row.get("name", "")
        for row in cases
    ]
    normalized = [
        node.split("::", 1)[0].removesuffix(".py") + "::" + node.split("::", 1)[1]
        for node in expected
    ]
    # pytest JUnit represents test classes in classname rather than name.
    normalized = [node.rsplit("::", 1)[0].replace("::", "/") + "::" + node.rsplit("::", 1)[1] for node in normalized]
    if Counter(junit_ids) != Counter(normalized):
        errors.append("JUnit identities do not match collection")
    if counts["failures"] or counts["errors"] or counts["tests"] != len(expected):
        errors.append("JUnit failure/error/count mismatch")
    typed_skips = sum(row["outcome"] == "skipped" for row in execution["reports"])
    if counts["skipped"] != typed_skips:
        errors.append("JUnit/typed skip mismatch")
    # Equal inventories and totals do not prove that the same cases skipped.
    # Join normalized identities AND outcomes; a skip moved from an optional
    # case onto a required case must fail even when the total stays unchanged.
    typed_skipped_ids = {
        row["nodeid"] for row in execution["reports"] if row["outcome"] == "skipped"
    }
    typed_outcomes = Counter(
        (identity, ("skipped",) if nodeid in typed_skipped_ids else ())
        for nodeid, identity in zip(expected, normalized)
    )
    junit_outcomes = Counter(
        (identity, tuple(
            outcome for outcome in ("failure", "error", "skipped")
            if case.find(outcome) is not None
        ))
        for identity, case in zip(junit_ids, cases)
    )
    if junit_outcomes != typed_outcomes:
        errors.append("JUnit outcomes do not match typed execution per node")
    return {**result, "counts": counts, "skips": skips, "errors": errors, "valid": not errors}


def main():
    destination = Path(sys.argv[1]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    provenance = {
        "scope": "component only; no installed Foundation/host/media-graph qualification",
        "head": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_changes": git("status", "--porcelain", "--untracked-files=no"),
        "python": sys.version,
        "platform": platform.platform(),
        "github": {key: os.environ.get(key) for key in (
            "GITHUB_SHA", "GITHUB_EVENT_NAME", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT",
            "PR_HEAD_SHA",
        )},
        "dependencies": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        },
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    assert importlib.metadata.version("amplifier-core") == "2.0.2"
    expected_head = os.environ.get("PR_HEAD_SHA") or os.environ.get("GITHUB_SHA")
    if expected_head and expected_head != provenance["head"]:
        raise RuntimeError("checked-out source does not match event head")
    if provenance["tracked_changes"]:
        raise RuntimeError("tracked source changed before component checks")
    # Both suites run even when one fails; neither failure is suppressed.
    results = {
        "required": run_suite("required", REQUIRED, destination),
        "full": run_suite("full", ("tests",), destination),
    }
    (destination / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    summary = [
        "## Component tests (not installed integration)",
        f"Head: `{provenance['head']}`; Python {platform.python_version()}; Core 2.0.2",
    ]
    for name, result in results.items():
        summary.append(f"- {name}: {json.dumps(result.get('counts'))}; exit={result['returncode']}")
        summary.append(f"  - Typed/JUnit validation: {'PASS' if result['valid'] else 'FAIL'}")
        for error in result.get("errors", [result.get("reason")] if result.get("reason") else []):
            summary.append(f"  - VALIDATION ERROR: {error}")
        for skip in result.get("skips", []):
            summary.append(f"  - SKIPPED {skip['class']}::{skip['name']}: {skip['reason']}")
    summary.append("Opt-in Foundation pairing/positive-graph skips are not qualified or passed.")
    rendered = "\n".join(summary) + "\n"
    print(rendered)
    (destination / "summary.txt").write_text(rendered)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
            stream.write(rendered)
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.iterdir()) if path.is_file() and path.name != "sha256.json"
    }
    (destination / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return 0 if all(result["valid"] for result in results.values()) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest-child":
        import pytest

        sys.exit(pytest.main(sys.argv[3:], plugins=[Recorder(sys.argv[2])]))
    sys.exit(main())