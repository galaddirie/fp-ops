"""Verify that each negative consumer fixture fails for the intended reasons."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]


def run_mypy() -> None:
    fixture = TESTS / "typecheck" / "mypy" / "bad.py"
    completed = subprocess.run(
        [sys.executable, "-m", "mypy", "--show-error-codes", str(fixture)],
        check=False,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, "mypy bad fixture unexpectedly passed"
    expected = {
        "then": "[arg-type]",
        'Missing positional argument "value"': "[call-arg]",
        "Incompatible types in assignment": "[assignment]",
        "with_value": "[misc]",
        "Unsupported operand types for &": "[operator]",
    }
    for message, code in expected.items():
        assert message in output and code in output, output


def run_pyright() -> None:
    fixture = TESTS / "typecheck" / "pyright" / "bad.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            "--pythonpath",
            sys.executable,
            "--outputjson",
            str(fixture),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0, "Pyright bad fixture unexpectedly passed"
    report = json.loads(completed.stdout)
    rules = {
        item.get("rule")
        for item in report["generalDiagnostics"]
        if item["severity"] == "error"
    }
    assert {
        "reportArgumentType",
        "reportCallIssue",
        "reportAssignmentType",
        "reportOperatorIssue",
    }.issubset(rules), report


if __name__ == "__main__":
    run_mypy()
    run_pyright()
