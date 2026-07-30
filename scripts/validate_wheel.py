"""Build and validate runtime and typing contracts from an isolated wheel."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(
    command: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def executable(environment: Path, name: str) -> Path:
    folder = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return environment / folder / f"{name}{suffix}"


def main() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', project, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("project version not found")
    expected_version = match.group(1)

    with tempfile.TemporaryDirectory(prefix="fp-ops-wheel-") as directory:
        work = Path(directory)
        distribution = work / "dist"
        run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(distribution),
                str(ROOT),
            ],
            cwd=work,
        )
        wheel = next(distribution.glob("fp_ops-*.whl"))

        environment = work / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = executable(environment, "python")
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                str(wheel),
                "mypy",
                "pyright",
            ],
            cwd=work,
        )

        run(
            [
                str(python),
                "-c",
                (
                    "import asyncio, fp_ops; "
                    f"assert fp_ops.__version__ == {expected_version!r}; "
                    "assert not hasattr(fp_ops.Operation, 'execute'); "
                    "result = asyncio.run(fp_ops.operation(lambda x: x + 1).run(1)); "
                    "assert result == fp_ops.Ok(2)"
                ),
            ],
            cwd=work,
        )

        fixtures = work / "fixtures"
        fixtures.mkdir()
        mypy_good = fixtures / "mypy_good.py"
        mypy_bad = fixtures / "mypy_bad.py"
        pyright_good = fixtures / "pyright_good.py"
        pyright_bad = fixtures / "pyright_bad.py"
        shutil.copy(ROOT / "tests/typecheck/mypy/good.py", mypy_good)
        shutil.copy(ROOT / "tests/typecheck/mypy/bad.py", mypy_bad)
        shutil.copy(ROOT / "tests/typecheck/pyright/good.py", pyright_good)
        shutil.copy(ROOT / "tests/typecheck/pyright/bad.py", pyright_bad)

        run([str(python), "-m", "mypy", "--strict", str(mypy_good)], cwd=work)
        bad_mypy = run(
            [
                str(python),
                "-m",
                "mypy",
                "--strict",
                "--show-error-codes",
                str(mypy_bad),
            ],
            cwd=work,
            check=False,
        )
        assert bad_mypy.returncode != 0, "installed-wheel mypy bad fixture passed"
        assert "[arg-type]" in bad_mypy.stdout
        assert "[assignment]" in bad_mypy.stdout
        assert "[call-arg]" in bad_mypy.stdout

        config = work / "pyrightconfig.json"
        config.write_text(
            json.dumps(
                {
                    "venvPath": str(work),
                    "venv": "venv",
                    "pythonVersion": "3.10",
                    "typeCheckingMode": "strict",
                }
            ),
            encoding="utf-8",
        )
        run(
            [
                str(python),
                "-m",
                "pyright",
                "--project",
                str(config),
                str(pyright_good),
            ],
            cwd=work,
        )
        bad_pyright = run(
            [
                str(python),
                "-m",
                "pyright",
                "--project",
                str(config),
                "--outputjson",
                str(pyright_bad),
            ],
            cwd=work,
            check=False,
        )
        assert (
            bad_pyright.returncode != 0
        ), "installed-wheel Pyright bad fixture passed"
        diagnostics = json.loads(bad_pyright.stdout)["generalDiagnostics"]
        rules = {item.get("rule") for item in diagnostics}
        assert {
            "reportArgumentType",
            "reportAssignmentType",
            "reportCallIssue",
        }.issubset(rules)


if __name__ == "__main__":
    main()
