"""Packaging and dependency-policy guarantees.

These lock in the promise that the core package is importable with numpy alone and that
no backend or plotting dependency can creep into the import path.
"""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

import pytest

import llmwatermark

# Anything here must never be imported as a side effect of importing llmwatermark.
FORBIDDEN_ON_IMPORT = ["torch", "transformers", "vllm", "matplotlib"]

EXPECTED_EXTRAS = {"transformers", "vllm", "viz", "dev"}


def test_version_is_exposed_and_pep440() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+|[ab]\d+|rc\d+)?", llmwatermark.__version__)


def test_installed_version_matches_package_attribute() -> None:
    assert importlib.metadata.version("llmwatermark") == llmwatermark.__version__


def test_core_import_pulls_no_backend_or_plotting_dependency() -> None:
    """Run in a subprocess so modules already imported by the test session don't mask it."""
    probe = (
        "import sys; import llmwatermark; "
        f"leaked = [m for m in {FORBIDDEN_ON_IMPORT!r} if m in sys.modules]; "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", f"importing llmwatermark leaked: {result.stdout.strip()}"


def test_only_numpy_is_an_unconditional_dependency() -> None:
    requirements = importlib.metadata.requires("llmwatermark") or []
    unconditional = [r for r in requirements if "extra ==" not in r]
    assert [r.split(maxsplit=1)[0].split(">")[0] for r in unconditional] == ["numpy"]


def test_declared_extras_cover_the_dependency_policy() -> None:
    declared = set(importlib.metadata.metadata("llmwatermark").get_all("Provides-Extra") or [])
    assert declared >= EXPECTED_EXTRAS


def test_package_ships_py_typed_marker() -> None:
    assert (Path(llmwatermark.__file__).parent / "py.typed").is_file()


@pytest.mark.parametrize("module", FORBIDDEN_ON_IMPORT)
def test_forbidden_modules_are_not_imported_by_the_test_session_itself(
    module: str, request: pytest.FixtureRequest
) -> None:
    """Core tests must stay pure-CPU and dependency-free; a stray import here is a bug.

    Only meaningful for the default run. Enabling a backend deliberately imports one.
    """
    if request.config.getoption("backends") or request.config.getoption("all_backends"):
        pytest.skip("a backend was enabled explicitly, so importing it is expected")
    assert module not in sys.modules
