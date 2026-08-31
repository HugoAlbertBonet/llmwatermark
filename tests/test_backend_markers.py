"""The requires_<backend> skip machinery in the root conftest.

Adapter tests must be skipped by default so the suite runs on any machine, and must run
when explicitly enabled. Both directions are exercised here against the real conftest
source, so this stays honest if the hooks are edited.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT_CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"


def _load_root_conftest() -> Any:
    """Import the rootdir conftest under a name of its own.

    A plain ``import conftest`` resolves to tests/conftest.py, which shadows it.
    """
    spec = importlib.util.spec_from_file_location("_root_conftest", ROOT_CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


requirement_of = _load_root_conftest().requirement_of

SUB_SUITE = """
import pytest

@pytest.mark.requires_demo
def test_needs_demo_backend():
    pass

def test_plain_core_test():
    pass
"""

SUB_INI = """
[pytest]
markers =
    requires_demo: needs the demo backend
"""


@pytest.fixture
def suite(pytester: pytest.Pytester) -> pytest.Pytester:
    """A throwaway suite wired up with this repo's real conftest hooks."""
    # pytest_plugins is only legal in a rootdir conftest, and the sub-suite has its own.
    source = "\n".join(
        line
        for line in ROOT_CONFTEST.read_text().splitlines()
        if not line.startswith("pytest_plugins")
    )
    pytester.makeconftest(source)
    pytester.makeini(SUB_INI)
    pytester.makepyfile(SUB_SUITE)
    return pytester


def test_backend_tests_are_skipped_by_default(suite: pytest.Pytester) -> None:
    result = suite.runpytest()
    result.assert_outcomes(passed=1, skipped=1)


def test_skip_reason_names_how_to_enable_the_backend(suite: pytest.Pytester) -> None:
    result = suite.runpytest("-rs")
    result.stdout.fnmatch_lines(["*--backend demo*"])


def test_named_backend_flag_enables_only_that_backend(suite: pytest.Pytester) -> None:
    result = suite.runpytest("--backend", "demo")
    result.assert_outcomes(passed=2)

    other = suite.runpytest("--backend", "something_else")
    other.assert_outcomes(passed=1, skipped=1)


def test_all_backends_flag_enables_every_marked_test(suite: pytest.Pytester) -> None:
    result = suite.runpytest("--all-backends")
    result.assert_outcomes(passed=2)


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("requires_transformers", "transformers"),
        ("requires_llama_cpp", "llama_cpp"),
        ("requires_viz", "viz"),
        ("benchmark", None),
        ("requires_", None),
        ("parametrize", None),
    ],
)
def test_requirement_of_only_matches_requires_prefixed_markers(
    marker: str, expected: str | None
) -> None:
    assert requirement_of(marker) == expected
