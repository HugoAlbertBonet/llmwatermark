"""Pytest configuration shared by the whole test suite.

Core tests (hashing, greenlist construction, determinism, the z-test, the detector,
config and fingerprint) are pure CPU, deterministic, and always run. Tests that need an
inference backend or an optional extra are marked ``requires_<name>`` and are skipped
unless explicitly enabled, so the suite stays runnable without eight accelerators.

Enable them with ``--backend <name>`` (repeatable) or ``--all-backends``.
"""

from __future__ import annotations

import pytest

# Enables the ``pytester`` fixture, used to test this file's own skip machinery.
pytest_plugins = ["pytester"]

MARKER_PREFIX = "requires_"


def requirement_of(marker_name: str) -> str | None:
    """Return the requirement a marker name gates, or None if it gates nothing.

    ``requires_transformers`` -> ``transformers``; ``benchmark`` -> ``None``.
    """
    if marker_name.startswith(MARKER_PREFIX) and len(marker_name) > len(MARKER_PREFIX):
        return marker_name[len(MARKER_PREFIX) :]
    return None


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("llmwatermark")
    group.addoption(
        "--backend",
        action="append",
        default=[],
        metavar="NAME",
        dest="backends",
        help=(
            "Run tests marked requires_NAME (e.g. --backend transformers). "
            "Repeat the flag to enable several."
        ),
    )
    group.addoption(
        "--all-backends",
        action="store_true",
        default=False,
        dest="all_backends",
        help="Run every requires_<name> test. Needs all the corresponding hardware.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("all_backends"):
        return

    enabled = set(config.getoption("backends"))
    for item in items:
        for marker in item.iter_markers():
            requirement = requirement_of(marker.name)
            if requirement is not None and requirement not in enabled:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"needs {requirement!r}; enable with "
                            f"--backend {requirement} or --all-backends"
                        )
                    )
                )
                break
