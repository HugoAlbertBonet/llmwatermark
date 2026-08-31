"""The README has to stay true as the code moves.

The spec's bar for it is that it is sufficient for someone to use the tool, so it is
verified rather than asserted. These tests do not run the examples that need a model - they
check the things that actually rot: renamed symbols, moved modules, code that no longer
parses, and figures the text refers to but that are not there.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def code_blocks(language: str) -> list[str]:
    text = README.read_text()
    return re.findall(rf"```{language}\n(.*?)```", text, flags=re.DOTALL)


def imported_symbols(source: str) -> list[tuple[str, str]]:
    """(module, name) for every `from llmwatermark... import name` in a snippet."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("llmwatermark"):
            found.extend((node.module or "", alias.name) for alias in node.names)
    return found


class TestCodeBlocks:
    def test_the_readme_has_runnable_looking_examples(self) -> None:
        assert len(code_blocks("python")) >= 3

    @pytest.mark.parametrize("index", range(len(code_blocks("python"))))
    def test_every_python_block_parses(self, index: int) -> None:
        """A README that does not parse is worse than no README."""
        ast.parse(code_blocks("python")[index])

    def test_every_imported_symbol_exists(self) -> None:
        """Catches the common rot: a rename lands, the README keeps the old name.

        Adapter modules raise ImportError when their backend is absent, which is the
        behaviour they are supposed to have. Those are skipped here and checked in the
        runs where the backend is installed.
        """
        missing, unchecked = [], []
        for block in code_blocks("python"):
            for module_name, symbol in imported_symbols(block):
                try:
                    module = importlib.import_module(module_name)
                except ImportError:
                    unchecked.append(module_name)
                    continue
                if not hasattr(module, symbol):
                    missing.append(f"{module_name}.{symbol}")
        assert missing == [], f"README imports symbols that do not exist: {missing}"
        if unchecked:
            pytest.skip(f"backend not installed, so not checked here: {sorted(set(unchecked))}")

    def test_the_install_commands_name_the_real_extras(self) -> None:
        import importlib.metadata

        declared = set(importlib.metadata.metadata("llmwatermark").get_all("Provides-Extra") or [])
        for extra in re.findall(r"llmwatermark\[([a-z]+)\]", README.read_text()):
            assert extra in declared, f"README names an extra that does not exist: {extra}"


class TestReferences:
    @pytest.mark.parametrize(
        "target", re.findall(r"\]\(((?:docs|tests|examples)/[^)]+)\)", README.read_text())
    )
    def test_linked_files_exist(self, target: str) -> None:
        assert (README.parent / target).exists(), f"README links to a missing file: {target}"

    @pytest.mark.parametrize("image", re.findall(r"!\[[^\]]*\]\(([^)]+)\)", README.read_text()))
    def test_referenced_figures_exist(self, image: str) -> None:
        assert (README.parent / image).exists(), f"README shows a missing figure: {image}"


class TestExamples:
    @pytest.mark.parametrize("script", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
    def test_every_example_parses(self, script: Path) -> None:
        ast.parse(script.read_text())

    @pytest.mark.parametrize("script", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
    def test_every_example_is_linked_from_the_readme(self, script: Path) -> None:
        assert script.name in README.read_text(), f"{script.name} is not mentioned in the README"


def test_the_decision_view_example_runs() -> None:
    """The one example with no model and no backend, so it can run in the core suite."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "show_decision.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.lstrip().startswith("<!doctype html>")
    assert "WATERMARKED" in result.stderr
