"""The inline decision view: showing how the detector reached its verdict.

This is the spec's primary answer to "show how the detector is making the decision", so
it ships in the core package and must pull no dependencies. It renders the per-token
records the detector already produced, so it can never disagree with the score.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import ClassVar

import pytest

from llmwatermark.detector import (
    SKIP_DUPLICATE_CONTEXT,
    SKIP_NO_CONTEXT,
    DetectionResult,
    TokenRecord,
)

ESC = "\x1b"


def record(
    position: int,
    piece: str,
    *,
    is_green: bool | None = True,
    scored: bool = True,
    skip_reason: str | None = None,
    token_id: int | None = None,
) -> TokenRecord:
    return TokenRecord(
        position=position,
        token_id=position * 7 if token_id is None else token_id,
        piece=piece,
        context=() if skip_reason == SKIP_NO_CONTEXT else (position - 1,),
        is_green=is_green,
        scored=scored,
        skip_reason=skip_reason,
    )


def build_result(tokens: tuple[TokenRecord, ...], **overrides: object) -> DetectionResult:
    scored = [token for token in tokens if token.scored]
    fields: dict[str, object] = {
        "z_score": 3.5,
        "p_value": 0.00023,
        "threshold": 4.0,
        "is_watermarked": False,
        "green_count": sum(1 for token in scored if token.is_green),
        "scored_count": len(scored),
        "total_tokens": len(tokens),
        "gamma": 0.25,
        "tokens": tokens,
    }
    fields.update(overrides)
    return DetectionResult(**fields)  # type: ignore[arg-type]


@pytest.fixture
def every_state() -> DetectionResult:
    """One token in each of the four visual states."""
    return build_result(
        (
            record(0, "the", is_green=None, scored=False, skip_reason=SKIP_NO_CONTEXT),
            record(1, "quick", is_green=True),
            record(2, "brown", is_green=False),
            record(3, "fox", is_green=True, scored=False, skip_reason=SKIP_DUPLICATE_CONTEXT),
            record(4, "jumps", is_green=False, scored=False, skip_reason=SKIP_DUPLICATE_CONTEXT),
        )
    )


# Snapshots of the rendered stream for the every_state fixture. Formatting drift shows up
# here as a diff rather than silently changing what users see.
GOLDEN_ANSI_STREAM = (
    "\x1b[2;37mthe\x1b[0m\x1b[1;32mquick\x1b[0m\x1b[1;31mbrown\x1b[0m"
    "\x1b[2;32mfox\x1b[0m\x1b[2;31mjumps\x1b[0m"
)

GOLDEN_HTML_STREAM = (
    '<span class="lw-tok lw-nocontext" title="#0 \u00b7 id 0 \u00b7 no greenlist\n'
    'no context window yet, so no greenlist is defined here">the</span>'
    '<span class="lw-tok lw-green" title="#1 \u00b7 id 7 \u00b7 green, scored\n'
    'context: 0">quick</span>'
    '<span class="lw-tok lw-red" title="#2 \u00b7 id 14 \u00b7 red, scored\n'
    'context: 1">brown</span>'
    '<span class="lw-tok lw-skip-green" title="#3 \u00b7 id 21 \u00b7 green, not scored\n'
    "context: 2\nthis context n-gram repeats an earlier one, so it reuses that greenlist "
    'and would count the same evidence twice">fox</span>'
    '<span class="lw-tok lw-skip-red" title="#4 \u00b7 id 28 \u00b7 red, not scored\n'
    "context: 3\nthis context n-gram repeats an earlier one, so it reuses that greenlist "
    'and would count the same evidence twice">jumps</span>'
)


class TestGoldenSnapshots:
    def test_ansi_stream_is_unchanged(self, every_state: DetectionResult) -> None:
        assert every_state.to_ansi().split("\n\n", 1)[1] == GOLDEN_ANSI_STREAM

    def test_html_stream_is_unchanged(self, every_state: DetectionResult) -> None:
        stream = every_state.to_html().split('<div class="lw-stream">', 1)[1]
        assert stream.split("</div>")[0] == GOLDEN_HTML_STREAM


class TestSummary:
    def test_states_the_verdict_and_the_evidence(self, every_state: DetectionResult) -> None:
        text = every_state.summary()
        assert "not watermarked" in text.lower()
        assert "3.5" in text
        assert "25" in text  # gamma
        assert str(every_state.scored_count) in text
        assert str(every_state.total_tokens) in text

    def test_a_positive_verdict_reads_as_watermarked(self) -> None:
        result = build_result((record(0, "a"),), z_score=9.0, is_watermarked=True)
        assert "not watermarked" not in result.summary().lower()
        assert "watermarked" in result.summary().lower()

    def test_summary_heads_both_renderings(self, every_state: DetectionResult) -> None:
        assert "z" in every_state.to_ansi(color=False).splitlines()[0].lower()
        assert "watermarked" in every_state.to_html().lower()


class TestAnsi:
    def test_plain_output_contains_no_escape_sequences(self, every_state: DetectionResult) -> None:
        assert ESC not in every_state.to_ansi(color=False)

    def test_colored_output_is_escaped_and_always_reset(self, every_state: DetectionResult) -> None:
        text = every_state.to_ansi()
        assert ESC in text
        assert text.count("\x1b[0m") >= every_state.total_tokens

    def test_the_four_states_are_visually_distinct(self, every_state: DetectionResult) -> None:
        codes = set(re.findall(r"\x1b\[([0-9;]+)m", every_state.to_ansi()))
        codes.discard("0")
        assert len(codes) >= 4

    def test_every_token_appears_in_order(self, every_state: DetectionResult) -> None:
        text = every_state.to_ansi(color=False)
        positions = [text.index(token.piece) for token in every_state.tokens]
        assert positions == sorted(positions)

    def test_whitespace_is_made_visible(self) -> None:
        """A green space is invisible; the colored stream would silently mislead."""
        result = build_result((record(0, " "), record(1, "\n"), record(2, "\t")))
        text = result.to_ansi(color=False)
        assert "·" in text
        assert "⏎" in text
        assert "⇥" in text

    def test_a_legend_explains_the_states(self, every_state: DetectionResult) -> None:
        text = every_state.to_ansi(color=False).lower()
        assert "green" in text
        assert "skipped" in text or "repeat" in text


class TestHtml:
    def test_is_a_fragment_by_default(self, every_state: DetectionResult) -> None:
        markup = every_state.to_html()
        assert "<!doctype" not in markup.lower()
        assert "<style" in markup

    def test_can_produce_a_standalone_document(self, every_state: DetectionResult) -> None:
        markup = every_state.to_html(full_document=True)
        assert markup.lower().startswith("<!doctype html>")
        assert "</html>" in markup

    def test_is_self_contained(self, every_state: DetectionResult) -> None:
        """No external assets: the file must render offline, from any directory."""
        markup = every_state.to_html(full_document=True)
        assert "http://" not in markup
        assert "https://" not in markup
        assert "<script" not in markup.lower()

    def test_every_token_carries_hover_detail(self, every_state: DetectionResult) -> None:
        markup = every_state.to_html()
        assert markup.count("title=") >= every_state.total_tokens
        assert "id 7" in markup  # token_id of position 1
        assert "context" in markup.lower()

    def test_skipped_positions_say_why(self, every_state: DetectionResult) -> None:
        markup = every_state.to_html().lower()
        assert "repeat" in markup or "duplicate" in markup
        assert "context window" in markup or "no context" in markup

    def test_repr_html_delegates_to_to_html(self, every_state: DetectionResult) -> None:
        assert every_state._repr_html_() == every_state.to_html()

    def test_renders_in_both_colour_schemes(self, every_state: DetectionResult) -> None:
        assert "prefers-color-scheme" in every_state.to_html()


class TestEscaping:
    """Token pieces are attacker-influenced text. This is where a render bug is a hole."""

    DANGEROUS: ClassVar[list[str]] = [
        "<script>alert(1)</script>",
        "</span><script>x</script>",
        "a & b",
        'quote" attr',
        "single' quote",
        '" onmouseover="alert(1)',
    ]

    @pytest.mark.parametrize("piece", DANGEROUS)
    def test_no_raw_markup_survives(self, piece: str) -> None:
        markup = build_result((record(0, piece),)).to_html()
        body = markup.split("</style>", 1)[1]
        assert "<script" not in body.lower()
        assert "alert(1)" not in body or "&" in body

    @pytest.mark.parametrize("piece", DANGEROUS)
    def test_attributes_cannot_be_broken_out_of(self, piece: str) -> None:
        markup = build_result((record(0, piece),)).to_html()
        for title in re.findall(r'title="([^"]*)"', markup):
            assert "<" not in title
            assert ">" not in title

    def test_ampersands_are_escaped_once(self) -> None:
        markup = build_result((record(0, "a & b"),)).to_html()
        assert "&amp;" in markup
        assert "&amp;amp;" not in markup

    def test_control_characters_do_not_leak_through(self) -> None:
        markup = build_result((record(0, "a\x00b\x07c"),)).to_html()
        assert "\x00" not in markup
        assert "\x07" not in markup


def stream_styles(markup: str) -> list[str]:
    """The style of each token span in the rendered stream, ignoring CSS and the legend."""
    stream = markup.split('<div class="lw-stream">', 1)[1]
    return re.findall(r'class="lw-tok (lw-[a-z-]+)"', stream)


class TestConsistencyWithTheScore:
    def test_green_styling_count_matches_the_green_count(
        self, every_state: DetectionResult
    ) -> None:
        """A rendering that disagrees with its own verdict is worse than none."""
        styles = stream_styles(every_state.to_html())
        assert styles.count("lw-green") == every_state.green_count

    def test_scored_styling_count_matches_the_scored_count(
        self, every_state: DetectionResult
    ) -> None:
        styles = stream_styles(every_state.to_html())
        scored = styles.count("lw-green") + styles.count("lw-red")
        assert scored == every_state.scored_count

    def test_one_span_per_token(self, every_state: DetectionResult) -> None:
        assert len(stream_styles(every_state.to_html())) == every_state.total_tokens

    def test_all_four_states_are_reachable(self, every_state: DetectionResult) -> None:
        styles = set(stream_styles(every_state.to_html()))
        assert styles == {
            "lw-green",
            "lw-red",
            "lw-skip-green",
            "lw-skip-red",
            "lw-nocontext",
        }

    def test_an_empty_result_still_renders(self) -> None:
        result = build_result((), scored_count=0, green_count=0, total_tokens=0)
        assert result.to_html()
        assert result.to_ansi(color=False)


class TestUnicode:
    @pytest.mark.parametrize("piece", ["日本語", "\U0001f600", "مرحبا", "é", "​", "a" * 500])
    def test_exotic_pieces_render(self, piece: str) -> None:
        result = build_result((record(0, piece),))
        assert result.to_html()
        assert result.to_ansi()


class TestNoDependencies:
    def test_rendering_pulls_no_optional_dependency(self) -> None:
        """The decision view ships in the core package and must stay dependency-free."""
        probe = (
            "import sys; from llmwatermark.detector import DetectionResult; "
            "import llmwatermark.render; "
            "print([m for m in ('matplotlib', 'torch', 'transformers') if m in sys.modules])"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "[]"
