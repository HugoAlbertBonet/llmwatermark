"""Rendering a detection decision so a person can see how it was reached.

Ships in the core package and pulls no dependencies: it is string building over the
per-token records the detector already produced. Nothing is recomputed here, so the
picture can never disagree with the score it illustrates.

Four visual states, not two. Green and red are the evidence; the other two are what make
the detector's behaviour legible:

* **green / red, scored** - counted for or against.
* **green / red, skipped** - the position's context n-gram repeats an earlier one, so it
  reuses that greenlist and would count the same evidence twice. Shown faded.
* **no context** - the first h positions have no full context window, so no greenlist is
  defined for them at all.

Seeing three quarters of a text visibly struck out is what turns "the score seems low"
into "this text is too repetitive to score", which no single number conveys.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from llmwatermark.detector import DetectionResult, TokenRecord

__all__ = ["summary", "to_ansi", "to_html"]

# One CSS class per visual state. The names are asserted against the aggregate counts in
# the tests, so a rendering cannot silently drift from the verdict it illustrates.
STYLE_GREEN: Final[str] = "lw-green"
STYLE_RED: Final[str] = "lw-red"
STYLE_SKIP_GREEN: Final[str] = "lw-skip-green"
STYLE_SKIP_RED: Final[str] = "lw-skip-red"
STYLE_NO_CONTEXT: Final[str] = "lw-nocontext"

_ANSI_RESET: Final[str] = "\x1b[0m"
_ANSI_CODES: Final[dict[str, str]] = {
    STYLE_GREEN: "\x1b[1;32m",
    STYLE_RED: "\x1b[1;31m",
    STYLE_SKIP_GREEN: "\x1b[2;32m",
    STYLE_SKIP_RED: "\x1b[2;31m",
    STYLE_NO_CONTEXT: "\x1b[2;37m",
}

# Whitespace carries the same colour as any other token, and a coloured space is
# invisible. Substituting a glyph is the difference between a readable stream and a
# misleading one.
_VISIBLE: Final[dict[str, str]] = {" ": "·", "\n": "⏎", "\t": "⇥", "\r": "␍"}


def style_of(record: TokenRecord) -> str:
    """The visual state of one position."""
    if record.is_green is None:
        return STYLE_NO_CONTEXT
    if record.scored:
        return STYLE_GREEN if record.is_green else STYLE_RED
    return STYLE_SKIP_GREEN if record.is_green else STYLE_SKIP_RED


def summary(result: DetectionResult) -> str:
    """The one-line verdict that heads both renderings."""
    verdict = "WATERMARKED" if result.is_watermarked else "not watermarked"
    return (
        f"{verdict}: z = {result.z_score:.2f} (threshold {result.threshold:.2f}), "
        f"p = {result.p_value:.3g}\n"
        f"{result.green_count} of {result.scored_count} scored tokens green "
        f"({result.green_fraction:.1%}, expected {result.gamma:.1%} by chance); "
        f"{result.total_tokens} tokens in total, {result.skipped_count} not scored."
    )


def to_ansi(result: DetectionResult, *, color: bool = True) -> str:
    """The decision as a colored token stream for a terminal.

    :param color: Set False for plain text, for piping or for a terminal without SGR.
    """
    parts = [summary(result), "\n", _ansi_legend(color), "\n\n"]
    for record in result.tokens:
        parts.append(_ansi_token(record, color=color))
    return "".join(parts)


def to_html(result: DetectionResult, *, full_document: bool = False) -> str:
    """The decision as self-contained HTML, with per-token hover detail.

    No external assets and no scripts, so it renders offline from anywhere: a notebook
    cell, an email, a file on disk.

    :param full_document: Wrap the fragment in a complete HTML document, for writing to a
        standalone file.
    """
    spans = "".join(_html_token(record) for record in result.tokens)
    body = (
        f'<div class="lw-report">'
        f'<pre class="lw-summary">{html.escape(summary(result))}</pre>'
        f"{_html_legend()}"
        f'<div class="lw-stream">{spans}</div>'
        f"</div>"
    )
    fragment = f"<style>{_CSS}</style>{body}"
    if not full_document:
        return fragment
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>llmwatermark detection</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )


def _ansi_token(record: TokenRecord, *, color: bool) -> str:
    text = _visible_text(record.piece)
    if not color:
        return text
    return f"{_ANSI_CODES[style_of(record)]}{text}{_ANSI_RESET}"


def _ansi_legend(color: bool) -> str:
    entries = (
        (STYLE_GREEN, "green (counted)"),
        (STYLE_RED, "red (counted)"),
        (STYLE_SKIP_GREEN, "skipped: context repeats"),
        (STYLE_NO_CONTEXT, "skipped: no context window"),
    )
    rendered = []
    for style, label in entries:
        rendered.append(f"{_ANSI_CODES[style]}{label}{_ANSI_RESET}" if color else label)
    return "  |  ".join(rendered)


def _html_token(record: TokenRecord) -> str:
    style = style_of(record)
    return (
        f'<span class="lw-tok {style}" title="{html.escape(_tooltip(record), quote=True)}">'
        f"{html.escape(_visible_text(record.piece))}</span>"
    )


def _html_legend() -> str:
    entries = (
        (STYLE_GREEN, "green, counted"),
        (STYLE_RED, "red, counted"),
        (STYLE_SKIP_GREEN, "skipped: context n-gram repeats"),
        (STYLE_NO_CONTEXT, "skipped: no context window"),
    )
    items = "".join(
        f'<span class="lw-tok {style}">{html.escape(label)}</span>' for style, label in entries
    )
    return f'<div class="lw-legend">{items}</div>'


def _tooltip(record: TokenRecord) -> str:
    lines = [f"#{record.position} · id {record.token_id} · {_verdict_text(record)}"]
    if record.context:
        lines.append("context: " + ", ".join(str(value) for value in record.context))
    if record.skip_reason is not None:
        lines.append(_SKIP_EXPLANATIONS.get(record.skip_reason, record.skip_reason))
    return "\n".join(lines)


def _verdict_text(record: TokenRecord) -> str:
    if record.is_green is None:
        return "no greenlist"
    colour = "green" if record.is_green else "red"
    return f"{colour}, scored" if record.scored else f"{colour}, not scored"


_SKIP_EXPLANATIONS: Final[dict[str, str]] = {
    "no_context": "no context window yet, so no greenlist is defined here",
    "duplicate_context": (
        "this context n-gram repeats an earlier one, so it reuses that greenlist and "
        "would count the same evidence twice"
    ),
}


def _visible_text(piece: str) -> str:
    """Substitute glyphs for whitespace and control characters."""
    out = []
    for character in piece:
        if character in _VISIBLE:
            out.append(_VISIBLE[character])
        elif character < " " or character == "\x7f":
            out.append(f"\\x{ord(character):02x}")
        else:
            out.append(character)
    return "".join(out)


# Colours are given as custom properties and redefined for dark mode, so the same markup
# is legible in a notebook, an email client and a file opened in a browser.
_CSS: Final[str] = """
.lw-report{--lw-green-bg:#d7f5dd;--lw-green-fg:#0b5f26;--lw-red-bg:#fbdcdc;
--lw-red-fg:#8a1414;--lw-dim:#6b7280;--lw-border:#d1d5db;
font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;
line-height:1.9;color:inherit}
@media (prefers-color-scheme:dark){.lw-report{--lw-green-bg:#123a1f;--lw-green-fg:#7ee29b;
--lw-red-bg:#40161a;--lw-red-fg:#f4a3a3;--lw-dim:#9ca3af;--lw-border:#374151}}
.lw-summary{white-space:pre-wrap;margin:0 0 .6em;font:inherit}
.lw-legend{margin:0 0 .8em;display:flex;flex-wrap:wrap;gap:.4em}
.lw-stream{white-space:pre-wrap;word-break:break-word;
border:1px solid var(--lw-border);border-radius:6px;padding:.6em}
.lw-tok{padding:.05em .12em;border-radius:3px}
.lw-green{background:var(--lw-green-bg);color:var(--lw-green-fg)}
.lw-red{background:var(--lw-red-bg);color:var(--lw-red-fg)}
.lw-skip-green{color:var(--lw-green-fg);opacity:.45;
border-bottom:1px dashed currentColor}
.lw-skip-red{color:var(--lw-red-fg);opacity:.45;border-bottom:1px dashed currentColor}
.lw-nocontext{color:var(--lw-dim);opacity:.6}
""".strip()
