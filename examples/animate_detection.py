"""Animate the watermark being read out of text, token by token.

Produces the GIF used in the README. Two passages generated from the same prompt - one
watermarked, one not - appear a token at a time, each token coloured by whether it landed on
that position's greenlist. Underneath, the z-score accumulates and crosses the threshold.

The point is that no single token proves anything: greens and reds appear in both panels.
What separates them is the *rate*, and the curve is what turns that rate into a decision.

    python examples/animate_detection.py --out docs/figures/detection.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from llmwatermark.config import WatermarkConfig
from llmwatermark.detector import DEFAULT_Z_THRESHOLD, WatermarkDetector

KEY = b"quality-evaluation-key-0123456789"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

INK = "#1b2a27"
GREEN = "#12703f"
RED = "#a8393a"
GREY = "#8b9793"
PAPER = "#ffffff"

# Tuned so a monospace character at the font size below fills 1/COLUMNS of the panel
# width. Too few columns and the words drift apart; too many and they overlap.
COLUMNS = 60
LINES = 8
FONT_SIZE = 7.2


def cumulative_z(result, gamma: float) -> np.ndarray:
    greens = np.array([r.is_green for r in result.tokens if r.scored], dtype=float)
    counts = np.arange(1, len(greens) + 1)
    return (greens.cumsum() - gamma * counts) / np.sqrt(counts * gamma * (1 - gamma))


def display_tokens(result, tokenizer, limit: int) -> list[tuple[str, bool | None]]:
    """(text, was_green) per token, decoded so it reads as prose rather than byte pieces."""
    tokens = []
    for record in result.tokens[:limit]:
        piece = tokenizer.decode([record.token_id]).replace("\n", " ")
        tokens.append((piece, record.is_green))
    return tokens


def lay_out(tokens, axes) -> list[plt.Text]:
    """Place each token at its own position, wrapping by character count.

    One artist per token, because a single text artist cannot carry two colours - and the
    colour per token is the whole point of the panel.
    """
    artists, column, line = [], 0, 0
    for piece, is_green in tokens:
        width = max(len(piece), 1)
        if column + width > COLUMNS:
            column, line = 0, line + 1
        if line >= LINES:
            break
        colour = GREY if is_green is None else (GREEN if is_green else RED)
        artist = axes.text(
            column / COLUMNS,
            1 - line * (1 / LINES),
            piece,
            transform=axes.transAxes,
            fontsize=FONT_SIZE,
            family="monospace",
            color=colour,
            va="top",
            visible=False,
        )
        artists.append(artist)
        column += width
    return artists


def panel(axes, title: str) -> None:
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_facecolor(PAPER)
    for spine in axes.spines.values():
        spine.set_color("#dfe5e2")
    axes.set_title(title, color=INK, fontsize=9.5, loc="left", pad=6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default="tests/benchmarks/eval/generations.jsonl")
    parser.add_argument("--out", default="docs/figures/detection.gif")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--index", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=110)
    parser.add_argument("--stride", type=int, default=2)
    arguments = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    rows = [json.loads(line) for line in Path(arguments.generations).read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    config = WatermarkConfig.from_tokenizer(
        tokenizer,
        vocab_size=int(AutoConfig.from_pretrained(arguments.model).vocab_size),
        secret_key=KEY,
    )
    detector = WatermarkDetector(config, tokenizer)
    gamma = config.effective_gamma
    row = rows[arguments.index]

    marked_result = detector.detect(row["delta_2"]["text"])
    plain_result = detector.detect(row["delta_0"]["text"])
    marked_z = cumulative_z(marked_result, gamma)
    plain_z = cumulative_z(plain_result, gamma)
    length = min(len(marked_z), len(plain_z), arguments.tokens)
    marked_z, plain_z = marked_z[:length], plain_z[:length]

    figure = plt.figure(figsize=(8.4, 5.0), dpi=95)
    figure.patch.set_facecolor(PAPER)
    grid = GridSpec(2, 2, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.12, figure=figure)
    marked_axes = figure.add_subplot(grid[0, 0])
    plain_axes = figure.add_subplot(grid[0, 1])
    curve_axes = figure.add_subplot(grid[1, :])

    panel(marked_axes, "watermarked")
    panel(plain_axes, "unwatermarked")
    marked_artists = lay_out(display_tokens(marked_result, tokenizer, length), marked_axes)
    plain_artists = lay_out(display_tokens(plain_result, tokenizer, length), plain_axes)

    curve_axes.set_facecolor(PAPER)
    for side in ("top", "right"):
        curve_axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        curve_axes.spines[side].set_color(GREY)
    curve_axes.grid(True, color="#e9eeec", linewidth=0.8)
    curve_axes.set_axisbelow(True)
    curve_axes.set_xlim(0, length)
    curve_axes.set_ylim(min(-3.0, plain_z.min() - 1), max(marked_z.max() + 1.2, 6))
    curve_axes.set_xlabel("scored tokens", color=INK, fontsize=9)
    curve_axes.set_ylabel("cumulative z", color=INK, fontsize=9)
    curve_axes.tick_params(colors=INK, labelsize=8)
    curve_axes.axhline(DEFAULT_Z_THRESHOLD, color=RED, linewidth=1.2, linestyle="--")
    curve_axes.annotate(
        f"threshold  z = {DEFAULT_Z_THRESHOLD:g}",
        xy=(0.995, DEFAULT_Z_THRESHOLD),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        color=RED,
        fontsize=8,
        ha="right",
    )
    (marked_line,) = curve_axes.plot([], [], color=GREEN, linewidth=1.8)
    (plain_line,) = curve_axes.plot([], [], color=GREY, linewidth=1.8)

    marked_verdict = marked_axes.set_title("watermarked", color=INK, fontsize=9.5, loc="left")
    plain_verdict = plain_axes.set_title("unwatermarked", color=INK, fontsize=9.5, loc="left")
    figure.text(
        0.5,
        0.012,
        "green = on this position's greenlist    red = not    grey = no context window yet",
        ha="center",
        color=GREY,
        fontsize=8,
    )

    steps = np.arange(1, length + 1)

    def draw(frame: int):
        for artists in (marked_artists, plain_artists):
            for index, artist in enumerate(artists):
                artist.set_visible(index < frame)
        marked_line.set_data(steps[:frame], marked_z[:frame])
        plain_line.set_data(steps[:frame], plain_z[:frame])
        current = marked_z[frame - 1] if frame else 0.0
        other = plain_z[frame - 1] if frame else 0.0
        state = "WATERMARKED" if current >= DEFAULT_Z_THRESHOLD else "not yet conclusive"
        marked_verdict.set_text(f"watermarked      z = {current:5.2f}   {state}")
        marked_verdict.set_color(GREEN if current >= DEFAULT_Z_THRESHOLD else INK)
        plain_verdict.set_text(f"unwatermarked    z = {other:5.2f}   no evidence")
        return ()

    frames = [*range(1, length + 1, arguments.stride), *([length] * 12)]
    animation = FuncAnimation(figure, draw, frames=frames, interval=80, blit=False)
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    animation.save(destination, writer=PillowWriter(fps=12))
    plt.close(figure)
    _compress(destination)
    size = destination.stat().st_size / 1e3
    print(f"wrote {destination} ({size:.0f} KB, {len(frames)} frames, {length} tokens)")


def _compress(path: Path, colours: int = 48) -> None:
    """Requantise the palette. This lives in a README, so file size is a real constraint."""
    from PIL import Image

    animation = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(animation.convert("RGB").quantize(colors=colours, method=Image.MEDIANCUT))
            animation.seek(animation.tell() + 1)
    except EOFError:
        pass
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=animation.info.get("duration", 83),
        loop=0,
        optimize=True,
    )


if __name__ == "__main__":
    main()
