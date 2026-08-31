"""Animate the detector's confidence building token by token.

Produces the GIF used in the README. Each frame adds one scored token and redraws the
cumulative z-score, so the viewer sees the evidence accumulate and cross the threshold
rather than being handed a final number.

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

from llmwatermark.config import WatermarkConfig
from llmwatermark.detector import DEFAULT_Z_THRESHOLD, WatermarkDetector

KEY = b"quality-evaluation-key-0123456789"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
INK, GREEN, RED, GREY, PAPER = "#1b2a27", "#1b7f4b", "#b23a3a", "#7b8785", "#ffffff"


def cumulative_z(result, gamma: float) -> np.ndarray:
    greens = np.array([r.is_green for r in result.tokens if r.scored], dtype=float)
    counts = np.arange(1, len(greens) + 1)
    return (greens.cumsum() - gamma * counts) / np.sqrt(counts * gamma * (1 - gamma))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default="tests/benchmarks/eval/generations.jsonl")
    parser.add_argument("--out", default="docs/figures/detection.gif")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--index", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2, help="tokens advanced per frame")
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
    marked = cumulative_z(detector.detect(row["delta_2"]["text"]), gamma)
    plain = cumulative_z(detector.detect(row["delta_0"]["text"]), gamma)
    length = min(len(marked), len(plain), 150)
    marked, plain = marked[:length], plain[:length]

    figure, axes = plt.subplots(figsize=(6.4, 3.5), dpi=90)
    axes.set_facecolor(PAPER)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GREY)
    axes.grid(True, color="#e6ebe9", linewidth=0.8)
    axes.set_axisbelow(True)
    axes.set_xlim(0, length)
    axes.set_ylim(min(-3, plain.min() - 1), max(marked.max() + 1.5, 6))
    axes.set_xlabel("scored tokens", color=INK, fontsize=10)
    axes.set_ylabel("cumulative z-score", color=INK, fontsize=10)
    axes.tick_params(colors=INK, labelsize=9)
    axes.axhline(DEFAULT_Z_THRESHOLD, color=RED, linewidth=1.3, linestyle="--")
    axes.annotate(
        f"threshold  z = {DEFAULT_Z_THRESHOLD:g}",
        xy=(0.99, DEFAULT_Z_THRESHOLD),
        xycoords=("axes fraction", "data"),
        xytext=(0, 6),
        textcoords="offset points",
        color=RED,
        fontsize=9,
        ha="right",
    )

    (marked_line,) = axes.plot([], [], color=GREEN, linewidth=1.8, label="watermarked")
    (plain_line,) = axes.plot([], [], color=GREY, linewidth=1.8, label="unwatermarked")
    verdict = axes.text(0.02, 0.94, "", transform=axes.transAxes, fontsize=11, color=INK)
    axes.legend(frameon=False, fontsize=9, loc="lower right")
    figure.tight_layout()

    steps = np.arange(1, length + 1)

    def draw(frame: int):
        marked_line.set_data(steps[:frame], marked[:frame])
        plain_line.set_data(steps[:frame], plain[:frame])
        current = marked[frame - 1] if frame else 0.0
        state = "WATERMARKED" if current >= DEFAULT_Z_THRESHOLD else "not enough evidence yet"
        colour = GREEN if current >= DEFAULT_Z_THRESHOLD else GREY
        verdict.set_text(f"{frame:>3} tokens    z = {current:5.2f}    {state}")
        verdict.set_color(colour)
        return marked_line, plain_line, verdict

    # Advance several tokens per frame: the curve reads the same and the file is a
    # fraction of the size, which matters for something embedded in a README.
    frames = list(range(1, length + 1, arguments.stride)) + [length] * 12
    animation = FuncAnimation(figure, draw, frames=frames, interval=70, blit=True)
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    animation.save(destination, writer=PillowWriter(fps=14))
    plt.close(figure)
    print(f"wrote {destination} ({destination.stat().st_size / 1e3:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
