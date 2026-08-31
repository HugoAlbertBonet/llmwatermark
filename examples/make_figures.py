"""Regenerate the figures used in the README.

Reads the evaluation generations produced by tests/benchmarks/eval/eval_generate.py. That
file is large and not committed, so the rendered figures in docs/figures/ are the artefact
of record; this script is kept so they can be redrawn if the data is ever regenerated.

    python examples/make_figures.py --generations tests/benchmarks/eval/generations.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llmwatermark.config import WatermarkConfig
from llmwatermark.detector import DEFAULT_Z_THRESHOLD, WatermarkDetector
from llmwatermark.errors import DetectionError

KEY = b"quality-evaluation-key-0123456789"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

INK = "#1b2a27"
GREEN = "#1b7f4b"
RED = "#b23a3a"
GREY = "#7b8785"
PAPER = "#ffffff"

# Measured in tests/benchmarks/eval/EVAL.md.
DELTAS = [0, 1, 2, 4, 6]
TPR_AT_1E4 = [0.0, 56.2, 95.0, 100.0, 100.0]
PERPLEXITY_RATIO = [1.00, 1.11, 1.44, 4.38, 8.66]


def style(axes: plt.Axes) -> None:
    axes.set_facecolor(PAPER)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GREY)
    axes.tick_params(colors=INK, labelsize=9)
    axes.grid(True, color="#e6ebe9", linewidth=0.8)
    axes.set_axisbelow(True)


def cumulative_z(result, gamma: float) -> np.ndarray:
    """z after each scored token, which is how confidence actually accrues."""
    greens = np.array([record.is_green for record in result.tokens if record.scored], dtype=float)
    counts = np.arange(1, len(greens) + 1)
    return (greens.cumsum() - gamma * counts) / np.sqrt(counts * gamma * (1 - gamma))


def figure_confidence(rows, detector, gamma: float, out: Path) -> None:
    figure, axes = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    style(axes)
    arms = (("delta_2", GREEN, "watermarked"), ("delta_0", GREY, "unwatermarked"))
    for arm, colour, label in arms:
        drawn = 0
        for row in rows:
            if drawn >= 12:
                break
            try:
                curve = cumulative_z(detector.detect(row[arm]["text"]), gamma)
            except DetectionError:
                continue
            axes.plot(
                np.arange(1, len(curve) + 1),
                curve,
                color=colour,
                linewidth=1.1,
                alpha=0.75,
                label=label if drawn == 0 else None,
            )
            drawn += 1
    axes.axhline(DEFAULT_Z_THRESHOLD, color=RED, linewidth=1.4, linestyle="--")
    axes.annotate(
        f"decision threshold  z = {DEFAULT_Z_THRESHOLD:g}",
        xy=(0.99, DEFAULT_Z_THRESHOLD),
        xycoords=("axes fraction", "data"),
        xytext=(0, 7),
        textcoords="offset points",
        color=RED,
        fontsize=9,
        ha="right",
    )
    axes.set_xlabel("scored tokens", color=INK, fontsize=10)
    axes.set_ylabel("cumulative z-score", color=INK, fontsize=10)
    axes.set_title("Confidence accumulates with length", color=INK, fontsize=12, loc="left", pad=12)
    axes.legend(frameon=False, fontsize=9, loc="upper left")
    figure.tight_layout()
    figure.savefig(out, facecolor=PAPER)
    plt.close(figure)


def figure_separation(rows, detector, gamma: float, out: Path) -> None:
    def fractions(getter) -> np.ndarray:
        values = []
        for row in rows:
            try:
                values.append(detector.detect(getter(row)).green_fraction)
            except DetectionError:
                continue
        return np.array(values)

    human = fractions(lambda row: row["human"])
    plain = fractions(lambda row: row["delta_0"]["text"])
    marked = fractions(lambda row: row["delta_2"]["text"])

    figure, axes = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    style(axes)
    bins = np.linspace(0.05, 0.85, 45)
    for values, colour, label in (
        (human, "#4a6fa5", f"human  (mean {human.mean():.1%})"),
        (plain, GREY, f"unwatermarked  (mean {plain.mean():.1%})"),
        (marked, GREEN, f"watermarked, delta=2  (mean {marked.mean():.1%})"),
    ):
        axes.hist(values, bins=bins, color=colour, alpha=0.6, label=label)
    axes.axvline(gamma, color=RED, linewidth=1.4, linestyle="--")
    axes.text(gamma + 0.008, axes.get_ylim()[1] * 0.92, "gamma = 25%", color=RED, fontsize=9)
    axes.set_xlabel("fraction of tokens on the greenlist", color=INK, fontsize=10)
    axes.set_ylabel("texts", color=INK, fontsize=10)
    axes.set_title("The two populations separate", color=INK, fontsize=12, loc="left", pad=12)
    axes.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(out, facecolor=PAPER)
    plt.close(figure)


def figure_tradeoff(out: Path) -> None:
    figure, left = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    style(left)
    right = left.twinx()
    right.spines["top"].set_visible(False)
    right.tick_params(colors=INK, labelsize=9)

    left.plot(DELTAS, TPR_AT_1E4, color=GREEN, marker="o", linewidth=1.8, label="detection")
    left.set_ylabel("detected at a 0.01% false-positive rate (%)", color=GREEN, fontsize=10)
    left.set_ylim(-5, 108)
    left.tick_params(axis="y", colors=GREEN)

    right.plot(DELTAS, PERPLEXITY_RATIO, color=RED, marker="s", linewidth=1.8, label="quality cost")
    right.set_ylabel("perplexity, relative to unwatermarked", color=RED, fontsize=10)
    right.tick_params(axis="y", colors=RED)
    right.axhline(1.0, color=GREY, linewidth=0.9, linestyle=":")

    left.axvspan(1.6, 2.4, color="#dff0e6", zorder=0)
    left.text(2.0, 52, "default", color=INK, fontsize=9, ha="center")
    left.set_xlabel("delta", color=INK, fontsize=10)
    left.set_xticks(DELTAS)
    left.set_title(
        "Detection saturates before quality falls away", color=INK, fontsize=12, loc="left", pad=12
    )
    figure.tight_layout()
    figure.savefig(out, facecolor=PAPER)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default="tests/benchmarks/eval/generations.jsonl")
    parser.add_argument("--out-dir", default="docs/figures")
    parser.add_argument("--model", default=MODEL)
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

    directory = Path(arguments.out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    figure_confidence(rows, detector, gamma, directory / "confidence.png")
    figure_separation(rows, detector, gamma, directory / "separation.png")
    figure_tradeoff(directory / "tradeoff.png")
    print(f"wrote three figures to {directory}")


if __name__ == "__main__":
    main()
