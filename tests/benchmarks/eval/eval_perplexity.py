"""Perplexity of each arm under an independent judge model.

The judge must be a *different family* with a *different tokenizer*. Scoring text with the
model that produced it is circular, and scoring with a sibling is partly circular: a model
rates its own family's phrasing as unsurprising regardless of whether the text is good.

Loaded after generation, since the generator and the judge do not fit in 8 GB together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

JUDGE = "HuggingFaceTB/SmolLM2-1.7B"


def perplexity(model, tokenizer, texts: list[str], window: int = 512) -> np.ndarray:
    """Per-text perplexity, scored one at a time so padding cannot bias the mean."""
    scores = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=window)
        ids = {key: value.to(model.device) for key, value in ids.items()}
        if ids["input_ids"].shape[1] < 8:
            continue
        with torch.no_grad():
            loss = model(**ids, labels=ids["input_ids"]).loss
        scores.append(float(torch.exp(loss)))
    return np.array(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default=JUDGE)
    parser.add_argument("--generations", default="tests/benchmarks/eval/generations.jsonl")
    arguments = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in Path(arguments.generations).read_text().splitlines()]
    arms = [
        "human",
        *sorted(
            (key for key in rows[0] if key.startswith("delta_")),
            key=lambda key: float(key.split("_")[1]),
        ),
    ]

    tokenizer = AutoTokenizer.from_pretrained(arguments.judge)
    model = (
        AutoModelForCausalLM.from_pretrained(arguments.judge, dtype=torch.float16)
        .to("cuda" if torch.cuda.is_available() else "cpu")
        .eval()
    )
    print(f"judge: {arguments.judge}   texts per arm: {len(rows)}\n")
    print(f"| {'arm':>9} | {'median PPL':>10} | {'mean PPL':>9} | {'vs delta 0':>10} |")
    print(f"|{'-' * 11}|{'-' * 12}|{'-' * 11}|{'-' * 12}|")

    reference = None
    for arm in arms:
        texts = [row["human"] if arm == "human" else row[arm]["text"] for row in rows]
        values = perplexity(model, tokenizer, texts)
        median = float(np.median(values))
        if arm == "delta_0":
            reference = median
        relative = "-" if reference is None or arm == "human" else f"{median / reference:+.2f}x"
        if arm == "delta_0":
            relative = "1.00x"
        print(f"| {arm:>9} | {median:>10.2f} | {float(values.mean()):>9.2f} | {relative:>10} |")


if __name__ == "__main__":
    main()
