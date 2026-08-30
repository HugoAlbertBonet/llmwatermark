"""False-positive rate of the detector on real human writing.

The calibration in the core test suite uses uniformly random token IDs, and gets a
textbook standard normal. Real prose is not uniform: it repeats, it has boilerplate, it
reuses phrases. The n-gram deduplication is meant to absorb exactly that, and until now it
had never been measured against actual human text.

If the null distribution is wider than N(0, 1) here, every false-positive rate this project
publishes is wrong, and the fix is a higher threshold rather than a nicer README.

Data: databricks/databricks-dolly-15k, human-written instruction responses, CC BY-SA 3.0.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from llmwatermark.config import WatermarkConfig
from llmwatermark.detector import DEFAULT_Z_THRESHOLD, WatermarkDetector
from llmwatermark.errors import DetectionError

DATASET = "databricks/databricks-dolly-15k"
KEY = b"human-false-positive-evaluation"


def human_responses(minimum_words: int, limit: int) -> list[tuple[str, str]]:
    """Human answers long enough to score, stratified across Dolly's categories."""
    from datasets import load_dataset

    rows = load_dataset(DATASET, split="train")
    by_category: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        text = row["response"].strip()
        if len(text.split()) < minimum_words:
            continue
        by_category.setdefault(row["category"], []).append((row["category"], text))

    per_category = max(1, limit // max(len(by_category), 1))
    sample: list[tuple[str, str]] = []
    for entries in by_category.values():
        sample.extend(entries[:per_category])
    return sample[:limit]


def normal_tail(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--min-words", type=int, default=60)
    parser.add_argument("--keys", type=int, default=4, help="independent keys to average over")
    arguments = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    vocab_size = int(AutoConfig.from_pretrained(arguments.model).vocab_size)
    sample = human_responses(arguments.min_words, arguments.limit)
    print(f"model: {arguments.model}   vocab_size: {vocab_size}")
    print(f"human responses scored: {len(sample)} (>= {arguments.min_words} words)\n")

    scores: list[float] = []
    categories: list[str] = []
    refused = 0
    for index in range(arguments.keys):
        config = WatermarkConfig.from_tokenizer(
            tokenizer, vocab_size=vocab_size, secret_key=KEY + bytes([index])
        )
        detector = WatermarkDetector(config, tokenizer)
        for category, text in sample:
            try:
                scores.append(detector.detect(text).z_score)
                categories.append(category)
            except DetectionError:
                refused += 1

    values = np.array(scores)
    print(f"scored {len(values)} texts across {arguments.keys} independent keys")
    print(f"refused as too short or too repetitive: {refused}\n")
    print(f"  mean z   {values.mean():+.3f}   (nominal 0)")
    print(f"  sd   z   {values.std():.3f}   (nominal 1)")
    print(f"  max  z   {values.max():+.3f}\n")

    print(f"| {'threshold':>9} | {'nominal FPR':>12} | {'observed FPR':>13} | {'count':>6} |")
    print(f"|{'-' * 11}|{'-' * 14}|{'-' * 15}|{'-' * 8}|")
    for threshold in (1.645, 2.0, 3.0, DEFAULT_Z_THRESHOLD, 5.0):
        hits = int((values > threshold).sum())
        print(
            f"| {threshold:>9.3f} | {normal_tail(threshold):>12.2e} | "
            f"{hits / len(values):>13.2e} | {hits:>6} |"
        )

    print("\nby category (mean z / sd z / count):")
    labels = np.array(categories)
    for category in sorted(set(categories)):
        subset = values[labels == category]
        print(f"  {category:<24} {subset.mean():+.3f}  {subset.std():.3f}  {len(subset):>5}")


if __name__ == "__main__":
    main()
