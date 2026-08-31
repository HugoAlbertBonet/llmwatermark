"""Score the generated arms: detection power, and what the watermark costs in quality.

Reads the JSONL written by eval_generate.py so every metric is computed over the same
generations. ROC and AUC are computed in numpy rather than pulled from scikit-learn - the
project's dependency discipline is worth more than fifteen lines of code.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from llmwatermark.config import WatermarkConfig
from llmwatermark.detector import DEFAULT_Z_THRESHOLD, WatermarkDetector
from llmwatermark.errors import DetectionError

KEY = b"quality-evaluation-key-0123456789"


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Area under the ROC curve, as the Mann-Whitney statistic (ties count a half)."""
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks over ties
    values = np.concatenate([positive, negative])[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    positive_ranks = ranks[: len(positive)].sum()
    return float(
        (positive_ranks - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative))
    )


def tpr_at_fpr(positive: np.ndarray, negative: np.ndarray, target: float) -> float:
    """True-positive rate at the threshold that gives at most `target` false positives."""
    threshold = np.quantile(negative, 1.0 - target)
    return float((positive > threshold).mean())


def distinct_n(token_ids: list[int], order: int) -> float:
    if len(token_ids) <= order:
        return 0.0
    grams = [tuple(token_ids[i : i + order]) for i in range(len(token_ids) - order + 1)]
    return len(set(grams)) / len(grams)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--generations", default="tests/benchmarks/eval/generations.jsonl")
    arguments = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    rows = [json.loads(line) for line in Path(arguments.generations).read_text().splitlines()]
    arms = sorted(
        (key for key in rows[0] if key.startswith("delta_")),
        key=lambda key: float(key.split("_")[1]),
    )
    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    vocab_size = int(AutoConfig.from_pretrained(arguments.model).vocab_size)
    config = WatermarkConfig.from_tokenizer(tokenizer, vocab_size=vocab_size, secret_key=KEY)
    detector = WatermarkDetector(config, tokenizer)

    def score(texts: list[str]) -> np.ndarray:
        values = []
        for text in texts:
            try:
                values.append(detector.detect(text).z_score)
            except DetectionError:
                continue
        return np.array(values)

    human = score([row["human"] for row in rows])
    print(f"prompts: {len(rows)}   categories: {len(Counter(r['category'] for r in rows))}")
    print(f"human reference texts scored: {len(human)}  mean z {human.mean():+.2f}\n")

    baseline = score([row["delta_0"]["text"] for row in rows])
    negatives = np.concatenate([human, baseline])

    print(
        f"| {'delta':>5} | {'mean z':>7} | {'detected':>9} | {'AUC':>6} | "
        f"{'TPR@1%':>7} | {'TPR@.01%':>8} | {'distinct-3':>10} | {'tokens':>6} |"
    )
    print(f"|{'-' * 7}|{'-' * 9}|{'-' * 11}|{'-' * 8}|{'-' * 9}|{'-' * 10}|{'-' * 12}|{'-' * 8}|")
    for arm in arms:
        delta = float(arm.split("_")[1])
        values = score([row[arm]["text"] for row in rows])
        # Re-tokenize the decoded text: the stored token_ids are padded to the generation
        # cap, so measuring length or diversity on them measures the padding.
        encoded = [
            tokenizer(row[arm]["text"], add_special_tokens=False)["input_ids"] for row in rows
        ]
        lengths = [len(ids) for ids in encoded]
        diversity = float(np.mean([distinct_n(ids, 3) for ids in encoded]))
        detected = float((values > DEFAULT_Z_THRESHOLD).mean())
        if delta == 0:
            auc = tpr1 = tpr001 = float("nan")
        else:
            auc = roc_auc(values, negatives)
            tpr1 = tpr_at_fpr(values, negatives, 0.01)
            tpr001 = tpr_at_fpr(values, negatives, 0.0001)
        print(
            f"| {delta:>5.0f} | {values.mean():>+7.2f} | {detected:>9.1%} | {auc:>6.3f} | "
            f"{tpr1:>7.1%} | {tpr001:>8.1%} | {diversity:>10.3f} | {np.mean(lengths):>6.0f} |"
        )


if __name__ == "__main__":
    main()
