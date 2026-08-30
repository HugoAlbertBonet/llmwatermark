"""End-to-end throughput on transformers: watermark off against watermark on.

The number a user actually experiences. The gap between this and bench_kernel.py is the
cost of the integration rather than the arithmetic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _harness import compare, describe_device, noise_floor

from llmwatermark.adapters.transformers import (
    config_for_model,
    unwatermark,
    watermark,
)

PROMPT = "Explain in detail how a modern operating system schedules processes:"
NEW_TOKENS = 128
REPEATS = 20


def run(model: object, tokenizer: object, batch: int) -> None:
    prompts = [PROMPT] * batch
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)  # type: ignore[operator,attr-defined]

    def generate() -> None:
        torch.manual_seed(0)
        with torch.no_grad():
            model.generate(  # type: ignore[attr-defined]
                **encoded,
                max_new_tokens=NEW_TOKENS,
                min_new_tokens=NEW_TOKENS,
                do_sample=True,
                top_k=0,
            )

    config = config_for_model(model, tokenizer, secret_key=b"benchmark-key-0123456789")

    def baseline() -> None:
        unwatermark(model)
        generate()

    def watermarked() -> None:
        watermark(model, config)
        generate()

    off, on, overhead = compare(baseline, watermarked, repeats=REPEATS, warmup=2)
    tokens = batch * NEW_TOKENS
    budget = 5.0 if batch == 1 else 2.0
    noise = noise_floor(off, on)
    within_budget = "within" if overhead <= budget else "OVER"
    verdict = "below noise" if abs(overhead) < noise else within_budget
    print(
        f"| {batch:>5} | {off.median:>8.0f} ms | {on.median:>8.0f} ms | "
        f"{tokens / off.median * 1e3:>8.0f} | {tokens / on.median * 1e3:>8.0f} | "
        f"{overhead:>+6.2f}% | {noise:>5.2f}% | {budget:>4.0f}% | {verdict} |"
    )
    unwatermark(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--batches", default="1,8,32")
    arguments = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            arguments.model, dtype=torch.float16 if device == "cuda" else torch.float32
        )
        .to(device)
        .eval()
    )

    print(f"device: {describe_device()}")
    print(f"model: {arguments.model}, {NEW_TOKENS} new tokens per sequence\n")
    print(
        f"| {'batch':>5} | {'baseline':>11} | {'watermark':>11} | {'tok/s':>8} | "
        f"{'tok/s':>8} | {'delta':>7} | {'noise':>6} | {'budget':>6} | verdict |"
    )
    print(
        f"|{'-' * 7}|{'-' * 13}|{'-' * 13}|{'-' * 10}|{'-' * 10}|"
        f"{'-' * 9}|{'-' * 8}|{'-' * 8}|---------|"
    )
    for batch in (int(value) for value in arguments.batches.split(",")):
        run(model, tokenizer, batch)


if __name__ == "__main__":
    main()
