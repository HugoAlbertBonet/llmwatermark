"""Generate the watermarked and unwatermarked answer sets for evaluation.

Same prompts, same seeds, same token budget across every arm - only delta changes. Writes
one JSONL row per prompt carrying the human reference and every generated arm, so the
scoring, detection and judging steps all read the same file and cannot drift apart.

Data: databricks/databricks-dolly-15k, CC BY-SA 3.0.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from llmwatermark.adapters.transformers import config_for_model, unwatermark, watermark

DATASET = "databricks/databricks-dolly-15k"
KEY = b"quality-evaluation-key-0123456789"
DELTAS = (0.0, 1.0, 2.0, 4.0, 6.0)


def prompts(limit: int, minimum_words: int) -> list[dict[str, str]]:
    """A stratified sample across Dolly's categories, with the human answer kept."""
    from datasets import load_dataset

    rows = load_dataset(DATASET, split="train")
    by_category: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["context"] or len(row["response"].split()) < minimum_words:
            continue
        by_category.setdefault(row["category"], []).append(
            {
                "instruction": row["instruction"],
                "human": row["response"].strip(),
                "category": row["category"],
            }
        )
    per_category = max(1, limit // max(len(by_category), 1))
    sample: list[dict[str, str]] = []
    for entries in by_category.values():
        sample.extend(entries[:per_category])
    return sample[:limit]


def generate(model, tokenizer, batch, new_tokens: int, seed: int) -> list[dict]:
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["instruction"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in batch
    ]
    encoded = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[:, encoded["input_ids"].shape[1] :]
    results = []
    for row in generated:
        text = tokenizer.decode(row, skip_special_tokens=True)
        # Trim the padding generate() adds so rows reach a common width; measuring length
        # or diversity on padded ids measures the padding.
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        results.append({"text": text, "token_ids": ids})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default="tests/benchmarks/eval/generations.jsonl")
    arguments = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(arguments.model, dtype=torch.float16)
        .to("cuda" if torch.cuda.is_available() else "cpu")
        .eval()
    )

    items = prompts(arguments.limit, minimum_words=40)
    print(f"model: {arguments.model}\nprompts: {len(items)}   deltas: {DELTAS}")

    base = config_for_model(model, tokenizer, secret_key=KEY)
    # "control" is a second unwatermarked pass with different seeds. Pairing it against
    # delta_0 gives the judge a comparison that is indistinguishable by construction: if a
    # judge does not score those near 50/50, its verdict on the real arms means nothing.
    arms: list[tuple[str, float, int]] = [("control", 0.0, 10_000)]
    arms += [(f"delta_{delta:g}", delta, 0) for delta in DELTAS]

    for arm, delta, seed_offset in arms:
        unwatermark(model)
        if delta > 0:
            watermark(model, replace(base, delta=delta))
        done = 0
        for start in range(0, len(items), arguments.batch):
            chunk = items[start : start + arguments.batch]
            for item, result in zip(
                chunk,
                generate(model, tokenizer, chunk, arguments.new_tokens, seed=start + seed_offset),
                strict=True,
            ):
                item[arm] = result
            done += len(chunk)
        print(f"  {arm}: {done} generations")
    unwatermark(model)

    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
