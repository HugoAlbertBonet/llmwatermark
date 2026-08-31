"""End-to-end throughput on llama.cpp: watermark off against watermark on.

A different cost profile from the other two backends. llama.cpp's logits processors run on
the host over numpy, so the watermark is not competing with a GPU kernel - it is plain CPU
work between decode steps, and the fixed ~0.25 ms per step measured under torch does not
carry over.

    python tests/benchmarks/bench_llama_cpp.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _harness import compare, noise_floor

from llmwatermark.adapters.llama_cpp import config_for_llama, unwatermark, watermark

REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
PROMPT = "Explain in detail how a modern operating system schedules processes:"
NEW_TOKENS = 128
REPEATS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--filename", default=FILENAME)
    parser.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    arguments = parser.parse_args()

    import llama_cpp
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    offload = llama_cpp.llama_supports_gpu_offload()
    llama = Llama(
        model_path=hf_hub_download(arguments.repo, arguments.filename),
        n_ctx=1024,
        n_gpu_layers=-1 if offload else 0,
        verbose=False,
        seed=0,
    )
    config = config_for_llama(llama, secret_key=b"benchmark-key-0123456789", delta=2.0)

    def run() -> None:
        llama.create_completion(PROMPT, max_tokens=arguments.new_tokens, temperature=0.8, seed=0)

    def baseline() -> None:
        unwatermark(llama)
        run()

    def watermarked() -> None:
        watermark(llama, config)
        run()

    print(f"model: {arguments.repo}/{arguments.filename}")
    print(f"gpu offload: {offload}   vocab_size: {config.vocab_size}")
    print(f"{arguments.new_tokens} new tokens, {arguments.repeats} interleaved repeats\n")

    off, on, overhead = compare(baseline, watermarked, repeats=arguments.repeats, warmup=2)
    unwatermark(llama)
    noise = noise_floor(off, on)
    tokens = arguments.new_tokens
    print(f"| {'':>12} | {'wall':>10} | {'tok/s':>8} |")
    print(f"|{'-' * 14}|{'-' * 12}|{'-' * 10}|")
    for label, timing in (("baseline", off), ("watermarked", on)):
        print(f"| {label:>12} | {timing.median:>7.0f} ms | {tokens / timing.median * 1e3:>8.1f} |")
    print(f"\noverhead: {overhead:+.2f}%   (noise floor {noise:.2f}%)")
    print(f"per step: {(on.median - off.median) / tokens * 1e3:+.0f} us")


if __name__ == "__main__":
    main()
