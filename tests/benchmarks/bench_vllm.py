"""End-to-end throughput on vLLM: watermark off against watermark on.

The production number. vLLM's continuous batching means the watermark runs against a batch
whose composition changes every step, so this exercises the tracker as well as the kernel.

Two engines are built in sequence rather than side by side - an 8 GB card will not hold
both - so the arms cannot be interleaved here. Thermal drift is therefore a real confound,
and the reported noise floor is what says whether a difference means anything.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

PROMPT = "Explain in detail how a modern operating system schedules processes:"
NEW_TOKENS = 128
REPEATS = 8


def throughput(
    model: str, batch: int, watermarked: bool, compile_mode: object, eager: bool = True
) -> list[float]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from llmwatermark.adapters.vllm import watermark_llm_kwargs
    from llmwatermark.config import WatermarkConfig

    extra: dict[str, object] = {}
    if watermarked:
        tokenizer = AutoTokenizer.from_pretrained(model)
        from transformers import AutoConfig

        config = WatermarkConfig.from_tokenizer(
            tokenizer,
            vocab_size=int(AutoConfig.from_pretrained(model).vocab_size),
            secret_key=b"benchmark-key-0123456789",
            delta=2.0,
        )
        extra = watermark_llm_kwargs(config, compile=compile_mode)

    engine = LLM(
        model=model,
        gpu_memory_utilization=0.60,
        max_model_len=1024,
        enforce_eager=eager,
        disable_log_stats=True,
        **extra,
    )
    params = SamplingParams(
        max_tokens=NEW_TOKENS, min_tokens=NEW_TOKENS, ignore_eos=True, temperature=0.9, seed=0
    )
    prompts = [PROMPT] * batch

    engine.generate(prompts, params)  # warm up
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        engine.generate(prompts, params)
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--compile", default="auto")
    parser.add_argument("--cuda-graphs", action="store_true")
    arguments = parser.parse_args()

    compile_mode: object = arguments.compile
    if arguments.compile in ("True", "False"):
        compile_mode = arguments.compile == "True"

    eager = not arguments.cuda_graphs
    print(
        f"model: {arguments.model}, batch {arguments.batch}, compile={compile_mode!r}, "
        f"enforce_eager={eager}"
    )
    baseline = throughput(arguments.model, arguments.batch, False, compile_mode, eager)
    treated = throughput(arguments.model, arguments.batch, True, compile_mode, eager)

    off, on = statistics.median(baseline), statistics.median(treated)
    noise = max(statistics.pstdev(baseline), statistics.pstdev(treated)) / off * 100
    tokens = arguments.batch * NEW_TOKENS
    print(f"  baseline  : {off:8.0f} ms   {tokens / off * 1e3:8.0f} tok/s")
    print(f"  watermark : {on:8.0f} ms   {tokens / on * 1e3:8.0f} tok/s")
    print(f"  overhead  : {(on - off) / off * 100:+7.2f}%   (noise floor {noise:.2f}%)")


if __name__ == "__main__":
    main()
