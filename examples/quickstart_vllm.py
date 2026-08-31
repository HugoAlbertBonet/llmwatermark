"""Watermark a vLLM engine, generate, and detect.

    pip install "llmwatermark[vllm]"
    python examples/quickstart_vllm.py

vLLM builds its logits processors inside a separate engine process, so the watermark is
wired in through watermark_llm_kwargs() rather than by handing over an object. The secret
key travels in an environment variable, which that helper sets: vLLM logs its engine
configuration at startup, and a key placed in that configuration would be in every log.
"""

from __future__ import annotations

from transformers import AutoConfig, AutoTokenizer

from llmwatermark import WatermarkConfig, generate_secret_key
from llmwatermark.adapters.vllm import watermark_llm_kwargs
from llmwatermark.detector import WatermarkDetector

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPTS = [
    "Explain in a few sentences why the sky is blue.",
    "Give three practical tips for learning to swim as an adult.",
]


def main() -> None:
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    key = generate_secret_key()
    config = WatermarkConfig.from_tokenizer(
        tokenizer,
        # The size the model generates over, which exceeds len(tokenizer) on padded models.
        vocab_size=int(AutoConfig.from_pretrained(MODEL).vocab_size),
        secret_key=key,
    )

    llm = LLM(model=MODEL, max_model_len=1024, **watermark_llm_kwargs(config))
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=220, temperature=0.8, seed=0))

    detector = WatermarkDetector(config, tokenizer)
    for output in outputs:
        text = output.outputs[0].text
        print(f"\n{text.strip()[:200]}...\n")
        print(detector.detect(text).summary())


if __name__ == "__main__":
    main()
