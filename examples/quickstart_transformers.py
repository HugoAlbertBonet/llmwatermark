"""Watermark a transformers model, generate, and detect. The whole loop in one file.

pip install "llmwatermark[transformers]"
python examples/quickstart_transformers.py
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmwatermark import generate_secret_key
from llmwatermark.adapters.transformers import config_for_model, unwatermark, watermark
from llmwatermark.detector import WatermarkDetector

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Explain in a few sentences why the sky is blue."


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).eval()

    # Keep this key. Anyone holding it can both detect and forge the watermark, and
    # without it the text you generate today can never be attributed.
    key = generate_secret_key()
    config = config_for_model(model, tokenizer, secret_key=key)

    watermark(model, config)
    inputs = tokenizer(PROMPT, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.8)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    print(text.strip(), "\n")

    # Detection needs the tokenizer and the key. Not the model, not torch, not a GPU.
    result = WatermarkDetector(config, tokenizer).detect(text)
    print(result.summary())

    # Same prompt, watermark removed, for comparison.
    unwatermark(model)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.8)
    plain = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print("\nunwatermarked, same prompt:")
    print(WatermarkDetector(config, tokenizer).detect(plain).summary())


if __name__ == "__main__":
    main()
