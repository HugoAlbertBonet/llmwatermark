"""Detect a watermark in text you were handed, with no model anywhere.

This is the real use case: someone gives you an essay and asks whether a model wrote it.
You will not have the token IDs - if you did, you would already know the answer.

    pip install llmwatermark          # numpy only
    python examples/detect_text.py "some text to score"
"""

from __future__ import annotations

import sys

from transformers import AutoTokenizer  # only to obtain a tokenizer

from llmwatermark import WatermarkConfig
from llmwatermark.detector import WatermarkDetector
from llmwatermark.errors import DetectionError

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VOCAB_SIZE = 151936  # model.config.vocab_size, which exceeds len(tokenizer)


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    config = WatermarkConfig.from_tokenizer(
        tokenizer, vocab_size=VOCAB_SIZE, secret_key=b"replace-me-with-your-own-key"
    )
    try:
        result = WatermarkDetector(config, tokenizer).detect(text)
    except DetectionError as error:
        # Refusing beats returning a confident number the data cannot support.
        print(f"cannot score this text: {error}")
        raise SystemExit(1) from error

    print(result.summary())
    print()
    print(result.to_ansi())


if __name__ == "__main__":
    main()
