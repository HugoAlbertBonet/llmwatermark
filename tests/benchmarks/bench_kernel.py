"""The watermark step in isolation: greenlist plus bias, no model.

Establishes the floor. If the kernel is already a large share of a decode step here, no
integration work can rescue the budget; if it is small here but the end-to-end cost is
large, the cost is in the integration rather than the arithmetic.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _harness import describe_device, measure

from llmwatermark.config import MixWidth, WatermarkConfig
from llmwatermark.processor import WatermarkProcessor

VOCAB_SIZE = 128256
BATCHES = (1, 2, 4, 8, 16, 32, 64)


def build(width: MixWidth) -> WatermarkConfig:
    return WatermarkConfig(
        secret_key=b"benchmark-key-0123456789",
        vocab_size=VOCAB_SIZE,
        vocab_fingerprint="0" * 64,
        delta=2.0,
        mix_width=width,
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {describe_device()}")
    print(f"vocab_size: {VOCAB_SIZE}, logits dtype: float32\n")

    print(f"| {'batch':>5} | {'width':>5} | {'eager':>12} | {'compiled':>12} | {'speedup':>7} |")
    print(f"|{'-' * 7}|{'-' * 7}|{'-' * 14}|{'-' * 14}|{'-' * 9}|")

    for width in (MixWidth.BITS32, MixWidth.BITS64):
        config = build(width)
        eager_processor = WatermarkProcessor(config, compile=False)
        compiled_processor = WatermarkProcessor(config, compile=True)
        for batch in BATCHES:
            logits = torch.zeros(batch, VOCAB_SIZE, device=device)
            context = torch.full((batch, 1), 7, device=device)

            eager = measure(partial(eager_processor.apply, logits, context))
            compiled = measure(partial(compiled_processor.apply, logits, context))
            ratio = eager.median / compiled.median
            print(
                f"| {batch:>5} | {int(width):>5} | {eager.median:>9.3f} ms | "
                f"{compiled.median:>9.3f} ms | {ratio:>6.1f}x |"
            )


if __name__ == "__main__":
    main()
