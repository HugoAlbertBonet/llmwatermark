"""Render how the detector reached its verdict, as a self-contained HTML page.

Every token is coloured by whether it was on the greenlist, whether it counted toward the
score, and if not, why not. Hovering shows the context that seeded each position.

    python examples/show_decision.py > decision.html
"""

from __future__ import annotations

import sys

import numpy as np

from llmwatermark import WatermarkConfig
from llmwatermark.detector import WatermarkDetector
from llmwatermark.greenlist import green_mask, token_id_range
from llmwatermark.seeding import SeedTable

VOCAB_SIZE = 4096
KEY = b"decision-view-example-key"


class ToyTokenizer:
    """A synthetic vocabulary, so the example runs with no download and no model."""

    def __init__(self, size: int) -> None:
        self._pieces = [f"tok{index} " for index in range(size)]

    def __len__(self) -> int:
        return len(self._pieces)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._pieces[index] for index in ids]


def watermarked_ids(config: WatermarkConfig, length: int) -> list[int]:
    """Sample from each position's greenlist, which is what a watermarked model does."""
    rng = np.random.default_rng(0)
    table = SeedTable.for_config(config)
    ids = token_id_range(config.vocab_size, config.mix_width)
    sequence = [int(rng.integers(config.vocab_size))]
    for step in range(220):
        # A repeated stretch part way through, so the deduplication rule is visible.
        if 120 < step < 160:
            sequence.append(sequence[-2] if len(sequence) > 1 else 7)
            continue
        seeds = table.seeds([sequence[-config.h :]], config.scheme)
        mask = green_mask(seeds, ids, config.green_divisor, config.mix_width)[0]
        sequence.append(int(rng.choice(np.flatnonzero(mask))))
    return sequence


def main() -> None:
    tokenizer = ToyTokenizer(VOCAB_SIZE)
    config = WatermarkConfig.from_tokenizer(
        tokenizer, vocab_size=VOCAB_SIZE, secret_key=KEY, delta=2.0
    )
    result = WatermarkDetector(config, tokenizer).detect(watermarked_ids(config, 220))
    print(result.summary(), file=sys.stderr)
    print(result.to_html(full_document=True))


if __name__ == "__main__":
    main()
