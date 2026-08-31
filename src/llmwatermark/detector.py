"""Detection: deciding whether a token sequence carries the watermark.

Detection needs only a tokenizer and the secret key - never the model, never torch, never
an inference backend. It is deliberately cheap to run anywhere.

It is also asymmetric with generation. Generation must decide green or red for every
candidate in the vocabulary at every step. Detection only asks about the token that was
actually emitted, so the whole pass is ``O(tokens)`` and no greenlist is ever built.

The score is the KGW one-proportion z-test. Under the null - text not produced by this
key - each scored position is an independent Bernoulli(gamma) trial, so::

    z = (green_count - gamma * T) / sqrt(T * gamma * (1 - gamma))

Two details decide whether that number means anything:

* **gamma is the effective gamma.** The greenlist rule is an integer modulus, so a
  requested gamma of 0.3 is really 1/3. Scoring against the requested value biases every
  result.
* **Repeated context n-grams are scored once.** A repeated context reuses the same
  greenlist, so the trials stop being independent and the z-score inflates. On repetitive
  human text that produces confident false accusations. This is a known KGW artifact, not
  an edge case.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from llmwatermark import render
from llmwatermark.config import WatermarkConfig
from llmwatermark.errors import DetectionError, SeedingError
from llmwatermark.greenlist import is_green
from llmwatermark.seeding import SeedTable, gather_seeds
from llmwatermark.vocab import encode_text, piece_text, resolve_id_to_piece

__all__ = [
    "DEFAULT_Z_THRESHOLD",
    "MIN_SCORED_TOKENS",
    "SKIP_DUPLICATE_CONTEXT",
    "SKIP_NO_CONTEXT",
    "DetectionResult",
    "TokenRecord",
    "WatermarkDetector",
    "detect",
]

# The KGW paper's operating point. As a one-sided normal tail this is a false-positive
# rate of about 3e-5: roughly one in 30,000 unwatermarked texts is wrongly flagged.
#
# For reference, at gamma = 0.25:  z = 2.0 -> 1 in 44      z = 3.0 -> 1 in 741
#                                  z = 4.0 -> 1 in 31,600  z = 5.0 -> 1 in 3,500,000
#
# Raise it when a false accusation is expensive and you can afford to miss short or
# lightly watermarked passages; lower it only when a human reviews every hit. Override per
# call with detect(..., threshold=...).
DEFAULT_Z_THRESHOLD: Final[float] = 4.0

# Below this many scored tokens the normal approximation behind the z-test is poor and the
# score is not meaningful. Detection refuses rather than guessing.
MIN_SCORED_TOKENS: Final[int] = 16

SKIP_NO_CONTEXT: Final[str] = "no_context"
SKIP_DUPLICATE_CONTEXT: Final[str] = "duplicate_context"


@dataclass(frozen=True, slots=True)
class TokenRecord:
    """One position's contribution to the decision.

    Produced as a by-product of scoring, so the visualization layer can render exactly
    what the detector did without recomputing anything.
    """

    position: int
    token_id: int
    piece: str
    context: tuple[int, ...]
    is_green: bool | None
    """Whether this token was in its greenlist. ``None`` when no greenlist is defined."""
    scored: bool
    """Whether this position contributed to the z-score."""
    skip_reason: str | None
    """``"no_context"``, ``"duplicate_context"``, or ``None`` when the position counted."""


@dataclass(frozen=True)
class DetectionResult:
    """The decision, the evidence behind it, and the per-token reasoning."""

    z_score: float
    p_value: float
    threshold: float
    is_watermarked: bool
    green_count: int
    scored_count: int
    total_tokens: int
    gamma: float
    """The effective greenlist fraction the score was computed against."""
    tokens: tuple[TokenRecord, ...]

    @property
    def green_fraction(self) -> float:
        """The observed green rate among scored positions. Compare against ``gamma``."""
        return self.green_count / self.scored_count if self.scored_count else 0.0

    @property
    def skipped_count(self) -> int:
        return self.total_tokens - self.scored_count

    def summary(self) -> str:
        """The verdict and the evidence behind it, as one short paragraph."""
        return render.summary(self)

    def to_ansi(self, *, color: bool = True) -> str:
        """The per-token decision as a colored stream for a terminal."""
        return render.to_ansi(self, color=color)

    def to_html(self, *, full_document: bool = False) -> str:
        """The per-token decision as self-contained HTML, with hover detail."""
        return render.to_html(self, full_document=full_document)

    def _repr_html_(self) -> str:
        """Render automatically in a Jupyter notebook."""
        return self.to_html()

    def __repr__(self) -> str:
        verdict = "watermarked" if self.is_watermarked else "not watermarked"
        return (
            f"{type(self).__name__}({verdict}, z={self.z_score:.2f}, p={self.p_value:.3g}, "
            f"green={self.green_count}/{self.scored_count} "
            f"({self.green_fraction:.1%} vs gamma {self.gamma:.1%}), "
            f"tokens={self.total_tokens})"
        )


class WatermarkDetector:
    """Detects one watermark, given the key and the tokenizer that defines its vocabulary.

    The tokenizer is checked against the config's vocabulary fingerprint at construction,
    so a mismatched tokenizer fails immediately rather than producing a confident wrong
    answer later.
    """

    def __init__(self, config: WatermarkConfig, tokenizer: object) -> None:
        config.verify_tokenizer(tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        self._table = SeedTable.for_config(config)
        self._id_to_piece = resolve_id_to_piece(tokenizer)

    def detect(
        self,
        sequence: str | Sequence[int] | Any,
        *,
        threshold: float = DEFAULT_Z_THRESHOLD,
        min_tokens: int = MIN_SCORED_TOKENS,
    ) -> DetectionResult:
        """Score a text or token-ID sequence.

        :param sequence: Text to tokenize, or the token IDs directly. Passing IDs avoids
            the retokenization caveat: detection normally has to re-derive IDs from text,
            and that round trip is not guaranteed to recover the IDs the model emitted.
        :param threshold: z above which the text is called watermarked. Defaults to
            :data:`DEFAULT_Z_THRESHOLD`.
        :param min_tokens: Refuse below this many scored tokens. Lower it only knowingly.
        """
        ids = self._as_token_ids(sequence)
        window = self.config.h
        total = len(ids)

        if total <= window:
            raise DetectionError(
                f"too few tokens to score: {total} token(s) give 0 positions with a full "
                f"{window}-token context window, and at least {min_tokens} are needed for "
                "the z-test to mean anything."
            )

        # Reads values, so it happens here on the host and never on a hot path.
        self._table.validate_context_values(ids)

        contexts = np.ascontiguousarray(sliding_window_view(ids, window)[:-1])
        observed = ids[window:]

        seeds = gather_seeds(self._table.values, contexts, self.config.scheme)
        green = is_green(seeds, observed, self.config.green_divisor, self.config.mix_width)
        scored = _first_occurrences(contexts)

        scored_count = int(scored.sum())
        if scored_count < min_tokens:
            raise DetectionError(
                f"too few tokens to score: {scored_count} of {total} survived, but at "
                f"least {min_tokens} are needed for the z-test to mean anything. "
                f"({total - window - scored_count} position(s) were dropped because their "
                "context n-gram repeats, and the first "
                f"{window} have no context window.) Score a longer or less repetitive "
                "passage, or lower min_tokens knowingly."
            )

        gamma = self.config.effective_gamma
        green_count = int(green[scored].sum())
        z_score = (green_count - gamma * scored_count) / math.sqrt(
            scored_count * gamma * (1.0 - gamma)
        )
        return DetectionResult(
            z_score=z_score,
            p_value=0.5 * math.erfc(z_score / math.sqrt(2.0)),
            threshold=threshold,
            is_watermarked=z_score >= threshold,
            green_count=green_count,
            scored_count=scored_count,
            total_tokens=total,
            gamma=gamma,
            tokens=self._records(ids, contexts, green, scored),
        )

    def _records(
        self,
        ids: np.ndarray,
        contexts: np.ndarray,
        green: np.ndarray,
        scored: np.ndarray,
    ) -> tuple[TokenRecord, ...]:
        window = self.config.h
        pieces = [piece_text(piece) for piece in self._id_to_piece(ids.tolist())]
        records = [
            TokenRecord(
                position=position,
                token_id=int(ids[position]),
                piece=pieces[position],
                context=(),
                is_green=None,
                scored=False,
                skip_reason=SKIP_NO_CONTEXT,
            )
            for position in range(min(window, len(ids)))
        ]
        for offset in range(len(contexts)):
            position = offset + window
            was_scored = bool(scored[offset])
            records.append(
                TokenRecord(
                    position=position,
                    token_id=int(ids[position]),
                    piece=pieces[position],
                    context=tuple(int(value) for value in contexts[offset]),
                    is_green=bool(green[offset]),
                    scored=was_scored,
                    skip_reason=None if was_scored else SKIP_DUPLICATE_CONTEXT,
                )
            )
        return tuple(records)

    def _as_token_ids(self, sequence: str | Sequence[int] | Any) -> np.ndarray:
        if isinstance(sequence, str):
            return np.asarray(encode_text(self.tokenizer, sequence), dtype=np.int64)
        ids = np.asarray(sequence, dtype=np.int64)
        if ids.ndim > 1:
            raise SeedingError(
                f"expected a single token sequence, got shape {ids.shape}. Detection "
                "scores one sequence at a time."
            )
        return ids

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(vocab_size={self.config.vocab_size}, "
            f"gamma={self.config.gamma}, scheme={self.config.scheme.value!r})"
        )


def detect(
    sequence: str | Sequence[int] | Any,
    tokenizer: object,
    config: WatermarkConfig,
    *,
    threshold: float = DEFAULT_Z_THRESHOLD,
    min_tokens: int = MIN_SCORED_TOKENS,
) -> DetectionResult:
    """Score one text against one watermark.

    Convenience wrapper around :class:`WatermarkDetector`. Build a detector directly when
    scoring many texts, so the tokenizer is verified once.
    """
    return WatermarkDetector(config, tokenizer).detect(
        sequence, threshold=threshold, min_tokens=min_tokens
    )


def _first_occurrences(contexts: np.ndarray) -> np.ndarray:
    """Mark the first appearance of each distinct context n-gram.

    Repeated contexts reuse one greenlist, so scoring them all would count the same
    evidence many times over and inflate the z-score on repetitive text.
    """
    mask = np.zeros(len(contexts), dtype=bool)
    if len(contexts):
        _, first = np.unique(contexts, axis=0, return_index=True)
        mask[first] = True
    return mask
