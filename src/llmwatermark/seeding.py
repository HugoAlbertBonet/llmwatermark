"""Seed derivation: the secret half of the watermark.

The greenlist at position *t* is a pure function of ``(secret_key, the last h token IDs)``.
This module computes the seed that stands for that context. Nothing here is cached per
row, per batch index or per request: the watermark is stateless, which is what lets it
survive vLLM preemption, beam reordering and speculative rollback.

Determinism is the hard requirement. The seed must be byte-identical across operating
systems, Python versions, numpy versions and torch versions, or text watermarked on one
machine stops detecting on another. Only ``hmac`` and ``hashlib`` decide the value; both
are stdlib and specified by RFC 2104 and FIPS 180-4.

Layout, which is part of the on-disk format and must not change silently::

    token_hash = HMAC-SHA256(secret_key, SEED_DOMAIN || uint32_be(token_id))
                 first 8 bytes, big-endian, masked to 63 bits

The two schemes then differ only in which token hashes they combine:

* LeftHash (h=1): the hash of the single preceding token.
* MinHash (h>=1): the minimum over the hashes of the last h tokens.

Because the primitive is keyed by token ID alone, and not by position, all of its values
are precomputed once into a :class:`SeedTable`. Per-step seeding is then an array gather
with no HMAC on the hot path at all.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from llmwatermark.arrays import is_torch, row_min
from llmwatermark.config import HashScheme, normalize_secret_key, validate_vocab_size
from llmwatermark.errors import SeedingError

if TYPE_CHECKING:
    from llmwatermark.config import WatermarkConfig

__all__ = [
    "MAX_TOKEN_ID",
    "SEED_BITS",
    "SEED_DOMAIN",
    "SEED_MASK",
    "SeedTable",
    "context_matrix",
    "gather_seeds",
    "token_hash",
    "validate_context_shape",
]

# Domain separation: this digest must never collide with the vocabulary fingerprint.
# Bump the suffix only alongside a deliberate format change - doing so invalidates every
# watermark ever issued.
SEED_DOMAIN: Final[bytes] = b"llmwatermark/seed/v1"

# 63 bits, not 64: torch has no unsigned 64-bit integer type, so a 63-bit value is the
# widest that crosses numpy -> torch as a positive int64 with no sign reinterpretation.
SEED_BITS: Final[int] = 63
SEED_MASK: Final[int] = (1 << SEED_BITS) - 1

# The width of the fixed serialization. No production vocabulary approaches this.
TOKEN_ID_BYTES: Final[int] = 4
MAX_TOKEN_ID: Final[int] = (1 << (8 * TOKEN_ID_BYTES)) - 1

_SEED_DIGEST_BYTES: Final[int] = 8


def token_hash(secret_key: bytes | str, token_id: int) -> int:
    """The keyed hash of one token ID, as a 63-bit non-negative integer.

    This is the only place secrecy enters the watermark. Recovering greenlists means
    recovering these values, which means breaking HMAC-SHA256 or guessing the key.
    """
    key = normalize_secret_key(secret_key)
    _validate_token_id(token_id)
    message = SEED_DOMAIN + int(token_id).to_bytes(TOKEN_ID_BYTES, "big")
    digest = hmac.new(key, message, hashlib.sha256).digest()
    return int.from_bytes(digest[:_SEED_DIGEST_BYTES], "big") & SEED_MASK


class SeedTable:
    """Every token ID's keyed hash, precomputed once for one key and vocabulary.

    With h=1 there are only ``vocab_size`` distinct seeds, and MinHash takes the minimum
    over those same per-token values, so one table serves both schemes. Building it costs
    ``vocab_size`` HMAC evaluations once; after that, seeding a whole batch is a gather.

    The table is a value cache - a pure function of ``(secret_key, vocab_size)`` - not
    watermark state. It holds nothing about rows, requests or positions.
    """

    def __init__(self, secret_key: bytes | str, vocab_size: int) -> None:
        key = normalize_secret_key(secret_key)
        validate_vocab_size(vocab_size)
        self._vocab_size = int(vocab_size)
        self._values = _build_table(key, self._vocab_size)
        self._device_tables: dict[str, Any] = {}

    @property
    def values(self) -> np.ndarray:
        """The read-only int64 table, indexed by token ID."""
        return self._values

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @classmethod
    def for_config(cls, config: WatermarkConfig) -> SeedTable:
        """Return the shared table for a config, building it at most once per key.

        Detection creates a table per call; rebuilding 128k HMACs each time would make
        repeated detection unusable, so tables are memoized by key and vocabulary size.
        The key is already resident in the config, so this retains nothing new.
        """
        return _cached_table(config.secret_key, config.vocab_size)

    def on(self, like: Any = None) -> Any:
        """The table as an array of ``like``'s library and device, cached per device.

        Keeping the table resident on the accelerator is what lets the per-step gather
        happen there, with no host transfer and no synchronisation.
        """
        if like is None or not is_torch(like):
            return self._values
        import torch

        device = str(like.device)
        table = self._device_tables.get(device)
        if table is None:
            # copy(): torch refuses to share memory with a read-only numpy array.
            table = torch.from_numpy(self._values.copy()).to(like.device)
            self._device_tables[device] = table
        return table

    def seeds(self, context: Any, scheme: HashScheme | str) -> np.ndarray:
        """Seed a whole batch at once.

        :param context: Integer array of shape ``(batch, h)`` holding each row's last h
            token IDs, oldest first.
        :param scheme: Which context hash to apply.
        :returns: int64 array of shape ``(batch,)``.

        Fully vectorized: no Python loop over the batch, and no host/device transfer.
        """
        matrix = np.asarray(context)
        resolved = HashScheme.parse(scheme)
        validate_context_shape(matrix, resolved)
        self.validate_context_values(matrix)
        result: np.ndarray = gather_seeds(self._values, matrix, resolved)
        return result

    def validate_context_values(self, matrix: Any) -> None:
        """Check that every context token ID is inside the watermark vocabulary.

        Reads array values, so callers must not run this on device tensors: it would force
        a host synchronisation on the hot path.
        """
        if matrix.size == 0:
            return
        lowest = int(matrix.min())
        highest = int(matrix.max())
        if lowest < 0 or highest >= self._vocab_size:
            offender = lowest if lowest < 0 else highest
            raise SeedingError(
                f"context token ID {offender} lies outside the watermark vocabulary of "
                f"size {self._vocab_size}. Token IDs are never clamped, because clamping "
                "would silently seed a greenlist the detector cannot reproduce. Check that "
                "the config's vocab_size matches the model that produced these tokens."
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(vocab_size={self._vocab_size})"


def gather_seeds(values: Any, context: Any, scheme: HashScheme) -> Any:
    """Seed each row from its context window. Works on numpy arrays and torch tensors.

    Deliberately free of validation: it runs on the hot path, where reading array values
    would mean a device synchronisation. Callers validate on the host beforehand.
    """
    if scheme is HashScheme.LEFTHASH:
        # Width is 1, so gather the single column rather than the whole row.
        return values[context[:, -1]]
    return row_min(values[context])


def validate_context_shape(matrix: Any, scheme: HashScheme) -> None:
    """Check the context array's rank, dtype and width. Never reads its values.

    Safe to call on a device tensor: every check here reads metadata only, so it cannot
    force a host synchronisation.
    """
    if matrix.ndim != 2:
        raise SeedingError(
            f"context must have shape (batch, h), got shape {tuple(matrix.shape)}. Pass "
            "one row per sequence holding its last h token IDs."
        )
    if _is_floating(matrix):
        raise SeedingError(
            f"context must be an integer array of token IDs, got dtype {matrix.dtype}."
        )
    width = matrix.shape[1]
    if width < 1:
        raise SeedingError("context must be at least 1 token wide, got width 0.")
    if scheme is HashScheme.LEFTHASH and width != 1:
        raise SeedingError(
            f"scheme LEFTHASH seeds from the single preceding token, so context must "
            f"have width 1, got {width}. Use scheme='minhash' for a wider window."
        )


def _is_floating(matrix: Any) -> bool:
    if is_torch(matrix):
        return bool(matrix.dtype.is_floating_point or matrix.dtype.is_complex)
    return not np.issubdtype(matrix.dtype, np.integer)


def context_matrix(histories: Sequence[Sequence[int]], h: int) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the last h token IDs of each row, flagging rows with too little history.

    The first h positions of a sequence have no full context window, so no greenlist is
    defined for them. Rather than inventing one, those rows come back marked invalid:
    generation leaves their logits untouched and detection skips those positions.

    Only for ragged host-side histories, as vLLM and SGLang supply them. Backends that
    already hold a rectangular ``(batch, time)`` tensor should slice ``[:, -h:]`` on the
    device instead and never call this.

    :returns: ``(context, valid)`` of shapes ``(batch, h)`` int64 and ``(batch,)`` bool.
        Invalid rows are zero-filled; their seeds are meaningless and must not be used.
    """
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        raise SeedingError(f"h, the context width, must be a positive integer, got {h!r}.")

    rows = list(histories)
    context = np.zeros((len(rows), h), dtype=np.int64)
    valid = np.zeros(len(rows), dtype=np.bool_)
    # Loops over the batch, not over the vocabulary: h ints copied per row.
    for position, history in enumerate(rows):
        if len(history) >= h:
            context[position] = history[-h:]
            valid[position] = True
    return context, valid


def _validate_token_id(token_id: object) -> None:
    if not isinstance(token_id, (int, np.integer)) or isinstance(token_id, bool):
        raise SeedingError(f"token_id must be an integer, got {type(token_id).__name__}.")
    if not 0 <= int(token_id) <= MAX_TOKEN_ID:
        raise SeedingError(f"token_id must lie in [0, {MAX_TOKEN_ID}], got {token_id}.")


def _build_table(secret_key: bytes, vocab_size: int) -> np.ndarray:
    """Evaluate the keyed hash for every token ID, once.

    The HMAC prototype is copied per message so the key's two padded blocks are absorbed
    once rather than re-derived ``vocab_size`` times.
    """
    prototype = hmac.new(secret_key, digestmod=hashlib.sha256)
    values = np.empty(vocab_size, dtype=np.int64)
    for token_id in range(vocab_size):
        mac = prototype.copy()
        mac.update(SEED_DOMAIN + token_id.to_bytes(TOKEN_ID_BYTES, "big"))
        values[token_id] = int.from_bytes(mac.digest()[:_SEED_DIGEST_BYTES], "big") & SEED_MASK
    values.flags.writeable = False
    return values


@lru_cache(maxsize=4)
def _cached_table(secret_key: bytes, vocab_size: int) -> SeedTable:
    return SeedTable(secret_key, vocab_size)
