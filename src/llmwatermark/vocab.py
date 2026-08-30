"""Vocabulary identity: turning a tokenizer into a fingerprint the detector can check.

The greenlist partition is computed over token IDs. The ID -> string map is stable across
backends because it is baked into the embedding matrix, but the reported *vocabulary size*
is not: models pad the embedding matrix beyond the real tokenizer (Llama-3 has
``len(tokenizer) == 128000`` and ``config.vocab_size == 128256``) and backends disagree on
which number they report. Partitioning 128000 IDs instead of 128256 yields entirely
different greenlists and silent detection failure.

The fingerprint pins both numbers down: the declared size, and a sample of the ID -> string
map wide enough to notice a different tokenizer of the same size.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any, Final

from llmwatermark.errors import TokenizerInterfaceError, VocabMismatchError

__all__ = [
    "FINGERPRINT_DOMAIN",
    "FINGERPRINT_SAMPLE_SIZE",
    "encode_text",
    "fingerprint_from_tokenizer",
    "fingerprint_sample_ids",
    "observed_vocab_size",
    "piece_text",
    "resolve_id_to_piece",
]

# Domain separation: this digest must never collide with another use of sha256 here.
# Bump the suffix only alongside a deliberate, documented format change - doing so
# invalidates every config issued so far.
FINGERPRINT_DOMAIN: Final[bytes] = b"llmwatermark/vocab-fingerprint/v2"

# Wide enough to catch a different tokenizer, small enough to be free to compute.
FINGERPRINT_SAMPLE_SIZE: Final[int] = 256

# A batched map from token IDs to their pieces. Pieces may be str (transformers,
# ExLlamaV2, sentencepiece) or bytes (llama.cpp).
IdToPiece = Callable[[Sequence[int]], Sequence[str | bytes]]


def fingerprint_sample_ids(span: int) -> list[int]:
    """The token IDs the fingerprint is computed over, across ``span`` IDs.

    A pure function of ``span`` - no RNG - so two machines sample identically. Both
    boundaries are always included: ID 0 and ID ``span - 1``.

    ``span`` is the number of IDs the *tokenizer* can map, which for a padded model is
    smaller than the declared ``vocab_size``. The declared size is folded into the digest
    separately, so a padded and an unpadded configuration still fingerprint differently
    without asking the tokenizer for pieces it does not have.
    """
    if span < 1:
        raise VocabMismatchError(f"vocab_size must be a positive integer, got {span}.")

    count = min(span, FINGERPRINT_SAMPLE_SIZE)
    if count == 1:
        return [0]
    # Evenly spaced in integer arithmetic, so the spacing cannot drift with float width.
    return sorted({(index * (span - 1)) // (count - 1) for index in range(count)})


def observed_vocab_size(tokenizer: object) -> int | None:
    """Best-effort size reported by a tokenizer, for diagnostics only.

    Never used to decide the partition - ``vocab_size`` is always explicit - but a
    mismatch error is far more useful when it can name the number actually seen.
    """
    getter = getattr(tokenizer, "get_id_to_piece_list", None)
    if callable(getter):
        return len(getter())
    try:
        return len(tokenizer)  # type: ignore[arg-type]
    except TypeError:
        pass
    size = getattr(tokenizer, "vocab_size", None)
    return size if isinstance(size, int) and not isinstance(size, bool) else None


def resolve_id_to_piece(tokenizer: object) -> IdToPiece:
    """Adapt any supported tokenizer object to one batched ID -> piece callable.

    Supported, in priority order:

    * ``convert_ids_to_tokens(ids)`` - transformers, and every backend that exposes an
      HF tokenizer (vLLM, SGLang, TensorRT-LLM, LMDeploy, mlx-lm).
    * ``get_id_to_piece_list()`` - ExLlamaV2/V3.
    * ``id_to_piece(id)`` - sentencepiece, and llama.cpp wrappers.
    * a plain callable taking a sequence of IDs and returning a sequence of pieces, for
      anything else (for example ``lambda ids: [llama_token_get_text(model, i) for i in ids]``).
    """
    batched = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(batched):
        return lambda ids: _check_pieces(batched(list(ids)), ids, "convert_ids_to_tokens")

    listing = getattr(tokenizer, "get_id_to_piece_list", None)
    if callable(listing):
        pieces = listing()
        return lambda ids: _check_pieces([pieces[i] for i in ids], ids, "get_id_to_piece_list")

    single = getattr(tokenizer, "id_to_piece", None)
    if callable(single):
        return lambda ids: _check_pieces([single(i) for i in ids], ids, "id_to_piece")

    if callable(tokenizer):
        return lambda ids: _check_pieces(tokenizer(list(ids)), ids, "callable")

    raise TokenizerInterfaceError(
        f"{type(tokenizer).__name__} exposes no token ID to token string mapping. Pass a "
        "tokenizer with convert_ids_to_tokens(ids) (transformers), get_id_to_piece_list() "
        "(ExLlamaV2), or id_to_piece(id) (sentencepiece), or a callable taking a sequence "
        "of token IDs and returning a sequence of pieces."
    )


def _check_pieces(pieces: Any, ids: Sequence[int], source: str) -> Sequence[str | bytes]:
    """Reject a mapping that returned something other than one piece per requested ID."""
    if isinstance(pieces, (str, bytes)):
        raise TokenizerInterfaceError(
            f"{source} returned a single {type(pieces).__name__} for {len(ids)} token IDs. "
            "It must accept a sequence of token IDs and return a sequence of pieces."
        )
    if not isinstance(pieces, Sequence) or len(pieces) != len(ids):
        got = len(pieces) if isinstance(pieces, Sequence) else type(pieces).__name__
        raise TokenizerInterfaceError(
            f"{source} returned {got} pieces for {len(ids)} token IDs. It must accept a "
            "sequence of token IDs and return one piece per ID, in the same order."
        )
    return pieces


def encode_text(tokenizer: object, text: str) -> list[int]:
    """Tokenize text to IDs for detection.

    Special tokens are suppressed where the tokenizer supports it: they were never
    produced by the model, and letting them into the sequence shifts every context window
    by one, which changes every greenlist.
    """
    encoder = getattr(tokenizer, "encode", None)
    if not callable(encoder):
        raise TokenizerInterfaceError(
            f"{type(tokenizer).__name__} has no encode() method, so text cannot be "
            "tokenized. Pass token IDs directly, or use a tokenizer that can encode."
        )
    try:
        ids = encoder(text, add_special_tokens=False)
    except TypeError:
        ids = encoder(text)
    return _flatten_ids(ids)


def _flatten_ids(ids: Any) -> list[int]:
    """Normalize whatever the tokenizer returned into a flat list of ints."""
    listed = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    while listed and isinstance(listed[0], list):
        if len(listed) != 1:
            raise TokenizerInterfaceError(
                f"encode() returned {len(listed)} sequences; detection scores one at a time."
            )
        listed = listed[0]
    return [int(value) for value in listed]


def piece_text(piece: str | bytes) -> str:
    """A displayable form of a token piece, for the per-token detection records."""
    if isinstance(piece, str):
        return piece
    return bytes(piece).decode("utf-8", errors="replace")


def _piece_bytes(piece: str | bytes) -> bytes:
    """Normalize a piece to bytes so str and bytes backends fingerprint identically."""
    if isinstance(piece, str):
        return piece.encode("utf-8")
    if isinstance(piece, (bytes, bytearray)):
        return bytes(piece)
    raise TokenizerInterfaceError(f"token pieces must be str or bytes, got {type(piece).__name__}.")


def fingerprint_from_tokenizer(tokenizer: object, vocab_size: int) -> str:
    """Compute the vocabulary fingerprint as 64 lowercase hex characters.

    Byte layout, which is part of the on-disk format and must not change silently::

        sha256(
            FINGERPRINT_DOMAIN
            || uint64_be(vocab_size)       # what the model generates over
            || uint64_be(span)             # how many IDs the tokenizer can map
            || uint32_be(number of sampled IDs)
            || for each sampled ID, ascending:
                   uint32_be(token_id)
                   || uint32_be(len(piece_bytes))
                   || piece_bytes            # utf-8 for str pieces, raw for bytes pieces
        )

    Every field is length-prefixed so that two different vocabularies cannot serialize to
    the same byte string (``("ab", "c")`` must not collide with ``("a", "bc")``).

    Models pad the embedding matrix past the real tokenizer, so ``vocab_size`` is
    routinely larger than the number of pieces the tokenizer can produce - OPT-125m
    generates over 50272 IDs from 50265 pieces, Llama-3 over 128256 from 128000. Sampling
    is therefore bounded by the tokenizer, while ``vocab_size`` enters the digest as a
    field of its own: partitioning 128000 IDs still fingerprints differently from
    partitioning 128256, and no piece is ever requested that does not exist.
    """
    if vocab_size < 1:
        raise VocabMismatchError(f"vocab_size must be a positive integer, got {vocab_size}.")
    id_to_piece = resolve_id_to_piece(tokenizer)

    observed = observed_vocab_size(tokenizer)
    span = vocab_size if observed is None else min(vocab_size, observed)
    ids = fingerprint_sample_ids(span)

    try:
        pieces = id_to_piece(ids)
    except IndexError as exc:  # a mapping that cannot report its own size
        raise VocabMismatchError(
            f"declared vocab_size={vocab_size} but this tokenizer has no piece for one of "
            f"the sampled token IDs (up to {ids[-1]}). Pass the vocab_size the text was "
            "generated with, or the tokenizer that matches the watermark."
        ) from exc

    digest = hashlib.sha256()
    digest.update(FINGERPRINT_DOMAIN)
    digest.update(vocab_size.to_bytes(8, "big"))
    digest.update(span.to_bytes(8, "big"))
    digest.update(len(ids).to_bytes(4, "big"))
    for token_id, piece in zip(ids, pieces, strict=True):
        raw = _piece_bytes(piece)
        digest.update(token_id.to_bytes(4, "big"))
        digest.update(len(raw).to_bytes(4, "big"))
        digest.update(raw)
    return digest.hexdigest()
