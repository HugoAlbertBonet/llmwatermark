"""Fixtures shared by the core test suite.

The fakes here stand in for the tokenizer interfaces the real backends expose, so the
fingerprint logic can be tested without downloading a model or installing a backend.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest


def make_pieces(count: int, prefix: str = "tok") -> list[str]:
    """A deterministic, synthetic vocabulary of `count` pieces."""
    return [f"{prefix}{index}" for index in range(count)]


class HFStyleTokenizer:
    """Mimics `transformers`: a batched `convert_ids_to_tokens` plus `__len__`."""

    def __init__(self, pieces: Sequence[str]) -> None:
        self._pieces = list(pieces)

    def __len__(self) -> int:
        return len(self._pieces)

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
        return [self._pieces[index] for index in ids]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Whitespace round trip, so detection can be tested from text as well as IDs."""
        lookup = {piece: index for index, piece in enumerate(self._pieces)}
        return [lookup[piece] for piece in text.split()]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._pieces[index] for index in ids)


class ExLlamaStyleTokenizer:
    """Mimics ExLlamaV2: one call returning the whole id -> piece list."""

    def __init__(self, pieces: Sequence[str]) -> None:
        self._pieces = list(pieces)

    def get_id_to_piece_list(self) -> list[str]:
        return list(self._pieces)


class SentencePieceStyleTokenizer:
    """Mimics sentencepiece: a single-id `id_to_piece`."""

    def __init__(self, pieces: Sequence[str]) -> None:
        self._pieces = list(pieces)

    def id_to_piece(self, token_id: int) -> str:
        return self._pieces[token_id]


class LlamaCppStyleTokenizer:
    """Mimics llama.cpp: pieces come back as raw bytes, not str."""

    def __init__(self, pieces: Sequence[str]) -> None:
        self._pieces = [piece.encode("utf-8") for piece in pieces]

    def id_to_piece(self, token_id: int) -> bytes:
        return self._pieces[token_id]


@pytest.fixture
def pieces() -> list[str]:
    return make_pieces(512)


@pytest.fixture
def tokenizer(pieces: list[str]) -> HFStyleTokenizer:
    return HFStyleTokenizer(pieces)


@pytest.fixture
def lazy_id_to_piece() -> Callable[[Sequence[int]], list[str]]:
    """A vocabulary too large to materialize, exposed as a plain callable."""

    def id_to_piece(ids: Sequence[int]) -> list[str]:
        return [f"tok{index}" for index in ids]

    return id_to_piece
