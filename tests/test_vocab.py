"""The vocabulary fingerprint: the cross-backend safety check.

A watermark is a partition of *token IDs*. If the generator partitioned 128000 IDs and
the detector partitions 128256, every greenlist differs and detection silently fails.
The fingerprint exists to turn that silent failure into a loud one.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Callable, Sequence

import pytest

from conftest import (
    ExLlamaStyleTokenizer,
    HFStyleTokenizer,
    LlamaCppStyleTokenizer,
    SentencePieceStyleTokenizer,
    make_pieces,
)
from llmwatermark.errors import TokenizerInterfaceError
from llmwatermark.vocab import (
    FINGERPRINT_DOMAIN,
    FINGERPRINT_SAMPLE_SIZE,
    fingerprint_from_tokenizer,
    fingerprint_sample_ids,
    resolve_id_to_piece,
)

VOCAB_SIZES = [1, 2, 7, 255, 256, 257, 512, 128000, 128256]

# Golden vectors over the synthetic vocabulary {id -> f"tok{id}"}. These pin the digest
# format. Changing them invalidates every config ever issued, so they are hardcoded and
# any drift must be a deliberate, documented format bump (see FINGERPRINT_DOMAIN).
GOLDEN_FINGERPRINTS = {
    512: "527307761e39dd314ce40ea32c469de4c4c8f7c7cf48dd84661b30a54ad5ec75",
    128256: "c18cb10f4b382be38bd6effea384882d51594224fbbbba834bcd9cb96f3fc1e9",
}


class TestFingerprintSampleIds:
    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_sample_is_sorted_unique_and_in_range(self, vocab_size: int) -> None:
        ids = fingerprint_sample_ids(vocab_size)
        assert ids == sorted(set(ids))
        assert all(0 <= token_id < vocab_size for token_id in ids)

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_sample_always_covers_both_boundaries(self, vocab_size: int) -> None:
        """The last ID is what distinguishes a padded embedding matrix from a real one."""
        ids = fingerprint_sample_ids(vocab_size)
        assert ids[0] == 0
        assert ids[-1] == vocab_size - 1

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_sample_size_is_capped(self, vocab_size: int) -> None:
        ids = fingerprint_sample_ids(vocab_size)
        assert len(ids) == min(vocab_size, FINGERPRINT_SAMPLE_SIZE)

    def test_sample_is_a_pure_function_of_vocab_size(self) -> None:
        assert fingerprint_sample_ids(128256) == fingerprint_sample_ids(128256)

    @pytest.mark.parametrize("vocab_size", [0, -1])
    def test_rejects_non_positive_vocab_size(self, vocab_size: int) -> None:
        with pytest.raises(ValueError, match="vocab_size"):
            fingerprint_sample_ids(vocab_size)


class TestFingerprintFormat:
    def test_matches_the_documented_byte_layout(self, tokenizer: HFStyleTokenizer) -> None:
        """Recompute the digest independently from the layout documented in vocab.py."""
        vocab_size = len(tokenizer)
        ids = fingerprint_sample_ids(vocab_size)
        digest = hashlib.sha256()
        digest.update(FINGERPRINT_DOMAIN)
        digest.update(vocab_size.to_bytes(8, "big"))
        digest.update(len(ids).to_bytes(4, "big"))
        for token_id in ids:
            piece = tokenizer.convert_ids_to_tokens([token_id])[0].encode("utf-8")
            digest.update(token_id.to_bytes(4, "big"))
            digest.update(len(piece).to_bytes(4, "big"))
            digest.update(piece)

        assert fingerprint_from_tokenizer(tokenizer, vocab_size) == digest.hexdigest()

    def test_is_64_lowercase_hex_characters(self, tokenizer: HFStyleTokenizer) -> None:
        fingerprint = fingerprint_from_tokenizer(tokenizer, len(tokenizer))
        assert len(fingerprint) == 64
        assert set(fingerprint) <= set("0123456789abcdef")

    def test_is_stable_across_processes(self) -> None:
        """PYTHONHASHSEED must not reach this digest."""
        probe = (
            "from llmwatermark.vocab import fingerprint_from_tokenizer; "
            "print(fingerprint_from_tokenizer(lambda ids: [f'tok{i}' for i in ids], 512))"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", probe], capture_output=True, text=True, check=True
            ).stdout.strip()
            for _ in range(2)
        }
        assert len(runs) == 1
        assert runs.pop() == fingerprint_from_tokenizer(lambda ids: [f"tok{i}" for i in ids], 512)

    @pytest.mark.parametrize("vocab_size", sorted(GOLDEN_FINGERPRINTS))
    def test_matches_the_committed_golden_vectors(
        self, vocab_size: int, lazy_id_to_piece: Callable[[Sequence[int]], list[str]]
    ) -> None:
        """Locks the digest format across Python and library versions, forever."""
        assert (
            fingerprint_from_tokenizer(lazy_id_to_piece, vocab_size)
            == GOLDEN_FINGERPRINTS[vocab_size]
        )

    def test_length_prefixing_prevents_concatenation_collisions(self) -> None:
        """('ab', 'c') and ('a', 'bc') must not hash alike."""
        left = fingerprint_from_tokenizer(HFStyleTokenizer(["ab", "c"]), 2)
        right = fingerprint_from_tokenizer(HFStyleTokenizer(["a", "bc"]), 2)
        assert left != right

    def test_handles_non_ascii_and_byte_level_pieces(self) -> None:
        exotic = ["▁café", "\U0001f642", "Ġthe", " ", "a" * 300]
        fingerprint = fingerprint_from_tokenizer(HFStyleTokenizer(exotic), len(exotic))
        assert len(fingerprint) == 64

    def test_str_and_bytes_pieces_agree(self) -> None:
        """llama.cpp hands back bytes; transformers hands back str. Same vocab, same digest."""
        pieces = make_pieces(64)
        from_str = fingerprint_from_tokenizer(HFStyleTokenizer(pieces), 64)
        from_bytes = fingerprint_from_tokenizer(LlamaCppStyleTokenizer(pieces), 64)
        assert from_str == from_bytes


class TestFingerprintSensitivity:
    def test_padded_vocab_size_changes_the_fingerprint(
        self, lazy_id_to_piece: Callable[[Sequence[int]], list[str]]
    ) -> None:
        """The exact Llama-3 trap: len(tokenizer)=128000 vs config.vocab_size=128256."""
        real = fingerprint_from_tokenizer(lazy_id_to_piece, 128000)
        padded = fingerprint_from_tokenizer(lazy_id_to_piece, 128256)
        assert real != padded

    def test_off_by_one_vocab_size_changes_the_fingerprint(
        self, lazy_id_to_piece: Callable[[Sequence[int]], list[str]]
    ) -> None:
        smaller = fingerprint_from_tokenizer(lazy_id_to_piece, 512)
        larger = fingerprint_from_tokenizer(lazy_id_to_piece, 513)
        assert smaller != larger

    def test_changing_one_sampled_piece_changes_the_fingerprint(self) -> None:
        pieces = make_pieces(256)
        original = fingerprint_from_tokenizer(HFStyleTokenizer(pieces), 256)
        pieces[128] = "MUTATED"
        assert fingerprint_from_tokenizer(HFStyleTokenizer(pieces), 256) != original

    def test_a_different_vocabulary_of_the_same_size_changes_the_fingerprint(self) -> None:
        other = fingerprint_from_tokenizer(HFStyleTokenizer(make_pieces(256, "piece")), 256)
        assert other != fingerprint_from_tokenizer(HFStyleTokenizer(make_pieces(256)), 256)


class TestTokenizerInterfaces:
    def test_every_interface_agrees_on_the_same_vocabulary(self, pieces: list[str]) -> None:
        """Cross-backend agreement is the point; which API supplied the pieces must not matter."""
        fingerprints = {
            fingerprint_from_tokenizer(HFStyleTokenizer(pieces), 512),
            fingerprint_from_tokenizer(ExLlamaStyleTokenizer(pieces), 512),
            fingerprint_from_tokenizer(SentencePieceStyleTokenizer(pieces), 512),
            fingerprint_from_tokenizer(LlamaCppStyleTokenizer(pieces), 512),
            fingerprint_from_tokenizer(lambda ids: [pieces[i] for i in ids], 512),
        }
        assert len(fingerprints) == 1

    def test_unsupported_object_names_the_accepted_interfaces(self) -> None:
        with pytest.raises(TokenizerInterfaceError) as excinfo:
            resolve_id_to_piece(object())
        message = str(excinfo.value)
        assert "convert_ids_to_tokens" in message
        assert "get_id_to_piece_list" in message
        assert "id_to_piece" in message

    def test_per_id_callable_is_rejected_with_a_signature_hint(self) -> None:
        """A bare callable must map a *sequence* of IDs, not a single ID."""
        with pytest.raises(TokenizerInterfaceError, match="sequence"):
            fingerprint_from_tokenizer(lambda token_id: f"tok{token_id}", 8)

    def test_tokenizer_smaller_than_declared_vocab_size_is_reported_clearly(self) -> None:
        """Sampling ID 511 from a 256-piece tokenizer must not surface as a raw IndexError."""
        with pytest.raises(ValueError, match="512"):
            fingerprint_from_tokenizer(HFStyleTokenizer(make_pieces(256)), 512)
