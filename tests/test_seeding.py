"""Seed derivation: the root of the whole watermark.

Every greenlist is a pure function of (secret_key, the last h token IDs). If this module's
output shifts by one bit between two machines, two library versions or two Python
versions, every watermark ever issued stops detecting. The tests here are therefore
heavier on determinism than on behaviour.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import subprocess
import sys
import tokenize
from pathlib import Path

import numpy as np
import pytest

from llmwatermark.config import HashScheme, WatermarkConfig
from llmwatermark.errors import ConfigError, SeedingError
from llmwatermark.seeding import (
    SEED_DOMAIN,
    SEED_MASK,
    SeedTable,
    context_matrix,
    token_hash,
)

KEY = b"demo-key"
VOCAB_SIZE = 512

# Golden vectors under KEY. These pin the byte layout of the seed. Any drift here
# invalidates every watermark ever generated, so they are hardcoded rather than derived.
GOLDEN_TOKEN_HASHES = {
    0: 5429984429020031435,
    1: 1489109541951707350,
    42: 9162896786199901550,
    128255: 6597952119908538536,
}


def reference_token_hash(secret_key: bytes, token_id: int) -> int:
    """The documented layout, recomputed independently of the implementation."""
    message = SEED_DOMAIN + token_id.to_bytes(4, "big")
    digest = hmac.new(secret_key, message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") & SEED_MASK


class TestTokenHash:
    @pytest.mark.parametrize("token_id", [0, 1, 42, 255, 4096, 128255, 2**32 - 1])
    def test_matches_the_documented_hmac_layout(self, token_id: int) -> None:
        assert token_hash(KEY, token_id) == reference_token_hash(KEY, token_id)

    @pytest.mark.parametrize(("token_id", "expected"), sorted(GOLDEN_TOKEN_HASHES.items()))
    def test_matches_the_committed_golden_vectors(self, token_id: int, expected: int) -> None:
        assert token_hash(KEY, token_id) == expected

    @pytest.mark.parametrize("token_id", [0, 1, 42, 128255, 2**32 - 1])
    def test_fits_in_a_positive_int64(self, token_id: int) -> None:
        """torch has no uint64; a 63-bit value crosses numpy -> torch unchanged."""
        value = token_hash(KEY, token_id)
        assert 0 <= value <= SEED_MASK
        assert np.int64(value) == value

    def test_is_stable_across_processes(self) -> None:
        probe = "from llmwatermark.seeding import token_hash; print(token_hash(b'demo-key', 42))"
        outputs = {
            subprocess.run(
                [sys.executable, "-c", probe], capture_output=True, text=True, check=True
            ).stdout.strip()
            for _ in range(2)
        }
        assert outputs == {str(token_hash(KEY, 42))}

    def test_adjacent_token_ids_give_unrelated_seeds(self) -> None:
        """Consecutive tokens must not share greenlist structure."""
        distances = [
            bin(token_hash(KEY, i) ^ token_hash(KEY, i + 1)).count("1") for i in range(256)
        ]
        assert 20 < sum(distances) / len(distances) < 43  # ~31.5 expected over 63 bits

    def test_a_one_bit_key_change_changes_every_seed(self) -> None:
        other = bytes([KEY[0] ^ 0x01]) + KEY[1:]
        assert all(token_hash(KEY, i) != token_hash(other, i) for i in range(256))

    @pytest.mark.parametrize("bad_id", [-1, 2**32, 2**64])
    def test_out_of_range_token_id_is_rejected(self, bad_id: int) -> None:
        with pytest.raises(SeedingError, match="token_id"):
            token_hash(KEY, bad_id)

    @pytest.mark.parametrize("bad_id", [1.0, "42", None, True])
    def test_non_integer_token_id_is_rejected(self, bad_id: object) -> None:
        with pytest.raises(SeedingError, match="token_id"):
            token_hash(KEY, bad_id)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_key", [b"", "", None, 42])
    def test_invalid_secret_key_is_rejected(self, bad_key: object) -> None:
        with pytest.raises(ConfigError, match="secret_key"):
            token_hash(bad_key, 0)  # type: ignore[arg-type]

    def test_str_key_is_accepted_and_matches_its_utf8_bytes(self) -> None:
        assert token_hash("demo-key", 42) == token_hash(b"demo-key", 42)


class TestSeedTable:
    @pytest.fixture
    def table(self) -> SeedTable:
        return SeedTable(KEY, VOCAB_SIZE)

    def test_every_entry_equals_the_primitive(self, table: SeedTable) -> None:
        expected = [token_hash(KEY, i) for i in range(VOCAB_SIZE)]
        assert table.values.tolist() == expected

    def test_is_an_int64_array_of_vocab_size(self, table: SeedTable) -> None:
        assert table.values.dtype == np.int64
        assert table.values.shape == (VOCAB_SIZE,)

    def test_table_is_read_only(self, table: SeedTable) -> None:
        """A mutated table would silently desynchronise the detector from the generator."""
        with pytest.raises(ValueError):
            table.values[0] = 1

    def test_for_config_reuses_one_table_per_key_and_vocab_size(self) -> None:
        """Rebuilding 128k HMACs on every detect() call would be unusable."""
        base = {"vocab_size": VOCAB_SIZE, "vocab_fingerprint": "0" * 64}
        first = SeedTable.for_config(WatermarkConfig(secret_key=KEY, **base))
        second = SeedTable.for_config(WatermarkConfig(secret_key=KEY, **base))
        assert first is second

    def test_for_config_separates_different_keys(self) -> None:
        base = {"vocab_size": VOCAB_SIZE, "vocab_fingerprint": "0" * 64}
        first = SeedTable.for_config(WatermarkConfig(secret_key=b"key-a", **base))
        second = SeedTable.for_config(WatermarkConfig(secret_key=b"key-b", **base))
        assert first is not second
        assert not np.array_equal(first.values, second.values)

    @pytest.mark.parametrize("bad_size", [0, 1, -5])
    def test_unusable_vocab_size_is_rejected(self, bad_size: int) -> None:
        with pytest.raises(ConfigError, match="vocab_size"):
            SeedTable(KEY, bad_size)

    def test_seeds_are_uniform_enough_over_the_greenlist_divisor(self) -> None:
        """A truncation bug that zeroed low bits would show up as a skewed partition."""
        values = SeedTable(KEY, 20000).values
        counts = np.bincount(values % 4, minlength=4)
        assert counts.min() > 0.9 * 5000
        assert counts.max() < 1.1 * 5000


class TestLeftHash:
    @pytest.fixture
    def table(self) -> SeedTable:
        return SeedTable(KEY, VOCAB_SIZE)

    def test_seed_is_the_preceding_token_hash(self, table: SeedTable) -> None:
        context = np.array([[7], [11], [0]], dtype=np.int64)
        expected = [token_hash(KEY, 7), token_hash(KEY, 11), token_hash(KEY, 0)]
        assert table.seeds(context, HashScheme.LEFTHASH).tolist() == expected

    def test_context_wider_than_one_is_rejected(self, table: SeedTable) -> None:
        context = np.zeros((2, 4), dtype=np.int64)
        with pytest.raises(SeedingError, match="LEFTHASH"):
            table.seeds(context, HashScheme.LEFTHASH)


class TestMinHash:
    @pytest.fixture
    def table(self) -> SeedTable:
        return SeedTable(KEY, VOCAB_SIZE)

    def test_seed_is_the_minimum_of_the_window_token_hashes(self, table: SeedTable) -> None:
        window = [3, 17, 250, 8]
        context = np.array([window], dtype=np.int64)
        expected = min(token_hash(KEY, token_id) for token_id in window)
        assert table.seeds(context, HashScheme.MINHASH).tolist() == [expected]

    def test_seed_is_invariant_under_permutation_of_the_window(self, table: SeedTable) -> None:
        window = np.array([[3, 17, 250, 8]], dtype=np.int64)
        shuffled = np.array([[250, 8, 3, 17]], dtype=np.int64)
        assert table.seeds(window, HashScheme.MINHASH) == table.seeds(shuffled, HashScheme.MINHASH)

    def test_editing_the_minimum_token_changes_the_seed(self, table: SeedTable) -> None:
        window = [3, 17, 250, 8]
        hashes = [token_hash(KEY, token_id) for token_id in window]
        argmin = int(np.argmin(hashes))
        edited = list(window)
        edited[argmin] = 500
        original = table.seeds(np.array([window], dtype=np.int64), HashScheme.MINHASH)
        assert table.seeds(np.array([edited], dtype=np.int64), HashScheme.MINHASH) != original

    def test_editing_a_non_minimum_token_leaves_the_seed_alone(self, table: SeedTable) -> None:
        """MinHash's documented edit behaviour: only the argmin position matters."""
        window = [3, 17, 250, 8]
        hashes = [token_hash(KEY, token_id) for token_id in window]
        argmin = int(np.argmin(hashes))
        minimum = hashes[argmin]
        replacement = next(
            token_id for token_id in range(VOCAB_SIZE) if token_hash(KEY, token_id) > minimum
        )
        edited = list(window)
        edited[(argmin + 1) % len(window)] = replacement
        original = table.seeds(np.array([window], dtype=np.int64), HashScheme.MINHASH)
        assert table.seeds(np.array([edited], dtype=np.int64), HashScheme.MINHASH) == original

    def test_repeated_tokens_in_the_window_are_fine(self, table: SeedTable) -> None:
        context = np.array([[5, 5, 5, 5]], dtype=np.int64)
        assert table.seeds(context, HashScheme.MINHASH).tolist() == [token_hash(KEY, 5)]


class TestBatchedSeeding:
    @pytest.fixture
    def table(self) -> SeedTable:
        return SeedTable(KEY, VOCAB_SIZE)

    @pytest.mark.parametrize(
        ("scheme", "width"), [(HashScheme.LEFTHASH, 1), (HashScheme.MINHASH, 4)]
    )
    def test_batched_rows_match_one_row_at_a_time(
        self, table: SeedTable, scheme: HashScheme, width: int
    ) -> None:
        rng = np.random.default_rng(0)
        context = rng.integers(0, VOCAB_SIZE, size=(16, width), dtype=np.int64)
        batched = table.seeds(context, scheme)
        one_by_one = [int(table.seeds(row[None, :], scheme)[0]) for row in context]
        assert batched.tolist() == one_by_one

    def test_returns_int64_of_batch_length(self, table: SeedTable) -> None:
        seeds = table.seeds(np.zeros((5, 4), dtype=np.int64), HashScheme.MINHASH)
        assert seeds.dtype == np.int64
        assert seeds.shape == (5,)

    def test_empty_batch_is_allowed(self, table: SeedTable) -> None:
        seeds = table.seeds(np.zeros((0, 4), dtype=np.int64), HashScheme.MINHASH)
        assert seeds.shape == (0,)

    def test_accepts_a_plain_nested_list(self, table: SeedTable) -> None:
        assert table.seeds([[7]], HashScheme.LEFTHASH).tolist() == [token_hash(KEY, 7)]

    @pytest.mark.parametrize("bad_id", [-1, VOCAB_SIZE, VOCAB_SIZE + 100])
    def test_token_id_outside_the_vocabulary_names_the_value_and_the_size(
        self, table: SeedTable, bad_id: int
    ) -> None:
        context = np.array([[bad_id, 1, 2, 3]], dtype=np.int64)
        with pytest.raises(SeedingError) as excinfo:
            table.seeds(context, HashScheme.MINHASH)
        message = str(excinfo.value)
        assert str(bad_id) in message
        assert str(VOCAB_SIZE) in message

    @pytest.mark.parametrize("shape", [(4,), (2, 2, 2)])
    def test_wrongly_shaped_context_is_rejected(
        self, table: SeedTable, shape: tuple[int, ...]
    ) -> None:
        with pytest.raises(SeedingError, match="shape"):
            table.seeds(np.zeros(shape, dtype=np.int64), HashScheme.MINHASH)

    def test_zero_width_context_is_rejected(self, table: SeedTable) -> None:
        with pytest.raises(SeedingError, match="context"):
            table.seeds(np.zeros((3, 0), dtype=np.int64), HashScheme.MINHASH)

    def test_float_context_is_rejected(self, table: SeedTable) -> None:
        with pytest.raises(SeedingError, match="integer"):
            table.seeds(np.zeros((2, 4), dtype=np.float32), HashScheme.MINHASH)


class TestContextMatrix:
    def test_takes_the_last_h_tokens_of_each_history(self) -> None:
        context, valid = context_matrix([[1, 2, 3, 4, 5], [9, 8, 7, 6]], h=3)
        assert context.tolist() == [[3, 4, 5], [8, 7, 6]]
        assert valid.tolist() == [True, True]

    def test_history_of_exactly_h_tokens_is_valid(self) -> None:
        context, valid = context_matrix([[1, 2]], h=2)
        assert context.tolist() == [[1, 2]]
        assert valid.tolist() == [True]

    def test_short_history_is_marked_invalid(self) -> None:
        """The first h positions have no full context window and get no greenlist."""
        context, valid = context_matrix([[1], [1, 2, 3]], h=3)
        assert valid.tolist() == [False, True]
        assert context[0].tolist() == [0, 0, 0]

    def test_empty_history_is_marked_invalid(self) -> None:
        _, valid = context_matrix([[]], h=1)
        assert valid.tolist() == [False]

    def test_empty_batch_is_allowed(self) -> None:
        context, valid = context_matrix([], h=4)
        assert context.shape == (0, 4)
        assert valid.shape == (0,)

    def test_result_dtypes_are_stable(self) -> None:
        context, valid = context_matrix([[1, 2, 3]], h=2)
        assert context.dtype == np.int64
        assert valid.dtype == np.bool_

    @pytest.mark.parametrize("bad_h", [0, -1])
    def test_invalid_width_is_rejected(self, bad_h: int) -> None:
        with pytest.raises(SeedingError, match="h"):
            context_matrix([[1, 2]], h=bad_h)


class TestDeterminismGuards:
    """The spec forbids nondeterministic sources outright; check the code, not just output."""

    FORBIDDEN_NAMES = frozenset(
        {"random", "randperm", "shuffle", "getrandbits", "hash", "id", "time", "uuid"}
    )

    def test_seeding_module_names_no_nondeterministic_construct(self) -> None:
        """Python's hash() is randomized per process; random/time streams are not pinned."""
        import llmwatermark.seeding as module

        source = Path(module.__file__).read_text()
        names = {
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.NAME
        }
        assert names & self.FORBIDDEN_NAMES == set()

    def test_seeding_module_imports_only_pinned_primitives(self) -> None:
        import llmwatermark.seeding as module

        source = Path(module.__file__).read_text()
        imported = {
            line.split()[1].split(".")[0]
            for line in source.splitlines()
            if line.startswith(("import ", "from "))
        }
        assert imported <= {
            "__future__",
            "collections",
            "functools",
            "hashlib",
            "hmac",
            "numpy",
            "typing",
            "llmwatermark",
        }
