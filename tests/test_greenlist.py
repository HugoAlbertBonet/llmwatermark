"""Greenlist construction: the hot path, and the piece the detector must agree with.

Generation asks "which of the 128k candidates are green" (O(batch x vocab) per step).
Detection asks "was this one emitted token green" (O(tokens)). Both go through the same
expression here, so they cannot drift apart.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from llmwatermark.config import MixWidth
from llmwatermark.errors import SeedingError
from llmwatermark.greenlist import green_mask, is_green, mix, token_id_range

VOCAB_SIZE = 128256
SEED = 0x5A17BEEF

WIDTHS = [MixWidth.BITS32, MixWidth.BITS64]

# Golden vectors pinning the mixer's constants, shifts and sign masking. Drift here means
# every previously watermarked text stops detecting, so the values are hardcoded.
GOLDEN_MIX: dict[MixWidth, dict[tuple[int, int], int]] = {
    MixWidth.BITS32: {
        (0, 0): 0,
        (SEED, 0): 1180388923,
        (SEED, 1): 927355871,
        (SEED, 128255): 1390166083,
        ((1 << 63) - 1, 42): 1263387180,
    },
    MixWidth.BITS64: {
        (0, 0): 0,
        (SEED, 0): 7570096605030653657,
        (SEED, 1): 3710221190730908572,
        (SEED, 128255): 8878638492413451030,
        ((1 << 63) - 1, 42): 7743110097144617672,
    },
}


@pytest.fixture(params=WIDTHS, ids=lambda width: f"bits{int(width)}")
def width(request: pytest.FixtureRequest) -> MixWidth:
    return request.param


def seed_sample(count: int, high: int = 2**62) -> np.ndarray:
    return np.random.default_rng(0).integers(0, high, count, dtype=np.int64)


class TestMixer:
    def test_output_is_deterministic(self, width: MixWidth) -> None:
        ids = np.arange(1024)
        assert np.array_equal(mix(SEED, ids, width), mix(SEED, ids, width))

    def test_output_is_stable_across_processes(self, width: MixWidth) -> None:
        probe = (
            "import numpy as np; from llmwatermark.greenlist import mix; "
            f"print(mix(np.int64({SEED}), np.int64(12345), {int(width)}).tolist())"
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-c", probe], capture_output=True, text=True, check=True
            ).stdout.strip()
            for _ in range(2)
        }
        assert len(outputs) == 1

    def test_matches_the_committed_golden_vectors(self, width: MixWidth) -> None:
        for (seed, token_id), expected in GOLDEN_MIX[width].items():
            actual = int(mix(np.int64(seed), np.int64(token_id), width)[0])
            assert actual == expected, f"mix({seed}, {token_id}) at {int(width)} bits"

    def test_output_is_non_negative(self, width: MixWidth) -> None:
        """The sign bit is masked off so that `%` behaves the same in numpy and torch."""
        values = mix(seed_sample(64)[:, None], np.arange(1024)[None, :], width)
        assert int(values.min()) >= 0

    def test_dtype_matches_the_requested_width(self, width: MixWidth) -> None:
        values = mix(np.int64(SEED), np.arange(16), width)
        expected = np.int32 if width is MixWidth.BITS32 else np.int64
        assert values.dtype == expected

    def test_flipping_one_token_id_bit_flips_half_the_output_bits(self, width: MixWidth) -> None:
        """Avalanche is what makes adjacent token IDs land in unrelated places."""
        ids = np.arange(20000)
        changed = mix(SEED, ids, width) ^ mix(SEED, ids ^ 1, width)
        bits = np.unpackbits(np.ascontiguousarray(changed).view(np.uint8)).reshape(len(ids), -1)
        mean_flipped = bits.sum(axis=1).mean()
        assert abs(mean_flipped - int(width) / 2) < int(width) * 0.05

    def test_flipping_one_seed_bit_flips_half_the_output_bits(self, width: MixWidth) -> None:
        ids = np.arange(20000)
        changed = mix(SEED, ids, width) ^ mix(SEED ^ 1, ids, width)
        bits = np.unpackbits(np.ascontiguousarray(changed).view(np.uint8)).reshape(len(ids), -1)
        assert abs(bits.sum(axis=1).mean() - int(width) / 2) < int(width) * 0.05

    def test_the_two_widths_are_different_watermarks(self) -> None:
        ids = np.arange(4096)
        thirty_two = mix(SEED, ids, MixWidth.BITS32).astype(np.int64)
        sixty_four = mix(SEED, ids, MixWidth.BITS64)
        assert not np.array_equal(thirty_two, sixty_four)

    def test_broadcasts_seeds_against_token_ids(self, width: MixWidth) -> None:
        seeds = seed_sample(8)
        ids = np.arange(64)
        matrix = mix(seeds[:, None], ids[None, :], width)
        assert matrix.shape == (8, 64)
        for row, seed in enumerate(seeds):
            assert np.array_equal(matrix[row], mix(seed, ids, width))

    def test_unknown_width_is_rejected(self) -> None:
        with pytest.raises(Exception, match="mix_width"):
            mix(SEED, np.arange(4), 16)  # type: ignore[arg-type]


class TestGreenFraction:
    @pytest.mark.parametrize("divisor", [2, 4, 10])
    def test_green_fraction_matches_one_over_the_divisor(
        self, divisor: int, width: MixWidth
    ) -> None:
        ids = token_id_range(VOCAB_SIZE, width)
        fractions = np.array(
            [green_mask(np.array([seed]), ids, divisor, width).mean() for seed in seed_sample(40)]
        )
        target = 1 / divisor
        tolerance = 6 * np.sqrt(target * (1 - target) / VOCAB_SIZE)
        assert abs(fractions.mean() - target) < tolerance
        assert np.abs(fractions - target).max() < tolerance

    def test_greenlist_is_not_a_periodic_stripe(self, width: MixWidth) -> None:
        """Without mixing, `% divisor` would green exactly every divisor-th token ID."""
        ids = token_id_range(VOCAB_SIZE, width)
        green = green_mask(np.array([SEED]), ids, 4, width)[0]
        for residue in range(4):
            share = green[residue::4].mean()
            assert 0.2 < share < 0.3

    def test_two_seeds_are_independent(self, width: MixWidth) -> None:
        """The z-test's null needs each position to be an independent Bernoulli(gamma)."""
        ids = token_id_range(VOCAB_SIZE, width)
        seeds = seed_sample(2)
        first, second = green_mask(seeds, ids, 4, width)
        both = float((first & second).mean())
        assert abs(both - 0.25**2) < 0.004

    def test_a_different_seed_gives_a_different_greenlist(self, width: MixWidth) -> None:
        ids = token_id_range(4096, width)
        first, second = green_mask(np.array([SEED, SEED + 1]), ids, 4, width)
        assert not np.array_equal(first, second)


class TestGeneratorDetectorAgreement:
    def test_is_green_matches_the_full_mask_exhaustively(self, width: MixWidth) -> None:
        """The single guarantee that detection reproduces generation."""
        vocab_size = 512
        ids = token_id_range(vocab_size, width)
        mask = green_mask(np.array([SEED]), ids, 4, width)[0]
        per_token = is_green(np.full(vocab_size, SEED, dtype=np.int64), ids, 4, width)
        assert np.array_equal(mask, per_token)

    def test_is_green_is_elementwise_over_positions(self, width: MixWidth) -> None:
        seeds = seed_sample(6)
        observed = np.array([3, 17, 250, 8, 1, 0])
        expected = [
            bool(green_mask(np.array([seed]), np.array([token]), 4, width)[0, 0])
            for seed, token in zip(seeds, observed, strict=True)
        ]
        assert is_green(seeds, observed, 4, width).tolist() == expected


class TestShapesAndDtypes:
    def test_green_mask_is_a_boolean_batch_by_vocab_matrix(self, width: MixWidth) -> None:
        mask = green_mask(seed_sample(5), token_id_range(64, width), 4, width)
        assert mask.dtype == np.bool_
        assert mask.shape == (5, 64)

    def test_empty_batch_is_allowed(self, width: MixWidth) -> None:
        mask = green_mask(np.zeros(0, dtype=np.int64), token_id_range(64, width), 4, width)
        assert mask.shape == (0, 64)

    @pytest.mark.parametrize("vocab_size", [2, 3, 1024])
    def test_tiny_vocabularies_work(self, vocab_size: int, width: MixWidth) -> None:
        mask = green_mask(np.array([SEED]), token_id_range(vocab_size, width), 2, width)
        assert mask.shape == (1, vocab_size)

    @pytest.mark.parametrize("seed", [0, 1, 2**62, (1 << 63) - 1])
    def test_extreme_seeds_are_handled(self, seed: int, width: MixWidth) -> None:
        mask = green_mask(np.array([seed], dtype=np.int64), token_id_range(1024, width), 4, width)
        assert 0.2 < mask.mean() < 0.3

    def test_divisor_below_two_is_rejected(self, width: MixWidth) -> None:
        with pytest.raises(SeedingError, match="divisor"):
            green_mask(np.array([SEED]), token_id_range(16, width), 1, width)

    def test_float_token_ids_are_rejected(self, width: MixWidth) -> None:
        with pytest.raises(SeedingError, match="integer"):
            is_green(np.array([SEED]), np.array([1.5]), 4, width)


class TestTokenIdRange:
    def test_dtype_follows_the_width(self) -> None:
        assert token_id_range(16, MixWidth.BITS32).dtype == np.int32
        assert token_id_range(16, MixWidth.BITS64).dtype == np.int64

    def test_covers_every_token_id(self, width: MixWidth) -> None:
        assert token_id_range(2048, width).tolist() == list(range(2048))

    def test_is_cached_so_the_hot_path_never_reallocates(self, width: MixWidth) -> None:
        assert token_id_range(2048, width) is token_id_range(2048, width)

    def test_cached_range_is_read_only(self, width: MixWidth) -> None:
        with pytest.raises(ValueError):
            token_id_range(2048, width)[0] = 5

    @pytest.mark.parametrize("bad_size", [0, 1, -4])
    def test_unusable_vocab_size_is_rejected(self, bad_size: int) -> None:
        with pytest.raises(Exception, match="vocab_size"):
            token_id_range(bad_size, MixWidth.BITS32)


@pytest.mark.requires_torch
class TestTorchAgreement:
    """numpy and torch must produce bit-identical greenlists, or nothing detects."""

    def test_mixer_is_bit_identical(self, width: MixWidth) -> None:
        import torch

        seeds = seed_sample(64)
        ids = np.arange(4096)
        from_numpy = mix(seeds[:, None], ids[None, :], width)
        from_torch = mix(torch.from_numpy(seeds)[:, None], torch.from_numpy(ids)[None, :], width)
        assert np.array_equal(from_numpy, from_torch.numpy())

    def test_negative_intermediates_agree(self, width: MixWidth) -> None:
        """Arithmetic vs logical shift, and C vs Python modulo, both bite here."""
        import torch

        seeds = np.array([(1 << 62) - 1, 1 << 61, 0, 7], dtype=np.int64)
        ids = np.arange(1, 5000)
        from_numpy = mix(seeds[:, None], ids[None, :], width)
        from_torch = mix(torch.from_numpy(seeds)[:, None], torch.from_numpy(ids)[None, :], width)
        assert np.array_equal(from_numpy, from_torch.numpy())

    def test_green_mask_is_bit_identical(self, width: MixWidth) -> None:
        import torch

        seeds = seed_sample(16)
        ids = np.arange(20000)
        from_numpy = green_mask(seeds, ids, 4, width)
        from_torch = green_mask(torch.from_numpy(seeds), torch.from_numpy(ids), 4, width)
        assert np.array_equal(from_numpy, from_torch.numpy())

    def test_token_id_range_follows_the_array_it_is_given(self, width: MixWidth) -> None:
        import torch

        like = torch.zeros(1, dtype=torch.int64)
        ids = token_id_range(64, width, like=like)
        assert isinstance(ids, torch.Tensor)
        assert ids.device == like.device
        assert ids.dtype == (torch.int32 if width is MixWidth.BITS32 else torch.int64)
