"""The numpy fast path, checked against the shared implementation it replaces.

This module exists to be fast, and the only thing that makes that safe is that it is
provably the same function. A greenlist that differs from :mod:`llmwatermark.greenlist` by
one token does not fail loudly - it produces text that the detector scores as unmarked, on
a backend the author may never run. So the central test here is not "does it work" but
"does it agree", over whole vocabularies rather than samples.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from llmwatermark.config import MixWidth, WatermarkConfig
from llmwatermark.errors import SeedingError
from llmwatermark.fastpath import Scratch, green_mask, green_mask_into
from llmwatermark.greenlist import is_green, token_id_range
from llmwatermark.processor import WatermarkProcessor, _biased

VOCAB_SIZE = 32768
# 4 is the default (gamma = 0.25) and takes the bitwise-AND path; 3 and 5 are odd, so
# Lemire's test applies with no rotation; 10 is even but not a power of two, which is the
# only case that rotates. 2 and 16 keep the AND path honest at its edges.
DIVISORS = (2, 3, 4, 5, 10, 16)


def reference(seeds: np.ndarray, ids: np.ndarray, divisor: int, width: MixWidth) -> np.ndarray:
    return is_green(seeds[:, None], ids[None, :], divisor, width)


class TestAgreesWithTheSharedPath:
    @pytest.mark.parametrize("width", list(MixWidth))
    @pytest.mark.parametrize("divisor", DIVISORS)
    def test_over_the_whole_vocabulary(self, width: MixWidth, divisor: int) -> None:
        ids = token_id_range(VOCAB_SIZE, width)
        seeds = np.random.default_rng(divisor).integers(0, 1 << 63, 4, dtype=np.int64)
        expected = reference(seeds, ids, divisor, width)
        assert np.array_equal(green_mask(seeds, ids, divisor, width), expected)

    @pytest.mark.parametrize("width", list(MixWidth))
    def test_at_a_realistic_vocabulary_size(self, width: MixWidth) -> None:
        """151936 is Qwen2.5's, where this path was measured; sizes are not interchangeable."""
        ids = token_id_range(151936, width)
        seeds = np.array([0, 1, 1 << 62, (1 << 63) - 1], dtype=np.int64)
        assert np.array_equal(green_mask(seeds, ids, 4, width), reference(seeds, ids, 4, width))

    def test_the_green_fraction_is_one_over_the_divisor(self) -> None:
        """Agreement with a wrong reference would be invisible; this is independent."""
        ids = token_id_range(VOCAB_SIZE)
        seeds = np.random.default_rng(0).integers(0, 1 << 63, 32, dtype=np.int64)
        for divisor in DIVISORS:
            share = green_mask(seeds, ids, divisor, MixWidth.BITS32).mean()
            assert share == pytest.approx(1 / divisor, abs=0.01)

    def test_a_single_row_matches_a_batched_row(self) -> None:
        ids = token_id_range(VOCAB_SIZE)
        seeds = np.array([11, 22, 33], dtype=np.int64)
        batched = green_mask(seeds, ids, 4, MixWidth.BITS32)
        for index, seed in enumerate(seeds):
            alone = green_mask(seed[None], ids, 4, MixWidth.BITS32)
            assert np.array_equal(alone[0], batched[index])


class TestLemireDivisibility:
    """The one step that is an identity rather than a transcription.

    ``rotr(n * qinv, k) <= (2**32 - 1) // d`` replaces a division. It was checked against
    ``%`` over all 2**32 unsigned values before being adopted; that sweep is too slow for a
    test suite, so this samples widely instead.
    """

    @pytest.mark.parametrize("divisor", [3, 5, 6, 7, 10, 100, 1000])
    def test_matches_the_modulus_on_a_wide_sample(self, divisor: int) -> None:
        values = np.random.default_rng(divisor).integers(0, 1 << 31, 2_000_000, dtype=np.int64)
        seeds = np.zeros(1, dtype=np.int64)
        # Feed the values in as token IDs so they travel the real predicate path.
        mixed = is_green(seeds[:, None], values.astype(np.int32)[None, :], divisor)
        fast = green_mask(seeds, values.astype(np.int32), divisor, MixWidth.BITS32)
        assert np.array_equal(mixed, fast)

    def test_every_residue_class_is_represented(
        self,
    ) -> None:
        """A test that only ever saw non-multiples would pass on a broken predicate."""
        ids = token_id_range(VOCAB_SIZE)
        mask = green_mask(np.array([7], dtype=np.int64), ids, 10, MixWidth.BITS32)
        assert mask.any() and not mask.all()


class TestScratch:
    def test_buffers_are_reused_across_calls(self) -> None:
        scratch = Scratch()
        ids = token_id_range(VOCAB_SIZE)
        seeds = np.array([1], dtype=np.int64)
        first = green_mask_into(seeds, ids, 4, MixWidth.BITS32, scratch)
        pointer = first.__array_interface__["data"][0]
        second = green_mask_into(seeds, ids, 4, MixWidth.BITS32, scratch)
        assert second.__array_interface__["data"][0] == pointer

    def test_a_smaller_batch_reuses_the_same_buffer(self) -> None:
        scratch = Scratch()
        ids = token_id_range(VOCAB_SIZE)
        green_mask_into(np.arange(8, dtype=np.int64), ids, 4, MixWidth.BITS32, scratch)
        before = repr(scratch)
        result = green_mask_into(np.arange(3, dtype=np.int64), ids, 4, MixWidth.BITS32, scratch)
        assert result.shape[0] == 3
        assert repr(scratch) == before, "a shrinking batch must not reallocate"

    def test_a_larger_batch_grows_and_stays_correct(self) -> None:
        scratch = Scratch()
        ids = token_id_range(VOCAB_SIZE)
        green_mask_into(np.arange(2, dtype=np.int64), ids, 4, MixWidth.BITS32, scratch)
        seeds = np.arange(40, dtype=np.int64)
        grown = green_mask_into(seeds, ids, 4, MixWidth.BITS32, scratch)
        assert np.array_equal(grown, reference(seeds, ids, 4, MixWidth.BITS32))

    def test_stale_rows_never_leak_into_a_later_batch(self) -> None:
        scratch = Scratch()
        ids = token_id_range(VOCAB_SIZE)
        green_mask_into(np.arange(16, dtype=np.int64), ids, 4, MixWidth.BITS32, scratch)
        seeds = np.array([5, 6], dtype=np.int64)
        result = green_mask_into(seeds, ids, 4, MixWidth.BITS32, scratch)
        assert np.array_equal(result, reference(seeds, ids, 4, MixWidth.BITS32))

    def test_a_changed_vocabulary_size_reallocates(self) -> None:
        scratch = Scratch()
        seeds = np.array([3], dtype=np.int64)
        green_mask_into(seeds, token_id_range(1024), 4, MixWidth.BITS32, scratch)
        wider = token_id_range(VOCAB_SIZE)
        assert np.array_equal(
            green_mask_into(seeds, wider, 4, MixWidth.BITS32, scratch),
            reference(seeds, wider, 4, MixWidth.BITS32),
        )

    def test_threads_do_not_share_buffers(self) -> None:
        """A processor is documented as shareable between threads, so this must hold."""
        scratch = Scratch()
        ids = token_id_range(VOCAB_SIZE)
        failures: list[str] = []

        def check(seed: int) -> None:
            seeds = np.array([seed], dtype=np.int64)
            expected = reference(seeds, ids, 4, MixWidth.BITS32)
            for _ in range(20):
                got = green_mask_into(seeds, ids, 4, MixWidth.BITS32, scratch)
                if not np.array_equal(got, expected):
                    failures.append(f"seed {seed} produced a mask from another thread")
                    return

        threads = [threading.Thread(target=check, args=(seed,)) for seed in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not failures

    def test_green_mask_returns_an_array_the_caller_owns(self) -> None:
        """green_mask_into hands back a buffer; green_mask must not."""
        ids = token_id_range(1024)
        first = green_mask(np.array([1], dtype=np.int64), ids, 4, MixWidth.BITS32)
        held = first.copy()
        green_mask(np.array([2], dtype=np.int64), ids, 4, MixWidth.BITS32)
        assert np.array_equal(first, held)


class TestRejectsBadInput:
    @pytest.mark.parametrize("divisor", [1, 0, -4])
    def test_a_divisor_below_two_is_refused(self, divisor: int) -> None:
        with pytest.raises(SeedingError):
            green_mask(np.array([1], dtype=np.int64), token_id_range(64), divisor, MixWidth.BITS32)


class TestProcessorUsesIt:
    """The processor must route numpy through the fast path and torch through the shared one."""

    @pytest.fixture
    def config(self) -> WatermarkConfig:
        return WatermarkConfig(
            secret_key=b"fastpath-key-0123456789012345678",
            vocab_size=VOCAB_SIZE,
            vocab_fingerprint="0" * 64,
        )

    @pytest.mark.parametrize("batch", [1, 5])
    def test_apply_matches_the_shared_kernel(self, config: WatermarkConfig, batch: int) -> None:
        processor = WatermarkProcessor(config)
        context = np.arange(batch, dtype=np.int64).reshape(batch, 1) + 17
        logits = np.random.default_rng(1).standard_normal((batch, VOCAB_SIZE)).astype(np.float32)
        expected = _biased(
            logits.copy(),
            context,
            None,
            processor._table.on(logits),
            token_id_range(VOCAB_SIZE, config.mix_width),
            config.scheme,
            config.green_divisor,
            config.delta,
            config.mix_width,
        )
        assert np.array_equal(processor.apply(logits.copy(), context), expected)

    def test_rows_marked_invalid_are_left_alone(self, config: WatermarkConfig) -> None:
        processor = WatermarkProcessor(config)
        context = np.array([[3], [4]], dtype=np.int64)
        logits = np.zeros((2, VOCAB_SIZE), dtype=np.float32)
        processor.apply(logits, context, np.array([True, False]))
        assert logits[1].any() is np.False_ or not logits[1].any()
        assert logits[0].any()

    def test_the_logits_dtype_is_preserved(self, config: WatermarkConfig) -> None:
        """A float64 promotion would silently double the memory the sampler then reads."""
        processor = WatermarkProcessor(config)
        for dtype in (np.float32, np.float64):
            logits = np.zeros((1, VOCAB_SIZE), dtype=dtype)
            assert processor.apply(logits, np.array([[9]], dtype=np.int64)).dtype == dtype
