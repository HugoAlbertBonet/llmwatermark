"""The logits processor: where delta actually meets the model's logits.

The bugs this guards against are silent ones. A watermark that desynchronises under
batching, row reordering or preemption produces text that simply does not detect, with no
error anywhere. So most of these tests are about statelessness rather than arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from llmwatermark.config import HashScheme, MixWidth, WatermarkConfig
from llmwatermark.errors import ConfigError, SeedingError
from llmwatermark.greenlist import green_mask, token_id_range
from llmwatermark.processor import WatermarkProcessor
from llmwatermark.seeding import SeedTable

VOCAB_SIZE = 512
KEY = b"processor-test-key"
DELTA = 2.0


def make_config(**overrides: object) -> WatermarkConfig:
    fields: dict[str, object] = {
        "secret_key": KEY,
        "vocab_size": VOCAB_SIZE,
        "vocab_fingerprint": "0" * 64,
        "delta": DELTA,
    }
    fields.update(overrides)
    return WatermarkConfig(**fields)  # type: ignore[arg-type]


def expected_green(config: WatermarkConfig, context: np.ndarray) -> np.ndarray:
    """The greenlist, computed straight from the M2/M3 primitives."""
    seeds = SeedTable.for_config(config).seeds(context, config.scheme)
    ids = token_id_range(config.vocab_size, config.mix_width)
    return green_mask(seeds, ids, config.green_divisor, config.mix_width)


def zero_logits(batch: int, dtype: object = np.float32) -> np.ndarray:
    return np.zeros((batch, VOCAB_SIZE), dtype=dtype)  # type: ignore[arg-type]


@pytest.fixture
def config() -> WatermarkConfig:
    return make_config()


@pytest.fixture
def processor(config: WatermarkConfig) -> WatermarkProcessor:
    return WatermarkProcessor(config)


@pytest.fixture
def context() -> np.ndarray:
    return np.array([[7], [11], [300], [0]], dtype=np.int64)


class TestBias:
    def test_delta_lands_on_green_tokens_and_nowhere_else(
        self, processor: WatermarkProcessor, config: WatermarkConfig, context: np.ndarray
    ) -> None:
        logits = zero_logits(len(context))
        processor.apply(logits, context)
        green = expected_green(config, context)
        assert np.array_equal(logits[green], np.full(int(green.sum()), DELTA, dtype=np.float32))
        assert np.array_equal(logits[~green], np.zeros(int((~green).sum()), dtype=np.float32))

    def test_bias_is_added_not_assigned(
        self, processor: WatermarkProcessor, config: WatermarkConfig, context: np.ndarray
    ) -> None:
        base = np.linspace(-5, 5, VOCAB_SIZE, dtype=np.float32)
        logits = np.tile(base, (len(context), 1))
        processor.apply(logits, context)
        green = expected_green(config, context)
        assert np.allclose(logits - np.tile(base, (len(context), 1)), green * DELTA)

    def test_zero_delta_leaves_logits_untouched(self, context: np.ndarray) -> None:
        processor = WatermarkProcessor(make_config(delta=0.0))
        logits = np.linspace(-1, 1, len(context) * VOCAB_SIZE, dtype=np.float32).reshape(
            len(context), VOCAB_SIZE
        )
        before = logits.copy()
        processor.apply(logits, context)
        assert np.array_equal(logits, before)

    def test_masked_tokens_stay_masked(
        self, processor: WatermarkProcessor, context: np.ndarray
    ) -> None:
        """A token an earlier processor set to -inf must not be rescued by delta."""
        logits = zero_logits(len(context))
        logits[:, :100] = -np.inf
        processor.apply(logits, context)
        assert np.all(np.isneginf(logits[:, :100]))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
    def test_logits_dtype_is_preserved(
        self, processor: WatermarkProcessor, context: np.ndarray, dtype: object
    ) -> None:
        logits = zero_logits(len(context), dtype)
        processor.apply(logits, context)
        assert logits.dtype == dtype
        assert logits.max() == pytest.approx(DELTA)

    def test_apply_mutates_in_place_and_returns_the_same_object(
        self, processor: WatermarkProcessor, context: np.ndarray
    ) -> None:
        logits = zero_logits(len(context))
        assert processor.apply(logits, context) is logits

    @pytest.mark.parametrize("scheme", [HashScheme.LEFTHASH, HashScheme.MINHASH])
    @pytest.mark.parametrize("width", [MixWidth.BITS32, MixWidth.BITS64])
    def test_every_scheme_and_width_combination_biases_its_own_greenlist(
        self, scheme: HashScheme, width: MixWidth
    ) -> None:
        config = make_config(scheme=scheme, mix_width=width)
        context = np.array([[3, 17, 250, 8][: config.h], [1, 2, 3, 4][: config.h]], dtype=np.int64)
        logits = zero_logits(2)
        WatermarkProcessor(config).apply(logits, context)
        assert np.array_equal(logits > 0, expected_green(config, context))


class TestStatelessness:
    """The watermark is a pure function of (key, last h tokens). Nothing else may leak in."""

    def test_a_batch_equals_the_same_rows_run_one_at_a_time(
        self, processor: WatermarkProcessor, context: np.ndarray
    ) -> None:
        batched = zero_logits(len(context))
        processor.apply(batched, context)
        for row in range(len(context)):
            single = zero_logits(1)
            processor.apply(single, context[row : row + 1])
            assert np.array_equal(batched[row], single[0])

    def test_reordering_rows_only_reorders_the_result(
        self, processor: WatermarkProcessor, context: np.ndarray
    ) -> None:
        """Beam search reorders rows between steps."""
        order = np.array([2, 0, 3, 1])
        straight = zero_logits(len(context))
        processor.apply(straight, context)
        shuffled = zero_logits(len(context))
        processor.apply(shuffled, context[order])
        assert np.array_equal(shuffled, straight[order])

    def test_applying_twice_to_fresh_logits_gives_the_same_answer(
        self, processor: WatermarkProcessor, context: np.ndarray
    ) -> None:
        first = zero_logits(len(context))
        processor.apply(first, context)
        second = zero_logits(len(context))
        processor.apply(second, context)
        assert np.array_equal(first, second)

    def test_a_row_dropped_and_re_added_keeps_its_own_greenlist(
        self, processor: WatermarkProcessor, context: np.ndarray
    ) -> None:
        """vLLM preempts and reschedules sequences; the row index is meaningless."""
        full = zero_logits(len(context))
        processor.apply(full, context)

        shrunk = zero_logits(2)
        processor.apply(shrunk, context[:2])
        restored = zero_logits(len(context))
        processor.apply(restored, context)
        assert np.array_equal(restored, full)

    def test_batch_size_changing_between_calls_changes_nothing(
        self, processor: WatermarkProcessor, context: np.ndarray
    ) -> None:
        for size in (1, 4, 2, 3, 4):
            logits = zero_logits(size)
            processor.apply(logits, context[:size])
            reference = zero_logits(size)
            processor.apply(reference, context[:size])
            assert np.array_equal(logits, reference)


class TestShortContext:
    def test_rows_without_a_full_context_window_are_left_alone(
        self, processor: WatermarkProcessor, context: np.ndarray
    ) -> None:
        logits = zero_logits(len(context))
        valid = np.array([True, False, True, False])
        processor.apply(logits, context, valid)
        assert np.array_equal(logits[1], np.zeros(VOCAB_SIZE, dtype=np.float32))
        assert np.array_equal(logits[3], np.zeros(VOCAB_SIZE, dtype=np.float32))
        assert logits[0].max() == pytest.approx(DELTA)

    def test_valid_rows_match_an_unmasked_run(
        self, processor: WatermarkProcessor, context: np.ndarray
    ) -> None:
        masked = zero_logits(len(context))
        processor.apply(masked, context, np.array([True, False, True, False]))
        unmasked = zero_logits(len(context))
        processor.apply(unmasked, context)
        assert np.array_equal(masked[0], unmasked[0])
        assert np.array_equal(masked[2], unmasked[2])

    def test_histories_shorter_than_the_context_window_are_skipped(
        self, processor: WatermarkProcessor
    ) -> None:
        logits = zero_logits(2)
        processor.apply_to_histories(logits, [[], [4, 5, 6]])
        assert np.array_equal(logits[0], np.zeros(VOCAB_SIZE, dtype=np.float32))
        assert logits[1].max() == pytest.approx(DELTA)

    def test_histories_path_matches_the_explicit_context_path(
        self, processor: WatermarkProcessor
    ) -> None:
        histories = [[1, 2, 3, 7], [9, 11]]
        from_histories = zero_logits(2)
        processor.apply_to_histories(from_histories, histories)
        from_context = zero_logits(2)
        processor.apply(from_context, np.array([[7], [11]], dtype=np.int64))
        assert np.array_equal(from_histories, from_context)


class TestValidation:
    def test_logits_vocabulary_must_match_the_config(
        self, processor: WatermarkProcessor, context: np.ndarray
    ) -> None:
        with pytest.raises(ConfigError) as excinfo:
            processor.apply(np.zeros((4, VOCAB_SIZE + 8), dtype=np.float32), context)
        message = str(excinfo.value)
        assert str(VOCAB_SIZE) in message
        assert str(VOCAB_SIZE + 8) in message

    def test_context_width_must_match_the_scheme(self, processor: WatermarkProcessor) -> None:
        with pytest.raises(SeedingError, match="context"):
            processor.apply(zero_logits(2), np.zeros((2, 3), dtype=np.int64))

    def test_batch_sizes_must_agree(
        self, processor: WatermarkProcessor, context: np.ndarray
    ) -> None:
        with pytest.raises(SeedingError, match="batch"):
            processor.apply(zero_logits(2), context)

    def test_logits_must_be_two_dimensional(
        self, processor: WatermarkProcessor, context: np.ndarray
    ) -> None:
        with pytest.raises(SeedingError, match="shape"):
            processor.apply(np.zeros(VOCAB_SIZE, dtype=np.float32), context)

    def test_valid_mask_length_must_match_the_batch(
        self, processor: WatermarkProcessor, context: np.ndarray
    ) -> None:
        with pytest.raises(SeedingError, match="valid"):
            processor.apply(zero_logits(4), context, np.array([True, False]))

    def test_out_of_range_context_token_is_reported_on_the_host_path(
        self, processor: WatermarkProcessor
    ) -> None:
        with pytest.raises(SeedingError, match=str(VOCAB_SIZE)):
            processor.apply(zero_logits(1), np.array([[VOCAB_SIZE + 1]], dtype=np.int64))


class TestCompileMode:
    def test_numpy_input_is_never_compiled(self, processor: WatermarkProcessor) -> None:
        processor.apply(zero_logits(1), np.array([[7]], dtype=np.int64))
        assert processor.is_compiled is False

    def test_compile_mode_is_visible(self, config: WatermarkConfig) -> None:
        """Compilation is on by default; users must be able to see and change that."""
        assert WatermarkProcessor(config).compile_mode == "auto"
        assert WatermarkProcessor(config, compile=False).compile_mode == "never"
        assert WatermarkProcessor(config, compile=True).compile_mode == "always"

    def test_repr_states_the_compile_mode(self, processor: WatermarkProcessor) -> None:
        assert "compile=" in repr(processor)

    def test_unknown_compile_mode_is_rejected(self, config: WatermarkConfig) -> None:
        with pytest.raises(ConfigError, match="compile"):
            WatermarkProcessor(config, compile="sometimes")  # type: ignore[arg-type]


@pytest.mark.requires_torch
class TestTorchAgreement:
    def test_torch_matches_numpy(self, processor: WatermarkProcessor) -> None:
        import torch

        context = np.array([[7], [11], [300]], dtype=np.int64)
        reference = zero_logits(3)
        processor.apply(reference, context)

        tensor = torch.zeros(3, VOCAB_SIZE, dtype=torch.float32)
        processor.apply(tensor, torch.from_numpy(context))
        assert np.array_equal(tensor.numpy(), reference)

    def test_torch_apply_is_in_place(self, processor: WatermarkProcessor) -> None:
        import torch

        tensor = torch.zeros(2, VOCAB_SIZE)
        assert processor.apply(tensor, torch.tensor([[7], [11]])) is tensor

    def test_half_precision_is_preserved(self, processor: WatermarkProcessor) -> None:
        import torch

        tensor = torch.zeros(2, VOCAB_SIZE, dtype=torch.float16)
        processor.apply(tensor, torch.tensor([[7], [11]]))
        assert tensor.dtype == torch.float16
        assert float(tensor.max()) == pytest.approx(DELTA)

    def test_compiled_output_is_identical_to_eager(self, config: WatermarkConfig) -> None:
        import torch

        context = torch.tensor([[7], [11], [300], [0]])
        eager = torch.zeros(4, VOCAB_SIZE)
        WatermarkProcessor(config, compile=False).apply(eager, context)

        compiled_processor = WatermarkProcessor(config, compile=True)
        compiled = torch.zeros(4, VOCAB_SIZE)
        compiled_processor.apply(compiled, context)
        assert compiled_processor.is_compiled is True
        assert torch.equal(compiled, eager)

    def test_changing_batch_size_does_not_recompile(self, config: WatermarkConfig) -> None:
        """A serving workload changes batch size constantly; recompiling each time is fatal."""
        import torch
        import torch._dynamo

        processor = WatermarkProcessor(config, compile=True)
        for size in (2, 3, 5):
            processor.apply(torch.zeros(size, VOCAB_SIZE), torch.full((size, 1), 7))
        before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        for size in (4, 6, 7, 8):
            processor.apply(torch.zeros(size, VOCAB_SIZE), torch.full((size, 1), 7))
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == before


@pytest.mark.requires_cuda
class TestDeviceHotPath:
    """The performance contract, asserted rather than assumed."""

    @staticmethod
    def _cuda_setup(config: WatermarkConfig) -> tuple[object, object, object]:
        import torch

        processor = WatermarkProcessor(config)
        logits = torch.zeros(32, VOCAB_SIZE, device="cuda")
        context = torch.full((32, 1), 7, device="cuda")
        processor.apply(logits, context)  # warm up: build tables and compile
        return processor, logits, context

    def test_the_device_path_never_synchronises_with_the_host(
        self, config: WatermarkConfig
    ) -> None:
        """A sync per decode step would cost more than the watermark itself."""
        import torch

        processor, logits, context = self._cuda_setup(config)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            processor.apply(logits, context)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_no_batch_by_vocabulary_float_temporary_is_allocated(
        self, config: WatermarkConfig
    ) -> None:
        import torch

        processor, logits, context = self._cuda_setup(config)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        processor.apply(logits, context)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        full_float_temporary = 32 * VOCAB_SIZE * 4
        assert peak < full_float_temporary, (
            f"allocated {peak} bytes; a batch x vocab float temporary would be "
            f"{full_float_temporary}"
        )

    def test_compilation_is_on_by_default_on_cuda(self, config: WatermarkConfig) -> None:
        processor, _, _ = self._cuda_setup(config)
        assert processor.compile_mode == "auto"
        assert processor.is_compiled is True

    def test_compiled_cuda_output_matches_the_numpy_reference(
        self, config: WatermarkConfig
    ) -> None:
        import torch

        context = np.array([[7], [11], [300], [0]], dtype=np.int64)
        reference = zero_logits(len(context))
        WatermarkProcessor(config, compile=False).apply(reference, context)

        logits = torch.zeros(len(context), VOCAB_SIZE, device="cuda")
        WatermarkProcessor(config).apply(logits, torch.from_numpy(context).cuda())
        assert np.array_equal(logits.cpu().numpy(), reference)
