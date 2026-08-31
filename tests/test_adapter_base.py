"""The pieces shared by every backend adapter.

The staging class exists because getting this wrong is expensive and silent: a pageable
host-to-device copy blocks the stream, and inside a live sampler that stall cost about seven
percent of vLLM's throughput before it was found. Writing it once and testing it here is what
stops each new adapter rediscovering it.
"""

from __future__ import annotations

import numpy as np
import pytest

from llmwatermark.adapters.base import HostContextStaging, check_vocabulary, require_backend
from llmwatermark.config import WatermarkConfig
from llmwatermark.errors import ConfigError

VOCAB_SIZE = 512


@pytest.fixture
def config() -> WatermarkConfig:
    return WatermarkConfig(
        secret_key=b"adapter-base-key", vocab_size=VOCAB_SIZE, vocab_fingerprint="0" * 64
    )


class TestRequireBackend:
    def test_names_the_package_and_the_extra(self) -> None:
        with pytest.raises(ImportError) as excinfo:
            require_backend("llama-cpp-python", "llama-cpp")
        message = str(excinfo.value)
        assert "llama-cpp-python" in message
        assert 'pip install "llmwatermark[llama-cpp]"' in message

    def test_keeps_the_original_import_error_as_the_cause(self) -> None:
        cause = ModuleNotFoundError("No module named 'whatever'")
        with pytest.raises(ImportError) as excinfo:
            require_backend("whatever", "whatever", cause)
        assert excinfo.value.__cause__ is cause


class TestCheckVocabulary:
    def test_a_matching_size_passes(self, config: WatermarkConfig) -> None:
        check_vocabulary(VOCAB_SIZE, config, "some backend")

    def test_an_unknown_size_passes(self, config: WatermarkConfig) -> None:
        """A backend that cannot report its vocabulary is not an error by itself."""
        check_vocabulary(None, config, "some backend")

    def test_a_mismatch_names_both_numbers_and_the_backend(self, config: WatermarkConfig) -> None:
        with pytest.raises(ConfigError) as excinfo:
            check_vocabulary(VOCAB_SIZE + 64, config, "some backend")
        message = str(excinfo.value)
        assert str(VOCAB_SIZE) in message
        assert str(VOCAB_SIZE + 64) in message
        assert "some backend" in message


@pytest.mark.requires_torch
class TestHostContextStaging:
    def test_the_staged_tensors_carry_the_right_values(self) -> None:
        import torch

        staging = HostContextStaging()
        context = np.array([[7], [11], [300]], dtype=np.int64)
        valid = np.array([True, False, True])
        device_context, device_valid = staging.stage(context, valid, torch.device("cpu"))
        assert device_context.tolist() == context.tolist()
        assert device_valid.tolist() == valid.tolist()

    def test_shapes_and_dtypes_are_preserved(self) -> None:
        import torch

        staging = HostContextStaging()
        context = np.zeros((5, 4), dtype=np.int64)
        valid = np.ones(5, dtype=np.bool_)
        staged_context, staged_valid = staging.stage(context, valid, torch.device("cpu"))
        assert staged_context.shape == (5, 4)
        assert staged_valid.shape == (5,)
        assert staged_context.dtype == torch.int64
        assert staged_valid.dtype == torch.bool

    def test_buffers_are_reused_across_calls(self) -> None:
        """Allocating per step would put an allocator call on the hot path."""
        import torch

        staging = HostContextStaging()
        first, _ = staging.stage(
            np.zeros((8, 1), dtype=np.int64), np.ones(8, dtype=np.bool_), torch.device("cpu")
        )
        second, _ = staging.stage(
            np.ones((8, 1), dtype=np.int64), np.ones(8, dtype=np.bool_), torch.device("cpu")
        )
        assert first.data_ptr() == second.data_ptr()

    def test_a_smaller_batch_reuses_the_same_buffer(self) -> None:
        import torch

        staging = HostContextStaging()
        staging.stage(
            np.zeros((30, 1), dtype=np.int64), np.ones(30, dtype=np.bool_), torch.device("cpu")
        )
        before = repr(staging)
        staged, _ = staging.stage(
            np.zeros((4, 1), dtype=np.int64), np.ones(4, dtype=np.bool_), torch.device("cpu")
        )
        assert staged.shape[0] == 4
        assert repr(staging) == before, "a shrinking batch must not reallocate"

    def test_growing_past_capacity_reallocates_once(self) -> None:
        import torch

        staging = HostContextStaging()
        staging.stage(
            np.zeros((8, 1), dtype=np.int64), np.ones(8, dtype=np.bool_), torch.device("cpu")
        )
        big = 200
        staged, valid = staging.stage(
            np.arange(big, dtype=np.int64).reshape(big, 1),
            np.ones(big, dtype=np.bool_),
            torch.device("cpu"),
        )
        assert staged.tolist() == [[value] for value in range(big)]
        assert valid.shape == (big,)

    def test_a_changed_context_width_reallocates(self) -> None:
        """LeftHash and MinHash have different window widths; a reused buffer would break."""
        import torch

        staging = HostContextStaging()
        staging.stage(
            np.zeros((4, 1), dtype=np.int64), np.ones(4, dtype=np.bool_), torch.device("cpu")
        )
        staged, _ = staging.stage(
            np.arange(16, dtype=np.int64).reshape(4, 4),
            np.ones(4, dtype=np.bool_),
            torch.device("cpu"),
        )
        assert staged.shape == (4, 4)
        assert staged.tolist() == np.arange(16).reshape(4, 4).tolist()

    def test_stale_rows_never_leak_into_a_later_batch(self) -> None:
        """Buffers are reused, so a shorter batch must not expose the previous one's rows."""
        import torch

        staging = HostContextStaging()
        staging.stage(
            np.full((16, 1), 99, dtype=np.int64), np.ones(16, dtype=np.bool_), torch.device("cpu")
        )
        staged, valid = staging.stage(
            np.array([[1], [2]], dtype=np.int64), np.array([True, False]), torch.device("cpu")
        )
        assert staged.tolist() == [[1], [2]]
        assert valid.tolist() == [True, False]
