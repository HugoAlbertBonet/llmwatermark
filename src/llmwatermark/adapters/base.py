"""Pieces every backend adapter needs, so none of them reinvents one.

Adapters differ in exactly one thing that matters: where a request's recent token IDs live.
Backends that hold a rectangular ``(batch, time)`` tensor on the device - transformers - slice
it and hand the slice straight to the processor. Backends that keep per-request Python lists
on the host - vLLM, and most of the rest - have to copy that context to the device every
decode step, and *how* they copy it is not a detail: a pageable copy blocks, and inside a
pipelined sampler that stall cost roughly seven percent of throughput before it was found.

:class:`HostContextStaging` is that copy done once, correctly, for every adapter that needs
it. The rest here is the shared error handling that otherwise gets reworded slightly in each
adapter until the messages disagree.
"""

from __future__ import annotations

from typing import Any, NoReturn

from llmwatermark.config import WatermarkConfig
from llmwatermark.errors import ConfigError

__all__ = ["HostContextStaging", "check_vocabulary", "require_backend"]


def require_backend(package: str, extra: str, error: BaseException | None = None) -> NoReturn:
    """Raise the install message for a missing optional backend.

    Adapters import their backend at module import, so importing the adapter is how a user
    opts in; this is what they see when they have not installed it.
    """
    raise ImportError(
        f"the {extra} adapter needs the {package} package, which is an optional extra. "
        f'Install it with:\n\n    pip install "llmwatermark[{extra}]"\n'
    ) from error


def check_vocabulary(actual: int | None, config: WatermarkConfig, source: str) -> None:
    """Fail loudly when a backend generates over a different number of token IDs.

    The greenlist partitions token IDs, so a mismatch does not degrade detection - it
    produces a different partition entirely and detection silently returns nothing.
    """
    if actual is None or int(actual) == config.vocab_size:
        return
    raise ConfigError(
        f"{source} generates over {int(actual)} token IDs but the watermark config declares "
        f"vocab_size={config.vocab_size}. The greenlist partitions token IDs, so the two must "
        "match exactly. Build the config from the model rather than the tokenizer: a padded "
        "embedding matrix makes model vocab_size larger than len(tokenizer)."
    )


class HostContextStaging:
    """Move host-assembled context to the device without stalling the pipeline.

    Handing a pageable numpy array to ``torch.as_tensor(..., device=...)`` makes the copy
    *blocking*: it waits for the stream. The copy itself is tens of microseconds, but inside
    a live sampler it drains whatever work is queued, and that bubble is measured in
    milliseconds per decode step.

    Staging through pinned buffers keeps the copy asynchronous, and reusing the buffers means
    nothing is allocated on the hot path either. Buffers grow geometrically and are never
    shrunk, so a batch that fluctuates does not reallocate.
    """

    __slots__ = ("_buffers",)

    def __init__(self) -> None:
        self._buffers: tuple[Any, Any, Any, Any] | None = None

    def stage(self, context: Any, valid: Any, device: Any) -> tuple[Any, Any]:
        """Return ``(context, valid)`` as device tensors of the same shape."""

        rows, window = context.shape
        buffers = self._buffers
        if buffers is None or buffers[0].shape[0] < rows or buffers[0].shape[1] != window:
            buffers = self._allocate(rows, window, device)
            self._buffers = buffers

        host_context, host_valid, device_context, device_valid = buffers
        host_context.numpy()[:rows] = context
        host_valid.numpy()[:rows] = valid
        device_context[:rows].copy_(host_context[:rows], non_blocking=True)
        device_valid[:rows].copy_(host_valid[:rows], non_blocking=True)
        return device_context[:rows], device_valid[:rows]

    def _allocate(self, rows: int, window: int, device: Any) -> tuple[Any, Any, Any, Any]:
        import torch

        previous = self._buffers[0].shape[0] if self._buffers else 0
        capacity = max(rows, 2 * previous, 32)
        try:
            host_context = torch.empty((capacity, window), dtype=torch.int64, pin_memory=True)
            host_valid = torch.empty(capacity, dtype=torch.bool, pin_memory=True)
        except RuntimeError:
            # Pinned memory needs a CUDA context; without one the copy is synchronous
            # anyway, so a plain buffer costs nothing.
            host_context = torch.empty((capacity, window), dtype=torch.int64)
            host_valid = torch.empty(capacity, dtype=torch.bool)
        return (
            host_context,
            host_valid,
            torch.empty((capacity, window), dtype=torch.int64, device=device),
            torch.empty(capacity, dtype=torch.bool, device=device),
        )

    def __repr__(self) -> str:
        capacity = self._buffers[0].shape if self._buffers else "unallocated"
        return f"{type(self).__name__}({capacity})"
