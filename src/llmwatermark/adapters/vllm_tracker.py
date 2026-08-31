"""Row bookkeeping for vLLM's persistent batch.

vLLM hands a logits processor the logits for the whole batch and nothing else. Which
request sits in which row changes every step: finished requests leave, waiting ones take
their slots, and preempted ones are resumed somewhere else entirely. A row index is a
seat, not a passenger.

This class mirrors that layout from vLLM's own add / remove / move events. It deliberately
imports nothing from vLLM, so the bookkeeping - the part whose failure mode is a silently
unreadable watermark - is testable on CPU without a GPU or a served model.

**Nothing derived is cached.** The tracker holds references to each request's live output
token list, the list vLLM itself appends to, and reads the tail of it on every call. So a
request that is evicted and resumed still seeds from whatever tokens it actually has, and
no stale seed or greenlist can survive a reschedule.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from llmwatermark.errors import SeedingError

__all__ = ["MOVE_SWAP", "MOVE_UNIDIRECTIONAL", "RequestTracker"]

MOVE_UNIDIRECTIONAL: Final[str] = "unidirectional"
MOVE_SWAP: Final[str] = "swap"


class _Row:
    """One occupied slot: a request's prompt, and a live reference to its output tokens."""

    __slots__ = ("output", "prompt")

    def __init__(self, prompt: Any, output: Any) -> None:
        self.prompt = prompt if prompt is not None else ()
        self.output = output

    def tail(self, window: int) -> list[int] | None:
        """The last ``window`` token IDs, or None when there is not that much history.

        Reads only the tail, so cost is O(window) rather than O(sequence length).
        """
        output = self.output
        produced = len(output)
        if produced >= window:
            return [int(value) for value in output[produced - window :]]
        needed = window - produced
        prompt = self.prompt
        if len(prompt) < needed:
            return None
        return [int(value) for value in prompt[len(prompt) - needed :]] + [
            int(value) for value in output
        ]


class RequestTracker:
    """Tracks which request occupies which row of vLLM's persistent batch."""

    def __init__(self) -> None:
        self._rows: list[_Row | None] = []
        self._size = 0

    @property
    def batch_size(self) -> int:
        return self._size

    def apply(
        self,
        *,
        batch_size: int,
        removed: Any = (),
        added: Any = (),
        moved: Any = (),
    ) -> None:
        """Apply one batch update.

        Operations run in the order vLLM documents: **removed, then added, then moved**.
        Added and moved requests may legitimately replace whatever occupied the index.

        :param added: ``(index, prompt_token_ids, output_token_ids)`` per request. The
            prompt may be None - vLLM only materializes prompt IDs when a penalty or
            another processor asks for them.
        """
        touched = [int(index) for index in removed]
        touched += [int(entry[0]) for entry in added]
        for source, target, _ in moved:
            touched += [int(source), int(target)]
        self._ensure_capacity(max([batch_size, *touched], default=0))

        for index in removed:
            self._rows[int(index)] = None

        for index, prompt, output in added:
            self._rows[int(index)] = _Row(prompt, output)

        for source, target, direction in moved:
            self._move(int(source), int(target), direction)

        self._size = int(batch_size)
        # Clear anything past the batch so a later growth cannot resurrect a stale row.
        for index in range(self._size, len(self._rows)):
            self._rows[index] = None

    def contexts(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        """The last ``window`` token IDs of every row, and which rows have that many.

        :returns: ``(context, valid)`` of shapes ``(batch_size, window)`` int64 and
            ``(batch_size,)`` bool. Invalid rows are zero-filled and must not be biased:
            they are empty slots, or requests without a full context window yet.
        """
        if window < 1:
            raise SeedingError(f"window must be a positive integer, got {window!r}.")
        context = np.zeros((self._size, window), dtype=np.int64)
        valid = np.zeros(self._size, dtype=np.bool_)
        for index in range(self._size):
            row = self._rows[index]
            if row is None:
                continue
            tail = row.tail(window)
            if tail is None:
                continue
            context[index] = tail
            valid[index] = True
        return context, valid

    def _move(self, source: int, target: int, direction: Any) -> None:
        name = getattr(direction, "name", direction)
        if isinstance(name, str):
            name = name.lower()
        if name == MOVE_SWAP:
            self._rows[source], self._rows[target] = self._rows[target], self._rows[source]
        elif name == MOVE_UNIDIRECTIONAL:
            self._rows[target] = self._rows[source]
            self._rows[source] = None
        else:
            raise SeedingError(
                f"unknown move direction {direction!r}; expected "
                f"{MOVE_UNIDIRECTIONAL!r} or {MOVE_SWAP!r}."
            )

    def _ensure_capacity(self, size: int) -> None:
        if size > len(self._rows):
            self._rows.extend([None] * (size - len(self._rows)))

    def __repr__(self) -> str:
        occupied = sum(1 for row in self._rows[: self._size] if row is not None)
        return f"{type(self).__name__}(batch_size={self._size}, occupied={occupied})"
