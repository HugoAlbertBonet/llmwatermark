"""The greenlist and its bias, written for numpy's execution model rather than torch's.

:mod:`llmwatermark.greenlist` is written once for both libraries, and on CUDA
``torch.compile`` fuses the whole expression into a single kernel that keeps every
intermediate in registers. numpy has no such compiler: each operator is a separate loop
that writes a full ``batch x vocab_size`` array to memory, and at a 152k vocabulary that
is roughly sixteen passes over 600 KB per decode step. Backends whose sampler is host-side
- llama.cpp, and CPU transformers - pay that in full, which measured as **+52% throughput
on llama.cpp with GPU offload**, where the model step is short enough for the watermark to
dominate.

This module is the same function with the passes counted. It is not a different watermark:
every mask it produces is bit-identical to :func:`llmwatermark.greenlist.is_green`, and
:mod:`tests.test_fastpath` asserts that over the whole vocabulary for both mixer widths.
Three things account for the difference, and each was measured rather than assumed:

**Unsigned arithmetic removes the sign-extension mask.** The shared path emulates a logical
shift by masking off the bits ``>>`` sign-extends, because torch has no unsigned 32-bit
integer. Viewing the same bytes as ``uint32`` makes ``>>`` logical already, so three
mask-and-xor stages become three shift-and-xor ones. The bytes are untouched by the view.

**Reused buffers remove the page faults.** Every numpy temporary here is 600 KB, too large
for numpy's small-array cache, so a fresh one faults in its pages on first touch. Writing
through ``out=`` into buffers held across calls took the mixer from 326 us to 134 us -
allocation, not arithmetic, was the larger half. The buffers are thread-local, so a
processor shared between threads still behaves as its docstring promises.

**Divisibility without division.** ``mix(...) % divisor == 0`` is the one operation numpy
cannot vectorize: integer division has no SIMD form, and it alone cost 219 us of a 526 us
mask. Two exact replacements avoid it. When the divisor is a power of two the modulus is a
bitwise AND (25 us). Otherwise Lemire's test applies: for ``d = 2**k * q`` with ``q`` odd,
``n`` is divisible by ``d`` exactly when ``rotr32(n * qinv, k) <= (2**32 - 1) // d``, where
``qinv`` is the inverse of ``q`` modulo ``2**32``. That is 35 us for an odd divisor. It is
an identity rather than an approximation, and it was checked here against ``%`` over all
2**32 unsigned values for several divisors before being used - zero mismatches.

The result is a mask in 150 us against 526 us, and a bias in 45 us against 180 us. The
bias figure is not a trick: the shared path's ``green * delta`` promotes a bool array to
**float64** against a Python float, then casts 600 KB back down to float32.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np

from llmwatermark.config import MixWidth
from llmwatermark.greenlist import _PARAMS, _validate_divisor

__all__ = ["Scratch", "green_mask", "green_mask_into"]


class _Width(NamedTuple):
    """The mixer's constants for one width, as unsigned values of that width."""

    dtype: Any
    bits: int
    golden: Any
    first_constant: Any
    second_constant: Any
    shifts: tuple[int, int, int]
    sign_mask: Any
    span_mask: int


@lru_cache(maxsize=4)
def _width(mix_width: MixWidth) -> _Width:
    params = _PARAMS[mix_width]
    dtype = np.uint32 if params.bits == 32 else np.uint64
    span = (1 << params.bits) - 1
    return _Width(
        dtype=dtype,
        bits=params.bits,
        golden=dtype(params.golden & span),
        first_constant=dtype(params.first_constant & span),
        second_constant=dtype(params.second_constant & span),
        shifts=params.shifts,
        sign_mask=dtype(params.sign_mask),
        span_mask=span,
    )


class _Divisibility(NamedTuple):
    """How to test ``value % divisor == 0`` without dividing.

    ``multiplier`` is None for the bitwise-AND form, which is what tells the two apart;
    ``threshold`` is then unused. They are typed loosely because they hold numpy scalars of
    whichever width the mixer runs at.
    """

    mask: Any
    multiplier: Any
    rotation: int
    threshold: Any


@lru_cache(maxsize=16)
def _divisibility(divisor: int, mix_width: MixWidth) -> _Divisibility:
    """Precompute the divisibility test for one divisor.

    A power-of-two divisor reduces to an AND, and the sign mask the shared path applies
    first is redundant there: it clears only the top bit, which sits above every bit the
    AND reads. Anything else uses Lemire's test, which needs the sign mask kept because it
    is defined over the value the shared path actually takes the modulus of.
    """
    width = _width(mix_width)
    dtype = width.dtype
    if divisor & (divisor - 1) == 0 and divisor - 1 <= int(width.sign_mask):
        return _Divisibility(mask=dtype(divisor - 1), multiplier=None, rotation=0, threshold=None)
    rotation = (divisor & -divisor).bit_length() - 1
    odd_part = divisor >> rotation
    span = 1 << width.bits
    return _Divisibility(
        mask=width.sign_mask,
        multiplier=dtype(pow(odd_part, -1, span)),
        rotation=rotation,
        threshold=dtype((span - 1) // divisor),
    )


class Scratch:
    """Per-thread working buffers for the mask, kept across decode steps.

    The buffers are the reason this path is fast, and thread-local because a processor is
    documented as safe to share between threads. Buffers grow to the largest batch seen
    and are never shrunk, so a fluctuating batch does not refault its pages.
    """

    __slots__ = ("_local",)

    def __init__(self) -> None:
        self._local = threading.local()

    def take(self, rows: int, columns: int, dtype: Any) -> tuple[Any, Any, Any]:
        """``(value, spare, mask)`` buffers of shape ``(rows, columns)``."""
        held = getattr(self._local, "buffers", None)
        if (
            held is None
            or held[0].dtype != dtype
            or held[0].shape[1] != columns
            or held[0].shape[0] < rows
        ):
            capacity = max(rows, held[0].shape[0] * 2 if held is not None else 0, 1)
            held = (
                np.empty((capacity, columns), dtype=dtype),
                np.empty((capacity, columns), dtype=dtype),
                np.empty((capacity, columns), dtype=np.bool_),
            )
            self._local.buffers = held
        return held[0][:rows], held[1][:rows], held[2][:rows]

    def __repr__(self) -> str:
        held = getattr(self._local, "buffers", None)
        return f"{type(self).__name__}({held[0].shape if held else 'unallocated'})"


def green_mask_into(
    seeds: Any,
    token_ids: Any,
    divisor: int,
    mix_width: MixWidth,
    scratch: Scratch,
) -> Any:
    """The ``(batch, vocab_size)`` greenlist, written into ``scratch``'s buffers.

    The returned array is a view of a reused buffer: it is valid until the next call on
    this thread. Callers consume it immediately, which every caller here does.
    """
    _validate_divisor(divisor)
    width = _width(mix_width)
    ids = np.asarray(token_ids)
    unsigned_ids = ids.view(width.dtype) if ids.dtype.itemsize == width.bits // 8 else None
    if unsigned_ids is None:  # pragma: no cover - token_id_range always matches the width
        unsigned_ids = ids.astype(width.dtype)

    rows = int(np.asarray(seeds).shape[0])
    value, spare, out = scratch.take(rows, unsigned_ids.shape[0], width.dtype)

    # The seed is narrowed and multiplied per row, so this stays (batch, 1) - cheap
    # regardless of the vocabulary - and only the xor against the ids is full width. The
    # shared path narrows to a *signed* value of the width and relies on a wrapping
    # multiply; taking the low bits unsigned leaves the same bits.
    narrowed = np.asarray(seeds).astype(np.uint64) & np.uint64(width.span_mask)
    starts = narrowed.astype(width.dtype)
    np.bitwise_xor(unsigned_ids[None, :], (starts * width.golden)[:, None], out=value)

    first, second, third = width.shifts
    for distance, constant in (
        (first, width.first_constant),
        (second, width.second_constant),
        (third, None),
    ):
        np.right_shift(value, distance, out=spare)
        np.bitwise_xor(value, spare, out=value)
        if constant is not None:
            np.multiply(value, constant, out=value)

    test = _divisibility(divisor, mix_width)
    if test.multiplier is None:
        np.bitwise_and(value, test.mask, out=value)
        return np.equal(value, 0, out=out)

    np.bitwise_and(value, test.mask, out=value)
    np.multiply(value, test.multiplier, out=value)
    if test.rotation:
        np.right_shift(value, test.rotation, out=spare)
        np.left_shift(value, width.bits - test.rotation, out=value)
        np.bitwise_or(value, spare, out=value)
    return np.less_equal(value, test.threshold, out=out)


def green_mask(seeds: Any, token_ids: Any, divisor: int, mix_width: MixWidth) -> Any:
    """A freshly allocated greenlist, for callers that keep the result.

    :func:`green_mask_into` is what the hot path uses; this exists so tests and callers
    outside a decode loop are not handed a buffer that the next call overwrites.
    """
    return green_mask_into(seeds, token_ids, divisor, mix_width, Scratch()).copy()
