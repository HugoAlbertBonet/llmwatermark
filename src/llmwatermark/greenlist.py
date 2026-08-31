"""Greenlist construction: which token IDs are boosted at one position.

Given the seed for a position (see :mod:`llmwatermark.seeding`), a token is green iff::

    mix(seed, token_id) % round(1 / gamma) == 0

Picking one residue class out of ``round(1/gamma)`` is what makes the greenlist a gamma
fraction of the vocabulary. Nothing is sorted, counted or permuted: the fraction falls out
of the mixer's output being uniform. That is what keeps this affordable at 128k tokens
times batch size, every decode step.

The mixer is a murmur3-style bit avalanche, deliberately *not* cryptographic. It runs over
the whole vocabulary every step, where an HMAC would cost orders of magnitude too much.
All secrecy lives in the seed, which is HMAC-SHA256 keyed by the user's secret.

Two shapes, one expression:

* generation broadcasts ``(batch, 1)`` seeds against ``(1, vocab_size)`` token IDs to get
  the full ``(batch, vocab_size)`` mask;
* detection asks only about the token actually emitted, elementwise over positions, and so
  never materializes a greenlist at all.

Both go through :func:`mix`, so the detector cannot drift away from the generator.

The same code runs on numpy arrays and torch tensors: it uses only multiplication, xor,
shifts, masking and comparison, which both libraries spell identically. Three details
decide whether the two agree bit for bit, and all three are handled explicitly below:

* ``>>`` is an *arithmetic* shift on signed integers in both libraries, so the mixer's
  logical shift is emulated by masking off the sign-extended bits;
* the murmur constants exceed the signed range and are written as their two's-complement
  negatives, relying on wrapping multiplication, which both libraries provide;
* ``%`` disagrees on negative operands (numpy follows Python, torch follows C), so the
  sign bit is masked off before any modulus is taken.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Final, NamedTuple

import numpy as np

from llmwatermark.arrays import as_dtype, is_torch
from llmwatermark.config import MixWidth, validate_vocab_size
from llmwatermark.errors import SeedingError

__all__ = ["green_mask", "is_green", "mix", "token_id_range"]

# The smallest divisor that partitions anything: gamma <= 0.5.
MIN_DIVISOR: Final[int] = 2


class _MixParams(NamedTuple):
    """Constants for one mixer width, as signed values of that width."""

    bits: int
    golden: int
    first_constant: int
    second_constant: int
    shifts: tuple[int, int, int]
    sign_mask: int


def _signed(value: int, bits: int) -> int:
    """Reinterpret an unsigned constant as the signed integer with the same bits."""
    return value - (1 << bits) if value >= (1 << (bits - 1)) else value


_PARAMS: Final[dict[MixWidth, _MixParams]] = {
    MixWidth.BITS32: _MixParams(
        bits=32,
        golden=_signed(0x9E3779B1, 32),
        first_constant=_signed(0x85EBCA6B, 32),
        second_constant=_signed(0xC2B2AE35, 32),
        shifts=(16, 13, 16),
        sign_mask=(1 << 31) - 1,
    ),
    MixWidth.BITS64: _MixParams(
        bits=64,
        golden=_signed(0x9E3779B97F4A7C15, 64),
        first_constant=_signed(0xFF51AFD7ED558CCD, 64),
        second_constant=_signed(0xC4CEB9FE1A85EC53, 64),
        shifts=(33, 29, 32),
        sign_mask=(1 << 63) - 1,
    ),
}


def mix(seeds: Any, token_ids: Any, width: MixWidth | int = MixWidth.BITS32) -> Any:
    """Avalanche a seed and a token ID into a uniformly distributed integer.

    Accepts numpy arrays or torch tensors, and broadcasts them against each other. The
    result is non-negative and of the requested width, so that the caller's ``%`` behaves
    identically in both libraries.
    """
    params = _PARAMS[MixWidth.parse(width)]
    ids = _as_array(token_ids, "token_ids")
    seed_values = _as_array(seeds, "seeds")

    target = _mix_dtype(ids, params)
    # The seed is 63 bits wide; narrowing happens in the wider type, before the cast, so
    # no value is ever cast out of range.
    narrowed = _narrow(seed_values, params.bits)

    value = (as_dtype(narrowed, target) * params.golden) ^ as_dtype(ids, target)
    first, second, third = params.shifts
    value = value ^ _logical_shift_right(value, first, params.bits)
    value = value * params.first_constant
    value = value ^ _logical_shift_right(value, second, params.bits)
    value = value * params.second_constant
    value = value ^ _logical_shift_right(value, third, params.bits)
    # Drop the sign bit rather than the low bits: the modulus reads the low end, and a
    # non-negative operand is the only way numpy and torch agree on `%`.
    return value & params.sign_mask


def is_green(
    seeds: Any, token_ids: Any, divisor: int, width: MixWidth | int = MixWidth.BITS32
) -> Any:
    """Elementwise green/red decision, broadcasting seeds against token IDs.

    :param divisor: ``round(1 / gamma)``, from :attr:`WatermarkConfig.green_divisor`.
    :returns: A boolean array of the broadcast shape.

    Detection calls this with one seed and one observed token per position, which is
    ``O(tokens)`` and never builds a greenlist.
    """
    _validate_divisor(divisor)
    return mix(seeds, token_ids, width) % divisor == 0


def green_mask(
    seeds: Any, token_ids: Any, divisor: int, width: MixWidth | int = MixWidth.BITS32
) -> Any:
    """The full ``(batch, vocab_size)`` greenlist for one decode step.

    :param seeds: One seed per row, shape ``(batch,)``.
    :param token_ids: The vocabulary, shape ``(vocab_size,)`` - pass
        :func:`token_id_range` so it is allocated once rather than per step.

    Fully vectorized, with no Python loop over the batch and no host/device transfer. The
    mask is built wherever ``token_ids`` lives, which is how it ends up on the same device
    as the logits.
    """
    seed_values = _as_array(seeds, "seeds")
    ids = _as_array(token_ids, "token_ids")
    if seed_values.ndim != 1:
        raise SeedingError(f"seeds must have shape (batch,), got shape {seed_values.shape}.")
    if ids.ndim != 1:
        raise SeedingError(f"token_ids must have shape (vocab_size,), got shape {ids.shape}.")
    return is_green(seed_values[:, None], ids[None, :], divisor, width)


def token_id_range(
    vocab_size: int, width: MixWidth | int = MixWidth.BITS32, *, like: Any = None
) -> Any:
    """The vocabulary as an array, cached so the hot path never reallocates it.

    :param like: An array or tensor whose library and device the range should match. The
        greenlist is built wherever this lives, so passing the logits tensor is what keeps
        the mask on the same device with no transfer.
    """
    validate_vocab_size(vocab_size)
    resolved = MixWidth.parse(width)
    if like is None or not is_torch(like):
        return _numpy_range(vocab_size, resolved)
    return _torch_range(vocab_size, resolved, str(like.device))


@lru_cache(maxsize=8)
def _numpy_range(vocab_size: int, width: MixWidth) -> np.ndarray:
    values = np.arange(vocab_size, dtype=np.int32 if width is MixWidth.BITS32 else np.int64)
    values.flags.writeable = False
    return values


@lru_cache(maxsize=8)
def _torch_range(vocab_size: int, width: MixWidth, device: str) -> Any:
    import torch

    dtype = torch.int32 if width is MixWidth.BITS32 else torch.int64
    return torch.arange(vocab_size, dtype=dtype, device=device)


def _logical_shift_right(value: Any, distance: int, bits: int) -> Any:
    """Shift right without sign extension.

    ``>>`` on a signed integer copies the sign bit into the vacated positions in both
    numpy and torch. Masking those positions off leaves exactly the logical shift.
    """
    return (value >> distance) & ((1 << (bits - distance)) - 1)


def _narrow(seeds: Any, bits: int) -> Any:
    """Take the low ``bits`` of a seed as a signed value of that width.

    Done in the seed's own (wider) type so the subsequent cast is always in range, rather
    than relying on either library's out-of-range cast behaviour.
    """
    if bits >= 64:
        return seeds
    low = seeds & ((1 << bits) - 1)
    return low - (((low >> (bits - 1)) & 1) << bits)


def _as_array(values: Any, name: str) -> Any:
    """Accept numpy arrays, torch tensors and plain Python integers or sequences.

    Scalars are promoted to shape ``(1,)``. The mixer relies on wrapping multiplication,
    which numpy performs silently on arrays but reports as an overflow warning on scalars.
    Broadcasting makes the promotion invisible to callers.
    """
    array = values if is_torch(values) else np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    dtype = array.dtype
    if is_torch(array):
        # torch dtypes carry these flags; numpy's do not, and mypy only knows which is
        # which when torch happens to be installed, so ask rather than assert.
        is_integer = not (
            getattr(dtype, "is_floating_point", False) or getattr(dtype, "is_complex", False)
        )
    else:
        is_integer = bool(np.issubdtype(dtype, np.integer))
    if not is_integer:
        raise SeedingError(f"{name} must be an integer array, got dtype {dtype}.")
    return array


def _mix_dtype(reference: Any, params: _MixParams) -> Any:
    if is_torch(reference):
        import torch

        return torch.int32 if params.bits == 32 else torch.int64
    return np.int32 if params.bits == 32 else np.int64


def _validate_divisor(divisor: int) -> None:
    if not isinstance(divisor, (int, np.integer)) or isinstance(divisor, bool):
        raise SeedingError(f"divisor must be an integer, got {type(divisor).__name__}.")
    if int(divisor) < MIN_DIVISOR:
        raise SeedingError(
            f"divisor must be at least {MIN_DIVISOR}, got {divisor}. It is round(1/gamma), "
            "so a smaller value would green the whole vocabulary."
        )
