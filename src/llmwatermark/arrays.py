"""Small helpers for code that must run on both numpy arrays and torch tensors.

The hot path is written once and executed by whichever library the backend hands us. The
two libraries agree on almost every operator this package needs; these helpers cover the
few places they do not, so no module needs its own dispatch logic.

Nothing here imports torch. Tensors are recognized by their module name, which keeps the
core package importable with numpy alone.
"""

from __future__ import annotations

from typing import Any

__all__ = ["as_dtype", "is_torch", "row_min"]


def is_torch(values: Any) -> bool:
    """True for torch tensors, without importing torch."""
    return type(values).__module__.split(".")[0] == "torch"


def as_dtype(values: Any, dtype: Any) -> Any:
    """Cast to a dtype of the value's own library (``astype`` vs ``to``)."""
    caster = getattr(values, "astype", None)
    return caster(dtype) if caster is not None else values.to(dtype)


def row_min(values: Any) -> Any:
    """Minimum along axis 1.

    numpy returns the values; torch returns a ``(values, indices)`` pair.
    """
    result = values.min(1)
    return getattr(result, "values", result)
