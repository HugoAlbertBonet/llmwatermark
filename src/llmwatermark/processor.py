"""The logits processor: where delta meets the model's logits.

One decode step: seed each row from its own last h token IDs, decide green or red for the
whole vocabulary, and add delta to the green entries in place. Backend adapters wrap this;
none of them reimplement it.

**Compilation is on by default.** On CUDA the processor wraps its kernel in
``torch.compile``. This is not an optimization, it is what makes the tool meet its stated
performance budget: eager mode runs eight separate elementwise kernels over a
``batch x vocab_size`` buffer, each writing its result to memory, and measures at
memory-bandwidth saturation. Compiled, the whole step fuses into a single kernel that
reads the logits, decides green or red in registers and writes the logits back - roughly
17x faster at batch 32, with bit-identical output. Pass ``compile=False`` to disable it,
``compile=True`` to force it on CPU as well. See :attr:`compile_mode`.

**Statelessness.** The greenlist at a position is a pure function of
``(secret_key, last h token IDs)``, read from the row's own history every step. Nothing
here is keyed by batch index, row position or request ID. That is what lets batching,
beam reordering, vLLM preemption and speculative rollback work without corrupting the
watermark - all of which silently produce undetectable text if state desynchronises.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Final, Literal

from llmwatermark.arrays import is_torch
from llmwatermark.config import HashScheme, MixWidth, WatermarkConfig
from llmwatermark.errors import ConfigError, SeedingError
from llmwatermark.greenlist import is_green, token_id_range
from llmwatermark.seeding import (
    SeedTable,
    context_matrix,
    gather_seeds,
    validate_context_shape,
)

__all__ = ["CompileMode", "WatermarkProcessor"]

CompileMode = Literal["auto", "always", "never"]

_COMPILE_MODES: Final[tuple[str, ...]] = ("auto", "always", "never")

_logger = logging.getLogger("llmwatermark")


def _biased(
    logits: Any,
    context: Any,
    valid: Any,
    table: Any,
    ids: Any,
    scheme: HashScheme,
    divisor: int,
    delta: float,
    width: MixWidth,
) -> Any:
    """One watermarking step, in place.

    Written as a single expression graph with no data-dependent branching, so
    ``torch.compile`` can fuse it into one kernel. Everything after the seed gather stays
    in registers: no ``batch x vocab_size`` temporary reaches memory.
    """
    seeds = gather_seeds(table, context, scheme)
    green = is_green(seeds[:, None], ids[None, :], divisor, width)
    if valid is not None:
        # Positions without a full context window get no greenlist, so no bias.
        green = green & valid[:, None]
    adder = getattr(logits, "add_", None)
    if adder is None:
        logits += green * delta
        return logits
    return adder(green, alpha=delta)


class WatermarkProcessor:
    """Applies one watermark's logit bias, on any array library and any device.

    :param config: The watermark to apply.
    :param compile: ``"auto"`` (default) compiles on CUDA and stays eager elsewhere;
        ``True`` always compiles; ``False`` never does. The default is on because the
        eager path does not meet the performance budget - see the module docstring.

    The processor is stateless with respect to the generation: it may be shared across
    requests, batches and threads, and holds nothing that can desynchronise.
    """

    def __init__(self, config: WatermarkConfig, *, compile: CompileMode | bool = "auto") -> None:
        self.config = config
        self._compile_mode = _normalize_compile_mode(compile)
        self._table = SeedTable.for_config(config)
        self._compiled: Any = None
        self._compile_failed = False

    @property
    def compile_mode(self) -> str:
        """``"auto"``, ``"always"`` or ``"never"``. Set with the ``compile`` argument."""
        return self._compile_mode

    @property
    def is_compiled(self) -> bool:
        """Whether a compiled kernel has actually been built and is in use."""
        return self._compiled is not None

    def apply(self, logits: Any, context: Any, valid: Any = None) -> Any:
        """Add delta to each row's green tokens, in place.

        :param logits: ``(batch, vocab_size)``, mutated in place and also returned.
        :param context: ``(batch, h)`` integer array of each row's last h token IDs,
            oldest first. May live on the same device as the logits, in which case
            nothing crosses to the host.
        :param valid: Optional ``(batch,)`` boolean mask. Rows marked False have too
            little history for a full context window and are left untouched.

        Delta is added to the raw logits. It must run *before* temperature, top-k and
        top-p: a green token already masked to -inf cannot be rescued, which makes the
        watermark weak and erratic at low top-p. Adapters are responsible for that
        ordering.
        """
        self._validate(logits, context, valid)
        if not is_torch(context):
            # Reads values, so only ever on the host path. On device this would sync.
            self._table.validate_context_values(context)

        table = self._table.on(logits)
        ids = token_id_range(self.config.vocab_size, self.config.mix_width, like=logits)
        kernel = self._kernel_for(logits)
        return kernel(
            logits,
            context,
            valid,
            table,
            ids,
            self.config.scheme,
            self.config.green_divisor,
            self.config.delta,
            self.config.mix_width,
        )

    def apply_to_histories(self, logits: Any, histories: Any) -> Any:
        """Apply the bias from ragged per-row token histories.

        For backends that hand back Python lists of generated token IDs (vLLM, SGLang)
        rather than a rectangular tensor. Backends that already hold ``(batch, time)`` on
        the device should slice ``[:, -h:]`` themselves and call :meth:`apply`, which
        avoids the host round trip entirely.
        """
        context, valid = context_matrix(histories, self.config.h)
        if is_torch(logits):
            import torch

            context = torch.as_tensor(context, device=logits.device)
            valid = torch.as_tensor(valid, device=logits.device)
        return self.apply(logits, context, valid)

    def _validate(self, logits: Any, context: Any, valid: Any) -> None:
        """Shape and dtype checks only. Never reads array values, so never synchronises."""
        if getattr(logits, "ndim", None) != 2:
            shape = getattr(logits, "shape", type(logits).__name__)
            raise SeedingError(f"logits must have shape (batch, vocab_size), got shape {shape}.")
        batch, vocab_size = int(logits.shape[0]), int(logits.shape[1])
        if vocab_size != self.config.vocab_size:
            raise ConfigError(
                f"logits have {vocab_size} columns but the watermark config declares "
                f"vocab_size={self.config.vocab_size}. The greenlist partitions token IDs, "
                "so the two must match exactly. Build the config with the vocab_size the "
                "model generates over, usually model.config.vocab_size."
            )
        validate_context_shape(context, self.config.scheme)
        if int(context.shape[0]) != batch:
            raise SeedingError(
                f"batch mismatch: logits have {batch} rows but context has {int(context.shape[0])}."
            )
        if int(context.shape[1]) != self.config.h:
            raise SeedingError(
                f"context must be {self.config.h} tokens wide for "
                f"{self.config.scheme.value}, got {int(context.shape[1])}."
            )
        if valid is not None and (valid.ndim != 1 or int(valid.shape[0]) != batch):
            raise SeedingError(f"valid must have shape ({batch},), got shape {tuple(valid.shape)}.")

    def _kernel_for(self, logits: Any) -> Any:
        """Pick the compiled or eager kernel, compiling once and caching the result."""
        if self._compile_mode == "never" or self._compile_failed or not is_torch(logits):
            return _biased
        if self._compile_mode == "auto" and not logits.is_cuda:
            return _biased
        if self._compiled is None:
            self._compiled = self._build_compiled()
        return self._compiled if self._compiled is not None else _biased

    def _build_compiled(self) -> Any:
        try:
            import torch

            _logger.info(
                "compiling the watermark kernel (compile=%r). The eager path does not meet "
                "the performance budget; pass compile=False to WatermarkProcessor to "
                "disable this.",
                self._compile_mode,
            )
            # dynamic=True: batch size changes constantly in serving, and recompiling per
            # batch size would cost far more than the watermark itself.
            return torch.compile(_biased, dynamic=True)
        except Exception as exc:  # pragma: no cover - depends on the torch build
            self._compile_failed = True
            warnings.warn(
                f"could not compile the watermark kernel ({exc!r}); falling back to the "
                "eager path, which is slower. Pass compile=False to silence this.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(vocab_size={self.config.vocab_size}, "
            f"gamma={self.config.gamma}, delta={self.config.delta}, "
            f"scheme={self.config.scheme.value!r}, compile={self._compile_mode!r})"
        )


def _normalize_compile_mode(value: CompileMode | bool) -> str:
    if value is True:
        return "always"
    if value is False:
        return "never"
    if value in _COMPILE_MODES:
        return str(value)
    raise ConfigError(
        f"compile must be True, False, or one of {', '.join(_COMPILE_MODES)}. Got {value!r}."
    )
