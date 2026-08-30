"""Watermarking for vLLM's V1 engine.

vLLM reschedules on every decode step, so a logits processor here receives the logits for
the whole batch and must work out for itself which request occupies which row. That
bookkeeping lives in :class:`~llmwatermark.adapters.vllm_tracker.RequestTracker`, which
imports nothing from vLLM and is tested on CPU.

Observed against **vLLM 0.28.0**, recorded rather than assumed:

* ``LogitsProcessor`` is constructed as ``(vllm_config, device, is_pin_memory)`` *inside
  the engine process*, and the processor class is pickled by reference to get there. A
  class built at runtime with a config bound to it fails to pickle, so the config travels
  as plain data instead - see :func:`watermark_llm_kwargs`.
* ``BatchUpdate`` carries ``batch_size``, ``removed``, ``added``, ``moved``, and documents
  that they must be applied in the order **removed, added, moved**.
* Each added request arrives as ``(index, sampling_params, prompt_token_ids,
  output_token_ids)``, where ``output_token_ids`` is a *live reference* to the list vLLM
  appends to. Reading its tail each step is what keeps the watermark stateless.
* ``prompt_token_ids`` is ``None`` unless a penalty or another processor asked vLLM to
  materialize it. See the note on the first h tokens below.

**Why is_argmax_invariant() must be False**, which is load-bearing twice over. vLLM splits
processors into two groups and applies them at different points in the sampler::

    logits = apply_logits_processors(...)   # non-argmax-invariant, i.e. us
    greedy_sampled = greedy_sample(logits)  # greedy requests sample here
    logits = apply_temperature(...)
    for p in logitsprocs.argmax_invariant:  # the other group runs here
        logits = p.apply(logits)
    random_sampled = topk_topp_sampler(...)

Declaring False puts the watermark before temperature and before greedy sampling, which
matches the delta-before-warpers contract and the transformers adapter. Declaring True
would move it after temperature *and* skip it entirely for greedy requests - which would
silently leave every greedy request unwatermarked.

**Where delta lands in vLLM's sampler**, measured on 0.28.0::

    apply_bad_words(...)                    # masks to -inf; survives delta, since -inf + d
    for p in logitsprocs.non_argmax_invariant:
        logits = p.apply(logits)            # <- the watermark
    logits = apply_penalties(...)           # frequency / presence / repetition

So delta reaches the logits before vLLM's penalties, the same relative position the
transformers adapter takes by inserting at index 0. Tokens already banned stay banned.

**The first h generated tokens.** Because vLLM usually passes ``prompt_token_ids=None``,
a request's context window is filled from its own output alone, so the first h generated
tokens receive no bias. transformers seeds those from the tail of the prompt instead. This
changes no greenlist and no detection arithmetic - the detector skips the first h
positions either way - only the watermark strength across h tokens of a generation.
"""

from __future__ import annotations

import os
from typing import Any, Final, NoReturn, cast

from llmwatermark.adapters.vllm_tracker import MOVE_SWAP, MOVE_UNIDIRECTIONAL, RequestTracker
from llmwatermark.config import WatermarkConfig
from llmwatermark.errors import ConfigError, SeedingError
from llmwatermark.processor import CompileMode, WatermarkProcessor

__all__ = [
    "PROCESSOR_PATH",
    "SECRET_KEY_VARIABLE",
    "WatermarkLogitsProcessor",
    "config_for_llm",
    "watermark_llm_kwargs",
]

# vLLM runs its engine in a separate process and pickles the processor class by reference,
# so the class must be importable by name and cannot be built on the fly with a config
# bound to it. The config travels as plain data in additional_config instead.
PROCESSOR_PATH: Final[str] = "llmwatermark.adapters.vllm:WatermarkLogitsProcessor"
CONFIG_KEY: Final[str] = "llmwatermark"

# The secret key is deliberately not part of additional_config: vLLM logs its engine
# configuration at startup, and a key in the logs is a key in every log aggregator.
SECRET_KEY_VARIABLE: Final[str] = "LLMWATERMARK_SECRET_KEY"


def _require_vllm(error: BaseException | None = None) -> NoReturn:
    raise ImportError(
        "the vLLM adapter needs the vllm package, which is an optional extra. "
        'Install it with:\n\n    pip install "llmwatermark[vllm]"\n'
    ) from error


try:
    from vllm.v1.sample.logits_processor import (
        BatchUpdate,
        LogitsProcessor,
        MoveDirectionality,
    )
except ModuleNotFoundError as _import_error:  # pragma: no cover - exercised without the extra
    _require_vllm(_import_error)


_DIRECTIONS = {
    MoveDirectionality.SWAP: MOVE_SWAP,
    MoveDirectionality.UNIDIRECTIONAL: MOVE_UNIDIRECTIONAL,
}


class WatermarkLogitsProcessor(LogitsProcessor):  # type: ignore[misc]
    """Applies the watermark inside vLLM's sampler.

    Do not instantiate directly - vLLM constructs logits processors itself inside its
    engine process. Use :func:`watermark_llm_kwargs` to wire it up.
    """

    def __init__(self, vllm_config: Any, device: Any, is_pin_memory: bool) -> None:
        config = _config_from(vllm_config)
        self.config = config
        self.device = device
        self.processor = WatermarkProcessor(config, compile=_compile_mode_from(vllm_config))
        self.tracker = RequestTracker()
        self._buffers: tuple[Any, Any, Any, Any] | None = None
        _check_vocabulary(vllm_config, config)

    def is_argmax_invariant(self) -> bool:
        """False: the watermark changes the argmax, and must run before temperature.

        See the module docstring - returning True would both move the bias after
        temperature and skip it entirely for greedy requests.
        """
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        """Mirror vLLM's batch layout. None means the layout did not change this step."""
        if batch_update is None:
            return
        self.tracker.apply(
            batch_size=batch_update.batch_size,
            removed=batch_update.removed,
            # vLLM's tuple is (index, sampling_params, prompt_ids, output_ids); the
            # sampling params are not ours to interpret.
            added=[(index, prompt, output) for index, _, prompt, output in batch_update.added],
            moved=[
                (source, target, _DIRECTIONS[direction])
                for source, target, direction in batch_update.moved
            ],
        )

    def apply(self, logits: Any) -> Any:
        """Add delta to each row's green tokens, in place."""
        rows = int(logits.shape[0])
        if rows != self.tracker.batch_size:
            raise SeedingError(
                f"vLLM passed {rows} rows of logits but the tracked batch holds "
                f"{self.tracker.batch_size}. The watermark cannot tell which request owns "
                "which row, so it refuses rather than biasing the wrong greenlists."
            )
        context, valid = self.tracker.contexts(self.config.h)
        if not valid.any():
            return logits
        device_context, device_valid = self._stage(context, valid, logits.device)
        return self.processor.apply(logits, device_context, device_valid)

    def _stage(self, context: Any, valid: Any, device: Any) -> tuple[Any, Any]:
        """Move this step's context to the device without stalling the pipeline.

        The context is assembled on the host, so it has to be copied every step. Handing
        a pageable array straight to ``torch.as_tensor(..., device=...)`` makes that copy
        *blocking*, which drains vLLM's queued GPU work once per decode step. The copy
        itself is tens of microseconds; the bubble it opens is measured in milliseconds.

        Staging through reusable pinned buffers keeps the copy asynchronous, and reusing
        them means no allocation on the hot path either.
        """
        import torch

        rows, window = context.shape
        buffers = self._buffers
        if buffers is None or buffers[0].shape[0] < rows or buffers[0].shape[1] != window:
            capacity = max(rows, 2 * (buffers[0].shape[0] if buffers else 0), 32)
            try:
                host_context = torch.empty((capacity, window), dtype=torch.int64, pin_memory=True)
                host_valid = torch.empty(capacity, dtype=torch.bool, pin_memory=True)
            except RuntimeError:  # pragma: no cover - pinned memory is not always available
                host_context = torch.empty((capacity, window), dtype=torch.int64)
                host_valid = torch.empty(capacity, dtype=torch.bool)
            buffers = (
                host_context,
                host_valid,
                torch.empty((capacity, window), dtype=torch.int64, device=device),
                torch.empty(capacity, dtype=torch.bool, device=device),
            )
            self._buffers = buffers

        host_context, host_valid, device_context, device_valid = buffers
        host_context.numpy()[:rows] = context
        host_valid.numpy()[:rows] = valid
        device_context[:rows].copy_(host_context[:rows], non_blocking=True)
        device_valid[:rows].copy_(host_valid[:rows], non_blocking=True)
        return device_context[:rows], device_valid[:rows]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.tracker!r})"


def watermark_llm_kwargs(
    config: WatermarkConfig, *, compile: CompileMode | bool = "auto"
) -> dict[str, Any]:
    """Keyword arguments that turn an ``LLM`` into a watermarked one.

        llm = LLM(model="...", **watermark_llm_kwargs(config))

    vLLM constructs logits processors inside its engine process, so a config cannot simply
    be bound to a class and handed over - a class built at runtime is not picklable across
    that boundary. The processor is therefore named by import path, and the watermark
    parameters travel as plain data in ``additional_config``.

    **The secret key travels separately**, in the ``LLMWATERMARK_SECRET_KEY`` environment
    variable, which this function sets in the current process so the engine subprocess
    inherits it. vLLM logs its engine configuration at startup; a key placed in that
    configuration would be a key in every log file. Set the variable yourself instead if
    your deployment already has a way to inject secrets.
    """
    os.environ[SECRET_KEY_VARIABLE] = config.secret_key.hex()
    payload = config.to_dict()
    payload["compile"] = compile if isinstance(compile, str) else bool(compile)
    return {
        "logits_processors": [PROCESSOR_PATH],
        "additional_config": {CONFIG_KEY: payload},
    }


def config_for_llm(
    llm: Any, tokenizer: Any = None, *, secret_key: bytes | str, **parameters: Any
) -> WatermarkConfig:
    """Build a config for a vLLM engine and its tokenizer.

    Takes the vocabulary size from the model config - the size vLLM actually generates
    over, which for a padded embedding matrix exceeds ``len(tokenizer)``. Partitioning the
    wrong one produces entirely different greenlists.
    """
    resolved = llm.get_tokenizer() if tokenizer is None else tokenizer
    return WatermarkConfig.from_tokenizer(
        resolved, vocab_size=_vocab_size_of(llm), secret_key=secret_key, **parameters
    )


def _payload_from(vllm_config: Any) -> dict[str, Any]:
    additional = getattr(vllm_config, "additional_config", None) or {}
    payload = additional.get(CONFIG_KEY) if isinstance(additional, dict) else None
    if not isinstance(payload, dict):
        raise ConfigError(
            "no watermark configuration reached the vLLM engine. Build the engine with "
            "LLM(model=..., **llmwatermark.adapters.vllm.watermark_llm_kwargs(config))."
        )
    return dict(payload)


def _config_from(vllm_config: Any) -> WatermarkConfig:
    payload = _payload_from(vllm_config)
    payload.pop("compile", None)
    key = os.environ.get(SECRET_KEY_VARIABLE)
    if not key:
        raise ConfigError(
            f"the watermark secret key is missing: {SECRET_KEY_VARIABLE} is not set in the "
            "engine process. watermark_llm_kwargs() sets it in the parent process so the "
            "engine inherits it; set it yourself if you launch the engine separately."
        )
    return WatermarkConfig.from_dict(payload, secret_key=bytes.fromhex(key))


def _compile_mode_from(vllm_config: Any) -> CompileMode | bool:
    mode = _payload_from(vllm_config).get("compile", "auto")
    if isinstance(mode, bool):
        return mode
    if mode in ("auto", "always", "never"):
        return cast("CompileMode", mode)
    return "auto"


def _vocab_size_of(source: Any) -> int:
    model_config = getattr(source, "model_config", None)
    if model_config is None:
        engine = getattr(source, "llm_engine", None)
        model_config = getattr(engine, "model_config", None)
    if model_config is None:
        raise ConfigError(
            f"could not read a vocabulary size from {type(source).__name__}. Pass "
            "vocab_size explicitly to WatermarkConfig.from_tokenizer()."
        )
    return int(model_config.get_vocab_size())


def _check_vocabulary(vllm_config: Any, config: WatermarkConfig) -> None:
    """Fail at construction rather than producing a wrong greenlist at every step."""
    try:
        actual = _vocab_size_of(vllm_config)
    except (ConfigError, AttributeError):  # pragma: no cover - depends on the vLLM build
        return
    if actual != config.vocab_size:
        raise ConfigError(
            f"vLLM generates over {actual} token IDs but the watermark config declares "
            f"vocab_size={config.vocab_size}. The greenlist partitions token IDs, so the "
            "two must match exactly. Build the config with "
            "llmwatermark.adapters.vllm.config_for_llm(llm, secret_key=...)."
        )
