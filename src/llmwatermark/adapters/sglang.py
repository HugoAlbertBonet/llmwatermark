"""Watermarking for SGLang.

The third serving shape, and the friendliest of them. SGLang has a first-class
``CustomLogitProcessor`` API, so unlike vLLM there is no argument about where the hook goes
- there is one, it is documented, and it runs in the right place.

Observed against **SGLang 0.5.18**, recorded rather than assumed. ``Sampler._preprocess_logits``
applies custom processors first::

    if sampling_info.has_custom_logit_processor:
        apply_custom_logit_processor(logits, sampling_info)   # us
    ...
    logits.div_(sampling_info.temperatures)                   # temperature, later
    top_k_renorm_prob / top_p_renorm_prob / min_p_sampling_from_probs

So delta reaches the raw logits before temperature and before every warper - the same
relative position the transformers, vLLM and llama.cpp adapters take, which is what makes a
single delta mean one thing across all four.

``apply_custom_logit_processor`` hands each processor only its own rows::

    logits[batch_mask] = processor(logits[batch_mask], [custom_params[i] for i in indices])

so row *i* of the logits is row *i* of ``custom_param_list``, already aligned. This adapter
therefore needs no request bookkeeping at all - the vLLM adapter's ``RequestTracker`` exists
because vLLM hands over batch *edits* instead.

Three constraints follow from that design, and all three are load-bearing:

* **The processor is constructed with no arguments.** ``CustomLogitProcessor.from_str``
  deserializes the class and calls it bare, so the configuration cannot arrive through
  ``__init__``. It travels in ``custom_params`` instead.
* **``custom_params`` must be JSON-safe**, because it crosses SGLang's msgpack IPC. The
  watermark parameters are plain scalars and survive; the secret key does not travel there
  at all, for the reason given in :mod:`llmwatermark.adapters.base`.
* **``custom_params`` must be a dict or the token history is unreachable.** SGLang injects
  the ``Req`` object into that dict as ``__req__`` only when it is already a dict
  (``schedule_batch.py``), and ``Req`` is the only route to the tokens generated so far.
  :func:`watermark_sampling_params` always supplies one.

The engine must be started with ``enable_custom_logit_processor=True``; SGLang rejects the
request otherwise, and :func:`watermark_engine_kwargs` sets it.

**The overlap scheduler has to be off, and this is not optional.** SGLang's default overlap
mode launches the next forward pass before the previous step's token has been appended to
``req.output_ids``, so a processor reading that history sees it one token stale. The
watermark then keys each greenlist to the token *two* positions back while the detector
keys it to one, and the two disagree everywhere. Measured on Qwen2.5-0.5B at delta 2: green
fraction 0.128 against a 0.25 chance rate, z = -2.48, undetectable - and with the scheduler
disabled, green 0.500 and z = +4.32 from the same code. Nothing raises, and the text looks
completely normal; it simply does not detect. That is the failure mode this project treats
as the worst kind, so :func:`watermark_engine_kwargs` disables overlap, and the processor
additionally verifies it at run time rather than trusting the engine was built correctly.
"""

from __future__ import annotations

from typing import Any, Final, NoReturn

from llmwatermark.adapters.base import (
    HostContextStaging,
    check_vocabulary,
    publish_secret_key,
    require_backend,
    resolve_vocab_size,
    secret_key_from_environment,
)
from llmwatermark.adapters.sglang_requests import check_row_alignment, histories_from
from llmwatermark.config import WatermarkConfig
from llmwatermark.errors import ConfigError
from llmwatermark.processor import CompileMode, WatermarkProcessor
from llmwatermark.seeding import context_matrix

__all__ = [
    "CONFIG_KEY",
    "WatermarkLogitProcessor",
    "config_for_engine",
    "watermark_engine_kwargs",
    "watermark_sampling_params",
]

# The key our payload sits under inside custom_params, kept distinct from any the caller
# may already be passing for their own processors.
CONFIG_KEY: Final[str] = "llmwatermark"

_SETUP: Final[str] = "watermark_engine_kwargs()"


def _require_sglang(error: BaseException | None = None) -> NoReturn:
    require_backend("sglang", "sglang", error)


try:
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
except ModuleNotFoundError as _import_error:  # pragma: no cover - exercised without the extra
    _require_sglang(_import_error)


class WatermarkLogitProcessor(CustomLogitProcessor):  # type: ignore[misc]
    """The watermark as an SGLang custom logit processor.

    Prefer :func:`watermark_engine_kwargs` and :func:`watermark_sampling_params`, which set
    this up on both sides. SGLang instantiates it with no arguments inside the scheduler
    process, so everything it needs is read from the first request's ``custom_params``.
    """

    def __init__(self) -> None:
        self._processor: WatermarkProcessor | None = None
        self._staging = HostContextStaging()

    def __call__(self, logits: Any, custom_param_list: Any = None) -> Any:
        if not custom_param_list:
            # No request in this group carries our parameters, so there is nothing to bias.
            return logits

        check_row_alignment(int(logits.shape[0]), len(custom_param_list))
        processor = self._processor_for(custom_param_list[0])
        context, valid = context_matrix(histories_from(custom_param_list), processor.config.h)
        staged_context, staged_valid = self._staging.stage(context, valid, logits.device)
        return processor.apply(logits, staged_context, staged_valid)

    def _processor_for(self, params: Any) -> WatermarkProcessor:
        """Build the processor once, from the first request that carries a payload."""
        if self._processor is None:
            _refuse_overlap_schedule()
            payload = _payload_from(params)
            compile_mode = payload.pop("compile", "auto")
            config = WatermarkConfig.from_dict(
                payload, secret_key=secret_key_from_environment(_SETUP)
            )
            self._processor = WatermarkProcessor(config, compile=_compile_mode(compile_mode))
        return self._processor

    def __repr__(self) -> str:
        state = "unconfigured" if self._processor is None else repr(self._processor.config)
        return f"{type(self).__name__}({state})"


def _refuse_overlap_schedule() -> None:
    """Refuse to run under the overlap scheduler, which makes the history one token stale.

    Checked once, when the processor is first configured. An engine built without
    :func:`watermark_engine_kwargs` would otherwise emit text that never detects, with no
    error anywhere - so this converts a silent failure into a loud one.
    """
    try:
        from sglang.srt.server_args import get_global_server_args

        overlapped = not bool(get_global_server_args().disable_overlap_schedule)
    except Exception:  # pragma: no cover - older releases without the accessor
        return
    if overlapped:
        raise ConfigError(
            "SGLang's overlap scheduler is enabled, and the watermark cannot run under it. "
            "Overlap launches the next forward before the previous token is appended to the "
            "request history, so the greenlist would be keyed one token behind the detector "
            "and the output would never detect. Build the engine with "
            "sglang.Engine(..., **watermark_engine_kwargs(secret_key)), which disables it."
        )


def watermark_engine_kwargs(secret_key: bytes | str) -> dict[str, Any]:
    """Keyword arguments that let an ``Engine`` run the watermark.

        engine = sglang.Engine(model_path="...", **watermark_engine_kwargs(key))
        config = config_for_engine(engine, secret_key=key)

    Takes the key rather than a config on purpose. The engine is what reports the vocabulary
    size a config needs, so the config cannot exist yet - and the key has to reach the
    environment before the scheduler process is forked, so it cannot wait until later.

    SGLang refuses a request carrying a custom logit processor unless the server was started
    with the feature enabled, so this turns it on. The key goes through the environment
    rather than IPC, where SGLang would log it.

    It also disables the overlap scheduler, which is a throughput cost taken deliberately:
    under overlap the request history is one token stale and the watermark silently stops
    being detectable. See the module docstring for the measurement.
    """
    publish_secret_key(secret_key)
    return {"enable_custom_logit_processor": True, "disable_overlap_schedule": True}


def watermark_sampling_params(
    config: WatermarkConfig,
    *,
    compile: CompileMode | bool = "auto",
    **parameters: Any,
) -> dict[str, Any]:
    """The per-request arguments that apply the watermark.

        out = engine.generate(
            prompt,
            **watermark_sampling_params(config, temperature=0.8, max_new_tokens=128),
        )

    Returns ``custom_logit_processor`` and a ``sampling_params`` dict; any extra keyword
    arguments are merged into the latter, so ordinary sampling settings pass straight
    through. Every watermarked request needs these - SGLang applies custom processors per
    request, not per engine.
    """
    payload = config.to_dict()
    payload["compile"] = compile if isinstance(compile, str) else bool(compile)
    return {
        "custom_logit_processor": WatermarkLogitProcessor.to_str(),
        "sampling_params": {**parameters, "custom_params": {CONFIG_KEY: payload}},
    }


def config_for_engine(
    engine: Any, tokenizer: Any = None, *, secret_key: bytes | str, **parameters: Any
) -> WatermarkConfig:
    """Build a config for an SGLang engine and its tokenizer.

    Takes the vocabulary size from the model configuration rather than the tokenizer: a
    padded embedding matrix makes the model's count larger, and partitioning the wrong one
    produces entirely different greenlists.
    """
    resolved = _tokenizer_of(engine) if tokenizer is None else tokenizer
    detected = _vocab_size_of(engine)
    size = resolve_vocab_size(parameters, detected)
    config = WatermarkConfig.from_tokenizer(
        resolved, vocab_size=size, secret_key=secret_key, **parameters
    )
    check_vocabulary(detected, config, "SGLang")
    return config


def _payload_from(params: Any) -> dict[str, Any]:
    payload = params.get(CONFIG_KEY) if isinstance(params, dict) else None
    if not isinstance(payload, dict):
        raise ConfigError(
            "no watermark configuration reached the SGLang scheduler. Build the request "
            "with engine.generate(prompt, **watermark_sampling_params(config))."
        )
    return dict(payload)


def _compile_mode(mode: Any) -> CompileMode | bool:
    if isinstance(mode, bool):
        return mode
    return mode if mode in ("auto", "always", "never") else "auto"


def _tokenizer_of(engine: Any) -> Any:
    tokenizer = getattr(getattr(engine, "tokenizer_manager", None), "tokenizer", None)
    if tokenizer is None:
        raise ConfigError(
            "could not read a tokenizer from this SGLang engine; pass one explicitly to "
            "config_for_engine(engine, tokenizer, secret_key=...)."
        )
    return tokenizer


def _vocab_size_of(engine: Any) -> int:
    manager = getattr(engine, "tokenizer_manager", None)
    model_config = getattr(manager, "model_config", None)
    size = getattr(model_config, "vocab_size", None)
    if size is None:
        raise ConfigError(
            "could not read a vocabulary size from this SGLang engine. Pass vocab_size "
            "explicitly to WatermarkConfig.from_tokenizer(); it must be the size the model "
            "generates over, which a padded embedding matrix makes larger than the tokenizer."
        )
    return int(size)
