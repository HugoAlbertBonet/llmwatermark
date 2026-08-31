"""Watermarking for llama.cpp, via ``llama-cpp-python``.

A third shape, and the simplest one. llama.cpp's logits processors are plain callables over
**numpy** arrays for a **single sequence**: ``(input_ids, scores) -> scores``, where
``input_ids`` is the whole token history as a 1-D array and ``scores`` is the vocabulary's
logits. Nothing is batched and nothing is on a device, so this adapter needs no torch, no
staging and no bookkeeping - the core's numpy path carries it end to end.

Observed against **llama-cpp-python 0.3.35**, recorded rather than assumed. ``_init_sampler``
builds the chain in this order::

    sampler.add_custom(...)        # every entry in logits_processor, i.e. us
    sampler.add_penalties(...)     # repetition, frequency, presence
    sampler.add_grammar(...)
    sampler.add_top_k / add_typical / add_top_p / add_min_p / add_temp / add_dist

So delta reaches the raw logits before the penalties and before every warper - the same
relative position the transformers adapter takes by inserting at index 0, and the same one
vLLM gives a non-argmax-invariant processor. All three backends agree on where delta lands,
which is what makes a single delta mean one thing across them.

``generate()`` is the chokepoint: ``create_completion``, ``__call__`` and
``create_chat_completion`` all route through it, so that is the single method
:func:`watermark` wraps.
"""

from __future__ import annotations

from typing import Any, NoReturn

import numpy as np

from llmwatermark.adapters.base import check_vocabulary, require_backend, resolve_vocab_size
from llmwatermark.config import WatermarkConfig
from llmwatermark.processor import WatermarkProcessor

__all__ = [
    "WatermarkLogitsProcessor",
    "config_for_llama",
    "installed_processor",
    "unwatermark",
    "watermark",
]

_ORIGINAL_ATTRIBUTE = "_llmwatermark_original_generate"
_PROCESSOR_ATTRIBUTE = "_llmwatermark_processor"


def _require_llama_cpp(error: BaseException | None = None) -> NoReturn:
    require_backend("llama-cpp-python", "llama-cpp", error)


try:
    import llama_cpp as _llama_cpp
    from llama_cpp import LogitsProcessorList
except ModuleNotFoundError as _import_error:  # pragma: no cover - exercised without the extra
    _require_llama_cpp(_import_error)

# llama_token_get_text was renamed to llama_vocab_get_text and now warns. Prefer the new
# name where it exists so the adapter works on both without emitting deprecations.
_token_text = getattr(_llama_cpp, "llama_vocab_get_text", None) or _llama_cpp.llama_token_get_text


class WatermarkLogitsProcessor:
    """The watermark as a llama.cpp logits processor.

    llama.cpp asks only for a callable, so there is no base class to inherit. Prefer
    :func:`watermark`, which installs this for every generation path at once.
    """

    def __init__(self, config: WatermarkConfig, *, compile: bool = False) -> None:
        self.config = config
        # Scores arrive as a host numpy array, so there is no device kernel to fuse and
        # nothing for torch.compile to do.
        self.processor = WatermarkProcessor(config, compile=compile)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        window = self.config.h
        history = np.asarray(input_ids, dtype=np.int64)
        if history.size < window:
            # No full context window yet, so no greenlist is defined for this step.
            return scores
        logits = np.asarray(scores)
        context = history[-window:].reshape(1, window)
        self.processor.apply(logits.reshape(1, -1), context)
        return logits

    def __repr__(self) -> str:
        return f"{type(self).__name__}(vocab_size={self.config.vocab_size})"


def watermark(llama: Any, config: WatermarkConfig, *, compile: bool = False) -> Any:
    """Install the watermark on a ``Llama``, returning the same object.

    Every subsequent completion is watermarked. Idempotent; reverse with
    :func:`unwatermark`.
    """
    unwatermark(llama)
    check_vocabulary(_vocab_size_of(llama), config, "llama.cpp")
    processor = WatermarkLogitsProcessor(config, compile=compile)
    original = llama.generate

    def generate(*args: Any, **kwargs: Any) -> Any:
        supplied = kwargs.get("logits_processor")
        existing = list(supplied) if supplied else []
        if not any(isinstance(entry, WatermarkLogitsProcessor) for entry in existing):
            # Front of the list, so delta lands on logits no other processor has touched.
            existing.insert(0, processor)
        kwargs["logits_processor"] = LogitsProcessorList(existing)
        return original(*args, **kwargs)

    setattr(llama, _ORIGINAL_ATTRIBUTE, original if "generate" in vars(llama) else None)
    setattr(llama, _PROCESSOR_ATTRIBUTE, processor)
    llama.generate = generate
    return llama


def unwatermark(llama: Any) -> Any:
    """Remove the watermark. Safe on an object that was never watermarked."""
    if _ORIGINAL_ATTRIBUTE not in vars(llama):
        return llama
    previous = vars(llama).pop(_ORIGINAL_ATTRIBUTE)
    vars(llama).pop(_PROCESSOR_ATTRIBUTE, None)
    if previous is None:
        vars(llama).pop("generate", None)
    else:
        llama.generate = previous
    return llama


def installed_processor(llama: Any) -> WatermarkLogitsProcessor | None:
    """The processor currently installed, or None."""
    processor = vars(llama).get(_PROCESSOR_ATTRIBUTE)
    return processor if isinstance(processor, WatermarkLogitsProcessor) else None


class LlamaCppVocabulary:
    """Adapts a loaded ``Llama`` to the tokenizer interface the fingerprint expects.

    ``llama_vocab_get_text`` returns the raw piece as **bytes**, which the fingerprint hashes
    directly - the same bytes a str piece encodes to, so a vocabulary fingerprints identically
    whether it arrived from here or from a transformers tokenizer.

    The length reported here is deliberately *not* ``n_vocab``. llama.cpp pads its vocabulary
    up to the model's embedding width and invents placeholder pieces - ``b"[PAD151665]"`` and
    so on - for the slots that carry no real token, where a transformers tokenizer simply has
    nothing. Fingerprinting over that region would make the two backends disagree about a
    vocabulary they actually share: measured on Qwen2.5-0.5B, zero of 256 sampled pieces
    differ below the padding, and every one differs inside it.

    Those slots are marked ``LLAMA_TOKEN_ATTR_UNUSED``, so the real piece count is found by
    walking down from the top while that attribute holds - a documented signal rather than a
    guess at the placeholder format.
    """

    def __init__(self, llama: Any) -> None:
        self._vocab = llama._model.vocab
        self._size = _piece_count(self._vocab, _vocab_size_of(llama) or 0)

    def __len__(self) -> int:
        return int(self._size)

    def id_to_piece(self, token_id: int) -> bytes:
        piece: bytes = _token_text(self._vocab, int(token_id))
        return piece


def config_for_llama(llama: Any, *, secret_key: bytes | str, **parameters: Any) -> WatermarkConfig:
    """Build a config from a loaded ``Llama``, taking the vocabulary from the model itself."""
    size = _vocab_size_of(llama)
    if size is None:
        raise ValueError(
            "could not read a vocabulary size from this Llama object; pass vocab_size "
            "explicitly to WatermarkConfig.from_tokenizer()."
        )
    config = WatermarkConfig.from_tokenizer(
        LlamaCppVocabulary(llama),
        vocab_size=resolve_vocab_size(parameters, size),
        secret_key=secret_key,
        **parameters,
    )
    check_vocabulary(size, config, "llama.cpp")
    return config


def _piece_count(vocab: Any, declared: int) -> int:
    """How many leading token IDs carry a real piece rather than padding."""
    unused = getattr(_llama_cpp, "LLAMA_TOKEN_ATTR_UNUSED", None)
    attribute = getattr(_llama_cpp, "llama_vocab_get_attr", None)
    if unused is None or not callable(attribute):  # pragma: no cover - older builds
        return declared
    count = declared
    # Padding sits at the top and is small; the bound stops a pathological vocabulary from
    # turning this into a full scan.
    limit = max(declared - 100_000, 0)
    while count > limit and attribute(vocab, count - 1) == unused:
        count -= 1
    return count


def _vocab_size_of(llama: Any) -> int | None:
    getter = getattr(llama, "n_vocab", None)
    if callable(getter):
        return int(getter())
    return None
