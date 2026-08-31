"""Watermarking for ``transformers``.

The watermark must reach the **raw** logits, before temperature, top-k and top-p. A green
token already masked to ``-inf`` cannot be rescued by adding delta, which would make the
watermark weak and erratic at low top-p and give delta a different meaning here than on
every other backend.

:func:`watermark` guarantees that by installing the processor at index 0 of the prepared
processor list, ahead of both the default processors and the warpers.

Measured placement, since this has moved between releases and is worth stating rather than
assuming. On transformers 4.57 and 5.16 a processor passed to
``generate(logits_processor=[...])`` lands *after* the default processors and *before* the
warpers::

    [0] RepetitionPenaltyLogitsProcessor
    [1] NoRepeatNGramLogitsProcessor
    [2] <- a processor passed to generate() arrives here
    [3] TemperatureLogitsWarper
    [4] TopKLogitsWarper
    [5] TopPLogitsWarper

So on those versions the naive route already satisfies the before-the-warpers contract.
Index 0 is kept because it is version-independent: it does not depend on where a given
release merges caller-supplied processors, and it needs no knowledge of which classes are
warpers - a distinction transformers no longer exposes, since the LogitsWarper marker base
class was removed. Guessing it wrong would apply delta to already-masked logits and
silently weaken the watermark.

Where a sign-dependent processor such as the repetition penalty is active, index 0 also
measures better. The penalty is multiplicative, so applying it to already-boosted logits
removes more: dividing a +6 green logit by 1.5 costs 2 logits, dividing a +0.5 raw logit
costs 0.17. Measured over 200-token generations, index 0 produced 178 distinct context
n-grams against 138 for the naive placement at delta=6 and penalty 1.5, for a green rate
0.2 points lower - more scored positions, and a higher z-score overall. With no
sign-dependent processor active the two placements are identical.

One rough edge to know about: delta pushes about 15% of previously-seen tokens' logits
across zero, which inverts the penalty's branch from multiply to divide for those. The
penalty is therefore not monotonic in delta, though the magnitude effect above dominates.

Installing means patching ``_get_logits_processor`` on the model instance. That is
intrusive, and deliberate: the alternative is documenting "please insert at index 0
yourself", which makes the correctness of the watermark depend on the user reading a
footnote. :func:`unwatermark` reverses it exactly, installing twice does not stack, and
:class:`WatermarkLogitsProcessor` stays public for anyone driving their own decode loop.

Watermarking is not output-preserving, and delta interacts with the rest of the pipeline:

* **Temperature rescales delta.** Delta lands on the raw logits and temperature divides
  them, so the sampler sees ``delta / temperature``. Lowering the temperature strengthens
  the watermark; raising it weakens it.
* **top-p and top-k select a different set**, not merely a reordered one. That is the
  point of running first, but the nucleus is no longer the model's nucleus.
* **Stopping behaviour shifts**, because the EOS token is green or red like any other.
* **Beam search scores include delta**, so ``sequences_scores`` are no longer the model's
  log-probabilities.
"""

from __future__ import annotations

from typing import Any, NoReturn

from llmwatermark.adapters.base import check_vocabulary, require_backend, resolve_vocab_size
from llmwatermark.config import WatermarkConfig
from llmwatermark.processor import CompileMode, WatermarkProcessor

__all__ = [
    "WatermarkLogitsProcessor",
    "config_for_model",
    "installed_processor",
    "unwatermark",
    "watermark",
]

_ORIGINAL_ATTRIBUTE = "_llmwatermark_original_get_logits_processor"
_PROCESSOR_ATTRIBUTE = "_llmwatermark_processor"


def _require_transformers(error: BaseException | None = None) -> NoReturn:
    require_backend("transformers", "transformers", error)


try:
    from transformers import LogitsProcessor
except ModuleNotFoundError as _import_error:  # pragma: no cover - exercised without the extra
    _require_transformers(_import_error)


class WatermarkLogitsProcessor(LogitsProcessor):  # type: ignore[misc]
    """The watermark as a ``transformers`` logits processor.

    Prefer :func:`watermark`, which pins this at index 0 regardless of where a given
    transformers release chooses to merge caller-supplied processors. Use this class
    directly when you drive the decode loop yourself and control the ordering.
    """

    def __init__(self, config: WatermarkConfig, *, compile: CompileMode | bool = "auto") -> None:
        self.config = config
        self.processor = WatermarkProcessor(config, compile=compile)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        window = self.config.h
        if int(input_ids.shape[1]) < window:
            # No full context window yet, so no greenlist is defined for this step.
            return scores
        # input_ids is already on the model's device, so the context slice, the greenlist
        # and the bias all stay there: no host transfer and no synchronisation.
        return self.processor.apply(scores, input_ids[:, -window:])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.processor!r})"


def watermark(model: Any, config: WatermarkConfig, *, compile: CompileMode | bool = "auto") -> Any:
    """Install the watermark on a model, returning the same model.

    Every subsequent ``model.generate(...)`` is watermarked, with the bias applied to the
    raw logits before temperature, top-k and top-p.

    Idempotent: installing twice replaces the first installation rather than stacking two
    biases. Reverse it with :func:`unwatermark`.
    """
    unwatermark(model)
    processor = WatermarkLogitsProcessor(config, compile=compile)
    original = model._get_logits_processor
    # None records that the method came from the class, so unwatermark knows to remove the
    # instance attribute rather than leave a bound method behind.
    previous = original if "_get_logits_processor" in vars(model) else None

    def _get_logits_processor(*args: Any, **kwargs: Any) -> Any:
        prepared = original(*args, **kwargs)
        prepared.insert(0, processor)
        return prepared

    setattr(model, _ORIGINAL_ATTRIBUTE, previous)
    setattr(model, _PROCESSOR_ATTRIBUTE, processor)
    model._get_logits_processor = _get_logits_processor
    return model


def unwatermark(model: Any) -> Any:
    """Remove the watermark, restoring the model exactly. Safe on an unwatermarked model."""
    if _ORIGINAL_ATTRIBUTE not in vars(model):
        return model
    previous = vars(model).pop(_ORIGINAL_ATTRIBUTE)
    vars(model).pop(_PROCESSOR_ATTRIBUTE, None)
    if previous is None:
        vars(model).pop("_get_logits_processor", None)
    else:
        model._get_logits_processor = previous
    return model


def installed_processor(model: Any) -> WatermarkLogitsProcessor | None:
    """The processor currently installed on a model, or None."""
    processor = vars(model).get(_PROCESSOR_ATTRIBUTE)
    return processor if isinstance(processor, WatermarkLogitsProcessor) else None


def config_for_model(
    model: Any, tokenizer: Any, *, secret_key: bytes | str, **parameters: Any
) -> WatermarkConfig:
    """Build a config for a model and its tokenizer.

    Reads ``model.config.vocab_size`` - the size the model actually generates over, which
    for a padded embedding matrix exceeds ``len(tokenizer)``. Partitioning the wrong one
    produces entirely different greenlists, so the value is taken from the model rather
    than the tokenizer, explicitly and in one documented place.
    """
    detected = int(model.config.vocab_size)
    config = WatermarkConfig.from_tokenizer(
        tokenizer,
        vocab_size=resolve_vocab_size(parameters, detected),
        secret_key=secret_key,
        **parameters,
    )
    check_vocabulary(detected, config, "this model")
    return config
