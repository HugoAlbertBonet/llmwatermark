"""Reading SGLang's per-request state, without importing SGLang.

The interesting parts of the SGLang adapter are not the ones that need a GPU: assembling
each row's token history, and refusing the case where the rows do not line up. Both are
pure bookkeeping over plain dicts, so they live here, where the default CPU suite can test
them. :mod:`llmwatermark.adapters.sglang` holds everything that genuinely needs the backend.

This mirrors :mod:`llmwatermark.adapters.vllm_tracker`, which exists for the same reason.
"""

from __future__ import annotations

from typing import Any

from llmwatermark.errors import ConfigError

__all__ = ["check_row_alignment", "histories_from"]

_SETUP = "watermark_sampling_params(config)"


def check_row_alignment(rows: int, requests: int) -> None:
    """Fail when the batch has more logits rows than requests.

    SGLang gives each request several draft positions under speculative decoding but still
    one parameter dict, so the two counts diverge. That is not a case to handle: the draft
    tokens are not in ``req.output_ids`` yet, so the greenlist for every position after the
    first is *unknowable*, not merely misaligned. Biasing the wrong rows would produce text
    that fails to detect, silently, which is the failure this project treats as the worst
    one - so it refuses instead.
    """
    if rows == requests:
        return
    raise ConfigError(
        f"the watermark received {rows} logits rows for {requests} requests, which happens "
        "under speculative decoding. The draft tokens are not yet in the request history, "
        "so the greenlist for those positions cannot be computed. Disable speculative "
        "decoding on a watermarked engine."
    )


def histories_from(custom_param_list: Any) -> list[list[int]]:
    """Each row's token IDs so far, prompt included, in batch order.

    SGLang attaches the request object to ``custom_params`` under ``__req__``, and only when
    ``custom_params`` is already a dict. A caller who builds requests by hand and passes
    nothing gets no history at all, so that case is named explicitly rather than surfacing
    later as an attribute error.
    """
    histories = []
    for index, params in enumerate(custom_param_list):
        request = params.get("__req__") if isinstance(params, dict) else None
        if request is None:
            raise ConfigError(
                f"request {index} carries no token history. SGLang attaches it to "
                "custom_params as '__req__', and only when custom_params is already a "
                f"dict. Build the request with {_SETUP}, which always supplies one."
            )
        histories.append([*request.origin_input_ids, *request.output_ids])
    return histories
