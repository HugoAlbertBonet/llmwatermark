"""SGLang's per-request bookkeeping, tested without SGLang.

The two things worth guarding here fail *silently* if they go wrong: a row whose history is
missing, and a batch whose rows do not correspond one-to-one with its requests. Neither
raises on its own - both would simply bias the wrong tokens and produce text the detector
reads as unmarked. So both are checked explicitly, on the CPU, in the default suite.
"""

from __future__ import annotations

from typing import Any

import pytest

from llmwatermark.adapters.sglang_requests import check_row_alignment, histories_from
from llmwatermark.errors import ConfigError


class FakeRequest:
    """Enough of SGLang's ``Req`` to read a token history from."""

    def __init__(self, prompt: list[int], output: list[int]) -> None:
        self.origin_input_ids = prompt
        self.output_ids = output


def params_for(prompt: list[int], output: list[int], **extra: Any) -> dict[str, Any]:
    return {"__req__": FakeRequest(prompt, output), **extra}


class TestHistories:
    def test_prompt_and_output_are_concatenated_in_order(self) -> None:
        histories = histories_from([params_for([1, 2, 3], [4, 5])])
        assert histories == [[1, 2, 3, 4, 5]]

    def test_a_request_that_has_generated_nothing_yet_is_just_its_prompt(self) -> None:
        assert histories_from([params_for([7, 8], [])]) == [[7, 8]]

    def test_batch_order_is_preserved(self) -> None:
        """Row i of the logits is row i of custom_param_list; reordering would misalign."""
        batch = [params_for([1], [10]), params_for([2], [20]), params_for([3], [30])]
        assert histories_from(batch) == [[1, 10], [2, 20], [3, 30]]

    def test_our_payload_riding_alongside_does_not_disturb_it(self) -> None:
        batch = [params_for([1], [2], llmwatermark={"gamma": 0.25})]
        assert histories_from(batch) == [[1, 2]]

    def test_a_missing_request_names_the_row_and_the_fix(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            histories_from([params_for([1], [2]), {"llmwatermark": {}}])
        message = str(excinfo.value)
        assert "request 1" in message
        assert "__req__" in message
        assert "watermark_sampling_params" in message

    def test_params_that_are_not_a_dict_are_refused_rather_than_skipped(self) -> None:
        """SGLang only injects __req__ when custom_params is already a dict."""
        with pytest.raises(ConfigError):
            histories_from([None])


class TestRowAlignment:
    def test_matching_counts_pass(self) -> None:
        check_row_alignment(4, 4)

    def test_an_empty_batch_passes(self) -> None:
        check_row_alignment(0, 0)

    @pytest.mark.parametrize(("rows", "requests"), [(8, 4), (3, 1), (2, 4)])
    def test_a_mismatch_is_refused(self, rows: int, requests: int) -> None:
        with pytest.raises(ConfigError):
            check_row_alignment(rows, requests)

    def test_the_message_names_both_counts_and_the_cause(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            check_row_alignment(8, 4)
        message = str(excinfo.value)
        assert "8" in message and "4" in message
        assert "speculative decoding" in message
        # The point is that this is unknowable, not merely unimplemented.
        assert "cannot be computed" in message
