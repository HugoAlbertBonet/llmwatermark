"""Row bookkeeping for the vLLM adapter.

vLLM's batched interface hands the processor logits for the whole batch and nothing else.
Which request occupies which row changes every step: requests finish, new ones take their
slots, and preempted ones come back somewhere else entirely. The tracker mirrors that
layout from vLLM's own add / remove / move events.

Nothing derived is cached. Contexts are read from each request's live token list on every
call, so a request that is evicted and resumed still seeds from whatever tokens it
actually has. Getting the bookkeeping wrong corrupts the watermark silently, which is why
it lives in a class with no vLLM import and is tested here on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from llmwatermark.adapters.vllm_tracker import MOVE_SWAP, MOVE_UNIDIRECTIONAL, RequestTracker
from llmwatermark.errors import SeedingError


def added(index: int, prompt: list[int], output: list[int]) -> tuple[int, list[int], list[int]]:
    return (index, prompt, output)


@pytest.fixture
def tracker() -> RequestTracker:
    return RequestTracker()


class TestAddingAndReading:
    def test_a_single_request_reports_its_own_tail(self, tracker: RequestTracker) -> None:
        tracker.apply(batch_size=1, added=[added(0, [1, 2], [7, 8, 9])])
        context, valid = tracker.contexts(2)
        assert context.tolist() == [[8, 9]]
        assert valid.tolist() == [True]

    def test_rows_keep_their_own_histories(self, tracker: RequestTracker) -> None:
        tracker.apply(
            batch_size=2,
            added=[added(0, [1], [10, 11]), added(1, [2], [20, 21])],
        )
        context, _ = tracker.contexts(2)
        assert context.tolist() == [[10, 11], [20, 21]]

    def test_the_output_list_is_read_live(self, tracker: RequestTracker) -> None:
        """vLLM appends to the same list object; the tracker must never snapshot it."""
        output: list[int] = [5]
        tracker.apply(batch_size=1, added=[added(0, [1, 2], output)])
        assert tracker.contexts(1)[0].tolist() == [[5]]
        output.append(6)
        assert tracker.contexts(1)[0].tolist() == [[6]]

    def test_the_prompt_supplies_the_tail_before_enough_output_exists(
        self, tracker: RequestTracker
    ) -> None:
        """The first generated token is seeded by the end of the prompt."""
        tracker.apply(batch_size=1, added=[added(0, [1, 2, 3, 4], [9])])
        assert tracker.contexts(3)[0].tolist() == [[3, 4, 9]]

    def test_a_history_shorter_than_the_window_is_invalid(self, tracker: RequestTracker) -> None:
        tracker.apply(batch_size=2, added=[added(0, [1], []), added(1, [1, 2, 3], [4])])
        context, valid = tracker.contexts(3)
        assert valid.tolist() == [False, True]
        assert context[0].tolist() == [0, 0, 0]

    def test_an_empty_row_is_invalid(self, tracker: RequestTracker) -> None:
        tracker.apply(batch_size=2, added=[added(1, [1, 2, 3], [4])])
        _, valid = tracker.contexts(2)
        assert valid.tolist() == [False, True]

    def test_result_shapes_and_dtypes_are_stable(self, tracker: RequestTracker) -> None:
        tracker.apply(batch_size=3, added=[added(0, [1, 2], [3])])
        context, valid = tracker.contexts(2)
        assert context.shape == (3, 2)
        assert context.dtype == np.int64
        assert valid.dtype == np.bool_

    def test_an_empty_batch_is_allowed(self, tracker: RequestTracker) -> None:
        tracker.apply(batch_size=0)
        context, valid = tracker.contexts(4)
        assert context.shape == (0, 4)
        assert valid.shape == (0,)


class TestRemoval:
    def test_a_removed_row_stops_reporting(self, tracker: RequestTracker) -> None:
        tracker.apply(batch_size=2, added=[added(0, [1], [10, 11]), added(1, [2], [20, 21])])
        tracker.apply(batch_size=2, removed=[0])
        _, valid = tracker.contexts(2)
        assert valid.tolist() == [False, True]

    def test_removing_several_rows_at_once(self, tracker: RequestTracker) -> None:
        tracker.apply(
            batch_size=3,
            added=[added(0, [1], [1, 1]), added(1, [2], [2, 2]), added(2, [3], [3, 3])],
        )
        tracker.apply(batch_size=3, removed=[2, 0])
        _, valid = tracker.contexts(2)
        assert valid.tolist() == [False, True, False]

    def test_a_freed_row_can_be_taken_by_a_new_request(self, tracker: RequestTracker) -> None:
        tracker.apply(batch_size=2, added=[added(0, [1], [10, 11]), added(1, [2], [20, 21])])
        tracker.apply(batch_size=2, removed=[0], added=[added(0, [9], [90, 91])])
        context, valid = tracker.contexts(2)
        assert valid.tolist() == [True, True]
        assert context.tolist() == [[90, 91], [20, 21]]


class TestMoves:
    def test_a_unidirectional_move_vacates_the_source(self, tracker: RequestTracker) -> None:
        tracker.apply(batch_size=2, added=[added(1, [2], [20, 21])])
        tracker.apply(batch_size=2, moved=[(1, 0, MOVE_UNIDIRECTIONAL)])
        context, valid = tracker.contexts(2)
        assert valid.tolist() == [True, False]
        assert context[0].tolist() == [20, 21]

    def test_a_swap_exchanges_two_rows(self, tracker: RequestTracker) -> None:
        tracker.apply(batch_size=2, added=[added(0, [1], [10, 11]), added(1, [2], [20, 21])])
        tracker.apply(batch_size=2, moved=[(0, 1, MOVE_SWAP)])
        context, _ = tracker.contexts(2)
        assert context.tolist() == [[20, 21], [10, 11]]

    def test_a_swap_with_an_empty_row_still_moves_the_occupant(
        self, tracker: RequestTracker
    ) -> None:
        tracker.apply(batch_size=2, added=[added(0, [1], [10, 11])])
        tracker.apply(batch_size=2, moved=[(0, 1, MOVE_SWAP)])
        context, valid = tracker.contexts(2)
        assert valid.tolist() == [False, True]
        assert context[1].tolist() == [10, 11]

    def test_moves_are_applied_in_the_order_given(self, tracker: RequestTracker) -> None:
        tracker.apply(
            batch_size=3,
            added=[added(0, [1], [10, 11]), added(1, [2], [20, 21]), added(2, [3], [30, 31])],
        )
        tracker.apply(
            batch_size=3,
            moved=[(2, 0, MOVE_UNIDIRECTIONAL), (1, 2, MOVE_UNIDIRECTIONAL)],
        )
        context, valid = tracker.contexts(2)
        assert context[0].tolist() == [30, 31]
        assert context[2].tolist() == [20, 21]
        assert valid.tolist() == [True, False, True]

    def test_an_unknown_move_direction_is_rejected(self, tracker: RequestTracker) -> None:
        tracker.apply(batch_size=2, added=[added(0, [1], [10])])
        with pytest.raises(SeedingError, match="direction"):
            tracker.apply(batch_size=2, moved=[(0, 1, "sideways")])


class TestSchedulerChurn:
    """The failure this class exists to prevent has no symptom other than a low z-score."""

    def test_a_preempted_request_resumed_elsewhere_keeps_its_own_context(
        self, tracker: RequestTracker
    ) -> None:
        output = [10, 11]
        tracker.apply(batch_size=2, added=[added(0, [1], output), added(1, [2], [20, 21])])
        tracker.apply(batch_size=2, removed=[0])  # preempted
        output.append(12)  # resumed, having generated more
        tracker.apply(batch_size=2, added=[added(1, [1], output)], removed=[1])

        context, valid = tracker.contexts(2)
        assert valid.tolist() == [False, True]
        assert context[1].tolist() == [11, 12]

    def test_row_indices_carry_no_meaning_across_steps(self, tracker: RequestTracker) -> None:
        """Row 0 is a seat, not a passenger."""
        first = [10, 11]
        second = [20, 21]
        tracker.apply(batch_size=1, added=[added(0, [1], first)])
        assert tracker.contexts(2)[0].tolist() == [[10, 11]]

        tracker.apply(batch_size=1, removed=[0], added=[added(0, [2], second)])
        assert tracker.contexts(2)[0].tolist() == [[20, 21]]

    def test_the_batch_can_grow_and_shrink(self, tracker: RequestTracker) -> None:
        tracker.apply(batch_size=1, added=[added(0, [1], [10, 11])])
        tracker.apply(batch_size=3, added=[added(1, [2], [20, 21]), added(2, [3], [30, 31])])
        assert tracker.contexts(2)[1].tolist() == [True, True, True]

        tracker.apply(batch_size=1, removed=[1, 2])
        context, valid = tracker.contexts(2)
        assert context.shape == (1, 2)
        assert valid.tolist() == [True]

    def test_a_long_churn_matches_a_naive_recomputation(self, tracker: RequestTracker) -> None:
        """Fuzz the scheduler against a dictionary that models the same thing directly."""
        rng = np.random.default_rng(0)
        size = 6
        window = 3
        model: dict[int, list[int]] = {}
        tracker.apply(batch_size=size)

        for step in range(400):
            removed, moved, adds = [], [], []
            for row in range(size):
                choice = rng.random()
                if row in model and choice < 0.12:
                    removed.append(row)
                    del model[row]
                elif row not in model and choice < 0.30:
                    history = rng.integers(1, 900, rng.integers(1, 8)).tolist()
                    adds.append((row, [], history))
                    model[row] = history
                elif row in model:
                    model[row].append(int(rng.integers(1, 900)))

            if len(model) >= 2 and rng.random() < 0.25:
                source, target = rng.choice(sorted(model), 2, replace=False)
                moved.append((int(source), int(target), MOVE_SWAP))
                model[int(source)], model[int(target)] = model[int(target)], model[int(source)]

            tracker.apply(batch_size=size, removed=removed, moved=moved, added=adds)

            context, valid = tracker.contexts(window)
            for row in range(size):
                # No prompt in this fuzz, so the output alone decides validity.
                history = model.get(row)
                expected = history is not None and len(history) >= window
                assert bool(valid[row]) == expected, f"row {row} at step {step}"
                if expected:
                    assert context[row].tolist() == history[-window:], f"row {row}, step {step}"
