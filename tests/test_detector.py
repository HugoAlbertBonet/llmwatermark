"""Detection: turning a token sequence into a defensible z-score.

Two failure modes matter more than the rest. Scoring against the requested gamma instead
of the effective one biases every score silently. And skipping the n-gram dedup turns
repetitive human text into confident accusations. Both have dedicated tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import HFStyleTokenizer, make_pieces
from llmwatermark.config import HashScheme, MixWidth, WatermarkConfig
from llmwatermark.detector import (
    DEFAULT_Z_THRESHOLD,
    MIN_SCORED_TOKENS,
    WatermarkDetector,
    detect,
)
from llmwatermark.errors import DetectionError, VocabMismatchError
from llmwatermark.greenlist import green_mask, token_id_range
from llmwatermark.seeding import SeedTable
from llmwatermark.vocab import fingerprint_from_tokenizer

VOCAB_SIZE = 8192
KEY = b"detector-test-key"


@pytest.fixture(scope="module")
def pieces() -> list[str]:
    return make_pieces(VOCAB_SIZE)


@pytest.fixture(scope="module")
def tokenizer(pieces: list[str]) -> HFStyleTokenizer:
    return HFStyleTokenizer(pieces)


@pytest.fixture(scope="module")
def config(tokenizer: HFStyleTokenizer) -> WatermarkConfig:
    return WatermarkConfig(
        secret_key=KEY,
        vocab_size=VOCAB_SIZE,
        vocab_fingerprint=fingerprint_from_tokenizer(tokenizer, VOCAB_SIZE),
    )


@pytest.fixture
def detector(config: WatermarkConfig, tokenizer: HFStyleTokenizer) -> WatermarkDetector:
    return WatermarkDetector(config, tokenizer)


def green_ids(config: WatermarkConfig, previous: int) -> np.ndarray:
    """The greenlist for a LeftHash context, straight from the M2/M3 primitives."""
    seeds = SeedTable.for_config(config).seeds([[previous]], config.scheme)
    ids = token_id_range(config.vocab_size, config.mix_width)
    mask = green_mask(seeds, ids, config.green_divisor, config.mix_width)[0]
    return np.flatnonzero(mask)


def watermarked_ids(
    config: WatermarkConfig, length: int, green_rate: float = 1.0, seed: int = 0
) -> list[int]:
    """A sequence whose tokens are green with the given probability."""
    rng = np.random.default_rng(seed)
    sequence = [int(rng.integers(config.vocab_size))]
    for _ in range(length - 1):
        if rng.random() < green_rate:
            choices = green_ids(config, sequence[-1])
            sequence.append(int(rng.choice(choices)))
        else:
            sequence.append(int(rng.integers(config.vocab_size)))
    return sequence


def plain_ids(length: int, seed: int = 0) -> list[int]:
    return np.random.default_rng(seed).integers(0, VOCAB_SIZE, length).tolist()


def naive_z(result: object) -> float:
    """The z-score dedup exists to prevent: every context scored, repeats included."""
    scored = [record for record in result.tokens if record.is_green is not None]  # type: ignore[attr-defined]
    total = len(scored)
    green = sum(1 for record in scored if record.is_green)
    gamma = result.gamma  # type: ignore[attr-defined]
    return (green - gamma * total) / math.sqrt(total * gamma * (1 - gamma))


class TestNullDistribution:
    """Under the null the z-score must be standard normal, or every p-value is a lie."""

    def test_unwatermarked_text_scores_near_zero(self, detector: WatermarkDetector) -> None:
        scores = np.array([detector.detect(plain_ids(400, seed)).z_score for seed in range(300)])
        assert abs(scores.mean()) < 0.2
        assert 0.8 < scores.std() < 1.25

    def test_false_positive_rate_matches_the_threshold(self, detector: WatermarkDetector) -> None:
        scores = np.array([detector.detect(plain_ids(400, seed)).z_score for seed in range(300)])
        assert (scores > DEFAULT_Z_THRESHOLD).sum() == 0
        assert (scores > 1.645).mean() < 0.12  # nominal 5% one-sided, sampling slack

    def test_a_wrong_key_does_not_detect_watermarked_text(
        self, config: WatermarkConfig, tokenizer: HFStyleTokenizer
    ) -> None:
        ids = watermarked_ids(config, 400)
        wrong = WatermarkConfig(
            secret_key=b"not-the-key",
            vocab_size=VOCAB_SIZE,
            vocab_fingerprint=config.vocab_fingerprint,
        )
        assert WatermarkDetector(wrong, tokenizer).detect(ids).z_score < DEFAULT_Z_THRESHOLD


class TestDetectionPower:
    def test_fully_green_text_scores_far_above_the_threshold(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        result = detector.detect(watermarked_ids(config, 400))
        assert result.z_score > 20
        assert result.is_watermarked

    def test_partially_watermarked_text_is_still_detected(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        result = detector.detect(watermarked_ids(config, 400, green_rate=0.5))
        assert result.is_watermarked

    def test_confidence_grows_with_the_square_root_of_length(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        short = detector.detect(watermarked_ids(config, 200, green_rate=0.5, seed=1))
        long = detector.detect(watermarked_ids(config, 800, green_rate=0.5, seed=1))
        ratio = long.z_score / short.z_score
        assert 1.5 < ratio < 2.8  # sqrt(4) = 2, with sampling slack


class TestScoreArithmetic:
    def test_z_matches_the_formula_over_the_reported_counts(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        result = detector.detect(watermarked_ids(config, 300, green_rate=0.6))
        gamma, total = result.gamma, result.scored_count
        expected = (result.green_count - gamma * total) / math.sqrt(total * gamma * (1 - gamma))
        assert result.z_score == pytest.approx(expected)

    def test_p_value_is_the_one_sided_normal_tail(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        result = detector.detect(watermarked_ids(config, 300, green_rate=0.4))
        assert result.p_value == pytest.approx(0.5 * math.erfc(result.z_score / math.sqrt(2)))

    def test_scoring_uses_the_effective_gamma_not_the_requested_one(
        self, tokenizer: HFStyleTokenizer
    ) -> None:
        """gamma=0.3 is really 1/3 under the integer greenlist rule."""
        config = WatermarkConfig(
            secret_key=KEY,
            vocab_size=VOCAB_SIZE,
            vocab_fingerprint=fingerprint_from_tokenizer(tokenizer, VOCAB_SIZE),
            gamma=0.3,
        )
        result = WatermarkDetector(config, tokenizer).detect(plain_ids(600))
        assert result.gamma == pytest.approx(1 / 3)
        assert abs(result.z_score) < 4

    def test_decision_follows_the_threshold(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        ids = watermarked_ids(config, 400, green_rate=0.5)
        assert detector.detect(ids, threshold=0.0).is_watermarked
        assert not detector.detect(ids, threshold=1e9).is_watermarked

    def test_default_threshold_is_the_documented_constant(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        result = detector.detect(watermarked_ids(config, 200))
        assert result.threshold == DEFAULT_Z_THRESHOLD
        assert DEFAULT_Z_THRESHOLD == 4.0


class TestContextDeduplication:
    """A known KGW artifact: repeated context n-grams reuse one greenlist and inflate z."""

    def test_repetition_that_would_false_positive_is_neutralized(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        unique_part = plain_ids(120, seed=3)
        pair = unique_part[-1]
        repeated = green_ids(config, pair)[:1].tolist() * 1
        # One green context, hammered: without dedup this alone drags z over the line.
        sequence = unique_part + [pair, repeated[0]] * 200
        result = detector.detect(sequence)
        assert naive_z(result) > DEFAULT_Z_THRESHOLD
        assert result.z_score < DEFAULT_Z_THRESHOLD

    def test_repeated_contexts_are_recorded_as_skipped(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        sequence = plain_ids(100, seed=4) + [7, 11] * 60
        result = detector.detect(sequence)
        duplicates = [r for r in result.tokens if r.skip_reason == "duplicate_context"]
        assert len(duplicates) > 100
        assert all(record.is_green is not None for record in duplicates)

    def test_a_sequence_without_repeats_is_scored_in_full(
        self, detector: WatermarkDetector
    ) -> None:
        """Distinct preceding tokens mean distinct contexts, so nothing is dropped."""
        sequence = list(range(1, 401))
        result = detector.detect(sequence)
        assert result.scored_count == len(sequence) - 1

    def test_pure_repetition_leaves_too_little_to_score(self, detector: WatermarkDetector) -> None:
        with pytest.raises(DetectionError, match=str(MIN_SCORED_TOKENS)):
            detector.detect([5, 9] * 300)


class TestRefusals:
    def test_short_input_names_the_floor_and_the_actual_count(
        self, detector: WatermarkDetector
    ) -> None:
        with pytest.raises(DetectionError) as excinfo:
            detector.detect(list(range(5)))
        message = str(excinfo.value)
        assert str(MIN_SCORED_TOKENS) in message
        assert "4" in message

    def test_the_floor_can_be_lowered_deliberately(self, detector: WatermarkDetector) -> None:
        result = detector.detect(list(range(1, 9)), min_tokens=4)
        assert result.scored_count == 7

    def test_empty_input_is_refused(self, detector: WatermarkDetector) -> None:
        with pytest.raises(DetectionError):
            detector.detect([])

    def test_a_mismatched_tokenizer_is_refused_before_scoring(
        self, config: WatermarkConfig
    ) -> None:
        other = HFStyleTokenizer(make_pieces(VOCAB_SIZE, prefix="other"))
        with pytest.raises(VocabMismatchError) as excinfo:
            WatermarkDetector(config, other)
        assert config.vocab_fingerprint in str(excinfo.value)

    def test_token_ids_outside_the_vocabulary_are_refused(
        self, detector: WatermarkDetector
    ) -> None:
        with pytest.raises(Exception, match=str(VOCAB_SIZE)):
            detector.detect([1, 2, VOCAB_SIZE + 5, *range(50)])


class TestTokenRecords:
    def test_there_is_one_record_per_token(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        ids = watermarked_ids(config, 120)
        result = detector.detect(ids)
        assert len(result.tokens) == len(ids)
        assert [record.token_id for record in result.tokens] == ids
        assert [record.position for record in result.tokens] == list(range(len(ids)))

    def test_the_first_h_positions_have_no_greenlist(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        result = detector.detect(watermarked_ids(config, 120))
        head = result.tokens[: config.h]
        assert all(record.is_green is None for record in head)
        assert all(record.skip_reason == "no_context" for record in head)
        assert all(not record.scored for record in head)

    def test_records_reconcile_with_the_aggregate_counts(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        """The visualization layer renders these; they must not disagree with the score."""
        result = detector.detect(watermarked_ids(config, 300, green_rate=0.5))
        scored = [record for record in result.tokens if record.scored]
        assert len(scored) == result.scored_count
        assert sum(1 for record in scored if record.is_green) == result.green_count

    def test_records_carry_the_context_that_seeded_them(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        ids = watermarked_ids(config, 60)
        result = detector.detect(ids)
        for record in result.tokens[config.h :]:
            assert record.context == tuple(ids[record.position - config.h : record.position])

    def test_records_carry_the_token_text(
        self, detector: WatermarkDetector, pieces: list[str]
    ) -> None:
        result = detector.detect(list(range(1, 60)))
        assert [record.piece for record in result.tokens] == pieces[1:60]


class TestInputForms:
    def test_text_and_token_ids_agree(
        self, detector: WatermarkDetector, config: WatermarkConfig, pieces: list[str]
    ) -> None:
        ids = watermarked_ids(config, 200)
        text = " ".join(pieces[index] for index in ids)
        assert detector.detect(text).z_score == pytest.approx(detector.detect(ids).z_score)

    def test_numpy_arrays_are_accepted(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        ids = watermarked_ids(config, 120)
        assert detector.detect(np.array(ids)).z_score == pytest.approx(detector.detect(ids).z_score)

    def test_the_module_level_helper_matches_the_detector(
        self, config: WatermarkConfig, tokenizer: HFStyleTokenizer
    ) -> None:
        ids = watermarked_ids(config, 120)
        direct = detect(ids, tokenizer, config)
        assert direct.z_score == pytest.approx(
            WatermarkDetector(config, tokenizer).detect(ids).z_score
        )

    def test_a_detector_is_reusable(
        self, detector: WatermarkDetector, config: WatermarkConfig
    ) -> None:
        ids = watermarked_ids(config, 150)
        assert detector.detect(ids).z_score == detector.detect(ids).z_score

    @pytest.mark.parametrize("scheme", [HashScheme.LEFTHASH, HashScheme.MINHASH])
    @pytest.mark.parametrize("width", [MixWidth.BITS32, MixWidth.BITS64])
    def test_every_scheme_and_width_detects_its_own_watermark(
        self, tokenizer: HFStyleTokenizer, scheme: HashScheme, width: MixWidth
    ) -> None:
        config = WatermarkConfig(
            secret_key=KEY,
            vocab_size=VOCAB_SIZE,
            vocab_fingerprint=fingerprint_from_tokenizer(tokenizer, VOCAB_SIZE),
            scheme=scheme,
            mix_width=width,
        )
        detector = WatermarkDetector(config, tokenizer)
        rng = np.random.default_rng(7)
        sequence = rng.integers(0, VOCAB_SIZE, config.h).tolist()
        ids = token_id_range(VOCAB_SIZE, width)
        table = SeedTable.for_config(config)
        for _ in range(400):
            seeds = table.seeds([sequence[-config.h :]], scheme)
            mask = green_mask(seeds, ids, config.green_divisor, width)[0]
            sequence.append(int(rng.choice(np.flatnonzero(mask))))
        assert detector.detect(sequence).is_watermarked


class TestEndToEnd:
    def test_text_generated_through_the_processor_detects(
        self, config: WatermarkConfig, tokenizer: HFStyleTokenizer
    ) -> None:
        """Greedy sampling over biased logits, then detection. No model required."""
        from llmwatermark.processor import WatermarkProcessor

        processor = WatermarkProcessor(config, compile=False)
        rng = np.random.default_rng(11)
        sequence = [int(rng.integers(VOCAB_SIZE))]
        for _ in range(300):
            logits = rng.normal(0.0, 1.0, (1, VOCAB_SIZE)).astype(np.float32)
            processor.apply(logits, np.array([[sequence[-1]]], dtype=np.int64))
            sequence.append(int(logits[0].argmax()))

        result = WatermarkDetector(config, tokenizer).detect(sequence)
        assert result.is_watermarked
        assert result.z_score > DEFAULT_Z_THRESHOLD

    def test_the_same_generation_does_not_detect_under_another_key(
        self, config: WatermarkConfig, tokenizer: HFStyleTokenizer
    ) -> None:
        from llmwatermark.processor import WatermarkProcessor

        processor = WatermarkProcessor(config, compile=False)
        rng = np.random.default_rng(11)
        sequence = [int(rng.integers(VOCAB_SIZE))]
        for _ in range(300):
            logits = rng.normal(0.0, 1.0, (1, VOCAB_SIZE)).astype(np.float32)
            processor.apply(logits, np.array([[sequence[-1]]], dtype=np.int64))
            sequence.append(int(logits[0].argmax()))

        other = WatermarkConfig(
            secret_key=b"a-different-key",
            vocab_size=VOCAB_SIZE,
            vocab_fingerprint=config.vocab_fingerprint,
        )
        assert not WatermarkDetector(other, tokenizer).detect(sequence).is_watermarked
