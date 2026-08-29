"""The transformers adapter.

transformers is the one target backend that appends caller-supplied logits processors
*after* its sampling warpers. A green token already masked to -inf cannot be rescued by
delta, so the watermark would go weak and erratic at low top-p. The adapter's whole job is
to put the watermark at index 0, and the ordering tests below are the reason it exists.

Models here are built from a config, never downloaded: two layers, 256 tokens, CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from conftest import HFStyleTokenizer, make_pieces
from llmwatermark.config import WatermarkConfig
from llmwatermark.detector import DEFAULT_Z_THRESHOLD, WatermarkDetector
from llmwatermark.errors import ConfigError
from llmwatermark.greenlist import is_green
from llmwatermark.seeding import SeedTable
from llmwatermark.vocab import fingerprint_from_tokenizer

pytestmark = pytest.mark.requires_transformers

VOCAB_SIZE = 256
KEY = b"transformers-adapter-key"
PROMPT = [3, 9, 27, 81]


@pytest.fixture(scope="module")
def tokenizer() -> HFStyleTokenizer:
    return HFStyleTokenizer(make_pieces(VOCAB_SIZE))


@pytest.fixture(scope="module")
def config(tokenizer: HFStyleTokenizer) -> WatermarkConfig:
    return WatermarkConfig(
        secret_key=KEY,
        vocab_size=VOCAB_SIZE,
        vocab_fingerprint=fingerprint_from_tokenizer(tokenizer, VOCAB_SIZE),
        delta=6.0,
    )


@pytest.fixture
def model() -> object:
    """A tiny randomly initialized model. Constructed locally, never downloaded."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    architecture = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=512,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
    )
    return GPT2LMHeadModel(architecture).eval()


def prompt_tensor(batch: int = 1) -> object:
    import torch

    return torch.tensor([PROMPT] * batch)


def generate(model: object, *, seed: int = 0, **kwargs: object) -> list[int]:
    import torch

    torch.manual_seed(seed)
    defaults: dict[str, object] = {"max_new_tokens": 220, "do_sample": True, "top_k": 0}
    defaults.update(kwargs)
    output = model.generate(prompt_tensor(), **defaults)  # type: ignore[attr-defined]
    return output[0].tolist()


def raw_green_fraction(config: WatermarkConfig, ids: list[int]) -> float:
    """Green rate over every full-context position, ignoring the detector's dedup.

    An untrained test model loops, so detection rightly refuses to score its output. For
    comparing the relative strength of two generations, count every position instead.
    """
    array = np.asarray(ids, dtype=np.int64)
    window = config.h
    contexts = np.ascontiguousarray(sliding_window_view(array, window)[:-1])
    seeds = SeedTable.for_config(config).seeds(contexts, config.scheme)
    green = is_green(seeds, array[window:], config.green_divisor, config.mix_width)
    return float(green.mean())


class TestInstallation:
    def test_watermark_returns_the_same_model(self, model: object, config: WatermarkConfig) -> None:
        from llmwatermark.adapters.transformers import watermark

        assert watermark(model, config) is model

    def test_the_processor_is_installed_at_index_zero(
        self, model: object, config: WatermarkConfig
    ) -> None:
        """Anywhere else and the sampling warpers have already masked green tokens."""
        from llmwatermark.adapters.transformers import WatermarkLogitsProcessor, watermark

        watermark(model, config)
        prepared = model._get_logits_processor(  # type: ignore[attr-defined]
            model.generation_config,  # type: ignore[attr-defined]
            input_ids_seq_length=len(PROMPT),
            encoder_input_ids=prompt_tensor(),
            prefix_allowed_tokens_fn=None,
            logits_processor=[],
            device="cpu",
        )
        assert isinstance(prepared[0], WatermarkLogitsProcessor)

    def test_installing_twice_does_not_stack(self, model: object, config: WatermarkConfig) -> None:
        from llmwatermark.adapters.transformers import WatermarkLogitsProcessor, watermark

        watermark(model, config)
        watermark(model, config)
        generated = generate(model, max_new_tokens=5)
        from llmwatermark.adapters.transformers import installed_processor

        assert isinstance(installed_processor(model), WatermarkLogitsProcessor)
        prepared = model._get_logits_processor(  # type: ignore[attr-defined]
            model.generation_config,  # type: ignore[attr-defined]
            input_ids_seq_length=len(PROMPT),
            encoder_input_ids=prompt_tensor(),
            prefix_allowed_tokens_fn=None,
            logits_processor=[],
            device="cpu",
        )
        watermarks = [p for p in prepared if isinstance(p, WatermarkLogitsProcessor)]
        assert len(watermarks) == 1
        assert len(generated) == len(PROMPT) + 5

    def test_unwatermark_restores_the_original_behaviour(
        self, model: object, config: WatermarkConfig
    ) -> None:
        from llmwatermark.adapters.transformers import unwatermark, watermark

        baseline = generate(model)
        watermark(model, config)
        assert generate(model) != baseline
        unwatermark(model)
        assert generate(model) == baseline

    def test_unwatermark_is_safe_on_a_clean_model(self, model: object) -> None:
        from llmwatermark.adapters.transformers import unwatermark

        unwatermark(model)

    def test_config_for_model_reads_the_model_vocabulary(
        self, model: object, tokenizer: HFStyleTokenizer
    ) -> None:
        from llmwatermark.adapters.transformers import config_for_model

        config = config_for_model(model, tokenizer, secret_key=KEY)
        assert config.vocab_size == VOCAB_SIZE
        config.verify_tokenizer(tokenizer)

    def test_a_vocabulary_mismatch_names_both_sizes(
        self, model: object, tokenizer: HFStyleTokenizer
    ) -> None:
        from llmwatermark.adapters.transformers import watermark

        wrong = WatermarkConfig(
            secret_key=KEY,
            vocab_size=VOCAB_SIZE // 2,
            vocab_fingerprint=fingerprint_from_tokenizer(tokenizer, VOCAB_SIZE // 2),
        )
        watermark(model, wrong)
        with pytest.raises(ConfigError) as excinfo:
            generate(model, max_new_tokens=2)
        assert str(VOCAB_SIZE) in str(excinfo.value)
        assert str(VOCAB_SIZE // 2) in str(excinfo.value)


class TestWarperOrdering:
    """The reason this adapter exists, pinned empirically rather than by inspection."""

    def test_delta_still_bites_under_top_k_one(
        self, model: object, config: WatermarkConfig
    ) -> None:
        """The decisive test.

        With top_k=1 the warper masks every alternative to -inf. If delta were applied
        after it, nothing could be rescued and the output would be bit-identical to the
        unwatermarked run. A difference proves delta reached the raw logits.
        """
        from llmwatermark.adapters.transformers import watermark

        baseline = generate(model, top_k=1, max_new_tokens=60)
        watermark(model, config)
        assert generate(model, top_k=1, max_new_tokens=60) != baseline

    def test_the_watermark_runs_ahead_of_every_other_processor(
        self, model: object, config: WatermarkConfig
    ) -> None:
        """Index 0 is ahead of the default processors too, not just the warpers.

        Measured on transformers 4.57 and 5.16, a processor passed to generate() lands
        after the default processors and before the warpers. Index 0 does not depend on
        that placement holding in any particular release.
        """
        from llmwatermark.adapters.transformers import WatermarkLogitsProcessor, watermark

        generation_config = model.generation_config  # type: ignore[attr-defined]
        generation_config.repetition_penalty = 1.2
        generation_config.no_repeat_ngram_size = 3
        generation_config.do_sample = True
        generation_config.top_p = 0.9

        watermark(model, config)
        prepared = model._get_logits_processor(  # type: ignore[attr-defined]
            generation_config,
            input_ids_seq_length=len(PROMPT),
            encoder_input_ids=prompt_tensor(),
            prefix_allowed_tokens_fn=None,
            logits_processor=[WatermarkLogitsProcessor(config, compile=False)],
            device="cpu",
        )
        placements = [
            index
            for index, processor in enumerate(prepared)
            if isinstance(processor, WatermarkLogitsProcessor)
        ]
        assert placements[0] == 0
        # The naively passed one lands somewhere later, which is what the adapter removes
        # any dependence on.
        assert placements[-1] > 0
        assert isinstance(prepared[0], WatermarkLogitsProcessor)

    def test_suppressed_tokens_stay_suppressed(
        self, model: object, config: WatermarkConfig
    ) -> None:
        """Processors that mask to -inf run after us and must win, green or not."""
        from llmwatermark.adapters.transformers import watermark

        watermark(model, config)
        banned = list(range(2, 200))
        generated = generate(model, max_new_tokens=40, suppress_tokens=banned)
        assert not set(generated[len(PROMPT) :]) & set(banned)

    def test_the_watermark_survives_a_repetition_penalty(
        self, model: object, config: WatermarkConfig, tokenizer: HFStyleTokenizer
    ) -> None:
        """Repetition penalty branches on the sign of a logit, which delta can flip."""
        from llmwatermark.adapters.transformers import watermark

        watermark(model, config)
        generated = generate(model, repetition_penalty=1.2)
        result = WatermarkDetector(config, tokenizer).detect(generated[len(PROMPT) :])
        assert result.is_watermarked

    def test_watermark_strength_falls_as_temperature_rises(
        self, model: object, tokenizer: HFStyleTokenizer
    ) -> None:
        """delta lands on raw logits and temperature divides them, so the sampler sees
        delta/temperature. A modest delta is used here so the effect is not saturated."""
        from llmwatermark.adapters.transformers import watermark

        gentle = WatermarkConfig(
            secret_key=KEY,
            vocab_size=VOCAB_SIZE,
            vocab_fingerprint=fingerprint_from_tokenizer(tokenizer, VOCAB_SIZE),
            delta=1.5,
        )
        watermark(model, gentle)
        rates = [
            raw_green_fraction(gentle, generate(model, temperature=temperature)[len(PROMPT) :])
            for temperature in (0.4, 1.0, 4.0)
        ]
        assert rates[0] > rates[1] > rates[2]


class TestEndToEnd:
    def test_generated_text_detects(
        self, model: object, config: WatermarkConfig, tokenizer: HFStyleTokenizer
    ) -> None:
        from llmwatermark.adapters.transformers import watermark

        watermark(model, config)
        generated = generate(model)[len(PROMPT) :]
        result = WatermarkDetector(config, tokenizer).detect(generated)
        assert result.is_watermarked
        assert result.z_score > DEFAULT_Z_THRESHOLD

    def test_the_same_text_does_not_detect_under_another_key(
        self, model: object, config: WatermarkConfig, tokenizer: HFStyleTokenizer
    ) -> None:
        from llmwatermark.adapters.transformers import watermark

        watermark(model, config)
        generated = generate(model)[len(PROMPT) :]
        other = WatermarkConfig(
            secret_key=b"a-different-key",
            vocab_size=VOCAB_SIZE,
            vocab_fingerprint=config.vocab_fingerprint,
        )
        assert not WatermarkDetector(other, tokenizer).detect(generated).is_watermarked

    def test_unwatermarked_generation_does_not_detect(
        self, model: object, config: WatermarkConfig, tokenizer: HFStyleTokenizer
    ) -> None:
        generated = generate(model)[len(PROMPT) :]
        result = WatermarkDetector(config, tokenizer).detect(generated)
        assert not result.is_watermarked

    def test_greedy_decoding_is_watermarked(
        self, model: object, config: WatermarkConfig, tokenizer: HFStyleTokenizer
    ) -> None:
        from llmwatermark.adapters.transformers import watermark

        watermark(model, config)
        generated = generate(model, do_sample=False)[len(PROMPT) :]
        assert WatermarkDetector(config, tokenizer).detect(generated).is_watermarked

    def test_every_row_of_a_batch_detects_independently(
        self, model: object, config: WatermarkConfig, tokenizer: HFStyleTokenizer
    ) -> None:
        import torch

        from llmwatermark.adapters.transformers import watermark

        watermark(model, config)
        torch.manual_seed(3)
        output = model.generate(  # type: ignore[attr-defined]
            prompt_tensor(4), max_new_tokens=220, do_sample=True, top_k=0
        )
        detector = WatermarkDetector(config, tokenizer)
        for row in output.tolist():
            assert detector.detect(row[len(PROMPT) :]).is_watermarked

    def test_beam_search_output_is_watermarked(
        self, model: object, config: WatermarkConfig, tokenizer: HFStyleTokenizer
    ) -> None:
        """Beam search reorders rows between steps; a stateless processor survives it."""
        import torch

        from llmwatermark.adapters.transformers import watermark

        watermark(model, config)
        torch.manual_seed(4)
        output = model.generate(  # type: ignore[attr-defined]
            prompt_tensor(), max_new_tokens=120, num_beams=3, do_sample=False
        )
        # Beam search on an untrained model loops, so score the raw green rate rather than
        # asking the detector, which rightly refuses such repetitive text.
        assert raw_green_fraction(config, output[0].tolist()[len(PROMPT) :]) > 0.5

    def test_a_prompt_shorter_than_the_context_window_still_works(
        self, model: object, tokenizer: HFStyleTokenizer
    ) -> None:
        """MinHash needs h tokens of history; the first positions simply get no bias."""
        import torch

        from llmwatermark.adapters.transformers import watermark

        config = WatermarkConfig(
            secret_key=KEY,
            vocab_size=VOCAB_SIZE,
            vocab_fingerprint=fingerprint_from_tokenizer(tokenizer, VOCAB_SIZE),
            scheme="minhash",
            context_width=4,
            delta=6.0,
        )
        watermark(model, config)
        torch.manual_seed(5)
        output = model.generate(  # type: ignore[attr-defined]
            torch.tensor([[7]]), max_new_tokens=220, do_sample=True, top_k=0
        )
        assert WatermarkDetector(config, tokenizer).detect(output[0].tolist()).is_watermarked


class TestMissingDependency:
    def test_the_error_names_the_right_extra(self) -> None:
        from llmwatermark.adapters.transformers import _require_transformers

        with pytest.raises(ImportError, match=r"llmwatermark\[transformers\]") as excinfo:
            _require_transformers(ModuleNotFoundError("No module named 'transformers'"))
        assert "transformers" in str(excinfo.value)


class TestQualityEffects:
    def test_the_watermark_changes_the_generated_text(
        self, model: object, config: WatermarkConfig
    ) -> None:
        """Stated plainly: watermarking is not output-preserving."""
        from llmwatermark.adapters.transformers import watermark

        baseline = generate(model)
        watermark(model, config)
        assert generate(model) != baseline

    def test_generation_stays_reproducible_under_a_seed(
        self, model: object, config: WatermarkConfig
    ) -> None:
        """The processor consumes no randomness, so seeded generation is still stable."""
        from llmwatermark.adapters.transformers import watermark

        watermark(model, config)
        assert generate(model, seed=7) == generate(model, seed=7)

    def test_green_rate_rises_with_delta(self, model: object, tokenizer: HFStyleTokenizer) -> None:
        from llmwatermark.adapters.transformers import unwatermark, watermark

        rates = []
        for delta in (0.0, 3.0, 8.0):
            config = WatermarkConfig(
                secret_key=KEY,
                vocab_size=VOCAB_SIZE,
                vocab_fingerprint=fingerprint_from_tokenizer(tokenizer, VOCAB_SIZE),
                delta=delta,
            )
            unwatermark(model)
            watermark(model, config)
            generated = generate(model)[len(PROMPT) :]
            rates.append(raw_green_fraction(config, generated))
        assert rates[0] < rates[1] < rates[2]
        assert np.isclose(rates[0], 0.25, atol=0.12)
