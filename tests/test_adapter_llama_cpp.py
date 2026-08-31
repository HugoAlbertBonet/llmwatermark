"""The llama.cpp adapter, on a real GGUF model.

This backend is the first that is neither PyTorch-shaped nor batched: processors are plain
callables over numpy arrays for a single sequence. It is also the first to exercise the
fingerprint's **bytes** piece path against a real vocabulary rather than a synthetic fake -
llama_token_get_text returns raw bytes where a transformers tokenizer returns str.

Needs the model file; marked and skipped by default.
"""

from __future__ import annotations

import numpy as np
import pytest

from llmwatermark.config import WatermarkConfig
from llmwatermark.detector import DEFAULT_Z_THRESHOLD, WatermarkDetector
from llmwatermark.errors import ConfigError

pytestmark = pytest.mark.requires_llama_cpp

REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
HF_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
KEY = b"llama-cpp-adapter-test-key"
PROMPT = "Explain in a few sentences how a compass works."


@pytest.fixture(scope="module")
def llama() -> object:
    import llama_cpp
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    # Offload only if this build actually has it. The prebuilt CUDA wheels are compiled
    # with AVX-512, which not every CPU has, so a plain CPU build is the portable choice.
    layers = -1 if llama_cpp.llama_supports_gpu_offload() else 0
    return Llama(
        model_path=hf_hub_download(REPO, FILENAME),
        n_ctx=1024,
        n_gpu_layers=layers,
        verbose=False,
        seed=0,
    )


@pytest.fixture(scope="module")
def config(llama: object) -> WatermarkConfig:
    from llmwatermark.adapters.llama_cpp import config_for_llama

    return config_for_llama(llama, secret_key=KEY, delta=4.0)


def generate(llama: object, **overrides: object) -> str:
    parameters = {"max_tokens": 220, "temperature": 0.8, "top_p": 0.95, "seed": 0}
    parameters.update(overrides)
    return llama(PROMPT, **parameters)["choices"][0]["text"]  # type: ignore[operator,index]


class TestVocabulary:
    def test_the_config_reads_the_vocabulary_from_the_model(
        self, llama: object, config: WatermarkConfig
    ) -> None:
        assert config.vocab_size == llama.n_vocab()  # type: ignore[attr-defined]

    def test_pieces_come_back_as_bytes(self, llama: object) -> None:
        """The path the fingerprint has only ever seen from a synthetic tokenizer."""
        from llmwatermark.adapters.llama_cpp import LlamaCppVocabulary

        vocabulary = LlamaCppVocabulary(llama)
        assert isinstance(vocabulary.id_to_piece(100), bytes)

    def test_the_fingerprint_matches_the_transformers_tokenizer(
        self, config: WatermarkConfig
    ) -> None:
        """The open question from M1: do two backends agree on the same vocabulary?

        llama.cpp hands back raw bytes and transformers hands back str. If the underlying
        pieces are the same, both fingerprint identically and a watermark crosses between
        them. If they do not, this is where it surfaces.
        """
        from transformers import AutoTokenizer

        from llmwatermark.vocab import fingerprint_from_tokenizer

        tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
        from_transformers = fingerprint_from_tokenizer(tokenizer, config.vocab_size)
        assert from_transformers == config.vocab_fingerprint

    def test_a_vocabulary_mismatch_is_refused(self, llama: object) -> None:
        from llmwatermark.adapters.llama_cpp import watermark

        wrong = WatermarkConfig(secret_key=KEY, vocab_size=1024, vocab_fingerprint="0" * 64)
        with pytest.raises(ConfigError) as excinfo:
            watermark(llama, wrong)
        assert "1024" in str(excinfo.value)


class TestInstallation:
    def test_watermark_returns_the_same_object(
        self, llama: object, config: WatermarkConfig
    ) -> None:
        from llmwatermark.adapters.llama_cpp import unwatermark, watermark

        try:
            assert watermark(llama, config) is llama
        finally:
            unwatermark(llama)

    def test_installing_twice_does_not_stack(self, llama: object, config: WatermarkConfig) -> None:
        from llmwatermark.adapters.llama_cpp import (
            WatermarkLogitsProcessor,
            installed_processor,
            unwatermark,
            watermark,
        )

        try:
            watermark(llama, config)
            watermark(llama, config)
            assert isinstance(installed_processor(llama), WatermarkLogitsProcessor)
            generate(llama, max_tokens=8)
        finally:
            unwatermark(llama)

    def test_unwatermark_restores_the_original_output(
        self, llama: object, config: WatermarkConfig
    ) -> None:
        from llmwatermark.adapters.llama_cpp import unwatermark, watermark

        baseline = generate(llama, max_tokens=40)
        watermark(llama, config)
        assert generate(llama, max_tokens=40) != baseline
        unwatermark(llama)
        assert generate(llama, max_tokens=40) == baseline

    def test_unwatermark_is_safe_on_a_clean_object(self, llama: object) -> None:
        from llmwatermark.adapters.llama_cpp import unwatermark

        unwatermark(llama)


class TestProcessor:
    def test_delta_lands_only_on_greenlist_tokens(self, config: WatermarkConfig) -> None:
        """The processor is pure numpy, so it can be checked without loading a model."""
        from llmwatermark.adapters.llama_cpp import WatermarkLogitsProcessor
        from llmwatermark.greenlist import green_mask, token_id_range
        from llmwatermark.seeding import SeedTable

        processor = WatermarkLogitsProcessor(config)
        history = np.array([5, 9, 11], dtype=np.intc)
        scores = np.zeros(config.vocab_size, dtype=np.single)
        processor(history, scores)

        seeds = SeedTable.for_config(config).seeds([[11]], config.scheme)
        ids = token_id_range(config.vocab_size, config.mix_width)
        expected = green_mask(seeds, ids, config.green_divisor, config.mix_width)[0]
        assert np.array_equal(scores > 0, expected)

    def test_a_history_shorter_than_the_window_is_left_alone(self) -> None:
        from llmwatermark.adapters.llama_cpp import WatermarkLogitsProcessor
        from llmwatermark.vocab import fingerprint_from_tokenizer

        pieces = [f"tok{index}" for index in range(256)]

        class Fake:
            def __len__(self) -> int:
                return len(pieces)

            def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
                return [pieces[index] for index in ids]

        fake = Fake()
        config = WatermarkConfig(
            secret_key=KEY,
            vocab_size=256,
            vocab_fingerprint=fingerprint_from_tokenizer(fake, 256),
            scheme="minhash",
            context_width=4,
        )
        processor = WatermarkLogitsProcessor(config)
        scores = np.zeros(256, dtype=np.single)
        processor(np.array([1, 2], dtype=np.intc), scores)
        assert not scores.any()


class TestEndToEnd:
    def test_generated_text_detects(self, llama: object, config: WatermarkConfig) -> None:
        from transformers import AutoTokenizer

        from llmwatermark.adapters.llama_cpp import unwatermark, watermark

        try:
            watermark(llama, config)
            text = generate(llama)
        finally:
            unwatermark(llama)

        tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
        result = WatermarkDetector(config, tokenizer).detect(text, min_tokens=8)
        assert result.is_watermarked
        assert result.z_score > DEFAULT_Z_THRESHOLD

    def test_unwatermarked_text_does_not_detect(
        self, llama: object, config: WatermarkConfig
    ) -> None:
        from transformers import AutoTokenizer

        from llmwatermark.adapters.llama_cpp import unwatermark

        unwatermark(llama)
        text = generate(llama)
        tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
        assert not WatermarkDetector(config, tokenizer).detect(text, min_tokens=8).is_watermarked

    def test_another_key_does_not_detect(self, llama: object, config: WatermarkConfig) -> None:
        from transformers import AutoTokenizer

        from llmwatermark.adapters.llama_cpp import unwatermark, watermark

        try:
            watermark(llama, config)
            text = generate(llama)
        finally:
            unwatermark(llama)

        other = WatermarkConfig(
            secret_key=b"a-different-key",
            vocab_size=config.vocab_size,
            vocab_fingerprint=config.vocab_fingerprint,
        )
        tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
        assert not WatermarkDetector(other, tokenizer).detect(text, min_tokens=8).is_watermarked

    def test_the_watermark_survives_a_repetition_penalty(
        self, llama: object, config: WatermarkConfig
    ) -> None:
        """llama.cpp applies its penalties after the custom sampler, so they compose."""
        from transformers import AutoTokenizer

        from llmwatermark.adapters.llama_cpp import unwatermark, watermark

        try:
            watermark(llama, config)
            text = generate(llama, repeat_penalty=1.2)
        finally:
            unwatermark(llama)

        tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
        assert WatermarkDetector(config, tokenizer).detect(text, min_tokens=8).is_watermarked
