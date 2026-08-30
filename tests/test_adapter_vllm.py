"""The vLLM adapter, end to end on a real engine.

These need a GPU and a served model, so they are marked and skipped by default. The
bookkeeping they depend on is tested separately and on CPU in test_vllm_tracker.py; what
is proved here is that the adapter wires that bookkeeping into vLLM correctly.

Two environment variables are set below. Both are inert off WSL2 and neither changes what
is being tested: pinned memory is opt-in on WSL2, and vLLM's FlashInfer sampler JIT-builds
a CUDA kernel that needs nvcc on PATH.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

from llmwatermark.config import WatermarkConfig
from llmwatermark.detector import DEFAULT_Z_THRESHOLD, WatermarkDetector
from llmwatermark.errors import ConfigError

pytestmark = pytest.mark.requires_vllm

MODEL = "facebook/opt-125m"
KEY = b"vllm-adapter-test-key"
PROMPTS = ["The history of France began", "In a distant galaxy there was"]


@pytest.fixture(scope="module")
def tokenizer() -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def vocab_size() -> int:
    """The size the model generates over, which exceeds len(tokenizer) for OPT."""
    from transformers import AutoConfig

    return int(AutoConfig.from_pretrained(MODEL).vocab_size)


@pytest.fixture(scope="module")
def config(tokenizer: object, vocab_size: int) -> WatermarkConfig:
    return WatermarkConfig.from_tokenizer(
        tokenizer, vocab_size=vocab_size, secret_key=KEY, delta=6.0
    )


@pytest.fixture(scope="module")
def llm(config: WatermarkConfig) -> object:
    """One watermarked engine for the whole module: startup costs tens of seconds."""
    from vllm import LLM

    from llmwatermark.adapters.vllm import watermark_llm_kwargs

    return LLM(
        model=MODEL,
        gpu_memory_utilization=0.45,
        max_model_len=512,
        enforce_eager=True,
        disable_log_stats=True,
        **watermark_llm_kwargs(config, compile=False),
    )


def generate(llm: object, prompts: list[str], **overrides: object) -> list[list[int]]:
    from vllm import SamplingParams

    params = {
        "max_tokens": 220,
        "min_tokens": 220,
        "ignore_eos": True,
        "temperature": 0.9,
        "seed": 0,
    }
    params.update(overrides)
    outputs = llm.generate(prompts, SamplingParams(**params))  # type: ignore[attr-defined]
    return [list(output.outputs[0].token_ids) for output in outputs]


class TestVocabulary:
    def test_the_model_generates_over_more_ids_than_the_tokenizer_has(
        self, tokenizer: object, vocab_size: int
    ) -> None:
        """The padded-vocabulary trap, on a real model: OPT-125m is 50272 vs 50265."""
        assert vocab_size > len(tokenizer)  # type: ignore[arg-type]

    def test_config_for_llm_takes_the_size_from_the_model(
        self, llm: object, config: WatermarkConfig
    ) -> None:
        from llmwatermark.adapters.vllm import config_for_llm

        built = config_for_llm(llm, secret_key=KEY, delta=6.0)
        assert built.vocab_size == config.vocab_size
        assert built.vocab_fingerprint == config.vocab_fingerprint

    def test_a_vocabulary_mismatch_is_refused(self, config: WatermarkConfig) -> None:
        from llmwatermark.adapters.vllm import _check_vocabulary

        class FakeModelConfig:
            @staticmethod
            def get_vocab_size() -> int:
                return config.vocab_size + 64

        class FakeVllmConfig:
            model_config = FakeModelConfig()

        with pytest.raises(ConfigError) as excinfo:
            _check_vocabulary(FakeVllmConfig(), config)
        assert str(config.vocab_size) in str(excinfo.value)
        assert str(config.vocab_size + 64) in str(excinfo.value)


class TestSamplerIntegration:
    def test_the_processor_is_not_argmax_invariant(self) -> None:
        """True would move the bias after temperature and skip greedy requests entirely."""
        from llmwatermark.adapters.vllm import WatermarkLogitsProcessor

        assert WatermarkLogitsProcessor.is_argmax_invariant(object()) is False  # type: ignore[arg-type]

    def test_generated_text_detects(
        self, llm: object, config: WatermarkConfig, tokenizer: object
    ) -> None:
        detector = WatermarkDetector(config, tokenizer)
        for ids in generate(llm, PROMPTS[:1]):
            result = detector.detect(ids)
            assert result.is_watermarked
            assert result.z_score > DEFAULT_Z_THRESHOLD

    def test_every_row_of_a_batch_detects(
        self, llm: object, config: WatermarkConfig, tokenizer: object
    ) -> None:
        """Continuous batching gives each request its own row; each must be watermarked."""
        detector = WatermarkDetector(config, tokenizer)
        for ids in generate(llm, PROMPTS * 3):
            assert detector.detect(ids, min_tokens=8).is_watermarked

    def test_greedy_requests_are_watermarked(
        self, llm: object, config: WatermarkConfig, tokenizer: object
    ) -> None:
        """vLLM samples greedy requests before temperature - and before argmax-invariant
        processors. Only a non-argmax-invariant processor reaches them."""
        detector = WatermarkDetector(config, tokenizer)
        ids = generate(llm, PROMPTS[:1], temperature=0.0)[0]
        assert detector.detect(ids, min_tokens=8).is_watermarked

    def test_another_key_does_not_detect(
        self, llm: object, config: WatermarkConfig, tokenizer: object
    ) -> None:
        ids = generate(llm, PROMPTS[:1])[0]
        other = WatermarkConfig(
            secret_key=b"a-different-key",
            vocab_size=config.vocab_size,
            vocab_fingerprint=config.vocab_fingerprint,
        )
        assert not WatermarkDetector(other, tokenizer).detect(ids).is_watermarked

    def test_detection_works_from_text_alone(
        self, llm: object, config: WatermarkConfig, tokenizer: object
    ) -> None:
        """The real use case: no token IDs, just the text someone handed you."""
        ids = generate(llm, PROMPTS[:1])[0]
        text = tokenizer.decode(ids)  # type: ignore[attr-defined]
        assert WatermarkDetector(config, tokenizer).detect(text).is_watermarked


class TestCrossBackendAgreement:
    """The payoff for the fingerprint, the golden vectors and the bit-identical mixer."""

    def test_transformers_output_detects_under_the_same_config(
        self, config: WatermarkConfig, tokenizer: object
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        from llmwatermark.adapters.transformers import watermark

        model = AutoModelForCausalLM.from_pretrained(MODEL).eval()
        watermark(model, config, compile=False)
        torch.manual_seed(0)
        prompt = tokenizer(PROMPTS[0], return_tensors="pt")  # type: ignore[operator]
        generated = model.generate(
            **prompt, max_new_tokens=220, min_new_tokens=220, do_sample=True, top_k=0
        )
        ids = generated[0].tolist()[prompt["input_ids"].shape[1] :]

        result = WatermarkDetector(config, tokenizer).detect(ids, min_tokens=8)
        assert result.is_watermarked, "text watermarked under transformers must detect"

    def test_both_engines_agree_on_the_vocabulary_fingerprint(
        self, llm: object, config: WatermarkConfig, tokenizer: object
    ) -> None:
        """A mismatch here is what silently breaks detection across a deployment change."""
        from llmwatermark.adapters.vllm import config_for_llm

        from_vllm = config_for_llm(llm, secret_key=KEY)
        assert from_vllm.vocab_fingerprint == config.vocab_fingerprint
        assert from_vllm.vocab_size == config.vocab_size
