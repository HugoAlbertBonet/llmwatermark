"""The SGLang adapter against a real engine.

Needs a GPU and a model download, so everything here is gated behind
``--backend sglang``. The bookkeeping that does not need SGLang is in
``test_adapter_sglang_requests.py`` and runs in the default suite.

The engine is built with Triton attention and PyTorch sampling because SGLang's default
FlashInfer path JIT-compiles CUDA and needs a toolkit that a plain install does not
provide. Those choices are about this machine, not about the watermark - the adapter does
not care which attention kernel runs underneath it.
"""

from __future__ import annotations

from typing import Any

import pytest

from llmwatermark.detector import WatermarkDetector
from llmwatermark.errors import ConfigError

pytestmark = pytest.mark.requires_sglang

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
KEY = b"sglang-adapter-test-key-01234567"
PROMPT = "Explain in detail how a modern operating system schedules processes:"
# 160 tokens leaves roughly 80 after n-gram dedup, where a single sample at delta 2 sits
# near z = 4 and crosses the threshold only most of the time - the same ~95% TPR the
# evaluation reports. These tests assert on one sample, so they use a length where the
# signal is not marginal; this is about test reliability, not about delta 2 being stronger.
NEW_TOKENS = 320


@pytest.fixture(scope="module")
def engine() -> Any:
    """One engine for the module: starting it costs a model load and a GPU allocation."""
    sglang = pytest.importorskip("sglang")

    from llmwatermark.adapters.sglang import watermark_engine_kwargs

    instance = sglang.Engine(
        model_path=MODEL,
        mem_fraction_static=0.55,
        log_level="error",
        attention_backend="triton",
        sampling_backend="pytorch",
        disable_cuda_graph=True,
        **watermark_engine_kwargs(KEY),
    )
    yield instance
    instance.shutdown()


@pytest.fixture(scope="module")
def config(engine: Any) -> Any:
    from llmwatermark.adapters.sglang import config_for_engine

    return config_for_engine(engine, secret_key=KEY, delta=2.0)


@pytest.fixture(scope="module")
def detector(engine: Any, config: Any) -> WatermarkDetector:
    return WatermarkDetector(config, engine.tokenizer_manager.tokenizer)


def generate(engine: Any, config: Any = None, prompts: Any = PROMPT, **overrides: Any) -> Any:
    """Generate with the watermark when a config is given, without it when not."""
    from llmwatermark.adapters.sglang import watermark_sampling_params

    settings = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": NEW_TOKENS, **overrides}
    if config is None:
        outputs = engine.generate(prompts, sampling_params=settings)
    else:
        outputs = engine.generate(prompts, **watermark_sampling_params(config, **settings))
    return outputs if isinstance(outputs, list) else [outputs]


class TestVocabulary:
    def test_the_config_takes_the_size_the_model_generates_over(
        self, engine: Any, config: Any
    ) -> None:
        """A padded embedding makes this larger than len(tokenizer); the model wins."""
        declared = engine.tokenizer_manager.model_config.vocab_size
        assert config.vocab_size == int(declared)

    def test_a_mismatched_config_is_refused(self, engine: Any) -> None:
        from llmwatermark.adapters.sglang import config_for_engine

        with pytest.raises(ConfigError):
            config_for_engine(engine, secret_key=KEY, vocab_size=1024)


class TestSamplerIntegration:
    def test_unwatermarked_text_is_not_detected(
        self, engine: Any, detector: WatermarkDetector
    ) -> None:
        result = detector.detect(generate(engine)[0]["text"])
        assert not result.is_watermarked

    def test_watermarked_text_is_detected(
        self, engine: Any, config: Any, detector: WatermarkDetector
    ) -> None:
        result = detector.detect(generate(engine, config)[0]["text"])
        assert result.is_watermarked
        assert result.z_score > result.threshold
        # Clearly above chance rather than at any particular level: delta 2 lands the green
        # fraction around 0.46-0.59 on 160 tokens, and the z-test above is the real check.
        assert result.green_fraction > 1.5 * config.effective_gamma

    def test_every_row_of_a_batch_is_watermarked(
        self, engine: Any, config: Any, detector: WatermarkDetector
    ) -> None:
        """SGLang hands the processor its own rows; a misalignment would mark only some."""
        outputs = generate(engine, config, prompts=[PROMPT] * 4)
        scores = [detector.detect(output["text"]).z_score for output in outputs]
        assert len(scores) == 4
        # Individual rows vary at this length, so the batch is judged as a batch.
        assert min(scores) > 0.0, scores
        assert sum(scores) / len(scores) > 3.0, scores

    def test_a_wrong_key_does_not_detect(self, engine: Any, config: Any) -> None:
        """The watermark is keyed, so a detector holding another key must see nothing."""
        from llmwatermark.config import WatermarkConfig

        text = generate(engine, config)[0]["text"]
        other = WatermarkConfig.from_tokenizer(
            engine.tokenizer_manager.tokenizer,
            vocab_size=config.vocab_size,
            secret_key=b"a-completely-different-key-98765",
            delta=config.delta,
        )
        wrong = WatermarkDetector(other, engine.tokenizer_manager.tokenizer).detect(text)
        assert not wrong.is_watermarked

    def test_delta_zero_produces_nothing_to_detect(self, engine: Any, config: Any) -> None:
        """A sanity check on the whole path: no bias, no signal, same plumbing."""
        from dataclasses import replace

        unbiased = replace(config, delta=0.0)
        detector = WatermarkDetector(unbiased, engine.tokenizer_manager.tokenizer)
        assert not detector.detect(generate(engine, unbiased)[0]["text"]).is_watermarked


class TestOverlapSchedulerGuard:
    """The engine option that silently destroys detection.

    Under SGLang's overlap scheduler the request history is one token stale, so the
    greenlist is keyed a position behind the detector and the output never detects while
    looking completely normal. The adapter refuses rather than let that happen.
    """

    def test_the_kwargs_disable_overlap(self) -> None:
        from llmwatermark.adapters.sglang import watermark_engine_kwargs

        assert watermark_engine_kwargs(KEY)["disable_overlap_schedule"] is True

    def test_an_overlapped_engine_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sglang.srt import server_args as sglang_server_args

        from llmwatermark.adapters import sglang as adapter

        class Overlapped:
            disable_overlap_schedule = False

        monkeypatch.setattr(
            sglang_server_args, "get_global_server_args", lambda: Overlapped(), raising=False
        )
        with pytest.raises(ConfigError) as excinfo:
            adapter._refuse_overlap_schedule()
        message = str(excinfo.value)
        assert "overlap" in message.lower()
        assert "watermark_engine_kwargs" in message
