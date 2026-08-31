"""Unsloth, which needs no adapter of its own - and the one reason that is not obvious.

Unsloth is a training and inference wrapper around transformers, not a separate runtime.
``FastLanguageModel.from_pretrained`` returns a patched ``AutoModelForCausalLM`` whose
``generate`` is replaced by ``unsloth_fast_generate``, and that wrapper delegates to the
model's original ``generate``. So HuggingFace's sampling loop still runs, still calls
``_get_logits_processor``, and the transformers adapter already covers it.

What makes that worth a test file rather than a sentence in the README is *how narrowly*
it holds. Unsloth reinstalls its own ``generate`` on every call, guarded only by::

    if model.generate.__name__ != "unsloth_fast_generate":
        model._old_generate = model.generate
        model.generate = types.MethodType(unsloth_fast_generate, model)

An adapter that wrapped ``model.generate`` would be silently replaced the next time the
user generated - and worse, captured as ``_old_generate``, so the watermark would appear
to work once and then vanish with no error and no exception. The transformers adapter
hooks ``_get_logits_processor`` instead, which HuggingFace calls from inside the sampling
loop, below anything Unsloth touches. That is the difference between a working integration
and one that silently emits unmarked text, so it is tested rather than assumed.

:class:`TestSurvivesGenerateRepatching` encodes that with no Unsloth and no GPU, because it
is the property that must not regress. The rest needs the real thing.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

from llmwatermark.config import WatermarkConfig

MODEL = "unsloth/Qwen2.5-0.5B-Instruct"
KEY = b"unsloth-test-key-0123456789abcd"
PROMPT = "Explain in detail how a modern operating system schedules processes:"


@pytest.mark.requires_transformers
class TestSurvivesGenerateRepatching:
    """The failure Unsloth would cause to an adapter hooked on the wrong method.

    This uses a stand-in that reproduces Unsloth's patch exactly, so it needs no GPU and no
    Unsloth - only transformers, which the adapter imports.
    """

    @pytest.fixture
    def model(self) -> Any:
        import transformers

        class Stub:
            """Enough of a HF model for the adapter to attach to."""

            def __init__(self) -> None:
                self.config = types.SimpleNamespace(vocab_size=64)

            def generate(self) -> str:
                return "original"

            def _get_logits_processor(self, *args: Any, **kwargs: Any) -> Any:
                return transformers.LogitsProcessorList()

        return Stub()

    @staticmethod
    def repatch(model: Any) -> None:
        """Unsloth's own guard, transcribed from unsloth/models/llama.py."""

        def unsloth_fast_generate(self: Any) -> str:
            return "unsloth"

        if model.generate.__name__ != "unsloth_fast_generate":
            model._old_generate = model.generate
            model.generate = types.MethodType(unsloth_fast_generate, model)

    def test_the_processor_is_still_installed_after_repatching(self, model: Any) -> None:
        from llmwatermark.adapters.transformers import installed_processor, watermark

        config = WatermarkConfig(secret_key=KEY, vocab_size=64, vocab_fingerprint="0" * 64)
        watermark(model, config)
        for _ in range(3):
            self.repatch(model)
            assert installed_processor(model) is not None
            assert len(model._get_logits_processor()) == 1

    def test_the_adapter_does_not_touch_generate(self, model: Any) -> None:
        """The whole point: hooking generate is what Unsloth would clobber."""
        from llmwatermark.adapters.transformers import watermark

        before = model.generate.__func__
        config = WatermarkConfig(secret_key=KEY, vocab_size=64, vocab_fingerprint="0" * 64)
        watermark(model, config)
        assert model.generate.__func__ is before, "the adapter must not wrap generate"
        assert "generate" not in vars(model), "generate must stay on the class"


@pytest.fixture(scope="module")
def loaded() -> Any:
    """The model, loaded once - it is a download and a GPU allocation."""
    # Unsloth patches transformers on import and insists on going first.
    unsloth = pytest.importorskip("unsloth")

    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        MODEL, max_seq_length=1024, load_in_4bit=False, dtype=None
    )
    unsloth.FastLanguageModel.for_inference(model)
    return model, tokenizer


@pytest.fixture(scope="module")
def config(loaded: Any) -> WatermarkConfig:
    from llmwatermark.adapters.transformers import config_for_model

    model, tokenizer = loaded
    return config_for_model(model, tokenizer, secret_key=KEY, delta=2.0)


@pytest.mark.requires_unsloth
class TestUnslothEndToEnd:
    """The real thing: load a model through Unsloth and read the watermark back out."""

    @staticmethod
    def generate(loaded: Any, seed: int) -> str:
        import torch

        model, tokenizer = loaded
        torch.manual_seed(seed)
        ids = tokenizer(PROMPT, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        return str(tokenizer.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True))

    def test_the_config_takes_the_padded_vocabulary_from_the_model(
        self, loaded: Any, config: WatermarkConfig
    ) -> None:
        """Unsloth resizes embeddings when a chat template adds tokens.

        Qwen2.5 ships this way already: 151936 embedding rows against 151665 tokenizer
        entries. Partitioning the tokenizer's count would produce a different greenlist.
        """
        model, tokenizer = loaded
        assert config.vocab_size == int(model.config.vocab_size)
        assert config.vocab_size == model.get_output_embeddings().out_features
        assert config.vocab_size >= len(tokenizer)

    def test_unwatermarked_text_is_not_detected(self, loaded: Any, config: WatermarkConfig) -> None:
        from llmwatermark.adapters.transformers import unwatermark
        from llmwatermark.detector import WatermarkDetector

        model, tokenizer = loaded
        unwatermark(model)
        result = WatermarkDetector(config, tokenizer).detect(self.generate(loaded, 0))
        assert not result.is_watermarked

    def test_watermarked_text_is_detected(self, loaded: Any, config: WatermarkConfig) -> None:
        from llmwatermark.adapters.transformers import unwatermark, watermark
        from llmwatermark.detector import WatermarkDetector

        model, tokenizer = loaded
        watermark(model, config)
        try:
            result = WatermarkDetector(config, tokenizer).detect(self.generate(loaded, 1))
        finally:
            unwatermark(model)
        assert result.is_watermarked
        assert result.z_score > result.threshold
        assert result.green_fraction > 2 * config.effective_gamma

    def test_it_survives_unsloths_repatching_across_calls(
        self, loaded: Any, config: WatermarkConfig
    ) -> None:
        """Unsloth reinstalls its generate wrapper on every call. Two calls, both marked."""
        from llmwatermark.adapters.transformers import installed_processor, unwatermark, watermark
        from llmwatermark.detector import WatermarkDetector

        model, tokenizer = loaded
        detector = WatermarkDetector(config, tokenizer)
        watermark(model, config)
        try:
            results = [detector.detect(self.generate(loaded, seed)) for seed in (2, 3)]
            assert installed_processor(model) is not None
        finally:
            unwatermark(model)
        assert all(result.is_watermarked for result in results), [
            result.z_score for result in results
        ]

    def test_removing_it_restores_the_model(self, loaded: Any, config: WatermarkConfig) -> None:
        from llmwatermark.adapters.transformers import installed_processor, unwatermark, watermark
        from llmwatermark.detector import WatermarkDetector

        model, tokenizer = loaded
        watermark(model, config)
        unwatermark(model)
        assert installed_processor(model) is None
        result = WatermarkDetector(config, tokenizer).detect(self.generate(loaded, 4))
        assert not result.is_watermarked
        assert np.isfinite(result.z_score)
