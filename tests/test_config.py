"""WatermarkConfig: the object both sides of the watermark must agree on exactly.

Generation and detection are only interoperable if every field here means the same thing
on both machines, so validation is strict and the errors say how to fix the problem.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from conftest import HFStyleTokenizer, make_pieces
from llmwatermark.config import (
    DEFAULT_DELTA,
    DEFAULT_GAMMA,
    HashScheme,
    MixWidth,
    WatermarkConfig,
    generate_secret_key,
)
from llmwatermark.errors import ConfigError, VocabMismatchError
from llmwatermark.vocab import fingerprint_from_tokenizer

VOCAB_SIZE = 512
SECRET_KEY = b"unit-test-key"


@pytest.fixture
def fingerprint(tokenizer: HFStyleTokenizer) -> str:
    return fingerprint_from_tokenizer(tokenizer, VOCAB_SIZE)


@pytest.fixture
def config(fingerprint: str) -> WatermarkConfig:
    return WatermarkConfig(
        secret_key=SECRET_KEY, vocab_size=VOCAB_SIZE, vocab_fingerprint=fingerprint
    )


def build(**overrides: object) -> WatermarkConfig:
    """A valid config with individual fields overridden, for validation tests."""
    fields: dict[str, object] = {
        "secret_key": SECRET_KEY,
        "vocab_size": VOCAB_SIZE,
        "vocab_fingerprint": "0" * 64,
    }
    fields.update(overrides)
    return WatermarkConfig(**fields)  # type: ignore[arg-type]


class TestDefaults:
    def test_defaults_match_the_specified_watermark(self, config: WatermarkConfig) -> None:
        assert config.scheme is HashScheme.LEFTHASH
        assert config.context_width == 1
        assert config.mix_width is MixWidth.BITS32
        assert config.gamma == DEFAULT_GAMMA
        assert config.delta == DEFAULT_DELTA

    def test_config_is_frozen(self, config: WatermarkConfig) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.gamma = 0.5  # type: ignore[misc]

    def test_fields_are_keyword_only(self, fingerprint: str) -> None:
        """Positional construction would make gamma/delta silently swappable."""
        with pytest.raises(TypeError):
            WatermarkConfig(SECRET_KEY, VOCAB_SIZE, fingerprint)  # type: ignore[misc]

    def test_configs_compare_by_value_and_are_hashable(self, fingerprint: str) -> None:
        first = build(vocab_fingerprint=fingerprint)
        second = build(vocab_fingerprint=fingerprint)
        assert first == second
        assert len({first, second}) == 1
        assert first != build(vocab_fingerprint=fingerprint, gamma=0.5)


class TestSecretKey:
    def test_str_keys_are_normalized_to_utf8_bytes(self) -> None:
        assert build(secret_key="hunter2").secret_key == b"hunter2"

    def test_str_and_bytes_keys_produce_equal_configs(self) -> None:
        assert build(secret_key="hunter2") == build(secret_key=b"hunter2")

    @pytest.mark.parametrize("bad_key", [b"", ""])
    def test_empty_key_is_rejected(self, bad_key: bytes | str) -> None:
        with pytest.raises(ConfigError, match="secret_key"):
            build(secret_key=bad_key)

    @pytest.mark.parametrize("bad_key", [None, 1234, ["k"]])
    def test_non_string_key_is_rejected(self, bad_key: object) -> None:
        with pytest.raises(ConfigError, match="secret_key"):
            build(secret_key=bad_key)

    def test_generate_secret_key_is_random_and_long_enough(self) -> None:
        keys = {generate_secret_key() for _ in range(8)}
        assert len(keys) == 8
        assert all(len(key) >= 32 for key in keys)

    def test_repr_never_leaks_the_key(self, config: WatermarkConfig) -> None:
        text = repr(config)
        assert "unit-test-key" not in text
        assert "REDACTED" in text


class TestGamma:
    @pytest.mark.parametrize("bad_gamma", [0.0, 1.0, -0.1, 1.5, float("nan")])
    def test_gamma_outside_the_open_unit_interval_is_rejected(self, bad_gamma: float) -> None:
        with pytest.raises(ConfigError, match="gamma"):
            build(gamma=bad_gamma)

    @pytest.mark.parametrize("bad_gamma", [0.7, 0.9, 0.99])
    def test_gamma_that_would_green_the_whole_vocabulary_is_rejected(
        self, bad_gamma: float
    ) -> None:
        """round(1/gamma) == 1 means every token is green, which is not a watermark."""
        with pytest.raises(ConfigError, match="gamma"):
            build(gamma=bad_gamma)

    @pytest.mark.parametrize(
        ("gamma", "expected"),
        [(0.5, 0.5), (0.25, 0.25), (0.1, 0.1), (0.3, 1 / 3), (0.2, 0.2), (0.4, 0.5)],
    )
    def test_effective_gamma_reflects_the_integer_greenlist_rule(
        self, gamma: float, expected: float
    ) -> None:
        """A token is green iff hash % round(1/gamma) == 0, so gamma is quantized."""
        assert build(gamma=gamma).effective_gamma == pytest.approx(expected)

    def test_green_divisor_is_an_integer_at_least_two(self) -> None:
        assert build(gamma=0.25).green_divisor == 4
        assert build(gamma=0.3).green_divisor == 3


class TestDelta:
    @pytest.mark.parametrize("bad_delta", [-1.0, float("nan"), float("inf")])
    def test_negative_or_non_finite_delta_is_rejected(self, bad_delta: float) -> None:
        with pytest.raises(ConfigError, match="delta"):
            build(delta=bad_delta)

    def test_zero_delta_is_allowed_as_an_explicit_no_op(self) -> None:
        assert build(delta=0.0).delta == 0.0


class TestVocabSize:
    @pytest.mark.parametrize("bad_size", [0, -1])
    def test_non_positive_vocab_size_is_rejected(self, bad_size: int) -> None:
        with pytest.raises(ConfigError, match="vocab_size"):
            build(vocab_size=bad_size)

    @pytest.mark.parametrize("bad_size", [512.0, "512", True, None])
    def test_non_integer_vocab_size_is_rejected(self, bad_size: object) -> None:
        """Notably bool, which is an int subclass and would otherwise slip through."""
        with pytest.raises(ConfigError, match="vocab_size"):
            build(vocab_size=bad_size)

    def test_unpartitionable_vocab_size_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="vocab_size"):
            build(vocab_size=1)


class TestSchemeAndContextWidth:
    def test_lefthash_is_defined_only_for_a_single_context_token(self) -> None:
        with pytest.raises(ConfigError, match="LEFTHASH"):
            build(scheme=HashScheme.LEFTHASH, context_width=4)

    def test_minhash_defaults_to_the_specified_width(self) -> None:
        assert build(scheme=HashScheme.MINHASH).context_width == 4

    def test_minhash_width_is_user_selectable(self) -> None:
        assert build(scheme=HashScheme.MINHASH, context_width=8).context_width == 8

    @pytest.mark.parametrize("bad_width", [0, -1, 2.0, "4"])
    def test_invalid_context_width_is_rejected(self, bad_width: object) -> None:
        with pytest.raises(ConfigError, match="context_width"):
            build(scheme=HashScheme.MINHASH, context_width=bad_width)

    def test_scheme_accepts_its_string_name(self) -> None:
        assert build(scheme="minhash").scheme is HashScheme.MINHASH

    def test_unknown_scheme_lists_the_supported_ones(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            build(scheme="selfhash")
        assert "lefthash" in str(excinfo.value)
        assert "minhash" in str(excinfo.value)


class TestMixWidth:
    def test_accepts_the_plain_integer(self) -> None:
        assert build(mix_width=64).mix_width is MixWidth.BITS64

    @pytest.mark.parametrize("bad", [16, 128, 0, -32, "32", 32.0, None, True])
    def test_unsupported_width_lists_the_supported_ones(self, bad: object) -> None:
        with pytest.raises(ConfigError, match="mix_width") as excinfo:
            build(mix_width=bad)
        assert "32" in str(excinfo.value)
        assert "64" in str(excinfo.value)

    def test_width_is_part_of_config_identity(self) -> None:
        """Changing the width changes every greenlist, so it cannot compare equal."""
        assert build(mix_width=32) != build(mix_width=64)

    def test_width_survives_a_json_round_trip(self) -> None:
        original = build(mix_width=64)
        assert WatermarkConfig.from_json(original.to_json(), secret_key=SECRET_KEY) == original


class TestFingerprintField:
    @pytest.mark.parametrize("bad", ["", "abc", "z" * 64, "0" * 63, None, 12345])
    def test_malformed_fingerprint_is_rejected(self, bad: object) -> None:
        with pytest.raises(ConfigError, match="vocab_fingerprint"):
            build(vocab_fingerprint=bad)

    def test_uppercase_fingerprint_is_normalized(self) -> None:
        assert build(vocab_fingerprint="A" * 64).vocab_fingerprint == "a" * 64


class TestFromTokenizer:
    def test_computes_the_fingerprint(self, tokenizer: HFStyleTokenizer, fingerprint: str) -> None:
        config = WatermarkConfig.from_tokenizer(
            tokenizer, vocab_size=VOCAB_SIZE, secret_key=SECRET_KEY
        )
        assert config.vocab_fingerprint == fingerprint
        assert config.vocab_size == VOCAB_SIZE

    def test_vocab_size_is_required_and_never_inferred(self, tokenizer: HFStyleTokenizer) -> None:
        with pytest.raises(TypeError):
            WatermarkConfig.from_tokenizer(tokenizer, secret_key=SECRET_KEY)  # type: ignore[call-arg]

    def test_forwards_watermark_parameters(self, tokenizer: HFStyleTokenizer) -> None:
        config = WatermarkConfig.from_tokenizer(
            tokenizer,
            vocab_size=VOCAB_SIZE,
            secret_key=SECRET_KEY,
            gamma=0.5,
            delta=4.0,
            scheme=HashScheme.MINHASH,
            context_width=2,
        )
        assert (config.gamma, config.delta, config.context_width) == (0.5, 4.0, 2)
        assert config.scheme is HashScheme.MINHASH


class TestSerialization:
    def test_to_dict_omits_the_secret_key_by_default(self, config: WatermarkConfig) -> None:
        payload = config.to_dict()
        assert "secret_key" not in payload
        assert payload["vocab_size"] == VOCAB_SIZE
        assert payload["scheme"] == "lefthash"
        assert payload["mix_width"] == 32

    def test_to_dict_can_include_the_key_explicitly(self, config: WatermarkConfig) -> None:
        payload = config.to_dict(include_secret_key=True)
        assert payload["secret_key"] == SECRET_KEY.hex()

    def test_round_trip_with_the_key_supplied_separately(self, config: WatermarkConfig) -> None:
        restored = WatermarkConfig.from_json(config.to_json(), secret_key=SECRET_KEY)
        assert restored == config

    def test_round_trip_with_an_embedded_key(self, config: WatermarkConfig) -> None:
        restored = WatermarkConfig.from_json(config.to_json(include_secret_key=True))
        assert restored == config

    def test_from_json_without_any_key_is_rejected(self, config: WatermarkConfig) -> None:
        with pytest.raises(ConfigError, match="secret_key"):
            WatermarkConfig.from_json(config.to_json())

    def test_explicit_key_overrides_an_embedded_one(self, config: WatermarkConfig) -> None:
        payload = config.to_json(include_secret_key=True)
        restored = WatermarkConfig.from_json(payload, secret_key=b"other-key")
        assert restored.secret_key == b"other-key"

    def test_json_is_stable_and_human_readable(self, config: WatermarkConfig) -> None:
        payload = json.loads(config.to_json())
        assert payload["gamma"] == DEFAULT_GAMMA
        assert config.to_json() == config.to_json()

    def test_unknown_json_field_is_rejected_by_name(self, config: WatermarkConfig) -> None:
        payload = json.loads(config.to_json())
        payload["aggressiveness"] = 3
        with pytest.raises(ConfigError, match="aggressiveness"):
            WatermarkConfig.from_json(json.dumps(payload), secret_key=SECRET_KEY)

    def test_missing_json_field_is_rejected_by_name(self, config: WatermarkConfig) -> None:
        payload = json.loads(config.to_json())
        del payload["vocab_fingerprint"]
        with pytest.raises(ConfigError, match="vocab_fingerprint"):
            WatermarkConfig.from_json(json.dumps(payload), secret_key=SECRET_KEY)

    def test_malformed_json_is_rejected_clearly(self) -> None:
        with pytest.raises(ConfigError, match="JSON"):
            WatermarkConfig.from_json("{not json", secret_key=SECRET_KEY)


class TestVerifyTokenizer:
    def test_matching_tokenizer_passes(
        self, config: WatermarkConfig, tokenizer: HFStyleTokenizer
    ) -> None:
        config.verify_tokenizer(tokenizer)

    def test_different_vocabulary_names_both_fingerprints(self, config: WatermarkConfig) -> None:
        other = HFStyleTokenizer(make_pieces(VOCAB_SIZE, prefix="piece"))
        with pytest.raises(VocabMismatchError) as excinfo:
            config.verify_tokenizer(other)
        message = str(excinfo.value)
        assert config.vocab_fingerprint in message
        assert fingerprint_from_tokenizer(other, VOCAB_SIZE) in message

    def test_padded_vocab_size_names_both_sizes_and_how_to_fix_it(
        self, tokenizer: HFStyleTokenizer
    ) -> None:
        """The Llama-3 case: config says 128256, the tokenizer reports 128000."""
        config = WatermarkConfig(
            secret_key=SECRET_KEY,
            vocab_size=VOCAB_SIZE + 16,
            vocab_fingerprint="0" * 64,
        )
        with pytest.raises(VocabMismatchError) as excinfo:
            config.verify_tokenizer(tokenizer)
        message = str(excinfo.value)
        assert str(VOCAB_SIZE + 16) in message
        assert str(VOCAB_SIZE) in message
        assert "vocab_size" in message
