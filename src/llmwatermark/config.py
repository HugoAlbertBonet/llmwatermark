"""The watermark configuration: the contract shared by the generator and the detector.

Generation and detection are only interoperable when both sides agree on every field
here, byte for byte. Validation is therefore strict and errors name the offending field
and how to fix it, rather than letting a subtly wrong config produce a confidently wrong
detection result.
"""

from __future__ import annotations

import dataclasses
import json
import math
import secrets
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Final

from llmwatermark.errors import ConfigError, VocabMismatchError
from llmwatermark.vocab import fingerprint_from_tokenizer, observed_vocab_size

__all__ = [
    "DEFAULT_DELTA",
    "DEFAULT_GAMMA",
    "DEFAULT_MINHASH_CONTEXT_WIDTH",
    "SECRET_KEY_BYTES",
    "HashScheme",
    "MixWidth",
    "WatermarkConfig",
    "generate_secret_key",
    "normalize_secret_key",
    "validate_vocab_size",
]

# KGW paper defaults: a quarter of the vocabulary greened, biased by 2.0 logits.
DEFAULT_GAMMA: Final[float] = 0.25
DEFAULT_DELTA: Final[float] = 2.0

# MinHash needs a width; the specified alternative to LeftHash is h=4.
DEFAULT_MINHASH_CONTEXT_WIDTH: Final[int] = 4

# 256 bits: forging a watermark means guessing this key.
SECRET_KEY_BYTES: Final[int] = 32

# A greenlist covering (almost) the whole vocabulary is not a watermark. See green_divisor.
MIN_GREEN_DIVISOR: Final[int] = 2

_FINGERPRINT_HEX_LENGTH: Final[int] = 64
_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdef")


class HashScheme(str, Enum):
    """How the seed's context window is taken from the preceding tokens.

    Wider context gives more distinct greenlists, better text quality and harder
    reverse-engineering, but any single edited token inside the window breaks that
    position. SelfHash is deliberately absent: it needs a hash per candidate token per
    step over the whole vocabulary, which the performance budget rules out.
    """

    LEFTHASH = "lefthash"
    """Seed from the single preceding token (h=1). Cheapest and most edit-robust."""

    MINHASH = "minhash"
    """Seed from the minimum over an h-token window. Resists greenlist recovery."""

    @classmethod
    def parse(cls, value: HashScheme | str) -> HashScheme:
        """Accept either the enum member or its name, with a listing error otherwise."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
        supported = ", ".join(member.value for member in cls)
        raise ConfigError(
            f"scheme must be one of: {supported}. Got {value!r}. "
            "(SelfHash is not implemented; see the hash scheme section of the README.)"
        )


class MixWidth(IntEnum):
    """Integer width of the greenlist bit-mixer.

    This is part of the watermark format, not a tuning knob: the two widths use different
    constants and shift distances and therefore produce entirely different greenlists.
    Text watermarked at one width does not detect at the other.

    32 bits is the default. The mixer materializes a ``batch x vocab_size`` integer buffer
    every decode step, so halving the element size halves the dominant memory traffic on
    the hot path. The seed is truncated to 32 bits for the mix only; the key stays 256-bit
    and the HMAC in :mod:`llmwatermark.seeding` is untouched.
    """

    BITS32 = 32
    BITS64 = 64

    @classmethod
    def parse(cls, value: MixWidth | int) -> MixWidth:
        """Accept the enum member or the plain integer 32 / 64."""
        if isinstance(value, cls):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            try:
                return cls(value)
            except ValueError:
                pass
        supported = ", ".join(str(member.value) for member in cls)
        raise ConfigError(f"mix_width must be one of: {supported}. Got {value!r}.")


def generate_secret_key(num_bytes: int = SECRET_KEY_BYTES) -> bytes:
    """Generate a cryptographically random watermark key.

    Anyone holding this key can both detect and forge the watermark, so keep it secret
    and reuse the same key everywhere a given watermark must be detectable.
    """
    if not isinstance(num_bytes, int) or isinstance(num_bytes, bool) or num_bytes < 16:
        raise ConfigError(f"num_bytes must be an integer of at least 16, got {num_bytes!r}.")
    return secrets.token_bytes(num_bytes)


@dataclass(frozen=True, kw_only=True, repr=False)
class WatermarkConfig:
    """Everything needed to place or detect one watermark.

    :param secret_key: The watermark key. ``str`` is accepted and encoded as UTF-8.
        Prefer :func:`generate_secret_key`.
    :param vocab_size: The number of token IDs the greenlist partitions. Required and
        never inferred: a padded model reports two different sizes and choosing the wrong
        one silently invalidates the watermark.
    :param vocab_fingerprint: 64 hex characters identifying the vocabulary. Use
        :meth:`from_tokenizer` to compute it.
    :param gamma: Target greenlist fraction. Quantized by the integer greenlist rule -
        see :attr:`effective_gamma`.
    :param delta: Logit bias added to green tokens, before any sampling warper.
    :param scheme: Which context hash to use.
    :param context_width: How many preceding tokens seed the greenlist. ``None`` selects
        the scheme's default (1 for LeftHash, 4 for MinHash); after construction it is
        always an int.
    :param mix_width: Integer width of the greenlist mixer. Changes every greenlist, so
        it must match between generation and detection.
    """

    secret_key: bytes
    vocab_size: int
    vocab_fingerprint: str
    gamma: float = DEFAULT_GAMMA
    delta: float = DEFAULT_DELTA
    scheme: HashScheme = HashScheme.LEFTHASH
    context_width: int | None = None
    mix_width: MixWidth = MixWidth.BITS32

    def __post_init__(self) -> None:
        # Normalize first so validation and equality see canonical values.
        self._set("secret_key", normalize_secret_key(self.secret_key))
        self._set("scheme", HashScheme.parse(self.scheme))
        self._set("vocab_fingerprint", _normalized_fingerprint(self.vocab_fingerprint))
        self._set("context_width", _normalized_context_width(self.context_width, self.scheme))
        self._set("mix_width", MixWidth.parse(self.mix_width))
        validate_vocab_size(self.vocab_size)
        _validate_gamma(self.gamma)
        _validate_delta(self.delta)

    def _set(self, field: str, value: object) -> None:
        """Assign to a frozen field during __post_init__ normalization."""
        object.__setattr__(self, field, value)

    # -- derived values ---------------------------------------------------------------

    @property
    def h(self) -> int:
        """The resolved context width, as a plain int for hot-path code."""
        assert self.context_width is not None  # guaranteed by __post_init__
        return self.context_width

    @property
    def green_divisor(self) -> int:
        """A token is green iff ``int_hash(seed, token_id) % green_divisor == 0``."""
        return round(1.0 / self.gamma)

    @property
    def effective_gamma(self) -> float:
        """The greenlist fraction actually realized, ``1 / green_divisor``.

        The greenlist rule is an integer modulus, so a requested gamma of 0.3 becomes
        1/3. The detector must score against this value, not the requested one, or the
        z-score is biased.
        """
        return 1.0 / self.green_divisor

    # -- construction -----------------------------------------------------------------

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: object,
        *,
        vocab_size: int,
        secret_key: bytes | str,
        gamma: float = DEFAULT_GAMMA,
        delta: float = DEFAULT_DELTA,
        scheme: HashScheme | str = HashScheme.LEFTHASH,
        context_width: int | None = None,
        mix_width: MixWidth | int = MixWidth.BITS32,
    ) -> WatermarkConfig:
        """Build a config, computing the vocabulary fingerprint from ``tokenizer``.

        ``vocab_size`` stays explicit on purpose. Pass the size the model generates over
        (usually ``model.config.vocab_size``), which may exceed ``len(tokenizer)``.
        """
        return cls(
            secret_key=secret_key,  # type: ignore[arg-type]
            vocab_size=vocab_size,
            vocab_fingerprint=fingerprint_from_tokenizer(tokenizer, vocab_size),
            gamma=gamma,
            delta=delta,
            scheme=scheme,  # type: ignore[arg-type]
            context_width=context_width,
            mix_width=mix_width,  # type: ignore[arg-type]
        )

    def verify_tokenizer(self, tokenizer: object) -> None:
        """Raise unless ``tokenizer`` is the vocabulary this watermark was built over.

        Called before any scoring. A mismatch means the detector would partition a
        different set of token IDs than the generator did.
        """
        actual = fingerprint_from_tokenizer(tokenizer, self.vocab_size)
        if actual == self.vocab_fingerprint:
            return
        observed = observed_vocab_size(tokenizer)
        seen = "unknown" if observed is None else str(observed)
        raise VocabMismatchError(
            "this tokenizer is not the vocabulary the watermark was built over.\n"
            f"  expected: vocab_size={self.vocab_size} fingerprint={self.vocab_fingerprint}\n"
            f"  actual:   vocab_size={seen} fingerprint={actual}\n"
            "Load the tokenizer of the model that generated the text, and pass the same "
            "vocab_size that was used at generation time (for a padded model that is "
            "config.vocab_size, not len(tokenizer))."
        )

    # -- serialization ----------------------------------------------------------------

    def to_dict(self, *, include_secret_key: bool = False) -> dict[str, Any]:
        """Serialize to plain JSON-compatible types.

        The secret key is omitted by default: a config is meant to be shareable, and the
        key is the one field that must not be. Opt in only for private storage.
        """
        payload: dict[str, Any] = {
            "vocab_size": self.vocab_size,
            "vocab_fingerprint": self.vocab_fingerprint,
            "gamma": self.gamma,
            "delta": self.delta,
            "scheme": self.scheme.value,
            "context_width": self.h,
            "mix_width": int(self.mix_width),
        }
        if include_secret_key:
            payload["secret_key"] = self.secret_key.hex()
        return payload

    def to_json(self, *, include_secret_key: bool = False) -> str:
        return json.dumps(self.to_dict(include_secret_key=include_secret_key), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str, *, secret_key: bytes | str | None = None) -> WatermarkConfig:
        """Rebuild a config from :meth:`to_json`.

        ``secret_key`` takes precedence over any key embedded in the payload, so a shared
        config file can be combined with a locally held key.
        """
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"could not parse the watermark config as JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(
                f"a watermark config must be a JSON object, got {type(data).__name__}."
            )
        return cls.from_dict(data, secret_key=secret_key)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, secret_key: bytes | str | None = None
    ) -> WatermarkConfig:
        """Rebuild a config from :meth:`to_dict`."""
        fields = dict(data)
        embedded = fields.pop("secret_key", None)
        if secret_key is None:
            secret_key = _key_from_hex(embedded)

        expected = {field.name for field in dataclasses.fields(cls)} - {"secret_key"}
        unknown = sorted(set(fields) - expected)
        if unknown:
            raise ConfigError(
                f"unknown watermark config field(s): {', '.join(unknown)}. "
                f"Expected only: {', '.join(sorted(expected))}."
            )
        missing = sorted(expected - set(fields))
        if missing:
            raise ConfigError(
                f"missing watermark config field(s): {', '.join(missing)}. "
                "A config must be serialized with WatermarkConfig.to_json()."
            )
        # __post_init__ accepts and normalizes a str key; the field annotation is the
        # post-normalization type.
        return cls(secret_key=secret_key, **fields)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        """Redact the key: configs end up in logs, tracebacks and notebook output."""
        return (
            f"{type(self).__name__}(secret_key=<REDACTED {len(self.secret_key)} bytes>, "
            f"vocab_size={self.vocab_size}, vocab_fingerprint={self.vocab_fingerprint!r}, "
            f"gamma={self.gamma}, delta={self.delta}, scheme={self.scheme.value!r}, "
            f"context_width={self.h}, mix_width={int(self.mix_width)})"
        )


def normalize_secret_key(secret_key: object) -> bytes:
    """Validate a watermark key and return it as bytes. Shared with the seeding module."""
    if isinstance(secret_key, str):
        secret_key = secret_key.encode("utf-8")
    if not isinstance(secret_key, (bytes, bytearray)):
        raise ConfigError(
            f"secret_key must be bytes or str, got {type(secret_key).__name__}. "
            "Use llmwatermark.generate_secret_key() to create one."
        )
    if not secret_key:
        raise ConfigError(
            "secret_key must not be empty. Use llmwatermark.generate_secret_key() to "
            "create one, and keep it secret: it both detects and forges the watermark."
        )
    return bytes(secret_key)


def _key_from_hex(embedded: object) -> bytes:
    if embedded is None:
        raise ConfigError(
            "no secret_key: the config was serialized without one. Pass "
            "secret_key=... to from_json(), or serialize with include_secret_key=True."
        )
    if not isinstance(embedded, str):
        raise ConfigError(
            f"secret_key in a serialized config must be a hex string, "
            f"got {type(embedded).__name__}."
        )
    try:
        return bytes.fromhex(embedded)
    except ValueError as exc:
        raise ConfigError(f"secret_key in a serialized config is not valid hex: {exc}") from exc


def _normalized_fingerprint(fingerprint: object) -> str:
    if not isinstance(fingerprint, str):
        raise ConfigError(
            f"vocab_fingerprint must be a {_FINGERPRINT_HEX_LENGTH}-character hex string, "
            f"got {type(fingerprint).__name__}. Build the config with "
            "WatermarkConfig.from_tokenizer() to compute it."
        )
    normalized = fingerprint.lower()
    if len(normalized) != _FINGERPRINT_HEX_LENGTH or not set(normalized) <= _HEX_DIGITS:
        raise ConfigError(
            f"vocab_fingerprint must be {_FINGERPRINT_HEX_LENGTH} hex characters, got "
            f"{fingerprint!r}. Build the config with WatermarkConfig.from_tokenizer()."
        )
    return normalized


def _normalized_context_width(context_width: object, scheme: HashScheme) -> int:
    if context_width is None:
        return 1 if scheme is HashScheme.LEFTHASH else DEFAULT_MINHASH_CONTEXT_WIDTH
    if not isinstance(context_width, int) or isinstance(context_width, bool):
        raise ConfigError(f"context_width must be an integer, got {type(context_width).__name__}.")
    if context_width < 1:
        raise ConfigError(f"context_width must be at least 1, got {context_width}.")
    if scheme is HashScheme.LEFTHASH and context_width != 1:
        raise ConfigError(
            f"scheme LEFTHASH seeds from the single preceding token, so context_width "
            f"must be 1, got {context_width}. Use scheme='minhash' for a wider context."
        )
    return context_width


def validate_vocab_size(vocab_size: object) -> None:
    """Validate a vocabulary size. Shared with the seeding module."""
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
        raise ConfigError(
            f"vocab_size must be an integer, got {type(vocab_size).__name__}. Pass the "
            "size the model generates over, usually model.config.vocab_size."
        )
    if vocab_size < 2:
        raise ConfigError(f"vocab_size must be at least 2 to be partitioned, got {vocab_size}.")


def _validate_gamma(gamma: object) -> None:
    if not isinstance(gamma, (int, float)) or isinstance(gamma, bool):
        raise ConfigError(f"gamma must be a float, got {type(gamma).__name__}.")
    if not 0.0 < float(gamma) < 1.0:
        raise ConfigError(
            f"gamma is the greenlist fraction and must lie strictly between 0 and 1, got {gamma}."
        )
    if round(1.0 / float(gamma)) < MIN_GREEN_DIVISOR:
        raise ConfigError(
            f"gamma={gamma} would green the entire vocabulary (round(1/gamma) < "
            f"{MIN_GREEN_DIVISOR}), which is not a watermark. Use gamma <= 0.5; the "
            f"default is {DEFAULT_GAMMA}."
        )


def _validate_delta(delta: object) -> None:
    if not isinstance(delta, (int, float)) or isinstance(delta, bool):
        raise ConfigError(f"delta must be a float, got {type(delta).__name__}.")
    value = float(delta)
    if not math.isfinite(value) or value < 0.0:
        raise ConfigError(
            f"delta is the logit bias added to green tokens and must be finite and "
            f"non-negative, got {delta}. The default is {DEFAULT_DELTA}; delta=0 disables "
            "the bias."
        )
