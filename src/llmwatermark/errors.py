"""Exception hierarchy.

Every error this package raises derives from :class:`LLMWatermarkError`, so callers can
catch the whole family, and also from the builtin exception a user would naturally expect
(``ValueError``, ``TypeError``), so existing error handling keeps working.

Messages are expected to state what went wrong, what was seen versus what was expected,
and how to resolve it.
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "LLMWatermarkError",
    "SeedingError",
    "TokenizerInterfaceError",
    "VocabMismatchError",
]


class LLMWatermarkError(Exception):
    """Base class for every error raised by llmwatermark."""


class ConfigError(LLMWatermarkError, ValueError):
    """A watermark configuration is invalid or internally inconsistent."""


class VocabMismatchError(LLMWatermarkError, ValueError):
    """The tokenizer does not match the vocabulary the watermark was built over.

    Raised before any scoring happens. Continuing would partition a different set of
    token IDs than the generator did, producing a confidently wrong answer.
    """


class SeedingError(LLMWatermarkError, ValueError):
    """A seed cannot be derived from the given context.

    Raised for token IDs outside the watermark vocabulary, malformed context arrays and
    context windows that do not match the configured scheme. Continuing would seed a
    greenlist the other side of the watermark cannot reproduce.
    """


class TokenizerInterfaceError(LLMWatermarkError, TypeError):
    """The given object does not expose a usable token ID to token string mapping."""
