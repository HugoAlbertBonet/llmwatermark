"""llmwatermark: plug-and-play KGW-style watermarking and detection for language models.

The top level of this package is intentionally dependency-light: importing it must never
pull in torch, an inference backend or matplotlib. Backend adapters live in
``llmwatermark.adapters`` and import their backend lazily.
"""

from llmwatermark.config import (
    DEFAULT_DELTA,
    DEFAULT_GAMMA,
    HashScheme,
    MixWidth,
    WatermarkConfig,
    generate_secret_key,
)
from llmwatermark.errors import (
    ConfigError,
    LLMWatermarkError,
    TokenizerInterfaceError,
    VocabMismatchError,
)
from llmwatermark.vocab import fingerprint_from_tokenizer

__version__ = "0.1.0.dev0"

__all__ = [
    "DEFAULT_DELTA",
    "DEFAULT_GAMMA",
    "ConfigError",
    "HashScheme",
    "LLMWatermarkError",
    "MixWidth",
    "TokenizerInterfaceError",
    "VocabMismatchError",
    "WatermarkConfig",
    "__version__",
    "fingerprint_from_tokenizer",
    "generate_secret_key",
]
