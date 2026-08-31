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
from llmwatermark.detector import (
    DEFAULT_Z_THRESHOLD,
    MIN_SCORED_TOKENS,
    DetectionResult,
    TokenRecord,
    WatermarkDetector,
    detect,
)
from llmwatermark.errors import (
    ConfigError,
    DetectionError,
    LLMWatermarkError,
    SeedingError,
    TokenizerInterfaceError,
    VocabMismatchError,
)
from llmwatermark.processor import WatermarkProcessor
from llmwatermark.vocab import fingerprint_from_tokenizer

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_DELTA",
    "DEFAULT_GAMMA",
    "DEFAULT_Z_THRESHOLD",
    "MIN_SCORED_TOKENS",
    "ConfigError",
    "DetectionError",
    "DetectionResult",
    "HashScheme",
    "LLMWatermarkError",
    "MixWidth",
    "SeedingError",
    "TokenRecord",
    "TokenizerInterfaceError",
    "VocabMismatchError",
    "WatermarkConfig",
    "WatermarkDetector",
    "WatermarkProcessor",
    "__version__",
    "detect",
    "fingerprint_from_tokenizer",
    "generate_secret_key",
]
