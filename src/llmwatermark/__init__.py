"""llmwatermark: plug-and-play KGW-style watermarking and detection for language models.

The top level of this package is intentionally dependency-light: importing it must never
pull in torch, an inference backend or matplotlib. Backend adapters live in
``llmwatermark.adapters`` and import their backend lazily.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
