# llmwatermark

Plug-and-play [KGW-style](https://arxiv.org/abs/2301.10226) watermarking and detection
for language models.

> **Status: pre-alpha.** The package scaffold is in place; the watermark and detector
> are not implemented yet. This README is a placeholder and will be replaced with full
> usage documentation once the core is complete.

## Installation

```bash
pip install "llmwatermark @ git+https://github.com/HugoAlbertBonet/llmwatermark"
```

The core package and the detector depend on **numpy only**. Each inference backend is
an optional extra:

```bash
pip install "llmwatermark[transformers]"
pip install "llmwatermark[vllm]"
pip install "llmwatermark[viz]"          # matplotlib, for plots and animations
```

## Development

```bash
pip install -e ".[dev]"
pytest                       # core tests: pure CPU, deterministic, always run
ruff check . && mypy         # lint and type-check
```

Tests that need an inference backend are marked `requires_<backend>` and are skipped by
default. Opt in per backend, or enable all of them:

```bash
pytest --backend transformers
pytest --all-backends
```

## License

MIT. See [LICENSE](LICENSE).
