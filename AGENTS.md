# Language Model Personal Watermark

This project's objective is to create a tool that allows easy, plug-and-play watermarking of language models using a KGW-style watermark.

The package is named `llmwatermark` (import name and distribution name).


## Technical context

### The core idea (KGW-style), roughly:

- At each generation step, before sampling the next token, hash the last k tokens (context window) combined with your secret key to seed a pseudo-random function.
- Use that seed to split the vocabulary into a "greenlist" and "redlist" (e.g. 50/50, or skewed).
- Bias the logits — boost greenlist token probabilities slightly (a "bias" parameter δ) before sampling.
- Over many tokens, watermarked text ends up with a statistically detectable excess of greenlist tokens.

### Detector side:

- Walk through the text, recompute the same greenlist/redlist split at each position using your key + preceding context.
- Count how many observed tokens fall in the greenlist.
- Run a z-test (or similar) against the expected ratio under the null (unwatermarked text) — high z-score = likely watermarked.


## Features of the tool

- Must be written in python, compatible with at least all 3.10+ versions.
- Must be compatible with transformers, llama.cpp, mlx-lm, ExLlamaV2/V3, vLLM, SGLang, TensorRT-LLM, LMDeploy python libraries.
- First demo version must work on transformers and vLLM, and once that works we move to the rest of the libraries.
- Must allow watermarking personalization (secret-key generation, aggressiveness selection by greenlist percentage of the total token list and bias parameter, context window size), must be easy to just plug on the model (whichever it is), and must include a detector function to easily detect if the text is produced by the watermarked model. It also must include a way to visualize how the watermarking is working and how the detector is making the decision.


## Watermark specification

These decisions are fixed. They exist so that a watermark generated on one backend detects identically on another. Do not deviate without explicit user confirmation.

### Vocabulary domain and the fingerprint

The greenlist partition is computed over **token IDs**, not token strings. Hashing strings cannot be vectorized over a 128k vocabulary per decode step and would break the performance budget.

The token ID to token string map is stable across backends — it is baked into the embedding matrix, so a backend that reordered it would produce garbage output. What is **not** stable is the reported vocabulary size. Models pad the embedding matrix beyond the real tokenizer (Llama-3: `len(tokenizer)` = 128000, `config.vocab_size` = 128256), and backends disagree on which number they report. Partitioning over 128000 vs 128256 produces entirely different greenlists.

Therefore:

- `vocab_size` is an explicit, required field of the watermark config. It is never inferred silently at generation time.
- The config carries a **vocab fingerprint**: `sha256(vocab_size ‖ id→string for a fixed, deterministic sample of token IDs)`.
- The detector recomputes the fingerprint from the tokenizer it was given and compares. On mismatch it raises immediately with a message naming both the expected and actual vocab size and fingerprint, and how to resolve it.

The ID→string map needed for the fingerprint is reachable on every target backend: transformers, vLLM (`get_tokenizer()`), SGLang, TensorRT-LLM, LMDeploy and mlx-lm all expose an HF tokenizer; llama.cpp exposes `llama_token_get_text`; ExLlamaV2 exposes `get_id_to_piece_list()`.

### Hash scheme

Default: **LeftHash with h=1** (seed derived from the single preceding token). It is the cheapest, the most robust to edits and paraphrase, and matches the original KGW paper, so the detector can be validated against published z-scores.

A `scheme` enum is exposed with **MinHash h=4** as the alternative for users who need resistance to greenlist reverse-engineering. The tradeoff across schemes: larger context width h gives more distinct greenlists, better text quality and harder reverse-engineering, but any single edited token within the h-window breaks that position.

SelfHash is deliberately **not** implemented. It requires a hash per candidate token per step over the whole vocabulary, which conflicts directly with the performance budget. Revisit only if spoofing resistance becomes a stated requirement.

### PRNG determinism

The seed derivation must be byte-identical across operating systems, Python versions, numpy versions and torch versions. A watermark generated on one machine must detect on any other.

- Seed derivation is **HMAC-SHA256(secret_key, context_token_ids)** from `hmac`/`hashlib`, truncated to the needed width.
- Never use Python's built-in `hash()` — it is randomized per process for str/bytes via `PYTHONHASHSEED`.
- Never rely on `random.seed()`, unseeded `torch.randperm`, or any generator whose stream is not guaranteed stable across library versions.
- Context token IDs are serialized to bytes in a fixed, documented byte order before hashing.

### Greenlist construction

A token is green iff `int_hash(seed, token_id) mod round(1/γ) == 0`, evaluated as a vectorized integer operation over the entire vocabulary on the same device as the logits.

Do **not** use the reference implementation's `torch.randperm(V, generator=g)[:γV]` per row per step. A 128k-element permutation per batch row per token makes the tool unusable in production.

### Where δ is applied

**δ is added to the raw logits before any sampling warper** — before temperature, top-k and top-p.

This is a strength-reproducibility contract, not a correctness one: the detector never sees δ, it only counts greenlist hits. But applied *after* top-p or top-k, a green token already masked to `-inf` cannot be rescued, so the watermark becomes weak and erratic at low top-p. Applied before, green tokens can enter the nucleus. It also means δ has the same meaning on every backend rather than being silently rescaled by temperature.

Seven of the eight target backends run logits processors before the sampler and satisfy this naturally (vLLM, SGLang, llama.cpp where the sampler chain order is explicit, mlx-lm, ExLlamaV2/V3, TensorRT-LLM, LMDeploy).

`transformers` is the exception: custom processors passed via `logits_processor=` are appended *after* the default warpers. The transformers adapter must insert the watermark processor at index 0. A test that pins this ordering empirically is required before the transformers adapter is considered working.

### Statelessness and batching

**The watermark is stateless.** The greenlist at position *t* is a pure function of `(secret_key, last h token IDs)`. The last h tokens are read from the row's own token history on every step.

Never cache watermark state keyed by batch index, row position or request ID. vLLM preempts and reschedules sequences, beam search reorders rows, and speculative decoding rolls tokens back — any mutable per-row state desynchronises and silently corrupts the watermark with no visible error.

This rule is what makes batched and streaming generation work for free, including vLLM V1's batched `LogitsProcessor` interface, where `apply()` receives the logits for the whole batch at once.


## Detector specification

The detector requires **only a tokenizer and the secret key — never the model**. Keep it free of heavy dependencies so detection can run anywhere.

- Skip the first h positions of the sequence. They have no full context window and no greenlist is defined for them.
- **Deduplicate repeated context n-grams** before scoring. Repeated contexts reuse the same greenlist and inflate the z-score, producing false positives on repetitive text. This is a known KGW artifact, not an edge case.
- Score: `z = (|s|_G - γT) / sqrt(T · γ · (1 - γ))`, where T is the number of scored tokens after skipping and dedup.
- Convert z to a one-sided p-value. The default decision threshold is a named, documented constant, not a magic number inline.
- **Minimum token count**: below a documented floor the z-score is not meaningful. Return a clear error naming the floor and the count actually seen, rather than a confident-looking wrong answer.
- Verify the vocab fingerprint before scoring and fail loudly on mismatch.
- The detector reports the per-token greenlist decisions alongside the aggregate score, so the visualization layer can render how the decision was reached without recomputing anything.

Retokenization caveat to document: detection tokenizes text that was produced as token IDs. Round-tripping through text is not guaranteed to recover the original IDs, especially where a backend's tokenizer differs on edge cases from the HF implementation. Document this limit honestly in the README.


## Performance budget

The watermark adds one O(V) integer hash plus a mask add per decode step. Against a forward pass this should be negligible; a naive implementation lands at 20–50% throughput loss and is not shippable.

Targets, measured as tokens/sec degradation against the unwatermarked baseline:

- **≤2% at batch size ≥32**
- **≤5% at batch size 1** (fixed kernel-launch cost is not amortized)

Hard constraints:

- No Python loop over the batch.
- No host↔device synchronisation on the hot path.
- The greenlist mask is built on the same device as the logits.
- No `B×V` float temporary where an in-place add works.

Benchmarks live alongside the tests and are run when the hot path changes.


## Dependency policy

The core package depends on **numpy only**. Every backend adapter is an optional extra, so installing the tool never pulls torch, tensorrt, mlx and eight inference stacks:

```
pip install llmwatermark                  # core + detector
pip install "llmwatermark[transformers]"
pip install "llmwatermark[vllm]"
pip install "llmwatermark[viz]"        # matplotlib, for plots and animations
```

Adapters import their backend lazily and raise a clear install message naming the right extra when the backend is missing.


## Testing and hardware

Test-driven development, as described below, has to remain executable without eight different accelerators.

- **Core tests** (hashing, greenlist construction, determinism, z-test, detector, config, fingerprint) are pure CPU, deterministic, and always run.
- **Adapter tests** are marked `@pytest.mark.requires_<backend>` and skipped by default. Each declares the hardware it needs.
- Determinism is tested explicitly: the same key and context must produce the same greenlist across processes and across runs.
- Cross-backend agreement is tested where hardware permits: text generated on one backend must detect with the same score on another.


## Visualization

Two layers, split so that the zero-dependency one is always available.

### Inline decision view (core, no extra dependencies)

`DetectionResult` renders the detector's per-token reasoning directly:

- `to_ansi()` — colored token stream for the terminal, green tokens vs red tokens, skipped positions dimmed.
- `to_html()` — a self-contained HTML string, no external assets, with per-token hover showing the position, the context that seeded it, and whether it was scored or skipped by the dedup rule.
- `_repr_html_()` delegates to `to_html()`, so a `DetectionResult` renders automatically in a Jupyter notebook.

This is the primary answer to "show how the detector is making the decision". It ships in the core package and pulls no dependencies.

### Plots (optional `[viz]` extra, matplotlib)

- Cumulative z-score against token index, with the decision threshold drawn as a horizontal line. Shows how confidence builds with text length.
- Green-fraction distribution, watermarked output against an unwatermarked baseline, showing the two populations separating.

### Animation

One example script producing a gif: the cumulative z-score curve building token by token during streaming generation, via matplotlib's animation writer. Lives in `examples/` as a standalone script and is the asset embedded in the README.

Nothing here may become a required dependency of the core package or of the detector.


## Extra features

Abstain from creating features not described in this file. If you realize a new feature might be beneficial, you need explicit confirmation from the user to proceed with its implementation.


## Documentation

Keep all the code well documented by using comments and a README.md file. The README.md file must be sufficient for anyone to understand how to use the tool.

Include some demonstration scripts, plots, and animations for better understanding.


## Usage

This tool is intended to be a python library, that you can import into a python file and use together with the libraries mentioned above. Therefore, it must allow the user to pip install it using the github repository URL, import it in a file, add to the model to alter the generation, and use to detect if a text is created using the watermarked model.

Make the code as simple and optimized for efficiency as possible, so it can be suited for production deployment. Try to make the wrapper around every library as similar as possible, but find the best way to adapt it to each of them.


## Test-Driven Development

Always plan your next steps in advance, with a clear path of what you want to achieve, why that must be achieved, how to achieve it and how to verified it is achieved. Everytime you are going to implement a new feature or solve a bug, make sure you first create a test in the folder tests/ so you can ensure no bugs are present in the code with pytest. While creating the tests, think of any edge cases that can be detrimental for the tool, and take them into account in the tests.

Handle errors professionally and cleanly, so any possible error is ensured to return a clear and concise message about why it happened and how to solve it.


## How to face uncertainty

When you face uncertainty in any decision, dont act on your own. In that case, you must ask the user for his opinion, and then proceed accordingly.

The decisions already fixed in "Watermark specification", "Detector specification", "Performance budget" and "Dependency policy" are settled. Follow them without re-asking.


## Repo structure

Always keep in mind to organize the repository in a professional way:
- Before creating any new file, function or chunk of code, look if there is another file, function or chunk of code that already contains the solution. Instead of create redundant code, use modular structure to reuse as much code as possible, always taking into account the efficiency trade-off.
- Keep an organized folder structure, with readable and understandable folder and file names so it is easy and pleasant for a human to navigate through the files to find what they need.
- Only create the minimal necessary files and code, to keep the repository clean. If a code snippet is going to be used only once, then abstain from creating a file. This does not apply to `examples/`, where each demonstration script is expected to be a standalone file.
- Use OOP principles so all code is organized in classes, methods, properties, dataclasses and functions with readable names and a clear name structure.
