# Throughput results

Measured with the scripts in this directory. Reproduce with:

```bash
python tests/benchmarks/bench_kernel.py
python tests/benchmarks/bench_transformers.py --model Qwen/Qwen2.5-1.5B-Instruct
python tests/benchmarks/bench_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct --batch 32
```

**Hardware**: NVIDIA RTX 5070 Laptop, 8 GB, 291 GB/s measured copy bandwidth, WSL2,
torch 2.13.0+cu129. A laptop GPU, not a serving part. See the caveats at the end.

## The headline: a fixed cost per decode step

The watermark adds a **fixed ~0.25 ms per decode step** — roughly 0.06–0.10 ms of kernel
plus ~0.19 ms of Python and kernel-launch dispatch. It does not scale with model size.

So whether it fits a 2% budget is a property of *the model's step time*, not of the
watermark:

| decode step | watermark share |
|---|---|
| 5.8 ms (125M params) | ~4% |
| 12 ms | ~2% |
| 23 ms (1.5B params) | ~1% |
| 40 ms+ (7B params and up) | under 0.6% |

This is more useful than a single percentage, because the percentage belongs to the
machine it was measured on and the ratio does not. **Below roughly a 12 ms decode step the
2% budget is not met; above it, it is** — and production serving is well above it.

## Budget verdicts

Targets: ≤2% at batch ≥32, ≤5% at batch 1.

### Qwen2.5-1.5B-Instruct, transformers, 128 new tokens

| batch | baseline | watermarked | tok/s off | tok/s on | delta | noise floor | budget | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | 2992 ms | 2985 ms | 43 | 43 | −0.22% | 2.70% | 5% | below noise |
| 8 | 3097 ms | 3079 ms | 331 | 333 | −0.58% | 1.07% | 2% | below noise |

The measured overhead is smaller than the run-to-run spread, so it cannot be quoted as a
figure. What *can* be said is stronger than it sounds: even the pessimistic bound
(`|delta| + noise`) is **1.65% against a 2% budget** at batch 8, and **2.92% against 5%**
at batch 1. The error bar itself fits inside the budget.

A negative delta is not a speed-up. It is the measurement saying the effect is below its
own resolution.

### Qwen2.5-1.5B-Instruct, vLLM, batch 32, 128 new tokens

Each row changes one thing from the row above it.

| configuration | baseline | watermarked | tok/s off | tok/s on | overhead | noise |
|---|---|---|---|---|---|---|
| eager kernel, pageable copy, no CUDA graphs | 2087 ms | 2485 ms | 1963 | 1648 | +19.08% | 3.00% |
| compiled kernel | 2001 ms | 2316 ms | 2047 | 1769 | +15.71% | 1.90% |
| pinned async copy | 1972 ms | 2141 ms | 2077 | 1913 | +8.60% | 2.38% |
| CUDA graphs (vLLM's default) | 1508 ms | 1587 ms | 2716 | 2581 | **+5.22%** | 0.91% |

**vLLM is at 5.22% against a 2% budget** - still over, and the one place the project
misses its own target, but 3.7x better than where the milestone started.

Three of the four points recovered came from configuration or integration mistakes, all of
them defaults chosen defensively without measuring:

* **compile=False** in the M8 tests, inherited by the benchmark. Worth 3.4 points.
* **A pageable host-to-device copy** of the context, every decode step. The transformers
  adapter slices its context from a device tensor; vLLM must assemble it on the host, and
  `torch.as_tensor(..., device=...)` from pageable memory *blocks*. That copy measures 27
  microseconds on an idle GPU and drains the queue once per step inside a live engine.
  Reusable pinned buffers with non-blocking copies: worth 7.1 points.
* **enforce_eager=True** in the benchmark, set for faster engine startup. It disables CUDA
  graphs, so vLLM launches every model kernel individually, the host becomes the
  bottleneck, and our launches block waiting for queue space. Production does not run this
  way. Worth 3.4 points.

Measured from inside the engine process, using a timing subclass vLLM loads by import
path:

| | |
|---|---|
| `processor.apply`, in engine | 3231 us/call |
| the same compiled kernel, same tensors, same process | **165 us** |
| the same kernel, eager | 1574 us |
| the same kernel, `dynamic=False` | 143 us |

The kernel is 165 us; the call around it was blocking for 3231 us. That is host stalling,
not GPU work, and it is what the CUDA-graphs row addresses.

It also disproves what this document previously offered as the leading hypothesis - that
`dynamic=True` produced a poorly fused kernel. Static shapes are worth 22 microseconds,
not milliseconds.

**What remains.** At 11.8 ms per step the 165 us kernel is 1.4%, and the logits read plus
write it performs (38.9 MB at 291 GB/s = 0.134 ms) is very close to a bandwidth floor, so
the kernel itself has little left to give. The measured 5.22% leaves roughly 0.45 ms per
step above that floor, still unexplained, and now the largest known item in the project.
A profiler trace from inside the engine subprocess is the next step rather than more
hypotheses.

### SGLang - not measured

The SGLang adapter is verified for correctness (`tests/test_adapter_sglang.py`, nine tests
against a live engine) but has no throughput row yet, and the gap is deliberate rather than
overlooked: quoting a number here would take more care than the other backends did.

Watermarked SGLang has to run with `disable_overlap_schedule=True`, because under overlap
the request history is one token stale and the output stops detecting. Overlap is a
throughput feature, so the honest comparison is not watermark-off against watermark-on. It
is three arms - stock SGLang, SGLang with overlap disabled, and SGLang with overlap disabled
plus the watermark - which separates what the watermark costs from what disabling overlap
costs. Reporting the two together as one figure would overstate the watermark, and reporting
only the last two would hide a cost the user pays to have it.

The environment is also not the one a serving benchmark should use: FlashInfer needs a CUDA
toolkit this machine lacks, so the engine ran on Triton attention with PyTorch sampling and
CUDA graphs disabled. That is fine for correctness and wrong for throughput.

### Qwen2.5-0.5B-Instruct-GGUF (Q4_K_M), llama.cpp, 128 new tokens

| model | build | step | baseline | watermarked | overhead | per step | noise |
|---|---|---|---|---|---|---|---|
| 0.5B Q4 | CPU | 9.0 ms | 1155 ms | 1230 ms | **+6.45%** | +582 us | 4.78% |
| 0.5B Q4 | CUDA offload | 2.9 ms | 375 ms | 473 ms | **+26.13%** | +765 us | 2.93% |
| 1.5B Q4 | CUDA offload | 5.6 ms | 711 ms | 826 ms | **+16.27%** | +904 us | 3.00% |

llama.cpp is the most expensive backend in the project, and GPU offload makes it worse
rather than better. That is not a paradox, and the three rows above are the demonstration:
the watermark's cost per step barely moves across them, while the *step* changes by 3x and
the percentage follows it almost exactly.

**Read the percentage against the step time, not against the other backends.** The
llama.cpp rows are not comparable with the transformers and vLLM ones. Those run a 1.5B
model at 16 bits - transformers at `float16`, vLLM at the `bfloat16` its config declares -
where the 0.5B row here is 4-bit `Q4_K_M`. Its decode step is 2.9 ms where vLLM's is 11.8 ms
and transformers' is 23.4 ms. Against vLLM's 5.22% the decomposition is a 1.2x larger
numerator and a 4.0x smaller denominator, and 1.2 x 4.0 is the 5x that separates them.

The third row controls for **parameter count only**: the same backend on a 1.5B model,
where the step roughly doubles and the overhead roughly halves, to +16.27%. It does not
control for precision, and precision turns out to be the larger term. Batch-1 decode is
bound by streaming the weights, and Q4_K_M is 1.1 GB against bf16's 2.9 GB - 2.76x less
traffic per step. That accounts for most of the 4.18x gap between this row's 5.6 ms step
and transformers' 23.4 ms one, with llama.cpp's leaner C++ loop taking the rest.

So a like-for-like comparison against transformers does not exist in this table. The
closest honest statement is the ratio the whole document is built on: the watermark costs
roughly 900 us per step on this path, so it is 16% of a 5.6 ms step, would be about 4% of
transformers' 23.4 ms one, and would be under 2% of a 7B model's. Quantizing a model
shrinks every part of the step *except* the watermark, which is why it shows up worst
exactly where the model has been made cheapest.

Two real effects remain underneath, and they are why llama.cpp is genuinely the worst case
rather than merely the smallest-model case:

* **Nothing is amortized.** llama.cpp is single-sequence, so its greenlist serves one
  token. vLLM at batch 32 runs one `(32, 151936)` kernel per step and divides its cost by
  32 - about 19 us per generated token, against roughly 900 us here.
* **Nothing is fused.** numpy runs eight separate passes over a 600 KB array where
  `torch.compile` emits one kernel that keeps the whole thing in registers.

The per-step cost is also *not* quite fixed on this path: it grew from 765 us to 904 us
between the 0.5B and 1.5B rows, with an identical vocabulary and identical work. That is
consistent with the 139 us cold-cache penalty measured directly - a larger model evicts
more of the buffers between steps - and it means the host path pays more, not less, as the
model grows.

The cause is measured, not guessed. llama.cpp hands its logits processors a numpy array on
the host, so the greenlist runs on the CPU where `torch.compile` cannot fuse it. Adding a
*no-op* logits processor costs only 75 us per step, so llama.cpp's own marshalling is not
the problem; the arithmetic is.

**These figures are after optimization.** The first measurement was +11.82% on CPU and
+52.33% with offload, at +1162 us and +1531 us per step. `llmwatermark.fastpath` roughly
halved both by rewriting the same function for numpy's cost model - see that module for
what changed and why each part was worth it. The mask itself went from 526 us to 150 us and
the bias from 180 us to 45 us, so `apply` at batch 1 costs 258 us where it cost 1208 us.

Three measured results shaped it, and two rejected candidates are worth recording:

| change | effect | kept |
|---|---|---|
| `% divisor` to bitwise AND (power-of-two) or Lemire's test | 219 us to 25 / 35 us | yes |
| reused thread-local buffers instead of fresh 600 KB temporaries | mixer 326 us to 134 us | yes |
| unsigned view, removing the sign-extension masks | 3 fewer passes | yes |
| bias scale as `np.float32` rather than a Python float | 180 us to 45 us | yes |
| blocking the loop so the working set fits L2 | best case 8%, worse below 16k | no |
| a reused float buffer for `green * scale` | 44.7 us to 41.0 us | no |

Integer division was the single largest item and the least obvious: it has no SIMD form, so
it alone cost 219 us of a 526 us mask. Lemire's replacement is an identity, not an
approximation, and was checked against `%` over all 2**32 unsigned values for six divisors
before adoption - zero mismatches.

**Roughly 430 us per step is still unaccounted for**, above the 258 us the kernel costs in
isolation. Two parts of it are measured: llama.cpp's own processor marshalling is 75 us,
and running with caches cold - which is the real situation, since the model's forward pass
evicts everything between steps - adds 139 us. The remaining ~215 us is not explained.
Blocking was the obvious remedy and did not work, so the next candidate is reducing the
number of full-width passes rather than making each one cheaper.

Nothing here changes a single greenlist decision. `tests/test_fastpath.py` asserts the fast
path agrees with the shared implementation over whole vocabularies, for both mixer widths
and six divisors, and the golden vectors are untouched.

### facebook/opt-125m, transformers, 128 new tokens

| batch | delta | noise floor | budget | verdict |
|---|---|---|---|---|
| 1 | +4.34% | 3.03% | 5% | within |
| 8 | +4.26% | 2.98% | 2% | **over** |
| 32 | +3.86% | 6.73% | 2% | below noise |

Reported because it is the honest worst case, not because it is representative. A 125M
model has a 5.8 ms decode step, which is almost entirely launch overhead already, so the
watermark's fixed cost lands as ~4%. The overhead being **constant across batch sizes**
while the kernel cost varies is what identifies it as fixed dispatch rather than GPU work.

## Kernel in isolation

`bench_kernel.py`, vocab_size 128256, float32 logits, per-call median.

| batch | width | eager | compiled | speedup |
|---|---|---|---|---|
| 1 | 32 | 0.303 ms | **0.061 ms** | 5.0x |
| 8 | 32 | 0.316 ms | **0.060 ms** | 5.3x |
| 32 | 32 | 0.960 ms | **0.096 ms** | 10.0x |
| 64 | 32 | 3.218 ms | **0.204 ms** | 15.7x |
| 1 | 64 | 0.227 ms | 0.059 ms | 3.9x |
| 32 | 64 | 3.242 ms | 0.143 ms | 22.7x |
| 64 | 64 | 6.353 ms | 0.275 ms | 23.1x |

Two things worth noting. **Compilation is not optional**: eager misses the budget by an
order of magnitude, which is why `compile="auto"` is the default. And under fusion the
32-bit and 64-bit mixers are close to parity — the 32-bit advantage is an eager-mode
effect, because fused intermediates never reach memory.

## Asserted, not just reported

Two properties are tests rather than measurements, because a silent regression in either
is correctness-adjacent. Both run under `--backend transformers --backend cuda`:

- **no host/device synchronisation** in the adapter's real call path, enforced with
  `torch.cuda.set_sync_debug_mode("error")`;
- **no `batch x vocab_size` float temporary** allocated per step, enforced against
  `torch.cuda.max_memory_allocated()`.

The percentages stay reported rather than asserted: a throughput threshold enforced in CI
on shared runners is a flake generator, and CI has no GPU.

## Method

Sloppy baselines make percentages meaningless, so the rules are in `_harness.py`:

- warm-up runs discarded — compilation, allocator growth and autotuning all land early;
- median reported, with the spread beside it;
- baseline and watermarked runs **interleaved**, so thermal drift cannot land on one arm
  and masquerade as a result;
- the sync amortized across an inner loop, because a decode loop does not synchronise
  every step and doing so adds tens of microseconds to a measurement of the same order;
- identical seeds, prompts and token counts across arms, with `min_new_tokens` pinned so
  an early EOS cannot change the denominator.

Three measurement bugs were caught by these rules rather than shipped:

1. **Too few repeats.** An early run reported +5.12% at batch 1 with five repeats. The
   run-to-run spread is larger than that. The noise floor column exists so a number
   smaller than its own error bar is labelled rather than quoted.
2. **Sync per call.** Synchronising after every call inflated the kernel measurement by
   roughly 3x at small batch.
3. **A contended GPU.** The first 1.5B run reported 15 tok/s with a 26.7% noise floor,
   while 7.7 GB of 8 GB was in use by another process. On a free GPU the same benchmark
   gives 43 tok/s and a 2.7% noise floor.

## Caveats

- **One laptop GPU, one small model.** Absolute latencies do not transfer. The *ratio*
  should, since both the watermark kernel and a decode step are bandwidth-bound.
- **Batch 64 goes superlinear here** (3x the cost for 2x the work) — an 8 GB card
  artifact, not a property of the watermark.
- **These are throughput numbers only.** They say nothing about whether the watermarked
  text is any good. That is measured separately.

## Open

- **SGLang has no throughput figure at all**, and needs a three-arm measurement to get one
  that means anything, since the watermark requires the overlap scheduler off.
- **llama.cpp is at 6.5% on CPU and 26.1% with GPU offload**, down from 11.8% and 52.3%
  but still over budget. About 430 us per step sits above what the kernel costs in
  isolation; 75 us of that is llama.cpp's marshalling and 139 us is cold caches, and the
  rest is unexplained.
- **vLLM is at 5.22% against a 2% budget.** The largest known gap among the torch backends,
  down from 19.08%. Roughly 0.45 ms per step remains above the kernel's bandwidth floor and is
  unexplained; see above for what has been ruled out by measurement.
- **Speculative decoding is unmeasured.** The vLLM adapter refuses loudly if the row count
  disagrees with the tracker rather than biasing the wrong rows, but that path has not been
  exercised.
