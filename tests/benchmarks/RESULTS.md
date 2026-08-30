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

| configuration | baseline | watermarked | tok/s off | tok/s on | overhead | noise |
|---|---|---|---|---|---|---|
| eager, pageable copy | 2087 ms | 2485 ms | 1963 | 1648 | +19.08% | 3.00% |
| compiled, pageable copy | 2001 ms | 2316 ms | 2047 | 1769 | +15.71% | 1.90% |
| compiled, pinned async copy | 1972 ms | 2141 ms | 2077 | 1913 | **+8.60%** | 2.38% |

**vLLM does not meet the 2% budget.** It is over by four times, and this is the one
place the project misses its own target.

Two of the three points are understood. Compilation is worth 3.4 points, and the vLLM
tests had been passing `compile=False` - a default chosen defensively in M8 without
measuring, and wrong. Staging the context through reusable pinned buffers is worth a
further 7.1 points: the transformers adapter slices its context from a device tensor,
while vLLM has to assemble it on the host and copy it every step, and a pageable copy is
*blocking*. In isolation that copy costs 27 microseconds; inside a pipelined sampler it
drains the queue once per decode step, and the bubble is measured in milliseconds. An
idle-GPU microbenchmark cannot see this, and initially told us the host path was
irrelevant.

The remaining ~1.3 ms per step is **not explained**. Ruled out by measurement: the
compiled kernel (~0.15 ms at this vocabulary), context assembly (11 us), pinned staging
(17 us), and compilation failing to engage - the engine log confirms the kernel is
compiled inside the engine process. The leading untested hypothesis is that
`dynamic=True`, needed because vLLM's batch size varies between prefill and decode steps,
produces a less well fused kernel than static shapes would. Resolving it needs a profiler
trace from inside the engine subprocess, which the in-process instrumentation used here
cannot reach.

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

- **vLLM is at 8.60% against a 2% budget.** The largest known gap in the project. The
  residual is unexplained; see above for what has been ruled out.
- **Speculative decoding is unmeasured.** The vLLM adapter refuses loudly if the row count
  disagrees with the tracker rather than biasing the wrong rows, but that path has not been
  exercised.
