"""Timing discipline shared by the benchmark scripts.

A sloppy baseline makes every percentage meaningless, so the rules live in one place:

* warm-up runs are discarded - compilation, allocator growth and autotuning all land in
  the first few iterations;
* the median is reported, not the mean, with the spread beside it;
* baseline and treatment runs are **interleaved**, so thermal drift on a laptop GPU cannot
  land entirely on one arm and masquerade as a result;
* CUDA work is synchronised before the clock is read.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover - torch is optional
        pass


@dataclass(frozen=True)
class Timing:
    """Milliseconds per call: the median, and how much the samples spread."""

    median: float
    spread: float
    samples: int

    def __str__(self) -> str:
        return f"{self.median:8.3f} ms +/- {self.spread:.3f}"


def measure(
    call: Callable[[], Any], *, repeats: int = 20, warmup: int = 10, inner: int = 20
) -> Timing:
    """Time one callable, discarding warm-up and reporting the median per call.

    Each sample times ``inner`` back-to-back calls and synchronises once. Synchronising
    per call would add a fixed cost of tens of microseconds that a real decode loop never
    pays, which at this scale is the same order as the thing being measured.
    """
    for _ in range(warmup):
        call()
    synchronize()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(inner):
            call()
        synchronize()
        samples.append((time.perf_counter() - start) * 1e3 / inner)
    return Timing(
        median=statistics.median(samples),
        spread=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        samples=len(samples),
    )


def compare(
    baseline: Callable[[], Any],
    treatment: Callable[[], Any],
    *,
    repeats: int = 10,
    warmup: int = 2,
) -> tuple[Timing, Timing, float]:
    """Interleave two arms and report both timings and the treatment's overhead.

    Interleaving matters: run back to back, a GPU that heats up during the second arm
    reports that heating as the effect under test.
    """
    for _ in range(warmup):
        baseline()
        treatment()
    synchronize()

    baseline_samples, treatment_samples = [], []
    for _ in range(repeats):
        for collect, call in ((baseline_samples, baseline), (treatment_samples, treatment)):
            start = time.perf_counter()
            call()
            synchronize()
            collect.append((time.perf_counter() - start) * 1e3)

    first = Timing(
        statistics.median(baseline_samples), statistics.pstdev(baseline_samples), repeats
    )
    second = Timing(
        statistics.median(treatment_samples), statistics.pstdev(treatment_samples), repeats
    )
    overhead = (second.median - first.median) / first.median * 100.0
    return first, second, overhead


def noise_floor(baseline: Timing, treatment: Timing) -> float:
    """How large an apparent overhead the run-to-run spread alone could produce.

    An end-to-end generation on a laptop GPU varies by several percent between identical
    runs. A measured overhead smaller than this is not a result, and reporting it as one
    would be worse than reporting nothing.
    """
    spread = max(baseline.spread, treatment.spread)
    return spread / baseline.median * 100.0


def describe_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"{name}, {total:.1f} GB, torch {torch.__version__}"
        return f"CPU, torch {torch.__version__}"
    except ImportError:  # pragma: no cover
        return "CPU, torch not installed"
