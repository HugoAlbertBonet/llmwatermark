# Quality and detection evaluation

What the watermark costs, and what it buys, measured rather than argued.

Reproduce with:

```bash
python tests/benchmarks/eval/eval_human_fpr.py       # false positives on human writing
python tests/benchmarks/eval/eval_generate.py        # the answer sets
python tests/benchmarks/eval/eval_report.py          # detection power, length, diversity
python tests/benchmarks/eval/eval_perplexity.py      # quality under a held-out judge
python tests/benchmarks/eval/eval_judge.py export    # blinded pairs for a frontier judge
python tests/benchmarks/eval/eval_judge.py score --verdicts <file>
```

**Setup.** Generator Qwen2.5-1.5B-Instruct, 160 instructions sampled across five Dolly-15k
categories (CC BY-SA 3.0), 256 new tokens, temperature 0.8, top-p 0.95. Same prompts, same
seeds and the same token budget in every arm - only delta changes. RTX 5070 Laptop, 8 GB.

## Headline

**delta = 2, the library default, is the right operating point.** Detection saturates at
delta = 2 and the quality cost arrives between delta = 2 and delta = 4. Going higher buys
0.995 -> 1.000 in AUC and costs a great deal.

## Does the detector accuse innocent people?

3200 human-written Dolly responses scored under four independent keys.

```
mean z   +0.010   (nominal 0)
sd   z    1.001   (nominal 1)
max  z   +4.000
```

| threshold | nominal FPR | observed FPR | count |
|---|---|---|---|
| 1.645 | 5.00e-02 | 5.50e-02 | 176 |
| 2.000 | 2.28e-02 | 2.38e-02 | 76 |
| 3.000 | 1.35e-03 | 1.87e-03 | 6 |
| 4.000 (default) | 3.17e-05 | 0.00e+00 | 0 |

The core test suite calibrates against *uniformly random token IDs* and gets a textbook
standard normal. Real prose is not uniform - it repeats, it has boilerplate, it reuses
phrasing - and the n-gram deduplication is what is supposed to absorb that. This is the
first measurement of whether it does. It does: sd 1.001 against a nominal 1.0, and every
category mean within 0.13 of zero, including the repetitive ones (information extraction,
summarization, classification).

Had the null been wider than 1.0 here, every false-positive rate this project publishes
would have been wrong.

## Detection power

Positives are the watermarked arm; negatives are human text pooled with the unwatermarked
arm.

| delta | mean z | detected at z=4 | AUC | TPR@1% FPR | TPR@0.01% FPR | distinct-3 | tokens |
|---|---|---|---|---|---|---|---|
| 0 | -0.35 | 0.0% | - | - | - | 0.958 | 206 |
| 1 | +2.24 | 8.8% | 0.943 | 63.1% | 56.2% | 0.968 | 203 |
| **2** | **+5.74** | **77.5%** | **0.995** | **96.2%** | **95.0%** | 0.967 | 207 |
| 4 | +15.90 | 99.4% | 1.000 | 100% | 100% | 0.978 | 216 |
| 6 | +20.93 | 100% | 1.000 | 100% | 100% | 0.977 | 216 |

Detection is already near ceiling at delta = 2 on 256-token answers. Length rises with
delta (206 -> 216 tokens), which is the EOS perturbation the transformers adapter
documents: the end-of-sequence token is green or red like any other. Diversity does not
fall, so delta is not inducing repetition loops.

## Quality, instrument 1: perplexity under a held-out model

Scored with SmolLM2-1.7B - a different family with a different tokenizer. Scoring text with
the model that produced it is circular; scoring with a sibling is partly circular.

| arm | median PPL | mean PPL | vs delta 0 |
|---|---|---|---|
| human | 8.86 | 10.08 | - |
| 0 | 5.13 | 5.37 | 1.00x |
| 1 | 5.69 | 6.00 | 1.11x |
| **2** | **7.39** | **8.16** | **1.44x** |
| 4 | 22.44 | 23.37 | 4.38x |
| 6 | 44.37 | 46.54 | 8.66x |

**At delta = 2 the watermarked text is still less perplexing than human writing.** The cost
is roughly flat to delta = 2 and then explodes - 1.11x, 1.44x, **4.38x**, 8.66x. There is a
cliff between delta = 2 and delta = 4, and the default sits just before it.

## Quality, instrument 2: blinded pairwise judging

Every comparison is judged twice with the sides swapped; nothing in the prompt mentions
watermarking. Alongside the real comparisons sit **control pairs** - unwatermarked against
unwatermarked from different seeds, indistinguishable by construction. A judge that does
not straddle 50% there is reading position or length, and its other verdicts are discarded.

Confidence intervals are over *comparisons*, not verdicts: the two orderings of one
comparison are not independent observations, and counting both would halve the intervals
on no extra information. Comparisons whose two orderings disagree are dropped as
unresolved, which is what the agreement rate below measures.

| judge | comparisons | order-swapped agreement | control | delta = 2 | delta = 4 |
|---|---|---|---|---|---|
| Claude Opus 5 | 16 | 100% | 50.0% | 40.0% [11.8-76.9] | 0.0% [0.0-49.0] |
| judge A *(model to be recorded)* | 119 | 78.2% | 46.4% | 24.2% [12.8-41.0] | 18.8% [8.9-35.3] |
| judge B *(model to be recorded)* | 120 | **94.2%** | 58.3% | 43.2% [28.7-59.1] | 15.8% [7.4-30.4] |

All three controls straddle 50%, so all three are admissible.

**delta = 4 degrades quality, robustly.** Three independent judges, win rates 0%, 18.8% and
15.8%, every upper bound below 50%. Perplexity says the same thing at 4.38x.

**delta = 2 shows no detectable degradation** in the two high-consistency runs. The only run
that found a cost at delta = 2 is the one that agreed with itself 78.2% of the time, where
22% of comparisons flipped on a side swap. Self-consistency is the tie-breaker here, and
the ~80% bar was set before these results were seen, not after.

Perplexity agrees independently: 1.44x at delta = 2, still under human text.

## Guidance

| delta | detection | quality | use when |
|---|---|---|---|
| 1 | AUC 0.943 | negligible cost | detection is a nice-to-have |
| **2** | **AUC 0.995, 95% TPR at 0.01% FPR** | **no detectable cost** | **default** |
| 4 | AUC 1.000 | clear degradation, 4.4x perplexity | never, on this evidence |
| 6 | AUC 1.000 | severe, 8.7x perplexity | never, on this evidence |

## Caveats

- **One 1.5B model on a laptop GPU.** Quality conclusions transfer less well across scale
  than latency ratios do: larger models have more acceptable next tokens at each step and
  should tolerate delta better. Treat these as pessimistic for production models.
- **256-token answers.** Detection power grows as sqrt(length), so shorter passages need a
  larger delta or a lower threshold, and both cost something.
- **Judges rarely tie.** Two ties in 120 comparisons for judge B, none for judge A, against
  an explicit instruction that ties were expected. Forced discrimination inflates apparent
  differences - though it inflates them for the control row too, which stayed at 50%.
- **English only**, single-turn instructions, one dataset.
