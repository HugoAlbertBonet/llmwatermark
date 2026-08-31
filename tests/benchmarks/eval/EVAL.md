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

**Per-key offset.** Texts scored under one key share a single greenlist, so they are not
independent observations and a per-key bias would not average away. Measured over 24 keys:

| corpus | per-key offset sd | observed range of per-key mean z |
|---|---|---|
| human writing | 0.05 - 0.08 | -0.29 to +0.12 |
| unwatermarked model output | 0.154 | -0.29 to +0.35 |

Model output carries roughly three times the offset of human writing, which follows from
its token distribution being more concentrated: fewer effective independent draws, so a
larger deviation from gamma for any particular greenlist.

The consequence is small but worth stating precisely. On human text - the case where a
false positive means accusing a person - the marginal null widens from N(0, 1) to about
N(0, 1.001) and the false-positive rate at z = 4 moves from 3.2e-05 to 3.3e-05. The worst
single key observed, +0.345 on model text, gives 1.3e-04 at that threshold: one in 7,700
rather than one in 31,600.

**So the published false-positive rate is per-key approximate, not exact.** It is not a
guarantee that holds identically for every key, and a deployment needing an exact bound
should measure its own key against a sample of in-domain text rather than trusting the
nominal figure. Nothing here changes any conclusion in this document.

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

All judging was run on 2026-08-29, driven through the Codex CLI over the chunked export.

| judge | comparisons | order-swapped agreement | ties | control | delta = 2 | delta = 4 |
|---|---|---|---|---|---|---|
| Claude Opus 5 (high) | 120 | **100%** | 17 | 65.5% [47.3-80.1] | 45.7% [30.5-61.8] | 5.1% [1.4-16.9] |
| GPT-5.6 Luna (high) | 119 | 78.2% | 0 | 46.4% [29.5-64.2] | 24.2% [12.8-41.0] | 18.8% [8.9-35.3] |
| **GPT-5.6 Sol (medium)** | 120 | 94.2% | 2 | 58.3% [42.2-72.9] | 43.2% [28.7-59.1] | 15.8% [7.4-30.4] |

All three controls straddle 50%, so all three are admissible.

Every run was blinded and executed in a fresh session with no knowledge of the project,
the hypothesis, or which side carried the watermark, so no judge's verdicts were shaped by
the conclusions drawn here. What the runs *do* share is the harness: Claude Opus 5 chose
what gets compared, what counts as resolved and how ties are broken. That design shapes all
four rows equally rather than any one of them specially, and it is the part a sceptical
reader should scrutinise.

Two observations for anyone repeating this. The higher-effort configuration was *less*
self-consistent: Luna at high effort flipped its verdict on 22% of comparisons when the
sides were swapped, Sol at medium effort on 6%. Reasoning effort is not a proxy for judge
reliability. And tie usage tracks reliability in the expected direction - Opus used TIE 17
times, most often on control pairs (11) and least on delta = 4 (1), which is what a
calibrated judge should do when the answers really are indistinguishable. Luna never tied
at all, against an explicit instruction that ties were expected.

**delta = 4 degrades quality, robustly.** Three judges over 240 comparisons each, win rates
5.1%, 15.8% and 18.8%, every upper bound below 50%. Perplexity says the same thing at
4.38x. This is the firmest result in the evaluation.

**delta = 2 shows no detectable degradation** in the two high-consistency runs (Opus at
100%, Sol at 94.2%). The only run that found a cost at delta = 2 is Luna at 78.2%, where
22% of comparisons flipped on a side swap. Self-consistency is the tie-breaker, and the
~80% bar was set before any of these results were seen, not after.

Perplexity agrees independently: 1.44x at delta = 2, still under human text.

### Against human writing

The comparisons above are model against model, which measures the watermark's cost relative
to the model's own baseline. A separate 480-pair run (GPT-5.6 Sol, medium, 240 comparisons,
89.2% agreement, control exactly 16-16) anchors the same question against human answers.
Win rates are for the *model*:

| comparison | model win rate | 95% CI |
|---|---|---|
| human vs delta = 0 | 13.9% | 6.1 - 28.7 |
| human vs delta = 2 | 15.8% | 7.4 - 30.4 |
| human vs delta = 4 | 5.3% | 1.5 - 17.3 |

delta = 0 and delta = 2 are indistinguishable against a fixed external reference; delta = 4
falls away from it. That is the same cliff the model-against-model rows and the perplexity
scores find, now measured against human writing rather than the model's own output.

Two things worth recording. This document previously predicted that judges would favour the
model's long, structured answers over Dolly's terse human ones, and warned that the
confound would make these rows hard to read. **The opposite happened** - human answers win
86% of the time - so the prediction was wrong in direction, and the rows are more
interpretable than claimed. The judging prompt for this run also added an explicit
instruction not to reward length, structure or formatting in themselves, which may account
for some of it.

And the generator loses to human answers 86% of the time *even unwatermarked*.
Qwen2.5-1.5B-Instruct is a small model with narrow quality headroom; a stronger model has
more acceptable next tokens at each step and may tolerate delta differently.

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
- **Two of three judges rarely tie.** Two ties in 120 comparisons for Sol, none at all for
  Luna, against an explicit instruction that ties were expected. Forced discrimination
  inflates apparent differences - though it inflates them for the control row too, which
  straddled 50% in every run.
- **The control row leans high for two judges** (58.3% and 65.5%, both straddling 50% but
  from above). Control pairs differ only by sampling seed, so there is nothing for a judge
  to prefer; treat small effects in the delta rows with corresponding caution.
- **English only**, single-turn instructions, one dataset.
