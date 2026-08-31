"""Blinded pairwise judging: export pairs, then score the verdicts.

Perplexity measures "does this look like text the judge model would have written", which
is a weak proxy for whether an answer is any good. A frontier model reading both answers
is a better instrument - provided it is used carefully, which here means:

* **Blinding.** Nothing in the exported pair says which side is watermarked, or that a
  watermark exists at all.
* **Order randomisation, and every pair judged twice with the sides swapped.** The
  self-consistency rate that falls out is a free measure of how much the judge is reading
  position rather than content.
* **Control pairs.** Alongside every watermarked-versus-unwatermarked comparison sit pairs
  that are unwatermarked-versus-unwatermarked from different seeds - indistinguishable by
  construction. A judge that does not land near 50/50 on those has a position or verbosity
  bias, and its verdict on the real arms is not evidence.
* **Ties allowed.** Forced choice manufactures differences that are not there.

Usage::

    python eval_judge.py export --out pairs.jsonl

    # either drive an OpenAI-compatible endpoint directly (resumable, names the model)
    OPENAI_API_KEY=... python eval_judge.py judge --model gpt-5 --out verdicts.jsonl

    # or split the pairs into files small enough for an agent to read and judge by hand
    python eval_judge.py chunks --size 30 --out-dir chunks/

    python eval_judge.py score --verdicts verdicts.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

PROMPT_TEMPLATE = """You are grading two candidate answers to the same instruction. Judge \
only the quality of the answers. You are not told, and must not guess, how either was \
produced.

INSTRUCTION:
{instruction}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Judge on, in priority order:
 1. Does it follow the instruction and answer what was asked?
 2. Is it factually coherent and internally consistent?
 3. Is it fluent, well organised, and free of degenerate repetition?

Do not reward length. Do not reward confident tone. If the two are of comparable quality, \
say TIE - ties are expected and useful.

Reply with exactly two lines:
VERDICT: A | B | TIE
REASON: <one sentence, under 25 words>"""


def export(arguments: argparse.Namespace) -> None:
    """Write the blinded pairs.

    Model-against-model comparisons ask what the watermark costs relative to the same model
    unwatermarked. The optional human comparisons ask a different and more end-user
    question: does watermarking change how the model fares against a person?

    Those human rows carry a heavy confound - Dolly answers are terse where the model is
    long and structured, and judges reward structure - so the absolute win rate against
    human text says little. The *difference* between the human-vs-delta-0 row and the
    human-vs-delta-N rows is the informative part, since both share the confound.
    """
    rows = [json.loads(line) for line in Path(arguments.generations).read_text().splitlines()]
    comparisons: list[tuple[str, str, str]] = [("control", "delta_0", "control")]
    comparisons += [(f"delta_{v:g}", "delta_0", f"delta_{v:g}") for v in arguments.deltas]
    if arguments.include_human:
        comparisons += [("human_vs_delta_0", "human", "delta_0")]
        comparisons += [
            (f"human_vs_delta_{v:g}", "human", f"delta_{v:g}") for v in arguments.deltas
        ]

    generator = random.Random(arguments.seed)
    pairs = []
    for index, row in enumerate(rows[: arguments.limit]):
        for label, baseline_arm, arm in comparisons:
            if arm not in row or baseline_arm not in row:
                continue
            baseline = row[baseline_arm] if baseline_arm == "human" else row[baseline_arm]["text"]
            treatment = row[arm]["text"]
            if not baseline.strip() or not treatment.strip():
                continue
            # Each pair twice, sides swapped: disagreement between the two is position bias.
            for repeat, flipped in enumerate((False, True)):
                first, second = (treatment, baseline) if flipped else (baseline, treatment)
                pairs.append(
                    {
                        "pair_id": f"{index}:{label}:{repeat}",
                        "arm": label,
                        "treatment_side": "A" if flipped else "B",
                        "category": row["category"],
                        "prompt": PROMPT_TEMPLATE.format(
                            instruction=row["instruction"], answer_a=first, answer_b=second
                        ),
                    }
                )
    generator.shuffle(pairs)
    destination = Path(arguments.out)
    with destination.open("w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair) + "\n")
    print(f"wrote {len(pairs)} blinded pairs to {destination}")
    print(f"arms: {sorted({pair['arm'] for pair in pairs})}")


def judge(arguments: argparse.Namespace) -> None:
    """Judge every pair through an OpenAI-compatible chat endpoint.

    Resumable: verdicts already in the output file are skipped, so an interrupted run
    continues where it stopped rather than paying for the whole set again.
    """
    import os
    import urllib.error
    import urllib.request

    key = os.environ.get(arguments.api_key_env)
    if not key:
        raise SystemExit(f"{arguments.api_key_env} is not set")

    pairs = [json.loads(line) for line in Path(arguments.pairs).read_text().splitlines()]
    destination = Path(arguments.out)
    done = set()
    if destination.exists():
        done = {
            json.loads(line)["pair_id"] for line in destination.read_text().splitlines() if line
        }
        print(f"resuming: {len(done)} verdicts already recorded")

    with destination.open("a") as handle:
        for index, pair in enumerate(pairs):
            if pair["pair_id"] in done:
                continue
            body = json.dumps(
                {
                    "model": arguments.model,
                    "messages": [{"role": "user", "content": pair["prompt"]}],
                    "max_completion_tokens": 200,
                }
            ).encode()
            request = urllib.request.Request(
                f"{arguments.base_url.rstrip('/')}/chat/completions",
                data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.load(response)
            except (urllib.error.URLError, TimeoutError) as error:
                print(f"  {pair['pair_id']}: request failed ({error}); skipping")
                continue
            reply = payload["choices"][0]["message"]["content"]
            verdict = next(
                (
                    line.split(":", 1)[1].strip().upper()
                    for line in reply.splitlines()
                    if line.strip().upper().startswith("VERDICT")
                ),
                "",
            )
            if verdict not in ("A", "B", "TIE"):
                print(f"  {pair['pair_id']}: unparseable verdict {reply[:60]!r}; skipping")
                continue
            handle.write(json.dumps({"pair_id": pair["pair_id"], "verdict": verdict}) + "\n")
            handle.flush()
            if (index + 1) % 20 == 0:
                print(f"  {index + 1}/{len(pairs)}")
    print(f"wrote verdicts to {destination}")


def chunks(arguments: argparse.Namespace) -> None:
    """Split the pairs into plain-text files small enough for an agent to read directly."""
    pairs = [json.loads(line) for line in Path(arguments.pairs).read_text().splitlines()]
    directory = Path(arguments.out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for start in range(0, len(pairs), arguments.size):
        batch = pairs[start : start + arguments.size]
        path = directory / f"chunk_{start // arguments.size:02d}.txt"
        with path.open("w") as handle:
            for pair in batch:
                handle.write(f"===== PAIR {pair['pair_id']} =====\n{pair['prompt']}\n\n")
        written += 1
    print(f"wrote {written} chunks of up to {arguments.size} pairs to {directory}")
    print("Judge each pair, then record one JSON object per line:")
    print('  {"pair_id": "...", "verdict": "A"}')


def wilson(successes: int, total: int) -> tuple[float, float]:
    """Wilson score interval, which behaves at the extremes where normal approximation does not."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.96
    rate = successes / total
    centre = rate + z * z / (2 * total)
    spread = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    denominator = 1 + z * z / total
    return ((centre - spread) / denominator, (centre + spread) / denominator)


def score(arguments: argparse.Namespace) -> None:
    pairs = {
        json.loads(line)["pair_id"]: json.loads(line)
        for line in Path(arguments.pairs).read_text().splitlines()
    }
    verdicts = {
        json.loads(line)["pair_id"]: json.loads(line)["verdict"].strip().upper()
        for line in Path(arguments.verdicts).read_text().splitlines()
    }

    # Each comparison is judged twice, once per side order. Those two verdicts are not
    # independent observations, so the tally counts *comparisons*, not verdicts: a pair
    # counts once when both orders agree, and is discarded as unresolved when they do not.
    # Counting all verdicts would halve the confidence intervals on nothing.
    outcomes: dict[str, list[str]] = defaultdict(list)
    arm_of: dict[str, str] = {}
    for pair_id, verdict in verdicts.items():
        pair = pairs.get(pair_id)
        if pair is None or verdict not in ("A", "B", "TIE"):
            continue
        if verdict == "TIE":
            outcome = "tie"
        else:
            outcome = "treatment" if verdict == pair["treatment_side"] else "baseline"
        comparison = pair_id.rsplit(":", 1)[0]
        outcomes[comparison].append(outcome)
        arm_of[comparison] = pair["arm"]

    tally: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    both_orders = [values for values in outcomes.values() if len(values) == 2]
    agreed = sum(1 for values in both_orders if len(set(values)) == 1)
    for comparison, values in outcomes.items():
        if len(values) == 2 and len(set(values)) == 1:
            tally[arm_of[comparison]][values[0]] += 1
        elif len(values) == 2:
            tally[arm_of[comparison]]["unresolved"] += 1

    agreement = agreed / max(len(both_orders), 1)
    print(
        f"verdicts: {len(verdicts)}   comparisons: {len(both_orders)}   "
        f"order-swapped agreement: {agreement:.1%}\n"
    )
    header = f"| {'arm':>9} | {'treatment wins':>14} | {'ties':>6} |"
    print(f"{header} {'baseline wins':>13} | {'win rate 95% CI':>18} |")
    print(f"|{'-' * 11}|{'-' * 16}|{'-' * 8}|{'-' * 15}|{'-' * 20}|")
    for arm in sorted(tally, key=lambda name: (name != "control", name)):
        counts = tally[arm]
        decided = counts["treatment"] + counts["baseline"]
        low, high = wilson(counts["treatment"], decided)
        print(
            f"| {arm:>9} | {counts['treatment']:>14} | {counts['tie']:>6} | "
            f"{counts['baseline']:>13} | {low:>8.1%} - {high:<8.1%} |"
        )
    print(
        "\nThe control row must straddle 50%. If it does not, the judge is reading position\n"
        "or length rather than quality, and the other rows are not evidence."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    exporter = sub.add_parser("export")
    exporter.add_argument("--generations", default="tests/benchmarks/eval/generations.jsonl")
    exporter.add_argument("--out", default="tests/benchmarks/eval/judge_pairs.jsonl")
    exporter.add_argument("--deltas", type=float, nargs="*", default=[2.0, 4.0])
    exporter.add_argument("--limit", type=int, default=60)
    exporter.add_argument("--seed", type=int, default=0)
    exporter.add_argument("--include-human", action="store_true")
    exporter.set_defaults(handler=export)

    api = sub.add_parser("judge")
    api.add_argument("--pairs", default="tests/benchmarks/eval/judge_pairs.jsonl")
    api.add_argument("--out", default="tests/benchmarks/eval/judge_verdicts.jsonl")
    api.add_argument("--model", required=True)
    api.add_argument("--base-url", default="https://api.openai.com/v1")
    api.add_argument("--api-key-env", default="OPENAI_API_KEY")
    api.set_defaults(handler=judge)

    splitter = sub.add_parser("chunks")
    splitter.add_argument("--pairs", default="tests/benchmarks/eval/judge_pairs.jsonl")
    splitter.add_argument("--out-dir", default="tests/benchmarks/eval/chunks")
    splitter.add_argument("--size", type=int, default=30)
    splitter.set_defaults(handler=chunks)

    scorer = sub.add_parser("score")
    scorer.add_argument("--pairs", default="tests/benchmarks/eval/judge_pairs.jsonl")
    scorer.add_argument("--verdicts", default="tests/benchmarks/eval/judge_verdicts.jsonl")
    scorer.set_defaults(handler=score)

    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
