"""Summarize interchangeable tuning strategies for neural_texture.

Input:  results_tunable.jsonl
Output: strategy_comparison_table.md
"""

from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass


THIS_DIR = pathlib.Path(__file__).parent
RESULT = THIS_DIR / "results_tunable.jsonl"
OUT_TABLE = THIS_DIR / "strategy_comparison_table.md"
RANDOM_BUDGET = 12
RANDOM_SEED = 0


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    measured: int
    budget: str
    best_config: str
    throughput: float
    normalized: float
    note: str


def load_rows(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if "_meta" not in row and "error" not in row:
                rows.append(row)
    if not rows:
        raise ValueError(f"no result rows in {path}")
    return rows


def compare_strategies(rows: list[dict]) -> list[StrategyResult]:
    exhaustive_best = max(rows, key=lambda r: r["msamples_per_s"])
    exhaustive_throughput = exhaustive_best["msamples_per_s"]

    shuffled = list(rows)
    random.Random(RANDOM_SEED).shuffle(shuffled)
    random_sample = shuffled[:RANDOM_BUDGET]
    random_best = max(random_sample, key=lambda r: r["msamples_per_s"])

    return [
        StrategyResult(
            strategy="Exhaustive",
            measured=len(rows),
            budget="complete space",
            best_config=exhaustive_best["config"],
            throughput=exhaustive_throughput,
            normalized=1.0,
            note="default evaluation; exposes full spread",
        ),
        StrategyResult(
            strategy=f"Random, seed {RANDOM_SEED}",
            measured=len(random_sample),
            budget=f"{RANDOM_BUDGET} of {len(rows)} configs",
            best_config=random_best["config"],
            throughput=random_best["msamples_per_s"],
            normalized=random_best["msamples_per_s"] / exhaustive_throughput,
            note="same reflected space; bounded stochastic policy",
        ),
    ]


def write_table(results: list[StrategyResult]) -> None:
    lines = [
        "# Neural Texture Strategy Comparison",
        "",
        "| strategy | measured configs | budget | best config found | best throughput | vs. exhaustive best | note |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.strategy} | {result.measured} | {result.budget} | "
            f"`{result.best_config}` | {result.throughput:.2f} MSamples/s | "
            f"{result.normalized * 100.0:.1f}% | {result.note} |"
        )
    OUT_TABLE.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = load_rows(RESULT)
    results = compare_strategies(rows)
    write_table(results)
    print(f"wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
