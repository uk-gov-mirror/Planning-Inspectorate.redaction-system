"""Check how sensitive the ranking of sweep variants is to the optimisation objective.

Scores sweep rows with the repo's scoring, computes the objective

    J = clip(F2 - hard_negative_weight * HN rate - unlocatable_weight * U, 0, 1)

for a grid of weights, and reports whether the ranking (and the top variant) changes. A
ranking that holds across the grid means the weights do not decide the outcome. The
penalties are only informative if the variants differ in hard-negative or unlocatable
rates, so also read the per-variant components.

    python -m eval.objective_sensitivity --corpus eval/data/corpus.jsonl \
        --results eval/results/coarse.jsonl --out-dir eval/results/objective_sensitivity
"""

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Callable
from itertools import product
from pathlib import Path

import numpy as np

from .corpus import RULE_NAMES, load_corpus
from .scoring import (
    HARD_NEGATIVE_WEIGHT,
    UNLOCATABLE_WEIGHT,
    ScoreCounts,
    objective_score,
)
from .summarise_chunk_sweep import ALL_RULES, ScoredRun, Variant, score_rows

Weights = tuple[float, float]


def pooled_objective(runs: list[ScoredRun], weights: Weights) -> float:
    """Objective of counts summed over runs (how the gate and sweep pool)"""
    counts = ScoreCounts()
    for run in runs:
        counts += run.counts
    return objective_score(
        counts.f_beta(), counts.hard_negative_fp_rate, counts.unlocatable_rate, *weights
    )


def macro_objective(runs: list[ScoredRun], weights: Weights) -> float:
    """Mean of the per-run objective (how the optimiser averages over records)"""
    return float(
        np.mean(
            [
                objective_score(
                    run.counts.f_beta(),
                    run.counts.hard_negative_fp_rate,
                    run.counts.unlocatable_rate,
                    *weights,
                )
                for run in runs
            ]
        )
    )


AGGREGATIONS: dict[str, Callable[[list[ScoredRun], Weights], float]] = {
    "pooled": pooled_objective,
    "macro": macro_objective,
}


def objective_grid(
    scored: list[ScoredRun],
    rule: str,
    aggregation: str,
    hard_negative_weights: list[float],
    unlocatable_weights: list[float],
) -> tuple[list[Variant], dict[Weights, np.ndarray]]:
    """Objective of every variant (sorted) at every weight pair, for one rule"""
    by_variant = defaultdict(list)
    for run in scored:
        if run.rule == rule:
            by_variant[run.variant].append(run)
    variants = sorted(by_variant)
    aggregate = AGGREGATIONS[aggregation]
    return variants, {
        weights: np.array([aggregate(by_variant[v], weights) for v in variants])
        for weights in product(hard_negative_weights, unlocatable_weights)
    }


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall's tau-b between two score vectors (NaN if either is constant)"""
    concordant = discordant = ties_a = ties_b = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            ties_a += da == 0
            ties_b += db == 0
            concordant += da * db > 0
            discordant += da * db < 0
    pairs = len(a) * (len(a) - 1) // 2
    denominator = np.sqrt((pairs - ties_a) * (pairs - ties_b))
    return float((concordant - discordant) / denominator) if denominator else np.nan


def rank_stability(
    variants: list[Variant], grid: dict[Weights, np.ndarray], default: Weights
) -> dict:
    """How the ranking at each weight pair compares with the ranking at the default"""
    base = grid[default]
    taus = [kendall_tau(base, values) for values in grid.values()]
    top_counts: dict[Variant, int] = defaultdict(int)
    for values in grid.values():
        top_counts[variants[int(np.argmax(values))]] += 1
    return {
        "j_range_at_default": float(np.ptp(base)),
        "default_best": variants[int(np.argmax(base))],
        "tau_min": float(np.nanmin(taus)),
        "tau_median": float(np.nanmedian(taus)),
        "settings": len(grid),
        "top_counts": dict(top_counts),
    }


def component_rows(scored: list[ScoredRun], rule: str) -> list[dict]:
    """Pooled F2, precision, recall, HN rate and U per variant, for context"""
    by_variant = defaultdict(ScoreCounts)
    for run in scored:
        if run.rule == rule:
            by_variant[run.variant] += run.counts
    return [
        {
            "rule": rule,
            "chunk_size": v[0],
            "chunk_overlap": v[1],
            "f2": round(c.f_beta(), 4),
            "precision": round(c.precision, 4),
            "recall": round(c.recall, 4),
            "hard_negative_fp_rate": round(c.hard_negative_fp_rate, 4),
            "unlocatable_rate": round(c.unlocatable_rate, 4),
            "unlocatable_predictions": c.unlocatable_predictions,
            "predictions": c.predictions,
        }
        for v, c in sorted(by_variant.items())
    ]


def main(argv=None):
    """Command line entry point: print and optionally write the sensitivity check"""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--results", required=True, nargs="+")
    parser.add_argument("--out-dir", help="Write components.csv and grid.csv here")
    parser.add_argument(
        "--hard-negative-weights",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--unlocatable-weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0]
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    args = parser.parse_args(argv)

    default = (HARD_NEGATIVE_WEIGHT, UNLOCATABLE_WEIGHT)
    if default[0] not in args.hard_negative_weights:
        args.hard_negative_weights.append(default[0])
    if default[1] not in args.unlocatable_weights:
        args.unlocatable_weights.append(default[1])

    documents = {d.id: d for d in load_corpus(args.corpus)}
    rows = []
    for results_path in args.results:
        with Path(results_path).open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    scored = score_rows(rows, documents, args.coverage_threshold)
    rules = [r for r in (ALL_RULES, *RULE_NAMES) if any(s.rule == r for s in scored)]

    components = [row for rule in rules for row in component_rows(scored, rule)]
    grid_rows = []
    print(f"Default weights: hard negative {default[0]}, unlocatable {default[1]}")
    for aggregation in AGGREGATIONS:
        print(f"\n== {aggregation} ==")
        for rule in rules:
            variants, grid = objective_grid(
                scored,
                rule,
                aggregation,
                args.hard_negative_weights,
                args.unlocatable_weights,
            )
            stability = rank_stability(variants, grid, default)
            print(
                f"{rule}: J range at default {stability['j_range_at_default']:.4f}, "
                f"best {stability['default_best']}, tau vs default "
                f"min {stability['tau_min']:.2f} / median {stability['tau_median']:.2f}, "
                f"top variant over {stability['settings']} settings: "
                f"{stability['top_counts']}"
            )
            grid_rows += [
                {
                    "aggregation": aggregation,
                    "rule": rule,
                    "hard_negative_weight": weights[0],
                    "unlocatable_weight": weights[1],
                    "chunk_size": v[0],
                    "chunk_overlap": v[1],
                    "objective": round(float(value), 5),
                }
                for weights, values in grid.items()
                for v, value in zip(variants, values)
            ]

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, table in (("components", components), ("grid", grid_rows)):
            with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
        print(f"\nWrote {out_dir / 'components.csv'} and {out_dir / 'grid.csv'}")


if __name__ == "__main__":
    main()
