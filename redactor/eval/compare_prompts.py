"""Compare the metrics of several prompt configs, e.g. basic, default and optimised.

Each config in `src/config` is run over a corpus split with `run_chunk_sweep` at the same
chunk settings, so only the prompt differs. The per-document results are scored with the
repo's scoring and saved as a plot and table. It uses no MLflow, and `--plot-only` makes
no LLM calls. Run from the `redactor` directory, inside the VNet when running configs:

    python -m eval.compare_prompts --configs basic default optimised
    python -m eval.compare_prompts --plot-only
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.redaction.config import LLMTextRedactionConfigBase

from .azure_auth import check_azure_auth
from .corpus import RULE_NAMES, load_corpus
from .run_chunk_sweep import overlap_for, run_sweep
from .run_chunk_sweep import parse_args as parse_sweep_args
from .scoring import (
    BETA,
    HARD_NEGATIVE_WEIGHT,
    UNLOCATABLE_WEIGHT,
    objective_score,
)
from .summarise_chunk_sweep import ALL_RULES, ScoredRun, score_rows, summarise

RESULTS_DIR = Path("eval/results")
COMPARE_CONFIGS = ("basic", "default", "optimised")
COMPARE_METRICS = {
    "f2": "F2",
    "recall": "Recall",
    "precision": "Precision",
    "hard_negative_hit_rate": "Hard-negative hit rate (lower is better)",
    "unlocatable_rate": "Unlocatable rate (lower is better)",
    "cost_per_1k_doc_tokens": "Cost per 1,000 document tokens (GBP)",
    "objective": "Objective (F2 less penalties)",
}
"""Metric names are those of the scorers in `mlflow_scorers`"""
UNIT_RANGE_METRICS = ("f2", "recall", "precision", "objective")


def comparison_rows(
    scored_by_config: dict[str, list[ScoredRun]],
    hard_negative_weight: float = HARD_NEGATIVE_WEIGHT,
    unlocatable_weight: float = UNLOCATABLE_WEIGHT,
) -> list[dict]:
    """
    The compared metrics for each config, for each rule and for all rules together.

    Counts are pooled over documents and repeats, as in the gate and the sweep summary.
    Each config's runs must be of a single chunk variant.
    """
    rows = []
    for config, scored in scored_by_config.items():
        summaries = {rule: s for (_, rule), s in summarise(scored, BETA).items()}
        for rule in (*RULE_NAMES, ALL_RULES):
            if rule not in summaries:
                continue
            summary, counts = summaries[rule], summaries[rule].counts
            rows.append(
                {
                    "config": config,
                    "rule": rule,
                    "runs": summary.runs,
                    "f2": counts.f_beta(BETA),
                    "recall": counts.recall,
                    "precision": counts.precision,
                    "hard_negative_hit_rate": counts.hard_negative_fp_rate,
                    "unlocatable_rate": counts.unlocatable_rate,
                    "cost_per_1k_doc_tokens": summary.cost_per_1k_doc_tokens,
                    "objective": objective_score(
                        counts.f_beta(BETA),
                        counts.hard_negative_fp_rate,
                        counts.unlocatable_rate,
                        hard_negative_weight,
                        unlocatable_weight,
                    ),
                }
            )
    return rows


def plot_comparison(rows: list[dict], out_path: Path, title: str = ""):
    """Save a grid of bar charts, one per metric, with a group of bars per rule (and
    for all rules) and one bar per config"""
    configs = list(dict.fromkeys(row["config"] for row in rows))
    rules = [
        rule
        for rule in (*RULE_NAMES, ALL_RULES)
        if any(row["rule"] == rule for row in rows)
    ]
    values = {(row["config"], row["rule"]): row for row in rows}
    positions = np.arange(len(rules))
    width = 0.8 / len(configs)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    for ax, (metric, label) in zip(axes.flat, COMPARE_METRICS.items()):
        for i, config in enumerate(configs):
            heights = [
                values[(config, rule)][metric] if (config, rule) in values else np.nan
                for rule in rules
            ]
            bars = ax.bar(
                positions + (i - (len(configs) - 1) / 2) * width,
                heights,
                width,
                label=config,
            )
            ax.bar_label(
                bars,
                labels=["" if np.isnan(h) else f"{h:.3f}" for h in heights],
                fontsize=6,
                padding=1,
            )
        ax.set_xticks(
            positions,
            ["all rules" if r == ALL_RULES else r.replace("_", "\n") for r in rules],
            fontsize=8,
        )
        ax.set_title(label, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        if metric in UNIT_RANGE_METRICS:
            ax.set_ylim(0, 1.05)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", title="Prompt config")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def sweep_argv(args: argparse.Namespace, config: str, out_path: Path) -> list[str]:
    """Arguments for `run_chunk_sweep` to run one config at the compared chunk variant"""
    size, overlap = compare_chunk_settings(args)
    argv = [
        "--corpus",
        args.corpus,
        "--split",
        args.split,
        "--out",
        str(out_path),
        "--config",
        config,
        "--chunk-sizes",
        str(size),
        "--overlap-ratios",
        str(overlap / size),
        "--repeats",
        str(args.repeats),
        "--max-workers",
        str(args.max_workers),
    ]
    for flag, value in (
        ("--limit", args.limit),
        ("--budget", args.budget),
        ("--max-completion-tokens", args.max_completion_tokens),
        ("--reasoning-effort", args.reasoning_effort),
    ):
        if value is not None:
            argv += [flag, str(value)]
    return argv


def compare_chunk_settings(args: argparse.Namespace) -> tuple[int, int]:
    """The chunk size and overlap every config is run at, so only the prompt differs"""
    fields = LLMTextRedactionConfigBase.model_fields
    size = args.chunk_size or fields["chunk_size"].default
    overlap = (
        args.chunk_overlap
        if args.chunk_overlap is not None
        else fields["chunk_overlap"].default
    )
    return size, overlap_for(size, overlap / size)


def cmd_compare(args: argparse.Namespace):
    """Run each prompt config over the corpus (unless --plot-only), then score the
    results and save a comparison plot and table. Runs are resumable, and recorded
    per document so the results can be re-scored without more LLM calls"""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size, overlap = compare_chunk_settings(args)
    documents = {d.id: d for d in load_corpus(args.corpus)}

    scored_by_config = {}
    for config in args.configs:
        results_path = out_dir / f"{config}.jsonl"
        if not args.plot_only:
            run_sweep(parse_sweep_args(sweep_argv(args, config, results_path)))
        if not results_path.exists():
            raise SystemExit(f"No results for '{config}' at {results_path}")
        with results_path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows = [
            r for r in rows if (r["chunk_size"], r["chunk_overlap"]) == (size, overlap)
        ]
        if not rows:
            raise SystemExit(
                f"{results_path} has no rows for chunk size {size}, overlap {overlap}"
            )
        scored_by_config[config] = score_rows(rows, documents, args.coverage_threshold)

    runs = {
        config: len({(s.repeat, s.doc_id) for s in scored if s.rule == ALL_RULES})
        for config, scored in scored_by_config.items()
    }
    if len(set(runs.values())) > 1:
        print(
            f"Warning: configs have different numbers of complete runs {runs}, so "
            "the comparison is not like for like. Re-run to finish them."
        )

    rows = comparison_rows(
        scored_by_config, args.hard_negative_weight, args.unlocatable_weight
    )
    table_path, plot_path = out_dir / "comparison.csv", out_dir / "comparison.png"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}
            for row in rows
        )
    plot_comparison(
        rows,
        plot_path,
        f"Prompt configs on the {args.split} split, chunk size {size}, overlap {overlap}",
    )

    print(" | ".join(["config", "rule", *COMPARE_METRICS]))
    for row in rows:
        print(
            " | ".join(
                [row["config"], row["rule"]]
                + [f"{row[metric]:.4f}" for metric in COMPARE_METRICS]
            )
        )
    print(f"Wrote {table_path} and {plot_path}")


def parse_args(argv=None) -> argparse.Namespace:
    """Define and parse the command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(COMPARE_CONFIGS),
        help="Redaction config names under src/config, in plot order",
    )
    parser.add_argument("--corpus", default="eval/data/prompt_corpus.jsonl")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "compare"),
        help="Where per-config results (<config>.jsonl), the table and plot go",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Score and plot the results already in --out-dir, with no LLM calls",
    )
    parser.add_argument("--chunk-size", type=int, help="Override the chunk size")
    parser.add_argument("--chunk-overlap", type=int, help="Override the overlap")
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument(
        "--reasoning-effort", choices=["minimal", "low", "medium", "high"]
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Only use the first N documents")
    parser.add_argument("--budget", type=float, help="Stop each config at this cost")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    parser.add_argument(
        "--hard-negative-weight", type=float, default=HARD_NEGATIVE_WEIGHT
    )
    parser.add_argument("--unlocatable-weight", type=float, default=UNLOCATABLE_WEIGHT)
    return parser.parse_args(argv)


def main(argv=None):
    """Command line entry point: run the configs, unless --plot-only, then plot them"""
    args = parse_args(argv)
    if not args.plot_only:
        check_azure_auth()
    cmd_compare(args)


if __name__ == "__main__":
    main()
