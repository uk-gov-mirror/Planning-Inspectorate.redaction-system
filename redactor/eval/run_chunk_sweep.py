"""Run LLM text redaction over a labelled corpus at several chunk sizes.

Each (repeat, document, chunk size, rule) combination is one row of JSONL output,
containing the raw predictions plus token usage, so scoring can be re-run with
`summarise_chunk_sweep` without calling the LLM again. Runs are resumable: rows already
present in the output file are skipped.

Example (run from the `redactor` directory, with Azure credentials available):

    python -m tests.eval.run_chunk_sweep \
        --corpus tests/eval/data/corpus.jsonl --split dev \
        --chunk-sizes 500 1000 1500 2500 4000 6000 8000 12000 \
        --repeats 2 --out tests/eval/results/coarse.jsonl --budget 50
"""

import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from tiktoken import get_encoding

from src.analysis.text import LLMTextAnalyser
from src.monitoring import LoggingUtil
from src.redaction.config import LLMTextRedactionConfig
from src.redaction.config_processor import ConfigProcessor

from .corpus import RULE_NAMES, load_corpus
from .scoring import boundary_split_entities


def load_rule_configs(
    config_name: str, rules: list[str]
) -> dict[str, LLMTextRedactionConfig]:
    """Load the named config and return its LLM rule configs, keyed by rule name

    :raises ValueError: If any requested rule is not in the config.
    """
    parsed = ConfigProcessor.validate_and_parse_redaction_config(
        ConfigProcessor.load_config(config_name)
    )
    configs = {
        c.name: c
        for c in parsed
        if type(c) is LLMTextRedactionConfig and c.name in rules
    }
    missing = set(rules) - set(configs)
    if missing:
        raise ValueError(f"Rules not found in config '{config_name}': {missing}")
    return configs


def overlap_for(chunk_size: int, overlap_ratio: float) -> int:
    """Chunk overlap in tokens for a ratio of the chunk size, kept below the size"""
    return min(round(chunk_size * overlap_ratio), chunk_size - 1)


def row_key(row: dict) -> tuple:
    """Identify a result row by the run it represents, used to skip finished work"""
    return (
        row["rule"],
        row["chunk_size"],
        row["chunk_overlap"],
        row["repeat"],
        row["doc_id"],
    )


def load_completed_keys(out_path: Path) -> set[tuple]:
    """Return the keys of rows already in the output file (empty if it is missing)"""
    if not out_path.exists():
        return set()
    with out_path.open(encoding="utf-8") as f:
        return {row_key(json.loads(line)) for line in f if line.strip()}


class AnalyserPool:
    """Caches one LLMTextAnalyser per rule"""

    def __init__(self):
        """Start with no cached analysers"""
        self._analysers: dict[str, LLMTextAnalyser] = {}

    def get(self, rule: str, config: LLMTextRedactionConfig) -> LLMTextAnalyser:
        """Return the rule's analyser, with its token and cost counters reset to zero"""
        analyser = self._analysers.get(rule)
        if analyser is None:
            analyser = LLMTextAnalyser(config)
            self._analysers[rule] = analyser
        analyser.input_token_count = 0
        analyser.completion_token_count = 0
        analyser.total_token_count = 0
        analyser.cached_input_token_count = 0
        analyser.cache_write_token_count = 0
        analyser.reasoning_token_count = 0
        analyser.total_cost = 0.0
        return analyser


# Per-process state, set by `init_worker` in each worker process
_worker: dict = {}


def init_worker(
    rule_configs: dict[str, LLMTextRedactionConfig],
    prompt_tokens: dict[str, int],
    config_name: str,
):
    """Give a worker process its own analyser pool and tokenizer"""
    _worker.update(
        rule_configs=rule_configs,
        prompt_tokens=prompt_tokens,
        config_name=config_name,
        pool=AnalyserPool(),
        encoding=get_encoding(next(iter(rule_configs.values())).token_encoding_name),
    )


def run_task(
    repeat: int, document, chunk_size: int, chunk_overlap: int, rule: str
) -> tuple[dict, str]:
    """Run one (repeat, document, chunk size, overlap, rule) run, returning its row and logs"""
    config = _worker["rule_configs"][rule].model_copy(
        update={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    )
    chunks = config.create_text_splitter().split_text(document.text)
    analyser = _worker["pool"].get(rule, config)
    started = time.perf_counter()
    result = analyser.analyse_text(config.create_system_prompt(), chunks)
    latency = time.perf_counter() - started
    metadata = result.metadata
    # Workers are separate processes, so hand their logs back to the parent
    logger = LoggingUtil()
    logs = logger.get_log_bytes().decode("utf-8")
    logger.clear_logs()
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config_name": _worker["config_name"],
        "model": config.model,
        "rule": rule,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "repeat": repeat,
        "doc_id": document.id,
        "split": document.split,
        "doc_tokens": len(_worker["encoding"].encode(document.text)),
        "prompt_tokens": _worker["prompt_tokens"][rule],
        "chunk_count": len(chunks),
        "max_completion_tokens": config.max_tokens,
        "reasoning_effort": config.reasoning_effort,
        "include_rationale": config.include_rationale,
        "predictions": list(result.redaction_strings),
        "boundary_split": sorted(boundary_split_entities(document.entities, chunks)),
        "input_tokens": metadata.input_token_count,
        "cached_input_tokens": metadata.cached_input_token_count,
        "cache_write_tokens": metadata.cache_write_token_count,
        "completion_tokens": metadata.completion_token_count,
        "reasoning_tokens": metadata.reasoning_token_count,
        "total_tokens": metadata.total_token_count,
        "reported_cost_gbp": metadata.total_cost,
        "failed_chunks": metadata.failed_chunk_count,
        "latency_s": round(latency, 3),
    }
    if config.include_rationale:
        row["decisions"] = list(result.redaction_decisions)
        row["excluded"] = list(result.excluded_decisions)
    return row, logs


def run_sweep(args: argparse.Namespace):
    """
    Run every rule over each corpus document at each chunk size, logging results.

    Each (rule, chunk size, repeat, document) run appends one JSON row (predictions,
    token counts, cost, latency) to the output file. Runs already in the file are
    skipped, so the sweep can be resumed, and it stops once the budget is spent.
    """
    documents = load_corpus(args.corpus, split=args.split)
    if args.limit:
        documents = documents[: args.limit]
    rule_configs = load_rule_configs(args.config, args.rules)
    if args.max_completion_tokens:
        rule_configs = {
            rule: config.model_copy(update={"max_tokens": args.max_completion_tokens})
            for rule, config in rule_configs.items()
        }
    if args.max_concurrent_requests:
        rule_configs = {
            rule: config.model_copy(
                update={"max_concurrent_requests": args.max_concurrent_requests}
            )
            for rule, config in rule_configs.items()
        }
    if args.reasoning_effort:
        rule_configs = {
            rule: config.model_copy(update={"reasoning_effort": args.reasoning_effort})
            for rule, config in rule_configs.items()
        }
    if args.include_rationale:
        rule_configs = {
            rule: config.model_copy(update={"include_rationale": True})
            for rule, config in rule_configs.items()
        }

    encoding = get_encoding(next(iter(rule_configs.values())).token_encoding_name)
    prompt_tokens = {
        rule: len(encoding.encode(config.create_system_prompt()))
        for rule, config in rule_configs.items()
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(out_path)

    print(
        f"{len(documents)} documents, rules={args.rules}, sizes={args.chunk_sizes}, "
        f"overlap ratios={args.overlap_ratios}"
    )
    print(f"System prompt tokens per rule: {prompt_tokens}")

    # Small sizes can round different ratios to the same overlap, so dedupe
    variants = sorted(
        {
            (size, overlap_for(size, ratio))
            for size in args.chunk_sizes
            for ratio in args.overlap_ratios
        }
    )

    # Interleave variants within each document so time-varying latency/throttling
    # affects every variant equally
    tasks = [
        (repeat, document, chunk_size, chunk_overlap, rule)
        for repeat in range(args.repeats)
        for document in documents
        for chunk_size, chunk_overlap in variants
        for rule in rule_configs
        if (rule, chunk_size, chunk_overlap, repeat, document.id) not in completed
    ]

    log_path = Path(args.log_file) if args.log_file else out_path.with_suffix(".log")
    log_sections: list[str] = []
    spent = 0.0
    pending = set()
    task_iter = iter(tasks)
    try:
        with (
            out_path.open("a", encoding="utf-8") as out,
            ProcessPoolExecutor(
                max_workers=args.max_workers,
                initializer=init_worker,
                initargs=(rule_configs, prompt_tokens, args.config),
            ) as executor,
        ):
            # Submit lazily so the budget can stop the sweep before all work is queued
            while True:
                while len(pending) < args.max_workers and not (
                    args.budget and spent >= args.budget
                ):
                    task = next(task_iter, None)
                    if task is None:
                        break
                    pending.add(executor.submit(run_task, *task))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row, logs = future.result()
                    log_sections.append(
                        f"=== repeat={row['repeat']} doc={row['doc_id']} "
                        f"size={row['chunk_size']} overlap={row['chunk_overlap']} "
                        f"rule={row['rule']} failed={row['failed_chunks']} ===\n{logs}"
                    )
                    # Auth failures aren't model results, so leave the run to be redone
                    if "AuthenticationError" in logs:
                        raise RuntimeError(
                            "Authentication failed (401), stopping. Not recording the "
                            "failed run, so rerunning will resume from it."
                        )
                    spent += row["reported_cost_gbp"]
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    print(
                        f"repeat={row['repeat']} doc={row['doc_id']} "
                        f"size={row['chunk_size']} overlap={row['chunk_overlap']} "
                        f"rule={row['rule']} chunks={row['chunk_count']} "
                        f"preds={len(row['predictions'])} "
                        f"failed={row['failed_chunks']} £{spent:.2f}"
                    )
    finally:
        # Written even if the sweep is interrupted, so failures can still be diagnosed
        log_path.write_text("\n".join(log_sections), encoding="utf-8")
        print(f"Wrote redactor logs to {log_path}")
    if args.budget and spent >= args.budget:
        print(f"Budget of £{args.budget:.2f} reached, stopping.")


def parse_args(argv=None) -> argparse.Namespace:
    """Define and parse the command line arguments for the sweep"""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True, help="Path to a JSONL corpus")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--out", required=True, help="JSONL file to append rows to")
    parser.add_argument(
        "--log-file", help="Where to write redactor logs. Defaults to --out with .log"
    )
    parser.add_argument("--config", default="default", help="Redaction config name")
    parser.add_argument("--rules", nargs="+", default=list(RULE_NAMES))
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[500, 1000, 1500, 2500, 4000, 6000, 8000, 12000],
        help="Chunk sizes in tokens",
    )
    parser.add_argument(
        "--overlap-ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.04, 0.1],
        help="Chunk overlaps to try, as fractions of the chunk size",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Only use the first N documents")
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        help="Override the configured max completion tokens. Keep fixed across a sweep",
    )
    parser.add_argument("--max-concurrent-requests", type=int)
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        help="Override the configured reasoning effort. Unset uses the config's value. "
        "Use a separate --out file per setting",
    )
    parser.add_argument(
        "--include-rationale",
        action="store_true",
        help="Ask for a reason, clause IDs and confidence per redaction, and write "
        "them to each row. This changes the prompt, so use a separate --out file",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of (repeat, document, chunk size, rule) runs in flight at once",
    )
    parser.add_argument("--budget", type=float, help="Stop after this cost in GBP")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_sweep(parse_args())
