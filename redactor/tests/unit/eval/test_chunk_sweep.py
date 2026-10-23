import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import ClassVar
from unittest.mock import Mock, patch

import pytest
from eval import run_chunk_sweep, summarise_chunk_sweep
from eval.corpus import Document, Entity, write_corpus
from eval.scoring import ScoreCounts
from eval.summarise_chunk_sweep import (
    ALL_RULES,
    Guardrails,
    ScoredRun,
    bootstrap_f_beta_differences,
    estimate_cost_gbp,
    select_variant,
    summarise,
)

from src.types import LLMTextRedactionResult


def test_overlap_for():
    assert run_chunk_sweep.overlap_for(1500, 0.04) == 60
    assert run_chunk_sweep.overlap_for(10, 1.0) == 9


def test_estimate_cost_gbp__applies_cached_discount():
    row = {
        "model": "gpt-5.6-luna",
        "input_tokens": 1_000_000,
        "cached_input_tokens": 500_000,
        "completion_tokens": 0,
    }

    # 500k uncached at 16 and 500k cached at 2, in units of 1e-8 GBP per token
    assert estimate_cost_gbp(row) == pytest.approx(500_000 * 16e-8 + 500_000 * 2e-8)


def test_estimate_cost_gbp__reasoning_tokens_not_double_counted():
    """Reasoning tokens are already part of the completion tokens"""
    row = {
        "model": "gpt-5.6-luna",
        "input_tokens": 0,
        "completion_tokens": 1000,
        "reasoning_tokens": 900,
    }

    assert estimate_cost_gbp(row) == pytest.approx(1000 * 91e-8)


def test_estimate_cost_gbp__cache_writes_not_charged_as_input():
    row = {
        "model": "gpt-5.6-luna",
        "input_tokens": 1000,
        "cache_write_tokens": 400,
        "completion_tokens": 0,
    }

    assert estimate_cost_gbp(row) == pytest.approx(600 * 16e-8 + 400 * 19e-8)


def _run(variant, doc_id, tp, positives, cost, repeat=0, rule=ALL_RULES, fp=0):
    return ScoredRun(
        variant=variant,
        rule=rule,
        repeat=repeat,
        doc_id=doc_id,
        counts=ScoreCounts(
            positives=positives,
            true_positives=tp,
            false_negatives=positives - tp,
            correct_predictions=tp,
            false_positives=fp,
        ),
        cost_gbp=cost,
        doc_tokens=1000,
        chunk_count=1,
        failed_chunks=0,
        latency_s=1.0,
        outcomes={f"e{i}": i < tp for i in range(positives)},
    )


def _scored_runs():
    runs = []
    for doc in range(20):
        # Small chunks: perfect but expensive. Medium: near-perfect and cheap.
        # Large: cheapest but clearly misses entities.
        runs.append(_run((500, 20), f"d{doc}", 10, 10, cost=3.0))
        runs.append(_run((1500, 60), f"d{doc}", 10 if doc else 9, 10, cost=1.0))
        runs.append(_run((8000, 320), f"d{doc}", 5, 10, cost=0.5))
    return runs


def test_select_variant__picks_cheapest_not_significantly_worse():
    scored = _scored_runs()
    summaries = summarise(scored)
    intervals = bootstrap_f_beta_differences(scored, ALL_RULES, (500, 20))

    chosen, reasons = select_variant(summaries, intervals, Guardrails())

    assert chosen == (1500, 60)
    assert reasons[(8000, 320)] == ["significantly worse than best F-beta"]


def test_select_variant__applies_recall_floor():
    scored = _scored_runs()
    summaries = summarise(scored)
    intervals = bootstrap_f_beta_differences(scored, ALL_RULES, (500, 20))

    chosen, reasons = select_variant(
        summaries, intervals, Guardrails(recall_floors={ALL_RULES: 0.999})
    )

    assert chosen == (500, 20)
    assert any("recall" in r for r in reasons[(1500, 60)])


def test_summarise__consistency_across_repeats():
    runs = [
        _run((500, 20), "d0", tp=2, positives=2, cost=1, repeat=0),
        _run((500, 20), "d0", tp=1, positives=2, cost=1, repeat=1),
    ]

    summary = summarise(runs)[((500, 20), ALL_RULES)]

    assert summary.consistency == pytest.approx(0.5)
    assert summary.cost_per_1k_doc_tokens == pytest.approx(1.0)


class _OracleAnalyser:
    """Predicts every positive label that fits wholly inside a chunk"""

    labels: ClassVar[list[str]] = []

    def __init__(self, config):
        self.config = config

    def analyse_text(self, system_prompt, text_chunks):
        predictions = tuple(
            label
            for label in self.labels
            if any(label in chunk for chunk in text_chunks)
        )
        metadata = LLMTextRedactionResult.LLMResultMetadata(
            request_count=len(text_chunks),
            input_token_count=100 * len(text_chunks),
            completion_token_count=10 * len(text_chunks),
            total_cost=0.001 * len(text_chunks),
        )
        return LLMTextRedactionResult(
            rule_name="",
            run_metrics={},
            redaction_strings=predictions,
            metadata=metadata,
        )


def _sweep_document() -> Document:
    filler = "The traffic assessment considered peak hour flows on the A-road. " * 40
    return Document(
        id="doc-1",
        doc_type="relevant_representation",
        text=f"My neighbour Joan Whitcombe objects. {filler} Cllr Raj Patel agreed.",
        entities=(
            Entity("Joan Whitcombe", "names", "third_party", True),
            Entity("Cllr Raj Patel", "names", "councillor", False),
        ),
    )


def _sweep_args(tmp_path, *extra_args):
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(corpus, [_sweep_document()])
    _OracleAnalyser.labels = ["Joan Whitcombe"]
    return run_chunk_sweep.parse_args(
        [
            "--corpus",
            str(corpus),
            "--out",
            str(tmp_path / "results.jsonl"),
            "--rules",
            "names",
            "--chunk-sizes",
            "100",
            "1000",
            "--overlap-ratios",
            "0.04",
            "--repeats",
            "2",
            "--max-workers",
            "1",
            *extra_args,
        ]
    )


@contextmanager
def _in_process_sweep(worker_logs: bytes = b""):
    """Run the sweep with threads, since patches don't reach spawned worker processes"""
    mock_logging_util = Mock()
    mock_logging_util.return_value.get_log_bytes.return_value = worker_logs
    with (
        patch.object(run_chunk_sweep, "LLMTextAnalyser", _OracleAnalyser),
        patch.object(run_chunk_sweep, "ProcessPoolExecutor", ThreadPoolExecutor),
        patch.object(run_chunk_sweep, "LoggingUtil", mock_logging_util),
    ):
        yield


def test_sweep_and_summarise_end_to_end(tmp_path):
    """
    - Given a small corpus and an oracle LLM in place of the real one
    - When I run the sweep, run it again (to resume), then summarise
    - Then the rows are written once, and a recommendation is produced
    """
    args = _sweep_args(tmp_path)
    corpus = args.corpus
    results = tmp_path / "results.jsonl"

    with _in_process_sweep():
        run_chunk_sweep.run_sweep(args)
        run_chunk_sweep.run_sweep(args)

    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(rows) == 4
    small = next(r for r in rows if r["chunk_size"] == 100)
    large = next(r for r in rows if r["chunk_size"] == 1000)
    assert small["chunk_count"] > large["chunk_count"] == 1
    assert large["predictions"] == ["Joan Whitcombe"]

    out_dir = tmp_path / "summary"
    summarise_chunk_sweep.main(
        [
            "--corpus",
            str(corpus),
            "--results",
            str(results),
            "--out-dir",
            str(out_dir),
            "--bootstrap-iterations",
            "50",
        ]
    )

    recommendation = json.loads((out_dir / "recommendation.json").read_text())
    assert recommendation[ALL_RULES]["recommended_variant"] == [1000, 40]
    assert (out_dir / "summary.csv").exists()


def _read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_sweep__reasoning_effort_defaults_to_config(tmp_path):
    """Without --reasoning-effort the config's own setting is used and recorded"""
    args = _sweep_args(tmp_path)

    with _in_process_sweep():
        run_chunk_sweep.run_sweep(args)

    rows = _read_rows(tmp_path / "results.jsonl")
    assert len(rows) == 4
    assert {row["reasoning_effort"] for row in rows} == {None}


def test_sweep__reasoning_effort_override(tmp_path):
    args = _sweep_args(tmp_path, "--reasoning-effort", "low")

    with _in_process_sweep():
        run_chunk_sweep.run_sweep(args)

    rows = _read_rows(tmp_path / "results.jsonl")
    assert {row["reasoning_effort"] for row in rows} == {"low"}


def test_sweep__invalid_reasoning_effort(tmp_path):
    with pytest.raises(SystemExit):
        _sweep_args(tmp_path, "--reasoning-effort", "extreme")


def test_sweep__include_rationale(tmp_path):
    """With --include-rationale the flag is recorded and decisions are written"""
    args = _sweep_args(tmp_path, "--include-rationale")

    with _in_process_sweep():
        run_chunk_sweep.run_sweep(args)

    rows = _read_rows(tmp_path / "results.jsonl")
    assert {row["include_rationale"] for row in rows} == {True}
    assert all(row["decisions"] == [] and row["excluded"] == [] for row in rows)


def test_sweep__no_rationale_omits_decisions(tmp_path):
    args = _sweep_args(tmp_path)

    with _in_process_sweep():
        run_chunk_sweep.run_sweep(args)

    rows = _read_rows(tmp_path / "results.jsonl")
    assert {row["include_rationale"] for row in rows} == {False}
    assert not any("decisions" in row for row in rows)


def test_sweep__writes_worker_logs_to_log_file(tmp_path):
    """
    - Given the workers capture some redactor logs for each run
    - When the sweep finishes
    - Then every run's logs are written, labelled by run, to the default log file
    """
    args = _sweep_args(tmp_path)

    with _in_process_sweep(worker_logs=b"INFO: hello\n"):
        run_chunk_sweep.run_sweep(args)

    log = (tmp_path / "results.log").read_text()
    assert log.count("INFO: hello") == 4
    assert "doc=doc-1 size=100 overlap=4 rule=names failed=0" in log


def test_sweep__custom_log_file(tmp_path):
    log_path = tmp_path / "custom.log"
    args = _sweep_args(tmp_path, "--log-file", str(log_path))

    with _in_process_sweep(worker_logs=b"INFO: hello\n"):
        run_chunk_sweep.run_sweep(args)

    assert log_path.exists()
    assert not (tmp_path / "results.log").exists()


def test_sweep__stops_on_authentication_error_without_recording_row(tmp_path):
    """
    - Given the workers report an authentication failure
    - When the sweep runs
    - Then it stops, records no row for the failed run, and still writes the logs
    """
    args = _sweep_args(tmp_path)

    with (
        _in_process_sweep(worker_logs=b"openai.AuthenticationError: Error code: 401"),
        pytest.raises(RuntimeError, match="Authentication failed"),
    ):
        run_chunk_sweep.run_sweep(args)

    assert (tmp_path / "results.jsonl").read_text() == ""
    assert "AuthenticationError" in (tmp_path / "results.log").read_text()
