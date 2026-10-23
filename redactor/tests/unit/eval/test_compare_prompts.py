import csv
import json
from unittest.mock import patch

import pytest
from azure.core.exceptions import ClientAuthenticationError
from eval import azure_auth, compare_prompts
from eval.corpus import Document, Entity, write_corpus
from eval.scoring import ScoreCounts
from eval.summarise_chunk_sweep import ScoredRun


def _scored_run(rule, doc_id, tp, positives, fp=0, hits=0, hard_negatives=0, cost=1.0):
    return ScoredRun(
        variant=(1500, 60),
        rule=rule,
        repeat=0,
        doc_id=doc_id,
        counts=ScoreCounts(
            positives=positives,
            true_positives=tp,
            false_negatives=positives - tp,
            predictions=tp + fp,
            correct_predictions=tp,
            false_positives=fp,
            hard_negatives=hard_negatives,
            hard_negative_hits=hits,
        ),
        cost_gbp=cost,
        doc_tokens=1000,
        chunk_count=1,
        failed_chunks=0,
        latency_s=1.0,
    )


def test_comparison_rows__metrics_match_pooled_counts():
    """
    - Given a perfect config and one that misses an entity and redacts a hard negative
    - When I compute the comparison rows
    - Then each metric is that of the counts pooled over the config's documents
    """
    scored = {
        "good": [_scored_run("names", f"d{i}", 2, 2, cost=1.0) for i in range(2)],
        "bad": [
            _scored_run("names", "d0", 1, 2, fp=1, hits=1, hard_negatives=1, cost=3.0),
            _scored_run("names", "d1", 2, 2, hard_negatives=1, cost=3.0),
        ],
    }

    rows = {
        row["config"]: row for row in compare_prompts.comparison_rows(scored, 0.5, 0.25)
    }

    assert rows["good"]["f2"] == 1.0
    assert rows["good"]["objective"] == 1.0
    assert rows["good"]["cost_per_1k_doc_tokens"] == pytest.approx(1.0)
    bad = rows["bad"]
    assert bad["recall"] == pytest.approx(3 / 4)
    assert bad["precision"] == pytest.approx(3 / 4)
    assert bad["hard_negative_hit_rate"] == pytest.approx(1 / 2)
    assert bad["unlocatable_rate"] == 0
    assert bad["cost_per_1k_doc_tokens"] == pytest.approx(3.0)
    assert bad["objective"] == pytest.approx(bad["f2"] - 0.5 * 0.5)
    assert set(compare_prompts.COMPARE_METRICS) <= bad.keys()


def _result_rows(config, predictions_by_rule, chunk_size=1500, chunk_overlap=60):
    """Sweep rows for one document in which every rule gets the given predictions"""
    return [
        {
            "config_name": config,
            "model": "gpt-5.6-luna",
            "rule": rule,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "repeat": 0,
            "doc_id": "doc-1",
            "doc_tokens": 1000,
            "chunk_count": 1,
            "predictions": predictions_by_rule.get(rule, []),
            "boundary_split": [],
            "input_tokens": 1000,
            "completion_tokens": 100,
            "failed_chunks": 0,
            "latency_s": 1.0,
        }
        for rule in compare_prompts.RULE_NAMES
    ]


def _compare_args(tmp_path, *extra_args):
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(
        corpus,
        [
            Document(
                id="doc-1",
                doc_type="relevant_representation",
                text="My neighbour Joan Whitcombe objects. Cllr Raj Patel agreed.",
                entities=(
                    Entity("Joan Whitcombe", "names", "third_party", True),
                    Entity("Cllr Raj Patel", "names", "councillor", False),
                ),
                split="test",
            )
        ],
    )
    return compare_prompts.parse_args(
        [
            "--corpus",
            str(corpus),
            "--out-dir",
            str(tmp_path / "compare"),
            "--configs",
            "basic",
            "default",
            *extra_args,
        ]
    )


def _write_results(tmp_path, config, rows):
    out_dir = tmp_path / "compare"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{config}.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )


def test_compare__plot_only_scores_results_and_saves_table_and_plot(tmp_path, capsys):
    _write_results(
        tmp_path, "basic", _result_rows("basic", {"names": ["Cllr Raj Patel"]})
    )
    _write_results(
        tmp_path, "default", _result_rows("default", {"names": ["Joan Whitcombe"]})
    )
    args = _compare_args(tmp_path, "--plot-only")

    with patch.object(compare_prompts, "run_sweep") as mock_run_sweep:
        compare_prompts.cmd_compare(args)

    mock_run_sweep.assert_not_called()
    out_dir = tmp_path / "compare"
    assert (out_dir / "comparison.png").stat().st_size > 0
    with (out_dir / "comparison.csv").open(encoding="utf-8") as f:
        table = {(r["config"], r["rule"]): r for r in csv.DictReader(f)}
    assert float(table["default", "names"]["f2"]) == 1.0
    assert float(table["basic", "names"]["f2"]) == 0.0
    assert float(table["basic", "names"]["hard_negative_hit_rate"]) == 1.0
    assert ("default", "__all__") in table
    assert "Wrote" in capsys.readouterr().out


def test_compare__runs_each_config_at_the_same_chunk_settings(tmp_path):
    for config in ("basic", "default"):
        _write_results(tmp_path, config, _result_rows(config, {}))
    args = _compare_args(
        tmp_path, "--chunk-size", "1500", "--chunk-overlap", "60", "--budget", "5"
    )

    with patch.object(compare_prompts, "run_sweep") as mock_run_sweep:
        compare_prompts.cmd_compare(args)

    sweeps = [call.args[0] for call in mock_run_sweep.call_args_list]
    assert [s.config for s in sweeps] == ["basic", "default"]
    for sweep in sweeps:
        assert (sweep.chunk_sizes, sweep.overlap_ratios) == ([1500], [0.04])
        assert sweep.budget == 5
        assert sweep.split == "test"


def test_compare__ignores_rows_of_other_chunk_variants(tmp_path):
    rows = _result_rows("basic", {"names": ["Joan Whitcombe"]})
    other = _result_rows("basic", {}, chunk_size=500, chunk_overlap=20)
    _write_results(tmp_path, "basic", rows + other)
    _write_results(tmp_path, "default", rows)

    compare_prompts.cmd_compare(_compare_args(tmp_path, "--plot-only"))

    with (tmp_path / "compare" / "comparison.csv").open(encoding="utf-8") as f:
        table = {(r["config"], r["rule"]): r for r in csv.DictReader(f)}
    assert table["basic", "names"]["runs"] == "1"


def test_compare__warns_if_configs_have_different_numbers_of_runs(tmp_path, capsys):
    rows = _result_rows("basic", {})
    _write_results(tmp_path, "basic", rows + [dict(r, repeat=1) for r in rows])
    _write_results(tmp_path, "default", rows)

    compare_prompts.cmd_compare(_compare_args(tmp_path, "--plot-only"))

    assert "not like for like" in capsys.readouterr().out


def test_compare__plot_only_needs_existing_results(tmp_path):
    with pytest.raises(SystemExit, match="No results for 'basic'"):
        compare_prompts.cmd_compare(_compare_args(tmp_path, "--plot-only"))


def test_main__plot_only_makes_no_azure_calls():
    with (
        patch.object(azure_auth, "ChainedTokenCredential") as mock_credential,
        patch.object(compare_prompts, "cmd_compare") as mock_compare,
    ):
        compare_prompts.main(["--plot-only"])

    mock_credential.assert_not_called()
    mock_compare.assert_called_once()


def test_main__checks_azure_auth_when_running():
    error = ClientAuthenticationError(message="no credentials")

    with (
        patch.object(azure_auth, "ChainedTokenCredential") as mock_credential,
        patch.object(compare_prompts, "cmd_compare") as mock_compare,
    ):
        mock_credential.return_value.get_token.side_effect = error
        with pytest.raises(SystemExit, match="Azure authentication failed"):
            compare_prompts.main([])

    mock_compare.assert_not_called()
