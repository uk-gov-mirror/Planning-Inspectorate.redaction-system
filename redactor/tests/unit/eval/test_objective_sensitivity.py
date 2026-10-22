import numpy as np
import pytest
from eval.objective_sensitivity import (
    kendall_tau,
    macro_objective,
    objective_grid,
    pooled_objective,
    rank_stability,
)
from eval.scoring import (
    HARD_NEGATIVE_WEIGHT,
    UNLOCATABLE_WEIGHT,
    ScoreCounts,
    objective_score,
)
from eval.summarise_chunk_sweep import ScoredRun


def _run(variant, doc_id, rule="names", **counts):
    return ScoredRun(
        variant=variant,
        rule=rule,
        repeat=0,
        doc_id=doc_id,
        counts=ScoreCounts(**counts),
        cost_gbp=0.0,
        doc_tokens=0,
        chunk_count=1,
        failed_chunks=0,
        latency_s=0.0,
    )


def test_objective_score__penalties_and_clipping():
    assert objective_score(0.9, 0.2, 0.4, 0.5, 0.25) == pytest.approx(0.9 - 0.1 - 0.1)
    assert objective_score(0.1, 1.0, 1.0, 0.5, 0.25) == 0.0
    assert objective_score(1.0, 0.0, 0.0) == 1.0
    assert objective_score(0.8, 0.2, 0.4) == objective_score(
        0.8, 0.2, 0.4, HARD_NEGATIVE_WEIGHT, UNLOCATABLE_WEIGHT
    )


def test_unlocatable_rate():
    assert (
        ScoreCounts(predictions=4, unlocatable_predictions=1).unlocatable_rate == 0.25
    )
    assert ScoreCounts().unlocatable_rate == 0.0


def test_kendall_tau():
    a = np.array([1.0, 2.0, 3.0])

    assert kendall_tau(a, a) == pytest.approx(1.0)
    assert kendall_tau(a, a[::-1]) == pytest.approx(-1.0)
    assert np.isnan(kendall_tau(a, np.array([1.0, 1.0, 1.0])))


def _counts(**overrides):
    base = {
        "positives": 10,
        "true_positives": 9,
        "false_negatives": 1,
        "predictions": 10,
        "correct_predictions": 9,
        "false_positives": 1,
        "hard_negatives": 10,
    }
    return base | overrides


def test_pooled_and_macro_objective_differ_by_aggregation():
    """Pooling weights documents by label count, macro averaging weights them equally"""
    runs = [
        _run(
            "v", "big", **_counts(positives=100, true_positives=100, false_negatives=0)
        ),
        _run("v", "small", **_counts(positives=1, true_positives=0, false_negatives=1)),
    ]

    assert pooled_objective(runs, (0.0, 0.0)) > macro_objective(runs, (0.0, 0.0))


def test_rank_stability__penalty_changes_ranking():
    """Variant b has a better F2 but redacts more hard negatives, so a large enough
    hard-negative weight makes a the winner"""
    runs = [
        _run("a", "d1", **_counts(true_positives=8, false_negatives=2)),
        _run(
            "b",
            "d1",
            **_counts(true_positives=9, false_negatives=1, hard_negative_hits=5),
        ),
    ]

    variants, grid = objective_grid(runs, "names", "pooled", [0.0, 0.5, 2.0], [0.0])
    stability = rank_stability(variants, grid, (0.5, 0.0))

    assert variants == ["a", "b"]
    assert stability["default_best"] == "a"
    assert stability["top_counts"] == {"b": 1, "a": 2}
    assert grid[(0.0, 0.0)].argmax() == 1
    assert grid[(2.0, 0.0)].argmax() == 0
