import json
import math

import numpy as np
import pytest
from eval.cost_model import (
    ChunkCountModel,
    build_observations,
    config_of,
    fit_cost_model,
    load_rows,
    main,
    single_pass_mask,
)

DOC_TOKENS = [400, 900, 1800, 3500, 6000, 9000, 14000, 20000]
SETTINGS = [(1000, 40), (3000, 120)]
SCALE = 1.02
PER_CHUNK_INPUT = 1700
OVERLAP_FACTOR = 0.95
PER_CHUNK_COMPLETION = 400
COMPLETION_PER_TOKEN = 0.1


def _chunks(doc_tokens, size, overlap):
    return max(1, math.ceil(SCALE * (doc_tokens - overlap) / (size - overlap) - 1e-9))


def _row(doc_index, doc_tokens, size, overlap, repeat, retried=False, **overrides):
    n = _chunks(doc_tokens, size, overlap)
    noise = 30 * ((doc_index + repeat) % 3 - 1)
    input_tokens = PER_CHUNK_INPUT * n + doc_tokens + OVERLAP_FACTOR * (n - 1) * overlap
    completion = PER_CHUNK_COMPLETION * n + COMPLETION_PER_TOKEN * doc_tokens + noise
    factor = 2 if retried else 1
    row = {
        "rule": "names",
        "model": "gpt-5.6-luna",
        "reasoning_effort": None,
        "include_rationale": False,
        "max_completion_tokens": 4000,
        "prompt_tokens": 1610,
        "doc_id": f"doc-{doc_index}",
        "repeat": repeat,
        "doc_tokens": doc_tokens,
        "chunk_size": size,
        "chunk_overlap": overlap,
        "chunk_count": n,
        "input_tokens": input_tokens * factor,
        "cached_input_tokens": 0.5 * input_tokens,
        "cache_write_tokens": 0,
        "completion_tokens": completion * factor,
        "reasoning_tokens": 0.9 * completion,
        "failed_chunks": 0,
    }
    return row | overrides


def _rows(retry_every=0):
    """Rows for every document, setting and two repeats; every Nth row in the larger
    chunk setting is a retry"""
    rows = []
    for size, overlap in SETTINGS:
        for repeat in range(2):
            for index, doc_tokens in enumerate(DOC_TOKENS):
                retried = bool(
                    retry_every and size == 3000 and (index + repeat) % retry_every == 0
                )
                rows.append(_row(index, doc_tokens, size, overlap, repeat, retried))
    return rows


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_rows__drops_duplicate_runs(tmp_path):
    rows = _rows()
    _write(tmp_path / "a.jsonl", rows)
    _write(tmp_path / "b.jsonl", rows[:5])

    loaded = load_rows([tmp_path / "a.jsonl", tmp_path / "b.jsonl"])

    assert len(loaded) == len(rows)


def test_build_observations__groups_by_configuration():
    """A different prompt is a different configuration; older rows lacking the newer
    fields fall back to defaults"""
    rows = _rows()
    old_row = _row(0, 900, 1000, 40, 0, prompt_tokens=1500)
    del old_row["include_rationale"], old_row["reasoning_effort"]

    observations = build_observations([*rows, old_row])

    assert len(observations) == 2
    old_config = config_of(old_row)
    assert old_config.include_rationale is False
    assert old_config.reasoning_effort is None
    assert len(observations[old_config]) == 1
    assert len(observations[config_of(rows[0])]) == len(rows)


def test_chunk_count_model__fits_observed_counts():
    """The smallest scale reproducing every observed count is chosen; with few
    documents it need not equal the generating scale"""
    observations = next(iter(build_observations(_rows()).values()))

    model = ChunkCountModel.fit(observations)

    assert model.exact_rate == 1.0
    assert 1.0 < model.scale <= SCALE
    assert model.predict(400, 1000, 40) == 1
    assert np.array_equal(
        model.predict(
            observations.doc_tokens,
            observations.chunk_size,
            observations.chunk_overlap,
        ),
        observations.chunk_count,
    )


def test_single_pass_mask__finds_retried_rows():
    rows = _rows(retry_every=3)
    observations = next(iter(build_observations(rows).values()))

    mask = single_pass_mask(observations)

    retried = np.array([r["input_tokens"] > r["cached_input_tokens"] * 3 for r in rows])
    assert np.array_equal(mask, ~retried)


def test_single_pass_mask__excludes_failed_chunks():
    rows = _rows()
    rows[0]["failed_chunks"] = 1
    observations = next(iter(build_observations(rows).values()))

    assert not single_pass_mask(observations)[0]


def test_fit_cost_model__recovers_coefficients_and_retry_factor():
    """
    - Given rows from known coefficients, with some retried rows billed twice
    - When I fit the cost model
    - Then the structural coefficients come from the single-pass rows, and the retries
      appear only in the factor for the chunk setting that had them
    """
    rows = _rows(retry_every=3)
    config, observations = next(iter(build_observations(rows).items()))

    model = fit_cost_model(config, observations)

    assert model.input_tokens.coefficients == pytest.approx(
        (PER_CHUNK_INPUT, 1.0, OVERLAP_FACTOR), rel=1e-6
    )
    assert model.input_tokens.r_squared == pytest.approx(1.0)
    assert model.completion_tokens.coefficients == pytest.approx(
        (PER_CHUNK_COMPLETION, COMPLETION_PER_TOKEN), rel=0.05
    )
    assert model.input_tokens.inflation[(1000, 40)] == pytest.approx(1.0)
    assert model.input_tokens.inflation[(3000, 120)] > 1.2
    assert model.documents == len(DOC_TOKENS)
    assert model.doc_token_range == (400, 20000)
    assert model.sweep_cached_share < 0.5


def test_cost_model__predict_tokens():
    """Predictions use the swept setting's factor, fall back to the pooled factor for
    an unswept setting, and flag lengths outside the fitted range"""
    config, observations = next(iter(build_observations(_rows(retry_every=3)).items()))
    model = fit_cost_model(config, observations)

    swept = model.predict_tokens([900, 50000], 3000, 120)
    unswept = model.predict_tokens([900], 2000, 80)

    n = _chunks(900, 3000, 120)
    single_pass_input = PER_CHUNK_INPUT * n + 900 + OVERLAP_FACTOR * (n - 1) * 120
    factor = model.input_tokens.inflation[(3000, 120)]
    assert swept["chunk_count"][0] == n
    assert swept["input_tokens"][0] == pytest.approx(single_pass_input * factor)
    assert swept["per_chunk_input_tokens"][0] == pytest.approx(
        PER_CHUNK_INPUT * n * factor
    )
    assert swept["in_range"].tolist() == [True, False]
    assert unswept["input_tokens"][0] == pytest.approx(
        model.input_tokens.predict_single_pass(np.array([[n, 900, (n - 1) * 80]]))[0]
        * model.input_tokens.pooled_inflation
    )


def test_main__prints_a_summary_per_configuration(tmp_path, capsys):
    _write(tmp_path / "results.jsonl", _rows(retry_every=3))

    main(["--results", str(tmp_path / "results.jsonl")])

    out = capsys.readouterr().out
    assert out.startswith("names (gpt-5.6-luna")
    assert "per_chunk=1700.000" in out
