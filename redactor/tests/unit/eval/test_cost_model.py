import json
import math
from unittest.mock import patch

import numpy as np
import pytest
from eval import cost_model
from eval.cost_model import (
    ChunkCountModel,
    CostConfig,
    PipelineCostModel,
    TokenPrices,
    assign_folds,
    band_index,
    band_labels,
    build_observations,
    cacheable_prefix_tokens,
    config_of,
    cross_validate,
    error_metrics,
    fit_cost_model,
    fit_cost_models,
    format_validation,
    load_rows,
    main,
    select_configs,
    single_pass_mask,
)
from eval.summarise_chunk_sweep import estimate_cost_gbp

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


PRICES = TokenPrices(
    uncached_input=1.0, cached_input=0.1, cache_write=2.0, completion=3.0
)


def _rows_for(rule, prompt_tokens=1610, retry_every=0, **overrides):
    return [
        row | {"rule": rule, "prompt_tokens": prompt_tokens} | overrides
        for row in _rows(retry_every)
    ]


def _fitted(rule="names", prompt_tokens=1610):
    rows = _rows_for(rule, prompt_tokens)
    config, observations = next(iter(build_observations(rows).items()))
    return fit_cost_model(config, observations)


@pytest.mark.parametrize(
    ("prompt_tokens", "expected"),
    [(784, 0), (1023, 0), (1024, 1024), (1151, 1024), (1152, 1152), (1610, 1536)],
)
def test_cacheable_prefix_tokens(prompt_tokens, expected):
    assert cacheable_prefix_tokens(prompt_tokens) == expected


def test_token_prices__match_the_sweep_cost_estimate():
    """Pricing observed tokens reproduces `estimate_cost_gbp`, so predicted and
    measured costs are on the same footing"""
    row = {
        "model": "gpt-5.6-luna",
        "input_tokens": 10_000,
        "cached_input_tokens": 6_000,
        "cache_write_tokens": 500,
        "completion_tokens": 1_200,
    }

    prices = TokenPrices.for_model("gpt-5.6-luna")

    assert prices.cost(10_000, 6_000, 500, 1_200) == pytest.approx(
        estimate_cost_gbp(row)
    )


def test_token_prices__unknown_model():
    with pytest.raises(ValueError, match="No prices"):
        TokenPrices.for_model("not-a-model")


def test_predict_cost__caching_scenarios():
    """
    - Given a fitted model
    - When I price documents with no caching, prompt caching and the sweep's caching
    - Then cached tokens are 0, the cacheable prompt prefix per chunk (capped at the
      input), and the sweep's cached share, and each is billed at the cached rate
    """
    model = _fitted()
    docs = [900, 9000]

    none = model.predict_cost(docs, 3000, 120, "none", PRICES)
    prompt = model.predict_cost(docs, 3000, 120, "prompt", PRICES)
    sweep = model.predict_cost(docs, 3000, 120, "sweep", PRICES)

    assert none.cached_input_tokens.tolist() == [0, 0]
    assert none.cost_gbp == pytest.approx(
        none.input_tokens + 3 * none.completion_tokens
    )
    inflation = model.input_tokens.inflation_for(3000, 120)
    expected_cached = np.minimum(
        prompt.input_tokens, 1536 * prompt.chunk_count * inflation
    )
    assert prompt.cached_input_tokens == pytest.approx(expected_cached)
    assert prompt.cost_gbp == pytest.approx(
        (prompt.input_tokens - expected_cached)
        + 0.1 * expected_cached
        + 3 * prompt.completion_tokens
    )
    assert sweep.cached_input_tokens == pytest.approx(0.5 * sweep.input_tokens)
    assert sweep.cache_write_tokens.tolist() == [0, 0]
    assert (none.cost_gbp > prompt.cost_gbp).all()


def test_predict_cost__short_prompts_are_not_cached():
    model = _fitted(prompt_tokens=900)

    estimate = model.predict_cost([5000], 1000, 40, "prompt", PRICES)

    assert estimate.cached_input_tokens.tolist() == [0]


def _two_rule_rows():
    return _rows_for("names") + _rows_for("sensitive_info", prompt_tokens=900)


def test_pipeline_cost_model__sums_rules():
    pipeline = PipelineCostModel.build(
        fit_cost_models(build_observations(_two_rule_rows()))
    )

    cost = pipeline.predict_cost([900, 5000], (1000, 40), "none", PRICES)

    expected = sum(
        model.predict_cost([900, 5000], 1000, 40, "none", PRICES).cost_gbp
        for model in pipeline.models.values()
    )
    assert set(cost.rules) == {"names", "sensitive_info"}
    assert cost.cost_gbp == pytest.approx(expected)
    assert cost.in_range.tolist() == [True, True]
    assert pipeline.predict_cost([50000], (1000, 40)).in_range.tolist() == [False]


def test_pipeline_cost_model__chunk_setting_per_rule():
    pipeline = PipelineCostModel.build(
        fit_cost_models(build_observations(_two_rule_rows()))
    )

    cost = pipeline.predict_cost(
        [9000], {"names": (1000, 40), "sensitive_info": (3000, 120)}, "none", PRICES
    )

    assert (
        cost.rules["names"].chunk_count[0] > cost.rules["sensitive_info"].chunk_count[0]
    )
    assert cost.rules["sensitive_info"].cost_gbp == pytest.approx(
        pipeline.models["sensitive_info"]
        .predict_cost([9000], 3000, 120, "none", PRICES)
        .cost_gbp
    )


def test_pipeline_cost_model__build_needs_one_configuration_per_rule():
    rows = _rows_for("names") + _rows_for("names", max_completion_tokens=2500)
    models = fit_cost_models(build_observations(rows))

    with pytest.raises(ValueError, match="2 configurations"):
        PipelineCostModel.build(models)
    with pytest.raises(ValueError, match="No configuration"):
        PipelineCostModel.build(models, rules=["sensitive_info"])
    chosen = PipelineCostModel.build(models, max_completion_tokens=2500)
    assert chosen.models["names"].config.max_completion_tokens == 2500


def test_select_configs__filters_to_one_configuration_per_rule():
    rows = _two_rule_rows() + _rows_for("names", max_completion_tokens=2500)
    observations = build_observations(rows)

    selected = select_configs(observations, max_completion_tokens=4000)

    assert sorted(config.rule for config in selected) == ["names", "sensitive_info"]
    assert all(isinstance(config, CostConfig) for config in selected)


def test_assign_folds__keeps_documents_whole():
    doc_ids = [f"doc-{i}" for i in range(8)] * 4

    leave_one_out, n_folds = assign_folds(doc_ids, None)
    three_fold, three = assign_folds(doc_ids, 3)

    assert n_folds == 8
    assert sorted(leave_one_out.values()) == list(range(8))
    assert three == 3
    assert set(three_fold.values()) == {0, 1, 2}
    assert len(three_fold) == 8
    assert assign_folds(doc_ids, 20)[1] == 8
    with pytest.raises(ValueError, match="two documents"):
        assign_folds(["a", "a"], None)
    with pytest.raises(ValueError, match="two folds"):
        assign_folds(doc_ids, 1)


def test_error_metrics():
    """Rows with nothing observed, or no prediction, are ignored"""
    metrics = error_metrics([100, 200, 0, 50], [110, 180, 5, np.nan])

    assert metrics.rows == 2
    assert metrics.mape == pytest.approx(0.1)
    assert metrics.median_ape == pytest.approx(0.1)
    assert metrics.bias == pytest.approx(290 / 300 - 1)
    assert metrics.r_squared == pytest.approx(0.9)


def test_error_metrics__nothing_to_compare():
    metrics = error_metrics([0.0], [1.0])

    assert metrics.rows == 0
    assert math.isnan(metrics.mape)


def test_length_bands():
    edges = (1000, 3000, 10000)

    assert band_labels(edges) == ["<1000", "1000-3000", "3000-10000", ">=10000"]
    assert band_index(
        np.array([999, 1000, 2999, 3000, 10000, 50000]), edges
    ).tolist() == [
        0,
        1,
        1,
        2,
        3,
        3,
    ]


def test_cross_validate__holds_out_whole_documents():
    """Every model is fitted without exactly one document, and every document is
    held out once, so no document informs its own prediction"""
    selected = select_configs(build_observations(_two_rule_rows()))
    trained = []
    real_fit = cost_model.fit_cost_model

    def recording_fit(config, observations):
        trained.append(set(observations.doc_id))
        return real_fit(config, observations)

    with patch.object(cost_model, "fit_cost_model", side_effect=recording_fit):
        cross_validate(selected)

    all_docs = {f"doc-{i}" for i in range(len(DOC_TOKENS))}
    held_out = [all_docs - docs for docs in trained]
    assert all(len(docs) == 1 for docs in held_out)
    assert set().union(*held_out) == all_docs


def test_cross_validate__beats_the_baseline_on_short_documents():
    """
    - Given costs that are a fixed amount per chunk plus an amount per token
    - When I cross-validate
    - Then held-out cost is predicted closely, and better than a cost-per-token
      baseline, most of all for the shortest documents where the fixed cost dominates
    """
    selected = select_configs(build_observations(_two_rule_rows()))

    validation = cross_validate(selected)

    assert validation.documents == len(DOC_TOKENS)
    assert validation.folds == len(DOC_TOKENS)
    assert set(validation.rules) == {"names", "sensitive_info"}
    assert validation.rules["names"]["input_tokens"].mape < 0.05
    model, baseline = validation.pipeline["model"], validation.pipeline["baseline"]
    assert model.mape < 0.1
    assert model.mape < baseline.mape
    short = validation.pipeline_by_length["<1000"]
    assert short["model"].mape < short["baseline"].mape / 2
    assert validation.rules_by_length["names"]["<1000"]["model"].rows > 0


def test_cross_validate__requires_one_configuration_per_rule():
    rows = _rows_for("names") + _rows_for("names", max_completion_tokens=2500)

    with pytest.raises(ValueError, match="exactly one configuration"):
        cross_validate(build_observations(rows))


def test_format_validation():
    selected = select_configs(build_observations(_two_rule_rows()))

    text = format_validation(cross_validate(selected, folds=4))

    assert "Cross-validation: 8 documents, 4 folds" in text
    assert "Pipeline cost (sum of rules)" in text
    assert "<1000 model" in text
    assert "baseline_cost_gbp" in text


def test_main__validate(tmp_path, capsys):
    _write(tmp_path / "results.jsonl", _two_rule_rows())

    main(["--results", str(tmp_path / "results.jsonl"), "--validate"])

    assert "Cross-validation: 8 documents" in capsys.readouterr().out


def test_main__validate_needs_one_configuration_per_rule(tmp_path):
    rows = _rows_for("names") + _rows_for("names", max_completion_tokens=2500)
    _write(tmp_path / "results.jsonl", rows)
    path = str(tmp_path / "results.jsonl")

    with pytest.raises(SystemExit):
        main(["--results", path, "--validate"])
    main(["--results", path, "--validate", "--max-completion-tokens", "4000"])
