import json
import math
from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest
from eval import cost_model
from eval.corpus import Document, write_corpus
from eval.cost_model import (
    ChunkCountModel,
    CostConfig,
    CostForecaster,
    DocumentResampler,
    PipelineCostModel,
    ProductionSettings,
    TokenPrices,
    assign_folds,
    band_index,
    band_labels,
    bootstrap_pipeline_models,
    build_observations,
    cacheable_prefix_tokens,
    compatibility_warnings,
    config_of,
    corpus_token_lengths,
    cross_validate,
    error_metrics,
    fit_cost_model,
    fit_cost_models,
    format_expected_cost,
    format_validation,
    load_rows,
    main,
    plot_cost_curve,
    plot_validation,
    production_settings,
    read_lengths,
    select_configs,
    single_pass_mask,
)
from eval.summarise_chunk_sweep import estimate_cost_gbp
from tiktoken import get_encoding

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


def _noisy_rows(seed, doc_count=40):
    """Documents with lognormal lengths, whose completion tokens scatter around the
    model by a lognormal factor, so real costs spread around the prediction"""
    rng = np.random.default_rng(seed)
    lengths = np.clip(rng.lognormal(np.log(3000), 1.0, doc_count), 400, 20000)
    rows = []
    for rule, prompt_tokens in (("names", 1610), ("sensitive_info", 900)):
        for index, tokens in enumerate(lengths.round()):
            for size, overlap in SETTINGS:
                row = _row(index, int(tokens), size, overlap, 0)
                row |= {"rule": rule, "prompt_tokens": prompt_tokens}
                row["completion_tokens"] *= math.exp(rng.normal(0, 0.3))
                rows.append(row)
    return rows


def _true_costs(rows, setting):
    """Document lengths and the real pipeline cost, without caching, at one setting"""
    prices = TokenPrices.for_model("gpt-5.6-luna")
    tokens, costs = {}, {}
    for row in rows:
        if (row["chunk_size"], row["chunk_overlap"]) == setting:
            tokens[row["doc_id"]] = row["doc_tokens"]
            costs[row["doc_id"]] = costs.get(row["doc_id"], 0) + prices.cost(
                row["input_tokens"], 0.0, 0.0, row["completion_tokens"]
            )
    ids = sorted(tokens)
    return np.array([tokens[i] for i in ids]), np.array([costs[i] for i in ids])


def _forecaster(rows=None, iterations=10, seed=0):
    selected = select_configs(build_observations(rows or _two_rule_rows()))
    return CostForecaster.fit(selected, iterations=iterations, seed=seed)


def test_document_resampler__keeps_documents_whole_and_aligned_across_rules():
    """Each rule gets the same drawn documents, with every row of each, so a rule's
    uncertainty is correlated with the others', as in a real pipeline"""
    selected = select_configs(build_observations(_two_rule_rows()))
    resampler = DocumentResampler(selected)

    sample = resampler.sample(np.random.default_rng(0))

    counts = [Counter(map(str, obs.doc_id)) for obs in sample.values()]
    rows_per_document = len(_rows()) // len(DOC_TOKENS)
    assert counts[0] == counts[1]
    assert sum(counts[0].values()) == len(_rows())
    assert all(n % rows_per_document == 0 for n in counts[0].values())
    again = resampler.sample(np.random.default_rng(0))
    assert all(np.array_equal(sample[c].doc_id, again[c].doc_id) for c in sample)


def test_bootstrap_pipeline_models__one_pipeline_per_iteration():
    selected = select_configs(build_observations(_two_rule_rows()))

    replicates = bootstrap_pipeline_models(selected, iterations=5, seed=1)

    assert len(replicates) == 5
    assert all(set(r.models) == {"names", "sensitive_info"} for r in replicates)


def test_cost_forecaster__fit_needs_an_iteration():
    selected = select_configs(build_observations(_two_rule_rows()))

    with pytest.raises(ValueError, match="at least one bootstrap"):
        CostForecaster.fit(selected, iterations=0)


def test_cost_forecaster__prediction_interval_covers_new_documents():
    """
    - Given costs that scatter around the model by a known relative amount
    - When I forecast a fresh set of documents from the same process
    - Then about the nominal share of their real costs fall inside the prediction
      interval, which is wider than the interval on the expected cost
    """
    forecaster = _forecaster(_noisy_rows(seed=1), iterations=60, seed=3)
    tokens, truth = _true_costs(_noisy_rows(seed=2, doc_count=300), (1000, 40))

    forecast = forecaster.predict_documents(tokens, (1000, 40), "none", level=0.9)

    inside = (truth >= forecast.pi_lower) & (truth <= forecast.pi_upper)
    assert 0.75 <= inside.mean() <= 0.99
    assert (forecast.ci_lower <= forecast.ci_upper).all()
    assert (forecast.pi_upper - forecast.pi_lower).mean() > (
        forecast.ci_upper - forecast.ci_lower
    ).mean()


def test_cost_forecaster__predict_documents_is_reproducible():
    forecaster = _forecaster()

    first = forecaster.predict_documents([900, 5000], (1000, 40), "none", seed=5)
    second = forecaster.predict_documents([900, 5000], (1000, 40), "none", seed=5)

    assert np.array_equal(first.pi_lower, second.pi_lower)
    assert first.level == 0.95
    assert first.cost_gbp.shape == (2,)


def test_cost_forecaster__flags_lengths_outside_the_fitted_range():
    forecaster = _forecaster()

    forecast = forecaster.predict_documents([100, 900, 50000], (1000, 40), "none")
    expected = forecaster.expected_cost(
        np.array([100, 900, 900, 50000]), (1000, 40), "none"
    )

    assert forecaster.pipeline.doc_token_range == (400, 20000)
    assert forecast.in_range.tolist() == [False, True, False]
    assert expected.out_of_range_share == pytest.approx(0.5)


def test_cost_forecaster__draws_errors_from_documents_of_a_similar_length():
    """Short documents' errors come from short documents. A band with too few held-out
    errors falls back to all of them"""
    forecaster = _forecaster().model_copy(
        update={
            "ratios": np.array([0.5] * 30 + [2.0] * 30 + [7.0] * 5),
            "ratio_tokens": np.array([500] * 30 + [5000] * 30 + [50000] * 5),
        }
    )
    rng = np.random.default_rng(0)

    drawn = forecaster._draw_ratios(np.array([600, 6000, 60000, 60000]), rng)

    assert drawn[:2].tolist() == [0.5, 2.0]
    assert set(drawn[2:]) <= {0.5, 2.0, 7.0}


def test_cost_forecaster__expected_cost_averages_over_lengths():
    """
    - Given lengths where most documents are short and some are long
    - When I forecast the expected cost
    - Then the mean cost exceeds the cost of a document of the mean length, because
      every document pays a fixed per-chunk cost, and the intervals bracket the mean
    """
    forecaster = _forecaster(iterations=30)
    lengths = np.array([400] * 70 + [5000] * 30)

    expected = forecaster.expected_cost(
        lengths, (3000, 120), "none", level=0.9, batch_size=50
    )

    assert expected.mean_cost_gbp > expected.cost_at_mean_length_gbp * 1.03
    assert (
        expected.mean_interval[0] < expected.mean_cost_gbp < expected.mean_interval[1]
    )
    assert expected.total_cost_gbp == pytest.approx(50 * expected.mean_cost_gbp)
    assert expected.total_interval[0] < expected.total_cost_gbp
    assert expected.total_cost_gbp < expected.total_interval[1]
    assert (expected.documents, expected.batch_size) == (100, 50)
    assert expected.out_of_range_share == 0
    assert sum(expected.by_rule_gbp.values()) == pytest.approx(expected.mean_cost_gbp)


def test_cost_forecaster__expected_cost_needs_lengths():
    with pytest.raises(ValueError, match="at least one document length"):
        _forecaster().expected_cost(np.array([]), (1000, 40))


def test_compatibility_warnings():
    pipeline = PipelineCostModel(models={"names": _fitted()})

    def settings(**overrides):
        base = {
            "chunk_size": 1000,
            "chunk_overlap": 40,
            "model": "gpt-5.6-luna",
            "max_tokens": 4000,
            "reasoning_effort": None,
            "include_rationale": False,
            "prompt_tokens": 1610,
        }
        return ProductionSettings(**(base | overrides))

    assert compatibility_warnings(pipeline, {"names": settings()}) == []
    warnings = compatibility_warnings(
        pipeline, {"names": settings(max_tokens=2000, prompt_tokens=1700)}
    )
    assert len(warnings) == 2
    assert "max_completion_tokens=4000 but production uses 2000" in warnings[0]
    assert "prompt_tokens=1610 but production uses 1700" in warnings[1]
    assert compatibility_warnings(pipeline, {}) == [
        "names: not in the production config"
    ]


def test_production_settings__reads_the_redaction_config():
    settings = production_settings("default", ["names"])["names"]

    assert settings.chunk_size > settings.chunk_overlap >= 0
    assert settings.prompt_tokens > 0
    with pytest.raises(ValueError, match="not_a_rule"):
        production_settings("default", ["not_a_rule"])


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("[100, 250.5]", [100, 250.5]),
        ("100\n250\n300", [100, 250, 300]),
        ("100 250", [100, 250]),
        ("640", [640]),
    ],
)
def test_read_lengths(tmp_path, contents, expected):
    (tmp_path / "lengths.txt").write_text(contents, encoding="utf-8")

    assert read_lengths(tmp_path / "lengths.txt").tolist() == expected


def test_read_lengths__converts_characters_to_tokens(tmp_path):
    (tmp_path / "chars.txt").write_text("[4000, 8000]", encoding="utf-8")

    lengths = read_lengths(tmp_path / "chars.txt", characters=True, chars_per_token=4)

    assert lengths.tolist() == [1000, 2000]


@pytest.mark.parametrize("contents", ["[]", "[100, 0]", "[-5]"])
def test_read_lengths__rejects_empty_or_non_positive(tmp_path, contents):
    (tmp_path / "lengths.txt").write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="positive document length"):
        read_lengths(tmp_path / "lengths.txt")


def test_corpus_token_lengths__counts_tokens_for_the_split(tmp_path):
    texts = {"a": "The cable route crosses two fields.", "b": "Short.", "c": "x " * 80}
    documents = [
        Document(id=name, doc_type="test", text=text, split=split)
        for (name, text), split in zip(texts.items(), ["dev", "dev", "test"])
    ]
    write_corpus(tmp_path / "corpus.jsonl", documents)
    encoding = get_encoding("o200k_base")

    dev = corpus_token_lengths(tmp_path / "corpus.jsonl", "dev", "gpt-5.6-luna")
    everything = corpus_token_lengths(tmp_path / "corpus.jsonl", None, "gpt-5.6-luna")

    assert dev.tolist() == [len(encoding.encode(texts[n])) for n in ("a", "b")]
    assert len(everything) == 3


def test_format_expected_cost():
    forecaster = _forecaster()
    lengths = np.array([400, 900, 3500, 9000])
    expected = forecaster.expected_cost(lengths, (1000, 40), "none")

    text = format_expected_cost(expected, lengths, forecaster.pipeline.doc_token_range)

    assert "Expected cost per document (caching: none, 95% intervals)" in text
    assert "4 documents; tokens min 400, median 2,200, mean 3,450, max 9,000" in text
    assert "fitted range 400-20,000 tokens; 0.0% of documents are outside it" in text
    assert "against the mean cost" in text
    assert "next 4 documents" in text
    assert "by rule: names" in text


def test_plot_validation__writes_a_plot_for_the_pipeline_and_each_rule(tmp_path):
    validation = cross_validate(select_configs(build_observations(_two_rule_rows())))

    paths = plot_validation(validation, tmp_path / "plots")

    assert sorted(path.name for path in paths) == [
        "cost_validation_names.png",
        "cost_validation_pipeline.png",
        "cost_validation_sensitive_info.png",
    ]
    assert all(path.stat().st_size > 0 for path in paths)


@pytest.mark.parametrize("caching", ["none", "prompt"])
def test_plot_cost_curve(tmp_path, caching):
    forecaster = _forecaster(iterations=5)

    path = plot_cost_curve(
        forecaster, (1000, 40), tmp_path / "plots" / "curve.png", caching
    )

    assert path.stat().st_size > 0


def test_main__predict_from_a_lengths_file(tmp_path, capsys):
    _write(tmp_path / "results.jsonl", _two_rule_rows())
    (tmp_path / "lengths.txt").write_text("[400, 900, 3000, 12000]", encoding="utf-8")

    main(
        [
            *["--results", str(tmp_path / "results.jsonl"), "--predict"],
            *["--lengths", str(tmp_path / "lengths.txt"), "--skip-config"],
            *["--chunk-size", "1000", "--chunk-overlap", "40"],
            *["--iterations", "10", "--caching", "none"],
            *["--plots-dir", str(tmp_path / "plots")],
        ]
    )

    out = capsys.readouterr().out
    assert "Expected cost per document (caching: none" in out
    assert "by rule: names" in out
    assert "Warning" not in out
    assert (tmp_path / "plots" / "cost_curve.png").exists()
    assert (tmp_path / "plots" / "cost_validation_pipeline.png").exists()


def test_main__predict_from_a_corpus(tmp_path, capsys):
    _write(tmp_path / "results.jsonl", _two_rule_rows())
    documents = [
        Document(id=f"d{i}", doc_type="test", text="word " * (200 * i), split="dev")
        for i in range(1, 4)
    ]
    write_corpus(tmp_path / "corpus.jsonl", documents)

    main(
        [
            *["--results", str(tmp_path / "results.jsonl"), "--predict"],
            *["--corpus", str(tmp_path / "corpus.jsonl"), "--split", "dev"],
            *["--skip-config", "--chunk-size", "1000", "--chunk-overlap", "40"],
            *["--iterations", "5"],
        ]
    )

    assert "3 documents" in capsys.readouterr().out


def test_main__predict_uses_the_production_chunk_settings(tmp_path, capsys):
    _write(tmp_path / "results.jsonl", _two_rule_rows())
    (tmp_path / "lengths.txt").write_text("[900, 3000]", encoding="utf-8")

    main(
        [
            *["--results", str(tmp_path / "results.jsonl"), "--predict"],
            *["--lengths", str(tmp_path / "lengths.txt"), "--iterations", "5"],
        ]
    )

    assert "Expected cost per document (caching: prompt" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (
            ["--skip-config", "--chunk-size", "1000", "--chunk-overlap", "40"],
            "--lengths",
        ),
        (["--lengths", "{lengths}", "--skip-config"], "--chunk-size"),
        (["--lengths", "{lengths}", "--chunk-size", "1000"], "both --chunk-size"),
    ],
)
def test_main__predict_needs_lengths_and_chunk_settings(
    tmp_path, capsys, extra_args, message
):
    _write(tmp_path / "results.jsonl", _two_rule_rows())
    (tmp_path / "lengths.txt").write_text("[900]", encoding="utf-8")
    extra = [
        arg.replace("{lengths}", str(tmp_path / "lengths.txt")) for arg in extra_args
    ]

    with pytest.raises(SystemExit):
        main(
            [
                *["--results", str(tmp_path / "results.jsonl"), "--predict"],
                *["--iterations", "5", *extra],
            ]
        )

    assert message in capsys.readouterr().err
