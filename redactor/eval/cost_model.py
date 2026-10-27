"""Predict the LLM token usage of redacting a document from its length.

Fits, per rule and LLM configuration, a small mechanistic model on `run_chunk_sweep`
results:

- chunk count: ``max(1, ceil(scale * (T - O) / (C - O)))`` for document tokens T, chunk
  size C and overlap O
- input tokens: ``a*n + b*T + c*(n - 1)*O``, i.e. a fixed cost per chunk (the system
  prompt and message overhead), the document itself and the duplicated overlap
- completion tokens (including reasoning): ``a*n + b*T``

Structural coefficients are fitted on single-pass rows only. Retries (e.g. after a
truncated response) are billed too, so each chunk setting gets an empirical factor,
total observed / total single-pass prediction, applied on top. The factor also absorbs
any misfit specific to that setting, so it can be slightly below 1.

Pricing prices those token predictions under an explicit caching scenario and sums
the rules into a per-document pipeline cost. `cross_validate` holds out whole
documents to check the model, and compares it with a cost-per-token baseline.

Example (run from the `redactor` directory):

    python -m eval.cost_model --results eval/results/optimised_sweep.jsonl --validate
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.text import LLMTextAnalyser

ChunkSetting = tuple[int, int]
"""(chunk_size, chunk_overlap)"""

RETRY_TOLERANCE = 0.1
"""Rows using more than this fraction above the single-pass input fit are retries"""


class CostConfig(BaseModel):
    """Everything that changes token usage apart from the document and chunking"""

    model_config = ConfigDict(frozen=True)

    rule: str
    model: str
    reasoning_effort: str | None
    include_rationale: bool
    max_completion_tokens: int | None
    prompt_tokens: int


class CostObservations(BaseModel):
    """One configuration's sweep rows, as parallel arrays"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_id: np.ndarray
    repeat: np.ndarray
    doc_tokens: np.ndarray
    chunk_size: np.ndarray
    chunk_overlap: np.ndarray
    chunk_count: np.ndarray
    input_tokens: np.ndarray
    cached_input_tokens: np.ndarray
    cache_write_tokens: np.ndarray
    completion_tokens: np.ndarray
    reasoning_tokens: np.ndarray
    failed_chunks: np.ndarray

    def __len__(self) -> int:
        return len(self.doc_id)

    def select(self, rows: np.ndarray) -> "CostObservations":
        """The observations in the rows picked by a boolean mask"""
        return CostObservations(
            **{name: getattr(self, name)[rows] for name in type(self).model_fields}
        )

    @property
    def chunk_settings(self) -> list[ChunkSetting]:
        """Distinct (chunk size, overlap) pairs, sorted"""
        return sorted(
            {(int(c), int(o)) for c, o in zip(self.chunk_size, self.chunk_overlap)}
        )


def config_of(row: dict) -> CostConfig:
    """The configuration a sweep row was run with (older rows lack some fields)"""
    return CostConfig(
        rule=row["rule"],
        model=row["model"],
        reasoning_effort=row.get("reasoning_effort"),
        include_rationale=bool(row.get("include_rationale", False)),
        max_completion_tokens=row.get("max_completion_tokens"),
        prompt_tokens=row["prompt_tokens"],
    )


def load_rows(paths: list[Path | str]) -> list[dict]:
    """Read sweep rows from JSONL files, dropping exact repeats of the same run

    Result files are sometimes copied or merged, so a run can appear more than once.
    """
    rows, seen = [], set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    config_of(row),
                    row["chunk_size"],
                    row["chunk_overlap"],
                    row["repeat"],
                    row["doc_id"],
                )
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    return rows


def build_observations(rows: list[dict]) -> dict[CostConfig, CostObservations]:
    """Group rows by configuration into arrays ready for fitting"""
    grouped: dict[CostConfig, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[config_of(row)].append(row)

    def column(group: list[dict], key: str, dtype=float) -> np.ndarray:
        return np.array([row.get(key, 0) for row in group], dtype=dtype)

    return {
        config: CostObservations(
            doc_id=np.array([row["doc_id"] for row in group], dtype=object),
            repeat=column(group, "repeat", int),
            doc_tokens=column(group, "doc_tokens"),
            chunk_size=column(group, "chunk_size"),
            chunk_overlap=column(group, "chunk_overlap"),
            chunk_count=column(group, "chunk_count"),
            input_tokens=column(group, "input_tokens"),
            cached_input_tokens=column(group, "cached_input_tokens"),
            cache_write_tokens=column(group, "cache_write_tokens"),
            completion_tokens=column(group, "completion_tokens"),
            reasoning_tokens=column(group, "reasoning_tokens"),
            failed_chunks=column(group, "failed_chunks", int),
        )
        for config, group in grouped.items()
    }


def ideal_chunk_ratio(
    doc_tokens: np.ndarray, chunk_size: np.ndarray, chunk_overlap: np.ndarray
) -> np.ndarray:
    """Chunks needed if every chunk were filled exactly, before rounding up"""
    return (np.asarray(doc_tokens) - chunk_overlap) / (
        np.asarray(chunk_size) - chunk_overlap
    )


class ChunkCountModel(BaseModel):
    """Chunk count from document length, scaled because the splitter breaks at
    separators and so rarely fills a chunk completely"""

    model_config = ConfigDict(frozen=True)

    scale: float
    exact_rate: float
    """Share of fitted rows whose chunk count was predicted exactly"""

    def predict(
        self,
        doc_tokens: np.ndarray | float,
        chunk_size: np.ndarray | float,
        chunk_overlap: np.ndarray | float,
    ) -> np.ndarray:
        ratio = ideal_chunk_ratio(doc_tokens, chunk_size, chunk_overlap)
        # Subtract a little so float error can't push an exact fit up a chunk
        return np.maximum(1, np.ceil(self.scale * ratio - 1e-9))

    @classmethod
    def fit(
        cls,
        observations: CostObservations,
        scales: np.ndarray | None = None,
    ) -> "ChunkCountModel":
        """Choose the scale that predicts the observed chunk counts exactly most
        often, preferring the smallest such scale"""
        if scales is None:
            scales = np.round(np.arange(1.0, 1.5001, 0.005), 3)
        best = None
        for scale in scales:
            predicted = cls(scale=float(scale), exact_rate=0.0).predict(
                observations.doc_tokens,
                observations.chunk_size,
                observations.chunk_overlap,
            )
            rate = float(np.mean(predicted == observations.chunk_count))
            if best is None or rate > best.exact_rate:
                best = cls(scale=float(scale), exact_rate=rate)
        return best


class TokenComponent(BaseModel):
    """Linear model of one token component, with retry inflation per chunk setting"""

    model_config = ConfigDict(frozen=True)

    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    r_squared: float
    """Fit quality on the single-pass rows used to estimate the coefficients"""
    single_pass_rows: int
    inflation: dict[ChunkSetting, float] = Field(default_factory=dict)
    pooled_inflation: float = 1.0
    """Used for chunk settings that were not swept"""

    def inflation_for(self, chunk_size: int, chunk_overlap: int) -> float:
        return self.inflation.get((chunk_size, chunk_overlap), self.pooled_inflation)

    def predict_single_pass(self, features: np.ndarray) -> np.ndarray:
        return features @ np.array(self.coefficients)

    def predict(
        self, features: np.ndarray, chunk_size: np.ndarray, chunk_overlap: np.ndarray
    ) -> np.ndarray:
        """Expected tokens including retries"""
        inflation = np.array(
            [
                self.inflation_for(int(c), int(o))
                for c, o in zip(np.atleast_1d(chunk_size), np.atleast_1d(chunk_overlap))
            ]
        )
        return self.predict_single_pass(features) * inflation


def input_features(
    chunk_count: np.ndarray, doc_tokens: np.ndarray, chunk_overlap: np.ndarray
) -> np.ndarray:
    """Per-chunk fixed cost, the document, and the overlap sent twice"""
    chunk_count = np.atleast_1d(np.asarray(chunk_count, dtype=float))
    return np.column_stack(
        [
            chunk_count,
            np.broadcast_to(doc_tokens, chunk_count.shape),
            (chunk_count - 1) * np.broadcast_to(chunk_overlap, chunk_count.shape),
        ]
    )


def completion_features(chunk_count: np.ndarray, doc_tokens: np.ndarray) -> np.ndarray:
    """Per-chunk base output plus output that grows with the text analysed"""
    chunk_count = np.atleast_1d(np.asarray(chunk_count, dtype=float))
    return np.column_stack(
        [chunk_count, np.broadcast_to(doc_tokens, chunk_count.shape)]
    )


INPUT_FEATURES = ("per_chunk", "doc_tokens", "overlap_tokens")
COMPLETION_FEATURES = ("per_chunk", "doc_tokens")


def _least_squares(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(features, target, rcond=None)
    return coefficients


def _r_squared(target: np.ndarray, predicted: np.ndarray) -> float:
    total = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - float(np.sum((target - predicted) ** 2)) / total if total else 1.0


def _inflation(
    observations: CostObservations, observed: np.ndarray, predicted: np.ndarray
) -> tuple[dict[ChunkSetting, float], float]:
    """Total observed over total single-pass prediction, per chunk setting and pooled"""
    by_setting = {}
    for size, overlap in observations.chunk_settings:
        in_setting = (observations.chunk_size == size) & (
            observations.chunk_overlap == overlap
        )
        by_setting[(size, overlap)] = float(
            observed[in_setting].sum() / predicted[in_setting].sum()
        )
    return by_setting, float(observed.sum() / predicted.sum())


def single_pass_mask(
    observations: CostObservations, tolerance: float = RETRY_TOLERANCE
) -> np.ndarray:
    """Rows with no retries or failed chunks, found by trimming input-token outliers

    Input tokens are near-deterministic for a single pass, so a retry shows up as a
    large excess over a fit to the remaining rows.
    """
    features = input_features(
        observations.chunk_count, observations.doc_tokens, observations.chunk_overlap
    )
    mask = observations.failed_chunks == 0
    for _ in range(3):
        coefficients = _least_squares(features[mask], observations.input_tokens[mask])
        ratio = observations.input_tokens / (features @ coefficients)
        new_mask = (observations.failed_chunks == 0) & (ratio <= 1 + tolerance)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return mask


def fit_component(
    observations: CostObservations,
    features: np.ndarray,
    target: np.ndarray,
    names: tuple[str, ...],
    single_pass: np.ndarray,
) -> TokenComponent:
    """Fit coefficients on single-pass rows, then inflation on every row"""
    coefficients = _least_squares(features[single_pass], target[single_pass])
    predicted = features @ coefficients
    inflation, pooled = _inflation(observations, target, predicted)
    return TokenComponent(
        feature_names=names,
        coefficients=tuple(float(c) for c in coefficients),
        r_squared=_r_squared(target[single_pass], predicted[single_pass]),
        single_pass_rows=int(single_pass.sum()),
        inflation=inflation,
        pooled_inflation=pooled,
    )


CachingScenario = Literal["none", "prompt", "sweep"]
"""How much input is billed at the cached rate:

- none: nothing is cached, an upper bound
- prompt: the cacheable prefix of the system prompt on every chunk, as when production
  traffic keeps the cache warm. Cache writes are ignored
- sweep: the cached and cache-write shares seen in the sweep. This reproduces the
  sweep's own cost but overstates production caching, since the sweep resends
  identical chunks across repeats and settings
"""

MIN_CACHE_PREFIX = 1024
"""Shortest prompt prefix that Azure OpenAI caches"""
CACHE_INCREMENT = 128
"""Cached prefixes grow in steps of this many tokens above the minimum"""


def cacheable_prefix_tokens(prompt_tokens: int) -> int:
    """Tokens of a system prompt that can be served from the prompt cache"""
    if prompt_tokens < MIN_CACHE_PREFIX:
        return 0
    steps = (prompt_tokens - MIN_CACHE_PREFIX) // CACHE_INCREMENT
    return MIN_CACHE_PREFIX + CACHE_INCREMENT * steps


class TokenPrices(BaseModel):
    """GBP per token"""

    model_config = ConfigDict(frozen=True)

    uncached_input: float
    cached_input: float
    cache_write: float
    completion: float

    @classmethod
    def for_model(cls, model: str) -> "TokenPrices":
        """Rates from `LLMTextAnalyser.OPENAI_MODELS`, which are in 1e-8 GBP per token"""
        rates = LLMTextAnalyser.OPENAI_MODELS.get(model)
        if rates is None:
            raise ValueError(f"No prices for model '{model}'")
        scale = 1e-8
        return cls(
            uncached_input=rates["input_cost"] * scale,
            cached_input=rates["cached_input_cost"] * scale,
            cache_write=rates["cache_write_cost"] * scale,
            completion=rates["output_cost"] * scale,
        )

    def cost(
        self,
        input_tokens: np.ndarray | float,
        cached_input_tokens: np.ndarray | float,
        cache_write_tokens: np.ndarray | float,
        completion_tokens: np.ndarray | float,
    ) -> np.ndarray | float:
        """GBP for tokens, where cached and written tokens are part of the input.
        Completion tokens include reasoning tokens"""
        uncached = input_tokens - cached_input_tokens - cache_write_tokens
        return (
            uncached * self.uncached_input
            + cached_input_tokens * self.cached_input
            + cache_write_tokens * self.cache_write
            + completion_tokens * self.completion
        )


class CostEstimate(BaseModel):
    """Predicted tokens and cost per document for one rule"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    caching: CachingScenario
    doc_tokens: np.ndarray
    chunk_count: np.ndarray
    input_tokens: np.ndarray
    cached_input_tokens: np.ndarray
    cache_write_tokens: np.ndarray
    completion_tokens: np.ndarray
    cost_gbp: np.ndarray
    in_range: np.ndarray
    """Whether each document's length is inside the range the model was fitted on"""


class CostModel(BaseModel):
    """Token usage model for one rule and configuration"""

    model_config = ConfigDict(frozen=True)

    config: CostConfig
    chunk_count: ChunkCountModel
    input_tokens: TokenComponent
    completion_tokens: TokenComponent
    doc_token_range: tuple[float, float]
    rows: int
    documents: int
    sweep_cached_share: float
    """Cached share of input in the sweep. Inflated by identical requests repeated
    across repeats and settings, so not a production estimate"""
    sweep_cache_write_share: float = 0.0
    """Share of input in the sweep billed as cache writes"""

    def predict_tokens(
        self,
        doc_tokens: np.ndarray | float,
        chunk_size: int,
        chunk_overlap: int,
    ) -> dict[str, np.ndarray]:
        """Expected tokens per document, including retries, for one chunk setting"""
        doc_tokens = np.atleast_1d(np.asarray(doc_tokens, dtype=float))
        size = np.full(doc_tokens.shape, chunk_size)
        overlap = np.full(doc_tokens.shape, chunk_overlap)
        chunks = self.chunk_count.predict(doc_tokens, size, overlap)
        low, high = self.doc_token_range
        return {
            "chunk_count": chunks,
            "input_tokens": self.input_tokens.predict(
                input_features(chunks, doc_tokens, overlap), size, overlap
            ),
            "completion_tokens": self.completion_tokens.predict(
                completion_features(chunks, doc_tokens), size, overlap
            ),
            "in_range": (doc_tokens >= low) & (doc_tokens <= high),
        }

    def predict_cost(
        self,
        doc_tokens: np.ndarray | float,
        chunk_size: int,
        chunk_overlap: int,
        caching: CachingScenario = "prompt",
        prices: TokenPrices | None = None,
    ) -> CostEstimate:
        """Expected tokens and GBP cost per document for one chunk setting"""
        tokens = self.predict_tokens(doc_tokens, chunk_size, chunk_overlap)
        prices = prices or TokenPrices.for_model(self.config.model)
        input_tokens = tokens["input_tokens"]
        cached = np.zeros_like(input_tokens)
        written = np.zeros_like(input_tokens)
        if caching == "prompt":
            inflation = self.input_tokens.inflation_for(chunk_size, chunk_overlap)
            prefix = cacheable_prefix_tokens(self.config.prompt_tokens)
            cached = np.minimum(
                input_tokens, prefix * tokens["chunk_count"] * inflation
            )
        elif caching == "sweep":
            cached = self.sweep_cached_share * input_tokens
            written = self.sweep_cache_write_share * input_tokens
        return CostEstimate(
            caching=caching,
            doc_tokens=np.atleast_1d(np.asarray(doc_tokens, dtype=float)),
            chunk_count=tokens["chunk_count"],
            input_tokens=input_tokens,
            cached_input_tokens=cached,
            cache_write_tokens=written,
            completion_tokens=tokens["completion_tokens"],
            cost_gbp=prices.cost(
                input_tokens, cached, written, tokens["completion_tokens"]
            ),
            in_range=tokens["in_range"],
        )


def fit_cost_model(config: CostConfig, observations: CostObservations) -> CostModel:
    """Fit the chunk-count, input and completion models for one configuration"""
    single_pass = single_pass_mask(observations)
    input_component = fit_component(
        observations,
        input_features(
            observations.chunk_count,
            observations.doc_tokens,
            observations.chunk_overlap,
        ),
        observations.input_tokens,
        INPUT_FEATURES,
        single_pass,
    )
    completion_component = fit_component(
        observations,
        completion_features(observations.chunk_count, observations.doc_tokens),
        observations.completion_tokens,
        COMPLETION_FEATURES,
        single_pass,
    )
    return CostModel(
        config=config,
        chunk_count=ChunkCountModel.fit(observations),
        input_tokens=input_component,
        completion_tokens=completion_component,
        doc_token_range=(
            float(observations.doc_tokens.min()),
            float(observations.doc_tokens.max()),
        ),
        rows=len(observations),
        documents=len(set(observations.doc_id)),
        sweep_cached_share=float(
            observations.cached_input_tokens.sum() / observations.input_tokens.sum()
        ),
        sweep_cache_write_share=float(
            observations.cache_write_tokens.sum() / observations.input_tokens.sum()
        ),
    )


def fit_cost_models(
    observations: dict[CostConfig, CostObservations],
) -> dict[CostConfig, CostModel]:
    return {config: fit_cost_model(config, obs) for config, obs in observations.items()}


def _config_label(config: CostConfig) -> str:
    return (
        f"{config.model}, reasoning={config.reasoning_effort}, "
        f"rationale={config.include_rationale}, "
        f"max_completion={config.max_completion_tokens}, "
        f"prompt={config.prompt_tokens}"
    )


def _only_match(candidates: dict[CostConfig, object], rule: str, filters: dict):
    """The one configuration for a rule that matches the filters

    :raises ValueError: If there is none, or more than one.
    """
    matches = [
        config
        for config in candidates
        if config.rule == rule
        and all(getattr(config, key) == value for key, value in filters.items())
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No configuration for rule '{rule}' matches {filters}")
    options = "; ".join(_config_label(config) for config in matches)
    raise ValueError(
        f"{len(matches)} configurations for rule '{rule}' match {filters}: {options}. "
        "Add filters or use fewer result files"
    )


def select_configs(
    observations: dict[CostConfig, CostObservations],
    rules: Sequence[str] | None = None,
    **filters,
) -> dict[CostConfig, CostObservations]:
    """One configuration per rule (every rule present by default), by CostConfig field"""
    rules = rules or sorted({config.rule for config in observations})
    chosen = (_only_match(observations, rule, filters) for rule in rules)
    return {config: observations[config] for config in chosen}


class PipelineCost(BaseModel):
    """Predicted cost per document for every rule, and in total"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rules: dict[str, CostEstimate]

    def _sum(self, field_name: str) -> np.ndarray:
        return sum(getattr(estimate, field_name) for estimate in self.rules.values())

    @property
    def cost_gbp(self) -> np.ndarray:
        return self._sum("cost_gbp")

    @property
    def input_tokens(self) -> np.ndarray:
        return self._sum("input_tokens")

    @property
    def completion_tokens(self) -> np.ndarray:
        return self._sum("completion_tokens")

    @property
    def in_range(self) -> np.ndarray:
        return np.logical_and.reduce([e.in_range for e in self.rules.values()])


class PipelineCostModel(BaseModel):
    """Cost models for each rule of one redaction pipeline. Every rule reads the whole
    document, so a document's cost is the sum over rules"""

    model_config = ConfigDict(frozen=True)

    models: dict[str, CostModel]

    @classmethod
    def build(
        cls,
        models: dict[CostConfig, CostModel],
        rules: Sequence[str] | None = None,
        **filters,
    ) -> "PipelineCostModel":
        """Pick one fitted model per rule (every rule present by default)"""
        rules = rules or sorted({config.rule for config in models})
        return cls(
            models={rule: models[_only_match(models, rule, filters)] for rule in rules}
        )

    def predict_cost(
        self,
        doc_tokens: np.ndarray | float,
        chunk_settings: ChunkSetting | dict[str, ChunkSetting],
        caching: CachingScenario = "prompt",
        prices: TokenPrices | None = None,
    ) -> PipelineCost:
        """Expected cost per document. Give one (chunk size, overlap) for every rule,
        or a dict of them by rule"""
        estimates = {}
        for rule, model in self.models.items():
            size, overlap = (
                chunk_settings[rule]
                if isinstance(chunk_settings, dict)
                else chunk_settings
            )
            estimates[rule] = model.predict_cost(
                doc_tokens, size, overlap, caching, prices
            )
        return PipelineCost(rules=estimates)


DEFAULT_LENGTH_EDGES = (1000, 3000, 10000)
"""Document token counts that split documents into length bands for validation"""


def band_labels(edges: Sequence[int]) -> list[str]:
    inner = [f"{low}-{high}" for low, high in pairwise(edges)]
    return [f"<{edges[0]}", *inner, f">={edges[-1]}"]


def band_index(doc_tokens: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    """Index into `band_labels` of each document length"""
    return np.searchsorted(edges, doc_tokens, side="right")


class ErrorMetrics(BaseModel):
    rows: int
    mape: float
    """Mean absolute percentage error"""
    median_ape: float
    bias: float
    """Total predicted over total observed, minus 1"""
    r_squared: float
    """Out-of-sample. Negative if worse than predicting the mean"""


def error_metrics(observed: np.ndarray, predicted: np.ndarray) -> ErrorMetrics:
    """Prediction error, ignoring rows with no prediction or nothing observed"""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(predicted) & (observed > 0)
    if not valid.any():
        nan = float("nan")
        return ErrorMetrics(rows=0, mape=nan, median_ape=nan, bias=nan, r_squared=nan)
    observed, predicted = observed[valid], predicted[valid]
    ape = np.abs(predicted - observed) / observed
    return ErrorMetrics(
        rows=int(valid.sum()),
        mape=float(ape.mean()),
        median_ape=float(np.median(ape)),
        bias=float(predicted.sum() / observed.sum() - 1),
        r_squared=_r_squared(observed, predicted),
    )


class Validation(BaseModel):
    documents: int
    folds: int
    rules: dict[str, dict[str, ErrorMetrics]]
    """By rule, then target: input_tokens, completion_tokens, cost_gbp (the model) and
    baseline_cost_gbp. Costs are priced without caching, which is an assumption
    rather than something the model learns"""
    rules_by_length: dict[str, dict[str, dict[str, ErrorMetrics]]]
    """Cost error by rule, then document length band, then model or baseline"""
    pipeline: dict[str, ErrorMetrics]
    """Cost summed over rules, for the model and the baseline"""
    pipeline_by_length: dict[str, dict[str, ErrorMetrics]]


def assign_folds(
    doc_ids: Sequence[str], folds: int | None
) -> tuple[dict[str, int], int]:
    """Assign whole documents to folds, so a document's repeats never straddle them

    :param folds: Number of folds, or None to leave out one document at a time.
    """
    ordered = sorted({str(doc_id) for doc_id in doc_ids})
    if len(ordered) < 2:
        raise ValueError("Cross-validation needs at least two documents")
    n_folds = len(ordered) if folds is None else min(folds, len(ordered))
    if n_folds < 2:
        raise ValueError("Cross-validation needs at least two folds")
    return {doc_id: i % n_folds for i, doc_id in enumerate(ordered)}, n_folds


def _predict_rows(
    model: CostModel, observations: CostObservations
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted input and completion tokens for each observed row"""
    input_tokens = np.zeros(len(observations))
    completion_tokens = np.zeros(len(observations))
    for size, overlap in observations.chunk_settings:
        rows = (observations.chunk_size == size) & (
            observations.chunk_overlap == overlap
        )
        tokens = model.predict_tokens(observations.doc_tokens[rows], size, overlap)
        input_tokens[rows] = tokens["input_tokens"]
        completion_tokens[rows] = tokens["completion_tokens"]
    return input_tokens, completion_tokens


def _baseline_cost(
    train: CostObservations, train_cost: np.ndarray, test: CostObservations
) -> np.ndarray:
    """Naive cost: the training cost per document token at the same chunk setting"""
    pooled = train_cost.sum() / train.doc_tokens.sum()
    ratios = {}
    for size, overlap in train.chunk_settings:
        rows = (train.chunk_size == size) & (train.chunk_overlap == overlap)
        ratios[(size, overlap)] = train_cost[rows].sum() / train.doc_tokens[rows].sum()
    return np.array(
        [
            ratios.get((int(size), int(overlap)), pooled) * tokens
            for size, overlap, tokens in zip(
                test.chunk_size, test.chunk_overlap, test.doc_tokens
            )
        ]
    )


def _cross_validate_rule(
    config: CostConfig,
    observations: CostObservations,
    fold_of: dict[str, int],
    n_folds: int,
    prices: TokenPrices,
) -> dict[str, np.ndarray]:
    """Held-out predictions for every row, NaN where no model could be fitted"""
    count = len(observations)
    row_fold = np.array([fold_of[str(doc_id)] for doc_id in observations.doc_id])
    observed_cost = prices.cost(
        observations.input_tokens, 0.0, 0.0, observations.completion_tokens
    )
    predicted_input = np.full(count, np.nan)
    predicted_completion = np.full(count, np.nan)
    baseline = np.full(count, np.nan)
    for fold in range(n_folds):
        test_rows = row_fold == fold
        if not test_rows.any() or test_rows.all():
            continue
        train, test = observations.select(~test_rows), observations.select(test_rows)
        model = fit_cost_model(config, train)
        predicted_input[test_rows], predicted_completion[test_rows] = _predict_rows(
            model, test
        )
        baseline[test_rows] = _baseline_cost(train, observed_cost[~test_rows], test)
    return {
        "observed_cost": observed_cost,
        "predicted_input": predicted_input,
        "predicted_completion": predicted_completion,
        "predicted_cost": prices.cost(predicted_input, 0.0, 0.0, predicted_completion),
        "baseline_cost": baseline,
    }


def _by_length(
    doc_tokens: np.ndarray,
    observed: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
    edges: Sequence[int],
) -> dict[str, dict[str, ErrorMetrics]]:
    bands = band_index(doc_tokens, edges)
    result = {}
    for index, label in enumerate(band_labels(edges)):
        rows = bands == index
        if rows.any():
            result[label] = {
                "model": error_metrics(observed[rows], model[rows]),
                "baseline": error_metrics(observed[rows], baseline[rows]),
            }
    return result


def cross_validate(
    selected: dict[CostConfig, CostObservations],
    folds: int | None = None,
    length_edges: Sequence[int] = DEFAULT_LENGTH_EDGES,
) -> Validation:
    """
    Predict every row from a model fitted without its document.

    Whole documents are held out, so repeats of a document never inform its own
    prediction. Each held-out row is predicted from its document length and chunk
    setting alone, as a new document would be. The baseline is the training cost per
    document token at the same chunk setting.

    :param selected: Observations for exactly one configuration per rule.
    :param folds: Number of folds, or None to leave out one document at a time.
    :raises ValueError: If two configurations share a rule, or there are fewer than two
    documents.
    """
    rules = [config.rule for config in selected]
    if len(set(rules)) != len(rules):
        raise ValueError("Give exactly one configuration per rule")
    fold_of, n_folds = assign_folds(
        [doc_id for obs in selected.values() for doc_id in obs.doc_id], folds
    )

    per_rule, rule_metrics, rule_lengths = {}, {}, {}
    for config, observations in selected.items():
        prices = TokenPrices.for_model(config.model)
        result = _cross_validate_rule(config, observations, fold_of, n_folds, prices)
        per_rule[config.rule] = (observations, result)
        rule_metrics[config.rule] = {
            "input_tokens": error_metrics(
                observations.input_tokens, result["predicted_input"]
            ),
            "completion_tokens": error_metrics(
                observations.completion_tokens, result["predicted_completion"]
            ),
            "cost_gbp": error_metrics(
                result["observed_cost"], result["predicted_cost"]
            ),
            "baseline_cost_gbp": error_metrics(
                result["observed_cost"], result["baseline_cost"]
            ),
        }
        rule_lengths[config.rule] = _by_length(
            observations.doc_tokens,
            result["observed_cost"],
            result["predicted_cost"],
            result["baseline_cost"],
            length_edges,
        )

    # Sum the rules for each document run, keeping only runs where every rule has a
    # held-out prediction
    totals: dict[tuple, list[float]] = {}
    for observations, result in per_rule.values():
        for i in range(len(observations)):
            if not np.isfinite(result["predicted_cost"][i]):
                continue
            key = (
                str(observations.doc_id[i]),
                int(observations.repeat[i]),
                int(observations.chunk_size[i]),
                int(observations.chunk_overlap[i]),
            )
            entry = totals.setdefault(
                key, [0, 0.0, 0.0, 0.0, observations.doc_tokens[i]]
            )
            entry[0] += 1
            entry[1] += result["observed_cost"][i]
            entry[2] += result["predicted_cost"][i]
            entry[3] += result["baseline_cost"][i]
    complete = np.array([e for e in totals.values() if e[0] == len(per_rule)])
    if len(complete):
        observed, model, baseline, tokens = complete[:, 1:5].T
    else:
        observed = model = baseline = tokens = np.array([])

    return Validation(
        documents=len(fold_of),
        folds=n_folds,
        rules=rule_metrics,
        rules_by_length=rule_lengths,
        pipeline={
            "model": error_metrics(observed, model),
            "baseline": error_metrics(observed, baseline),
        },
        pipeline_by_length=_by_length(tokens, observed, model, baseline, length_edges),
    )


def _metrics_line(label: str, metrics: ErrorMetrics) -> str:
    return (
        f"  {label:<34} rows {metrics.rows:>5}  MAPE {metrics.mape:7.1%}  "
        f"median {metrics.median_ape:7.1%}  bias {metrics.bias:+7.1%}  "
        f"R2 {metrics.r_squared:8.4f}"
    )


def format_validation(validation: Validation) -> str:
    """Tables of held-out error for checking by eye"""
    header = (
        f"Cross-validation: {validation.documents} documents, "
        f"{validation.folds} folds. Costs are priced without caching."
    )
    lines = [header]
    for rule, targets in validation.rules.items():
        lines.append(rule)
        lines += [_metrics_line(name, metrics) for name, metrics in targets.items()]
    lines.append("Pipeline cost (sum of rules)")
    lines += [
        _metrics_line(name, metrics) for name, metrics in validation.pipeline.items()
    ]
    lines.append("Pipeline cost by document length (tokens)")
    for band, results in validation.pipeline_by_length.items():
        lines += [
            _metrics_line(f"{band} {name}", metrics)
            for name, metrics in results.items()
        ]
    return "\n".join(lines)


def _terms(component: TokenComponent, digits: int) -> str:
    return ", ".join(
        f"{name}={value:.{digits}f}"
        for name, value in zip(component.feature_names, component.coefficients)
    )


def _factor_range(component: TokenComponent) -> str:
    factors = sorted(component.inflation.values())
    return f"{factors[0]:.3f}-{factors[-1]:.3f}"


def describe(model: CostModel) -> str:
    """One-paragraph summary of a fitted model for checking by eye"""
    config = model.config
    input_c, completion_c = model.input_tokens, model.completion_tokens
    low, high = model.doc_token_range
    header = (
        f"{config.rule} ({_config_label(config)}): {model.rows} rows, "
        f"{model.documents} documents, doc tokens {low:.0f}-{high:.0f}"
    )
    chunks = (
        f"  chunks: scale {model.chunk_count.scale:.3f}, "
        f"exact {model.chunk_count.exact_rate:.1%}"
    )
    input_line = (
        f"  input: {_terms(input_c, 3)}; R2 {input_c.r_squared:.4f} on "
        f"{input_c.single_pass_rows} single-pass rows; "
        f"setting factor {_factor_range(input_c)}"
    )
    completion_line = (
        f"  completion: {_terms(completion_c, 4)}; R2 {completion_c.r_squared:.3f}; "
        f"setting factor {_factor_range(completion_c)}"
    )
    cached = (
        f"  sweep cached share {model.sweep_cached_share:.1%} "
        "(not a production estimate)"
    )
    return f"{header}\n{chunks}\n{input_line}\n{completion_line}\n{cached}"


def main(argv=None):
    """Fit models to sweep results, print a summary of each, and optionally validate"""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", required=True, nargs="+")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Cross-validate one configuration per rule, holding out whole documents",
    )
    parser.add_argument(
        "--folds",
        type=int,
        help="Folds for --validate. Default: leave one document out",
    )
    parser.add_argument("--model", help="Only validate configurations with this model")
    parser.add_argument(
        "--reasoning-effort", help="Only validate this reasoning effort"
    )
    parser.add_argument(
        "--include-rationale", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--max-completion-tokens", type=int)
    args = parser.parse_args(argv)

    observations = build_observations(load_rows(args.results))
    models = fit_cost_models(observations)
    for config in sorted(models, key=lambda c: (c.rule, str(c))):
        print(describe(models[config]))
    if not models:
        print("No rows found")
    if args.validate and models:
        filters = {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "include_rationale": args.include_rationale,
            "max_completion_tokens": args.max_completion_tokens,
        }
        try:
            selected = select_configs(
                observations, **{k: v for k, v in filters.items() if v is not None}
            )
            validation = cross_validate(selected, args.folds)
        except ValueError as e:
            parser.error(str(e))
        print()
        print(format_validation(validation))


if __name__ == "__main__":
    main()
