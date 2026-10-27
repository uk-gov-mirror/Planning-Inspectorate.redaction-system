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

Pricing, aggregation over rules and intervals build on these token predictions.

Example (run from the `redactor` directory):

    python -m eval.cost_model --results eval/results/optimised_sweep.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

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
        input_inflation = self.input_tokens.inflation_for(chunk_size, chunk_overlap)
        low, high = self.doc_token_range
        return {
            "chunk_count": chunks,
            "input_tokens": self.input_tokens.predict(
                input_features(chunks, doc_tokens, overlap), size, overlap
            ),
            "per_chunk_input_tokens": self.input_tokens.coefficients[0]
            * chunks
            * input_inflation,
            "completion_tokens": self.completion_tokens.predict(
                completion_features(chunks, doc_tokens), size, overlap
            ),
            "in_range": (doc_tokens >= low) & (doc_tokens <= high),
        }


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
    )


def fit_cost_models(
    observations: dict[CostConfig, CostObservations],
) -> dict[CostConfig, CostModel]:
    return {config: fit_cost_model(config, obs) for config, obs in observations.items()}


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
        f"{config.rule} ({config.model}, reasoning={config.reasoning_effort}, "
        f"rationale={config.include_rationale}, "
        f"max_completion={config.max_completion_tokens}, "
        f"prompt={config.prompt_tokens}): {model.rows} rows, "
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
    """Fit models to sweep results and print a summary of each"""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", required=True, nargs="+")
    args = parser.parse_args(argv)

    models = fit_cost_models(build_observations(load_rows(args.results)))
    for config in sorted(models, key=lambda c: (c.rule, str(c))):
        print(describe(models[config]))
    if not models:
        print("No rows found")


if __name__ == "__main__":
    main()
