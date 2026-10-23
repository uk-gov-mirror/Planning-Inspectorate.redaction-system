"""MLflow scorers for redaction prompts, built on the entity-level matching in `scoring`.

Each record's expectations hold the rule and the labelled entities in its text. The
rationale of the ``f2`` scorer lists every mistake with the label's category and
rationale, which is the feedback GEPA's reflection model learns from.
"""

from collections.abc import Callable
from typing import Any

from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

from .corpus import Document, Entity
from .scoring import (
    BETA,
    HARD_NEGATIVE_WEIGHT,
    UNLOCATABLE_WEIGHT,
    DocumentWords,
    ScoreCounts,
    find_subsequence,
    objective_score,
    overlaps,
    positive_outcomes,
    score_document,
)

COVERAGE_THRESHOLD = 0.8


def _document(inputs: dict, expectations: dict) -> Document:
    return Document(
        id=expectations["doc_id"],
        doc_type="record",
        text=inputs["text"],
        entities=tuple(Entity(**entity) for entity in expectations["entities"]),
    )


def _predictions(outputs: Any) -> list[str]:
    # A failed predict_fn call gives a string describing the error instead of a dict
    return outputs.get("redaction_strings", []) if isinstance(outputs, dict) else []


def _error(outputs: Any) -> str | None:
    return outputs.get("error") if isinstance(outputs, dict) else str(outputs)


def _counts(inputs: dict, outputs: Any, expectations: dict) -> ScoreCounts:
    return score_document(
        _document(inputs, expectations),
        _predictions(outputs),
        rule=expectations["rule"],
        coverage_threshold=COVERAGE_THRESHOLD,
    )


def _describe(entity: Entity) -> str:
    rationale = f": {entity.rationale}" if entity.rationale else ""
    return f"'{entity.text}' ({entity.category}){rationale}"


def describe_errors(document: Document, outputs: Any, rule: str) -> str:
    """List each missed label, wrongly redacted hard negative, unlabelled prediction
    and prediction that does not occur in the text"""
    predictions = _predictions(outputs)
    lines = []
    if error := _error(outputs):
        lines.append(f"The prediction failed: {error}")

    outcomes = positive_outcomes(document, predictions, rule, COVERAGE_THRESHOLD)
    lines += [
        f"Missed (must redact): {_describe(entity)}"
        for entity in document.positives(rule)
        if not outcomes[entity.text]
    ]

    document_words = DocumentWords(document.text)
    labelled_words = [document_words.of(e.text) for e in document.positives()]
    negatives = document.negatives(rule)
    hard_negative_hits = set()
    for prediction in dict.fromkeys(predictions):
        words = document_words.of(prediction)
        if not words:
            continue
        if not find_subsequence(document_words.words, words):
            lines.append(
                f"Not found verbatim in the text, so cannot be applied: '{prediction}'"
            )
        elif not any(overlaps(words, label) for label in labelled_words):
            hits = [e for e in negatives if overlaps(words, document_words.of(e.text))]
            if not hits:
                lines.append(f"Redacted with no matching label: '{prediction}'")
            hard_negative_hits.update(hits)
    lines += [
        f"Wrongly redacted (must keep): {_describe(entity)}"
        for entity in negatives
        if entity in hard_negative_hits
    ]
    return "\n".join(lines) or "Every labelled span was handled correctly."


@scorer
def f2(inputs: dict, outputs: Any, expectations: dict) -> Feedback:
    counts = _counts(inputs, outputs, expectations)
    rationale = describe_errors(
        _document(inputs, expectations), outputs, expectations["rule"]
    )
    return Feedback(value=counts.f_beta(BETA), rationale=rationale)


@scorer
def recall(inputs: dict, outputs: Any, expectations: dict) -> Feedback:
    counts = _counts(inputs, outputs, expectations)
    return Feedback(
        value=counts.recall,
        rationale=f"{counts.true_positives} of {counts.positives} labelled spans redacted",
    )


@scorer
def precision(inputs: dict, outputs: Any, expectations: dict) -> Feedback:
    counts = _counts(inputs, outputs, expectations)
    located = counts.correct_predictions + counts.false_positives
    return Feedback(
        value=counts.precision,
        rationale=f"{counts.correct_predictions} of {located} predictions match a label",
    )


@scorer
def hard_negative_hit_rate(inputs: dict, outputs: Any, expectations: dict) -> Feedback:
    counts = _counts(inputs, outputs, expectations)
    return Feedback(
        value=counts.hard_negative_fp_rate,
        rationale=(
            f"{counts.hard_negative_hits} of {counts.hard_negatives} must-keep spans "
            "redacted"
        ),
    )


@scorer
def unlocatable_rate(inputs: dict, outputs: Any, expectations: dict) -> Feedback:
    counts = _counts(inputs, outputs, expectations)
    return Feedback(
        value=counts.unlocatable_rate,
        rationale=(
            f"{counts.unlocatable_predictions} of {counts.predictions} predictions are "
            "not in the text"
        ),
    )


@scorer
def clause_citation_valid(outputs: Any) -> Feedback:
    """Share of decisions citing only existing clauses of the right kind: Terms (T#)
    or MustRedact (M#) for redactions, DoNotRedact (D#) for exclusions"""
    if not isinstance(outputs, dict):
        return Feedback(value=0.0, rationale="The prediction failed")
    valid_ids = set(outputs.get("prompt_clause_ids", []))
    decisions = [(d, "TM") for d in outputs.get("redactions", [])] + [
        (d, "D") for d in outputs.get("excluded", [])
    ]
    if not decisions:
        return Feedback(value=1.0, rationale="No decisions to check")
    invalid = [
        decision
        for decision, kinds in decisions
        if not decision["clause_ids"]
        or any(
            cid not in valid_ids or cid[0] not in kinds
            for cid in decision["clause_ids"]
        )
    ]
    rationale = (
        "Invalid or missing clause IDs: "
        + "; ".join(f"'{d['text']}' cites {d['clause_ids']}" for d in invalid)
        if invalid
        else "Every decision cites existing clauses of the right kind"
    )
    return Feedback(value=1 - len(invalid) / len(decisions), rationale=rationale)


SCORERS = [
    f2,
    recall,
    precision,
    hard_negative_hit_rate,
    unlocatable_rate,
    clause_citation_valid,
]


def make_objective(
    hard_negative_weight: float = HARD_NEGATIVE_WEIGHT,
    unlocatable_weight: float = UNLOCATABLE_WEIGHT,
) -> Callable[[dict[str, Any]], float]:
    """Single score for the optimiser, from the f2, hard_negative_hit_rate and
    unlocatable_rate scorers (see `objective_score`)"""

    def value(scores: dict[str, Any], name: str) -> float:
        score = scores.get(name)
        score = score.value if isinstance(score, Feedback) else score
        return float(score or 0.0)

    def objective(scores: dict[str, Any]) -> float:
        return objective_score(
            value(scores, "f2"),
            value(scores, "hard_negative_hit_rate"),
            value(scores, "unlocatable_rate"),
            hard_negative_weight,
            unlocatable_weight,
        )

    return objective
