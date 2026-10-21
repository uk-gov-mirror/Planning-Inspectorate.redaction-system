import pytest

pytest.importorskip("mlflow")

from eval.mlflow_scorers import (
    clause_citation_valid,
    f2,
    hard_negative_hit_rate,
    make_objective,
    precision,
    recall,
    unlocatable_rate,
)
from mlflow.entities import Feedback

TEXT = (
    "My neighbour Joan Whitcombe objects. Cllr Raj Patel agreed. "
    "Email me at joan@example.com."
)
INPUTS = {"text": TEXT}
EXPECTATIONS = {
    "rule": "names",
    "doc_id": "doc-1",
    "entities": [
        {
            "text": "Joan Whitcombe",
            "rule": "names",
            "category": "third_party",
            "redact": True,
            "rationale": "uninvolved neighbour",
        },
        {
            "text": "Cllr Raj Patel",
            "rule": "names",
            "category": "councillor",
            "redact": False,
            "rationale": "councillors are not redacted",
        },
        {
            "text": "joan@example.com",
            "rule": "contact_identity_info",
            "category": "personal_email",
            "redact": True,
        },
    ],
}


def _outputs(*predictions, **extra):
    return {"redaction_strings": list(predictions), **extra}


def _score(scorer, outputs):
    return scorer(inputs=INPUTS, outputs=outputs, expectations=EXPECTATIONS)


def test_scorers__perfect_prediction():
    outputs = _outputs("Joan Whitcombe")

    assert _score(f2, outputs).value == 1.0
    assert _score(recall, outputs).value == 1.0
    assert _score(precision, outputs).value == 1.0
    assert _score(hard_negative_hit_rate, outputs).value == 0.0
    assert _score(unlocatable_rate, outputs).value == 0.0
    assert _score(f2, outputs).rationale == "Every labelled span was handled correctly."


def test_scorers__cross_rule_prediction_is_not_a_false_positive():
    assert _score(precision, _outputs("Joan Whitcombe", "joan@example.com")).value == 1


def test_f2__rationale_lists_every_mistake():
    """
    - Given predictions that miss the label, redact a hard negative, redact unlabelled
      text and include text not in the chunk
    - When I score them with f2
    - Then the rationale describes each mistake with the label's category and rationale
    """
    outputs = _outputs("Cllr Raj Patel", "objects", "Mary Jones")

    feedback = _score(f2, outputs)

    assert feedback.value == 0.0
    assert feedback.rationale.splitlines() == [
        "Missed (must redact): 'Joan Whitcombe' (third_party): uninvolved neighbour",
        "Redacted with no matching label: 'objects'",
        "Not found verbatim in the text, so cannot be applied: 'Mary Jones'",
        "Wrongly redacted (must keep): 'Cllr Raj Patel' (councillor): "
        + "councillors are not redacted",
    ]
    assert _score(hard_negative_hit_rate, outputs).value == 1.0
    assert _score(unlocatable_rate, outputs).value == pytest.approx(1 / 3)


def test_scorers__failed_prediction():
    feedback = _score(f2, "Failed to invoke the predict_fn: boom")

    assert feedback.value == 0.0
    assert feedback.rationale.startswith("The prediction failed: Failed to invoke")
    assert _score(recall, "Failed").value == 0.0


@pytest.mark.parametrize(
    ("redaction_ids", "excluded_ids", "expected"),
    [
        (["T1", "M1"], ["D1"], 1.0),
        (["T9"], ["D1"], 0.5),
        ([], ["D1"], 0.5),
        (["T1"], ["T1"], 0.5),
    ],
)
def test_clause_citation_valid(redaction_ids, excluded_ids, expected):
    outputs = {
        "prompt_clause_ids": ["D1", "M1", "T1"],
        "redactions": [{"text": "Joan Whitcombe", "clause_ids": redaction_ids}],
        "excluded": [{"text": "Cllr Raj Patel", "clause_ids": excluded_ids}],
    }

    assert clause_citation_valid(outputs=outputs).value == expected


def test_clause_citation_valid__no_decisions():
    assert clause_citation_valid(outputs=_outputs()).value == 1.0


def test_make_objective():
    objective = make_objective(hard_negative_weight=0.5, unlocatable_weight=0.25)

    assert objective(
        {
            "f2": Feedback(value=0.8),
            "hard_negative_hit_rate": Feedback(value=0.2),
            "unlocatable_rate": Feedback(value=0.4),
        }
    ) == pytest.approx(0.8 - 0.1 - 0.1)
    assert objective({"f2": Feedback(value=0.1), "hard_negative_hit_rate": 1.0}) == 0
    assert objective({"f2": Feedback(value=None)}) == 0
