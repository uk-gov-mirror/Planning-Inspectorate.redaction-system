from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("mlflow")

from eval import mlflow_dataset
from eval.corpus import Document, Entity
from eval.mlflow_dataset import (
    PromptRunner,
    chunk_records,
    document_records,
    split_documents,
)

from src.redaction.config import LLMTextRedactionConfig
from src.types import (
    LLMDetailedRedactionResultFormat,
    LLMExclusionDecision,
    LLMRedactionDecision,
    LLMTextRedactionResult,
)

FILLER = "The traffic assessment considered peak hour flows on the A-road. " * 30


def _config(**kwargs) -> LLMTextRedactionConfig:
    return LLMTextRedactionConfig(
        name="names",
        redactor_type="LLMTextRedaction",
        model="gpt-5.6-luna",
        role="role",
        task="task",
        redaction_terms=["term"],
        **kwargs,
    )


def _document(doc_id="doc-1") -> Document:
    return Document(
        id=doc_id,
        doc_type="relevant_representation",
        text=(
            f"My neighbour Joan Whitcombe objects. {FILLER}"
            f"Cllr Raj Patel agreed. {FILLER}Email joan@example.com today. {FILLER}"
        ),
        entities=(
            Entity("Joan Whitcombe", "names", "third_party", True),
            Entity("Cllr Raj Patel", "names", "councillor", False),
            Entity("joan@example.com", "contact_identity_info", "email", True),
        ),
    )


def _labels(record: dict) -> list[str]:
    return [e["text"] for e in record["expectations"]["entities"]]


def test_chunk_records__labels_each_chunk_with_its_own_entities():
    """
    - Given a document whose entities fall in different chunks
    - When I build chunk records for one rule, keeping every empty chunk
    - Then each record holds only the entities in its chunk, including other rules'
    """
    records = chunk_records(
        [_document()], "names", _config(chunk_size=100, chunk_overlap=0), 1.0
    )

    assert len(records) > 3
    for record in records:
        assert all(text in record["inputs"]["text"] for text in _labels(record))
        assert record["expectations"]["rule"] == "names"
    labelled = [r for r in records if _labels(r)]
    assert {label for r in labelled for label in _labels(r)} == {
        "Joan Whitcombe",
        "Cllr Raj Patel",
        "joan@example.com",
    }


def test_chunk_records__samples_empty_chunks():
    config = _config(chunk_size=100, chunk_overlap=0)
    all_records = chunk_records([_document()], "names", config, 1.0)
    rule_labelled = [
        r
        for r in all_records
        if any(e["rule"] == "names" for e in r["expectations"]["entities"])
    ]

    records = chunk_records([_document()], "names", config, empty_chunk_ratio=0.5)

    # Equal numbers of labelled and empty chunks at a ratio of 0.5
    assert len(records) == 2 * len(rule_labelled)
    assert chunk_records([_document()], "names", config, 0.0) == rule_labelled


def test_document_records():
    records = document_records([_document()], "names")

    assert records[0]["inputs"]["text"] == _document().text
    assert len(records[0]["expectations"]["entities"]) == 3


def test_split_documents__splits_by_document():
    documents = [_document(f"doc-{i}") for i in range(10)]

    train, val = split_documents(documents, 0.3)

    assert len(val) == 3
    assert {d.id for d in train} | {d.id for d in val} == {d.id for d in documents}
    assert not {d.id for d in train} & {d.id for d in val}


def _parsed_response(parsed):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
    )


@patch.object(mlflow_dataset, "LLMTextAnalyser")
def test_prompt_runner__predict_chunk(mock_analyser_class):
    """The prompt template is read per call and decisions are returned with the
    clause IDs available in that prompt"""
    parsed = LLMDetailedRedactionResultFormat(
        redactions=[
            LLMRedactionDecision(
                text="Joan Whitcombe",
                data_subject="third_party",
                clause_ids=["T1"],
                reason="neighbour",
                confidence="high",
            )
        ],
        excluded=[
            LLMExclusionDecision(
                text="Cllr Raj Patel", clause_ids=["D1"], reason="councillor"
            )
        ],
    )
    analyser = mock_analyser_class.return_value
    analyser._analyse_text_chunk.return_value = (_parsed_response(parsed), [])
    analyser.total_cost = 0.0
    prompt = Mock(template="<Terms>\n- [T1] a\n</Terms>\n<DoNotRedact>\n- [D1] b")

    runner = PromptRunner(prompt, _config())
    outputs = runner.predict_chunk("some text")

    assert mock_analyser_class.call_args.args[0].include_rationale is True
    analyser._analyse_text_chunk.assert_called_once_with(prompt.template, "some text")
    assert outputs["redaction_strings"] == ["Joan Whitcombe"]
    assert outputs["excluded"][0]["text"] == "Cllr Raj Patel"
    assert outputs["prompt_clause_ids"] == ["D1", "T1"]


@patch.object(mlflow_dataset, "LLMTextAnalyser")
def test_prompt_runner__predict_chunk__failed_call(mock_analyser_class):
    mock_analyser_class.return_value._analyse_text_chunk.return_value = None
    mock_analyser_class.return_value.total_cost = 0.0

    outputs = PromptRunner(Mock(template=""), _config()).predict_chunk("text")

    assert outputs["redaction_strings"] == []
    assert outputs["error"] == "LLM call failed after all retries"


@patch.object(mlflow_dataset, "LLMTextAnalyser")
def test_prompt_runner__predict_document(mock_analyser_class):
    mock_analyser_class.return_value.total_cost = 0.0
    mock_analyser_class.return_value.analyse_text.return_value = LLMTextRedactionResult(
        rule_name="",
        run_metrics={},
        redaction_strings=("Joan Whitcombe",),
        metadata=LLMTextRedactionResult.LLMResultMetadata(failed_chunk_count=1),
        redaction_decisions=({"text": "Joan Whitcombe"},),
    )
    runner = PromptRunner(Mock(template=""), _config(chunk_size=100, chunk_overlap=0))

    outputs = runner.predict_document(_document().text)

    chunks = mock_analyser_class.return_value.analyse_text.call_args.args[1]
    assert len(chunks) == outputs["chunk_count"] > 1
    assert outputs["redaction_strings"] == ["Joan Whitcombe"]
    assert outputs["failed_chunks"] == 1


@patch.object(mlflow_dataset, "LLMTextAnalyser")
def test_prompt_runner__stops_when_budget_spent(mock_analyser_class):
    mock_analyser_class.return_value.total_cost = 5.0

    runner = PromptRunner(Mock(template=""), _config(), budget=5.0)

    with pytest.raises(RuntimeError, match="Budget"):
        runner.predict_chunk("text")
    mock_analyser_class.return_value._analyse_text_chunk.assert_not_called()
