from itertools import pairwise
from unittest.mock import Mock, patch

from src.analysis.text import LLMTextAnalyser
from src.monitoring import LoggingUtil
from src.redaction.config import (
    LLMTextRedactionConfig,
    RedactionConfig,
)
from src.redaction.redactor import LLMTextRedactor
from src.types import LLMTextRedactionResult


def test__llm_text_redactor__get_name():
    """
    - When get_name is called
    - The return value must be a string
    """
    assert isinstance(LLMTextRedactor.get_name(), str)


def test__llm_text_redactor__get_redaction_config_class():
    """
    - When get_redaction_config_class is called for the LLMTextRedactor class
    - The return value must be an instance of RedactionConfig
    """
    assert issubclass(LLMTextRedactor.get_redaction_config_class(), RedactionConfig)


@patch.object(LLMTextRedactor, "__init__", return_value=None)
def test__llm_text_redactor___analyse_text(mock_llm_text_redaction_config_init):
    """
    - Given I have some llm redaction config
    - When I call LLMTextRedactor.redact
    - Then I should receive a LLMTextRedactionResult with appropriate properties set
    """
    config = LLMTextRedactionConfig(
        name="config name",
        redactor_type="LLMTextRedaction",
        model="gpt-5.6-luna",
        role="some role",
        task="some task",
        redaction_terms=[
            "rule A",
            "rule B",
            "rule C",
        ],
    )
    mock_llm_result = LLMTextRedactionResult(
        rule_name="",
        run_metrics={},
        redaction_strings=[],
        metadata=LLMTextRedactionResult.LLMResultMetadata(
            request_count=0,
            input_token_count=0,
            completion_token_count=0,
            total_token_count=0,
            total_cost=0,
        ),
    )

    with (
        patch.object(
            LLMTextAnalyser, "__init__", return_value=None
        ) as mock_llm_text_analyser_init,
        patch.object(
            LLMTextAnalyser, "analyse_text", return_value=mock_llm_result
        ) as mock_analyse_text,
    ):
        LLMTextRedactor.config = config
        LLMTextRedactor()._analyse_text("some text to analyse")

    mock_llm_text_analyser_init.assert_called_once_with(config)
    mock_analyse_text.assert_called_once_with(
        config.create_system_prompt(), ["some text to analyse"]
    )


@patch.object(LLMTextRedactor, "__init__", return_value=None)
def test__llm_text_redactor___analyse_text__uses_config_chunk_size(
    mock_llm_text_redaction_config_init,
):
    """
    - Given I have llm redaction config with a small chunk size
    - When I call LLMTextRedactor._analyse_text with text longer than one chunk
    - Then the text should be split into multiple chunks before analysis
    """
    config = LLMTextRedactionConfig(
        name="config name",
        redactor_type="LLMTextRedaction",
        model="gpt-5.6-luna",
        role="some role",
        task="some task",
        redaction_terms=["rule A"],
        chunk_size=20,
        chunk_overlap=0,
    )
    mock_llm_result = LLMTextRedactionResult(
        rule_name="", run_metrics={}, redaction_strings=[]
    )

    with (
        patch.object(LLMTextAnalyser, "__init__", return_value=None),
        patch.object(
            LLMTextAnalyser, "analyse_text", return_value=mock_llm_result
        ) as mock_analyse_text,
    ):
        LLMTextRedactor.config = config
        LLMTextRedactor()._analyse_text(" ".join(f"word{i}" for i in range(200)))

    text_chunks = mock_analyse_text.call_args.args[1]
    assert len(text_chunks) > 1


@patch.object(LLMTextRedactor, "__init__", return_value=None)
def test__llm_text_redactor___analyse_text__empty_text_skips_analysis(
    mock_llm_text_redaction_config_init,
):
    """
    - Given I have empty text to analyse
    - When I call LLMTextRedactor._analyse_text with empty string
    - Then it should return an empty LLMTextRedactionResult without calling LLMTextAnalyser
    """
    config = LLMTextRedactionConfig(
        name="config name",
        redactor_type="LLMTextRedaction",
        model="gpt-5.6-luna",
        role="some role",
        task="some task",
        redaction_terms=["rule A"],
    )

    with (
        patch.object(
            LLMTextAnalyser, "__init__", return_value=None
        ) as mock_llm_text_analyser_init,
        patch.object(LLMTextAnalyser, "analyse_text") as mock_analyse_text,
    ):
        LLMTextRedactor.config = config
        result = LLMTextRedactor()._analyse_text("")

    # LLMTextAnalyser should never be instantiated or called
    mock_llm_text_analyser_init.assert_not_called()
    mock_analyse_text.assert_not_called()

    assert isinstance(result, LLMTextRedactionResult)
    assert result.rule_name == "config name"
    assert result.redaction_strings == ()
    assert result.run_metrics == {}


@patch.object(LLMTextRedactor, "__init__", return_value=None)
def test__llm_text_redactor___analyse_text__none_text_skips_analysis(
    mock_llm_text_redaction_config_init,
):
    """
    - Given text_to_analyse is None
    - When I call LLMTextRedactor._analyse_text with None
    - Then it should return an empty LLMTextRedactionResult without calling LLMTextAnalyser
    """
    config = LLMTextRedactionConfig(
        name="config name",
        redactor_type="LLMTextRedaction",
        model="gpt-5.6-luna",
        role="some role",
        task="some task",
        redaction_terms=["rule A"],
    )

    with patch.object(
        LLMTextAnalyser, "__init__", return_value=None
    ) as mock_llm_text_analyser_init:
        LLMTextRedactor.config = config
        result = LLMTextRedactor()._analyse_text(None)

    mock_llm_text_analyser_init.assert_not_called()
    assert result.redaction_strings == ()


def _create_config(**kwargs) -> LLMTextRedactionConfig:
    return LLMTextRedactionConfig(
        name="config name",
        redactor_type="LLMTextRedaction",
        model="gpt-5.6-luna",
        role="some role",
        task="some task",
        redaction_terms=["rule A"],
        **kwargs,
    )


def test__llm_text_redactor___locate_chunks__overlapping_chunks():
    """
    - Given text split into overlapping chunks by the configured splitter
    - When I locate the chunks in the text
    - Then each range covers exactly its chunk, and consecutive ranges overlap
    """
    text = " ".join(f"word{i}" for i in range(500))
    chunks = (
        _create_config(chunk_size=50, chunk_overlap=10)
        .create_text_splitter()
        .split_text(text)
    )

    chunk_ranges = LLMTextRedactor._locate_chunks(text, chunks)

    assert len(chunks) > 2
    assert [text[start:end] for start, end in chunk_ranges] == chunks
    assert all(next_start < end for (_, end), (next_start, _) in pairwise(chunk_ranges))


def test__llm_text_redactor___locate_chunks__repeated_text_moves_forward():
    """
    - Given text whose chunks are identical
    - When I locate the chunks in the text
    - Then each chunk is located after the previous one rather than at the first match
    """
    text = "\n\n".join(["The same paragraph repeats."] * 3)
    chunks = ["The same paragraph repeats."] * 3

    chunk_ranges = LLMTextRedactor._locate_chunks(text, chunks)

    assert chunk_ranges == [(0, 27), (29, 56), (58, 85)]


def test__llm_text_redactor___locate_chunks__missing_chunk():
    """
    - Given a chunk that does not appear in the text
    - When I locate the chunks in the text
    - Then its range is None, a warning is logged, and later chunks are still found
    """
    chunk_ranges = LLMTextRedactor._locate_chunks(
        "first second", ["first", "missing", "second"]
    )

    assert chunk_ranges == [(0, 5), None, (6, 12)]
    LoggingUtil.log_warning.assert_called_once_with(
        "Chunk 1 could not be located in the analysed text"
    )


def _occurrence(chunk_index, **kwargs):
    return {"chunk_index": chunk_index, "chunk_id": f"id{chunk_index}", **kwargs}


def test__llm_text_redactor___add_span_offsets__whole_words_only():
    """
    - Given a span that is also part of other words in a chunk
    - When I add the span offsets
    - Then only whole-word matches are found, offset by the chunk's start
    """
    chunk = "another her, there her."
    decisions = ({"text": "her", "occurrences": [{"chunk_index": 0}]},)

    [located] = LLMTextRedactor._add_span_offsets(decisions, [chunk], [(10, 32)])

    assert located["occurrences"][0]["span_offsets"] == [18, 29]


@patch.object(LLMTextRedactor, "__init__", return_value=None)
def test__llm_text_redactor___analyse_text__include_rationale_adds_offsets(
    mock_llm_text_redactor_init,
):
    """
    - Given an llm config with include_rationale enabled and text split into two
      overlapping chunks
    - When the analyser returns decisions traced to those chunks
    - Then each chunk summary gains its character range, each occurrence gains the
      offsets of its span in the analysed text, and the clause legend is attached
    """
    text = "Joan Brown met Mary. Later Joan Brown left."
    chunks = ["Joan Brown met Mary.", "Mary. Later Joan Brown left."]
    config = _create_config(include_rationale=True)
    analyser_result = LLMTextRedactionResult(
        rule_name="",
        run_metrics={},
        redaction_strings=("Joan Brown",),
        redaction_decisions=(
            {
                "text": "Joan Brown",
                "occurrences": [_occurrence(0), _occurrence(1)],
            },
        ),
        excluded_decisions=({"text": "Mary", "occurrences": [_occurrence(0)]},),
        chunk_summaries=(
            {"chunk_index": 0, "chunk_id": "id0", "status": "ok"},
            {"chunk_index": 1, "chunk_id": "id1", "status": "ok"},
        ),
    )

    with (
        patch.object(
            LLMTextRedactionConfig,
            "create_text_splitter",
            return_value=Mock(split_text=Mock(return_value=chunks)),
        ),
        patch.object(LLMTextAnalyser, "__init__", return_value=None),
        patch.object(LLMTextAnalyser, "analyse_text", return_value=analyser_result),
    ):
        redactor = LLMTextRedactor()
        redactor.config = config
        result = redactor._analyse_text(text)

    assert result.chunk_summaries == (
        {
            "chunk_index": 0,
            "chunk_id": "id0",
            "status": "ok",
            "char_start": 0,
            "char_end": 20,
        },
        {
            "chunk_index": 1,
            "chunk_id": "id1",
            "status": "ok",
            "char_start": 15,
            "char_end": 43,
        },
    )
    joan_offsets = [
        o["span_offsets"] for o in result.redaction_decisions[0]["occurrences"]
    ]
    assert joan_offsets == [[0], [27]]
    assert all(text[o : o + len("Joan Brown")] == "Joan Brown" for [o] in joan_offsets)
    assert result.excluded_decisions[0]["occurrences"][0]["span_offsets"] == [15]
    assert result.clause_legend == {"T1": "rule A"}


@patch.object(LLMTextRedactor, "__init__", return_value=None)
def test__llm_text_redactor___analyse_text__unlocated_chunk_has_no_offsets(
    mock_llm_text_redactor_init,
):
    """
    - Given a chunk that cannot be located in the analysed text
    - When the analyser returns a decision traced to that chunk
    - Then the chunk's range and the occurrence's span offsets are left empty
    """
    analyser_result = LLMTextRedactionResult(
        rule_name="",
        run_metrics={},
        redaction_decisions=({"text": "Joan Brown", "occurrences": [_occurrence(0)]},),
        chunk_summaries=({"chunk_index": 0, "chunk_id": "id0", "status": "ok"},),
    )

    with (
        patch.object(
            LLMTextRedactionConfig,
            "create_text_splitter",
            return_value=Mock(split_text=Mock(return_value=["Joan Brown, edited"])),
        ),
        patch.object(LLMTextAnalyser, "__init__", return_value=None),
        patch.object(LLMTextAnalyser, "analyse_text", return_value=analyser_result),
    ):
        redactor = LLMTextRedactor()
        redactor.config = _create_config(include_rationale=True)
        result = redactor._analyse_text("Joan Brown")

    assert result.chunk_summaries[0]["char_start"] is None
    assert result.chunk_summaries[0]["char_end"] is None
    assert result.redaction_decisions[0]["occurrences"][0]["span_offsets"] == []


@patch.object(LLMTextRedactor, "__init__", return_value=None)
def test__llm_text_redactor___analyse_text__no_rationale_has_no_clause_legend(
    mock_llm_text_redactor_init,
):
    with (
        patch.object(LLMTextAnalyser, "__init__", return_value=None),
        patch.object(
            LLMTextAnalyser,
            "analyse_text",
            return_value=LLMTextRedactionResult(rule_name="", run_metrics={}),
        ),
    ):
        redactor = LLMTextRedactor()
        redactor.config = _create_config()
        result = redactor._analyse_text("some text")

    assert result.clause_legend == {}
