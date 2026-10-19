from unittest.mock import patch

from src.analysis.text import LLMTextAnalyser
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
        system_prompt="some system prompt",
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
            output_token_count=0,
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
        system_prompt="some system prompt",
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
        system_prompt="some system prompt",
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
        system_prompt="some system prompt",
        redaction_terms=["rule A"],
    )

    with patch.object(
        LLMTextAnalyser, "__init__", return_value=None
    ) as mock_llm_text_analyser_init:
        LLMTextRedactor.config = config
        result = LLMTextRedactor()._analyse_text(None)

    mock_llm_text_analyser_init.assert_not_called()
    assert result.redaction_strings == ()
