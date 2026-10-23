import pytest
from pydantic import ValidationError
from tiktoken import get_encoding

from src.redaction.config import LLMTextRedactionConfig


def test_llm_text_redactor__create_system_prompt():
    """
    - Given I have some llm redaction config with every section populated
    - When I call LLMTextRedactionConfig.create_system_prompt
    - Then the returned system prompt should be correctly formatted, with every
      section present
    """
    config = LLMTextRedactionConfig(
        name="config name",
        redactor_type="LLMTextRedaction",
        model="gpt-5.6-luna",
        text="some text",
        role="Some role",
        task="Some task",
        redaction_terms=[
            "rule A",
            "rule B",
            "rule C",
        ],
        must_redact=[
            "must redact X",
            "must redact Y",
        ],
        do_not_redact=[
            "do not redact P",
            "do not redact Q",
        ],
        examples=[
            "example 1",
            "example 2",
        ],
    )

    expected_system_prompt = (
        "<Role>\nSome role\n</Role>\n\n"
        "<Task>\nSome task\n</Task>\n\n"
        "<Terms>\n- rule A\n- rule B\n- rule C\n</Terms>\n\n"
        "<MustRedact>\n- must redact X\n- must redact Y\n</MustRedact>\n\n"
        "<DoNotRedact>\n- do not redact P\n- do not redact Q\n</DoNotRedact>\n\n"
        "<Examples>\n- example 1\n- example 2\n</Examples>"
    )
    actual_system_prompt = config.create_system_prompt()

    assert expected_system_prompt == actual_system_prompt


def test_llm_text_redactor__create_system_prompt__include_rationale_numbers_clauses():
    """
    - Given I have llm redaction config with include_rationale enabled
    - When I call LLMTextRedactionConfig.create_system_prompt
    - Then every list item is labelled with a clause ID for its section
    """
    config = _create_config(
        include_rationale=True,
        must_redact=["must redact X"],
        do_not_redact=["- do not redact P", "do not redact Q"],
        examples=["example 1"],
    )

    expected_system_prompt = (
        "<Role>\nSome role\n</Role>\n\n"
        "<Task>\nSome task\n</Task>\n\n"
        "<Terms>\n- [T1] rule A\n</Terms>\n\n"
        "<MustRedact>\n- [M1] must redact X\n</MustRedact>\n\n"
        "<DoNotRedact>\n- [D1] do not redact P\n- [D2] do not redact Q\n"
        "</DoNotRedact>\n\n"
        "<Examples>\n- [E1] example 1\n</Examples>"
    )

    assert config.create_system_prompt() == expected_system_prompt


def _create_config(**kwargs) -> LLMTextRedactionConfig:
    return LLMTextRedactionConfig(
        name="config name",
        redactor_type="LLMTextRedaction",
        model="gpt-5.6-luna",
        role="Some role",
        task="Some task",
        redaction_terms=["rule A"],
        **kwargs,
    )


def test_llm_text_redaction_config__default_chunk_settings():
    """
    - Given I create an llm redaction config without chunk settings
    - Then the default token-based chunk size and overlap should be used
    """
    config = _create_config()

    assert config.chunk_size == 1500
    assert config.chunk_overlap == 60


def test_llm_text_redaction_config__default_completion_settings():
    """
    - Given I create an llm redaction config without completion settings
    - Then the default token limit applies and reasoning effort is left unset
    """
    config = _create_config()

    assert config.max_tokens == 1000
    assert config.reasoning_effort is None


def test_llm_text_redaction_config__completion_settings():
    """
    - Given I create an llm redaction config with a token limit and reasoning effort
    - Then both should be set, including when copied with an override as the sweep does
    """
    config = _create_config(max_tokens=4000, reasoning_effort="low")

    assert config.max_tokens == 4000
    assert config.reasoning_effort == "low"
    assert config.model_copy(
        update={"reasoning_effort": "minimal"}
    ).reasoning_effort == ("minimal")


def test_llm_text_redaction_config__invalid_reasoning_effort():
    with pytest.raises(ValidationError):
        _create_config(reasoning_effort="extreme")


def test_llm_text_redaction_config__invalid_chunk_overlap():
    """
    - Given I create an llm redaction config with an overlap not smaller than the chunk size
    - Then a validation error should be raised
    """
    with pytest.raises(ValidationError):
        _create_config(chunk_size=100, chunk_overlap=100)


def test_llm_text_redaction_config__create_text_splitter__chunks_by_tokens():
    """
    - Given I have an llm redaction config with a small token chunk size
    - When I split some text with the splitter from create_text_splitter
    - Then every chunk should be within the configured token limit, and all text kept
    """
    config = _create_config(chunk_size=50, chunk_overlap=0)
    encoding = get_encoding(config.token_encoding_name)
    text = " ".join(f"word{i}" for i in range(500))

    chunks = config.create_text_splitter().split_text(text)

    assert len(chunks) > 1
    assert all(len(encoding.encode(chunk)) <= 50 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_llm_text_redactor__create_system_prompt_only_required_fields():
    """
    - Given I have some llm redaction config with only the required fields set
    - When I call LLMTextRedactionConfig.create_system_prompt
    - Then the returned system prompt should omit the optional sections entirely
    """
    config = LLMTextRedactionConfig(
        name="config name",
        redactor_type="LLMTextRedaction",
        model="gpt-5.6-luna",
        text="some text",
        role="Some role",
        task="Some task",
        redaction_terms=[
            "rule A",
            "rule B",
            "rule C",
        ],
    )

    expected_system_prompt = (
        "<Role>\nSome role\n</Role>\n\n"
        "<Task>\nSome task\n</Task>\n\n"
        "<Terms>\n- rule A\n- rule B\n- rule C\n</Terms>"
    )
    actual_system_prompt = config.create_system_prompt()

    assert expected_system_prompt == actual_system_prompt


def test_llm_text_redaction_config__optional_sections_survive_a_round_trip():
    """A config without the optional sections can be rebuilt from its own dump, as
    the prompt tooling does to apply overrides"""
    config = _create_config()

    rebuilt = LLMTextRedactionConfig(**(config.model_dump() | {"chunk_size": 4500}))

    assert (rebuilt.must_redact, rebuilt.do_not_redact, rebuilt.examples) == (
        None,
        None,
        None,
    )
    assert rebuilt.chunk_size == 4500


def test_llm_text_redaction_config__extra_sections_follow_the_examples():
    """
    - Given a config with examples and two extra sections
    - When I create the system prompt, with and without clause IDs
    - Then the extra sections come last, in order, and never carry clause IDs
    """
    config = _create_config(
        examples=["example 1"],
        extra_sections={"EdgeCases": ["case A", "case B"], "Style": ["be brief"]},
    )
    extra = (
        "<EdgeCases>\n- case A\n- case B\n</EdgeCases>\n\n<Style>\n- be brief\n</Style>"
    )

    assert config.create_system_prompt() == (
        "<Role>\nSome role\n</Role>\n\n<Task>\nSome task\n</Task>\n\n"
        "<Terms>\n- rule A\n</Terms>\n\n<Examples>\n- example 1\n</Examples>\n\n"
        + extra
    )
    numbered = config.model_copy(update={"include_rationale": True})
    assert numbered.create_system_prompt().endswith(
        "<Examples>\n- [E1] example 1\n</Examples>\n\n" + extra
    )


def test_llm_text_redaction_config__no_extra_sections_leaves_prompt_unchanged():
    assert (
        _create_config().create_system_prompt()
        == _create_config(extra_sections={}).create_system_prompt()
    )


@pytest.mark.parametrize(
    ("extra_sections", "message"),
    [
        ({"edgeCases": ["x"]}, "CamelCase"),
        ({"Edge Cases": ["x"]}, "CamelCase"),
        ({"DoNotRedact": ["x"]}, "already a section"),
        ({"Task": ["x"]}, "already a section"),
        ({"EdgeCases": []}, "at least one item"),
    ],
)
def test_llm_text_redaction_config__invalid_extra_sections(extra_sections, message):
    with pytest.raises(ValidationError, match=message):
        _create_config(extra_sections=extra_sections)
