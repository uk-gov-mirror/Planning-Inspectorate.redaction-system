import pytest
from pydantic import ValidationError
from tiktoken import get_encoding

from src.redaction.config import LLMTextRedactionConfig


def test_llm_text_redactor__create_system_prompt():
    """
    - Given I have some llm redaction config
    - When I call LLMTextRedactionConfig._create_system_prompt
    - Then the returned system prompt should be correctly formatted
    """
    config = LLMTextRedactionConfig(
        name="config name",
        redactor_type="LLMTextRedaction",
        model="gpt-5.6-luna",
        text="some text",
        system_prompt="Some system prompt",
        redaction_terms=[
            "rule A",
            "rule B",
            "rule C",
        ],
        constraints=[
            "constraint X",
            "constraint Y",
        ],
    )

    expected_system_prompt = (
        "<SystemRole>\nSome system prompt\n</SystemRole>\n\n"
        "<Terms>\n- rule A\n- rule B\n- rule C\n</Terms>\n\n"
        f"{config.output_format}\n\n"
        "<Constraints>\n- constraint X\n- constraint Y\n</Constraints>"
    )
    actual_system_prompt = config.create_system_prompt()

    assert expected_system_prompt == actual_system_prompt


def test_llm_text_redactor__create_system_prompt_no_constraints():
    """
    - Given I have some llm redaction config
    - When I call LLMTextRedactionConfig._create_system_prompt with no constraints
    - Then the returned system prompt should be correctly formatted
    """
    config = LLMTextRedactionConfig(
        name="config name",
        redactor_type="LLMTextRedaction",
        model="gpt-5.6-luna",
        text="some text",
        system_prompt="Some system prompt",
        redaction_terms=[
            "rule A",
            "rule B",
            "rule C",
        ],
    )

    expected_system_prompt = (
        "<SystemRole>\nSome system prompt\n</SystemRole>\n\n"
        "<Terms>\n- rule A\n- rule B\n- rule C\n</Terms>\n\n"
        f"{config.output_format}"
    )
    actual_system_prompt = config.create_system_prompt()

    assert expected_system_prompt == actual_system_prompt


def _create_config(**kwargs) -> LLMTextRedactionConfig:
    return LLMTextRedactionConfig(
        name="config name",
        redactor_type="LLMTextRedaction",
        model="gpt-5.6-luna",
        system_prompt="System prompt",
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
