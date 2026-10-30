from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel

from src.analysis.text import LLMTextAnalyser, LLMTextAnalyserConfig
from src.types import LLMTextRedactionResult


class SampleResultFormat(BaseModel):
    some_strings: list[str]


def test__llm_text_analyser___num_tokens_consumed():
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)
    system_prompt = "This is a system prompt."
    user_prompt = "This is a user prompt."

    num_tokens = llm_text_analyser._num_tokens_consumed(
        llm_text_analyser.create_api_message(system_prompt, user_prompt)
    )
    assert (
        num_tokens == 1024
    )  # 1000 completion + 6 in system + 6 in user + 2x4 in start + 2 in reply


def test__llm_text_analyser__invoke_chain__responds():
    """
    Assert that LLMTextAnalyser allows communication with the chain endpoint

    - Given we have a simple system prompt, user prompt, and result format
    - When we invoke the LLMTextAnalyser
    - Then we should receive a ParsedChatCompletion response from the LLM
    """
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser_inst = LLMTextAnalyser(llm_text_analyser_config)

    api_messages = [
        {"role": "system", "content": "Respond with a simple json list"},
        {"role": "user", "content": "Hello there"},
    ]
    response_format = SampleResultFormat
    completion = llm_text_analyser_inst.invoke_chain(api_messages, response_format)

    assert isinstance(completion, ParsedChatCompletion)


def test__llm_text_analyser__invoke_chain__has_correct_response_format():
    """
    Assert that communication with the LLM via LLMTextAnalyser responds with a result with the correct format

    - Given we have a simple system prompt, user prompt, and result format
    - When we invoke the LLMTextAnalyser
    - Then the completion response should contain a value that is an instance of our supplied result format
    """
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser_inst = LLMTextAnalyser(llm_text_analyser_config)

    api_messages = [
        {"role": "system", "content": "Respond with a json list"},
        {"role": "user", "content": "Hello there"},
    ]
    response_format = SampleResultFormat

    completion = llm_text_analyser_inst.invoke_chain(api_messages, response_format)
    formatted_result = completion.choices[0].message.parsed

    assert isinstance(formatted_result, response_format)
    assert formatted_result.some_strings


def test__llm_text_analyser__invoke_chain__with_reasoning_effort():
    """
    Assert that a reasoning model accepts a configured reasoning_effort

    - Given an analyser config with a reasoning effort set
    - When we invoke the LLMTextAnalyser
    - Then we should receive a parsed completion in our supplied result format
    """
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna", reasoning_effort="low", max_tokens=4000
    )
    llm_text_analyser_inst = LLMTextAnalyser(llm_text_analyser_config)

    api_messages = [
        {"role": "system", "content": "Respond with a json list"},
        {"role": "user", "content": "Hello there"},
    ]
    completion = llm_text_analyser_inst.invoke_chain(
        api_messages, SampleResultFormat, llm_text_analyser_config.max_tokens
    )

    assert isinstance(completion, ParsedChatCompletion)
    assert isinstance(completion.choices[0].message.parsed, SampleResultFormat)


def test__llm_text_analyser__analyse_text():
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser_inst = LLMTextAnalyser(llm_text_analyser_config)

    system_prompt = "Identify redaction strings in the text"
    text_chunks = [
        "This is some sample text that contains a redaction string: SECRET123.",
        "Here is another redaction string: CONFIDENTIAL456.",
    ] * 10  # Repeat to increase number of chunks

    result = llm_text_analyser_inst.analyse_text(
        system_prompt,
        text_chunks,
    )

    assert 1 < llm_text_analyser_inst.config.max_concurrent_requests <= 32
    assert isinstance(result, LLMTextRedactionResult)
    assert set(result.redaction_strings) == {"SECRET123", "CONFIDENTIAL456"}


def test__llm_text_analyser__analyse_text__single_chunk_sequential():
    """
    When there is only 1 text chunk, processing should be sequential (max_workers=1)
    and still produce correct results.
    """
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser_inst = LLMTextAnalyser(llm_text_analyser_config)

    system_prompt = "Identify redaction strings in the text"
    text_chunks = [
        "This is some sample text that contains a redaction string: SECRET123.",
    ]

    result = llm_text_analyser_inst.analyse_text(
        system_prompt,
        text_chunks,
    )

    assert isinstance(result, LLMTextRedactionResult)
    assert result.metadata.request_count == 1
    assert "SECRET123" in result.redaction_strings


def test__llm_text_analyser__analyse_text__workers_capped_by_chunk_count():
    """
    When max_concurrent_requests exceeds the number of chunks,
    the actual worker count should be capped to the number of chunks.
    """
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
        max_concurrent_requests=10,
    )
    llm_text_analyser_inst = LLMTextAnalyser(llm_text_analyser_config)

    system_prompt = "Identify redaction strings in the text"
    text_chunks = [
        "This text contains SECRET123.",
        "This text contains CONFIDENTIAL456.",
    ]

    result = llm_text_analyser_inst.analyse_text(
        system_prompt,
        text_chunks,
    )

    assert isinstance(result, LLMTextRedactionResult)
    assert result.metadata.request_count == 2
    assert set(result.redaction_strings) == {"SECRET123", "CONFIDENTIAL456"}
