import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, Mock, patch

import pytest
from openai import LengthFinishReasonError, RateLimitError
from pydantic import ValidationError
from tenacity import stop_after_attempt, wait_none

from src.analysis.text import (
    LLMTextAnalyser,
    LLMTextAnalyserConfig,
    handle_last_retry_error,
    merge_redaction_decisions,
    update_max_tokens,
)
from src.monitoring import LoggingUtil
from src.types import (
    LLMDetailedRedactionResultFormat,
    LLMExclusionDecision,
    LLMRedactionDecision,
    LLMRedactionResultFormat,
    LLMTextRedactionResult,
)

MODULE = "src.analysis.text"


class MockLLMChatCompletion:
    def __init__(self, choices, usage):
        self.choices = choices
        self.usage = usage


class MockLLMChatCompletionChoice:
    def __init__(self, message):
        self.message = message


class MockLLMChatCompletionChoiceMessage:
    def __init__(self, parsed):
        self.parsed = parsed


class MockLLMChatCompletionUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class MockOpenAIAPIResponse:
    request = None
    status_code = None

    def __init__(self):
        self.headers = {}


TOKEN_RATE_LIMIT = 3000000
REQUEST_RATE_LIMIT = 3000


def test__handle_last_retry_error():
    retry_state = Mock()
    retry_state.outcome = Mock()
    retry_state.outcome.exception.return_value = Exception(
        "Test exception for last retry"
    )
    handle_last_retry_error(retry_state)
    LoggingUtil.log_info.assert_called_with(
        "All retry attempts failed: Test exception for last retry\n"
        "Returning None for this chunk."
    )


def test__update_max_tokens():
    retry_state = Mock()
    retry_state.kwargs = {"max_completion_tokens": 1000}
    update_max_tokens(retry_state)
    LoggingUtil.log_info.assert_called_with(
        "Updating max_completion_tokens to 2000 for next attempt."
    )
    assert retry_state.kwargs["max_completion_tokens"] == 2000


def test__update_max_tokens__limit():
    retry_state = Mock()
    retry_state.kwargs = {"max_completion_tokens": 5000}
    update_max_tokens(retry_state)
    LoggingUtil.log_info.assert_called_with(
        "Updating max_completion_tokens to 8000 for next attempt."
    )
    assert retry_state.kwargs["max_completion_tokens"] == 8000


def test__update_max_tokens__defaults_to_analyser_limit():
    retry_state = Mock()
    retry_state.kwargs = {}
    retry_state.args = (
        Mock(config=LLMTextAnalyserConfig(model="gpt-5.6-luna", max_tokens=1500)),
    )
    update_max_tokens(retry_state)
    LoggingUtil.log_info.assert_called_with(
        "Updating max_completion_tokens to 3000 for next attempt."
    )
    assert retry_state.kwargs["max_completion_tokens"] == 3000


def test__update_max_tokens__no_analyser_limit():
    retry_state = Mock()
    retry_state.kwargs = {}
    retry_state.args = (
        Mock(config=LLMTextAnalyserConfig(model="gpt-5.6-luna", max_tokens=None)),
    )
    update_max_tokens(retry_state)
    assert retry_state.kwargs["max_completion_tokens"] == 2000


def test__llm_text_analyser_config__reasoning_effort_default():
    assert LLMTextAnalyserConfig(model="gpt-5.6-luna").reasoning_effort is None


@pytest.mark.parametrize("reasoning_effort", ["minimal", "low", "medium", "high"])
def test__llm_text_analyser_config__reasoning_effort(reasoning_effort):
    config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna", reasoning_effort=reasoning_effort
    )
    assert config.reasoning_effort == reasoning_effort


def test__llm_text_analyser_config__invalid_reasoning_effort():
    with pytest.raises(ValidationError):
        LLMTextAnalyserConfig(model="gpt-5.6-luna", reasoning_effort="extreme")


@pytest.mark.parametrize(
    ("model", "reasoning_effort", "expected_kwargs"),
    [
        (
            "gpt-5.6-luna",
            "low",
            {"max_completion_tokens": 1000, "reasoning_effort": "low"},
        ),
        ("gpt-5.6-luna", None, {"max_completion_tokens": 1000}),
    ],
)
@patch.object(LLMTextAnalyser, "__init__", return_value=None)
def test__llm_text_analyser__invoke_chain__reasoning_effort(
    mock_llm_text_analyser_init, model, reasoning_effort, expected_kwargs
):
    """reasoning_effort is only sent when it is set"""
    llm_text_analyser = LLMTextAnalyser()
    llm_text_analyser.config = LLMTextAnalyserConfig(
        model=model, reasoning_effort=reasoning_effort
    )
    llm_text_analyser.llm = Mock()
    api_messages = [{"role": "user", "content": "hello"}]

    llm_text_analyser.invoke_chain(api_messages, LLMRedactionResultFormat, 1000)

    llm_text_analyser.llm.chat.completions.parse.assert_called_once_with(
        model=model,
        messages=api_messages,
        response_format=LLMRedactionResultFormat,
        **expected_kwargs,
    )


@patch.object(
    LLMTextAnalyser,
    "create_api_message",
    return_value=[{"role": "system", "content": "system prompt"}],
)
@patch.object(LLMTextAnalyser, "_num_tokens_consumed", return_value=10)
def test__llm_text_analyser___analyse_text_chunk__retry_starts_from_config_limit(
    mock_num_tokens, mock_api_message
):
    """A truncated first attempt retries at double the configured limit, not 2000"""
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna", max_tokens=1500
    )
    truncated = Mock(usage=None)
    successful = MagicMock(usage=None)
    successful.choices[0].message.parsed.redaction_strings = ["string A"]
    with patch.object(
        LLMTextAnalyser,
        "invoke_chain",
        side_effect=[LengthFinishReasonError(completion=truncated), successful],
    ):
        llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)
        llm_text_analyser._analyse_text_chunk.retry.wait = wait_none()
        llm_text_analyser._analyse_text_chunk.retry.stop = stop_after_attempt(2)

        llm_text_analyser._analyse_text_chunk(
            system_prompt="system prompt", user_prompt=""
        )

        calls = LLMTextAnalyser.invoke_chain.call_args_list
        assert [call.args[-1] for call in calls] == [1500, 3000]


def test__llm_text_analyser____init__():
    with (
        patch(f"{MODULE}.get_bearer_token_provider") as mock_get_token_provider,
        patch(f"{MODULE}.AzureOpenAI") as mock_azure_openai,
    ):
        llm_text_analyser_config = LLMTextAnalyserConfig(
            model="gpt-5.6-luna",
            token_rate_limit=2000,
        )
        llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)

    assert llm_text_analyser.config.token_rate_limit == 2000
    assert llm_text_analyser.config.token_encoding_name == "o200k_base"

    assert llm_text_analyser.input_token_cost == pytest.approx(16 * 1e-8)
    assert llm_text_analyser.output_token_cost == pytest.approx(91 * 1e-8)

    # A provider lets the client renew its own token instead of using one that expires
    client_kwargs = mock_azure_openai.call_args.kwargs
    assert (
        client_kwargs["azure_ad_token_provider"] is mock_get_token_provider.return_value
    )
    assert "azure_ad_token" not in client_kwargs


@patch.object(LLMTextAnalyser, "__init__", return_value=None)
def test__llm_text_analyser___set_model_details(mock_llm_text_analyser_init):
    llm_text_analyser = LLMTextAnalyser()
    llm_text_analyser.config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
        token_rate_limit=TOKEN_RATE_LIMIT / 10,
        request_rate_limit=REQUEST_RATE_LIMIT / 10,
    )

    llm_text_analyser._set_model_details()

    assert llm_text_analyser.config.token_rate_limit == TOKEN_RATE_LIMIT / 10
    assert llm_text_analyser.config.request_rate_limit == REQUEST_RATE_LIMIT / 10

    assert llm_text_analyser.input_token_cost == pytest.approx(16 * 1e-8)
    assert llm_text_analyser.output_token_cost == pytest.approx(91 * 1e-8)


@patch.object(LLMTextAnalyser, "__init__", return_value=None)
def test__llm_text_analyser___set_model_details__exceeds_token_rate_limit(
    mock_llm_text_analyser_init,
):
    llm_text_analyser = LLMTextAnalyser()
    llm_text_analyser.config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
        token_rate_limit=TOKEN_RATE_LIMIT * 3,
        request_rate_limit=REQUEST_RATE_LIMIT / 10,
    )

    llm_text_analyser._set_model_details()

    assert llm_text_analyser.config.token_rate_limit == TOKEN_RATE_LIMIT
    LoggingUtil.log_info.assert_called_with(
        "Token rate limit for model gpt-5.6-luna exceeds maximum. "
        f"Setting to maximum of {TOKEN_RATE_LIMIT} tokens per minute."
    )


@patch.object(LLMTextAnalyser, "__init__", return_value=None)
def test__llm_text_analyser___set_model_details__exceeds_request_rate_limit(
    mock_llm_text_analyser_init,
):
    llm_text_analyser = LLMTextAnalyser()
    llm_text_analyser.config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
        token_rate_limit=TOKEN_RATE_LIMIT / 10,
        request_rate_limit=REQUEST_RATE_LIMIT * 2,
    )

    llm_text_analyser._set_model_details()

    assert llm_text_analyser.config.request_rate_limit == REQUEST_RATE_LIMIT
    LoggingUtil.log_info.assert_called_with(
        "Request rate limit for model gpt-5.6-luna exceeds maximum. "
        f"Setting to maximum of {REQUEST_RATE_LIMIT} requests per minute."
    )


@patch.object(LLMTextAnalyser, "__init__", return_value=None)
def test__llm_text_analyser___set_model_details__zero_token_request_rate_limit(
    mock_llm_text_analyser_init,
):
    llm_text_analyser = LLMTextAnalyser()
    llm_text_analyser.config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
        token_rate_limit=0,
        request_rate_limit=0,
    )

    llm_text_analyser._set_model_details()

    assert llm_text_analyser.config.token_rate_limit == TOKEN_RATE_LIMIT * 0.5
    assert llm_text_analyser.config.request_rate_limit == REQUEST_RATE_LIMIT * 0.5


@patch.object(LLMTextAnalyser, "__init__", return_value=None)
def test__llm_text_analyser___set_model_details__invalid_model(
    mock_llm_text_analyser_init,
):
    llm_text_analyser = LLMTextAnalyser()
    llm_text_analyser.config = LLMTextAnalyserConfig(
        model="not-a-real-model",
    )

    with pytest.raises(ValueError) as exc:
        llm_text_analyser._set_model_details()

    assert "Model not-a-real-model is not supported." in str(exc.value)


@patch.object(LLMTextAnalyser, "__init__", return_value=None)
@patch(f"{MODULE}.os.cpu_count", return_value=8)
def test__llm_text_analyser___set_workers__none_given(
    mock_llm_text_analyser_init, mock_cpu_count
):
    llm_text_analyser = LLMTextAnalyser()
    llm_text_analyser.config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser._set_workers()

    assert llm_text_analyser.config.max_concurrent_requests == 12


@patch.object(LLMTextAnalyser, "__init__", return_value=None)
@patch(f"{MODULE}.os.cpu_count", return_value=8)
def test__llm_text_analyser___set_workers__exceeds_cpu_count(
    mock_llm_text_analyser_init, mock_cpu_count
):
    llm_text_analyser = LLMTextAnalyser()
    llm_text_analyser.config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser._set_workers(40)

    assert llm_text_analyser.config.max_concurrent_requests == 12


@patch.object(LLMTextAnalyser, "__init__", return_value=None)
@patch(f"{MODULE}.os.cpu_count", return_value=8)
def test__llm_text_analyser___set_workers__zero_cpu_count(
    mock_llm_text_analyser_init, mock_cpu_count
):
    llm_text_analyser = LLMTextAnalyser()
    llm_text_analyser.config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser._set_workers(0)

    assert llm_text_analyser.config.max_concurrent_requests == 1


@patch.object(LLMTextAnalyser, "__init__", return_value=None)
@patch(f"{MODULE}.os.cpu_count", return_value=40)
def test__llm_text_analyser___set_workers__high_cpu_count(
    mock_llm_text_analyser_init, mock_cpu_count
):
    llm_text_analyser = LLMTextAnalyser()
    llm_text_analyser.config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser._set_workers()

    assert llm_text_analyser.config.max_concurrent_requests == 32


@patch(f"{MODULE}.get_encoding")
def test__llm_text_analyser___num_tokens_consumed__exception(mock_get_encoding):
    mock_encoding = Mock()
    mock_encoding.encode.side_effect = Exception("Encoding error")
    mock_get_encoding.return_value = mock_encoding

    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)
    system_prompt = "This is a system prompt."
    user_prompt = "This is a user prompt."

    num_tokens = llm_text_analyser._num_tokens_consumed(
        llm_text_analyser.create_api_message(system_prompt, user_prompt)
    )
    assert num_tokens == 0


def test__create_api_message():
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)

    system_prompt = "This is a system prompt."
    user_prompt = "This is a user prompt."
    expected_message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    actual_message = llm_text_analyser.create_api_message(system_prompt, user_prompt)
    assert actual_message == expected_message


def create_mock_chat_completion(
    redaction_strings=None, prompt_tokens=5, completion_tokens=4
):
    if redaction_strings is None:
        redaction_strings = ["string A", "string B"]
    return MockLLMChatCompletion(
        choices=[
            MockLLMChatCompletionChoice(
                message=MockLLMChatCompletionChoiceMessage(
                    parsed=LLMRedactionResultFormat(redaction_strings=redaction_strings)
                )
            )
        ],
        usage=MockLLMChatCompletionUsage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


def test__llm_text_analyser___compute_costs():
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)
    llm_text_analyser.input_token_cost = 1
    llm_text_analyser.output_token_cost = 2

    mock_chat_completion = create_mock_chat_completion(
        prompt_tokens=10, completion_tokens=15
    )

    llm_text_analyser._compute_costs(mock_chat_completion.usage)

    assert llm_text_analyser.input_token_count == 10
    assert llm_text_analyser.completion_token_count == 15
    assert llm_text_analyser.total_cost == 40


def test__llm_text_analyser___compute_costs__cached_and_reasoning_tokens():
    """
    - Given an LLM usage object that includes cached input and reasoning token details
    - When I call _compute_costs
    - Then the cached and reasoning token counts should be accumulated
    """
    llm_text_analyser = LLMTextAnalyser(LLMTextAnalyserConfig(model="gpt-5.6-luna"))
    llm_text_analyser.input_token_cost = 1
    llm_text_analyser.output_token_cost = 2
    usage = MockLLMChatCompletionUsage(prompt_tokens=10, completion_tokens=15)
    usage.prompt_tokens_details = Mock(cached_tokens=6, cache_write_tokens=0)
    usage.completion_tokens_details = Mock(reasoning_tokens=9)

    llm_text_analyser._compute_costs(usage)
    llm_text_analyser._compute_costs(usage)

    assert llm_text_analyser.cached_input_token_count == 12
    assert llm_text_analyser.reasoning_token_count == 18


@patch.object(
    LLMTextAnalyser,
    "create_api_message",
    return_value=[{"role": "system", "content": "system prompt"}],
)
@patch.object(LLMTextAnalyser, "_num_tokens_consumed", return_value=10)
def test__llm_text_analyser___analyse_text_chunk(
    mock_num_tokens_consumed, mock_create_api_message
):
    mock_chat_completion = create_mock_chat_completion()
    redaction_strings = mock_chat_completion.choices[0].message.parsed.redaction_strings
    expected_result = (mock_chat_completion, redaction_strings)

    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)
    llm_text_analyser.request_semaphore = Mock()
    llm_text_analyser.token_semaphore = Mock()
    llm_text_analyser.input_token_cost = 1
    llm_text_analyser.output_token_cost = 2

    with patch.object(
        LLMTextAnalyser, "invoke_chain", return_value=mock_chat_completion
    ):
        actual_result = llm_text_analyser._analyse_text_chunk(
            system_prompt="system prompt", user_prompt=""
        )

    assert expected_result == actual_result

    assert llm_text_analyser.input_token_count == 5
    assert llm_text_analyser.completion_token_count == 4
    assert llm_text_analyser.total_cost == 13

    llm_text_analyser.request_semaphore.acquire.assert_called_once()
    llm_text_analyser.request_semaphore.release.assert_called_once()

    llm_text_analyser.token_semaphore.acquire.assert_called_once_with(10)
    llm_text_analyser.token_semaphore.release.assert_called_once_with(10)


@patch.object(
    LLMTextAnalyser,
    "create_api_message",
    return_value=[{"role": "system", "content": "system prompt"}],
)
@patch.object(LLMTextAnalyser, "_num_tokens_consumed", return_value=10)
def test__llm_text_analyser___analyse_text_chunk__timeout_on_request_semaphore(
    mock_num_tokens_consumed, mock_create_api_message
):
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
        request_timeout=1,
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)
    llm_text_analyser.request_semaphore = Mock()
    llm_text_analyser.token_semaphore = Mock()

    llm_text_analyser._analyse_text_chunk.retry.wait = wait_none()
    llm_text_analyser._analyse_text_chunk.retry.stop = stop_after_attempt(1)

    llm_text_analyser.request_semaphore.acquire.return_value = False

    user_prompt = ""

    result = llm_text_analyser._analyse_text_chunk(
        system_prompt="system prompt", user_prompt=""
    )

    assert result is None  # Timeout occurred, so None is returned

    llm_text_analyser.request_semaphore.acquire.assert_called_once()
    llm_text_analyser.request_semaphore.release.assert_not_called()

    llm_text_analyser.token_semaphore.acquire.assert_not_called()
    llm_text_analyser.token_semaphore.release.assert_not_called()

    logging_util_calls = LoggingUtil.log_exception.call_args_list
    logging_util_calls_as_string = [str(x) for x in logging_util_calls]
    timeout_error = str(
        TimeoutError(
            f"(chunk ID {hash(user_prompt)}) Timeout while waiting for request semaphore to be available."
        )
    )
    assert any(timeout_error in x for x in logging_util_calls_as_string), (
        f"Expected {timeout_error} to be called. Called list was {logging_util_calls}"
    )


@patch.object(
    LLMTextAnalyser, "invoke_chain", side_effect=Exception("Some LLM invocation error")
)
@patch.object(
    LLMTextAnalyser,
    "create_api_message",
    return_value=[{"role": "system", "content": "system prompt"}],
)
@patch.object(LLMTextAnalyser, "_num_tokens_consumed", return_value=10)
def test__llm_text_analyser___analyse_text_chunk__exception(
    mock_num_tokens_consumed, mock_create_api_message, mock_invoke_chain
):
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)

    llm_text_analyser._analyse_text_chunk.retry.stop = stop_after_attempt(1)

    with pytest.raises(Exception):  # noqa: B017
        llm_text_analyser._analyse_text_chunk(
            system_prompt="system prompt", user_prompt=""
        )
        LoggingUtil.log_exception.assert_called_with(
            "An error occurred while processing the chunk: Some LLM invocation error"
        )


@patch.object(
    LLMTextAnalyser,
    "create_api_message",
    return_value=[{"role": "system", "content": "system prompt"}],
)
@patch.object(LLMTextAnalyser, "_num_tokens_consumed", return_value=10)
def test__llm_text_analyser___analyse_text_chunk__length_finish_reason(
    mock_num_tokens_consumed, mock_create_api_message
):
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)

    llm_text_analyser._analyse_text_chunk.retry.stop = stop_after_attempt(1)
    completion = create_mock_chat_completion()

    with (
        pytest.raises(Exception),  # noqa: B017
        patch.object(
            LLMTextAnalyser,
            "invoke_chain",
            side_effect=LengthFinishReasonError(completion=completion),
        ),
    ):
        llm_text_analyser._analyse_text_chunk(
            system_prompt="system prompt", user_prompt=""
        )
        LoggingUtil.log_exception.assert_called_with(
            "An error occurred while processing the chunk: Could not parse content as"
            f"the length limit was reached - {completion.usage}"
        )


@pytest.mark.parametrize(
    "exception",
    [
        RateLimitError("message", response=MockOpenAIAPIResponse(), body="body"),
        TimeoutError("Some LLM invocation error"),
        LengthFinishReasonError(completion=create_mock_chat_completion()),
        AttributeError("'str' object has no attribute 'choices'"),
    ],
)
@patch.object(
    LLMTextAnalyser,
    "create_api_message",
    return_value=[{"role": "system", "content": "system prompt"}],
)
@patch.object(LLMTextAnalyser, "_num_tokens_consumed", return_value=10)
def test__llm_text_analyser___analyse_text_chunk__retry_on_exception(
    mock_num_tokens, mock_api_message, exception
):
    mock_chat_completion = create_mock_chat_completion()
    redaction_strings = mock_chat_completion.choices[0].message.parsed.redaction_strings

    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    with patch.object(
        LLMTextAnalyser,
        "invoke_chain",
        side_effect=[
            exception,
            create_mock_chat_completion(["string A", "string B"]),
        ],
    ):
        llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)

        llm_text_analyser._analyse_text_chunk.retry.wait = wait_none()
        llm_text_analyser._analyse_text_chunk.retry.stop = stop_after_attempt(2)

        actual_result = llm_text_analyser._analyse_text_chunk(
            system_prompt="system prompt", user_prompt="", max_completion_tokens=1000
        )

        assert LLMTextAnalyser.invoke_chain.call_count == 2
        assert isinstance(actual_result[0], MockLLMChatCompletion)
        assert actual_result[1] == redaction_strings

        if isinstance(exception, LengthFinishReasonError):
            # Last call should have updated max_completion_tokens to 2000
            assert LLMTextAnalyser.invoke_chain.call_args.args[-1] == 2000


def test__llm_text_analyser__analyse_text():
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)
    llm_text_analyser.request_semaphore = Mock()
    llm_text_analyser.token_semaphore = Mock()
    llm_text_analyser.input_token_cost = 1
    llm_text_analyser.output_token_cost = 2

    results = [
        (create_mock_chat_completion(["string A"]), ["string A"]),
        (create_mock_chat_completion(["string B"]), ["string B"]),
    ]
    results_iter = iter(results)

    def side_effect(*args, **kwargs):
        result = next(results_iter)
        llm_text_analyser._compute_costs(result[0].usage)
        return result

    with patch.object(
        LLMTextAnalyser, "_analyse_text_chunk"
    ) as mock_analyse_text_chunk:
        mock_analyse_text_chunk.side_effect = side_effect
        actual_result = llm_text_analyser.analyse_text(
            system_prompt="system prompt",
            text_chunks=["redaction string A", "redaction string B"],
        )

    assert actual_result.metadata == LLMTextRedactionResult.LLMResultMetadata(
        request_count=2,
        input_token_count=10,
        completion_token_count=8,
        total_token_count=18,
        total_cost=26.0,
    )

    # Output may be unordered due to parallel execution
    assert set(actual_result.redaction_strings) == {"string A", "string B"}

    assert llm_text_analyser.input_token_count == 10
    assert llm_text_analyser.completion_token_count == 8
    assert llm_text_analyser.total_cost == 26.0


def _decision(text, clause_ids=("T1",), confidence="high", reason="a reason"):
    return LLMRedactionDecision(
        text=text,
        data_subject="third_party",
        clause_ids=list(clause_ids),
        reason=reason,
        confidence=confidence,
    )


def _exclusion(text):
    return LLMExclusionDecision(text=text, clause_ids=["D1"], reason="councillor")


def create_mock_detailed_chat_completion(redactions, excluded=()):
    return MockLLMChatCompletion(
        choices=[
            MockLLMChatCompletionChoice(
                message=MockLLMChatCompletionChoiceMessage(
                    parsed=LLMDetailedRedactionResultFormat(
                        redactions=list(redactions), excluded=list(excluded)
                    )
                )
            )
        ],
        usage=MockLLMChatCompletionUsage(prompt_tokens=5, completion_tokens=4),
    )


def test__merge_redaction_decisions():
    """
    - Given two chunks that both redact a span with different clauses and confidence
    - When I merge their decisions
    - Then there is one decision for the span with the highest confidence, every
      clause, and the chunk count, and exclusions of redacted spans are dropped
    """
    first = LLMDetailedRedactionResultFormat(
        redactions=[_decision("Joan Brown", ["T1"], "low", "first reason")],
        excluded=[_exclusion("Cllr Raj Patel")],
    )
    second = LLMDetailedRedactionResultFormat(
        redactions=[
            _decision("Joan Brown", ["T1", "M2"], "high", "second reason"),
            _decision("Mary Jones"),
        ],
        excluded=[_exclusion("Mary Jones")],
    )

    redactions, excluded = merge_redaction_decisions([first, second])

    assert redactions == (
        {
            "text": "Joan Brown",
            "data_subject": "third_party",
            "clause_ids": ["T1", "M2"],
            "reason": "first reason",
            "confidence": "high",
            "chunk_count": 2,
        },
        {
            "text": "Mary Jones",
            "data_subject": "third_party",
            "clause_ids": ["T1"],
            "reason": "a reason",
            "confidence": "high",
            "chunk_count": 1,
        },
    )
    assert excluded == (
        {"text": "Cllr Raj Patel", "clause_ids": ["D1"], "reason": "councillor"},
    )


@patch.object(
    LLMTextAnalyser,
    "create_api_message",
    return_value=[{"role": "system", "content": "system prompt"}],
)
@patch.object(LLMTextAnalyser, "_num_tokens_consumed", return_value=10)
def test__llm_text_analyser___analyse_text_chunk__include_rationale(
    mock_num_tokens_consumed, mock_create_api_message
):
    """
    - Given an analyser with include_rationale enabled
    - When I analyse a chunk
    - Then the detailed format is requested, the strings are returned, and the
      decisions' reasons are not logged
    """
    completion = create_mock_detailed_chat_completion(
        [_decision("Joan Brown", reason="uninvolved neighbour")],
        [_exclusion("Cllr Raj Patel")],
    )
    llm_text_analyser = LLMTextAnalyser(
        LLMTextAnalyserConfig(model="gpt-5.6-luna", include_rationale=True)
    )
    llm_text_analyser.request_semaphore = Mock()
    llm_text_analyser.token_semaphore = Mock()

    with patch.object(
        LLMTextAnalyser, "invoke_chain", return_value=completion
    ) as mock_invoke_chain:
        response, redaction_strings = llm_text_analyser._analyse_text_chunk(
            system_prompt="system prompt", user_prompt=""
        )

    assert mock_invoke_chain.call_args.args[1] is LLMDetailedRedactionResultFormat
    assert response is completion
    assert redaction_strings == ["Joan Brown"]
    logged = " ".join(str(c) for c in LoggingUtil.log_info.call_args_list)
    assert "uninvolved neighbour" not in logged
    assert "Cllr Raj Patel" not in logged


def test__llm_text_analyser__analyse_text__include_rationale():
    """
    - Given an analyser with include_rationale enabled
    - When I analyse two chunks that both redact the same span
    - Then the result has one merged decision for it alongside the strings
    """
    llm_text_analyser = LLMTextAnalyser(
        LLMTextAnalyserConfig(model="gpt-5.6-luna", include_rationale=True)
    )
    results = iter(
        [
            (
                create_mock_detailed_chat_completion([_decision("Joan Brown")]),
                ["Joan Brown"],
            ),
            (
                create_mock_detailed_chat_completion(
                    [_decision("Joan Brown", confidence="low")],
                    [_exclusion("Cllr Raj Patel")],
                ),
                ["Joan Brown"],
            ),
        ]
    )

    with patch.object(
        LLMTextAnalyser,
        "_analyse_text_chunk",
        side_effect=lambda *args, **kwargs: next(results),
    ):
        actual_result = llm_text_analyser.analyse_text(
            system_prompt="system prompt", text_chunks=["chunk A", "chunk B"]
        )

    assert actual_result.redaction_strings == ("Joan Brown",)
    assert [d["text"] for d in actual_result.redaction_decisions] == ["Joan Brown"]
    assert actual_result.redaction_decisions[0]["confidence"] == "high"
    assert actual_result.redaction_decisions[0]["chunk_count"] == 2
    assert [d["text"] for d in actual_result.excluded_decisions] == ["Cllr Raj Patel"]


def test__llm_text_analyser__analyse_text__no_rationale_has_no_decisions():
    llm_text_analyser = LLMTextAnalyser(LLMTextAnalyserConfig(model="gpt-5.6-luna"))

    with patch.object(
        LLMTextAnalyser,
        "_analyse_text_chunk",
        return_value=(create_mock_chat_completion(["string A"]), ["string A"]),
    ):
        actual_result = llm_text_analyser.analyse_text(
            system_prompt="system prompt", text_chunks=["chunk A"]
        )

    assert actual_result.redaction_decisions == ()
    assert actual_result.excluded_decisions == ()


@patch.object(LLMTextAnalyser, "analyse_text", LLMTextAnalyser.analyse_text.__wrapped__)
def test__llm_text_analyser__analyse_text__check_pool_size():
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna", max_concurrent_requests=4
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)

    with (
        patch.object(
            ThreadPoolExecutor, "submit", return_value=None
        ) as mock_executor_submit,
        patch(f"{MODULE}.as_completed", return_value=[]),
        patch.object(
            ThreadPoolExecutor, "__init__", return_value=None
        ) as mock_executor_init,
        patch.object(
            ThreadPoolExecutor, "__exit__", return_value=None
        ) as mock_executor_exit,
    ):
        llm_text_analyser.analyse_text(
            system_prompt="system prompt",
            text_chunks=["redaction string A", "redaction string B"] * 20,
        )

    mock_executor_init.assert_called_once_with(max_workers=4)
    assert mock_executor_submit.call_count == 40
    mock_executor_exit.assert_called_once()


@patch.object(LLMTextAnalyser, "analyse_text", LLMTextAnalyser.analyse_text.__wrapped__)
@patch(f"{MODULE}.os.cpu_count", return_value=8)
def test__llm_text_analyser__analyse_text__override_pool_size(mock_cpu_count):
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna", max_concurrent_requests=4
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)

    # Override to test that the value is respected
    llm_text_analyser.config.max_concurrent_requests = 40

    with (
        patch.object(
            ThreadPoolExecutor, "submit", return_value=None
        ) as mock_executor_submit,
        patch(f"{MODULE}.as_completed", return_value=[]),
        patch.object(
            ThreadPoolExecutor, "__init__", return_value=None
        ) as mock_executor_init,
        patch.object(
            ThreadPoolExecutor, "__exit__", return_value=None
        ) as mock_executor_exit,
    ):
        llm_text_analyser.analyse_text(
            system_prompt="system prompt",
            text_chunks=["redaction string A", "redaction string B"] * 20,
        )

    max_workers = min(32, (os.cpu_count() or 1) + 4)
    assert llm_text_analyser.config.max_concurrent_requests == max_workers

    mock_executor_init.assert_called_once_with(max_workers=max_workers)
    assert mock_executor_submit.call_count == 40
    mock_executor_exit.assert_called_once()


@patch(f"{MODULE}.get_encoding")
@patch("time.sleep", return_value=None)
def test__llm_text_analyser__analyse_text__budget_exceeded(
    mock_time_sleep, mock_get_encoding
):
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
        budget=12.0,
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)
    llm_text_analyser.input_token_cost = 1
    llm_text_analyser.output_token_cost = 2

    with patch.object(LLMTextAnalyser, "invoke_chain") as mock_invoke_chain:
        mock_invoke_chain.side_effect = [
            create_mock_chat_completion(["string A"]),
            create_mock_chat_completion(["string B"]),
        ]
        actual_result = llm_text_analyser.analyse_text(
            system_prompt="system prompt",
            text_chunks=["redaction string A", "redaction string B"],
        )

    # Only first call processed
    assert (actual_result.redaction_strings == ("string A",)) or (
        actual_result.redaction_strings == ("string B",)
    )


def test__llm_text_analyser___check_budget__raises_when_exceeded():
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
        budget=10.0,
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)
    llm_text_analyser.total_cost = 10.0

    with pytest.raises(RuntimeError, match="Budget of £10.00 exceeded"):
        llm_text_analyser._check_budget()


def test__llm_text_analyser___check_budget__no_raise_when_under_budget():
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
        budget=10.0,
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)
    llm_text_analyser.total_cost = 5.0

    # Should not raise
    llm_text_analyser._check_budget()


def test__llm_text_analyser___check_budget__no_raise_when_no_budget_set():
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)
    llm_text_analyser.total_cost = 99999.0

    # Should not raise when no budget configured
    llm_text_analyser._check_budget()


@patch(f"{MODULE}.get_encoding")
def test__llm_text_analyser__analyse_text__single_chunk_sequential(mock_get_encoding):
    """When there is only 1 chunk, max_workers=1 and processing is sequential
    (no ThreadPoolExecutor)"""
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)
    llm_text_analyser.request_semaphore = Mock()
    llm_text_analyser.token_semaphore = Mock()
    llm_text_analyser.input_token_cost = 1
    llm_text_analyser.output_token_cost = 2

    with (
        patch.object(LLMTextAnalyser, "invoke_chain") as mock_invoke_chain,
        patch.object(
            ThreadPoolExecutor, "__init__", return_value=None
        ) as mock_executor_init,
    ):
        mock_invoke_chain.return_value = create_mock_chat_completion(["string A"])
        actual_result = llm_text_analyser.analyse_text(
            system_prompt="system prompt",
            text_chunks=["single chunk"],
        )

    # Should not have created a ThreadPoolExecutor
    mock_executor_init.assert_not_called()

    assert actual_result.redaction_strings == ("string A",)
    assert actual_result.metadata.request_count == 1


@patch(f"{MODULE}.get_encoding")
def test__llm_text_analyser__analyse_text__sequential_failed_chunk_counted(
    mock_get_encoding,
):
    """
    - Given a chunk whose analysis returns no result after all retries
    - When analyse_text processes chunks sequentially
    - Then processing should continue and the failure should be counted in metadata
    """
    llm_text_analyser = LLMTextAnalyser(
        LLMTextAnalyserConfig(model="gpt-5.6-luna", max_concurrent_requests=1)
    )

    with patch.object(
        LLMTextAnalyser,
        "_analyse_text_chunk",
        side_effect=[None, (create_mock_chat_completion(["string B"]), ["string B"])],
    ):
        actual_result = llm_text_analyser.analyse_text(
            system_prompt="system prompt",
            text_chunks=["chunk A", "chunk B"],
        )

    assert actual_result.redaction_strings == ("string B",)
    assert actual_result.metadata.request_count == 2
    assert actual_result.metadata.failed_chunk_count == 1


@patch.object(LLMTextAnalyser, "analyse_text", LLMTextAnalyser.analyse_text.__wrapped__)
def test__llm_text_analyser__analyse_text__parallel_failed_chunk_counted():
    """
    - Given a chunk whose analysis returns no result after all retries
    - When analyse_text processes chunks in parallel
    - Then the failure should be counted in metadata
    """
    llm_text_analyser = LLMTextAnalyser(
        LLMTextAnalyserConfig(model="gpt-5.6-luna", max_concurrent_requests=4)
    )
    results = {
        "chunk A": None,
        "chunk B": (create_mock_chat_completion(["string B"]), ["string B"]),
    }

    with patch.object(
        LLMTextAnalyser,
        "_analyse_text_chunk",
        side_effect=lambda system_prompt, user_prompt: results[user_prompt],
    ):
        actual_result = llm_text_analyser.analyse_text(
            system_prompt="system prompt",
            text_chunks=["chunk A", "chunk B"],
        )

    assert actual_result.redaction_strings == ("string B",)
    assert actual_result.metadata.failed_chunk_count == 1


@patch.object(LLMTextAnalyser, "analyse_text", LLMTextAnalyser.analyse_text.__wrapped__)
def test__llm_text_analyser__analyse_text__max_workers_limited_by_chunk_count():
    """max_workers should be min(max_concurrent_requests, chunk_count)"""
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna", max_concurrent_requests=10
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)

    with (
        patch.object(
            ThreadPoolExecutor, "submit", return_value=None
        ) as mock_executor_submit,
        patch(f"{MODULE}.as_completed", return_value=[]),
        patch.object(
            ThreadPoolExecutor, "__init__", return_value=None
        ) as mock_executor_init,
        patch.object(
            ThreadPoolExecutor, "__exit__", return_value=None
        ) as mock_executor_exit,
    ):
        llm_text_analyser.analyse_text(
            system_prompt="system prompt",
            text_chunks=["chunk A", "chunk B", "chunk C"],  # 3 chunks < 10 workers
        )

    # max_workers should be 3 (chunk count), not 10 (max_concurrent_requests)
    mock_executor_init.assert_called_once_with(max_workers=3)
    assert mock_executor_submit.call_count == 3
    mock_executor_exit.assert_called_once()


@patch(f"{MODULE}.get_encoding")
def test__llm_text_analyser__analyse_text__sequential_budget_exceeded(
    mock_get_encoding,
):
    """Budget check in sequential path should stop processing after budget is exceeded"""
    llm_text_analyser_config = LLMTextAnalyserConfig(
        model="gpt-5.6-luna",
        budget=12.0,
        max_concurrent_requests=1,
    )
    llm_text_analyser = LLMTextAnalyser(llm_text_analyser_config)
    llm_text_analyser.input_token_cost = 1
    llm_text_analyser.output_token_cost = 2

    with patch.object(LLMTextAnalyser, "invoke_chain") as mock_invoke_chain:
        mock_invoke_chain.side_effect = [
            create_mock_chat_completion(["string A"]),
            create_mock_chat_completion(["string B"]),
            create_mock_chat_completion(["string C"]),
        ]
        actual_result = llm_text_analyser.analyse_text(
            system_prompt="system prompt",
            text_chunks=["chunk A", "chunk B", "chunk C"],
        )

    # Budget of 12 is exceeded after first call (cost = 13), so only 1 chunk processed
    assert actual_result.redaction_strings == ("string A",)
    assert mock_invoke_chain.call_count == 1
