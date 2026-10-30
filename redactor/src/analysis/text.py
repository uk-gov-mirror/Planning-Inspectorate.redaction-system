import dataclasses
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
from typing import ClassVar, Literal

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from openai import (
    AzureOpenAI,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    RateLimitError,
)
from openai.types.chat.chat_completion import CompletionUsage
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_any,
    retry_if_exception_message,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tiktoken import get_encoding

from ..monitoring import LoggingUtil, TimerUtil, log_to_appins
from ..types import LLMRedactionResultFormat, LLMTextRedactionResult
from .utils import TokenSemaphore, get_max_workers

load_dotenv(verbose=True)


@log_to_appins
def handle_last_retry_error(retry_state):
    LoggingUtil().log_info(
        f"All retry attempts failed: {retry_state.outcome.exception()}\n"
        "Returning None for this chunk."
    )


@log_to_appins
def update_max_tokens(retry_state):
    # Double max completions for next retry attempt, up to max of 8000
    # Only used when LengthFinishReasonError is raised
    # On the first retry the kwarg is unset, so start from the analyser's own limit
    current = (
        retry_state.kwargs.get("max_completion_tokens")
        or retry_state.args[0].config.max_tokens
        or 1000
    )
    retry_state.kwargs.update({"max_completion_tokens": min(current * 2, 8000)})
    LoggingUtil().log_info(
        f"Updating max_completion_tokens to {retry_state.kwargs['max_completion_tokens']}"
        " for next attempt."
    )


class LLMTextAnalyserConfig(BaseModel):
    model: str
    """The LLM model to use"""
    max_tokens: int | None = 1000
    """Maximum number of tokens per completion, including reasoning tokens"""
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    """Reasoning effort for reasoning models. None uses the model's default"""
    request_rate_limit: int | None = None
    """Maximum number of requests per minute. Defaults to 20% of model max RPM."""
    token_rate_limit: int | None = None
    """Number of tokens allowed per minute. Defaults to 20% of model max TPM."""
    max_concurrent_requests: int | None = None
    """Number of concurrent requests to allow. Assigns the number of threads."""
    token_encoding_name: str | None = None
    """The token encoding name to use for estimating token counts"""
    n: int | None = 1
    """Number of completions to generate per prompt"""
    budget: float | None = None
    """The budget in GBP for LLM usage"""
    token_timeout: float | None = 60.0
    """The timeout in seconds for acquiring tokens from the token semaphore"""
    request_timeout: float | None = 60.0
    """The timeout in seconds for acquiring request semaphore"""

    def __init__(self, **data):
        super().__init__(**data)
        if self.token_encoding_name is None:
            if self.model.startswith("gpt-5") or self.model.startswith("gpt-4o"):
                self.token_encoding_name = "o200k_base"  # nosec: B105
            elif self.model.startswith("gpt-4") or self.model.startswith("gpt-3.5"):
                self.token_encoding_name = "cl100k_base"  # nosec: B105
            else:
                self.token_encoding_name = "p50k_base"  # nosec: B105


class LLMTextAnalyser:
    """
    Class that handles the interaction with a large-language model hosted on Azure
    """

    # Azure Foundry quota limits and cost in GBP per 1M tokens - correct on 06/01/26
    OPENAI_MODELS: ClassVar[dict[str, dict[str, int]]] = {
        "gpt-5.6-luna": {
            "token_rate_limit": 3000000,  # nosec: B105
            "request_rate_limit": 3000,
            "input_cost": 16,
            "output_cost": 91,
            "cached_input_cost": 2,
            "cache_write_cost": 19,
        },
    }
    USER_PROMPT_TEMPLATE = PromptTemplate(input_variables=["chunk"], template="{chunk}")

    def __init__(
        self,
        config: LLMTextAnalyserConfig,
    ):
        self.lock = Lock()
        self.config: LLMTextAnalyserConfig = config

        # Initialise OpenAI client for Azure
        self.azure_endpoint = os.environ.get("OPENAI_ENDPOINT", None)
        credential = ChainedTokenCredential(
            ManagedIdentityCredential(), AzureCliCredential()
        )
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        LoggingUtil().log_info(
            f"Establishing connection to the LLM at {self.azure_endpoint}"
        )
        self.llm = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_version="2024-12-01-preview",
            azure_ad_token_provider=token_provider,
        )

        # Validates and sets input_token_cost, completion_token_cost, token_rate_limit and request_rate_limit
        self._set_model_details()

        # Validate and set max concurrent requests
        self._set_workers(self.config.max_concurrent_requests)
        self.request_semaphore = Semaphore(self.config.max_concurrent_requests)

        self.token_semaphore = TokenSemaphore(
            self.config.token_rate_limit, self.config.token_timeout
        )

        self.input_token_count = 0
        self.cached_input_token_count = 0
        self.cache_write_token_count = 0
        self.reasoning_token_count = 0
        self.completion_token_count = 0
        self.total_token_count = 0
        self.total_cost = 0.0  # Total cost of LLM calls in GBP

    @log_to_appins
    def _set_model_details(self):
        instance_quota_allocation = 0.5
        try:
            # Get specified model
            model_details = self.OPENAI_MODELS[self.config.model]

            # Set cost per token in GBP
            multiplier = 1e-8
            self.input_token_cost = model_details["input_cost"] * multiplier
            self.completion_token_cost = model_details["output_cost"] * multiplier
            self.cached_input_token_cost = (
                model_details["cached_input_cost"] * multiplier
            )
            self.cache_write_token_cost = model_details["cache_write_cost"] * multiplier

            # Validate and set token rate limit per minute
            default_token_rate_limit = int(
                model_details["token_rate_limit"] * instance_quota_allocation
            )

            token_limit = self.config.token_rate_limit
            if token_limit is not None:
                if token_limit < 1:
                    self.config.token_rate_limit = default_token_rate_limit
                if token_limit > model_details["token_rate_limit"]:
                    self.config.token_rate_limit = model_details["token_rate_limit"]
                    LoggingUtil().log_info(
                        f"Token rate limit for model {self.config.model} exceeds maximum. "
                        f"Setting to maximum of {self.config.token_rate_limit} tokens per minute."
                    )
            else:  # default to 20% of max token rate limit
                self.config.token_rate_limit = default_token_rate_limit

            default_request_rate_limit = int(
                model_details["request_rate_limit"] * instance_quota_allocation
            )

            # Validate and set request rate limit per minute
            req_limit = self.config.request_rate_limit
            if req_limit is not None:
                if req_limit < 1:
                    self.config.request_rate_limit = default_request_rate_limit
                if req_limit > model_details["request_rate_limit"]:
                    self.config.request_rate_limit = model_details["request_rate_limit"]
                    LoggingUtil().log_info(
                        f"Request rate limit for model {self.config.model} exceeds maximum. "
                        f"Setting to maximum of {self.config.request_rate_limit} requests per minute."
                    )
            else:  # default to 20% of max request rate limit
                self.config.request_rate_limit = default_request_rate_limit
        except KeyError:
            raise ValueError(f"Model {self.config.model} is not supported.")

    def _set_workers(self, n: int | None = None) -> int:
        """Determine the number of worker threads to use, capped at 32 or
        (os.cpu_count() or 1) + 4."""
        self.config.max_concurrent_requests = get_max_workers(n)

    @log_to_appins
    def _num_tokens_consumed(
        self,
        api_messages: str,
    ):
        """
        Estimate the number of tokens consumed by a request to the LLM

        Based on https://github.com/openai/openai-cookbook/blob/970d8261fbf6206718fe205e88e37f4745f9cf76/examples/api_request_parallel_processor.py#L339
        """
        encoding = get_encoding(self.config.token_encoding_name)
        completion_tokens = self.config.n * self.config.max_tokens
        n_tokens = 0
        try:
            for message in api_messages:
                n_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    n_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        n_tokens += -1  # role is always required and always 1 token
            n_tokens += 2  # every reply is primed with <im_start>assistant

            total_tokens = n_tokens + completion_tokens
            return total_tokens
        except Exception as e:  # noqa: BLE001
            LoggingUtil().log_exception_with_message(
                "An error occurred while counting tokens:", e
            )
            return 0

    def create_api_message(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def invoke_chain(
        self,
        api_messages: str,
        response_format: BaseModel,
        max_completion_tokens: int | None = None,
    ) -> ParsedChatCompletion:
        kwargs = {
            "model": self.config.model,
            "messages": api_messages,
            "response_format": response_format,
            "max_completion_tokens": max_completion_tokens,
        }
        if self.config.reasoning_effort:
            kwargs["reasoning_effort"] = self.config.reasoning_effort

        return self.llm.chat.completions.parse(**kwargs)

    @log_to_appins
    # exponential backoff to increase wait time between retries https://platform.openai.com/docs/guides/rate-limits
    # Only retry if there is a rate limit exception. All other errors are logged and skipped
    @retry(
        retry=retry_any(
            retry_if_exception_type(
                (
                    RateLimitError,  # API rate limit exceeded
                    TimeoutError,  # Timeout while waiting for semaphore
                    LengthFinishReasonError,  # LLM response truncated due to length
                    ContentFilterFinishReasonError,  # LLM response blocked by content filter
                    ValidationError,  # LLM response could not be parsed into expected format
                )
            ),
            retry_if_exception_message(  # LLM response parsing errors
                "'str' object has no attribute 'choices'"
            ),
        ),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
        before_sleep=lambda retry_state: LoggingUtil().log_info(
            "Retrying LLM analysis..."
        ),
        retry_error_callback=handle_last_retry_error,
        after=lambda retry_state: (
            update_max_tokens(retry_state)
            if isinstance(retry_state.outcome.exception(), LengthFinishReasonError)
            else None
        ),
    )
    def _analyse_text_chunk(
        self,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int | None = None,
    ) -> tuple[ParsedChatCompletion, list[str]]:
        """Redact a single chunk of text using the LLM."""
        # Chunk hash to distinguish between messages when multithreading
        chunk_hash_string = f"(chunk ID {hash(user_prompt)})"

        # Estimate tokens for the request
        api_messages = self.create_api_message(system_prompt, user_prompt)
        estimated_tokens = self._num_tokens_consumed(api_messages)

        # Set completion tokens
        if max_completion_tokens is None:
            max_completion_tokens = self.config.max_tokens

        # Acquire request semaphore
        thread_available = self.request_semaphore.acquire(
            timeout=self.config.request_timeout
        )  # returns True if acquired, False on timeout
        if not thread_available:
            exception = TimeoutError(
                f"{chunk_hash_string} Timeout while waiting for request semaphore to be available."
            )
            LoggingUtil().log_exception(exception)
            raise TimeoutError

        try:
            # Acquire token semaphore
            try:
                self.token_semaphore.acquire(estimated_tokens)
            except TimeoutError as te:
                LoggingUtil().log_exception_with_message(
                    f"{chunk_hash_string} Timeout while waiting for tokens to be released :",
                    te,
                )
                raise

            # Invoke LLM
            try:
                LoggingUtil().log_info(
                    f"{chunk_hash_string} The following messages were sent to the LLM: {api_messages}"
                )
                response = self.invoke_chain(
                    api_messages, LLMRedactionResultFormat, max_completion_tokens
                )
                LoggingUtil().log_info(
                    f"{chunk_hash_string} LLM response received: {response}"
                )
                usage = response.usage

                response_cleaned: LLMRedactionResultFormat = response.choices[
                    0
                ].message.parsed
                redaction_strings = response_cleaned.redaction_strings
                return response, redaction_strings
            except LengthFinishReasonError as lfe:
                LoggingUtil().log_exception_with_message(
                    f"{chunk_hash_string} The LLM response was truncated due to length"
                    f" (completion tokens: {self.config.max_tokens}):",
                    lfe,
                )
                if lfe.completion and lfe.completion.usage:
                    usage = lfe.completion.usage
                else:
                    usage = None
                raise
            except Exception as e:
                LoggingUtil().log_exception_with_message(
                    f"{chunk_hash_string} An error occurred while processing the chunk:",
                    e,
                )
                usage = None
                raise
            finally:
                # Update token counts and costs
                with self.lock:
                    self._compute_costs(usage)
                # Release token semaphore
                self.token_semaphore.release(estimated_tokens)

        finally:
            # Release request semaphore
            self.request_semaphore.release()
            time.sleep(60 / self.config.request_rate_limit)  # Rate limiting delay

    def _compute_costs(self, usage: CompletionUsage = None):
        if usage is None:
            return

        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        cached = getattr(prompt_details, "cached_tokens", None) or 0
        cache_write = getattr(prompt_details, "cache_write_tokens", None) or 0
        reasoning = getattr(completion_details, "reasoning_tokens", None) or 0

        self.input_token_count += usage.prompt_tokens
        self.completion_token_count += usage.completion_tokens
        self.total_token_count += usage.total_tokens
        self.cached_input_token_count += cached
        self.cache_write_token_count += cache_write
        self.reasoning_token_count += reasoning

        self.total_cost += (
            self.input_token_cost * (usage.prompt_tokens - cached - cache_write)
            + self.completion_token_cost * usage.completion_tokens
            + self.cached_input_token_cost * cached
            + self.cache_write_token_cost * cache_write
        )

    def _check_budget(self):
        # Check budget after each request
        if self.config.budget and self.total_cost >= self.config.budget:
            raise RuntimeError(
                f"Budget of £{self.config.budget:.2f} exceeded with total cost "
                f"£{self.total_cost:.2f}. Stopping further requests."
            )

    @log_to_appins
    def analyse_text(
        self,
        system_prompt: str,
        text_chunks: list[str],
    ) -> LLMTextRedactionResult:
        """Analyse multiple text chunks for redaction in parallel using the LLM.

        Based on https://github.com/mahmoudhage21/Parallel-LLM-API-Requester/blob/main/src/Parallel_LLM_API_Requester.py
        """
        chunk_count = len(text_chunks)
        character_count = sum(len(chunk) for chunk in text_chunks)
        word_count = sum(
            len([x.strip() for x in chunk.split(" ")]) for chunk in text_chunks
        )
        with TimerUtil() as timer:
            chunk_hashes = [
                {"chunk": chunk, "hash": hash(chunk)} for chunk in text_chunks
            ]
            LoggingUtil().log_info(
                f"The following text chunks will be processed: {json.dumps(chunk_hashes, indent=4)}"
            )

            # Initialise LLM interface
            request_counter = 0
            text_to_redact = []
            responses: list[ParsedChatCompletion] = []
            failed_chunk_count = 0

            # Check max concurrent requests
            if self.config.max_concurrent_requests > 32:
                self._set_workers(self.config.max_concurrent_requests)
                LoggingUtil().log_info(
                    "Max concurrent requests exceeds maximum."
                    f" Setting to {self.config.max_concurrent_requests}."
                )

            # Set max workers to the minimum of max_concurrent_requests and number of chunks
            max_workers = min(self.config.max_concurrent_requests, chunk_count)
            LoggingUtil().log_info(
                f"Starting text analysis on {chunk_count} chunks with {max_workers} "
                "workers."
            )
            if max_workers == 0:
                LoggingUtil().log_info(
                    "No text chunks to process. Returning empty result."
                )
            elif max_workers == 1:
                # Process chunks sequentially if only one worker is allowed
                for chunk in text_chunks:
                    request_counter += 1
                    try:
                        response, redaction_strings = self._analyse_text_chunk(
                            system_prompt,
                            self.USER_PROMPT_TEMPLATE.format(chunk=chunk),
                        )
                        responses.append(response)
                        text_to_redact.extend(redaction_strings)
                    except Exception as e:  # noqa: BLE001
                        failed_chunk_count += 1
                        LoggingUtil().log_exception_with_message(
                            f"Error processing chunk {hash(chunk)}",
                            e,
                        )

                    # Check budget after each request
                    try:
                        self._check_budget()
                    except RuntimeError as re:
                        LoggingUtil().log_exception(re)
                        break
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit tasks to the executor
                    future_to_chunk = {
                        executor.submit(
                            self._analyse_text_chunk,
                            system_prompt,
                            self.USER_PROMPT_TEMPLATE.format(chunk=chunk),
                        ): chunk
                        for chunk in text_chunks
                    }
                    for future in as_completed(future_to_chunk):
                        chunk = future_to_chunk[future]
                        request_counter += 1

                        try:
                            # Get redaction result for chunk and append to overall results
                            response, redaction_strings = future.result()
                            responses.append(response)
                            text_to_redact.extend(redaction_strings)
                        except Exception as e:  # noqa: BLE001
                            failed_chunk_count += 1
                            LoggingUtil().log_exception_with_message(
                                f"Error processing chunk {hash(chunk)}",
                                e,
                            )

                        # Check budget after each request
                        try:
                            self._check_budget()
                        except RuntimeError as re:
                            LoggingUtil().log_exception(re)
                            break

            # Remove duplicates
            text_to_redact_cleaned = tuple(dict.fromkeys(text_to_redact))

        # Collect metrics
        metadata = LLMTextRedactionResult.LLMResultMetadata(
            request_count=request_counter,
            input_token_count=self.input_token_count,
            completion_token_count=self.completion_token_count,
            total_token_count=self.total_token_count,
            total_cost=self.total_cost,
            cached_input_token_count=self.cached_input_token_count,
            cache_write_token_count=self.cache_write_token_count,
            reasoning_token_count=self.reasoning_token_count,
            failed_chunk_count=failed_chunk_count,
        )
        run_metrics = {
            "llm_analysis_time": timer.elapsed_time,
            "llm_character_count": character_count,
            "llm_approx_text_word_count": word_count,
            "llm_text_chunk_count": chunk_count,
            "llm_metadata": dataclasses.asdict(metadata),
        }
        result = LLMTextRedactionResult(
            rule_name="",
            run_metrics=run_metrics,
            redaction_strings=text_to_redact_cleaned,
            metadata=metadata,
        )

        return result
