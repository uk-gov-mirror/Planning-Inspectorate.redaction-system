import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from src.analysis.utils import TokenSemaphore, get_max_workers


def test__token_semaphore__acquire():
    token_semaphore = TokenSemaphore(max_tokens=100)
    token_semaphore.acquire(50)
    assert token_semaphore.tokens == 50


def test__token_semaphore__release():
    token_semaphore = TokenSemaphore(max_tokens=100)
    token_semaphore.acquire(50)
    token_semaphore.release(30)
    assert token_semaphore.tokens == 80


def token_semaphore_task(self, tokens: int):
    self.acquire(tokens)
    # Simulate some processing
    self.release(tokens)


def test__token_semaphore__insufficient_tokens():
    # Test that in a parallel scenario, only one thread waits when tokens are insufficient
    # Define a task that tries to acquire more tokens than available
    token_semaphore = TokenSemaphore(max_tokens=100)
    token_semaphore.task = token_semaphore_task

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit tasks to the executor
        future_to_semaphore = {
            executor.submit(token_semaphore.task, token_semaphore, x): x
            for x in [80, 80]
        }
        for future in as_completed(future_to_semaphore):
            # Ensure the task completed successfully
            assert future.done()
            assert token_semaphore.tokens >= 0

    assert token_semaphore.tokens == 100


def test__token_semaphore__timeout():
    # Test that acquiring tokens times out appropriately when tokens are insufficient
    token_semaphore = TokenSemaphore(max_tokens=100, timeout=1)
    token_semaphore.acquire(100)

    with pytest.raises(TimeoutError):
        token_semaphore.acquire(10)

    assert token_semaphore.tokens == 0


def test__get_max_workers():
    max_workers = min(32, (os.cpu_count() or 1) + 4)
    assert get_max_workers(2) == 2
    assert get_max_workers(0) == 1
    assert get_max_workers(1000) == max_workers
    assert get_max_workers() == max_workers


def test__get_max_workers__env_var_replaces_cpu_cap(monkeypatch):
    monkeypatch.setenv("TEST_MAX_WORKERS", "16")
    assert get_max_workers(env_var="TEST_MAX_WORKERS") == 16
    assert get_max_workers(1000, env_var="TEST_MAX_WORKERS") == 16
    assert get_max_workers(2, env_var="TEST_MAX_WORKERS") == 2


def test__get_max_workers__env_var_is_capped_at_32(monkeypatch):
    monkeypatch.setenv("TEST_MAX_WORKERS", "100")
    assert get_max_workers(env_var="TEST_MAX_WORKERS") == 32
