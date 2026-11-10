import os
from threading import Condition, Lock

from ..monitoring import LoggingUtil, log_to_appins


class TokenSemaphore:
    """Semaphore for limiting the number of tokens used in parallel requests.

    Based on https://github.com/mahmoudhage21/Parallel-LLM-API-Requester/blob/main/src/Parallel_LLM_API_Requester.py
    """

    def __init__(self, max_tokens: int, timeout: float = 60.0):
        self.tokens = max_tokens
        self.timeout = timeout
        self._condition = Condition(Lock())

    def __repr__(self) -> str:
        cls = self.__class__
        return (
            f"<{cls.__module__}.{cls.__qualname__} at {id(self):#x}:"
            f" tokens={self.tokens}, timeout={self.timeout}>"
        )

    @log_to_appins
    def acquire(self, tokens: int) -> None:
        """Acquire the specified number of tokens from the semaphore."""
        with self._condition:
            # Wait until enough tokens are available
            while tokens > self.tokens:
                LoggingUtil().log_info("Waiting for tokens to be released...")
                # returns True if notified (tokens available), False on timeout
                available = self._condition.wait(timeout=self.timeout)
                if not available:
                    raise TimeoutError(
                        "Timeout while waiting for tokens to be released."
                    )
            self.tokens -= tokens

    __enter__ = acquire

    @log_to_appins
    def release(self, tokens: int) -> None:
        """Release the specified number of tokens back to the semaphore."""
        with self._condition:
            self.tokens += tokens
            self._condition.notify_all()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def get_max_workers(n: int | None = None, env_var: str | None = None) -> int:
    """Determine the number of worker threads to use, capped at 32 or
    (os.cpu_count() or 1) + 4.

    :param int n: Desired number of workers. If None, use system default.
    :param str env_var: Name of an environment variable that replaces the
    cpu-based cap when set. The cap is still limited to 32.
    :return int: The number of worker threads to use.
    """
    max_workers = min(32, (os.cpu_count() or 1) + 4)
    configured_cap = os.environ.get(env_var) if env_var else None
    if configured_cap:
        max_workers = min(32, max(1, int(configured_cap)))
    if n is not None:
        if n < 1:
            return 1
        elif n > max_workers:
            return max_workers
        else:
            return n
    return max_workers
