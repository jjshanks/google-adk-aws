"""Retry logic and circuit breaker for S3 operations."""

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Callable, Optional, Tuple, Type

from botocore.exceptions import ClientError

from .exceptions import S3ConnectionError, S3ThrottleError

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_errors: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_errors = retryable_errors or (
            ClientError,
            S3ConnectionError,
            S3ThrottleError,
        )


class CircuitBreaker:
    """Circuit breaker pattern for S3 operations."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            if self.state == "OPEN":
                if (
                    self.last_failure_time
                    and time.time() - self.last_failure_time < self.timeout
                ):
                    raise S3ConnectionError("Circuit breaker is OPEN")
                else:
                    self.state = "HALF_OPEN"

            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception:
                self._on_failure()
                raise

        return wrapper

    def _on_success(self) -> None:
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


def with_retry(config: Optional[RetryConfig] = None):  # type: ignore
    """Decorator to add retry logic to async functions."""
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_errors as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        logger.error(
                            f"Max retry attempts ({config.max_attempts}) "
                            f"exceeded for {func.__name__}"
                        )
                        break

                    # Calculate delay
                    delay = min(
                        config.base_delay * (config.exponential_base**attempt),
                        config.max_delay,
                    )

                    if config.jitter:
                        delay *= 0.5 + random.random() * 0.5

                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    await asyncio.sleep(delay)
                except Exception as e:
                    # Non-retryable error
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise

            if last_exception:
                raise last_exception
            else:
                raise Exception("No exception to raise")

        return wrapper

    return decorator


def is_throttle_error(error: ClientError) -> bool:
    """Check if error is due to S3 throttling."""
    error_code = error.response.get("Error", {}).get("Code", "")
    return error_code in [
        "Throttling",
        "RequestLimitExceeded",
        "TooManyRequests",
        "SlowDown",
    ]


def should_retry_error(error: Exception) -> bool:
    """Determine if error should trigger retry."""
    if isinstance(error, ClientError):
        error_code = error.response.get("Error", {}).get("Code", "")

        # Retryable S3 errors
        retryable_codes = [
            "InternalError",
            "ServiceUnavailable",
            "Throttling",
            "RequestTimeout",
            "RequestLimitExceeded",
        ]

        return error_code in retryable_codes

    return isinstance(error, (S3ConnectionError, S3ThrottleError))
