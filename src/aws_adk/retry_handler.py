"""Enhanced retry handler with comprehensive error context."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from .exceptions import (
    S3ConcurrencyError,
    S3ConnectionError,
    S3ThrottleError,
    map_boto3_error,
)

logger = logging.getLogger(__name__)


@dataclass
class RetryContext:
    """Context information for retry operations."""

    operation: str
    attempt: int
    total_attempts: int
    last_error: Optional[Exception]
    start_time: float
    delay: float

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


class RetryConfig:
    """Enhanced retry configuration with operation-specific settings.

    Provides comprehensive retry behavior configuration including multiple backoff
    strategies, time constraints, and error filtering.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_errors: Optional[Tuple[Type[Exception], ...]] = None,
        max_total_time: Optional[float] = None,
        backoff_strategy: str = "exponential",
    ):
        """Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts before giving up
            base_delay: Base delay in seconds for retry calculations
            max_delay: Maximum delay in seconds between retries
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to apply random jitter (±50%) to delay times
            retryable_errors: Tuple of exception types that should trigger retries
            max_total_time: Maximum total time in seconds for all retry attempts.
                If specified, retries will stop when this time limit is exceeded,
                even if max_attempts hasn't been reached. When None, only
                max_attempts limits retries.
            backoff_strategy: Strategy for calculating retry delays. Supported values:
                - "exponential": delay = base_delay * (exponential_base ** attempt)
                - "linear": delay = base_delay * (attempt + 1)
                - "fixed": delay = base_delay (constant delay)
                All strategies respect max_delay as an upper bound.
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.max_total_time = max_total_time
        self.backoff_strategy = backoff_strategy

        self.retryable_errors = retryable_errors or (
            S3ConnectionError,
            S3ThrottleError,
            S3ConcurrencyError,
        )

    def calculate_delay(self, attempt: int, context: RetryContext) -> float:
        """Calculate delay for next retry attempt.

        Uses the configured backoff strategy to determine delay, applies jitter
        if enabled, and respects both max_delay and max_total_time constraints.

        Args:
            attempt: Current attempt number (0-based)
            context: Retry context with timing information

        Returns:
            Delay in seconds before next retry attempt (>= 0)
        """
        if self.backoff_strategy == "exponential":
            delay = self.base_delay * (self.exponential_base**attempt)
        elif self.backoff_strategy == "linear":
            delay = self.base_delay * (attempt + 1)
        else:  # fixed
            delay = self.base_delay

        # Apply max delay limit
        delay = min(delay, self.max_delay)

        # Apply jitter to prevent thundering herd
        if self.jitter:
            delay *= 0.5 + random.random() * 0.5

        # Check total time constraint
        if self.max_total_time:
            remaining_time = self.max_total_time - context.elapsed_time
            delay = min(delay, remaining_time)

        return max(0, delay)

    def should_retry(self, error: Exception, context: RetryContext) -> bool:
        """Determine if error should trigger retry.

        Checks if the error type is retryable, if max_attempts limit hasn't been
        reached, and if max_total_time constraint hasn't been exceeded.

        Args:
            error: Exception that occurred during operation
            context: Retry context with attempt count and timing information

        Returns:
            True if retry should be attempted, False otherwise
        """
        # Check attempt limit
        if context.attempt >= self.max_attempts:
            return False

        # Check total time limit
        if self.max_total_time and context.elapsed_time >= self.max_total_time:
            return False

        # Check if error is retryable
        if not isinstance(error, self.retryable_errors):
            return False

        # Special handling for throttle errors - always retry with longer delay
        if isinstance(error, S3ThrottleError):
            return True

        return True


class CircuitBreaker:
    """Enhanced circuit breaker with state persistence and monitoring."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Union[
            Type[Exception], Tuple[Type[Exception], ...]
        ] = Exception,
        success_threshold: int = 2,  # Successful calls needed to close circuit
        monitoring_window: float = 300.0,  # 5 minutes
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.monitoring_window = monitoring_window

        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

        # Monitoring
        self.failure_history: List[Tuple[float, str]] = []
        self.success_history: List[float] = []

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Clean old history entries
            self._clean_history()

            # Check circuit state
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise S3ConnectionError(
                        message="Circuit breaker is OPEN - service unavailable",
                        error_code="CircuitBreakerOpen",
                        operation=func.__name__,
                    )

            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result

            except self.expected_exception as e:
                self._on_failure(e)
                raise
            except Exception as e:
                # Unexpected errors don't affect circuit breaker
                logger.warning(f"Unexpected error in circuit breaker: {e}")
                raise

        return wrapper

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout

    def _on_success(self) -> None:
        """Handle successful operation."""
        self.success_count += 1
        self.success_history.append(time.time())

        if self.state == "HALF_OPEN":
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery")
        elif self.state == "CLOSED":
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self, error: Exception) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.failure_history.append((time.time(), str(error)))

        if self.state == "HALF_OPEN":
            # Return to OPEN state on any failure during half-open
            self.state = "OPEN"
            logger.warning("Circuit breaker returned to OPEN state")
        elif self.state == "CLOSED":
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(
                    f"Circuit breaker OPENED after {self.failure_count} failures"
                )

    def _clean_history(self) -> None:
        """Clean old entries from history."""
        current_time = time.time()
        cutoff_time = current_time - self.monitoring_window

        self.failure_history = [
            entry for entry in self.failure_history if entry[0] > cutoff_time
        ]

        self.success_history = [
            entry for entry in self.success_history if entry > cutoff_time
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "recent_failures": len(self.failure_history),
            "recent_successes": len(self.success_history),
            "last_failure_time": self.last_failure_time,
        }


def with_retry(config: Optional[RetryConfig] = None) -> Callable:
    """Enhanced retry decorator with comprehensive error handling."""
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            context = RetryContext(
                operation=func.__name__,
                attempt=0,
                total_attempts=config.max_attempts,
                last_error=None,
                start_time=time.time(),
                delay=0,
            )

            while context.attempt < config.max_attempts:
                try:
                    logger.debug(
                        f"Executing {context.operation} (attempt {context.attempt + 1}/"
                        f"{context.total_attempts})"
                    )

                    result = await func(*args, **kwargs)

                    if context.attempt > 0:
                        logger.info(
                            f"Operation {context.operation} succeeded after "
                            f"{context.attempt + 1} attempts"
                        )

                    return result

                except Exception as e:
                    context.last_error = e
                    context.attempt += 1

                    # Map to appropriate exception type
                    mapped_error = map_boto3_error(e, context.operation)

                    # Check if we should retry
                    if not config.should_retry(mapped_error, context):
                        logger.error(
                            f"Operation {context.operation} failed permanently: "
                            f"{mapped_error}"
                        )
                        raise mapped_error

                    # Calculate delay for next attempt
                    context.delay = config.calculate_delay(context.attempt - 1, context)

                    if context.delay <= 0:
                        logger.error(f"Retry timeout exceeded for {context.operation}")
                        break

                    logger.warning(
                        f"Attempt {context.attempt} failed for {context.operation}: "
                        f"{mapped_error}. Retrying in {context.delay:.2f}s"
                    )

                    await asyncio.sleep(context.delay)

            # All retry attempts exhausted
            final_error = map_boto3_error(
                context.last_error or Exception("Unknown error"), context.operation
            )
            logger.error(
                f"Max retry attempts ({config.max_attempts}) exceeded for "
                f"{context.operation}: {final_error}"
            )
            raise final_error

        return wrapper

    return decorator
