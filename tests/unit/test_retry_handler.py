"""Unit tests for retry handler functionality."""

import asyncio
import time

import pytest

from aws_adk.exceptions import S3ConnectionError, S3ThrottleError
from aws_adk.retry_handler import CircuitBreaker, RetryConfig, RetryContext, with_retry


class TestRetryConfig:
    """Test RetryConfig functionality."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.max_total_time is None  # Default is None
        assert config.backoff_strategy == "exponential"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
            max_total_time=120.0,
            backoff_strategy="linear",
        )
        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
        assert config.max_total_time == 120.0
        assert config.backoff_strategy == "linear"

    @pytest.mark.parametrize(
        "strategy,attempt,expected_base",
        [
            ("exponential", 0, 1.0),
            ("exponential", 1, 2.0),
            ("exponential", 2, 4.0),
            ("exponential", 3, 8.0),
            ("linear", 0, 1.0),
            ("linear", 1, 2.0),
            ("linear", 2, 3.0),
            ("linear", 3, 4.0),
            ("fixed", 0, 1.0),
            ("fixed", 1, 1.0),
            ("fixed", 2, 1.0),
            ("fixed", 3, 1.0),
        ],
    )
    def test_calculate_delay_strategies(
        self, strategy: str, attempt: int, expected_base: float
    ) -> None:
        """Test different backoff strategies."""
        config = RetryConfig(base_delay=1.0, backoff_strategy=strategy, jitter=False)
        context = RetryContext(
            operation="test",
            attempt=attempt,
            total_attempts=5,
            last_error=None,
            start_time=time.time(),
            delay=0,
        )

        delay = config.calculate_delay(attempt, context)
        assert delay == expected_base

    def test_calculate_delay_with_max_delay(self) -> None:
        """Test delay calculation respects max_delay."""
        config = RetryConfig(
            base_delay=1.0, max_delay=5.0, backoff_strategy="exponential", jitter=False
        )
        context = RetryContext(
            operation="test",
            attempt=10,  # Would result in very high delay
            total_attempts=15,
            last_error=None,
            start_time=time.time(),
            delay=0,
        )

        delay = config.calculate_delay(10, context)
        assert delay == 5.0  # Capped at max_delay

    def test_calculate_delay_with_jitter(self) -> None:
        """Test delay calculation with jitter."""
        config = RetryConfig(base_delay=1.0, backoff_strategy="fixed", jitter=True)
        context = RetryContext(
            operation="test",
            attempt=1,
            total_attempts=5,
            last_error=None,
            start_time=time.time(),
            delay=0,
        )

        delay = config.calculate_delay(1, context)
        # With jitter, delay should be between 0.5 and 1.5 (base_delay ± 50%)
        assert 0.5 <= delay <= 1.5

    @pytest.mark.parametrize(
        "error_type,should_retry",
        [
            (S3ConnectionError, True),
            (S3ThrottleError, True),
            (ValueError, False),
            (RuntimeError, False),
        ],
    )
    def test_should_retry_error_types(
        self, error_type: type, should_retry: bool
    ) -> None:
        """Test should_retry with different error types."""
        config = RetryConfig()
        context = RetryContext(
            operation="test",
            attempt=1,
            total_attempts=5,
            last_error=None,
            start_time=time.time(),
            delay=0,
        )

        error = error_type("Test error")
        assert config.should_retry(error, context) == should_retry

    def test_should_retry_max_total_time_exceeded(self) -> None:
        """Test should_retry when max_total_time is exceeded."""
        config = RetryConfig(max_total_time=1.0)  # 1 second max
        start_time = time.time() - 2.0  # Started 2 seconds ago
        context = RetryContext(
            operation="test",
            attempt=1,
            total_attempts=5,
            last_error=None,
            start_time=start_time,
            delay=0,
        )

        error = S3ConnectionError("Test error")
        assert config.should_retry(error, context) is False


class TestRetryContext:
    """Test RetryContext functionality."""

    def test_elapsed_time(self) -> None:
        """Test elapsed_time calculation."""
        start_time = time.time() - 1.0  # 1 second ago
        context = RetryContext(
            operation="test",
            attempt=1,
            total_attempts=5,
            last_error=None,
            start_time=start_time,
            delay=0,
        )

        elapsed = context.elapsed_time
        assert 0.8 <= elapsed <= 1.2  # Allow more tolerance for timing


class TestWithRetry:
    """Test with_retry decorator functionality."""

    @pytest.mark.asyncio
    async def test_successful_execution_no_retry(self) -> None:
        """Test successful execution without retries."""
        config = RetryConfig(max_attempts=3)
        call_count = 0

        @with_retry(config)
        async def test_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await test_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self) -> None:
        """Test retry behavior on retryable errors."""
        config = RetryConfig(
            max_attempts=3, base_delay=0.01
        )  # Fast retries for testing
        call_count = 0

        @with_retry(config)
        async def test_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Use ClientError which maps to retryable S3ConnectionError
                from botocore.exceptions import ClientError

                raise ClientError(
                    {
                        "Error": {
                            "Code": "RequestTimeout",
                            "Message": "Connection failed",
                        }
                    },
                    "test_operation",
                )
            return "success"

        result = await test_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self) -> None:
        """Test no retry on non-retryable errors."""
        config = RetryConfig(max_attempts=3)
        call_count = 0

        @with_retry(config)
        async def test_func() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(
            Exception, match="Unexpected error"
        ):  # Maps to S3ArtifactError
            await test_func()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self) -> None:
        """Test behavior when max attempts are exceeded."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        call_count = 0

        @with_retry(config)
        async def test_func() -> None:
            nonlocal call_count
            call_count += 1
            from botocore.exceptions import ClientError

            raise ClientError(
                {"Error": {"Code": "RequestTimeout", "Message": "Always fails"}},
                "test_operation",
            )

        with pytest.raises(S3ConnectionError):
            await test_func()
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_total_time_exceeded(self) -> None:
        """Test behavior when max total time is exceeded."""
        config = RetryConfig(max_attempts=10, max_total_time=0.1, base_delay=0.05)
        call_count = 0

        @with_retry(config)
        async def test_func() -> None:
            nonlocal call_count
            call_count += 1
            from botocore.exceptions import ClientError

            raise ClientError(
                {"Error": {"Code": "RequestTimeout", "Message": "Connection failed"}},
                "test_operation",
            )

        with pytest.raises(S3ConnectionError):
            await test_func()
        # Should not reach max_attempts due to time limit
        assert call_count < 10


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    def test_initial_state(self) -> None:
        """Test circuit breaker initial state."""
        cb = CircuitBreaker(
            failure_threshold=3, timeout=10.0, expected_exception=S3ConnectionError
        )
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
        assert cb.success_count == 0

    @pytest.mark.asyncio
    async def test_success_in_closed_state(self) -> None:
        """Test successful execution in CLOSED state."""
        cb = CircuitBreaker(
            failure_threshold=3, timeout=10.0, expected_exception=S3ConnectionError
        )

        @cb
        async def test_func() -> str:
            return "success"

        result = await test_func()
        assert result == "success"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_failures_reach_threshold(self) -> None:
        """Test circuit breaker opens when failure threshold is reached."""
        cb = CircuitBreaker(
            failure_threshold=2, timeout=0.1, expected_exception=S3ConnectionError
        )

        @cb
        async def test_func() -> None:
            raise S3ConnectionError("Connection failed")

        # First failure
        with pytest.raises(S3ConnectionError):
            await test_func()
        assert cb.state == "CLOSED"
        assert cb.failure_count == 1

        # Second failure - should open circuit
        with pytest.raises(S3ConnectionError):
            await test_func()
        assert cb.state == "OPEN"
        assert cb.failure_count == 2

    @pytest.mark.asyncio
    async def test_open_state_raises_immediately(self) -> None:
        """Test circuit breaker raises immediately when OPEN."""
        cb = CircuitBreaker(
            failure_threshold=1, timeout=0.1, expected_exception=S3ConnectionError
        )

        @cb
        async def test_func() -> None:
            raise S3ConnectionError("Connection failed")

        # Trigger failure to open circuit
        with pytest.raises(S3ConnectionError):
            await test_func()
        assert cb.state == "OPEN"

        # Should raise S3ConnectionError immediately without calling function
        with pytest.raises(S3ConnectionError, match="Circuit breaker is OPEN"):
            await test_func()

    @pytest.mark.asyncio
    async def test_half_open_recovery(self) -> None:
        """Test circuit breaker recovery through HALF_OPEN state."""
        cb = CircuitBreaker(
            failure_threshold=1,
            timeout=0.01,  # Very short timeout for testing
            expected_exception=S3ConnectionError,
            success_threshold=1,
        )

        call_count = 0

        @cb
        async def test_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise S3ConnectionError("First failure")
            return "success"

        # First call fails, opens circuit
        with pytest.raises(S3ConnectionError):
            await test_func()
        assert cb.state == "OPEN"

        # Wait for timeout
        await asyncio.sleep(0.02)

        # Next call should enter HALF_OPEN and succeed, closing circuit
        result = await test_func()
        assert result == "success"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    def test_get_stats(self) -> None:
        """Test circuit breaker statistics."""
        cb = CircuitBreaker(
            failure_threshold=3, timeout=10.0, expected_exception=S3ConnectionError
        )

        stats = cb.get_stats()
        assert stats["state"] == "CLOSED"
        assert stats["failure_count"] == 0
        assert stats["success_count"] == 0
        assert stats["recent_failures"] == 0
        assert stats["recent_successes"] == 0
        assert stats["last_failure_time"] is None

    def test_clean_history(self) -> None:
        """Test history cleanup functionality."""
        cb = CircuitBreaker(
            failure_threshold=3,
            timeout=10.0,
            expected_exception=S3ConnectionError,
            monitoring_window=0.1,  # Very short window for testing
        )

        # Add some old entries
        old_time = time.time() - 1.0
        cb.failure_history = [(old_time, "old error")]
        cb.success_history = [old_time]

        # Clean history
        cb._clean_history()

        # Old entries should be removed
        assert len(cb.failure_history) == 0
        assert len(cb.success_history) == 0

    @pytest.mark.asyncio
    async def test_non_expected_exception_not_counted(self) -> None:
        """Test that non-expected exceptions don't count as failures."""
        cb = CircuitBreaker(
            failure_threshold=1, timeout=10.0, expected_exception=S3ConnectionError
        )

        @cb
        async def test_func() -> None:
            raise ValueError("Different error type")

        # ValueError should be raised but not count as circuit breaker failure
        with pytest.raises(ValueError):
            await test_func()
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_success_count_tracking(self) -> None:
        """Test that success count is properly tracked in HALF_OPEN state."""
        cb = CircuitBreaker(
            failure_threshold=1,
            timeout=0.01,
            expected_exception=S3ConnectionError,
            success_threshold=2,  # Need 2 successes to close
        )

        call_count = 0

        @cb
        async def test_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise S3ConnectionError("First failure")
            return f"success_{call_count}"

        # First call fails, opens circuit
        with pytest.raises(S3ConnectionError):
            await test_func()
        assert cb.state == "OPEN"
        assert cb.failure_count == 1

        # Wait for timeout
        await asyncio.sleep(0.02)

        # First success in HALF_OPEN - should stay HALF_OPEN
        result = await test_func()
        assert result == "success_2"
        assert cb.state == "HALF_OPEN"
        assert cb.success_count == 1

        # Second success in HALF_OPEN - should transition to CLOSED
        result = await test_func()
        assert result == "success_3"
        assert cb.state == "CLOSED"
        assert cb.success_count == 2  # Should retain count
        assert cb.failure_count == 0  # Should be reset
