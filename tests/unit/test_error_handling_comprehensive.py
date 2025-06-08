# mypy: ignore-errors
"""Comprehensive unit tests for error handling and exception scenarios."""

import asyncio
import time
from unittest.mock import patch

import pytest
from botocore.exceptions import BotoCoreError, NoCredentialsError

from aws_adk.exceptions import (
    S3ArtifactError,
    S3ArtifactNotFoundError,
    S3BucketError,
    S3ConcurrencyError,
    S3ConnectionError,
    S3PermissionError,
    S3StorageQuotaError,
    S3ThrottleError,
    S3ValidationError,
    map_boto3_error,
)
from aws_adk.retry_handler import CircuitBreaker, RetryConfig, with_retry
from tests.utils import (
    ErrorSimulator,
    TestMetricsCollector,
    assert_operation_fails_with_error,
    create_test_service,
    parametrize_error_scenarios,
)

pytestmark = pytest.mark.asyncio


class TestExceptionMapping:
    """Test boto3 error mapping to custom exceptions."""

    @parametrize_error_scenarios()
    async def test_all_boto3_error_mapping(self, error_code):
        """Test mapping of all boto3 error codes to appropriate exceptions."""
        client_error = ErrorSimulator.create_client_error(error_code)
        mapped_error = map_boto3_error(client_error, "test_operation")

        # Verify error is properly mapped
        assert isinstance(mapped_error, S3ArtifactError)
        assert mapped_error.error_code == error_code
        assert mapped_error.operation == "test_operation"
        assert mapped_error.cause == client_error

        # Test specific mappings
        expected_mappings = {
            "NoSuchBucket": S3BucketError,
            "AccessDenied": S3PermissionError,
            "NoSuchKey": S3ArtifactNotFoundError,
            "Throttling": S3ThrottleError,
            "ServiceUnavailable": S3ConnectionError,
            "InvalidRequest": S3ValidationError,
            "PreconditionFailed": S3ConcurrencyError,
            "QuotaExceeded": S3StorageQuotaError,
        }

        if error_code in expected_mappings:
            assert isinstance(mapped_error, expected_mappings[error_code])

    async def test_no_credentials_error_mapping(self):
        """Test mapping of NoCredentialsError."""
        error = NoCredentialsError()
        mapped_error = map_boto3_error(error, "test_operation")

        assert isinstance(mapped_error, S3ConnectionError)
        assert mapped_error.error_code == "NoCredentials"
        assert "credentials not found" in str(mapped_error).lower()

    async def test_botocore_error_mapping(self):
        """Test mapping of general BotoCoreError."""
        error = BotoCoreError()
        mapped_error = map_boto3_error(error, "test_operation")

        assert isinstance(mapped_error, S3ConnectionError)
        assert mapped_error.error_code == "BotoCoreError"

    async def test_unknown_error_mapping(self):
        """Test mapping of unknown exceptions."""
        error = ValueError("Unknown error")
        mapped_error = map_boto3_error(error, "test_operation")

        assert isinstance(mapped_error, S3ArtifactError)
        assert mapped_error.operation == "test_operation"
        assert mapped_error.cause == error


class TestErrorContextAndLogging:
    """Test error context preservation and logging."""

    async def test_error_context_preservation(self):
        """Test that error context is properly preserved."""
        context = {"bucket": "test", "key": "test-key"}
        error = S3ArtifactError(
            message="Test error",
            error_code="TestError",
            operation="test_operation",
            context=context,
        )

        assert error.error_code == "TestError"
        assert error.operation == "test_operation"
        assert error.context == context

    async def test_error_serialization(self):
        """Test error serialization to dictionary."""
        error = S3ValidationError(
            message="Validation failed",
            error_code="ValidationFailed",
            operation="validate_input",
            context={"field": "filename", "value": "invalid"},
        )

        error_dict = error.to_dict()

        assert error_dict["error"] == "S3ValidationError"
        assert error_dict["message"] == "Validation failed"
        assert error_dict["error_code"] == "ValidationFailed"
        assert error_dict["operation"] == "validate_input"
        assert error_dict["context"]["field"] == "filename"

    @patch("aws_adk.exceptions.logger")
    async def test_error_logging(self, mock_logger):
        """Test that errors are properly logged."""
        S3ArtifactError(
            message="Test error", error_code="TestError", operation="test_operation"
        )

        mock_logger.error.assert_called_once()
        log_call = mock_logger.error.call_args[0][0]
        assert "Test error" in log_call
        assert "TestError" in log_call


class TestServiceErrorHandling:
    """Test error handling in S3ArtifactService operations."""

    @pytest.fixture
    def mock_service(self, advanced_s3_mock):
        """Create service with mock S3 client."""
        service = create_test_service()
        service.s3_client = advanced_s3_mock
        return service

    async def test_save_artifact_bucket_not_found(self, mock_service, sample_artifact):
        """Test save_artifact with bucket not found error."""
        error = ErrorSimulator.create_client_error("NoSuchBucket")
        mock_service.s3_client.configure_failure(1, error)

        await assert_operation_fails_with_error(
            lambda: mock_service.save_artifact(
                app_name="test",
                user_id="test",
                session_id="test",
                filename="test.txt",
                artifact=sample_artifact,
            ),
            S3BucketError,
            "NoSuchBucket",
        )

    async def test_save_artifact_access_denied(self, mock_service, sample_artifact):
        """Test save_artifact with access denied error."""
        error = ErrorSimulator.create_client_error("AccessDenied")
        mock_service.s3_client.configure_failure(2, error)  # Fail after list_versions

        await assert_operation_fails_with_error(
            lambda: mock_service.save_artifact(
                app_name="test",
                user_id="test",
                session_id="test",
                filename="test.txt",
                artifact=sample_artifact,
            ),
            S3PermissionError,
            "AccessDenied",
        )

    async def test_load_artifact_not_found(self, mock_service):
        """Test load_artifact with artifact not found."""
        error = ErrorSimulator.create_client_error("NoSuchKey")
        mock_service.s3_client.configure_failure(1, error)

        # Should return None instead of raising exception for NotFound
        result = await mock_service.load_artifact(
            app_name="test",
            user_id="test",
            session_id="test",
            filename="nonexistent.txt",
        )
        assert result is None

    async def test_throttling_error_handling(self, mock_service, sample_artifact):
        """Test handling of throttling errors."""
        error = ErrorSimulator.create_client_error("Throttling", http_status=503)
        mock_service.s3_client.configure_failure(1, error)

        await assert_operation_fails_with_error(
            lambda: mock_service.save_artifact(
                app_name="test",
                user_id="test",
                session_id="test",
                filename="test.txt",
                artifact=sample_artifact,
            ),
            S3ThrottleError,
            "Throttling",
        )

    @pytest.mark.slow
    async def test_network_timeout_handling(self, mock_service, sample_artifact):
        """Test handling of network timeouts."""
        # Configure long delay to trigger timeout
        mock_service.s3_client.configure_delay(10.0)

        # Override retry config for faster testing
        mock_service.retry_config = RetryConfig(max_attempts=1, max_total_time=1.0)

        await assert_operation_fails_with_error(
            lambda: mock_service.save_artifact(
                app_name="test",
                user_id="test",
                session_id="test",
                filename="test.txt",
                artifact=sample_artifact,
            ),
            S3ConnectionError,
        )


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with error handling."""

    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after consecutive failures."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=3, timeout=1.0, expected_exception=S3ArtifactError
        )

        # Simulate failing function
        @circuit_breaker
        async def failing_operation():
            raise S3ConnectionError("Connection failed", error_code="ConnectionError")

        # First 3 calls should fail with original error
        for i in range(3):
            with pytest.raises(S3ConnectionError):
                await failing_operation()

        # 4th call should fail with circuit breaker error
        with pytest.raises(S3ConnectionError) as exc_info:
            await failing_operation()

        assert exc_info.value.error_code == "CircuitBreakerOpen"

    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=2,
            timeout=0.1,  # Short timeout for testing
            success_threshold=1,
        )

        failure_count = 0

        @circuit_breaker
        async def intermittent_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise S3ConnectionError("Connection failed")
            return "success"

        # Trigger circuit breaker
        with pytest.raises(S3ConnectionError):
            await intermittent_operation()
        with pytest.raises(S3ConnectionError):
            await intermittent_operation()

        # Should be open now
        with pytest.raises(S3ConnectionError) as exc_info:
            await intermittent_operation()
        assert exc_info.value.error_code == "CircuitBreakerOpen"

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Should recover after successful call
        result = await intermittent_operation()
        assert result == "success"

    async def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics collection."""
        circuit_breaker = CircuitBreaker(failure_threshold=2)

        stats = circuit_breaker.get_stats()
        assert stats["state"] == "CLOSED"
        assert stats["failure_count"] == 0

        # Trigger some failures
        @circuit_breaker
        async def failing_operation():
            raise S3ConnectionError("Test failure")

        with pytest.raises(S3ConnectionError):
            await failing_operation()

        stats = circuit_breaker.get_stats()
        assert stats["failure_count"] == 1
        assert stats["recent_failures"] >= 1


class TestRetryLogicComprehensive:
    """Comprehensive testing of retry logic and backoff strategies."""

    async def test_exponential_backoff_timing(self):
        """Test exponential backoff timing accuracy."""
        retry_config = RetryConfig(
            max_attempts=4,
            base_delay=0.1,
            exponential_base=2.0,
            jitter=False,  # Disable jitter for precise timing
        )

        call_times = []

        async def timed_operation():
            call_times.append(time.time())
            raise S3ConnectionError("Test error")

        with pytest.raises(S3ConnectionError):
            await with_retry(retry_config)(timed_operation)()

        # Verify timing intervals
        assert len(call_times) == 4  # Original + 3 retries

        # Check delays (with some tolerance for execution time)
        expected_delays = [0.1, 0.2, 0.4]  # Exponential backoff
        for i in range(1, len(call_times)):
            actual_delay = call_times[i] - call_times[i - 1]
            expected_delay = expected_delays[i - 1]
            assert abs(actual_delay - expected_delay) < 0.05  # 50ms tolerance

    async def test_retry_with_jitter(self):
        """Test retry logic with jitter enabled."""
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1, jitter=True)

        call_times = []

        async def timed_operation():
            call_times.append(time.time())
            raise S3ThrottleError("Rate limited")

        with pytest.raises(S3ThrottleError):
            await with_retry(retry_config)(timed_operation)()

        # With jitter, delays should vary
        delays = [call_times[i] - call_times[i - 1] for i in range(1, len(call_times))]

        # All delays should be at least base_delay/2 and at most base_delay
        for delay in delays:
            assert 0.05 <= delay <= 0.15  # Jitter range

    async def test_max_total_time_limit(self):
        """Test maximum total time limit for retries."""
        retry_config = RetryConfig(
            max_attempts=10,  # High attempt count
            base_delay=0.2,
            max_total_time=0.5,  # But short total time
        )

        start_time = time.time()

        async def slow_operation():
            raise S3ConnectionError("Persistent error")

        with pytest.raises(S3ConnectionError):
            await with_retry(retry_config)(slow_operation)()

        total_time = time.time() - start_time
        assert total_time <= 0.7  # Allow some overhead

    async def test_non_retryable_errors(self):
        """Test that non-retryable errors are not retried."""
        retry_config = RetryConfig(
            max_attempts=3,
            retryable_errors=(S3ConnectionError,),  # Only connection errors
        )

        call_count = 0

        async def validation_error_operation():
            nonlocal call_count
            call_count += 1
            raise S3ValidationError("Invalid input")

        with pytest.raises(S3ValidationError):
            await with_retry(retry_config)(validation_error_operation)()

        assert call_count == 1  # Should not retry validation errors

    async def test_successful_retry_recovery(self):
        """Test successful operation after retries."""
        failure_count = 0

        async def intermittent_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise S3ThrottleError("Throttled")
            return "success"

        result = await with_retry()(intermittent_operation)()
        assert result == "success"
        assert failure_count == 3  # Failed twice, succeeded third time


@pytest.mark.error_handling
class TestComprehensiveErrorScenarios:
    """Test all possible error scenarios comprehensively."""

    @pytest.fixture
    def metrics_collector(self):
        """Metrics collector for tracking test coverage."""
        return TestMetricsCollector()

    async def test_all_s3_error_codes(self, metrics_collector):
        """Test handling of all possible S3 error codes."""
        error_codes = ErrorSimulator.get_all_s3_error_codes()

        for error_code in error_codes:
            error = ErrorSimulator.create_client_error(error_code)
            mapped_error = map_boto3_error(error, "test_operation")

            # Record error scenario tested
            metrics_collector.record_error_scenario(
                type(mapped_error).__name__, error_code
            )

            # Verify error is properly handled
            assert isinstance(mapped_error, S3ArtifactError)
            assert mapped_error.error_code == error_code

        # Verify comprehensive coverage
        coverage_report = metrics_collector.get_coverage_report()
        assert coverage_report["error_scenarios"] == len(error_codes)
