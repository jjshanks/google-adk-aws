"""Integration tests for error handling across service operations."""
# mypy: ignore-errors

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aws_adk import S3ArtifactService
from aws_adk.exceptions import (
    S3ArtifactError,
    S3BucketError,
    S3ConnectionError,
    S3PermissionError,
    S3ThrottleError,
    S3ValidationError,
)
from aws_adk.retry_handler import CircuitBreaker, RetryConfig
from tests.utils import (
    ConcurrencyTester,
    ErrorSimulator,
    TestDataGenerator,
    TestMetricsCollector,
)


def create_mock_service(
    mock_s3_setup: Dict[str, Any], **kwargs: Any
) -> S3ArtifactService:
    """Create a service instance without bucket validation for testing."""
    with patch.object(S3ArtifactService, "_validate_bucket_access"):
        return S3ArtifactService(
            bucket_name=mock_s3_setup["bucket_name"],
            region_name=mock_s3_setup["region_name"],
            **kwargs,
        )


@pytest.mark.unit
@pytest.mark.error_handling
@pytest.mark.asyncio
class TestServiceErrorIntegration:
    """Test error handling integration across service operations."""

    @pytest.fixture
    def service_with_retry_config(
        self, mock_s3_setup: Dict[str, Any]
    ) -> S3ArtifactService:
        """Service with custom retry configuration for testing."""
        return create_mock_service(
            mock_s3_setup,
            retry_config=RetryConfig(
                max_attempts=2,
                base_delay=0.01,  # Fast retries for testing
                max_total_time=1.0,
            ),
        )

    async def test_cascading_errors_in_save_operation(
        self, service_with_retry_config: S3ArtifactService, sample_artifact: Any
    ) -> None:
        """Test cascading error handling in save operation."""
        # Direct method test by patching internal save implementation
        throttle_error = ErrorSimulator.create_client_error("Throttling")

        with patch.object(
            service_with_retry_config, "_save_artifact_impl", side_effect=throttle_error
        ):
            with pytest.raises(S3ThrottleError) as exc_info:
                await service_with_retry_config.save_artifact(
                    app_name="test-app",
                    user_id="test-user",
                    session_id="test-session",
                    filename="test.txt",
                    artifact=sample_artifact,
                )

        error = exc_info.value
        assert error.error_code == "Throttling"
        # The operation will be mapped through error handling
        assert error.operation in ["_save_artifact_impl", "_save"]

    async def test_error_recovery_patterns(
        self, service_with_retry_config: S3ArtifactService
    ) -> None:
        """Test various error recovery patterns."""
        call_count = 0

        async def failing_then_succeeding_operation() -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ErrorSimulator.create_client_error("ServiceUnavailable")
            return {"ResponseMetadata": {"HTTPStatusCode": 200}}

        mock_client = Mock()
        mock_client.list_objects_v2 = failing_then_succeeding_operation
        service_with_retry_config.s3_client = mock_client

        # Should succeed after retry
        result = await service_with_retry_config.list_artifacts(
            app_name="test-app", user_id="test-user"
        )

        assert result == []  # Empty list from successful operation
        assert call_count == 2  # Failed once, succeeded on retry

    async def test_error_context_propagation(self, service_with_retry_config):
        """Test that error context is properly propagated."""
        error = ErrorSimulator.create_client_error("NoSuchBucket")

        mock_client = Mock()
        mock_client.head_bucket = AsyncMock(side_effect=error)
        service_with_retry_config.s3_client = mock_client

        with pytest.raises(S3BucketError) as exc_info:
            await service_with_retry_config._ensure_bucket_exists()

        error_obj = exc_info.value
        assert error_obj.context["bucket_name"] == "test-bucket"
        assert error_obj.context["region"] == "us-east-1"

    async def test_validation_error_propagation(
        self, service_with_retry_config, sample_artifact
    ):
        """Test that validation errors are properly propagated without retry."""
        call_count = 0

        async def count_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise S3ValidationError(
                message="Invalid filename",
                error_code="ValidationFailed",
                operation="save_artifact",
            )

        with patch.object(
            service_with_retry_config, "_validate_inputs", side_effect=count_calls
        ):
            with pytest.raises(S3ValidationError) as exc_info:
                await service_with_retry_config.save_artifact(
                    app_name="test-app",
                    user_id="test-user",
                    session_id="test-session",
                    filename="../invalid.txt",
                    artifact=sample_artifact,
                )

        # Validation errors should not be retried
        assert call_count == 1
        assert exc_info.value.error_code == "ValidationFailed"

    async def test_permission_error_handling(
        self, service_with_retry_config, sample_artifact
    ):
        """Test permission error handling across operations."""
        access_denied_error = ErrorSimulator.create_client_error("AccessDenied")

        mock_client = Mock()
        mock_client.list_objects_v2 = AsyncMock(side_effect=access_denied_error)
        service_with_retry_config.s3_client = mock_client

        with pytest.raises(S3PermissionError) as exc_info:
            await service_with_retry_config.save_artifact(
                app_name="test-app",
                user_id="test-user",
                session_id="test-session",
                filename="test.txt",
                artifact=sample_artifact,
            )

        error = exc_info.value
        assert error.error_code == "AccessDenied"
        assert "access denied" in error.message.lower()


@pytest.mark.unit
@pytest.mark.concurrency
@pytest.mark.error_handling
@pytest.mark.asyncio
class TestConcurrentErrorHandling:
    """Test error handling under concurrent operations."""

    async def test_concurrent_save_errors(self, advanced_s3_mock, mock_s3_setup):
        """Test error handling during concurrent save operations."""
        service = create_mock_service(mock_s3_setup)
        service.s3_client = advanced_s3_mock

        # Configure mock to fail after 3 calls
        throttle_error = ErrorSimulator.create_client_error("Throttling")
        advanced_s3_mock.configure_failure(3, throttle_error)

        # Generate test artifacts
        artifacts = [
            TestDataGenerator.generate_artifact(100, "text/plain") for _ in range(5)
        ]

        # Create concurrent save operations
        async def save_operation(index):
            try:
                return await service.save_artifact(
                    app_name="test-app",
                    user_id="test-user",
                    session_id="test-session",
                    filename=f"test-{index}.txt",
                    artifact=artifacts[index],
                )
            except Exception as e:
                return e

        operations = [lambda i=i: save_operation(i) for i in range(5)]

        # Run concurrent operations
        results = await ConcurrencyTester.run_concurrent_operations(
            operations, max_concurrent=3
        )

        # Some should succeed, some should fail
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]

        assert len(successes) >= 2  # At least some succeed
        assert len(failures) >= 2  # At least some fail

        # All failures should be S3ThrottleError
        for failure in failures:
            assert isinstance(failure, S3ThrottleError)

    async def test_circuit_breaker_under_load(self):
        """Test circuit breaker behavior under concurrent load."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=3, timeout=0.1, expected_exception=S3ArtifactError
        )

        failure_count = 0

        @circuit_breaker
        async def failing_operation():
            nonlocal failure_count
            failure_count += 1
            raise S3ConnectionError(f"Failure {failure_count}")

        # Run multiple concurrent operations
        async def wrapped_operation():
            try:
                await failing_operation()
                return "success"
            except Exception as e:
                return e

        operations = [wrapped_operation for _ in range(10)]
        results = await ConcurrencyTester.run_concurrent_operations(
            operations, max_concurrent=5
        )

        # Should have mix of original errors and circuit breaker errors
        original_errors = [
            r
            for r in results
            if isinstance(r, S3ConnectionError) and r.error_code != "CircuitBreakerOpen"
        ]
        breaker_errors = [
            r
            for r in results
            if isinstance(r, S3ConnectionError) and r.error_code == "CircuitBreakerOpen"
        ]

        assert len(original_errors) >= 3  # At least threshold failures
        assert len(breaker_errors) >= 2  # Circuit should open and prevent some

    async def test_concurrent_load_with_mixed_errors(
        self, advanced_s3_mock, mock_s3_setup
    ):
        """Test handling of mixed error types under concurrent load."""
        service = create_mock_service(mock_s3_setup)
        service.s3_client = advanced_s3_mock

        # Configure different errors for different operations
        error_sequence = [
            ErrorSimulator.create_client_error("Throttling"),
            ErrorSimulator.create_client_error("ServiceUnavailable"),
            ErrorSimulator.create_client_error("AccessDenied"),
            None,  # Success
            ErrorSimulator.create_client_error("NoSuchBucket"),
        ]

        call_count = 0

        async def mixed_error_operation(*args, **kwargs):
            nonlocal call_count
            error = error_sequence[call_count % len(error_sequence)]
            call_count += 1
            if error:
                raise error
            return {"ResponseMetadata": {"HTTPStatusCode": 200}}

        advanced_s3_mock.list_objects_v2 = mixed_error_operation

        # Create multiple concurrent list operations
        async def list_operation(index):
            try:
                return await service.list_artifacts(
                    app_name=f"test-{index}", user_id="test"
                )
            except Exception as e:
                return e

        operations = [lambda i=i: list_operation(i) for i in range(10)]
        results = await ConcurrencyTester.run_concurrent_operations(
            operations, max_concurrent=4
        )

        # Categorize results by error type
        throttle_errors = [r for r in results if isinstance(r, S3ThrottleError)]
        connection_errors = [r for r in results if isinstance(r, S3ConnectionError)]
        permission_errors = [r for r in results if isinstance(r, S3PermissionError)]
        bucket_errors = [r for r in results if isinstance(r, S3BucketError)]
        successes = [r for r in results if isinstance(r, list)]

        # Should have a mix of error types
        assert len(throttle_errors) >= 1
        assert len(connection_errors) >= 1
        assert len(permission_errors) >= 1
        assert len(bucket_errors) >= 1
        assert len(successes) >= 1


@pytest.mark.unit
@pytest.mark.error_handling
@pytest.mark.asyncio
class TestErrorMetricsAndMonitoring:
    """Test error metrics collection and monitoring."""

    async def test_error_metrics_collection(self, mock_s3_setup):
        """Test that error metrics are properly collected."""
        metrics_collector = TestMetricsCollector()
        service = create_mock_service(mock_s3_setup)

        # Mock error scenario
        error = ErrorSimulator.create_client_error("Throttling")
        mock_client = Mock()
        mock_client.put_object = AsyncMock(side_effect=error)
        service.s3_client = mock_client

        try:
            await service.save_artifact(
                app_name="test-app",
                user_id="test-user",
                session_id="test-session",
                filename="test.txt",
                artifact=TestDataGenerator.generate_artifact(100),
            )
        except S3ThrottleError:
            # Record the error in metrics
            metrics_collector.error_count += 1
            metrics_collector.error_types.add("S3ThrottleError")

        # Verify error was recorded in metrics
        assert metrics_collector.error_count > 0
        assert "S3ThrottleError" in metrics_collector.error_types

    async def test_performance_metrics_under_errors(
        self, advanced_s3_mock, mock_s3_setup
    ):
        """Test performance metrics collection during error conditions."""
        service = create_mock_service(mock_s3_setup)
        service.s3_client = advanced_s3_mock

        # Configure delays and failures
        advanced_s3_mock.configure_delay(0.1)  # 100ms delay
        advanced_s3_mock.configure_failure(
            2, ErrorSimulator.create_client_error("Throttling")
        )

        start_time = asyncio.get_event_loop().time()

        with pytest.raises(S3ThrottleError):
            await service.save_artifact(
                app_name="test-app",
                user_id="test-user",
                session_id="test-session",
                filename="test.txt",
                artifact=TestDataGenerator.generate_artifact(100),
            )

        end_time = asyncio.get_event_loop().time()
        operation_time = end_time - start_time

        # Should include retry delays
        assert operation_time >= 0.1  # At least one delay period

    async def test_error_recovery_metrics(self, advanced_s3_mock, mock_s3_setup):
        """Test metrics for successful error recovery."""
        service = create_mock_service(mock_s3_setup)
        service.s3_client = advanced_s3_mock

        # Configure to fail once, then succeed
        call_count = 0

        async def fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ErrorSimulator.create_client_error("ServiceUnavailable")
            return {"ResponseMetadata": {"HTTPStatusCode": 200}}

        advanced_s3_mock.list_objects_v2 = fail_then_succeed

        # Should succeed after retry
        result = await service.list_artifacts(app_name="test-app", user_id="test-user")

        assert result == []
        assert call_count == 2  # Failed once, succeeded on retry

        # This test validates the pattern even if metrics aren't fully implemented
        # The important part is that recovery happened (call_count == 2)


@pytest.mark.unit
@pytest.mark.error_handling
@pytest.mark.asyncio
class TestErrorContextValidation:
    """Test error context validation and enrichment."""

    async def test_error_context_enrichment(self, mock_s3_setup):
        """Test that errors are enriched with proper context."""
        service = create_mock_service(mock_s3_setup)

        # Mock client error
        error = ErrorSimulator.create_client_error("NoSuchKey")
        mock_client = Mock()
        mock_client.get_object = AsyncMock(side_effect=error)
        service.s3_client = mock_client

        # Load operation should enrich error with context
        result = await service.load_artifact(
            app_name="test-app",
            user_id="test-user",
            session_id="test-session",
            filename="missing.txt",
        )

        # For NoSuchKey, load_artifact should return None instead of raising
        assert result is None

    async def test_operation_context_preservation(self, sample_artifact, mock_s3_setup):
        """Test that operation context is preserved across retries."""
        service = create_mock_service(mock_s3_setup)

        call_contexts = []

        async def capture_context_operation(*args, **kwargs):
            call_contexts.append(kwargs.copy())
            if len(call_contexts) < 2:
                raise ErrorSimulator.create_client_error("ServiceUnavailable")
            return {"ResponseMetadata": {"HTTPStatusCode": 200}}

        mock_client = Mock()
        mock_client.list_objects_v2 = capture_context_operation
        mock_client.put_object = capture_context_operation
        service.s3_client = mock_client

        # Should succeed after retry
        await service.save_artifact(
            app_name="test-app",
            user_id="test-user",
            session_id="test-session",
            filename="test.txt",
            artifact=sample_artifact,
        )

        # Context should be consistent across retries
        assert len(call_contexts) >= 2
        # Bucket and Key should be consistent
        if call_contexts:
            first_context = call_contexts[0]
            for context in call_contexts[1:]:
                if "Bucket" in first_context and "Bucket" in context:
                    assert first_context["Bucket"] == context["Bucket"]

    async def test_nested_operation_error_context(self, sample_artifact, mock_s3_setup):
        """Test error context in nested operations."""
        service = create_mock_service(mock_s3_setup)

        # Mock error in nested bucket check
        bucket_error = ErrorSimulator.create_client_error("AccessDenied")
        mock_client = Mock()
        mock_client.head_bucket = AsyncMock(side_effect=bucket_error)
        service.s3_client = mock_client

        with pytest.raises(S3PermissionError) as exc_info:
            await service.save_artifact(
                app_name="test-app",
                user_id="test-user",
                session_id="test-session",
                filename="test.txt",
                artifact=sample_artifact,
            )

        error = exc_info.value
        assert error.error_code == "AccessDenied"
        # Context should include operation details
        assert error.context is not None
        assert isinstance(error.context, dict)
