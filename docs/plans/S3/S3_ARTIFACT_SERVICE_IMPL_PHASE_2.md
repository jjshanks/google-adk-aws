# Google ADK AWS Integrations - S3 Artifact Service Implementation - Phase 2

## Overview

Phase 2 builds upon the foundational S3ArtifactService implementation from Phase 1, focusing on comprehensive testing, error handling, performance optimization, and production readiness. This phase transforms the basic implementation into a robust, enterprise-grade service ready for deployment.

**Duration**: 4-5 weeks
**Prerequisites**: Phase 1 completion (core S3ArtifactService implementation)
**Status**: Ready to begin following Phase 1 completion

## Phase 2 Objectives

1. **Comprehensive Testing**: Complete test suite with >95% coverage using moto S3 mocking
2. **Advanced Error Handling**: Robust error recovery, retry logic, and edge case management
3. **Performance Optimization**: Connection pooling, async improvements, and efficient operations
4. **Production Features**: Security hardening, monitoring, and deployment considerations
5. **Documentation Excellence**: Complete API documentation, examples, and migration guides
6. **CI/CD Pipeline**: Automated testing, quality gates, and package publishing

## Implementation Roadmap

### Sub-Phase 2.1: Advanced Testing Framework (Week 1)
**Duration**: 5-7 days
**Focus**: Comprehensive test coverage with mocking and integration testing

#### 2.1.1 Moto S3 Testing Infrastructure

**File**: `tests/unit/test_s3_artifact_service.py`
```python
"""Comprehensive unit tests for S3ArtifactService using moto mocking."""

import asyncio
import pytest
from unittest.mock import Mock, patch
from moto import mock_s3
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from google_adk_aws import S3ArtifactService
from google_adk_aws.s3_artifact_service import (
    S3ArtifactError,
    S3ConnectionError,
    S3PermissionError
)
from google.genai import types


@pytest.fixture
def mock_s3_setup():
    """Set up mocked S3 environment for testing."""
    with mock_s3():
        # Create S3 client and bucket
        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket_name = "test-artifacts-bucket"
        s3_client.create_bucket(Bucket=bucket_name)

        yield {
            "bucket_name": bucket_name,
            "region_name": "us-east-1",
            "s3_client": s3_client
        }


@pytest.fixture
def artifact_service(mock_s3_setup):
    """Create S3ArtifactService instance with mocked S3."""
    return S3ArtifactService(
        bucket_name=mock_s3_setup["bucket_name"],
        region_name=mock_s3_setup["region_name"]
    )


@pytest.fixture
def sample_artifact():
    """Create sample artifact for testing."""
    return types.Part.from_text("Sample artifact content", mime_type="text/plain")


class TestS3ArtifactServiceInitialization:
    """Test service initialization and configuration."""

    def test_successful_initialization(self, mock_s3_setup):
        """Test successful service initialization."""
        service = S3ArtifactService(
            bucket_name=mock_s3_setup["bucket_name"],
            region_name=mock_s3_setup["region_name"]
        )
        assert service.bucket_name == mock_s3_setup["bucket_name"]
        assert service.region_name == mock_s3_setup["region_name"]

    def test_initialization_with_credentials(self, mock_s3_setup):
        """Test initialization with explicit credentials."""
        service = S3ArtifactService(
            bucket_name=mock_s3_setup["bucket_name"],
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret"
        )
        assert service.aws_access_key_id == "test_key"

    def test_initialization_with_custom_endpoint(self, mock_s3_setup):
        """Test initialization with custom S3 endpoint."""
        service = S3ArtifactService(
            bucket_name=mock_s3_setup["bucket_name"],
            endpoint_url="http://localhost:9000"
        )
        assert service.endpoint_url == "http://localhost:9000"

    def test_bucket_access_verification_failure(self):
        """Test bucket access verification with non-existent bucket."""
        with pytest.raises(S3ArtifactError, match="does not exist"):
            S3ArtifactService(bucket_name="non-existent-bucket")


class TestS3ArtifactServiceOperations:
    """Test core artifact operations."""

    @pytest.mark.asyncio
    async def test_save_artifact_first_version(self, artifact_service, sample_artifact):
        """Test saving first version of an artifact."""
        version = await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact
        )
        assert version == 0

    @pytest.mark.asyncio
    async def test_save_artifact_version_increment(self, artifact_service, sample_artifact):
        """Test version increment on multiple saves."""
        # Save first version
        v1 = await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact
        )

        # Save second version
        v2 = await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact
        )

        assert v1 == 0
        assert v2 == 1

    @pytest.mark.asyncio
    async def test_save_user_namespace_artifact(self, artifact_service, sample_artifact):
        """Test saving artifact with user namespace."""
        version = await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="user:global_file.txt",
            artifact=sample_artifact
        )
        assert version == 0

    @pytest.mark.asyncio
    async def test_load_artifact_latest_version(self, artifact_service, sample_artifact):
        """Test loading latest version of artifact."""
        # Save artifact
        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact
        )

        # Load artifact (latest version)
        loaded = await artifact_service.load_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt"
        )

        assert loaded is not None
        assert loaded.inline_data.data == sample_artifact.inline_data.data

    @pytest.mark.asyncio
    async def test_load_artifact_specific_version(self, artifact_service, sample_artifact):
        """Test loading specific version of artifact."""
        # Save multiple versions
        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact
        )

        modified_artifact = types.Part.from_text("Modified content", mime_type="text/plain")
        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=modified_artifact
        )

        # Load first version
        loaded = await artifact_service.load_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            version=0
        )

        assert loaded.inline_data.data == sample_artifact.inline_data.data

    @pytest.mark.asyncio
    async def test_load_nonexistent_artifact(self, artifact_service):
        """Test loading non-existent artifact returns None."""
        loaded = await artifact_service.load_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="nonexistent.txt"
        )
        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_artifact_keys_empty(self, artifact_service):
        """Test listing artifact keys with no artifacts."""
        keys = await artifact_service.list_artifact_keys(
            app_name="test_app",
            user_id="user123",
            session_id="session456"
        )
        assert keys == []

    @pytest.mark.asyncio
    async def test_list_artifact_keys_with_artifacts(self, artifact_service, sample_artifact):
        """Test listing artifact keys with multiple artifacts."""
        # Save session-scoped artifacts
        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="file1.txt",
            artifact=sample_artifact
        )

        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="file2.txt",
            artifact=sample_artifact
        )

        # Save user-scoped artifact
        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="user:global.txt",
            artifact=sample_artifact
        )

        keys = await artifact_service.list_artifact_keys(
            app_name="test_app",
            user_id="user123",
            session_id="session456"
        )

        assert set(keys) == {"file1.txt", "file2.txt", "user:global.txt"}

    @pytest.mark.asyncio
    async def test_delete_artifact(self, artifact_service, sample_artifact):
        """Test deleting artifact with all versions."""
        # Save multiple versions
        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact
        )

        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact
        )

        # Delete artifact
        await artifact_service.delete_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt"
        )

        # Verify deletion
        loaded = await artifact_service.load_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt"
        )
        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_versions(self, artifact_service, sample_artifact):
        """Test listing versions of an artifact."""
        # Save multiple versions
        for i in range(3):
            await artifact_service.save_artifact(
                app_name="test_app",
                user_id="user123",
                session_id="session456",
                filename="test_file.txt",
                artifact=sample_artifact
            )

        versions = await artifact_service.list_versions(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt"
        )

        assert versions == [0, 1, 2]


class TestS3ArtifactServiceErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_save_artifact_s3_error(self, artifact_service, sample_artifact):
        """Test save artifact with S3 client error."""
        with patch.object(artifact_service.s3_client, 'put_object') as mock_put:
            mock_put.side_effect = ClientError(
                error_response={'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}},
                operation_name='PutObject'
            )

            with pytest.raises(S3ArtifactError, match="Failed to save artifact"):
                await artifact_service.save_artifact(
                    app_name="test_app",
                    user_id="user123",
                    session_id="session456",
                    filename="test_file.txt",
                    artifact=sample_artifact
                )

    @pytest.mark.asyncio
    async def test_load_artifact_s3_error(self, artifact_service):
        """Test load artifact with S3 client error."""
        with patch.object(artifact_service.s3_client, 'get_object') as mock_get:
            mock_get.side_effect = ClientError(
                error_response={'Error': {'Code': 'InternalError', 'Message': 'Internal error'}},
                operation_name='GetObject'
            )

            with pytest.raises(S3ArtifactError, match="Failed to load artifact"):
                await artifact_service.load_artifact(
                    app_name="test_app",
                    user_id="user123",
                    session_id="session456",
                    filename="test_file.txt"
                )

    def test_initialization_no_credentials(self):
        """Test initialization failure with no credentials."""
        with patch('boto3.Session') as mock_session:
            mock_session.side_effect = NoCredentialsError()

            with pytest.raises(S3ConnectionError, match="Failed to create S3 client"):
                S3ArtifactService(bucket_name="test-bucket")


class TestS3ArtifactServiceUtilities:
    """Test utility methods and edge cases."""

    def test_file_has_user_namespace(self, artifact_service):
        """Test user namespace detection."""
        assert artifact_service._file_has_user_namespace("user:test.txt") is True
        assert artifact_service._file_has_user_namespace("regular.txt") is False
        assert artifact_service._file_has_user_namespace("") is False

    def test_get_object_key_session_scoped(self, artifact_service):
        """Test object key generation for session-scoped artifacts."""
        key = artifact_service._get_object_key(
            app_name="app",
            user_id="user123",
            session_id="session456",
            filename="test.txt",
            version=1
        )
        assert key == "app/user123/session456/test.txt/1"

    def test_get_object_key_user_scoped(self, artifact_service):
        """Test object key generation for user-scoped artifacts."""
        key = artifact_service._get_object_key(
            app_name="app",
            user_id="user123",
            session_id="session456",
            filename="user:test.txt",
            version=1
        )
        assert key == "app/user123/user/user:test.txt/1"


class TestS3ArtifactServiceConcurrency:
    """Test concurrent operations and thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, artifact_service, sample_artifact):
        """Test concurrent save operations."""
        tasks = []
        for i in range(5):
            task = artifact_service.save_artifact(
                app_name="test_app",
                user_id="user123",
                session_id="session456",
                filename=f"file_{i}.txt",
                artifact=sample_artifact
            )
            tasks.append(task)

        versions = await asyncio.gather(*tasks)
        assert all(v == 0 for v in versions)  # Each file's first version

    @pytest.mark.asyncio
    async def test_concurrent_loads(self, artifact_service, sample_artifact):
        """Test concurrent load operations."""
        # Save artifact first
        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact
        )

        # Concurrent loads
        tasks = []
        for i in range(5):
            task = artifact_service.load_artifact(
                app_name="test_app",
                user_id="user123",
                session_id="session456",
                filename="test_file.txt"
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        assert all(r is not None for r in results)
        assert all(r.inline_data.data == sample_artifact.inline_data.data for r in results)
```

#### 2.1.2 Integration Testing

**File**: `tests/integration/test_s3_integration.py`
```python
"""Integration tests for S3ArtifactService with real S3 operations."""

import os
import pytest
import asyncio
from unittest.mock import patch

from google_adk_aws import S3ArtifactService
from google.genai import types


@pytest.fixture
def integration_config():
    """Configuration for integration tests."""
    return {
        "bucket_name": os.environ.get("S3_TEST_BUCKET", "google-adk-aws-integration-test"),
        "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    }


@pytest.fixture
def real_s3_service(integration_config):
    """Create S3ArtifactService for integration testing."""
    if not os.environ.get("RUN_INTEGRATION_TESTS"):
        pytest.skip("Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable.")

    return S3ArtifactService(
        bucket_name=integration_config["bucket_name"],
        region_name=integration_config["region_name"]
    )


class TestS3IntegrationBasic:
    """Basic integration tests with real S3."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_artifact_lifecycle(self, real_s3_service):
        """Test complete artifact lifecycle with real S3."""
        artifact = types.Part.from_text("Integration test content", mime_type="text/plain")

        # Save artifact
        version = await real_s3_service.save_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="lifecycle_test.txt",
            artifact=artifact
        )
        assert version == 0

        # Load artifact
        loaded = await real_s3_service.load_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="lifecycle_test.txt"
        )
        assert loaded is not None
        assert loaded.inline_data.data == artifact.inline_data.data

        # List artifacts
        keys = await real_s3_service.list_artifact_keys(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session"
        )
        assert "lifecycle_test.txt" in keys

        # List versions
        versions = await real_s3_service.list_versions(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="lifecycle_test.txt"
        )
        assert versions == [0]

        # Delete artifact
        await real_s3_service.delete_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="lifecycle_test.txt"
        )

        # Verify deletion
        loaded_after_delete = await real_s3_service.load_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="lifecycle_test.txt"
        )
        assert loaded_after_delete is None


class TestS3IntegrationPerformance:
    """Performance and scalability integration tests."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_artifact_handling(self, real_s3_service):
        """Test handling of large artifacts (>1MB)."""
        # Create 5MB test content
        large_content = "A" * (5 * 1024 * 1024)
        artifact = types.Part.from_text(large_content, mime_type="text/plain")

        # Save large artifact
        version = await real_s3_service.save_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="large_file.txt",
            artifact=artifact
        )

        # Load and verify
        loaded = await real_s3_service.load_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="large_file.txt"
        )

        assert loaded.inline_data.data == artifact.inline_data.data

        # Cleanup
        await real_s3_service.delete_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="large_file.txt"
        )

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_many_artifacts_performance(self, real_s3_service):
        """Test performance with many artifacts."""
        artifacts_count = 50
        artifact = types.Part.from_text("Performance test", mime_type="text/plain")

        # Save many artifacts concurrently
        tasks = []
        for i in range(artifacts_count):
            task = real_s3_service.save_artifact(
                app_name="integration_test",
                user_id="test_user",
                session_id="perf_session",
                filename=f"perf_file_{i:03d}.txt",
                artifact=artifact
            )
            tasks.append(task)

        versions = await asyncio.gather(*tasks)
        assert len(versions) == artifacts_count

        # List all artifacts
        keys = await real_s3_service.list_artifact_keys(
            app_name="integration_test",
            user_id="test_user",
            session_id="perf_session"
        )
        assert len(keys) == artifacts_count

        # Cleanup - delete all artifacts
        delete_tasks = []
        for i in range(artifacts_count):
            task = real_s3_service.delete_artifact(
                app_name="integration_test",
                user_id="test_user",
                session_id="perf_session",
                filename=f"perf_file_{i:03d}.txt"
            )
            delete_tasks.append(task)

        await asyncio.gather(*delete_tasks)
```

#### 2.1.3 Test Configuration and Automation

**File**: `tests/conftest.py`
```python
"""Shared test configuration and fixtures."""

import os
import pytest
import asyncio
from typing import Generator


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: Integration tests requiring real AWS resources")
    config.addinivalue_line("markers", "slow: Slow running tests (>10 seconds)")
    config.addinivalue_line("markers", "unit: Fast unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment variables between tests."""
    original_env = os.environ.copy()
    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
```

### Sub-Phase 2.2: Advanced Error Handling and Resilience (Week 2)
**Duration**: 5-7 days
**Focus**: Production-grade error handling, retry logic, and fault tolerance

#### 2.2.1 Enhanced Exception Hierarchy

**File**: `src/google_adk_aws/exceptions.py`
```python
"""Enhanced exception hierarchy for S3 artifact operations."""

from typing import Optional, Dict, Any


class S3ArtifactError(Exception):
    """Base exception for S3 artifact operations."""

    def __init__(self, message: str, error_code: Optional[str] = None, **kwargs):
        super().__init__(message)
        self.error_code = error_code
        self.details = kwargs


class S3ConnectionError(S3ArtifactError):
    """Raised when S3 connection fails."""
    pass


class S3PermissionError(S3ArtifactError):
    """Raised when S3 permissions are insufficient."""
    pass


class S3BucketError(S3ArtifactError):
    """Raised when bucket-related operations fail."""
    pass


class S3ObjectError(S3ArtifactError):
    """Raised when object-related operations fail."""
    pass


class S3ThrottleError(S3ArtifactError):
    """Raised when S3 rate limiting occurs."""
    pass


class S3ArtifactNotFoundError(S3ArtifactError):
    """Raised when requested artifact is not found."""
    pass


class S3ArtifactVersionError(S3ArtifactError):
    """Raised when version-related operations fail."""
    pass
```

#### 2.2.2 Retry Logic and Circuit Breaker

**File**: `src/google_adk_aws/retry_handler.py`
```python
"""Retry logic and circuit breaker for S3 operations."""

import asyncio
import logging
import time
from typing import Callable, Any, Optional, Type, Tuple
from functools import wraps
import random

from botocore.exceptions import ClientError
from .exceptions import S3ThrottleError, S3ConnectionError


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
        retryable_errors: Optional[Tuple[Type[Exception], ...]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_errors = retryable_errors or (
            ClientError,
            S3ConnectionError,
            S3ThrottleError
        )


class CircuitBreaker:
    """Circuit breaker pattern for S3 operations."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time < self.timeout:
                    raise S3ConnectionError("Circuit breaker is OPEN")
                else:
                    self.state = "HALF_OPEN"

            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise

        return wrapper

    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic to async functions."""
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_errors as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        logger.error(f"Max retry attempts ({config.max_attempts}) exceeded for {func.__name__}")
                        break

                    # Calculate delay
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )

                    if config.jitter:
                        delay *= (0.5 + random.random() * 0.5)

                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    await asyncio.sleep(delay)
                except Exception as e:
                    # Non-retryable error
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise

            raise last_exception

        return wrapper
    return decorator


def is_throttle_error(error: ClientError) -> bool:
    """Check if error is due to S3 throttling."""
    error_code = error.response.get("Error", {}).get("Code", "")
    return error_code in [
        "Throttling",
        "RequestLimitExceeded",
        "TooManyRequests",
        "SlowDown"
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
            "RequestLimitExceeded"
        ]

        return error_code in retryable_codes

    return isinstance(error, (S3ConnectionError, S3ThrottleError))
```

#### 2.2.3 Enhanced S3ArtifactService with Error Handling

**Update**: `src/google_adk_aws/s3_artifact_service.py` (Enhanced version)
```python
# Add to existing imports
from .retry_handler import with_retry, RetryConfig, CircuitBreaker, is_throttle_error
from .exceptions import (
    S3ArtifactError, S3ConnectionError, S3PermissionError,
    S3BucketError, S3ObjectError, S3ThrottleError,
    S3ArtifactNotFoundError, S3ArtifactVersionError
)

class S3ArtifactService(BaseArtifactService):
    """Enhanced S3-based implementation with robust error handling."""

    def __init__(self, *args, retry_config: Optional[RetryConfig] = None, **kwargs):
        self.retry_config = retry_config or RetryConfig()

        # Initialize circuit breakers for different operation types
        self.read_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=30.0,
            expected_exception=S3ConnectionError
        )

        self.write_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=60.0,
            expected_exception=S3ConnectionError
        )

        super().__init__(*args, **kwargs)

    def _handle_s3_error(self, error: Exception, operation: str) -> None:
        """Enhanced error handling with specific exception mapping."""
        if isinstance(error, ClientError):
            error_code = error.response["Error"]["Code"]
            error_message = error.response["Error"]["Message"]

            if error_code == "NoSuchBucket":
                raise S3BucketError(f"Bucket {self.bucket_name} does not exist") from error
            elif error_code in ["AccessDenied", "Forbidden"]:
                raise S3PermissionError(f"Access denied for {operation}") from error
            elif error_code == "NoSuchKey":
                raise S3ArtifactNotFoundError("Artifact not found") from error
            elif is_throttle_error(error):
                raise S3ThrottleError(f"S3 throttling during {operation}") from error
            elif error_code in ["InternalError", "ServiceUnavailable"]:
                raise S3ConnectionError(f"S3 service error during {operation}") from error
            else:
                raise S3ObjectError(f"S3 {operation} failed: {error_message}") from error
        else:
            raise S3ArtifactError(f"Unexpected error during {operation}: {error}") from error

    @with_retry()
    async def save_artifact(self, *args, **kwargs) -> int:
        """Enhanced save with retry logic and circuit breaker."""
        @self.write_circuit_breaker
        async def _save_with_protection():
            try:
                return await self._save_artifact_impl(*args, **kwargs)
            except Exception as e:
                self._handle_s3_error(e, "save_artifact")

        return await _save_with_protection()

    async def _save_artifact_impl(self, *, app_name: str, user_id: str,
                                  session_id: str, filename: str,
                                  artifact: types.Part) -> int:
        """Implementation of save artifact with enhanced error handling."""
        # Original implementation with additional error context
        try:
            # ... (same logic as original save_artifact)
            pass
        except Exception as e:
            logger.error(
                f"Save artifact failed - App: {app_name}, User: {user_id}, "
                f"Session: {session_id}, File: {filename}, Error: {e}"
            )
            raise

    @with_retry()
    async def load_artifact(self, *args, **kwargs) -> Optional[types.Part]:
        """Enhanced load with retry logic and circuit breaker."""
        @self.read_circuit_breaker
        async def _load_with_protection():
            try:
                return await self._load_artifact_impl(*args, **kwargs)
            except S3ArtifactNotFoundError:
                # Not found is not an error condition
                return None
            except Exception as e:
                self._handle_s3_error(e, "load_artifact")

        return await _load_with_protection()

    # Similar enhancements for other methods...
```

### Sub-Phase 2.3: Performance Optimization (Week 3)
**Duration**: 5-7 days
**Focus**: Connection pooling, async improvements, and efficiency

#### 2.3.1 Connection Pool Manager

**File**: `src/google_adk_aws/connection_pool.py`
```python
"""S3 connection pool management for optimized performance."""

import asyncio
import logging
import time
import threading
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.config import Config


logger = logging.getLogger(__name__)


class S3ConnectionPool:
    """Manages pooled S3 client connections for better performance."""

    def __init__(
        self,
        max_pool_connections: int = 50,
        max_workers: int = 20,
        connect_timeout: int = 60,
        read_timeout: int = 60,
        retries_config: Optional[Dict[str, Any]] = None
    ):
        self.max_pool_connections = max_pool_connections
        self.max_workers = max_workers
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

        # Default retries configuration
        self.retries_config = retries_config or {
            'max_attempts': 3,
            'mode': 'adaptive'
        }

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="s3-pool"
        )

        # Client cache with connection reuse
        self._clients: Dict[str, boto3.client] = {}
        self._client_lock = threading.Lock()

        # Connection pool statistics
        self._stats = {
            'total_connections': 0,
            'active_connections': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def get_client(
        self,
        region_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ) -> boto3.client:
        """Get or create optimized S3 client with connection pooling."""

        # Create cache key
        cache_key = f"{region_name}:{aws_access_key_id}:{endpoint_url}"

        with self._client_lock:
            if cache_key in self._clients:
                self._stats['cache_hits'] += 1
                return self._clients[cache_key]

            self._stats['cache_misses'] += 1

            # Create optimized boto3 config
            config = Config(
                region_name=region_name,
                retries=self.retries_config,
                max_pool_connections=self.max_pool_connections,
                connect_timeout=self.connect_timeout,
                read_timeout=self.read_timeout,
                # Enable signature_version for better performance
                signature_version='s3v4',
                # Use virtual hosted-style addressing
                addressing_style='virtual'
            )

            # Create session
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name
            )

            # Create client with configuration
            client_kwargs = {'config': config}
            if endpoint_url:
                client_kwargs['endpoint_url'] = endpoint_url

            client = session.client('s3', **client_kwargs)

            # Cache the client
            self._clients[cache_key] = client
            self._stats['total_connections'] += 1

            logger.info(f"Created new S3 client for region {region_name}")
            return client

    async def execute_async(self, func, *args, **kwargs):
        """Execute S3 operation asynchronously using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, func, *args, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._client_lock:
            stats = self._stats.copy()
            stats['active_connections'] = len(self._clients)
            return stats

    def close(self):
        """Close all connections and cleanup resources."""
        with self._client_lock:
            self._clients.clear()

        self._executor.shutdown(wait=True)
        logger.info("S3 connection pool closed")

    def __del__(self):
        """Cleanup on object destruction."""
        self.close()


# Global connection pool instance
_connection_pool: Optional[S3ConnectionPool] = None
_pool_lock = threading.Lock()


def get_connection_pool() -> S3ConnectionPool:
    """Get or create global S3 connection pool."""
    global _connection_pool

    with _pool_lock:
        if _connection_pool is None:
            _connection_pool = S3ConnectionPool()
        return _connection_pool


def close_connection_pool():
    """Close global connection pool."""
    global _connection_pool

    with _pool_lock:
        if _connection_pool is not None:
            _connection_pool.close()
            _connection_pool = None
```

#### 2.3.2 Batch Operations

**File**: `src/google_adk_aws/batch_operations.py`
```python
"""Batch operations for improved S3 performance."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import as_completed
import boto3
from botocore.exceptions import ClientError

from .exceptions import S3ArtifactError


logger = logging.getLogger(__name__)


class S3BatchOperations:
    """Optimized batch operations for S3 artifacts."""

    def __init__(self, s3_client: boto3.client, bucket_name: str, max_concurrent: int = 10):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def batch_delete(self, object_keys: List[str]) -> Dict[str, Any]:
        """Delete multiple objects in batches for efficiency."""
        if not object_keys:
            return {"deleted": [], "errors": []}

        results = {"deleted": [], "errors": []}

        # S3 supports up to 1000 objects per delete request
        batch_size = 1000

        for i in range(0, len(object_keys), batch_size):
            batch = object_keys[i:i + batch_size]

            try:
                await self._delete_batch(batch, results)
            except Exception as e:
                logger.error(f"Batch delete failed for batch starting at index {i}: {e}")
                # Add all objects in failed batch to errors
                for key in batch:
                    results["errors"].append({
                        "key": key,
                        "error": str(e)
                    })

        return results

    async def _delete_batch(self, object_keys: List[str], results: Dict[str, Any]):
        """Delete a single batch of objects."""
        async with self._semaphore:
            # Prepare delete request
            delete_request = {
                'Objects': [{'Key': key} for key in object_keys],
                'Quiet': False  # Return both successful and failed deletes
            }

            def _execute_delete():
                return self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete=delete_request
                )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _execute_delete)

            # Process results
            for deleted in response.get('Deleted', []):
                results["deleted"].append(deleted['Key'])

            for error in response.get('Errors', []):
                results["errors"].append({
                    "key": error['Key'],
                    "error": f"{error['Code']}: {error['Message']}"
                })

    async def batch_upload(
        self,
        upload_specs: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Upload multiple objects concurrently."""
        if not upload_specs:
            return {"uploaded": [], "errors": []}

        results = {"uploaded": [], "errors": []}

        # Create upload tasks with concurrency control
        tasks = []
        for spec in upload_specs:
            task = self._upload_single(spec, results, progress_callback)
            tasks.append(task)

        # Execute uploads with limited concurrency
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def _upload_single(
        self,
        upload_spec: Dict[str, Any],
        results: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ):
        """Upload a single object."""
        async with self._semaphore:
            try:
                def _execute_upload():
                    return self.s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=upload_spec['key'],
                        Body=upload_spec['body'],
                        ContentType=upload_spec.get('content_type', 'application/octet-stream'),
                        Metadata=upload_spec.get('metadata', {})
                    )

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, _execute_upload)

                results["uploaded"].append(upload_spec['key'])

                if progress_callback:
                    progress_callback(upload_spec['key'], "uploaded")

            except Exception as e:
                logger.error(f"Upload failed for {upload_spec['key']}: {e}")
                results["errors"].append({
                    "key": upload_spec['key'],
                    "error": str(e)
                })

                if progress_callback:
                    progress_callback(upload_spec['key'], "error", str(e))

    async def batch_list_objects(
        self,
        prefixes: List[str],
        max_keys_per_prefix: int = 1000
    ) -> Dict[str, List[Dict[str, Any]]]:
        """List objects for multiple prefixes concurrently."""
        if not prefixes:
            return {}

        results = {}

        # Create list tasks
        tasks = []
        for prefix in prefixes:
            task = self._list_objects_for_prefix(prefix, max_keys_per_prefix)
            tasks.append((prefix, task))

        # Execute listing operations
        for prefix, task in tasks:
            try:
                objects = await task
                results[prefix] = objects
            except Exception as e:
                logger.error(f"List objects failed for prefix {prefix}: {e}")
                results[prefix] = []

        return results

    async def _list_objects_for_prefix(
        self,
        prefix: str,
        max_keys: int
    ) -> List[Dict[str, Any]]:
        """List objects for a single prefix."""
        async with self._semaphore:
            def _execute_list():
                paginator = self.s3_client.get_paginator('list_objects_v2')
                objects = []

                for page in paginator.paginate(
                    Bucket=self.bucket_name,
                    Prefix=prefix,
                    MaxKeys=max_keys
                ):
                    objects.extend(page.get('Contents', []))

                return objects

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _execute_list)


class MultipartUploadManager:
    """Manages multipart uploads for large artifacts."""

    def __init__(self, s3_client: boto3.client, bucket_name: str):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.multipart_threshold = 100 * 1024 * 1024  # 100MB
        self.part_size = 10 * 1024 * 1024  # 10MB parts

    async def upload_large_artifact(
        self,
        object_key: str,
        data: bytes,
        content_type: str,
        metadata: Dict[str, str],
        progress_callback: Optional[callable] = None
    ) -> str:
        """Upload large artifact using multipart upload if necessary."""

        if len(data) < self.multipart_threshold:
            # Use regular upload for smaller files
            return await self._regular_upload(
                object_key, data, content_type, metadata
            )
        else:
            # Use multipart upload for large files
            return await self._multipart_upload(
                object_key, data, content_type, metadata, progress_callback
            )

    async def _regular_upload(
        self,
        object_key: str,
        data: bytes,
        content_type: str,
        metadata: Dict[str, str]
    ) -> str:
        """Regular S3 upload for smaller files."""
        def _execute_upload():
            return self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=data,
                ContentType=content_type,
                Metadata=metadata
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _execute_upload)
        return response['ETag']

    async def _multipart_upload(
        self,
        object_key: str,
        data: bytes,
        content_type: str,
        metadata: Dict[str, str],
        progress_callback: Optional[callable] = None
    ) -> str:
        """Multipart upload for large files."""

        # Initiate multipart upload
        def _initiate_upload():
            return self.s3_client.create_multipart_upload(
                Bucket=self.bucket_name,
                Key=object_key,
                ContentType=content_type,
                Metadata=metadata
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _initiate_upload)
        upload_id = response['UploadId']

        try:
            # Upload parts
            parts = await self._upload_parts(
                object_key, upload_id, data, progress_callback
            )

            # Complete multipart upload
            def _complete_upload():
                return self.s3_client.complete_multipart_upload(
                    Bucket=self.bucket_name,
                    Key=object_key,
                    UploadId=upload_id,
                    MultipartUpload={'Parts': parts}
                )

            response = await loop.run_in_executor(None, _complete_upload)
            return response['ETag']

        except Exception as e:
            # Abort multipart upload on error
            try:
                def _abort_upload():
                    self.s3_client.abort_multipart_upload(
                        Bucket=self.bucket_name,
                        Key=object_key,
                        UploadId=upload_id
                    )

                await loop.run_in_executor(None, _abort_upload)
            except Exception as abort_error:
                logger.error(f"Failed to abort multipart upload: {abort_error}")

            raise S3ArtifactError(f"Multipart upload failed: {e}") from e

    async def _upload_parts(
        self,
        object_key: str,
        upload_id: str,
        data: bytes,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Upload individual parts of multipart upload."""

        # Calculate parts
        total_size = len(data)
        num_parts = (total_size + self.part_size - 1) // self.part_size

        # Upload parts concurrently
        tasks = []
        for part_num in range(1, num_parts + 1):
            start = (part_num - 1) * self.part_size
            end = min(start + self.part_size, total_size)
            part_data = data[start:end]

            task = self._upload_part(
                object_key, upload_id, part_num, part_data, progress_callback
            )
            tasks.append(task)

        # Wait for all parts to complete
        parts = await asyncio.gather(*tasks)

        # Sort parts by part number
        return sorted(parts, key=lambda p: p['PartNumber'])

    async def _upload_part(
        self,
        object_key: str,
        upload_id: str,
        part_number: int,
        part_data: bytes,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Upload a single part."""

        def _execute_part_upload():
            return self.s3_client.upload_part(
                Bucket=self.bucket_name,
                Key=object_key,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=part_data
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _execute_part_upload)

        if progress_callback:
            progress_callback(part_number, len(part_data), "uploaded")

        return {
            'ETag': response['ETag'],
            'PartNumber': part_number
        }
```

### Sub-Phase 2.4: Production Features and Security (Week 4)
**Duration**: 5-7 days
**Focus**: Security hardening, monitoring, logging, and deployment features

#### 2.4.1 Security Enhancements

**File**: `src/google_adk_aws/security.py`
```python
"""Security features and hardening for S3 artifact service."""

import hashlib
import hmac
import logging
import os
import secrets
from typing import Dict, Optional, Any, List
import boto3
from botocore.exceptions import ClientError

from .exceptions import S3PermissionError, S3ArtifactError


logger = logging.getLogger(__name__)


class S3SecurityManager:
    """Manages security features for S3 artifact operations."""

    def __init__(self, s3_client: boto3.client, bucket_name: str):
        self.s3_client = s3_client
        self.bucket_name = bucket_name

    def validate_bucket_security(self) -> Dict[str, Any]:
        """Validate bucket security configuration."""
        security_status = {
            "encryption": False,
            "versioning": False,
            "public_access_blocked": False,
            "logging_enabled": False,
            "mfa_delete": False,
            "recommendations": []
        }

        try:
            # Check encryption
            try:
                encryption = self.s3_client.get_bucket_encryption(Bucket=self.bucket_name)
                security_status["encryption"] = True
            except ClientError as e:
                if e.response["Error"]["Code"] != "ServerSideEncryptionConfigurationNotFoundError":
                    raise
                security_status["recommendations"].append(
                    "Enable server-side encryption for enhanced security"
                )

            # Check versioning
            try:
                versioning = self.s3_client.get_bucket_versioning(Bucket=self.bucket_name)
                if versioning.get("Status") == "Enabled":
                    security_status["versioning"] = True
                    if versioning.get("MfaDelete") == "Enabled":
                        security_status["mfa_delete"] = True
                else:
                    security_status["recommendations"].append(
                        "Enable versioning for better data protection"
                    )
            except ClientError:
                pass

            # Check public access block
            try:
                public_access = self.s3_client.get_public_access_block(Bucket=self.bucket_name)
                block_config = public_access.get("PublicAccessBlockConfiguration", {})
                if all([
                    block_config.get("BlockPublicAcls", False),
                    block_config.get("IgnorePublicAcls", False),
                    block_config.get("BlockPublicPolicy", False),
                    block_config.get("RestrictPublicBuckets", False)
                ]):
                    security_status["public_access_blocked"] = True
                else:
                    security_status["recommendations"].append(
                        "Enable public access block for all settings"
                    )
            except ClientError:
                security_status["recommendations"].append(
                    "Configure public access block settings"
                )

            # Check logging
            try:
                logging_config = self.s3_client.get_bucket_logging(Bucket=self.bucket_name)
                if "LoggingEnabled" in logging_config:
                    security_status["logging_enabled"] = True
                else:
                    security_status["recommendations"].append(
                        "Enable access logging for audit trails"
                    )
            except ClientError:
                pass

        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            raise S3ArtifactError(f"Failed to validate bucket security: {e}") from e

        return security_status

    def generate_secure_object_key(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: int
    ) -> str:
        """Generate secure object key with sanitization."""

        # Sanitize inputs
        def sanitize(value: str) -> str:
            """Remove/replace unsafe characters."""
            # Allow alphanumeric, hyphens, underscores, dots
            import re
            return re.sub(r'[^a-zA-Z0-9\-_.]', '_', str(value))

        safe_app_name = sanitize(app_name)
        safe_user_id = sanitize(user_id)
        safe_session_id = sanitize(session_id)
        safe_filename = sanitize(filename)

        # Construct secure path
        if filename.startswith("user:"):
            return f"{safe_app_name}/{safe_user_id}/user/{safe_filename}/{version}"
        else:
            return f"{safe_app_name}/{safe_user_id}/{safe_session_id}/{safe_filename}/{version}"

    def validate_object_key(self, object_key: str) -> bool:
        """Validate object key for security compliance."""

        # Check length (S3 limit is 1024 characters)
        if len(object_key) > 1024:
            return False

        # Check for path traversal attempts
        dangerous_patterns = ["../", "..\\", "/..", "\\..", "//"]
        for pattern in dangerous_patterns:
            if pattern in object_key:
                return False

        # Ensure key starts with expected app prefix pattern
        key_parts = object_key.split("/")
        if len(key_parts) < 4:  # app/user/session/filename or app/user/user/filename
            return False

        return True

    def calculate_content_hash(self, content: bytes, algorithm: str = "sha256") -> str:
        """Calculate secure hash of content for integrity verification."""
        if algorithm == "sha256":
            return hashlib.sha256(content).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(content).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    def verify_content_integrity(
        self,
        content: bytes,
        expected_hash: str,
        algorithm: str = "sha256"
    ) -> bool:
        """Verify content integrity using hash comparison."""
        actual_hash = self.calculate_content_hash(content, algorithm)
        return hmac.compare_digest(actual_hash, expected_hash)


class AccessControlManager:
    """Manages access control and permissions for artifacts."""

    def __init__(self):
        self.access_patterns = {
            "session_scoped": "{app_name}/{user_id}/{session_id}/*",
            "user_scoped": "{app_name}/{user_id}/user/*",
            "admin_scoped": "{app_name}/*"
        }

    def check_access_permission(
        self,
        operation: str,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        user_permissions: List[str]
    ) -> bool:
        """Check if user has permission for the operation."""

        # Define required permissions for each operation
        operation_permissions = {
            "read": ["artifact:read", "artifact:*"],
            "write": ["artifact:write", "artifact:*"],
            "delete": ["artifact:delete", "artifact:*"],
            "list": ["artifact:list", "artifact:read", "artifact:*"]
        }

        required_perms = operation_permissions.get(operation, [])

        # Check if user has any required permission
        for perm in required_perms:
            if perm in user_permissions:
                return True

        # Check scope-specific permissions
        if filename.startswith("user:"):
            # User-scoped artifact
            scope_perm = f"artifact:{operation}:user"
            if scope_perm in user_permissions:
                return True
        else:
            # Session-scoped artifact
            scope_perm = f"artifact:{operation}:session"
            if scope_perm in user_permissions:
                return True

        return False

    def generate_presigned_url(
        self,
        s3_client: boto3.client,
        bucket_name: str,
        object_key: str,
        operation: str,
        expiration: int = 3600,
        user_permissions: Optional[List[str]] = None
    ) -> str:
        """Generate secure presigned URL for artifact access."""

        # Validate operation
        valid_operations = ["get_object", "put_object"]
        if operation not in valid_operations:
            raise ValueError(f"Invalid operation: {operation}")

        # Check permissions if provided
        if user_permissions:
            op_type = "read" if operation == "get_object" else "write"
            # Additional permission checking logic here

        try:
            # Generate presigned URL
            url = s3_client.generate_presigned_url(
                operation,
                Params={"Bucket": bucket_name, "Key": object_key},
                ExpiresIn=expiration
            )

            logger.info(f"Generated presigned URL for {operation} on {object_key}")
            return url

        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise S3ArtifactError(f"Presigned URL generation failed: {e}") from e


class EncryptionManager:
    """Manages client-side encryption for sensitive artifacts."""

    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or self._generate_key()

    def _generate_key(self) -> str:
        """Generate a new encryption key."""
        return secrets.token_hex(32)  # 256-bit key

    def encrypt_content(self, content: bytes) -> tuple[bytes, str]:
        """Encrypt content and return encrypted data with metadata."""
        # This is a simplified example - use proper encryption libraries
        # like cryptography for production use

        try:
            from cryptography.fernet import Fernet
            import base64

            # Use the key to create Fernet cipher
            key = base64.urlsafe_b64encode(self.encryption_key.encode()[:32].ljust(32, b'0'))
            cipher = Fernet(key)

            encrypted_content = cipher.encrypt(content)

            # Return encrypted content and encryption metadata
            metadata = {
                "encryption": "client-side",
                "algorithm": "fernet"
            }

            return encrypted_content, metadata

        except ImportError:
            raise S3ArtifactError(
                "cryptography package required for client-side encryption"
            )

    def decrypt_content(self, encrypted_content: bytes, metadata: Dict[str, str]) -> bytes:
        """Decrypt content using stored metadata."""

        if metadata.get("encryption") != "client-side":
            return encrypted_content  # Not encrypted

        try:
            from cryptography.fernet import Fernet
            import base64

            key = base64.urlsafe_b64encode(self.encryption_key.encode()[:32].ljust(32, b'0'))
            cipher = Fernet(key)

            return cipher.decrypt(encrypted_content)

        except ImportError:
            raise S3ArtifactError(
                "cryptography package required for client-side decryption"
            )
        except Exception as e:
            raise S3ArtifactError(f"Decryption failed: {e}") from e
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "phase2-doc-creation", "content": "Create comprehensive Phase 2 implementation plan document", "status": "completed", "priority": "high"}, {"id": "analyze-requirements", "content": "Analyze roadmap Phase 2 requirements (AI/ML services)", "status": "completed", "priority": "high"}, {"id": "define-deliverables", "content": "Define specific Phase 2 deliverables and timeline", "status": "completed", "priority": "medium"}]
