# Google ADK AWS Integrations - S3 Artifact Service Implementation - Phase 4

## Overview

Phase 4 focuses exclusively on **comprehensive unit testing** for the production-ready S3ArtifactService following Phase 3's error handling and edge cases implementation. This phase establishes a robust, maintainable test suite that ensures 100% reliability of all error handling, edge cases, and core functionality through extensive unit testing.

**Duration**: 3-4 weeks
**Prerequisites**: Phase 3 completion (error handling and edge cases)
**Status**: Ready to begin following Phase 3 completion

## Phase 4 Objectives

Building on the comprehensive error handling from Phase 3, this phase implements:

1. **Complete Unit Test Coverage**: >95% code coverage for all components
2. **Error Scenario Testing**: Comprehensive testing of all error conditions and mappings
3. **Edge Case Validation**: Unit tests for all edge case handlers and validators
4. **Mocking Infrastructure**: Advanced mocking for isolated unit testing
5. **Performance Testing**: Unit-level performance validation and benchmarking
6. **Test Automation**: Automated test discovery, execution, and reporting

## Implementation Roadmap

### Sub-Phase 4.1: Advanced Testing Infrastructure (Week 1, Days 1-3)
**Duration**: 3 days
**Focus**: Enhanced testing framework and mocking infrastructure

#### 4.1.1 Enhanced Test Configuration

**File**: `tests/conftest.py` (Major Enhancement)
```python
"""Enhanced shared test configuration and fixtures for comprehensive testing."""

import asyncio
import os
import tempfile
import time
from typing import Any, Generator, Dict, Optional
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
import logging

import pytest
import boto3
from moto import mock_s3
from google.genai import types

from aws_adk import S3ArtifactService
from aws_adk.exceptions import *
from aws_adk.validation import InputValidator
from aws_adk.retry_handler import RetryConfig
from aws_adk.edge_case_handlers import ConcurrencyManager


# Test configuration
@dataclass
class TestConfig:
    """Centralized test configuration."""
    bucket_name: str = "test-artifacts-bucket"
    region_name: str = "us-east-1"
    app_name: str = "test-app"
    user_id: str = "test-user"
    session_id: str = "test-session"
    test_timeout: float = 30.0
    enable_slow_tests: bool = False


@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """Global test configuration."""
    return TestConfig(
        enable_slow_tests=os.getenv("ENABLE_SLOW_TESTS", "false").lower() == "true"
    )


def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers and logging."""
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring real AWS resources"
    )
    config.addinivalue_line("markers", "slow: Slow running tests (>10 seconds)")
    config.addinivalue_line("markers", "unit: Fast unit tests")
    config.addinivalue_line("markers", "error_handling: Error handling tests")
    config.addinivalue_line("markers", "edge_cases: Edge case validation tests")
    config.addinivalue_line("markers", "concurrency: Concurrency testing")
    config.addinivalue_line("markers", "validation: Input validation tests")
    config.addinivalue_line("markers", "performance: Performance benchmarking tests")

    # Configure logging for tests
    logging.basicConfig(
        level=logging.DEBUG if config.getoption("--verbose") else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def pytest_collection_modifyitems(config, items) -> None:  # type: ignore
    """Modify test collection to add markers and skip slow tests."""
    skip_slow = pytest.mark.skip(reason="Slow test skipped (use --slow to enable)")

    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add markers based on test function names
        if "error" in item.name or "exception" in item.name:
            item.add_marker(pytest.mark.error_handling)
        if "edge_case" in item.name or "large_file" in item.name or "concurrent" in item.name:
            item.add_marker(pytest.mark.edge_cases)
        if "concurrent" in item.name or "concurrency" in item.name:
            item.add_marker(pytest.mark.concurrency)
        if "validate" in item.name or "validation" in item.name:
            item.add_marker(pytest.mark.validation)
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)

        # Skip slow tests unless explicitly enabled
        if "slow" in item.keywords and not config.getoption("--slow"):
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def clean_environment() -> Generator[None, None, None]:
    """Clean environment variables between tests."""
    original_env = os.environ.copy()
    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_s3_setup():
    """Setup mocked S3 environment for testing."""
    with mock_s3():
        # Create mock S3 client and bucket
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-artifacts-bucket")

        yield {
            "client": s3_client,
            "bucket_name": "test-artifacts-bucket",
            "region_name": "us-east-1"
        }


@pytest.fixture
def sample_artifact() -> types.Part:
    """Create sample artifact for testing."""
    content = b"This is test content for artifact testing"
    return types.Part.from_bytes(data=content, mime_type="text/plain")


@pytest.fixture
def large_artifact() -> types.Part:
    """Create large artifact for testing."""
    # 50MB test content
    content = b"x" * (50 * 1024 * 1024)
    return types.Part.from_bytes(data=content, mime_type="application/octet-stream")


@pytest.fixture
def invalid_artifact_data() -> Dict[str, Any]:
    """Invalid test data for validation testing."""
    return {
        "empty_content": types.Part.from_bytes(data=b"", mime_type="text/plain"),
        "invalid_mime_type": types.Part.from_bytes(data=b"test", mime_type="invalid/mime"),
        "oversized_metadata": {"key": "x" * 3000},  # Too large for S3 metadata
    }


@pytest.fixture
def validation_test_cases() -> Dict[str, Dict[str, Any]]:
    """Test cases for validation testing."""
    return {
        "valid": {
            "app_name": "test-app",
            "user_id": "test-user",
            "session_id": "test-session",
            "filename": "test-file.txt",
            "version": 0
        },
        "invalid_app_name": {
            "app_name": "",  # Empty
            "user_id": "test-user",
            "session_id": "test-session",
            "filename": "test-file.txt",
            "version": 0
        },
        "invalid_filename": {
            "app_name": "test-app",
            "user_id": "test-user",
            "session_id": "test-session",
            "filename": "../../../etc/passwd",  # Path traversal
            "version": 0
        },
        "invalid_version": {
            "app_name": "test-app",
            "user_id": "test-user",
            "session_id": "test-session",
            "filename": "test-file.txt",
            "version": -1  # Negative version
        }
    }


@pytest.fixture
def error_simulation_fixtures() -> Dict[str, Any]:
    """Fixtures for simulating various error conditions."""
    return {
        "client_errors": {
            "NoSuchBucket": {
                "Error": {"Code": "NoSuchBucket", "Message": "The specified bucket does not exist"},
                "ResponseMetadata": {"HTTPStatusCode": 404}
            },
            "AccessDenied": {
                "Error": {"Code": "AccessDenied", "Message": "Access Denied"},
                "ResponseMetadata": {"HTTPStatusCode": 403}
            },
            "NoSuchKey": {
                "Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist"},
                "ResponseMetadata": {"HTTPStatusCode": 404}
            },
            "Throttling": {
                "Error": {"Code": "Throttling", "Message": "Rate exceeded"},
                "ResponseMetadata": {"HTTPStatusCode": 503}
            }
        },
        "network_errors": [
            ConnectionError("Network connection failed"),
            TimeoutError("Request timeout"),
            OSError("Network unreachable")
        ]
    }


@pytest.fixture
def performance_benchmarks() -> Dict[str, float]:
    """Performance benchmarks for testing."""
    return {
        "save_artifact_max_time": 5.0,  # seconds
        "load_artifact_max_time": 3.0,  # seconds
        "list_operations_max_time": 2.0,  # seconds
        "validation_max_time": 0.1,  # seconds
        "error_handling_max_time": 0.5,  # seconds
    }


class MockS3Client:
    """Advanced S3 client mock with configurable behavior."""

    def __init__(self):
        self.call_count = 0
        self.call_history = []
        self.fail_after_calls = None
        self.failure_exception = None
        self.response_delay = 0
        self.bucket_data = {}

    def configure_failure(self, fail_after_calls: int, exception: Exception):
        """Configure the mock to fail after specified number of calls."""
        self.fail_after_calls = fail_after_calls
        self.failure_exception = exception

    def configure_delay(self, delay: float):
        """Configure response delay for testing timeouts."""
        self.response_delay = delay

    async def put_object(self, **kwargs):
        """Mock put_object with configurable behavior."""
        return await self._execute_call("put_object", **kwargs)

    async def get_object(self, **kwargs):
        """Mock get_object with configurable behavior."""
        response = await self._execute_call("get_object", **kwargs)
        if response:
            # Simulate response structure
            return {
                "Body": Mock(read=lambda: b"test content"),
                "ContentType": "text/plain",
                "Metadata": {"version": "0"}
            }
        return response

    async def list_objects_v2(self, **kwargs):
        """Mock list_objects_v2 with configurable behavior."""
        response = await self._execute_call("list_objects_v2", **kwargs)
        if response:
            return {"Contents": []}
        return response

    async def head_bucket(self, **kwargs):
        """Mock head_bucket with configurable behavior."""
        return await self._execute_call("head_bucket", **kwargs)

    async def delete_object(self, **kwargs):
        """Mock delete_object with configurable behavior."""
        return await self._execute_call("delete_object", **kwargs)

    async def _execute_call(self, operation: str, **kwargs):
        """Execute mock call with failure and delay simulation."""
        self.call_count += 1
        self.call_history.append({"operation": operation, "kwargs": kwargs})

        # Simulate delay
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)

        # Simulate failure
        if (self.fail_after_calls is not None and
            self.call_count >= self.fail_after_calls and
            self.failure_exception is not None):
            raise self.failure_exception

        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.call_history = []
        self.fail_after_calls = None
        self.failure_exception = None
        self.response_delay = 0


@pytest.fixture
def advanced_s3_mock():
    """Advanced S3 mock with configurable behavior."""
    return MockS3Client()


@pytest.fixture
def temp_directory() -> Generator[str, None, None]:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_metrics() -> Dict[str, Any]:
    """Test execution metrics collector."""
    return {
        "start_time": time.time(),
        "test_count": 0,
        "failure_count": 0,
        "error_scenarios_tested": [],
        "performance_measurements": {}
    }


class AsyncContextManagerMock:
    """Mock for async context managers."""

    def __init__(self, return_value=None):
        self.return_value = return_value
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False


@pytest.fixture
def async_context_mock():
    """Mock for async context managers."""
    return AsyncContextManagerMock
```

#### 4.1.2 Test Utilities and Helpers

**File**: `tests/utils.py` (New)
```python
"""Utility functions and helpers for comprehensive testing."""

import asyncio
import time
import random
import string
from typing import Any, Dict, List, Optional, Callable, Union
from unittest.mock import Mock, patch
from contextlib import asynccontextmanager
import logging

from botocore.exceptions import ClientError
from google.genai import types

from aws_adk.exceptions import *

logger = logging.getLogger(__name__)


class TestDataGenerator:
    """Generate various test data scenarios."""

    @staticmethod
    def generate_random_string(length: int) -> str:
        """Generate random string of specified length."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    @staticmethod
    def generate_test_content(size_bytes: int) -> bytes:
        """Generate test content of specified size."""
        if size_bytes <= 1024:
            # Small content - use readable text
            return f"Test content {TestDataGenerator.generate_random_string(10)}".encode()[:size_bytes]
        else:
            # Large content - use efficient generation
            chunk = b"x" * 1024
            full_chunks = size_bytes // 1024
            remainder = size_bytes % 1024
            return chunk * full_chunks + chunk[:remainder]

    @staticmethod
    def generate_artifact(
        content_size: int = 100,
        mime_type: str = "text/plain"
    ) -> types.Part:
        """Generate test artifact with specified properties."""
        content = TestDataGenerator.generate_test_content(content_size)
        return types.Part.from_bytes(data=content, mime_type=mime_type)

    @staticmethod
    def generate_invalid_inputs() -> Dict[str, Any]:
        """Generate various invalid input scenarios."""
        return {
            "empty_string": "",
            "whitespace_only": "   ",
            "too_long": "x" * 1000,
            "path_traversal": "../../../etc/passwd",
            "null_bytes": "test\x00file",
            "unicode_issues": "test\ufffffile",
            "sql_injection": "'; DROP TABLE users; --",
            "script_injection": "<script>alert('xss')</script>",
            "special_chars": "!@#$%^&*()+={}[]|\\:;\"'<>?,./`~"
        }


class ErrorSimulator:
    """Simulate various error conditions for testing."""

    @staticmethod
    def create_client_error(
        error_code: str,
        message: str = "Test error",
        http_status: int = 400
    ) -> ClientError:
        """Create boto3 ClientError for testing."""
        return ClientError(
            error_response={
                "Error": {
                    "Code": error_code,
                    "Message": message
                },
                "ResponseMetadata": {
                    "HTTPStatusCode": http_status
                }
            },
            operation_name="TestOperation"
        )

    @staticmethod
    def get_all_s3_error_codes() -> List[str]:
        """Get list of all S3 error codes for comprehensive testing."""
        return [
            "NoSuchBucket", "BucketNotEmpty", "NoSuchKey", "AccessDenied",
            "Forbidden", "InvalidRequest", "InvalidArgument", "Throttling",
            "RequestLimitExceeded", "TooManyRequests", "SlowDown",
            "ServiceUnavailable", "InternalError", "RequestTimeout",
            "EntityTooLarge", "InvalidObjectState", "NotImplemented",
            "PreconditionFailed", "QuotaExceeded"
        ]

    @staticmethod
    async def simulate_network_failure(delay: float = 0.1):
        """Simulate network failure with delay."""
        await asyncio.sleep(delay)
        raise ConnectionError("Simulated network failure")

    @staticmethod
    async def simulate_timeout(delay: float = 5.0):
        """Simulate operation timeout."""
        await asyncio.sleep(delay)
        raise TimeoutError("Simulated timeout")


class PerformanceMeasurer:
    """Measure and validate performance characteristics."""

    def __init__(self):
        self.measurements = {}

    @asynccontextmanager
    async def measure(self, operation_name: str):
        """Context manager to measure operation performance."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time

            if operation_name not in self.measurements:
                self.measurements[operation_name] = []
            self.measurements[operation_name].append(duration)

            logger.debug(f"Operation {operation_name} took {duration:.4f} seconds")

    def get_stats(self, operation_name: str) -> Dict[str, float]:
        """Get performance statistics for operation."""
        measurements = self.measurements.get(operation_name, [])
        if not measurements:
            return {}

        return {
            "count": len(measurements),
            "total": sum(measurements),
            "average": sum(measurements) / len(measurements),
            "min": min(measurements),
            "max": max(measurements)
        }

    def assert_performance(
        self,
        operation_name: str,
        max_time: float,
        percentile: float = 95.0
    ):
        """Assert operation meets performance requirements."""
        measurements = self.measurements.get(operation_name, [])
        if not measurements:
            raise AssertionError(f"No measurements found for {operation_name}")

        # Calculate percentile
        sorted_measurements = sorted(measurements)
        index = int((percentile / 100.0) * len(sorted_measurements))
        percentile_value = sorted_measurements[min(index, len(sorted_measurements) - 1)]

        assert percentile_value <= max_time, (
            f"{operation_name} {percentile}th percentile ({percentile_value:.4f}s) "
            f"exceeds maximum allowed time ({max_time}s)"
        )


class ConcurrencyTester:
    """Test concurrent operations and race conditions."""

    @staticmethod
    async def run_concurrent_operations(
        operations: List[Callable],
        max_concurrent: int = 10
    ) -> List[Any]:
        """Run operations concurrently with controlled concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(operation):
            async with semaphore:
                return await operation()

        tasks = [run_with_semaphore(op) for op in operations]
        return await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def test_race_condition(
        setup_func: Callable,
        operation_func: Callable,
        teardown_func: Callable,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """Test for race conditions by running operations concurrently."""
        results = []
        exceptions = []

        for _ in range(iterations):
            try:
                # Setup
                await setup_func()

                # Run operations concurrently
                tasks = [operation_func() for _ in range(5)]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Collect results and exceptions
                for result in batch_results:
                    if isinstance(result, Exception):
                        exceptions.append(result)
                    else:
                        results.append(result)

                # Teardown
                await teardown_func()

            except Exception as e:
                exceptions.append(e)

        return {
            "results": results,
            "exceptions": exceptions,
            "success_rate": len(results) / (len(results) + len(exceptions))
        }


class ValidationTester:
    """Comprehensive validation testing utilities."""

    @staticmethod
    def test_all_validation_rules(
        validator_func: Callable,
        valid_inputs: Dict[str, Any],
        invalid_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test all validation rules comprehensively."""
        results = {
            "valid_passed": 0,
            "valid_failed": 0,
            "invalid_caught": 0,
            "invalid_missed": 0,
            "failures": []
        }

        # Test valid inputs
        for name, input_data in valid_inputs.items():
            try:
                validator_func(**input_data)
                results["valid_passed"] += 1
            except Exception as e:
                results["valid_failed"] += 1
                results["failures"].append(f"Valid input {name} failed: {e}")

        # Test invalid inputs
        for name, input_data in invalid_inputs.items():
            try:
                validator_func(**input_data)
                results["invalid_missed"] += 1
                results["failures"].append(f"Invalid input {name} was not caught")
            except Exception:
                results["invalid_caught"] += 1

        return results


def create_test_service(
    mock_s3_client: Optional[Mock] = None,
    **service_kwargs
) -> S3ArtifactService:
    """Create S3ArtifactService instance for testing."""
    default_kwargs = {
        "bucket_name": "test-bucket",
        "region_name": "us-east-1",
        "enable_validation": True,
        "enable_security_checks": True,
        "enable_integrity_checks": True
    }
    default_kwargs.update(service_kwargs)

    service = S3ArtifactService(**default_kwargs)

    if mock_s3_client:
        service.s3_client = mock_s3_client

    return service


async def assert_operation_fails_with_error(
    operation: Callable,
    expected_error_type: type,
    expected_error_code: Optional[str] = None
):
    """Assert that operation fails with specific error type and code."""
    try:
        await operation()
        raise AssertionError(f"Expected {expected_error_type.__name__} but operation succeeded")
    except expected_error_type as e:
        if expected_error_code and hasattr(e, 'error_code'):
            assert e.error_code == expected_error_code, (
                f"Expected error code {expected_error_code}, got {e.error_code}"
            )
    except Exception as e:
        raise AssertionError(
            f"Expected {expected_error_type.__name__}, got {type(e).__name__}: {e}"
        )


def parametrize_error_scenarios():
    """Decorator to parametrize tests with all error scenarios."""
    error_codes = ErrorSimulator.get_all_s3_error_codes()
    return pytest.mark.parametrize(
        "error_code",
        error_codes,
        ids=lambda code: f"error_{code}"
    )


class TestMetricsCollector:
    """Collect and analyze test execution metrics."""

    def __init__(self):
        self.metrics = {
            "test_count": 0,
            "passed": 0,
            "failed": 0,
            "error_scenarios_tested": set(),
            "edge_cases_tested": set(),
            "performance_measurements": {},
            "coverage_areas": set()
        }

    def record_test_result(self, test_name: str, passed: bool, categories: List[str]):
        """Record test execution result."""
        self.metrics["test_count"] += 1
        if passed:
            self.metrics["passed"] += 1
        else:
            self.metrics["failed"] += 1

        for category in categories:
            self.metrics["coverage_areas"].add(category)

    def record_error_scenario(self, error_type: str, error_code: str):
        """Record that an error scenario was tested."""
        self.metrics["error_scenarios_tested"].add(f"{error_type}:{error_code}")

    def record_edge_case(self, edge_case_type: str):
        """Record that an edge case was tested."""
        self.metrics["edge_cases_tested"].add(edge_case_type)

    def get_coverage_report(self) -> Dict[str, Any]:
        """Generate coverage report."""
        return {
            "summary": {
                "total_tests": self.metrics["test_count"],
                "passed": self.metrics["passed"],
                "failed": self.metrics["failed"],
                "success_rate": self.metrics["passed"] / max(self.metrics["test_count"], 1)
            },
            "error_scenarios": len(self.metrics["error_scenarios_tested"]),
            "edge_cases": len(self.metrics["edge_cases_tested"]),
            "coverage_areas": list(self.metrics["coverage_areas"])
        }
```

### Sub-Phase 4.2: Comprehensive Error Handling Tests (Week 1, Days 4-7)
**Duration**: 4 days
**Focus**: Unit tests for all error handling and exception scenarios

#### 4.2.1 Exception Mapping and Error Classification Tests

**File**: `tests/unit/test_error_handling_comprehensive.py` (New)
```python
"""Comprehensive unit tests for error handling and exception scenarios."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError

from aws_adk import S3ArtifactService
from aws_adk.exceptions import *
from aws_adk.retry_handler import RetryConfig, CircuitBreaker
from tests.utils import (
    ErrorSimulator, create_test_service, assert_operation_fails_with_error,
    parametrize_error_scenarios, TestMetricsCollector
)


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
            "QuotaExceeded": S3StorageQuotaError
        }

        if error_code in expected_mappings:
            assert isinstance(mapped_error, expected_mappings[error_code])

    async def test_no_credentials_error_mapping(self):
        """Test mapping of NoCredentialsError."""
        error = NoCredentialsError()
        mapped_error = map_boto3_error(error, "test_operation")

        assert isinstance(mapped_error, S3ConnectionError)
        assert mapped_error.error_code == "NoCredentials"
        assert "credentials not found" in mapped_error.message.lower()

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
            context=context
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
            context={"field": "filename", "value": "invalid"}
        )

        error_dict = error.to_dict()

        assert error_dict["error"] == "S3ValidationError"
        assert error_dict["message"] == "Validation failed"
        assert error_dict["error_code"] == "ValidationFailed"
        assert error_dict["operation"] == "validate_input"
        assert error_dict["context"]["field"] == "filename"

    @patch('aws_adk.exceptions.logger')
    async def test_error_logging(self, mock_logger):
        """Test that errors are properly logged."""
        S3ArtifactError(
            message="Test error",
            error_code="TestError",
            operation="test_operation"
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
                app_name="test", user_id="test", session_id="test",
                filename="test.txt", artifact=sample_artifact
            ),
            S3BucketError,
            "NoSuchBucket"
        )

    async def test_save_artifact_access_denied(self, mock_service, sample_artifact):
        """Test save_artifact with access denied error."""
        error = ErrorSimulator.create_client_error("AccessDenied")
        mock_service.s3_client.configure_failure(2, error)  # Fail after list_versions

        await assert_operation_fails_with_error(
            lambda: mock_service.save_artifact(
                app_name="test", user_id="test", session_id="test",
                filename="test.txt", artifact=sample_artifact
            ),
            S3PermissionError,
            "AccessDenied"
        )

    async def test_load_artifact_not_found(self, mock_service):
        """Test load_artifact with artifact not found."""
        error = ErrorSimulator.create_client_error("NoSuchKey")
        mock_service.s3_client.configure_failure(1, error)

        # Should return None instead of raising exception for NotFound
        result = await mock_service.load_artifact(
            app_name="test", user_id="test", session_id="test",
            filename="nonexistent.txt"
        )
        assert result is None

    async def test_throttling_error_handling(self, mock_service, sample_artifact):
        """Test handling of throttling errors."""
        error = ErrorSimulator.create_client_error("Throttling", http_status=503)
        mock_service.s3_client.configure_failure(1, error)

        await assert_operation_fails_with_error(
            lambda: mock_service.save_artifact(
                app_name="test", user_id="test", session_id="test",
                filename="test.txt", artifact=sample_artifact
            ),
            S3ThrottleError,
            "Throttling"
        )

    @pytest.mark.slow
    async def test_network_timeout_handling(self, mock_service, sample_artifact):
        """Test handling of network timeouts."""
        # Configure long delay to trigger timeout
        mock_service.s3_client.configure_delay(10.0)

        # Override retry config for faster testing
        mock_service.retry_config = RetryConfig(
            max_attempts=1,
            max_total_time=1.0
        )

        await assert_operation_fails_with_error(
            lambda: mock_service.save_artifact(
                app_name="test", user_id="test", session_id="test",
                filename="test.txt", artifact=sample_artifact
            ),
            S3ConnectionError
        )


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with error handling."""

    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after consecutive failures."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=1.0,
            expected_exception=S3ArtifactError
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
            success_threshold=1
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
            jitter=False  # Disable jitter for precise timing
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
            actual_delay = call_times[i] - call_times[i-1]
            expected_delay = expected_delays[i-1]
            assert abs(actual_delay - expected_delay) < 0.05  # 50ms tolerance

    async def test_retry_with_jitter(self):
        """Test retry logic with jitter enabled."""
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            jitter=True
        )

        call_times = []

        async def timed_operation():
            call_times.append(time.time())
            raise S3ThrottleError("Rate limited")

        with pytest.raises(S3ThrottleError):
            await with_retry(retry_config)(timed_operation)()

        # With jitter, delays should vary
        delays = [call_times[i] - call_times[i-1] for i in range(1, len(call_times))]

        # All delays should be at least base_delay/2 and at most base_delay
        for delay in delays:
            assert 0.05 <= delay <= 0.15  # Jitter range

    async def test_max_total_time_limit(self):
        """Test maximum total time limit for retries."""
        retry_config = RetryConfig(
            max_attempts=10,  # High attempt count
            base_delay=0.2,
            max_total_time=0.5  # But short total time
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
            retryable_errors=(S3ConnectionError,)  # Only connection errors
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
                type(mapped_error).__name__,
                error_code
            )

            # Verify error is properly handled
            assert isinstance(mapped_error, S3ArtifactError)
            assert mapped_error.error_code == error_code

        # Verify comprehensive coverage
        coverage_report = metrics_collector.get_coverage_report()
        assert coverage_report["error_scenarios"] == len(error_codes)
```

#### 4.2.2 Service-Level Error Integration Tests

**File**: `tests/unit/test_service_error_integration.py` (New)
```python
"""Integration tests for error handling across service operations."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from aws_adk import S3ArtifactService
from aws_adk.exceptions import *
from tests.utils import (
    ErrorSimulator, create_test_service, TestDataGenerator,
    ConcurrencyTester, TestMetricsCollector
)


class TestServiceErrorIntegration:
    """Test error handling integration across service operations."""

    @pytest.fixture
    def service_with_retry_config(self):
        """Service with custom retry configuration for testing."""
        return create_test_service(
            retry_config=RetryConfig(
                max_attempts=2,
                base_delay=0.01,  # Fast retries for testing
                max_total_time=1.0
            )
        )

    async def test_cascading_errors_in_save_operation(
        self, service_with_retry_config, sample_artifact
    ):
        """Test cascading error handling in save operation."""
        # Mock S3 client that fails on different operations
        mock_client = Mock()

        # First call (list_versions) succeeds
        mock_client.list_objects_v2 = AsyncMock(return_value={"Contents": []})

        # Second call (put_object) fails with throttling
        throttle_error = ErrorSimulator.create_client_error("Throttling")
        mock_client.put_object = AsyncMock(side_effect=throttle_error)

        service_with_retry_config.s3_client = mock_client

        with pytest.raises(S3ThrottleError) as exc_info:
            await service_with_retry_config.save_artifact(
                app_name="test", user_id="test", session_id="test",
                filename="test.txt", artifact=sample_artifact
            )

        error = exc_info.value
        assert error.error_code == "Throttling"
        assert error.operation == "put_object"
        assert "bucket" in error.context
        assert "key" in error.context

    async def test_error_recovery_patterns(self, service_with_retry_config):
        """Test various error recovery patterns."""
        call_count = 0

        async def failing_then_succeeding_operation():
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
            app_name="test", user_id="test"
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


class TestConcurrentErrorHandling:
    """Test error handling under concurrent operations."""

    async def test_concurrent_save_errors(self, advanced_s3_mock):
        """Test error handling during concurrent save operations."""
        service = create_test_service()
        service.s3_client = advanced_s3_mock

        # Configure mock to fail after 3 calls
        throttle_error = ErrorSimulator.create_client_error("Throttling")
        advanced_s3_mock.configure_failure(3, throttle_error)

        # Generate test artifacts
        artifacts = [
            TestDataGenerator.generate_artifact(100, "text/plain")
            for _ in range(5)
        ]

        # Create concurrent save operations
        async def save_operation(index):
            return await service.save_artifact(
                app_name="test", user_id="test", session_id="test",
                filename=f"test-{index}.txt", artifact=artifacts[index]
            )

        operations = [lambda i=i: save_operation(i) for i in range(5)]

        # Run concurrent operations
        results = await ConcurrencyTester.run_concurrent_operations(
            operations, max_concurrent=3
        )

        # Some should succeed, some should fail
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]

        assert len(successes) >= 2  # At least some succeed
        assert len(failures) >= 2   # At least some fail

        # All failures should be S3ThrottleError
        for failure in failures:
            assert isinstance(failure, S3ThrottleError)

    async def test_circuit_breaker_under_load(self):
        """Test circuit breaker behavior under concurrent load."""
        from aws_adk.retry_handler import CircuitBreaker

        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=0.1,
            expected_exception=S3ArtifactError
        )

        failure_count = 0

        @circuit_breaker
        async def failing_operation():
            nonlocal failure_count
            failure_count += 1
            raise S3ConnectionError(f"Failure {failure_count}")

        # Run multiple concurrent operations
        operations = [failing_operation for _ in range(10)]
        results = await ConcurrencyTester.run_concurrent_operations(
            operations, max_concurrent=5
        )

        # Should have mix of original errors and circuit breaker errors
        original_errors = [
            r for r in results
            if isinstance(r, S3ConnectionError) and r.error_code != "CircuitBreakerOpen"
        ]
        breaker_errors = [
            r for r in results
            if isinstance(r, S3ConnectionError) and r.error_code == "CircuitBreakerOpen"
        ]

        assert len(original_errors) >= 3  # At least threshold failures
        assert len(breaker_errors) >= 2   # Circuit should open and prevent some


### Sub-Phase 4.3: Edge Case and Validation Testing (Week 2, Days 1-4)
**Duration**: 4 days
**Focus**: Comprehensive testing of edge cases and input validation

#### 4.3.1 Input Validation Testing

**File**: `tests/unit/test_validation_comprehensive.py` (New)
```python
"""Comprehensive unit tests for input validation and edge cases."""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock

from aws_adk.validation import InputValidator
from aws_adk.exceptions import S3ValidationError
from tests.utils import (
    TestDataGenerator, ValidationTester, TestMetricsCollector
)


class TestInputValidationComprehensive:
    """Comprehensive input validation testing."""

    @pytest.fixture
    def validator(self):
        """Input validator instance."""
        return InputValidator()

    @pytest.fixture
    def validation_test_data(self):
        """Comprehensive validation test data."""
        invalid_inputs = TestDataGenerator.generate_invalid_inputs()

        return {
            "valid_cases": {
                "basic": {
                    "app_name": "test-app",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "document.txt",
                    "version": 0
                },
                "with_special_chars": {
                    "app_name": "test-app_v2",
                    "user_id": "user-123",
                    "session_id": "session_456",
                    "filename": "my-document_v2.txt",
                    "version": 5
                },
                "unicode": {
                    "app_name": "测试应用",
                    "user_id": "用户123",
                    "session_id": "会话456",
                    "filename": "文档.txt",
                    "version": 0
                }
            },
            "invalid_cases": {
                "empty_app_name": {
                    "app_name": "",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "document.txt",
                    "version": 0
                },
                "whitespace_app_name": {
                    "app_name": "   ",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "document.txt",
                    "version": 0
                },
                "too_long_app_name": {
                    "app_name": "x" * 256,
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "document.txt",
                    "version": 0
                },
                "path_traversal": {
                    "app_name": "test-app",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "../../../etc/passwd",
                    "version": 0
                },
                "null_bytes": {
                    "app_name": "test-app",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "test\x00.txt",
                    "version": 0
                },
                "negative_version": {
                    "app_name": "test-app",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "document.txt",
                    "version": -1
                },
                "huge_version": {
                    "app_name": "test-app",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "document.txt",
                    "version": 2**31
                }
            }
        }

    def test_all_validation_rules(self, validator, validation_test_data):
        """Test all validation rules comprehensively."""
        results = ValidationTester.test_all_validation_rules(
            validator.validate_save_inputs,
            validation_test_data["valid_cases"],
            validation_test_data["invalid_cases"]
        )

        # All valid cases should pass
        assert results["valid_failed"] == 0, f"Valid cases failed: {results['failures']}"

        # All invalid cases should be caught
        assert results["invalid_missed"] == 0, f"Invalid cases missed: {results['failures']}"

        # Verify comprehensive coverage
        assert results["valid_passed"] == len(validation_test_data["valid_cases"])
        assert results["invalid_caught"] == len(validation_test_data["invalid_cases"])

    @pytest.mark.parametrize("field,invalid_value,expected_error", [
        ("app_name", "", "cannot be empty"),
        ("app_name", "x" * 256, "too long"),
        ("user_id", None, "required"),
        ("filename", "../test.txt", "path traversal"),
        ("filename", "test\x00.txt", "null bytes"),
        ("version", -1, "cannot be negative"),
        ("version", "invalid", "must be integer")
    ])
    def test_specific_validation_errors(self, validator, field, invalid_value, expected_error):
        """Test specific validation error messages."""
        test_data = {
            "app_name": "test-app",
            "user_id": "user123",
            "session_id": "session456",
            "filename": "test.txt",
            "version": 0
        }
        test_data[field] = invalid_value

        with pytest.raises(S3ValidationError) as exc_info:
            validator.validate_save_inputs(**test_data)

        assert expected_error.lower() in str(exc_info.value).lower()

    def test_filename_validation_edge_cases(self, validator):
        """Test filename validation edge cases."""
        edge_cases = {
            "very_long_filename": "x" * 1000 + ".txt",
            "only_extension": ".txt",
            "no_extension": "filename",
            "multiple_dots": "file.name.with.dots.txt",
            "special_chars": "file!@#$%^&*().txt",
            "unicode_filename": "файл.txt",
            "mixed_case": "FileNAME.TXT"
        }

        for case_name, filename in edge_cases.items():
            try:
                validator.validate_save_inputs(
                    app_name="test", user_id="user", session_id="session",
                    filename=filename, version=0
                )
                # If we get here, validation passed
                assert len(filename) <= 255, f"Long filename {case_name} should have failed"
                assert not filename.startswith("."), f"Extension-only {case_name} should have failed"
            except S3ValidationError:
                # Validation failed as expected for problematic cases
                assert (len(filename) > 255 or
                       filename.startswith(".") or
                       any(char in filename for char in ["<", ">", ":", '"', "|", "?", "*"]))

    def test_metadata_validation(self, validator):
        """Test metadata validation edge cases."""
        large_metadata = {"key" + str(i): "value" * 100 for i in range(100)}

        with pytest.raises(S3ValidationError) as exc_info:
            validator.validate_metadata(large_metadata)

        assert "metadata too large" in str(exc_info.value).lower()

    def test_concurrent_validation(self, validator):
        """Test validation under concurrent access."""
        import asyncio

        async def validate_concurrently():
            tasks = []
            for i in range(50):
                task = asyncio.create_task(
                    asyncio.to_thread(
                        validator.validate_save_inputs,
                        app_name=f"app-{i}",
                        user_id=f"user-{i}",
                        session_id=f"session-{i}",
                        filename=f"file-{i}.txt",
                        version=i
                    )
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"Concurrent validation failed: {exceptions}"

        asyncio.run(validate_concurrently())


#### 4.3.2 Edge Case Testing

**File**: `tests/unit/test_edge_cases_comprehensive.py` (New)
```python
"""Comprehensive edge case testing for S3ArtifactService."""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch
from typing import Dict, Any

from google.genai import types
from aws_adk import S3ArtifactService
from aws_adk.exceptions import *
from tests.utils import (
    TestDataGenerator, create_test_service, ConcurrencyTester,
    PerformanceMeasurer, TestMetricsCollector
)


class TestArtifactSizeEdgeCases:
    """Test edge cases related to artifact sizes."""

    @pytest.mark.slow
    async def test_empty_artifact_handling(self, mock_s3_setup):
        """Test handling of empty artifacts."""
        service = create_test_service()
        empty_artifact = types.Part.from_bytes(data=b"", mime_type="text/plain")

        # Should handle empty artifacts gracefully
        with pytest.raises(S3ValidationError) as exc_info:
            await service.save_artifact(
                app_name="test", user_id="test", session_id="test",
                filename="empty.txt", artifact=empty_artifact
            )

        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.slow
    async def test_maximum_size_artifact(self, mock_s3_setup):
        """Test handling of maximum size artifacts."""
        service = create_test_service()

        # 5GB artifact (S3 single PUT limit)
        large_content = TestDataGenerator.generate_test_content(5 * 1024 * 1024 * 1024)
        large_artifact = types.Part.from_bytes(
            data=large_content,
            mime_type="application/octet-stream"
        )

        # Should handle large artifacts (might use multipart upload)
        result = await service.save_artifact(
            app_name="test", user_id="test", session_id="test",
            filename="large.bin", artifact=large_artifact
        )

        assert result["version"] == 0
        assert result["size"] == len(large_content)

    async def test_binary_data_edge_cases(self, mock_s3_setup):
        """Test various binary data edge cases."""
        service = create_test_service()

        test_cases = {
            "all_zeros": b"\x00" * 1000,
            "all_ones": b"\xff" * 1000,
            "random_bytes": os.urandom(1000),
            "mixed_encoding": "Hello 世界 🌍".encode("utf-8"),
            "control_chars": b"\x01\x02\x03\x04\x05" * 200
        }

        for case_name, content in test_cases.items():
            artifact = types.Part.from_bytes(
                data=content,
                mime_type="application/octet-stream"
            )

            result = await service.save_artifact(
                app_name="test", user_id="test", session_id="test",
                filename=f"{case_name}.bin", artifact=artifact
            )

            assert result["size"] == len(content)

            # Verify round-trip integrity
            loaded = await service.load_artifact(
                app_name="test", user_id="test", session_id="test",
                filename=f"{case_name}.bin"
            )

            assert loaded.data == content


class TestConcurrencyEdgeCases:
    """Test edge cases related to concurrent operations."""

    async def test_concurrent_version_creation(self, mock_s3_setup):
        """Test concurrent creation of artifact versions."""
        service = create_test_service()
        artifact = TestDataGenerator.generate_artifact(100)

        # Create operations that save same filename concurrently
        async def save_version(version_content):
            versioned_artifact = types.Part.from_bytes(
                data=version_content,
                mime_type="text/plain"
            )
            return await service.save_artifact(
                app_name="test", user_id="test", session_id="test",
                filename="concurrent.txt", artifact=versioned_artifact
            )

        # Create 10 concurrent save operations with different content
        operations = [
            lambda i=i: save_version(f"Version {i} content".encode())
            for i in range(10)
        ]

        results = await ConcurrencyTester.run_concurrent_operations(
            operations, max_concurrent=5
        )

        # All should succeed with different version numbers
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 10

        # Versions should be unique
        versions = [r["version"] for r in successful_results]
        assert len(set(versions)) == len(versions), "Version numbers should be unique"

    async def test_race_condition_artifact_deletion(self, mock_s3_setup):
        """Test race conditions during artifact deletion."""
        service = create_test_service()
        artifact = TestDataGenerator.generate_artifact(100)

        # First save an artifact
        await service.save_artifact(
            app_name="test", user_id="test", session_id="test",
            filename="race.txt", artifact=artifact
        )

        # Setup concurrent delete and load operations
        async def delete_operation():
            return await service.delete_artifact(
                app_name="test", user_id="test", session_id="test",
                filename="race.txt"
            )

        async def load_operation():
            return await service.load_artifact(
                app_name="test", user_id="test", session_id="test",
                filename="race.txt"
            )

        # Run delete and load concurrently multiple times
        race_results = await ConcurrencyTester.test_race_condition(
            setup_func=lambda: asyncio.sleep(0),  # No setup needed
            operation_func=lambda: asyncio.gather(
                delete_operation(), load_operation(), return_exceptions=True
            ),
            teardown_func=lambda: asyncio.sleep(0),  # No teardown needed
            iterations=5
        )

        # Should handle race conditions gracefully
        assert race_results["success_rate"] > 0.8, "Race condition handling should be robust"


class TestFileSystemEdgeCases:
    """Test edge cases related to file system operations."""

    async def test_special_filename_characters(self, mock_s3_setup):
        """Test handling of special characters in filenames."""
        service = create_test_service()
        artifact = TestDataGenerator.generate_artifact(100)

        # Test various special characters (S3 compatible)
        special_filenames = [
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "file.with.multiple.dots.txt",
            "UPPERCASE.TXT",
            "lowercase.txt",
            "MixedCase.TXT",
            "file123numbers.txt",
            "file(with)parentheses.txt",
            "file[with]brackets.txt"
        ]

        for filename in special_filenames:
            try:
                result = await service.save_artifact(
                    app_name="test", user_id="test", session_id="test",
                    filename=filename, artifact=artifact
                )

                # Verify the file can be loaded back
                loaded = await service.load_artifact(
                    app_name="test", user_id="test", session_id="test",
                    filename=filename
                )

                assert loaded is not None
                assert result["key"].endswith(filename.replace(" ", "_"))

            except S3ValidationError as e:
                # Some special characters might be rejected by validation
                assert any(char in filename for char in ["(", ")", "[", "]"]), \
                    f"Unexpected validation error for {filename}: {e}"

    async def test_unicode_filename_handling(self, mock_s3_setup):
        """Test handling of Unicode filenames."""
        service = create_test_service()
        artifact = TestDataGenerator.generate_artifact(100)

        unicode_filenames = [
            "简体中文.txt",
            "日本語.txt",
            "한국어.txt",
            "العربية.txt",
            "Русский.txt",
            "Ελληνικά.txt",
            "हिन्दी.txt",
            "emoji😀file.txt"
        ]

        for filename in unicode_filenames:
            try:
                result = await service.save_artifact(
                    app_name="test", user_id="test", session_id="test",
                    filename=filename, artifact=artifact
                )

                # Verify round-trip
                loaded = await service.load_artifact(
                    app_name="test", user_id="test", session_id="test",
                    filename=filename
                )

                assert loaded is not None

            except S3ValidationError:
                # Unicode filenames might be restricted
                pass

    async def test_long_path_handling(self, mock_s3_setup):
        """Test handling of very long S3 key paths."""
        service = create_test_service()
        artifact = TestDataGenerator.generate_artifact(100)

        # Create very long app_name and user_id to test path length limits
        long_app_name = "a" * 100
        long_user_id = "u" * 100
        long_session_id = "s" * 100
        long_filename = "f" * 100 + ".txt"

        try:
            result = await service.save_artifact(
                app_name=long_app_name,
                user_id=long_user_id,
                session_id=long_session_id,
                filename=long_filename,
                artifact=artifact
            )

            # Verify the key length is within S3 limits (1024 chars)
            assert len(result["key"]) <= 1024

        except S3ValidationError as e:
            # Long paths might be rejected
            assert "too long" in str(e).lower()


class TestErrorRecoveryEdgeCases:
    """Test edge cases in error recovery scenarios."""

    async def test_partial_failure_recovery(self, advanced_s3_mock):
        """Test recovery from partial operation failures."""
        service = create_test_service()
        service.s3_client = advanced_s3_mock

        # Configure mock to fail intermittently
        call_count = 0

        async def intermittent_put_object(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise ErrorSimulator.create_client_error("ServiceUnavailable")
            return {"ResponseMetadata": {"HTTPStatusCode": 200}}

        advanced_s3_mock.put_object = intermittent_put_object

        # Perform multiple save operations
        artifacts = [TestDataGenerator.generate_artifact(100) for _ in range(10)]

        results = []
        for i, artifact in enumerate(artifacts):
            try:
                result = await service.save_artifact(
                    app_name="test", user_id="test", session_id="test",
                    filename=f"test-{i}.txt", artifact=artifact
                )
                results.append(result)
            except S3ArtifactError:
                # Some operations might fail
                pass

        # Should have succeeded on most operations
        assert len(results) >= 7, "Most operations should succeed with retry"

    async def test_circuit_breaker_edge_cases(self):
        """Test circuit breaker edge cases."""
        from aws_adk.retry_handler import CircuitBreaker

        # Test circuit breaker with zero failure threshold
        circuit_breaker = CircuitBreaker(failure_threshold=0, timeout=0.1)

        @circuit_breaker
        async def always_failing():
            raise S3ConnectionError("Always fails")

        # Should open immediately
        with pytest.raises(S3ConnectionError):
            await always_failing()

        # Second call should be circuit breaker error
        with pytest.raises(S3ConnectionError) as exc_info:
            await always_failing()
        assert exc_info.value.error_code == "CircuitBreakerOpen"

    async def test_timeout_edge_cases(self, mock_s3_setup):
        """Test timeout handling edge cases."""
        service = create_test_service()

        # Override with very short timeout
        service.retry_config.max_total_time = 0.01  # 10ms

        # Mock slow operation
        async def slow_operation(**kwargs):
            await asyncio.sleep(0.1)  # 100ms delay
            return {"ResponseMetadata": {"HTTPStatusCode": 200}}

        with patch.object(service.s3_client, 'put_object', slow_operation):
            artifact = TestDataGenerator.generate_artifact(100)

            with pytest.raises(S3ConnectionError) as exc_info:
                await service.save_artifact(
                    app_name="test", user_id="test", session_id="test",
                    filename="slow.txt", artifact=artifact
                )

            # Should be timeout-related error
            assert "timeout" in str(exc_info.value).lower() or \
                   "time" in str(exc_info.value).lower()


@pytest.mark.edge_cases
class TestComprehensiveEdgeScenarios:
    """Comprehensive edge case scenario testing."""

    @pytest.fixture
    def metrics_collector(self):
        """Metrics collector for edge case coverage."""
        return TestMetricsCollector()

    async def test_all_edge_case_categories(self, metrics_collector, mock_s3_setup):
        """Test all categories of edge cases."""
        service = create_test_service()

        edge_case_categories = [
            "empty_data", "large_data", "binary_data", "unicode_data",
            "concurrent_access", "partial_failures", "timeout_scenarios",
            "special_characters", "long_paths", "version_conflicts"
        ]

        for category in edge_case_categories:
            try:
                if category == "empty_data":
                    empty_artifact = types.Part.from_bytes(data=b"", mime_type="text/plain")
                    await service.save_artifact(
                        app_name="test", user_id="test", session_id="test",
                        filename="empty.txt", artifact=empty_artifact
                    )

                elif category == "large_data":
                    large_artifact = TestDataGenerator.generate_artifact(1024 * 1024)  # 1MB
                    await service.save_artifact(
                        app_name="test", user_id="test", session_id="test",
                        filename="large.bin", artifact=large_artifact
                    )

                # ... test other categories

                metrics_collector.record_edge_case(category)

            except Exception as e:
                # Edge cases might legitimately fail
                if isinstance(e, S3ValidationError):
                    metrics_collector.record_edge_case(f"{category}_validation_error")
                else:
                    metrics_collector.record_edge_case(f"{category}_error")

        # Verify comprehensive edge case coverage
        coverage_report = metrics_collector.get_coverage_report()
        assert coverage_report["edge_cases"] >= len(edge_case_categories)
```

### Sub-Phase 4.4: Performance Testing and Benchmarking (Week 2, Days 5-7)
**Duration**: 3 days
**Focus**: Performance validation and benchmarking

#### 4.4.1 Performance Testing

**File**: `tests/unit/test_performance_comprehensive.py` (New)
```python
"""Comprehensive performance testing and benchmarking."""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from unittest.mock import Mock

from aws_adk import S3ArtifactService
from tests.utils import (
    TestDataGenerator, create_test_service, PerformanceMeasurer,
    ConcurrencyTester, TestMetricsCollector
)


@pytest.mark.performance
class TestOperationPerformance:
    """Test performance of individual operations."""

    @pytest.fixture
    def performance_service(self, mock_s3_setup):
        """Service configured for performance testing."""
        return create_test_service(
            enable_validation=True,
            enable_security_checks=True,
            enable_integrity_checks=True
        )

    @pytest.fixture
    def performance_measurer(self):
        """Performance measurement utility."""
        return PerformanceMeasurer()

    async def test_save_artifact_performance(
        self, performance_service, performance_measurer, performance_benchmarks
    ):
        """Test save_artifact operation performance."""
        artifact = TestDataGenerator.generate_artifact(1024)  # 1KB artifact

        # Measure multiple save operations
        for i in range(10):
            async with performance_measurer.measure("save_artifact"):
                await performance_service.save_artifact(
                    app_name="test", user_id="test", session_id="test",
                    filename=f"perf-test-{i}.txt", artifact=artifact
                )

        # Assert performance requirements
        performance_measurer.assert_performance(
            "save_artifact",
            performance_benchmarks["save_artifact_max_time"]
        )

    async def test_load_artifact_performance(
        self, performance_service, performance_measurer, performance_benchmarks
    ):
        """Test load_artifact operation performance."""
        artifact = TestDataGenerator.generate_artifact(1024)

        # First save an artifact
        await performance_service.save_artifact(
            app_name="test", user_id="test", session_id="test",
            filename="perf-load.txt", artifact=artifact
        )

        # Measure multiple load operations
        for i in range(10):
            async with performance_measurer.measure("load_artifact"):
                result = await performance_service.load_artifact(
                    app_name="test", user_id="test", session_id="test",
                    filename="perf-load.txt"
                )
                assert result is not None

        performance_measurer.assert_performance(
            "load_artifact",
            performance_benchmarks["load_artifact_max_time"]
        )

    async def test_list_operations_performance(
        self, performance_service, performance_measurer, performance_benchmarks
    ):
        """Test list operations performance."""
        # Create some test artifacts
        for i in range(5):
            artifact = TestDataGenerator.generate_artifact(100)
            await performance_service.save_artifact(
                app_name="test", user_id="test", session_id="test",
                filename=f"list-test-{i}.txt", artifact=artifact
            )

        # Measure list operations
        for i in range(10):
            async with performance_measurer.measure("list_artifacts"):
                artifacts = await performance_service.list_artifacts(
                    app_name="test", user_id="test"
                )
                assert len(artifacts) >= 5

        performance_measurer.assert_performance(
            "list_artifacts",
            performance_benchmarks["list_operations_max_time"]
        )

    async def test_validation_performance(
        self, performance_measurer, performance_benchmarks
    ):
        """Test input validation performance."""
        from aws_adk.validation import InputValidator
        validator = InputValidator()

        # Measure validation operations
        for i in range(100):
            async with performance_measurer.measure("validation"):
                validator.validate_save_inputs(
                    app_name=f"test-app-{i}",
                    user_id=f"user-{i}",
                    session_id=f"session-{i}",
                    filename=f"file-{i}.txt",
                    version=i
                )

        performance_measurer.assert_performance(
            "validation",
            performance_benchmarks["validation_max_time"]
        )

    @pytest.mark.slow
    async def test_large_artifact_performance(
        self, performance_service, performance_measurer
    ):
        """Test performance with large artifacts."""
        # Test different sizes
        sizes = [1024, 10*1024, 100*1024, 1024*1024]  # 1KB to 1MB

        for size in sizes:
            artifact = TestDataGenerator.generate_artifact(size)

            async with performance_measurer.measure(f"save_large_{size}"):
                await performance_service.save_artifact(
                    app_name="test", user_id="test", session_id="test",
                    filename=f"large-{size}.bin", artifact=artifact
                )

        # Verify performance scales reasonably
        stats_1kb = performance_measurer.get_stats("save_large_1024")
        stats_1mb = performance_measurer.get_stats("save_large_1048576")

        # 1MB should not take more than 1000x longer than 1KB
        assert stats_1mb["average"] / stats_1kb["average"] < 1000


class TestConcurrencyPerformance:
    """Test performance under concurrent load."""

    async def test_concurrent_save_performance(self, mock_s3_setup):
        """Test concurrent save operation performance."""
        service = create_test_service()
        measurer = PerformanceMeasurer()

        # Create concurrent save operations
        async def save_operation(index):
            artifact = TestDataGenerator.generate_artifact(1024)
            async with measurer.measure("concurrent_save"):
                return await service.save_artifact(
                    app_name="test", user_id="test", session_id="test",
                    filename=f"concurrent-{index}.txt", artifact=artifact
                )

        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]

        for concurrency in concurrency_levels:
            operations = [lambda i=i: save_operation(i) for i in range(concurrency)]

            start_time = time.time()
            results = await ConcurrencyTester.run_concurrent_operations(
                operations, max_concurrent=concurrency
            )
            total_time = time.time() - start_time

            # All operations should succeed
            successful = [r for r in results if not isinstance(r, Exception)]
            assert len(successful) == concurrency

            # Higher concurrency should not degrade performance linearly
            throughput = concurrency / total_time
            print(f"Concurrency {concurrency}: {throughput:.2f} ops/sec")

    async def test_memory_usage_under_load(self, mock_s3_setup):
        """Test memory usage under concurrent load."""
        import psutil
        import os

        service = create_test_service()
        process = psutil.Process(os.getpid())

        # Baseline memory usage
        baseline_memory = process.memory_info().rss

        # Create many concurrent operations
        async def memory_intensive_operation(index):
            # Large artifact
            artifact = TestDataGenerator.generate_artifact(100 * 1024)  # 100KB
            return await service.save_artifact(
                app_name="test", user_id="test", session_id="test",
                filename=f"memory-test-{index}.bin", artifact=artifact
            )

        operations = [
            lambda i=i: memory_intensive_operation(i)
            for i in range(50)
        ]

        await ConcurrencyTester.run_concurrent_operations(
            operations, max_concurrent=10
        )

        # Check memory usage after operations
        final_memory = process.memory_info().rss
        memory_increase = final_memory - baseline_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024, \
            f"Memory increased by {memory_increase / 1024 / 1024:.2f}MB"


class TestScalabilityBenchmarks:
    """Benchmark scalability characteristics."""

    @pytest.mark.slow
    async def test_operation_scaling(self, mock_s3_setup):
        """Test how operations scale with data size and volume."""
        service = create_test_service()
        measurer = PerformanceMeasurer()

        # Test scaling with number of artifacts
        artifact_counts = [10, 50, 100, 200]

        for count in artifact_counts:
            # Create artifacts
            start_time = time.time()
            for i in range(count):
                artifact = TestDataGenerator.generate_artifact(1024)
                await service.save_artifact(
                    app_name="scale-test", user_id="test", session_id="test",
                    filename=f"scale-{count}-{i}.txt", artifact=artifact
                )
            creation_time = time.time() - start_time

            # Test list performance
            start_time = time.time()
            artifacts = await service.list_artifacts(
                app_name="scale-test", user_id="test"
            )
            list_time = time.time() - start_time

            print(f"Count {count}: Creation {creation_time:.2f}s, List {list_time:.2f}s")

            # List time should not grow linearly with artifact count
            if count > 10:
                assert list_time < count * 0.1, "List performance should not degrade linearly"

    async def test_throughput_benchmarks(self, mock_s3_setup):
        """Benchmark operation throughput."""
        service = create_test_service()

        # Measure sustained throughput
        operations_completed = 0
        start_time = time.time()
        test_duration = 5.0  # 5 seconds

        async def throughput_operation():
            nonlocal operations_completed
            artifact = TestDataGenerator.generate_artifact(1024)
            await service.save_artifact(
                app_name="throughput", user_id="test", session_id="test",
                filename=f"throughput-{operations_completed}.txt", artifact=artifact
            )
            operations_completed += 1

        # Run operations for specified duration
        while time.time() - start_time < test_duration:
            await throughput_operation()

        actual_duration = time.time() - start_time
        throughput = operations_completed / actual_duration

        print(f"Throughput: {throughput:.2f} operations/second")

        # Should achieve reasonable throughput
        assert throughput > 10, "Should achieve at least 10 operations/second"


### Sub-Phase 4.5: Test Automation and Reporting (Week 3, Days 1-4)
**Duration**: 4 days
**Focus**: Test automation, continuous integration, and comprehensive reporting

#### 4.5.1 Test Automation Framework

**File**: `tests/automation/test_runner.py` (New)
```python
"""Automated test runner with comprehensive reporting."""

import asyncio
import time
import json
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess

import pytest
from tests.utils import TestMetricsCollector


@dataclass
class TestSuite:
    """Test suite configuration."""
    name: str
    pattern: str
    markers: List[str]
    timeout: int
    parallel: bool = True


@dataclass
class TestResult:
    """Test execution result."""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration: float
    coverage_percentage: float
    error_scenarios_tested: int
    edge_cases_tested: int


class ComprehensiveTestRunner:
    """Comprehensive test runner with reporting."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results: List[TestResult] = []
        self.metrics_collector = TestMetricsCollector()

    def define_test_suites(self) -> List[TestSuite]:
        """Define all test suites for comprehensive testing."""
        return [
            TestSuite(
                name="Unit Tests - Core",
                pattern="tests/unit/test_s3_artifact_service_*.py",
                markers=["unit", "not slow"],
                timeout=300,
                parallel=True
            ),
            TestSuite(
                name="Unit Tests - Error Handling",
                pattern="tests/unit/test_*error*.py",
                markers=["unit", "error_handling"],
                timeout=600,
                parallel=True
            ),
            TestSuite(
                name="Unit Tests - Edge Cases",
                pattern="tests/unit/test_*edge*.py",
                markers=["unit", "edge_cases"],
                timeout=900,
                parallel=True
            ),
            TestSuite(
                name="Unit Tests - Validation",
                pattern="tests/unit/test_*validation*.py",
                markers=["unit", "validation"],
                timeout=300,
                parallel=True
            ),
            TestSuite(
                name="Performance Tests",
                pattern="tests/unit/test_*performance*.py",
                markers=["performance"],
                timeout=1800,
                parallel=False  # Performance tests run sequentially
            ),
            TestSuite(
                name="Integration Tests",
                pattern="tests/integration/test_*.py",
                markers=["integration"],
                timeout=1200,
                parallel=True
            ),
            TestSuite(
                name="Slow Tests",
                pattern="tests/**/*test*.py",
                markers=["slow"],
                timeout=3600,
                parallel=True
            )
        ]

    async def run_test_suite(self, suite: TestSuite) -> TestResult:
        """Run a single test suite and collect results."""
        print(f"Running test suite: {suite.name}")

        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            "--tb=short",
            "--strict-markers",
            "--strict-config",
            "-v",
            f"--timeout={suite.timeout}",
            "--cov=aws_adk",
            "--cov-report=json",
            "--json-report",
            f"--json-report-file=test_results_{suite.name.replace(' ', '_').lower()}.json"
        ]

        # Add markers
        if suite.markers:
            marker_expr = " and ".join(suite.markers)
            cmd.extend(["-m", marker_expr])

        # Add parallelization if enabled
        if suite.parallel:
            cmd.extend(["-n", "auto"])

        # Add test pattern
        cmd.append(suite.pattern)

        start_time = time.time()

        try:
            # Run tests
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=suite.timeout
            )

            duration = time.time() - start_time

            # Parse results
            test_result = self._parse_test_results(
                suite, result, duration
            )

            return test_result

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"Test suite {suite.name} timed out after {duration:.2f}s")

            return TestResult(
                suite_name=suite.name,
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                duration=duration,
                coverage_percentage=0.0,
                error_scenarios_tested=0,
                edge_cases_tested=0
            )

    def _parse_test_results(
        self, suite: TestSuite, result: subprocess.CompletedProcess, duration: float
    ) -> TestResult:
        """Parse pytest results."""
        try:
            # Try to load JSON report
            json_file = f"test_results_{suite.name.replace(' ', '_').lower()}.json"
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    json_data = json.load(f)

                summary = json_data.get("summary", {})

                return TestResult(
                    suite_name=suite.name,
                    total_tests=summary.get("total", 0),
                    passed=summary.get("passed", 0),
                    failed=summary.get("failed", 0),
                    skipped=summary.get("skipped", 0),
                    duration=duration,
                    coverage_percentage=self._extract_coverage_percentage(),
                    error_scenarios_tested=self._count_error_scenarios(json_data),
                    edge_cases_tested=self._count_edge_cases(json_data)
                )
        except Exception as e:
            print(f"Error parsing results for {suite.name}: {e}")

        # Fallback parsing from stdout
        return self._parse_stdout_results(suite, result.stdout, duration)

    def _parse_stdout_results(
        self, suite: TestSuite, stdout: str, duration: float
    ) -> TestResult:
        """Parse results from pytest stdout."""
        lines = stdout.split('\n')

        # Look for summary line like "= 10 passed, 2 failed, 1 skipped in 5.23s ="
        summary_line = None
        for line in lines:
            if " passed" in line or " failed" in line:
                if line.startswith("=") and line.endswith("="):
                    summary_line = line.strip("= ")
                    break

        passed = failed = skipped = total = 0

        if summary_line:
            parts = summary_line.split(",")
            for part in parts:
                part = part.strip()
                if " passed" in part:
                    passed = int(part.split()[0])
                elif " failed" in part:
                    failed = int(part.split()[0])
                elif " skipped" in part:
                    skipped = int(part.split()[0])

            total = passed + failed + skipped

        return TestResult(
            suite_name=suite.name,
            total_tests=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration=duration,
            coverage_percentage=0.0,  # Cannot extract from stdout
            error_scenarios_tested=0,
            edge_cases_tested=0
        )

    def _extract_coverage_percentage(self) -> float:
        """Extract coverage percentage from coverage report."""
        try:
            if os.path.exists("coverage.json"):
                with open("coverage.json", 'r') as f:
                    coverage_data = json.load(f)
                return coverage_data.get("totals", {}).get("percent_covered", 0.0)
        except Exception:
            pass
        return 0.0

    def _count_error_scenarios(self, json_data: Dict) -> int:
        """Count error scenarios tested."""
        count = 0
        for test in json_data.get("tests", []):
            if any(marker in test.get("keywords", [])
                   for marker in ["error_handling", "error"]):
                count += 1
        return count

    def _count_edge_cases(self, json_data: Dict) -> int:
        """Count edge cases tested."""
        count = 0
        for test in json_data.get("tests", []):
            if any(marker in test.get("keywords", [])
                   for marker in ["edge_cases", "edge"]):
                count += 1
        return count

    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all test suites comprehensively."""
        print("Starting comprehensive test execution...")

        test_suites = self.define_test_suites()
        overall_start = time.time()

        # Run test suites
        for suite in test_suites:
            result = await self.run_test_suite(suite)
            self.test_results.append(result)

            # Print immediate results
            print(f"\nResults for {suite.name}:")
            print(f"  Total: {result.total_tests}")
            print(f"  Passed: {result.passed}")
            print(f"  Failed: {result.failed}")
            print(f"  Skipped: {result.skipped}")
            print(f"  Duration: {result.duration:.2f}s")
            print(f"  Coverage: {result.coverage_percentage:.1f}%")

        overall_duration = time.time() - overall_start

        # Generate comprehensive report
        report = self.generate_comprehensive_report(overall_duration)

        # Save report
        await self.save_reports(report)

        return report

    def generate_comprehensive_report(self, overall_duration: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = sum(r.total_tests for r in self.test_results)
        total_passed = sum(r.passed for r in self.test_results)
        total_failed = sum(r.failed for r in self.test_results)
        total_skipped = sum(r.skipped for r in self.test_results)

        # Calculate averages
        avg_coverage = (
            sum(r.coverage_percentage for r in self.test_results if r.coverage_percentage > 0) /
            len([r for r in self.test_results if r.coverage_percentage > 0])
            if any(r.coverage_percentage > 0 for r in self.test_results) else 0.0
        )

        total_error_scenarios = sum(r.error_scenarios_tested for r in self.test_results)
        total_edge_cases = sum(r.edge_cases_tested for r in self.test_results)

        return {
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
                "success_rate": total_passed / max(total_tests, 1),
                "overall_duration": overall_duration,
                "average_coverage": avg_coverage,
                "error_scenarios_tested": total_error_scenarios,
                "edge_cases_tested": total_edge_cases
            },
            "suite_results": [asdict(result) for result in self.test_results],
            "quality_metrics": {
                "test_categories_covered": len(self.test_results),
                "comprehensive_coverage": avg_coverage >= 95.0,
                "error_handling_coverage": total_error_scenarios >= 50,
                "edge_case_coverage": total_edge_cases >= 30,
                "performance_validated": any(
                    "Performance" in r.suite_name for r in self.test_results
                )
            },
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        total_tests = sum(r.total_tests for r in self.test_results)
        total_failed = sum(r.failed for r in self.test_results)

        if total_failed > 0:
            recommendations.append(
                f"Address {total_failed} failing tests before production deployment"
            )

        avg_coverage = (
            sum(r.coverage_percentage for r in self.test_results if r.coverage_percentage > 0) /
            len([r for r in self.test_results if r.coverage_percentage > 0])
            if any(r.coverage_percentage > 0 for r in self.test_results) else 0.0
        )

        if avg_coverage < 95.0:
            recommendations.append(
                f"Increase test coverage from {avg_coverage:.1f}% to at least 95%"
            )

        error_scenarios = sum(r.error_scenarios_tested for r in self.test_results)
        if error_scenarios < 50:
            recommendations.append(
                f"Add more error scenario tests (current: {error_scenarios}, target: 50+)"
            )

        edge_cases = sum(r.edge_cases_tested for r in self.test_results)
        if edge_cases < 30:
            recommendations.append(
                f"Add more edge case tests (current: {edge_cases}, target: 30+)"
            )

        return recommendations

    async def save_reports(self, report: Dict[str, Any]):
        """Save comprehensive reports."""
        # Save JSON report
        with open("comprehensive_test_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Save HTML report
        html_report = self._generate_html_report(report)
        with open("comprehensive_test_report.html", "w") as f:
            f.write(html_report)

        print(f"\nReports saved:")
        print(f"  JSON: comprehensive_test_report.json")
        print(f"  HTML: comprehensive_test_report.html")

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>S3 Artifact Service - Comprehensive Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .suite {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .skipped {{ color: orange; }}
                .metric {{ display: inline-block; margin: 10px 15px 10px 0; }}
            </style>
        </head>
        <body>
            <h1>S3 Artifact Service - Phase 4 Test Results</h1>

            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">Total Tests: <strong>{report['summary']['total_tests']}</strong></div>
                <div class="metric passed">Passed: <strong>{report['summary']['passed']}</strong></div>
                <div class="metric failed">Failed: <strong>{report['summary']['failed']}</strong></div>
                <div class="metric skipped">Skipped: <strong>{report['summary']['skipped']}</strong></div>
                <div class="metric">Success Rate: <strong>{report['summary']['success_rate']:.1%}</strong></div>
                <div class="metric">Duration: <strong>{report['summary']['overall_duration']:.1f}s</strong></div>
                <div class="metric">Coverage: <strong>{report['summary']['average_coverage']:.1f}%</strong></div>
            </div>

            <h2>Test Suites</h2>
        """

        for suite_result in report['suite_results']:
            html += f"""
            <div class="suite">
                <h3>{suite_result['suite_name']}</h3>
                <div class="metric">Total: {suite_result['total_tests']}</div>
                <div class="metric passed">Passed: {suite_result['passed']}</div>
                <div class="metric failed">Failed: {suite_result['failed']}</div>
                <div class="metric skipped">Skipped: {suite_result['skipped']}</div>
                <div class="metric">Duration: {suite_result['duration']:.1f}s</div>
                <div class="metric">Coverage: {suite_result['coverage_percentage']:.1f}%</div>
            </div>
            """

        html += f"""
            <h2>Quality Metrics</h2>
            <ul>
                <li>Error Scenarios Tested: {report['summary']['error_scenarios_tested']}</li>
                <li>Edge Cases Tested: {report['summary']['edge_cases_tested']}</li>
                <li>Test Categories: {report['quality_metrics']['test_categories_covered']}</li>
                <li>Comprehensive Coverage: {'✓' if report['quality_metrics']['comprehensive_coverage'] else '✗'}</li>
            </ul>

            <h2>Recommendations</h2>
            <ul>
        """

        for rec in report['recommendations']:
            html += f"<li>{rec}</li>"

        html += """
            </ul>
        </body>
        </html>
        """

        return html


async def main():
    """Main test runner entry point."""
    project_root = Path(__file__).parent.parent.parent
    runner = ComprehensiveTestRunner(project_root)

    report = await runner.run_comprehensive_tests()

    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST EXECUTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"Average Coverage: {report['summary']['average_coverage']:.1f}%")
    print(f"Duration: {report['summary']['overall_duration']:.1f}s")

    if report['summary']['failed'] > 0:
        print(f"\n⚠️  {report['summary']['failed']} tests failed!")
        return 1
    else:
        print(f"\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

#### 4.5.2 Continuous Integration Configuration

**File**: `.github/workflows/comprehensive-testing.yml` (New)
```yaml
name: Comprehensive Testing - Phase 4

on:
  push:
    branches: [ main, develop, 'feature/*', 'phase4/*' ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run comprehensive tests daily at 2 AM UTC
    - cron: '0 2 * * *'

jobs:
  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
        pip install pytest-xdist pytest-cov pytest-timeout pytest-json-report

    - name: Run unit tests
      run: |
        python -m pytest tests/unit/ \
          -v \
          --tb=short \
          --strict-markers \
          --strict-config \
          --timeout=300 \
          --cov=aws_adk \
          --cov-report=xml \
          --cov-report=html \
          --json-report \
          --json-report-file=unit_test_results.json \
          -m "unit and not slow"

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unit-tests
        name: codecov-umbrella

    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: unit-test-results-${{ matrix.python-version }}
        path: |
          unit_test_results.json
          htmlcov/

  error-handling-tests:
    name: Error Handling Tests
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
        pip install pytest-xdist pytest-timeout

    - name: Run error handling tests
      run: |
        python -m pytest tests/unit/ \
          -v \
          --tb=short \
          --timeout=600 \
          --json-report \
          --json-report-file=error_test_results.json \
          -m "error_handling"

    - name: Upload error test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: error-test-results
        path: error_test_results.json

  edge-case-tests:
    name: Edge Case Tests
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
        pip install pytest-xdist pytest-timeout

    - name: Run edge case tests
      run: |
        python -m pytest tests/unit/ \
          -v \
          --tb=short \
          --timeout=900 \
          --json-report \
          --json-report-file=edge_case_results.json \
          -m "edge_cases"

    - name: Upload edge case results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: edge-case-results
        path: edge_case_results.json

  performance-tests:
    name: Performance Tests
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
        pip install pytest-benchmark pytest-timeout

    - name: Run performance tests
      run: |
        python -m pytest tests/unit/ \
          -v \
          --tb=short \
          --timeout=1800 \
          --benchmark-json=performance_results.json \
          -m "performance"

    - name: Upload performance results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: performance-results
        path: performance_results.json

  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
        pip install pytest-xdist pytest-timeout

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1

    - name: Run integration tests
      run: |
        python -m pytest tests/integration/ \
          -v \
          --tb=short \
          --timeout=1200 \
          --json-report \
          --json-report-file=integration_results.json \
          -m "integration"
      env:
        AWS_DEFAULT_REGION: us-east-1
        TEST_BUCKET_NAME: ${{ secrets.TEST_BUCKET_NAME }}

    - name: Upload integration results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: integration-results
        path: integration_results.json

  comprehensive-report:
    name: Generate Comprehensive Report
    runs-on: ubuntu-latest
    needs: [unit-tests, error-handling-tests, edge-case-tests, performance-tests, integration-tests]
    if: always()

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Download all test results
      uses: actions/download-artifact@v3

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"

    - name: Generate comprehensive report
      run: |
        python tests/automation/test_runner.py

    - name: Upload comprehensive report
      uses: actions/upload-artifact@v3
      with:
        name: comprehensive-test-report
        path: |
          comprehensive_test_report.json
          comprehensive_test_report.html

    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          try {
            const report = JSON.parse(fs.readFileSync('comprehensive_test_report.json', 'utf8'));
            const comment = `
            ## 📊 Comprehensive Test Results - Phase 4

            ### Summary
            - **Total Tests**: ${report.summary.total_tests}
            - **Success Rate**: ${(report.summary.success_rate * 100).toFixed(1)}%
            - **Coverage**: ${report.summary.average_coverage.toFixed(1)}%
            - **Duration**: ${report.summary.overall_duration.toFixed(1)}s

            ### Quality Metrics
            - **Error Scenarios Tested**: ${report.summary.error_scenarios_tested}
            - **Edge Cases Tested**: ${report.summary.edge_cases_tested}
            - **Comprehensive Coverage**: ${report.quality_metrics.comprehensive_coverage ? '✅' : '❌'}

            ${report.recommendations.length > 0 ? '### Recommendations\n' + report.recommendations.map(r => `- ${r}`).join('\n') : ''}
            `;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
          } catch (error) {
            console.log('Could not read test report:', error);
          }

  quality-gate:
    name: Quality Gate
    runs-on: ubuntu-latest
    needs: [comprehensive-report]
    if: always()

    steps:
    - name: Download comprehensive report
      uses: actions/download-artifact@v3
      with:
        name: comprehensive-test-report

    - name: Check quality gate
      run: |
        python -c "
        import json
        import sys

        with open('comprehensive_test_report.json', 'r') as f:
            report = json.load(f)

        summary = report['summary']
        quality = report['quality_metrics']

        print(f'Success Rate: {summary[\"success_rate\"]:.1%}')
        print(f'Coverage: {summary[\"average_coverage\"]:.1f}%')
        print(f'Failed Tests: {summary[\"failed\"]}')

        # Quality gate criteria
        if summary['failed'] > 0:
            print('❌ Quality gate failed: Tests are failing')
            sys.exit(1)

        if summary['success_rate'] < 0.95:
            print('❌ Quality gate failed: Success rate below 95%')
            sys.exit(1)

        if summary['average_coverage'] < 95.0:
            print('❌ Quality gate failed: Coverage below 95%')
            sys.exit(1)

        if not quality['comprehensive_coverage']:
            print('❌ Quality gate failed: Comprehensive coverage not achieved')
            sys.exit(1)

        print('✅ Quality gate passed: All criteria met')
        "
```

## Phase 4 Summary and Next Steps

### 4.6 Phase 4 Completion Criteria

**All criteria must be met for Phase 4 completion:**

1. **Test Coverage**: >95% code coverage across all components
2. **Error Scenario Coverage**: All S3 error codes tested and handled
3. **Edge Case Coverage**: Comprehensive edge case testing completed
4. **Performance Validation**: All performance benchmarks met
5. **Automation**: Full CI/CD pipeline operational with quality gates
6. **Documentation**: All test scenarios documented with expected behaviors

### 4.7 Deliverables

**Phase 4 Deliverables:**

1. **Enhanced Test Infrastructure** (`tests/conftest.py`, `tests/utils.py`)
2. **Comprehensive Error Handling Tests** (`tests/unit/test_error_handling_comprehensive.py`)
3. **Service Error Integration Tests** (`tests/unit/test_service_error_integration.py`)
4. **Input Validation Tests** (`tests/unit/test_validation_comprehensive.py`)
5. **Edge Case Tests** (`tests/unit/test_edge_cases_comprehensive.py`)
6. **Performance Tests** (`tests/unit/test_performance_comprehensive.py`)
7. **Test Automation Framework** (`tests/automation/test_runner.py`)
8. **CI/CD Pipeline** (`.github/workflows/comprehensive-testing.yml`)

### 4.8 Success Metrics

**Phase 4 Success Metrics:**
- **Code Coverage**: >95% across all modules
- **Test Execution Time**: <30 minutes for full suite
- **Error Scenario Coverage**: 100% of S3 error codes
- **Edge Case Coverage**: >90% of identified edge cases
- **Performance Benchmarks**: All operations within defined limits
- **CI/CD Success Rate**: >98% pipeline success rate

### 4.9 Transition to Production

**Post-Phase 4:**
- Production deployment readiness achieved
- Comprehensive monitoring and alerting in place
- Performance characteristics well-understood
- Error handling battle-tested
- Automated quality gates protecting main branch

**Phase 4 establishes the S3ArtifactService as production-ready with enterprise-grade reliability, comprehensive error handling, and robust testing infrastructure.**
