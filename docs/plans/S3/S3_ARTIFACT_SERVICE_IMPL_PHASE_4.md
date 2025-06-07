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

This covers Sub-Phase 4.1 and the beginning of 4.2. The plan continues with validation testing, edge case testing, performance testing, and test automation. Would you like me to continue with the remaining sub-phases?
