"""Enhanced shared test configuration and fixtures for comprehensive testing."""

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator
from unittest.mock import Mock

import boto3
import pytest
from google.genai import types
from moto import mock_aws

from aws_adk import S3ArtifactService
from aws_adk.exceptions import S3ArtifactError
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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
        if (
            "edge_case" in item.name
            or "large_file" in item.name
            or "concurrent" in item.name
        ):
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
        "--slow", action="store_true", default=False, help="Run slow tests"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests",
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
    with mock_aws():
        # Create mock S3 client and bucket
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-artifacts-bucket")

        yield {
            "client": s3_client,
            "bucket_name": "test-artifacts-bucket",
            "region_name": "us-east-1",
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
        "invalid_mime_type": types.Part.from_bytes(
            data=b"test", mime_type="invalid/mime"
        ),
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
            "version": 0,
        },
        "invalid_app_name": {
            "app_name": "",  # Empty
            "user_id": "test-user",
            "session_id": "test-session",
            "filename": "test-file.txt",
            "version": 0,
        },
        "invalid_filename": {
            "app_name": "test-app",
            "user_id": "test-user",
            "session_id": "test-session",
            "filename": "../../../etc/passwd",  # Path traversal
            "version": 0,
        },
        "invalid_version": {
            "app_name": "test-app",
            "user_id": "test-user",
            "session_id": "test-session",
            "filename": "test-file.txt",
            "version": -1,  # Negative version
        },
    }


@pytest.fixture
def error_simulation_fixtures() -> Dict[str, Any]:
    """Fixtures for simulating various error conditions."""
    return {
        "client_errors": {
            "NoSuchBucket": {
                "Error": {
                    "Code": "NoSuchBucket",
                    "Message": "The specified bucket does not exist",
                },
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "AccessDenied": {
                "Error": {"Code": "AccessDenied", "Message": "Access Denied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "NoSuchKey": {
                "Error": {
                    "Code": "NoSuchKey",
                    "Message": "The specified key does not exist",
                },
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "Throttling": {
                "Error": {"Code": "Throttling", "Message": "Rate exceeded"},
                "ResponseMetadata": {"HTTPStatusCode": 503},
            },
        },
        "network_errors": [
            ConnectionError("Network connection failed"),
            TimeoutError("Request timeout"),
            OSError("Network unreachable"),
        ],
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
                "Metadata": {"version": "0"},
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
        if (
            self.fail_after_calls is not None
            and self.call_count >= self.fail_after_calls
            and self.failure_exception is not None
        ):
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
        "performance_measurements": {},
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
