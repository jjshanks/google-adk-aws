# mypy: ignore-errors
"""Utility functions and helpers for comprehensive testing."""

import asyncio
import logging
import random
import string
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError
from google.genai import types

from aws_adk import S3ArtifactService

logger = logging.getLogger(__name__)


class TestDataGenerator:
    """Generate various test data scenarios."""

    @staticmethod
    def generate_random_string(length: int) -> str:
        """Generate random string of specified length."""
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    @staticmethod
    def generate_test_content(size_bytes: int) -> bytes:
        """Generate test content of specified size."""
        if size_bytes <= 1024:
            # Small content - use readable text
            return (
                f"Test content {TestDataGenerator.generate_random_string(10)}".encode()[
                    :size_bytes
                ]
            )
        else:
            # Large content - use efficient generation
            chunk = b"x" * 1024
            full_chunks = size_bytes // 1024
            remainder = size_bytes % 1024
            return chunk * full_chunks + chunk[:remainder]

    @staticmethod
    def generate_artifact(
        content_size: int = 100, mime_type: str = "text/plain"
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
            "special_chars": "!@#$%^&*()+={}[]|\\:;\"'<>?,./`~",
        }


class ErrorSimulator:
    """Simulate various error conditions for testing."""

    @staticmethod
    def create_client_error(
        error_code: str, message: str = "Test error", http_status: int = 400
    ) -> ClientError:
        """Create boto3 ClientError for testing."""
        return ClientError(
            error_response={
                "Error": {"Code": error_code, "Message": message},
                "ResponseMetadata": {"HTTPStatusCode": http_status},
            },
            operation_name="TestOperation",
        )

    @staticmethod
    def get_all_s3_error_codes() -> List[str]:
        """Get list of all S3 error codes for comprehensive testing."""
        return [
            "NoSuchBucket",
            "BucketNotEmpty",
            "NoSuchKey",
            "AccessDenied",
            "Forbidden",
            "InvalidRequest",
            "InvalidArgument",
            "Throttling",
            "RequestLimitExceeded",
            "TooManyRequests",
            "SlowDown",
            "ServiceUnavailable",
            "InternalError",
            "RequestTimeout",
            "EntityTooLarge",
            "InvalidObjectState",
            "NotImplemented",
            "PreconditionFailed",
            "QuotaExceeded",
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
            "max": max(measurements),
        }

    def assert_performance(
        self, operation_name: str, max_time: float, percentile: float = 95.0
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
        operations: List[Callable], max_concurrent: int = 10
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
        iterations: int = 10,
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
            "success_rate": len(results) / (len(results) + len(exceptions)),
        }


class ValidationTester:
    """Comprehensive validation testing utilities."""

    @staticmethod
    def test_all_validation_rules(
        validator_func: Callable,
        valid_inputs: Dict[str, Any],
        invalid_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Test all validation rules comprehensively."""
        results = {
            "valid_passed": 0,
            "valid_failed": 0,
            "invalid_caught": 0,
            "invalid_missed": 0,
            "failures": [],
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
    mock_s3_client: Optional[Mock] = None, **service_kwargs
) -> S3ArtifactService:
    """Create S3ArtifactService instance for testing."""
    default_kwargs = {
        "bucket_name": "test-bucket",
        "region_name": "us-east-1",
        "enable_validation": True,
        "enable_security_checks": True,
        "enable_integrity_checks": True,
    }
    default_kwargs.update(service_kwargs)

    service = S3ArtifactService(**default_kwargs)

    if mock_s3_client:
        service.s3_client = mock_s3_client

    return service


async def assert_operation_fails_with_error(
    operation: Callable,
    expected_error_type: type,
    expected_error_code: Optional[str] = None,
):
    """Assert that operation fails with specific error type and code."""
    try:
        await operation()
        raise AssertionError(
            f"Expected {expected_error_type.__name__} but operation succeeded"
        )
    except expected_error_type as e:
        if expected_error_code and hasattr(e, "error_code"):
            assert (
                e.error_code == expected_error_code
            ), f"Expected error code {expected_error_code}, got {e.error_code}"
    except Exception as e:
        raise AssertionError(
            f"Expected {expected_error_type.__name__}, got {type(e).__name__}: {e}"
        )


def parametrize_error_scenarios():
    """Decorator to parametrize tests with all error scenarios."""
    error_codes = ErrorSimulator.get_all_s3_error_codes()
    return pytest.mark.parametrize(
        "error_code", error_codes, ids=lambda code: f"error_{code}"
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
            "coverage_areas": set(),
        }
        self.error_count = 0
        self.error_types = set()

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
                "success_rate": self.metrics["passed"]
                / max(self.metrics["test_count"], 1),
            },
            "error_scenarios": len(self.metrics["error_scenarios_tested"]),
            "edge_cases": len(self.metrics["edge_cases_tested"]),
            "coverage_areas": list(self.metrics["coverage_areas"]),
        }
