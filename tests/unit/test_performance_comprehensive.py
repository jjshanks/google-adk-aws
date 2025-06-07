"""Comprehensive performance testing and benchmarking for S3ArtifactService."""
# mypy: ignore-errors

import os
import time
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from aws_adk import S3ArtifactService
from tests.utils import (
    ConcurrencyTester,
    PerformanceMeasurer,
    TestDataGenerator,
)


def create_mock_service_for_performance(
    mock_s3_setup: Dict[str, Any], **kwargs: Any
) -> S3ArtifactService:
    """Create a service instance without bucket validation for performance testing."""
    with patch.object(S3ArtifactService, "_validate_bucket_access"):
        return S3ArtifactService(
            bucket_name=mock_s3_setup["bucket_name"],
            region_name=mock_s3_setup["region_name"],
            **kwargs,
        )


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.asyncio
class TestOperationPerformance:
    """Test performance of individual operations."""

    @pytest.fixture
    def performance_service(self, mock_s3_setup: Dict[str, Any]) -> S3ArtifactService:
        """Service configured for performance testing."""
        return create_mock_service_for_performance(mock_s3_setup)

    @pytest.fixture
    def performance_measurer(self) -> PerformanceMeasurer:
        """Performance measurement utility."""
        return PerformanceMeasurer()

    @pytest.fixture
    def performance_benchmarks(self) -> Dict[str, float]:
        """Performance benchmark thresholds."""
        return {
            "save_artifact_max_time": 0.1,  # 100ms for save operations
            "load_artifact_max_time": 0.05,  # 50ms for load operations
            "list_operations_max_time": 0.02,  # 20ms for list operations
            "validation_max_time": 0.001,  # 1ms for validation
        }

    async def test_save_artifact_performance(
        self,
        performance_service: S3ArtifactService,
        performance_measurer: PerformanceMeasurer,
        performance_benchmarks: Dict[str, float],
    ) -> None:
        """Test save_artifact operation performance."""
        # Mock S3 operations for consistent performance testing
        mock_client = Mock()
        mock_paginator = Mock()
        mock_paginator.paginate = Mock(return_value=[{"Contents": []}])
        mock_client.get_paginator = Mock(return_value=mock_paginator)
        mock_client.put_object = Mock(
            return_value={"ResponseMetadata": {"HTTPStatusCode": 200}}
        )
        performance_service.s3_client = mock_client

        artifact = TestDataGenerator.generate_artifact(1024)  # 1KB artifact

        # Measure multiple save operations
        for i in range(10):
            async with performance_measurer.measure("save_artifact"):
                await performance_service.save_artifact(
                    app_name="test-app",
                    user_id="test-user",
                    session_id="test-session",
                    filename=f"perf-test-{i}.txt",
                    artifact=artifact,
                )

        # Assert performance requirements
        performance_measurer.assert_performance(
            "save_artifact", performance_benchmarks["save_artifact_max_time"]
        )

    async def test_load_artifact_performance(
        self,
        performance_service: S3ArtifactService,
        performance_measurer: PerformanceMeasurer,
        performance_benchmarks: Dict[str, float],
    ) -> None:
        """Test load_artifact operation performance."""
        # Mock S3 operations comprehensively
        test_content = b"test content for performance testing"
        mock_client = Mock()

        # Mock paginator for list_versions call
        mock_paginator = Mock()
        mock_paginator.paginate = Mock(
            return_value=[
                {
                    "Contents": [
                        {"Key": "test-app/test-user/test-session/perf-load.txt/0"}
                    ]
                }
            ]
        )
        mock_client.get_paginator = Mock(return_value=mock_paginator)

        # Mock get_object for load operation
        mock_client.get_object = Mock(
            return_value={
                "Body": Mock(read=lambda: test_content),
                "ContentType": "text/plain",
                "Metadata": {"version": "0"},
            }
        )
        performance_service.s3_client = mock_client

        # Measure multiple load operations
        for i in range(10):
            async with performance_measurer.measure("load_artifact"):
                result = await performance_service.load_artifact(
                    app_name="test-app",
                    user_id="test-user",
                    session_id="test-session",
                    filename="perf-load.txt",
                )
                assert result is not None

        performance_measurer.assert_performance(
            "load_artifact", performance_benchmarks["load_artifact_max_time"]
        )

    async def test_list_operations_performance(
        self,
        performance_service: S3ArtifactService,
        performance_measurer: PerformanceMeasurer,
        performance_benchmarks: Dict[str, float],
    ) -> None:
        """Test list operations performance."""
        # Mock S3 list operations comprehensively
        mock_client = Mock()

        # Mock list_objects_v2 for list_artifacts
        mock_client.list_objects_v2 = Mock(
            return_value={
                "Contents": [
                    {"Key": f"test-app/test-user/test-session/list-test-{i}.txt/0"}
                    for i in range(5)
                ]
            }
        )

        # Mock paginator for any internal list_versions calls
        mock_paginator = Mock()
        mock_paginator.paginate = Mock(return_value=[{"Contents": []}])
        mock_client.get_paginator = Mock(return_value=mock_paginator)

        performance_service.s3_client = mock_client

        # Measure list operations
        for i in range(10):
            async with performance_measurer.measure("list_artifacts"):
                artifacts = await performance_service.list_artifacts(
                    app_name="test-app", user_id="test-user"
                )
                assert len(artifacts) >= 5

        performance_measurer.assert_performance(
            "list_artifacts", performance_benchmarks["list_operations_max_time"]
        )

    async def test_validation_performance(
        self,
        performance_measurer: PerformanceMeasurer,
        performance_benchmarks: Dict[str, float],
    ) -> None:
        """Test input validation performance."""
        from aws_adk.validation import InputValidator

        validator = InputValidator()

        # Measure validation operations
        for i in range(100):
            async with performance_measurer.measure("validation"):
                validator.validate_artifact_params(
                    app_name=f"test-app-{i}",
                    user_id=f"user-{i}",
                    session_id=f"session-{i}",
                    filename=f"file-{i}.txt",
                    version=i,
                )

        performance_measurer.assert_performance(
            "validation", performance_benchmarks["validation_max_time"]
        )

    @pytest.mark.slow
    async def test_large_artifact_performance(
        self,
        performance_service: S3ArtifactService,
        performance_measurer: PerformanceMeasurer,
    ) -> None:
        """Test performance with large artifacts."""
        # Mock S3 operations
        mock_client = Mock()
        mock_paginator = Mock()
        mock_paginator.paginate = Mock(return_value=[{"Contents": []}])
        mock_client.get_paginator = Mock(return_value=mock_paginator)
        mock_client.put_object = Mock(
            return_value={"ResponseMetadata": {"HTTPStatusCode": 200}}
        )
        performance_service.s3_client = mock_client

        # Test different sizes
        sizes = [1024, 10 * 1024, 100 * 1024, 1024 * 1024]  # 1KB to 1MB

        for size in sizes:
            artifact = TestDataGenerator.generate_artifact(size)

            async with performance_measurer.measure(f"save_large_{size}"):
                await performance_service.save_artifact(
                    app_name="test-app",
                    user_id="test-user",
                    session_id="test-session",
                    filename=f"large-{size}.bin",
                    artifact=artifact,
                )

        # Verify performance scales reasonably
        stats_1kb = performance_measurer.get_stats("save_large_1024")
        stats_1mb = performance_measurer.get_stats("save_large_1048576")

        # 1MB should not take more than 1000x longer than 1KB
        if stats_1kb["average"] > 0:  # Avoid division by zero
            assert (
                stats_1mb["average"] / stats_1kb["average"] < 1000
            ), "Large artifact performance should scale reasonably"


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.concurrency
@pytest.mark.asyncio
class TestConcurrencyPerformance:
    """Test performance under concurrent load."""

    async def test_concurrent_save_performance(
        self, mock_s3_setup: Dict[str, Any]
    ) -> None:
        """Test concurrent save operation performance."""
        service = create_mock_service_for_performance(mock_s3_setup)
        measurer = PerformanceMeasurer()

        # Mock S3 operations
        mock_client = Mock()
        mock_paginator = Mock()
        mock_paginator.paginate = Mock(return_value=[{"Contents": []}])
        mock_client.get_paginator = Mock(return_value=mock_paginator)
        mock_client.put_object = Mock(
            return_value={"ResponseMetadata": {"HTTPStatusCode": 200}}
        )
        service.s3_client = mock_client

        # Create concurrent save operations
        async def save_operation(index: int) -> int:
            artifact = TestDataGenerator.generate_artifact(1024)
            async with measurer.measure("concurrent_save"):
                return await service.save_artifact(
                    app_name="test-app",
                    user_id="test-user",
                    session_id="test-session",
                    filename=f"concurrent-{index}.txt",
                    artifact=artifact,
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

    async def test_memory_usage_under_load(self, mock_s3_setup: Dict[str, Any]) -> None:
        """Test memory usage under concurrent load."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        service = create_mock_service_for_performance(mock_s3_setup)

        # Mock S3 operations
        mock_client = Mock()
        mock_paginator = Mock()
        mock_paginator.paginate = Mock(return_value=[{"Contents": []}])
        mock_client.get_paginator = Mock(return_value=mock_paginator)
        mock_client.put_object = Mock(
            return_value={"ResponseMetadata": {"HTTPStatusCode": 200}}
        )
        service.s3_client = mock_client

        process = psutil.Process(os.getpid())

        # Baseline memory usage
        baseline_memory = process.memory_info().rss

        # Create many concurrent operations
        async def memory_intensive_operation(index: int) -> int:
            # Large artifact
            artifact = TestDataGenerator.generate_artifact(100 * 1024)  # 100KB
            return await service.save_artifact(
                app_name="test-app",
                user_id="test-user",
                session_id="test-session",
                filename=f"memory-test-{index}.bin",
                artifact=artifact,
            )

        operations = [lambda i=i: memory_intensive_operation(i) for i in range(50)]

        await ConcurrencyTester.run_concurrent_operations(operations, max_concurrent=10)

        # Check memory usage after operations
        final_memory = process.memory_info().rss
        memory_increase = final_memory - baseline_memory

        # Memory increase should be reasonable (less than 100MB)
        assert (
            memory_increase < 100 * 1024 * 1024
        ), f"Memory increased by {memory_increase / 1024 / 1024:.2f}MB"


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.asyncio
class TestScalabilityBenchmarks:
    """Benchmark scalability characteristics."""

    async def test_operation_scaling(self, mock_s3_setup: Dict[str, Any]) -> None:
        """Test how operations scale with data size and volume."""
        service = create_mock_service_for_performance(mock_s3_setup)

        # Mock S3 operations with scaling simulation
        call_count = 0

        def mock_list_with_scaling(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate slight delay increase with more objects
            time.sleep(0.001 * min(call_count, 10))  # Max 10ms delay
            prefix = kwargs.get("Prefix", "")
            if "scale-test" in prefix:
                # Return increasing number of objects based on call count
                return {
                    "Contents": [
                        {"Key": f"scale-test/test/test/scale-{call_count}-{i}.txt"}
                        for i in range(min(call_count * 10, 200))
                    ]
                }
            return {"Contents": []}

        mock_client = Mock()
        mock_paginator = Mock()
        mock_paginator.paginate = Mock(return_value=[{"Contents": []}])
        mock_client.get_paginator = Mock(return_value=mock_paginator)
        mock_client.put_object = Mock(
            return_value={"ResponseMetadata": {"HTTPStatusCode": 200}}
        )
        mock_client.list_objects_v2 = mock_list_with_scaling
        service.s3_client = mock_client

        # Test scaling with number of artifacts
        artifact_counts = [10, 50, 100, 200]

        for count in artifact_counts:
            # Create artifacts (mocked)
            start_time = time.time()
            for i in range(count):
                artifact = TestDataGenerator.generate_artifact(1024)
                await service.save_artifact(
                    app_name="scale-test",
                    user_id="test-user",
                    session_id="test-session",
                    filename=f"scale-{count}-{i}.txt",
                    artifact=artifact,
                )
            creation_time = time.time() - start_time

            # Test list performance
            start_time = time.time()
            await service.list_artifacts(app_name="scale-test", user_id="test-user")
            list_time = time.time() - start_time

            print(
                f"Count {count}: Creation {creation_time:.2f}s, List {list_time:.2f}s"
            )

            # List time should not grow linearly with artifact count
            if count > 10:
                assert (
                    list_time < count * 0.1
                ), "List performance should not degrade linearly"

    async def test_throughput_benchmarks(self, mock_s3_setup: Dict[str, Any]) -> None:
        """Benchmark operation throughput."""
        service = create_mock_service_for_performance(mock_s3_setup)

        # Mock S3 operations for fast throughput testing
        mock_client = Mock()
        mock_paginator = Mock()
        mock_paginator.paginate = Mock(return_value=[{"Contents": []}])
        mock_client.get_paginator = Mock(return_value=mock_paginator)
        mock_client.put_object = Mock(
            return_value={"ResponseMetadata": {"HTTPStatusCode": 200}}
        )
        service.s3_client = mock_client

        # Measure sustained throughput
        operations_completed = 0
        start_time = time.time()
        test_duration = 2.0  # 2 seconds for faster testing

        async def throughput_operation() -> None:
            nonlocal operations_completed
            artifact = TestDataGenerator.generate_artifact(1024)
            await service.save_artifact(
                app_name="throughput",
                user_id="test-user",
                session_id="test-session",
                filename=f"throughput-{operations_completed}.txt",
                artifact=artifact,
            )
            operations_completed += 1

        # Run operations for specified duration
        while time.time() - start_time < test_duration:
            await throughput_operation()

        actual_duration = time.time() - start_time
        throughput = operations_completed / actual_duration

        print(f"Throughput: {throughput:.2f} operations/second")

        # Should achieve reasonable throughput (adjusted for mocked environment)
        assert (
            throughput > 50
        ), "Should achieve at least 50 operations/second in mocked environment"


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.asyncio
class TestPerformanceRegression:
    """Test for performance regressions."""

    async def test_baseline_performance_metrics(
        self, mock_s3_setup: Dict[str, Any]
    ) -> None:
        """Establish baseline performance metrics."""
        service = create_mock_service_for_performance(mock_s3_setup)
        measurer = PerformanceMeasurer()

        # Mock S3 operations
        mock_client = Mock()
        mock_paginator = Mock()
        mock_paginator.paginate = Mock(return_value=[{"Contents": []}])
        mock_client.get_paginator = Mock(return_value=mock_paginator)
        mock_client.put_object = Mock(
            return_value={"ResponseMetadata": {"HTTPStatusCode": 200}}
        )
        mock_client.get_object = Mock(
            return_value={
                "Body": Mock(read=lambda: b"test content"),
                "ContentType": "text/plain",
                "Metadata": {"version": "0"},
            }
        )
        mock_client.list_objects_v2 = Mock(return_value={"Contents": []})
        service.s3_client = mock_client

        # Baseline measurements
        operations = [
            (
                "save_artifact",
                lambda: service.save_artifact(
                    app_name="baseline",
                    user_id="test",
                    session_id="test",
                    filename="baseline.txt",
                    artifact=TestDataGenerator.generate_artifact(1024),
                ),
            ),
            (
                "load_artifact",
                lambda: service.load_artifact(
                    app_name="baseline",
                    user_id="test",
                    session_id="test",
                    filename="baseline.txt",
                ),
            ),
            (
                "list_artifacts",
                lambda: service.list_artifacts(app_name="baseline", user_id="test"),
            ),
        ]

        baseline_metrics = {}
        for op_name, operation in operations:
            # Run each operation multiple times
            for _ in range(5):
                async with measurer.measure(op_name):
                    await operation()

            baseline_metrics[op_name] = measurer.get_stats(op_name)

        # Store baseline metrics for regression detection
        assert baseline_metrics["save_artifact"]["average"] < 0.1
        assert baseline_metrics["load_artifact"]["average"] < 0.05
        assert baseline_metrics["list_artifacts"]["average"] < 0.02

        print("Baseline performance metrics:")
        for op_name, stats in baseline_metrics.items():
            print(f"  {op_name}: {stats['average']:.4f}s avg, {stats['max']:.4f}s max")

    async def test_performance_under_error_conditions(
        self, mock_s3_setup: Dict[str, Any]
    ) -> None:
        """Test performance degradation under error conditions."""
        service = create_mock_service_for_performance(mock_s3_setup)
        measurer = PerformanceMeasurer()

        # Mock client with intermittent failures
        call_count = 0

        def failing_operation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Intermittent failure")
            return {"ResponseMetadata": {"HTTPStatusCode": 200}}

        mock_client = Mock()
        mock_paginator = Mock()
        mock_paginator.paginate = Mock(return_value=[{"Contents": []}])
        mock_client.get_paginator = Mock(return_value=mock_paginator)
        mock_client.put_object = failing_operation
        service.s3_client = mock_client

        # Measure performance under error conditions
        successful_operations = 0
        for i in range(10):
            try:
                async with measurer.measure("error_condition_save"):
                    await service.save_artifact(
                        app_name="error-test",
                        user_id="test-user",
                        session_id="test-session",
                        filename=f"error-test-{i}.txt",
                        artifact=TestDataGenerator.generate_artifact(1024),
                    )
                successful_operations += 1
            except Exception:
                pass  # Expected failures

        # Should have some successful operations
        assert successful_operations > 0

        # Performance should still be reasonable for successful operations
        if measurer.measurements.get("error_condition_save"):
            stats = measurer.get_stats("error_condition_save")
            assert (
                stats["average"] < 0.2
            ), "Performance should remain reasonable under errors"
