# type: ignore
"""Integration tests for S3ArtifactService with real S3 operations."""

import asyncio
import os

import pytest
from google.genai import types

from aws_adk import S3ArtifactService


@pytest.fixture
def integration_config():
    """Configuration for integration tests."""
    return {
        "bucket_name": os.environ.get(
            "S3_TEST_BUCKET", "google-adk-aws-integration-test"
        ),
        "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    }


@pytest.fixture
def real_s3_service(integration_config):
    """Create S3ArtifactService for integration testing."""
    if not os.environ.get("RUN_INTEGRATION_TESTS"):
        pytest.skip(
            "Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable."
        )

    return S3ArtifactService(
        bucket_name=integration_config["bucket_name"],
        region_name=integration_config["region_name"],
    )


@pytest.fixture
def real_s3_service_enhanced(integration_config):
    """Create enhanced S3ArtifactService with encryption for integration testing."""
    if not os.environ.get("RUN_INTEGRATION_TESTS"):
        pytest.skip(
            "Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable."
        )

    return S3ArtifactService(
        bucket_name=integration_config["bucket_name"],
        region_name=integration_config["region_name"],
        enable_encryption=True,
    )


class TestS3IntegrationBasic:
    """Basic integration tests with real S3."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_artifact_lifecycle(self, real_s3_service):
        """Test complete artifact lifecycle with real S3."""
        artifact = types.Part.from_text(
            "Integration test content", mime_type="text/plain"
        )

        # Save artifact
        version = await real_s3_service.save_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="lifecycle_test.txt",
            artifact=artifact,
        )
        assert version == 0

        # Load artifact
        loaded = await real_s3_service.load_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="lifecycle_test.txt",
        )
        assert loaded is not None
        assert loaded.inline_data.data == artifact.inline_data.data

        # List artifacts
        keys = await real_s3_service.list_artifact_keys(
            app_name="integration_test", user_id="test_user", session_id="test_session"
        )
        assert "lifecycle_test.txt" in keys

        # List versions
        versions = await real_s3_service.list_versions(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="lifecycle_test.txt",
        )
        assert versions == [0]

        # Delete artifact
        await real_s3_service.delete_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="lifecycle_test.txt",
        )

        # Verify deletion
        loaded_after_delete = await real_s3_service.load_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="lifecycle_test.txt",
        )
        assert loaded_after_delete is None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_encrypted_artifact_lifecycle(self, real_s3_service_enhanced):
        """Test encrypted artifact lifecycle with real S3."""
        artifact = types.Part.from_text(
            "Encrypted test content", mime_type="text/plain"
        )

        # Save encrypted artifact
        version = await real_s3_service_enhanced.save_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="encrypted_test.txt",
            artifact=artifact,
        )
        assert version == 0

        # Load and verify decryption
        loaded = await real_s3_service_enhanced.load_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="encrypted_test.txt",
        )
        assert loaded is not None
        assert loaded.inline_data.data == artifact.inline_data.data

        # Cleanup
        await real_s3_service_enhanced.delete_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="encrypted_test.txt",
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_security_features(self, real_s3_service):
        """Test security features with real S3."""
        # Get security status
        security_status = await real_s3_service.get_security_status()
        assert isinstance(security_status, dict)
        assert "encryption" in security_status
        assert "recommendations" in security_status

        # Test presigned URL generation
        artifact = types.Part.from_text("Presigned URL test", mime_type="text/plain")

        await real_s3_service.save_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="presigned_test.txt",
            artifact=artifact,
        )

        url = await real_s3_service.generate_presigned_url(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="presigned_test.txt",
            expiration=300,  # 5 minutes
        )

        assert isinstance(url, str)
        assert "https" in url or "http" in url

        # Cleanup
        await real_s3_service.delete_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="presigned_test.txt",
        )


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
        await real_s3_service.save_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="large_file.txt",
            artifact=artifact,
        )

        # Load and verify
        loaded = await real_s3_service.load_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="large_file.txt",
        )

        assert loaded.inline_data.data == artifact.inline_data.data

        # Cleanup
        await real_s3_service.delete_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="test_session",
            filename="large_file.txt",
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
                artifact=artifact,
            )
            tasks.append(task)

        versions = await asyncio.gather(*tasks)
        assert len(versions) == artifacts_count

        # List all artifacts
        keys = await real_s3_service.list_artifact_keys(
            app_name="integration_test", user_id="test_user", session_id="perf_session"
        )
        assert len(keys) == artifacts_count

        # Test batch delete
        filenames = [f"perf_file_{i:03d}.txt" for i in range(artifacts_count)]
        result = await real_s3_service.batch_delete_artifacts(
            app_name="integration_test",
            user_id="test_user",
            session_id="perf_session",
            filenames=filenames,
        )

        assert len(result["deleted"]) == artifacts_count
        assert len(result["errors"]) == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, real_s3_service):
        """Test concurrent read/write operations."""
        artifact = types.Part.from_text("Concurrent test", mime_type="text/plain")

        # Save initial artifact
        await real_s3_service.save_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="concurrent_session",
            filename="concurrent_test.txt",
            artifact=artifact,
        )

        # Concurrent read operations
        read_tasks = []
        for i in range(10):
            task = real_s3_service.load_artifact(
                app_name="integration_test",
                user_id="test_user",
                session_id="concurrent_session",
                filename="concurrent_test.txt",
            )
            read_tasks.append(task)

        # Concurrent write operations (different files)
        write_tasks = []
        for i in range(5):
            task = real_s3_service.save_artifact(
                app_name="integration_test",
                user_id="test_user",
                session_id="concurrent_session",
                filename=f"concurrent_write_{i}.txt",
                artifact=artifact,
            )
            write_tasks.append(task)

        # Execute all operations concurrently
        read_results = await asyncio.gather(*read_tasks)
        write_results = await asyncio.gather(*write_tasks)

        # Verify results
        assert all(r is not None for r in read_results)
        assert all(
            r.inline_data.data == artifact.inline_data.data for r in read_results
        )
        assert all(v == 0 for v in write_results)  # First version for each file

        # Cleanup
        cleanup_files = ["concurrent_test.txt"] + [
            f"concurrent_write_{i}.txt" for i in range(5)
        ]
        await real_s3_service.batch_delete_artifacts(
            app_name="integration_test",
            user_id="test_user",
            session_id="concurrent_session",
            filenames=cleanup_files,
        )


class TestS3IntegrationErrorHandling:
    """Test error handling with real S3."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_network_resilience(self, real_s3_service):
        """Test resilience to network issues (simulated)."""
        artifact = types.Part.from_text("Resilience test", mime_type="text/plain")

        # This test would require actual network simulation
        # For now, just verify normal operation works
        version = await real_s3_service.save_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="resilience_session",
            filename="resilience_test.txt",
            artifact=artifact,
        )

        assert version == 0

        # Cleanup
        await real_s3_service.delete_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="resilience_session",
            filename="resilience_test.txt",
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connection_pool_stats(self, real_s3_service):
        """Test connection pool statistics tracking."""
        stats = real_s3_service.get_connection_stats()

        assert isinstance(stats, dict)
        assert "total_connections" in stats
        assert "active_connections" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats

        # Perform operations to change stats
        artifact = types.Part.from_text("Stats test", mime_type="text/plain")

        await real_s3_service.save_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="stats_session",
            filename="stats_test.txt",
            artifact=artifact,
        )

        # Check stats again
        new_stats = real_s3_service.get_connection_stats()
        assert new_stats["total_connections"] >= stats["total_connections"]

        # Cleanup
        await real_s3_service.delete_artifact(
            app_name="integration_test",
            user_id="test_user",
            session_id="stats_session",
            filename="stats_test.txt",
        )
