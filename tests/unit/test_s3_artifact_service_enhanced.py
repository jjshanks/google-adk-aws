# type: ignore
"""Comprehensive unit tests for enhanced S3ArtifactService using moto mocking."""

import asyncio
from unittest.mock import patch

import boto3
import pytest
from botocore.exceptions import ClientError
from google.genai import types
from moto import mock_s3

from aws_adk import S3ArtifactService
from aws_adk.exceptions import S3ArtifactError, S3ConnectionError, S3PermissionError


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
            "s3_client": s3_client,
        }


@pytest.fixture
def artifact_service(mock_s3_setup):
    """Create S3ArtifactService instance with mocked S3."""
    return S3ArtifactService(
        bucket_name=mock_s3_setup["bucket_name"],
        region_name=mock_s3_setup["region_name"],
    )


@pytest.fixture
def enhanced_artifact_service(mock_s3_setup):
    """Create enhanced S3ArtifactService with encryption enabled."""
    return S3ArtifactService(
        bucket_name=mock_s3_setup["bucket_name"],
        region_name=mock_s3_setup["region_name"],
        enable_encryption=True,
    )


@pytest.fixture
def sample_artifact():
    """Create sample artifact for testing."""
    return types.Part.from_text("Sample artifact content", mime_type="text/plain")


@pytest.fixture
def large_artifact():
    """Create large artifact for multipart upload testing."""
    # Create 5MB test content
    large_content = "A" * (5 * 1024 * 1024)
    return types.Part.from_text(large_content, mime_type="text/plain")


class TestS3ArtifactServiceInitialization:
    """Test service initialization and configuration."""

    def test_successful_initialization(self, mock_s3_setup):
        """Test successful service initialization."""
        service = S3ArtifactService(
            bucket_name=mock_s3_setup["bucket_name"],
            region_name=mock_s3_setup["region_name"],
        )
        assert service.bucket_name == mock_s3_setup["bucket_name"]
        assert service.region_name == mock_s3_setup["region_name"]
        assert service.security_manager is not None
        assert service.batch_operations is not None
        assert service.multipart_manager is not None

    def test_initialization_with_encryption(self, mock_s3_setup):
        """Test initialization with encryption enabled."""
        service = S3ArtifactService(
            bucket_name=mock_s3_setup["bucket_name"],
            enable_encryption=True,
            encryption_key="test_key_12345",
        )
        assert service.enable_encryption is True
        assert service.encryption_manager is not None

    def test_initialization_with_credentials(self, mock_s3_setup):
        """Test initialization with explicit credentials."""
        service = S3ArtifactService(
            bucket_name=mock_s3_setup["bucket_name"],
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
        )
        assert service.aws_access_key_id == "test_key"

    def test_initialization_with_custom_endpoint(self, mock_s3_setup):
        """Test initialization with custom S3 endpoint."""
        service = S3ArtifactService(
            bucket_name=mock_s3_setup["bucket_name"],
            endpoint_url="http://localhost:9000",
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
            artifact=sample_artifact,
        )
        assert version == 0

    @pytest.mark.asyncio
    async def test_save_artifact_with_encryption(
        self, enhanced_artifact_service, sample_artifact
    ):
        """Test saving artifact with encryption enabled."""
        version = await enhanced_artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="encrypted_file.txt",
            artifact=sample_artifact,
        )
        assert version == 0

        # Verify we can load it back
        loaded = await enhanced_artifact_service.load_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="encrypted_file.txt",
        )
        assert loaded is not None
        assert loaded.inline_data.data == sample_artifact.inline_data.data

    @pytest.mark.asyncio
    async def test_save_large_artifact_multipart(
        self, artifact_service, large_artifact
    ):
        """Test saving large artifact using multipart upload."""
        version = await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="large_file.txt",
            artifact=large_artifact,
        )
        assert version == 0

    @pytest.mark.asyncio
    async def test_save_artifact_version_increment(
        self, artifact_service, sample_artifact
    ):
        """Test version increment on multiple saves."""
        # Save first version
        v1 = await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact,
        )

        # Save second version
        v2 = await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact,
        )

        assert v1 == 0
        assert v2 == 1

    @pytest.mark.asyncio
    async def test_save_user_namespace_artifact(
        self, artifact_service, sample_artifact
    ):
        """Test saving artifact with user namespace."""
        version = await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="user:global_file.txt",
            artifact=sample_artifact,
        )
        assert version == 0

    @pytest.mark.asyncio
    async def test_load_artifact_latest_version(
        self, artifact_service, sample_artifact
    ):
        """Test loading latest version of artifact."""
        # Save artifact
        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact,
        )

        # Load artifact (latest version)
        loaded = await artifact_service.load_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
        )

        assert loaded is not None
        assert loaded.inline_data.data == sample_artifact.inline_data.data

    @pytest.mark.asyncio
    async def test_load_artifact_specific_version(
        self, artifact_service, sample_artifact
    ):
        """Test loading specific version of artifact."""
        # Save multiple versions
        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact,
        )

        modified_artifact = types.Part.from_text(
            "Modified content", mime_type="text/plain"
        )
        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=modified_artifact,
        )

        # Load first version
        loaded = await artifact_service.load_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            version=0,
        )

        assert loaded.inline_data.data == sample_artifact.inline_data.data

    @pytest.mark.asyncio
    async def test_load_nonexistent_artifact(self, artifact_service):
        """Test loading non-existent artifact returns None."""
        loaded = await artifact_service.load_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="nonexistent.txt",
        )
        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_artifact_keys_empty(self, artifact_service):
        """Test listing artifact keys with no artifacts."""
        keys = await artifact_service.list_artifact_keys(
            app_name="test_app", user_id="user123", session_id="session456"
        )
        assert keys == []

    @pytest.mark.asyncio
    async def test_list_artifact_keys_with_artifacts(
        self, artifact_service, sample_artifact
    ):
        """Test listing artifact keys with multiple artifacts."""
        # Save session-scoped artifacts
        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="file1.txt",
            artifact=sample_artifact,
        )

        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="file2.txt",
            artifact=sample_artifact,
        )

        # Save user-scoped artifact
        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="user:global.txt",
            artifact=sample_artifact,
        )

        keys = await artifact_service.list_artifact_keys(
            app_name="test_app", user_id="user123", session_id="session456"
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
            artifact=sample_artifact,
        )

        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact,
        )

        # Delete artifact
        await artifact_service.delete_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
        )

        # Verify deletion
        loaded = await artifact_service.load_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
        )
        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_versions(self, artifact_service, sample_artifact):
        """Test listing versions of an artifact."""
        # Save multiple versions
        for _ in range(3):
            await artifact_service.save_artifact(
                app_name="test_app",
                user_id="user123",
                session_id="session456",
                filename="test_file.txt",
                artifact=sample_artifact,
            )

        versions = await artifact_service.list_versions(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
        )

        assert versions == [0, 1, 2]


class TestS3ArtifactServiceErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_save_artifact_s3_error(self, artifact_service, sample_artifact):
        """Test save artifact with S3 client error."""
        with patch.object(artifact_service.s3_client, "put_object") as mock_put:
            mock_put.side_effect = ClientError(
                error_response={
                    "Error": {"Code": "AccessDenied", "Message": "Access denied"}
                },
                operation_name="PutObject",
            )

            with pytest.raises(S3PermissionError, match="Access denied"):
                await artifact_service.save_artifact(
                    app_name="test_app",
                    user_id="user123",
                    session_id="session456",
                    filename="test_file.txt",
                    artifact=sample_artifact,
                )

    @pytest.mark.asyncio
    async def test_load_artifact_s3_error(self, artifact_service):
        """Test load artifact with S3 client error."""
        with patch.object(artifact_service.s3_client, "get_object") as mock_get:
            mock_get.side_effect = ClientError(
                error_response={
                    "Error": {"Code": "InternalError", "Message": "Internal error"}
                },
                operation_name="GetObject",
            )

            with pytest.raises(S3ConnectionError, match="S3 service error"):
                await artifact_service.load_artifact(
                    app_name="test_app",
                    user_id="user123",
                    session_id="session456",
                    filename="test_file.txt",
                )


class TestS3ArtifactServiceEnhancedFeatures:
    """Test enhanced features like security, batch operations."""

    @pytest.mark.asyncio
    async def test_get_security_status(self, artifact_service):
        """Test getting bucket security status."""
        status = await artifact_service.get_security_status()
        assert isinstance(status, dict)
        assert "encryption" in status
        assert "recommendations" in status

    @pytest.mark.asyncio
    async def test_generate_presigned_url(self, artifact_service, sample_artifact):
        """Test generating presigned URLs."""
        # Save artifact first
        await artifact_service.save_artifact(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
            artifact=sample_artifact,
        )

        # Generate presigned URL
        url = await artifact_service.generate_presigned_url(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filename="test_file.txt",
        )

        assert isinstance(url, str)
        assert "test-artifacts-bucket" in url

    @pytest.mark.asyncio
    async def test_batch_delete_artifacts(self, artifact_service, sample_artifact):
        """Test batch deletion of multiple artifacts."""
        # Save multiple artifacts
        filenames = ["file1.txt", "file2.txt", "file3.txt"]
        for filename in filenames:
            await artifact_service.save_artifact(
                app_name="test_app",
                user_id="user123",
                session_id="session456",
                filename=filename,
                artifact=sample_artifact,
            )

        # Batch delete
        result = await artifact_service.batch_delete_artifacts(
            app_name="test_app",
            user_id="user123",
            session_id="session456",
            filenames=filenames,
        )

        assert "deleted" in result
        assert "errors" in result
        assert len(result["deleted"]) == 3

    def test_get_connection_stats(self, artifact_service):
        """Test getting connection pool statistics."""
        stats = artifact_service.get_connection_stats()
        assert isinstance(stats, dict)
        assert "total_connections" in stats
        assert "cache_hits" in stats


class TestS3ArtifactServiceUtilities:
    """Test utility methods and edge cases."""

    def test_file_has_user_namespace(self, artifact_service):
        """Test user namespace detection."""
        assert artifact_service._file_has_user_namespace("user:test.txt") is True
        assert artifact_service._file_has_user_namespace("regular.txt") is False
        assert artifact_service._file_has_user_namespace("") is False

    def test_security_object_key_generation(self, artifact_service):
        """Test secure object key generation."""
        key = artifact_service.security_manager.generate_secure_object_key(
            app_name="app",
            user_id="user123",
            session_id="session456",
            filename="test.txt",
            version=1,
        )
        assert key == "app/user123/session456/test.txt/1"

    def test_security_object_key_validation(self, artifact_service):
        """Test object key security validation."""
        # Valid key
        valid_key = "app/user123/session456/test.txt/1"
        assert artifact_service.security_manager.validate_object_key(valid_key) is True

        # Invalid key with path traversal
        invalid_key = "app/user123/../../../etc/passwd"
        assert (
            artifact_service.security_manager.validate_object_key(invalid_key) is False
        )

    def test_content_hash_calculation(self, artifact_service):
        """Test content hash calculation for integrity."""
        content = b"test content"
        hash_value = artifact_service.security_manager.calculate_content_hash(content)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256 hex length

    def test_content_integrity_verification(self, artifact_service):
        """Test content integrity verification."""
        content = b"test content"
        hash_value = artifact_service.security_manager.calculate_content_hash(content)

        # Valid verification
        assert (
            artifact_service.security_manager.verify_content_integrity(
                content, hash_value
            )
            is True
        )

        # Invalid verification
        wrong_hash = "invalid_hash"
        assert (
            artifact_service.security_manager.verify_content_integrity(
                content, wrong_hash
            )
            is False
        )


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
                artifact=sample_artifact,
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
            artifact=sample_artifact,
        )

        # Concurrent loads
        tasks = []
        for _ in range(5):
            task = artifact_service.load_artifact(
                app_name="test_app",
                user_id="user123",
                session_id="session456",
                filename="test_file.txt",
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        assert all(r is not None for r in results)
        assert all(
            r.inline_data.data == sample_artifact.inline_data.data for r in results
        )
