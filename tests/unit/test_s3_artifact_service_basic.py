"""Basic verification tests for S3ArtifactService."""

import pytest

from aws_adk import S3ArtifactService


class TestS3ArtifactServiceBasic:
    """Basic tests to verify package structure and imports."""

    def test_import_s3_artifact_service(self) -> None:
        """Test that S3ArtifactService can be imported."""
        assert S3ArtifactService is not None

    def test_s3_artifact_service_initialization(self) -> None:
        """Test basic initialization (without real S3 connection)."""
        # This will fail bucket verification, but that's expected
        with pytest.raises(Exception):
            S3ArtifactService(bucket_name="test-bucket")

    def test_s3_artifact_service_has_required_methods(self) -> None:
        """Test that all required BaseArtifactService methods exist."""
        required_methods = [
            "save_artifact",
            "load_artifact",
            "list_artifact_keys",
            "delete_artifact",
            "list_versions",
        ]

        for method_name in required_methods:
            assert hasattr(S3ArtifactService, method_name)
            method = getattr(S3ArtifactService, method_name)
            assert callable(method)
