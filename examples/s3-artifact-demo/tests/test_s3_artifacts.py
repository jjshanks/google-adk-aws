# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for S3 artifact operations."""

from unittest.mock import AsyncMock, Mock

import pytest
from s3_artifact_demo.tools.file_tools import (
    get_artifact_versions,
    list_artifact_files,
    load_artifact_data,
    save_data_to_artifact,
)


@pytest.mark.asyncio
class TestS3ArtifactTools:
    """Test cases for S3 artifact tool functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_service = Mock()
        self.mock_service.save_artifact = AsyncMock()
        self.mock_service.load_artifact = AsyncMock()
        self.mock_service.list_artifact_keys = AsyncMock()
        self.mock_service.list_versions = AsyncMock()

    async def test_save_data_to_artifact_success(self):
        """Test successful artifact saving."""
        # Setup
        self.mock_service.save_artifact.return_value = 0

        # Execute
        result = await save_data_to_artifact(
            data="test content", filename="test.txt", artifact_service=self.mock_service
        )

        # Verify
        assert result["status"] == "success"
        assert result["version"] == 0
        assert "test.txt" in result["message"]

        # Check that save_artifact was called with correct parameters
        self.mock_service.save_artifact.assert_called_once()
        call_args = self.mock_service.save_artifact.call_args
        assert call_args.kwargs["filename"] == "test.txt"
        assert call_args.kwargs["app_name"] == "s3_demo_app"

    async def test_save_data_to_artifact_empty_content(self):
        """Test saving empty content returns error."""
        result = await save_data_to_artifact(
            data="", filename="test.txt", artifact_service=self.mock_service
        )

        assert result["status"] == "error"
        assert "empty content" in result["message"]
        self.mock_service.save_artifact.assert_not_called()

    async def test_save_data_to_artifact_invalid_filename(self):
        """Test saving with invalid filename returns error."""
        result = await save_data_to_artifact(
            data="test content", filename="", artifact_service=self.mock_service
        )

        assert result["status"] == "error"
        assert "non-empty string" in result["message"]
        self.mock_service.save_artifact.assert_not_called()

    async def test_save_data_to_artifact_s3_error(self):
        """Test handling of S3 service errors."""
        from aws_adk.s3_artifact_service import S3ArtifactError

        self.mock_service.save_artifact.side_effect = S3ArtifactError("S3 error")

        result = await save_data_to_artifact(
            data="test content", filename="test.txt", artifact_service=self.mock_service
        )

        assert result["status"] == "error"
        assert "S3 operation failed" in result["message"]

    async def test_load_artifact_data_success(self):
        """Test successful artifact loading."""
        # Setup mock artifact
        mock_artifact = Mock()
        mock_artifact.text = "test content"
        mock_artifact.inline_data.mime_type = "text/plain"

        self.mock_service.load_artifact.return_value = mock_artifact
        self.mock_service.list_versions.return_value = [0, 1, 2]

        # Execute
        result = await load_artifact_data(
            filename="test.txt", artifact_service=self.mock_service
        )

        # Verify
        assert result["status"] == "success"
        assert result["content"] == "test content"
        assert result["mime_type"] == "text/plain"
        assert result["version"] == 2  # Latest version

    async def test_load_artifact_data_not_found(self):
        """Test loading non-existent artifact."""
        self.mock_service.load_artifact.return_value = None

        result = await load_artifact_data(
            filename="nonexistent.txt", artifact_service=self.mock_service
        )

        assert result["status"] == "not_found"
        assert "not found" in result["message"]

    async def test_load_artifact_data_specific_version(self):
        """Test loading specific version of artifact."""
        mock_artifact = Mock()
        mock_artifact.text = "version 1 content"
        mock_artifact.inline_data.mime_type = "text/plain"

        self.mock_service.load_artifact.return_value = mock_artifact

        result = await load_artifact_data(
            filename="test.txt", artifact_service=self.mock_service, version=1
        )

        assert result["status"] == "success"
        assert result["version"] == 1

        # Verify correct version was requested
        call_args = self.mock_service.load_artifact.call_args
        assert call_args.kwargs["version"] == 1

    async def test_list_artifact_files_success(self):
        """Test successful artifact listing."""
        self.mock_service.list_artifact_keys.return_value = [
            "file1.txt",
            "user:settings.json",
            "report.md",
        ]

        result = await list_artifact_files(artifact_service=self.mock_service)

        assert result["status"] == "success"
        assert result["count"] == 3
        assert "file1.txt" in result["files"]
        assert "user:settings.json" in result["files"]
        assert "report.md" in result["files"]

    async def test_list_artifact_files_empty(self):
        """Test listing when no artifacts exist."""
        self.mock_service.list_artifact_keys.return_value = []

        result = await list_artifact_files(artifact_service=self.mock_service)

        assert result["status"] == "success"
        assert result["count"] == 0
        assert result["files"] == []

    async def test_get_artifact_versions_success(self):
        """Test successful version listing."""
        self.mock_service.list_versions.return_value = [0, 1, 2, 3]

        result = await get_artifact_versions(
            filename="test.txt", artifact_service=self.mock_service
        )

        assert result["status"] == "success"
        assert result["versions"] == [0, 1, 2, 3]
        assert result["count"] == 4
        assert result["latest"] == 3

    async def test_get_artifact_versions_no_versions(self):
        """Test version listing for non-existent file."""
        self.mock_service.list_versions.return_value = []

        result = await get_artifact_versions(
            filename="nonexistent.txt", artifact_service=self.mock_service
        )

        assert result["status"] == "success"
        assert result["versions"] == []
        assert result["count"] == 0
        assert result["latest"] is None

    async def test_get_artifact_versions_invalid_filename(self):
        """Test version listing with invalid filename."""
        result = await get_artifact_versions(
            filename="", artifact_service=self.mock_service
        )

        assert result["status"] == "error"
        assert "non-empty string" in result["message"]
        self.mock_service.list_versions.assert_not_called()


@pytest.mark.unit
class TestArtifactCreation:
    """Test artifact creation in save_data_to_artifact."""

    @pytest.mark.asyncio
    async def test_artifact_creation(self):
        """Test that artifacts are created correctly."""
        mock_service = Mock()
        mock_service.save_artifact = AsyncMock(return_value=0)

        await save_data_to_artifact(
            data='{"key": "value"}', filename="data.json", artifact_service=mock_service
        )

        # Check that save_artifact was called with an artifact
        call_args = mock_service.save_artifact.call_args
        artifact = call_args.kwargs["artifact"]
        assert artifact is not None
        assert hasattr(artifact, "text")
        assert artifact.text == '{"key": "value"}'


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration test scenarios."""

    @pytest.mark.asyncio
    async def test_save_load_cycle(self):
        """Test complete save and load cycle."""
        mock_service = Mock()
        mock_service.save_artifact = AsyncMock(return_value=0)

        mock_artifact = Mock()
        mock_artifact.text = "test content"
        mock_artifact.inline_data.mime_type = "text/plain"
        mock_service.load_artifact = AsyncMock(return_value=mock_artifact)
        mock_service.list_versions = AsyncMock(return_value=[0])

        # Save
        save_result = await save_data_to_artifact(
            data="test content", filename="test.txt", artifact_service=mock_service
        )

        # Load
        load_result = await load_artifact_data(
            filename="test.txt", artifact_service=mock_service
        )

        # Verify
        assert save_result["status"] == "success"
        assert load_result["status"] == "success"
        assert load_result["content"] == "test content"

    @pytest.mark.asyncio
    async def test_versioning_workflow(self):
        """Test versioning workflow."""
        mock_service = Mock()

        # Simulate saving multiple versions
        mock_service.save_artifact = AsyncMock(side_effect=[0, 1, 2])
        mock_service.list_versions = AsyncMock(return_value=[0, 1, 2])

        # Save three versions
        for i in range(3):
            result = await save_data_to_artifact(
                data=f"content version {i}",
                filename="versioned.txt",
                artifact_service=mock_service,
            )
            assert result["status"] == "success"
            assert result["version"] == i

        # Check versions
        versions_result = await get_artifact_versions(
            filename="versioned.txt", artifact_service=mock_service
        )

        assert versions_result["status"] == "success"
        assert versions_result["count"] == 3
        assert versions_result["latest"] == 2
