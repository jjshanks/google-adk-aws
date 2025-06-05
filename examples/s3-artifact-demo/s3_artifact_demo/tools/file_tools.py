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

"""File operation tools for S3 artifact management."""

import logging
from typing import Any, Dict, Optional

from google.genai import types

from aws_adk.s3_artifact_service import S3ArtifactError, S3ArtifactService

logger = logging.getLogger(__name__)


async def save_data_to_artifact(
    data: str,
    filename: str,
    artifact_service: S3ArtifactService,
    app_name: str = "s3_demo_app",
    user_id: str = "demo_user",
    session_id: str = "demo_session",
) -> Dict[str, Any]:
    """
    Save text data as an artifact to S3.

    This function creates an artifact from text data and saves it to S3 with
    automatic versioning. The artifact will be accessible within the specified
    app, user, and session context.

    Args:
        data: Text content to save
        filename: Name for the artifact file
        artifact_service: S3ArtifactService instance
        app_name: Application identifier (default: "s3_demo_app")
        user_id: User identifier (default: "demo_user")
        session_id: Session identifier (default: "demo_session")

    Returns:
        dict: Result with status and version information
            - status: "success" or "error"
            - version: Version number assigned (if successful)
            - message: Human-readable description

    Example:
        >>> artifact_service = S3ArtifactService(bucket_name="my-bucket")
        >>> result = await save_data_to_artifact(
        ...     "Hello, World!",
        ...     "greeting.txt",
        ...     artifact_service
        ... )
        >>> print(result)
        {
            "status": "success",
            "version": 0,
            "message": "Saved greeting.txt version 0"
        }

    Raises:
        No exceptions are raised directly; all errors are captured in the
        return dictionary with status="error".
    """
    try:
        logger.info(f"Saving artifact: {filename} with {len(data)} characters")

        # Validate inputs
        if not data:
            return {"status": "error", "message": "Cannot save empty content"}

        if not filename or not isinstance(filename, str):
            return {"status": "error", "message": "Filename must be a non-empty string"}

        # Create artifact from text data
        artifact = types.Part(text=data)

        # Save to S3
        version = await artifact_service.save_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            artifact=artifact,
        )

        logger.info(f"Successfully saved {filename} as version {version}")
        return {
            "status": "success",
            "version": version,
            "message": f"Saved {filename} version {version}",
        }

    except S3ArtifactError as e:
        logger.error(f"S3 error saving artifact {filename}: {e}")
        return {"status": "error", "message": f"S3 operation failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error saving artifact {filename}: {e}")
        return {"status": "error", "message": f"Failed to save artifact: {str(e)}"}


async def load_artifact_data(
    filename: str,
    artifact_service: S3ArtifactService,
    version: Optional[int] = None,
    app_name: str = "s3_demo_app",
    user_id: str = "demo_user",
    session_id: str = "demo_session",
) -> Dict[str, Any]:
    """
    Load artifact data from S3.

    Retrieves an artifact from S3 storage and returns its content. If no version
    is specified, loads the latest version. Supports both session-scoped and
    user-scoped artifacts.

    Args:
        filename: Name of the artifact file to load
        artifact_service: S3ArtifactService instance
        version: Specific version to load (latest if None)
        app_name: Application identifier (default: "s3_demo_app")
        user_id: User identifier (default: "demo_user")
        session_id: Session identifier (default: "demo_session")

    Returns:
        dict: Result with status and content information
            - status: "success", "not_found", or "error"
            - content: The artifact content (if successful)
            - mime_type: MIME type of the content (if successful)
            - version: Version number that was loaded (if successful)
            - message: Human-readable description

    Example:
        >>> result = await load_artifact_data("greeting.txt", artifact_service)
        >>> if result["status"] == "success":
        ...     print(f"Content: {result['content']}")
        ...     print(f"Version: {result['version']}")

    Raises:
        No exceptions are raised directly; all errors are captured in the
        return dictionary with appropriate status.
    """
    try:
        logger.info(f"Loading artifact: {filename}, version: {version}")

        # Validate filename
        if not filename or not isinstance(filename, str):
            return {"status": "error", "message": "Filename must be a non-empty string"}

        # Load from S3
        artifact = await artifact_service.load_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            version=version,
        )

        if artifact is None:
            logger.debug(f"Artifact not found: {filename}")
            return {
                "status": "not_found",
                "message": f"Artifact '{filename}' not found",
            }

        # Extract content and metadata
        content = (
            artifact.text
            if hasattr(artifact, "text")
            else str(artifact.inline_data.data, "utf-8")
        )
        mime_type = artifact.inline_data.mime_type

        # If version wasn't specified, get the actual version loaded
        if version is None:
            versions = await artifact_service.list_versions(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
            )
            actual_version = max(versions) if versions else 0
        else:
            actual_version = version

        logger.info(f"Successfully loaded {filename} version {actual_version}")
        return {
            "status": "success",
            "content": content,
            "mime_type": mime_type,
            "version": actual_version,
            "message": f"Loaded {filename} version {actual_version}",
        }

    except S3ArtifactError as e:
        logger.error(f"S3 error loading artifact {filename}: {e}")
        return {"status": "error", "message": f"S3 operation failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error loading artifact {filename}: {e}")
        return {"status": "error", "message": f"Failed to load artifact: {str(e)}"}


async def list_artifact_files(
    artifact_service: S3ArtifactService,
    app_name: str = "s3_demo_app",
    user_id: str = "demo_user",
    session_id: str = "demo_session",
) -> Dict[str, Any]:
    """
    List all artifact files available for the current session.

    Returns a list of all artifact filenames accessible within the specified
    app, user, and session context. This includes both session-scoped and
    user-scoped artifacts.

    Args:
        artifact_service: S3ArtifactService instance
        app_name: Application identifier (default: "s3_demo_app")
        user_id: User identifier (default: "demo_user")
        session_id: Session identifier (default: "demo_session")

    Returns:
        dict: Result with status and file list
            - status: "success" or "error"
            - files: List of artifact filenames (if successful)
            - count: Number of files found (if successful)
            - message: Human-readable description

    Example:
        >>> result = await list_artifact_files(artifact_service)
        >>> if result["status"] == "success":
        ...     print(f"Found {result['count']} files:")
        ...     for filename in result["files"]:
        ...         print(f"  - {filename}")
    """
    try:
        logger.info("Listing available artifact files")

        # List all artifacts for the session
        artifact_keys = await artifact_service.list_artifact_keys(
            app_name=app_name, user_id=user_id, session_id=session_id
        )

        logger.info(f"Found {len(artifact_keys)} artifact files")
        return {
            "status": "success",
            "files": artifact_keys,
            "count": len(artifact_keys),
            "message": f"Found {len(artifact_keys)} artifact files",
        }

    except S3ArtifactError as e:
        logger.error(f"S3 error listing artifacts: {e}")
        return {"status": "error", "message": f"S3 operation failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error listing artifacts: {e}")
        return {"status": "error", "message": f"Failed to list artifacts: {str(e)}"}


async def get_artifact_versions(
    filename: str,
    artifact_service: S3ArtifactService,
    app_name: str = "s3_demo_app",
    user_id: str = "demo_user",
    session_id: str = "demo_session",
) -> Dict[str, Any]:
    """
    Get all available versions for a specific artifact.

    Returns a list of version numbers available for the specified artifact.
    This is useful for understanding the history of changes to a file.

    Args:
        filename: Name of the artifact file
        artifact_service: S3ArtifactService instance
        app_name: Application identifier (default: "s3_demo_app")
        user_id: User identifier (default: "demo_user")
        session_id: Session identifier (default: "demo_session")

    Returns:
        dict: Result with status and version information
            - status: "success" or "error"
            - versions: List of version numbers (if successful)
            - count: Number of versions found (if successful)
            - latest: Latest version number (if successful)
            - message: Human-readable description
    """
    try:
        logger.info(f"Getting versions for artifact: {filename}")

        # Validate filename
        if not filename or not isinstance(filename, str):
            return {"status": "error", "message": "Filename must be a non-empty string"}

        # List all versions
        versions = await artifact_service.list_versions(
            app_name=app_name, user_id=user_id, session_id=session_id, filename=filename
        )

        latest_version = max(versions) if versions else None

        logger.info(f"Found {len(versions)} versions for {filename}")
        return {
            "status": "success",
            "versions": versions,
            "count": len(versions),
            "latest": latest_version,
            "message": f"Found {len(versions)} versions for {filename}",
        }

    except S3ArtifactError as e:
        logger.error(f"S3 error getting versions for {filename}: {e}")
        return {"status": "error", "message": f"S3 operation failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error getting versions for {filename}: {e}")
        return {"status": "error", "message": f"Failed to get versions: {str(e)}"}
