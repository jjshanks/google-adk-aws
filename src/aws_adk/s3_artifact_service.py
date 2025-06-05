"""S3 Artifact Service implementation for Google ADK."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import boto3
import botocore.exceptions
from google.adk.artifacts import BaseArtifactService
from google.genai import types

logger = logging.getLogger(__name__)


class S3ArtifactError(Exception):
    """Base exception for S3 artifact operations."""

    pass


class S3ConnectionError(S3ArtifactError):
    """Raised when S3 connection fails."""

    pass


class S3PermissionError(S3ArtifactError):
    """Raised when S3 permissions are insufficient."""

    pass


class S3ArtifactService(BaseArtifactService):
    """S3-based implementation of ADK's BaseArtifactService.

    Provides artifact storage and retrieval using Amazon S3 or S3-compatible
    services. Supports multiple authentication methods and automatic versioning.

    Attributes:
        bucket_name: S3 bucket name for artifact storage
        region_name: AWS region for S3 operations
        s3_client: Boto3 S3 client instance
    """

    def __init__(
        self,
        bucket_name: str,
        region_name: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        endpoint_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize S3ArtifactService.

        Args:
            bucket_name: Name of S3 bucket to use for storage
            region_name: AWS region name (default: us-east-1)
            aws_access_key_id: AWS access key ID (optional, uses IAM if not provided)
            aws_secret_access_key: AWS secret access key (optional)
            aws_session_token: AWS session token for temporary credentials (optional)
            endpoint_url: Custom S3 endpoint URL for S3-compatible services (optional)
            **kwargs: Additional arguments passed to boto3.Session

        Raises:
            S3ConnectionError: If S3 client initialization fails
            S3PermissionError: If bucket access is denied
        """
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.endpoint_url = endpoint_url

        # Store credentials for session creation
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.session_kwargs = kwargs

        # Initialize S3 client
        self.s3_client = self._create_s3_client()

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="s3-artifact"
        )

        # Verify bucket access
        self._verify_bucket_access()

    def _create_s3_client(self) -> Any:
        """Create and configure S3 client with authentication.

        Returns:
            Configured boto3 S3 client

        Raises:
            S3ConnectionError: If client creation fails
        """
        try:
            # Create session with provided credentials
            session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                region_name=self.region_name,
                **self.session_kwargs,
            )

            # Configure client options
            client_config: dict[str, str] = {}
            if self.endpoint_url:
                client_config["endpoint_url"] = self.endpoint_url

            return session.client("s3", **client_config)

        except botocore.exceptions.ClientError as e:
            raise S3ConnectionError(f"Failed to create S3 client: {e}") from e
        except Exception as e:
            raise S3ConnectionError(f"Unexpected error creating S3 client: {e}") from e

    def _verify_bucket_access(self) -> None:
        """Verify that the bucket exists and is accessible.

        Raises:
            S3PermissionError: If bucket access is denied
            S3ArtifactError: If bucket doesn't exist or other error
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully verified access to bucket: {self.bucket_name}")
        except botocore.exceptions.ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "403":
                raise S3PermissionError(
                    f"Access denied to bucket {self.bucket_name}. "
                    "Check IAM permissions."
                ) from e
            elif error_code == "404":
                raise S3ArtifactError(
                    f"Bucket {self.bucket_name} does not exist"
                ) from e
            else:
                raise S3ArtifactError(f"Failed to access bucket: {e}") from e
        except Exception as e:
            raise S3ArtifactError(f"Unexpected error accessing bucket: {e}") from e

    def _file_has_user_namespace(self, filename: str) -> bool:
        """Check if filename has user namespace prefix.

        Args:
            filename: The filename to check

        Returns:
            True if filename starts with "user:", False otherwise
        """
        return filename.startswith("user:")

    def _get_object_key(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: int,
    ) -> str:
        """Construct S3 object key for artifact.

        Args:
            app_name: Application name
            user_id: User identifier
            session_id: Session identifier
            filename: Artifact filename
            version: Artifact version number

        Returns:
            S3 object key string
        """
        if self._file_has_user_namespace(filename):
            clean_filename = filename[5:] if filename.startswith("user:") else filename
            return f"{app_name}/{user_id}/user/{clean_filename}/{version}"
        return f"{app_name}/{user_id}/{session_id}/{filename}/{version}"

    async def save_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        artifact: types.Part,
    ) -> int:
        """Save artifact to S3 storage.

        Args:
            app_name: Application name
            user_id: User identifier
            session_id: Session identifier
            filename: Artifact filename
            artifact: Artifact data as types.Part

        Returns:
            Version number of saved artifact (starting from 0)

        Raises:
            S3ArtifactError: If save operation fails
        """
        try:
            # Get existing versions to determine next version
            existing_versions = await self.list_versions(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
            )
            version = 0 if not existing_versions else max(existing_versions) + 1

            # Construct object key
            object_key = self._get_object_key(
                app_name, user_id, session_id, filename, version
            )

            # Prepare metadata
            metadata = {
                "app-name": app_name,
                "user-id": user_id,
                "session-id": session_id,
                "filename": filename,
                "version": str(version),
            }

            # Upload to S3 asynchronously
            await asyncio.get_running_loop().run_in_executor(
                self._executor,
                lambda: self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_key,
                    Body=artifact.inline_data.data,
                    ContentType=artifact.inline_data.mime_type,
                    Metadata=metadata,
                ),
            )

            logger.info(f"Saved artifact {filename} version {version} to {object_key}")
            return version

        except Exception as e:
            logger.error(f"Failed to save artifact {filename}: {e}")
            raise S3ArtifactError(f"Failed to save artifact: {e}") from e

    async def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: int | None = None,
    ) -> types.Part | None:
        """Load artifact from S3 storage.

        Args:
            app_name: Application name
            user_id: User identifier
            session_id: Session identifier
            filename: Artifact filename
            version: Specific version to load (latest if None)

        Returns:
            Artifact as types.Part or None if not found

        Raises:
            S3ArtifactError: If load operation fails
        """
        try:
            # Determine version if not specified
            if version is None:
                existing_versions = await self.list_versions(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                    filename=filename,
                )
                if not existing_versions:
                    return None
                version = max(existing_versions)

            # Construct object key
            object_key = self._get_object_key(
                app_name, user_id, session_id, filename, version
            )

            # Download from S3 asynchronously
            response = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                lambda: self.s3_client.get_object(
                    Bucket=self.bucket_name, Key=object_key
                ),
            )

            # Read data and create types.Part
            data = response["Body"].read()
            content_type = response["ContentType"]

            logger.info(
                f"Loaded artifact {filename} version {version} from {object_key}"
            )
            return types.Part.from_bytes(data=data, mime_type=content_type)

        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.debug(f"Artifact not found: {filename} version {version}")
                return None
            logger.error(f"Failed to load artifact {filename}: {e}")
            raise S3ArtifactError(f"Failed to load artifact: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error loading artifact {filename}: {e}")
            raise S3ArtifactError(f"Failed to load artifact: {e}") from e

    async def list_artifact_keys(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> list[str]:
        """List all artifact filenames for a session.

        Args:
            app_name: Application name
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Sorted list of artifact filenames

        Raises:
            S3ArtifactError: If list operation fails
        """
        try:
            filenames: set[str] = set()

            # List session-scoped artifacts
            session_prefix = f"{app_name}/{user_id}/{session_id}/"
            await self._list_filenames_with_prefix(session_prefix, filenames)

            # List user-scoped artifacts
            user_prefix = f"{app_name}/{user_id}/user/"
            await self._list_filenames_with_prefix(user_prefix, filenames)

            result = sorted(list(filenames))
            logger.debug(f"Listed {len(result)} artifacts for session {session_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to list artifact keys: {e}")
            raise S3ArtifactError(f"Failed to list artifacts: {e}") from e

    async def _list_filenames_with_prefix(
        self, prefix: str, filenames: set[str]
    ) -> None:
        """Helper to list filenames with given prefix.

        Args:
            prefix: S3 key prefix to search
            filenames: Set to add found filenames to
        """

        def _list_objects() -> None:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    # Extract filename: app/user/session/filename/version
                    key_parts = obj["Key"].split("/")
                    if len(key_parts) >= 5:
                        filename = key_parts[-2]  # Second to last is filename
                        filenames.add(filename)

        await asyncio.get_running_loop().run_in_executor(self._executor, _list_objects)

    async def delete_artifact(
        self, *, app_name: str, user_id: str, session_id: str, filename: str
    ) -> None:
        """Delete all versions of an artifact.

        Args:
            app_name: Application name
            user_id: User identifier
            session_id: Session identifier
            filename: Artifact filename to delete

        Raises:
            S3ArtifactError: If delete operation fails
        """
        try:
            # Get all versions to delete
            versions = await self.list_versions(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
            )

            if not versions:
                logger.debug(f"No versions found for artifact {filename}")
                return

            # Delete all versions
            def _delete_versions() -> None:
                for version in versions:
                    object_key = self._get_object_key(
                        app_name, user_id, session_id, filename, version
                    )
                    self.s3_client.delete_object(
                        Bucket=self.bucket_name, Key=object_key
                    )

            await asyncio.get_running_loop().run_in_executor(
                self._executor, _delete_versions
            )

            logger.info(f"Deleted {len(versions)} versions of artifact {filename}")

        except Exception as e:
            logger.error(f"Failed to delete artifact {filename}: {e}")
            raise S3ArtifactError(f"Failed to delete artifact: {e}") from e

    async def list_versions(
        self, *, app_name: str, user_id: str, session_id: str, filename: str
    ) -> list[int]:
        """List all versions of an artifact.

        Args:
            app_name: Application name
            user_id: User identifier
            session_id: Session identifier
            filename: Artifact filename

        Returns:
            Sorted list of version numbers

        Raises:
            S3ArtifactError: If list operation fails
        """
        try:

            def _list_versions() -> list[int]:
                # Use empty string for version to get prefix
                prefix = self._get_object_key(
                    app_name, user_id, session_id, filename, 0
                ).rsplit("/", 1)[
                    0
                ]  # Remove last path segment robustly

                paginator = self.s3_client.get_paginator("list_objects_v2")
                versions = []

                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        # Extract version from key: .../filename/version
                        key_parts = obj["Key"].split("/")
                        if len(key_parts) >= 5:
                            try:
                                version = int(key_parts[-1])
                                versions.append(version)
                            except ValueError:
                                # Skip invalid version numbers
                                continue

                return sorted(versions)

            versions = await asyncio.get_running_loop().run_in_executor(
                self._executor, _list_versions
            )

            logger.debug(f"Found {len(versions)} versions for artifact {filename}")
            return versions

        except Exception as e:
            logger.error(f"Failed to list versions for {filename}: {e}")
            raise S3ArtifactError(f"Failed to list versions: {e}") from e

    def __del__(self) -> None:
        """Cleanup resources on object destruction."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
