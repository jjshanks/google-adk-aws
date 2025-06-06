"""S3 Artifact Service implementation for Google ADK."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, cast

import botocore.exceptions
from google.adk.artifacts import BaseArtifactService
from google.genai import types

from .batch_operations import MultipartUploadManager, S3BatchOperations
from .connection_pool import get_connection_pool
from .exceptions import (
    S3ArtifactError,
    S3ArtifactNotFoundError,
    S3BucketError,
    S3ConnectionError,
    S3ObjectError,
    S3PermissionError,
    S3ThrottleError,
)
from .retry_handler import CircuitBreaker, RetryConfig, is_throttle_error, with_retry
from .security import AccessControlManager, EncryptionManager, S3SecurityManager

logger = logging.getLogger(__name__)


class S3ArtifactService(BaseArtifactService):
    """Enhanced S3-based implementation of ADK's BaseArtifactService.

    Provides artifact storage and retrieval using Amazon S3 or S3-compatible
    services. Supports multiple authentication methods, automatic versioning,
    advanced error handling, performance optimization, and security features.

    Attributes:
        bucket_name: S3 bucket name for artifact storage
        region_name: AWS region for S3 operations
        s3_client: Boto3 S3 client instance
        retry_config: Configuration for retry behavior
        security_manager: Security features manager
        access_control: Access control manager
        batch_operations: Batch operations manager
        multipart_manager: Multipart upload manager
    """

    def __init__(
        self,
        bucket_name: str,
        region_name: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        endpoint_url: str | None = None,
        retry_config: Optional[RetryConfig] = None,
        enable_encryption: bool = False,
        encryption_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Enhanced S3ArtifactService.

        Args:
            bucket_name: Name of S3 bucket to use for storage
            region_name: AWS region name (default: us-east-1)
            aws_access_key_id: AWS access key ID (optional, uses IAM if not provided)
            aws_secret_access_key: AWS secret access key (optional)
            aws_session_token: AWS session token for temporary credentials (optional)
            endpoint_url: Custom S3 endpoint URL for S3-compatible services (optional)
            retry_config: Configuration for retry behavior (optional)
            enable_encryption: Enable client-side encryption (default: False)
            encryption_key: Custom encryption key (optional, generates if needed)
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

        # Initialize retry configuration
        self.retry_config = retry_config or RetryConfig()

        # Initialize circuit breakers for different operation types
        self.read_circuit_breaker = CircuitBreaker(
            failure_threshold=5, timeout=30.0, expected_exception=S3ConnectionError
        )

        self.write_circuit_breaker = CircuitBreaker(
            failure_threshold=3, timeout=60.0, expected_exception=S3ConnectionError
        )

        # Use connection pool for optimized S3 client
        connection_pool = get_connection_pool()
        self.s3_client = connection_pool.get_client(
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            endpoint_url=self.endpoint_url,
        )

        # Initialize enhanced features
        self.security_manager = S3SecurityManager(self.s3_client, self.bucket_name)
        self.access_control = AccessControlManager()
        self.batch_operations = S3BatchOperations(self.s3_client, self.bucket_name)
        self.multipart_manager = MultipartUploadManager(
            self.s3_client, self.bucket_name
        )

        # Initialize encryption if enabled
        self.enable_encryption = enable_encryption
        self.encryption_manager: Optional[EncryptionManager]
        if enable_encryption:
            self.encryption_manager = EncryptionManager(encryption_key)
        else:
            self.encryption_manager = None

        # Thread pool for async operations (kept for backwards compatibility)
        self._executor = ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="s3-artifact"
        )

        # Verify bucket access
        self._verify_bucket_access()

    def _handle_s3_error(self, error: Exception, operation: str) -> None:
        """Enhanced error handling with specific exception mapping."""
        if isinstance(error, botocore.exceptions.ClientError):
            error_code = error.response["Error"]["Code"]
            error_message = error.response["Error"]["Message"]

            if error_code == "NoSuchBucket":
                raise S3BucketError(
                    f"Bucket {self.bucket_name} does not exist"
                ) from error
            elif error_code in ["AccessDenied", "Forbidden"]:
                raise S3PermissionError(f"Access denied for {operation}") from error
            elif error_code == "NoSuchKey":
                raise S3ArtifactNotFoundError("Artifact not found") from error
            elif is_throttle_error(error):
                raise S3ThrottleError(f"S3 throttling during {operation}") from error
            elif error_code in ["InternalError", "ServiceUnavailable"]:
                raise S3ConnectionError(
                    f"S3 service error during {operation}"
                ) from error
            else:
                raise S3ObjectError(
                    f"S3 {operation} failed: {error_message}"
                ) from error
        else:
            raise S3ArtifactError(
                f"Unexpected error during {operation}: {error}"
            ) from error

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
        """Save artifact to S3 storage with enhanced features.

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

        @self.write_circuit_breaker
        @with_retry(self.retry_config)  # type: ignore[misc]
        async def _save_with_protection() -> int:
            try:
                # Get existing versions to determine next version
                existing_versions = await self.list_versions(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                    filename=filename,
                )
                version = 0 if not existing_versions else max(existing_versions) + 1

                # Use security manager for secure object key generation
                object_key = self.security_manager.generate_secure_object_key(
                    app_name, user_id, session_id, filename, version
                )

                # Validate object key for security
                if not self.security_manager.validate_object_key(object_key):
                    raise S3ArtifactError(f"Invalid object key: {object_key}")

                # Prepare artifact data
                if artifact.inline_data is None:
                    raise S3ArtifactError("Artifact has no inline data")
                artifact_data = artifact.inline_data.data
                if artifact_data is None:
                    raise S3ArtifactError("Artifact data is None")
                content_type = artifact.inline_data.mime_type
                if content_type is None:
                    raise S3ArtifactError("Artifact content type is None")

                # Apply encryption if enabled
                if self.enable_encryption and self.encryption_manager:
                    (
                        artifact_data,
                        encryption_metadata,
                    ) = self.encryption_manager.encrypt_content(artifact_data)
                else:
                    encryption_metadata = {}

                # Calculate content hash for integrity verification
                content_hash = self.security_manager.calculate_content_hash(
                    artifact_data
                )

                # Prepare metadata
                metadata = {
                    "app-name": app_name,
                    "user-id": user_id,
                    "session-id": session_id,
                    "filename": filename,
                    "version": str(version),
                    "content-hash": content_hash,
                    **encryption_metadata,
                }

                # Use multipart upload for large artifacts or regular upload
                # for smaller ones
                if len(artifact_data) > self.multipart_manager.multipart_threshold:
                    await self.multipart_manager.upload_large_artifact(
                        object_key=object_key,
                        data=artifact_data,
                        content_type=content_type,
                        metadata=metadata,
                    )
                else:
                    # Regular upload for smaller artifacts
                    await asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        lambda: self.s3_client.put_object(
                            Bucket=self.bucket_name,
                            Key=object_key,
                            Body=artifact_data,
                            ContentType=content_type,
                            Metadata=metadata,
                        ),
                    )

                logger.info(
                    f"Saved artifact {filename} version {version} to {object_key}"
                )
                return version

            except Exception as e:
                logger.error(
                    f"Save artifact failed - App: {app_name}, User: {user_id}, "
                    f"Session: {session_id}, File: {filename}, Error: {e}"
                )
                self._handle_s3_error(e, "save_artifact")
                raise  # This should never be reached due to _handle_s3_error raising

        result = await _save_with_protection()
        return int(result)

    async def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: int | None = None,
    ) -> types.Part | None:
        """Load artifact from S3 storage with enhanced features.

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

        @self.read_circuit_breaker
        @with_retry(self.retry_config)  # type: ignore[misc]
        async def _load_with_protection() -> types.Part | None:
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
                    actual_version = max(existing_versions)
                else:
                    actual_version = version

                # Use security manager for secure object key generation
                object_key = self.security_manager.generate_secure_object_key(
                    app_name, user_id, session_id, filename, actual_version
                )

                # Download from S3 asynchronously
                response = await asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    lambda: self.s3_client.get_object(
                        Bucket=self.bucket_name, Key=object_key
                    ),
                )

                # Read data and metadata
                data = response["Body"].read()
                content_type = response["ContentType"]
                metadata = response.get("Metadata", {})

                # Verify content integrity if hash is available
                stored_hash = metadata.get("content-hash")
                if stored_hash:
                    if not self.security_manager.verify_content_integrity(
                        data, stored_hash
                    ):
                        raise S3ArtifactError(
                            f"Content integrity verification failed for {filename}"
                        )

                # Apply decryption if needed
                if self.enable_encryption and self.encryption_manager:
                    data = self.encryption_manager.decrypt_content(data, metadata)

                logger.info(
                    f"Loaded artifact {filename} version {actual_version} "
                    f"from {object_key}"
                )
                return types.Part.from_bytes(data=data, mime_type=content_type)

            except S3ArtifactNotFoundError:
                # Not found is not an error condition for load
                logger.debug(f"Artifact not found: {filename} version {actual_version}")
                return None
            except Exception as e:
                logger.error(f"Failed to load artifact {filename}: {e}")
                self._handle_s3_error(e, "load_artifact")
                raise  # This should never be reached due to _handle_s3_error raising

        result = await _load_with_protection()
        return cast(types.Part | None, result)

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

    async def get_security_status(self) -> dict:
        """Get bucket security status and recommendations."""
        return self.security_manager.validate_bucket_security()

    async def generate_presigned_url(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int] = None,
        operation: str = "get_object",
        expiration: int = 3600,
    ) -> str:
        """Generate presigned URL for artifact access."""
        # Determine version if not specified
        if version is None:
            existing_versions = await self.list_versions(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
            )
            if not existing_versions:
                raise S3ArtifactNotFoundError(f"No versions found for {filename}")
            version = max(existing_versions)

        # Generate secure object key
        object_key = self.security_manager.generate_secure_object_key(
            app_name, user_id, session_id, filename, version
        )

        return self.access_control.generate_presigned_url(
            self.s3_client, self.bucket_name, object_key, operation, expiration
        )

    async def batch_delete_artifacts(
        self, app_name: str, user_id: str, session_id: str, filenames: list[str]
    ) -> dict:
        """Delete multiple artifacts using batch operations."""
        object_keys = []

        for filename in filenames:
            versions = await self.list_versions(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
            )

            for version in versions:
                object_key = self.security_manager.generate_secure_object_key(
                    app_name, user_id, session_id, filename, version
                )
                object_keys.append(object_key)

        return await self.batch_operations.batch_delete(object_keys)

    def get_connection_stats(self) -> dict:
        """Get connection pool statistics."""
        connection_pool = get_connection_pool()
        return connection_pool.get_stats()

    def __del__(self) -> None:
        """Cleanup resources on object destruction."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
