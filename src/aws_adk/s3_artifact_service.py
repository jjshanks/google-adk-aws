"""Enhanced S3ArtifactService with comprehensive error handling and edge cases."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, cast

import boto3
from botocore.exceptions import ClientError
from google.adk.artifacts import BaseArtifactService
from google.genai import types

from .batch_operations import MultipartUploadManager, S3BatchOperations
from .connection_pool import get_connection_pool
from .edge_case_handlers import (
    get_concurrency_manager,
    get_corruption_detector,
    get_large_file_handler,
    get_network_failure_handler,
)
from .exceptions import (
    S3ConnectionError,
    S3CorruptionError,
    S3ThrottleError,
    S3ValidationError,
    map_boto3_error,
)
from .retry_handler import CircuitBreaker, RetryConfig, with_retry
from .security import AccessControlManager, S3SecurityManager
from .validation import get_validator

logger = logging.getLogger(__name__)


class S3ArtifactService(BaseArtifactService):
    """Production-ready S3 artifact service with comprehensive error handling."""

    def __init__(
        self,
        bucket_name: str,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_validation: bool = True,
        enable_security_checks: bool = True,
        enable_integrity_checks: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize S3ArtifactService with comprehensive error handling."""

        # Store configuration
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.endpoint_url = endpoint_url

        # Feature flags
        self.enable_validation = enable_validation
        self.enable_security_checks = enable_security_checks
        self.enable_integrity_checks = enable_integrity_checks

        # Initialize components
        self.connection_pool = get_connection_pool()
        self.validator = get_validator(strict_mode=enable_validation)
        self.concurrency_manager = get_concurrency_manager()
        self.large_file_handler = get_large_file_handler()
        self.corruption_detector = get_corruption_detector()
        self.network_failure_handler = get_network_failure_handler()

        # Retry configuration
        self.retry_config = retry_config or RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=60.0,
            backoff_strategy="exponential",
        )

        # Circuit breakers for different operation types
        self.read_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=30.0,
            expected_exception=(S3ConnectionError, S3ThrottleError),
        )

        self.write_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=60.0,
            expected_exception=(S3ConnectionError, S3ThrottleError),
        )

        # Initialize S3 client and components
        try:
            self.s3_client = self._create_s3_client()
            self.batch_operations = S3BatchOperations(self.s3_client, self.bucket_name)
            self.multipart_manager = MultipartUploadManager(
                self.s3_client, self.bucket_name
            )
            self.security_manager = S3SecurityManager(self.s3_client, self.bucket_name)
            self.access_control = AccessControlManager()

            # Validate bucket access
            self._validate_bucket_access()

        except Exception as e:
            mapped_error = map_boto3_error(e, "initialization")
            logger.error(f"Failed to initialize S3ArtifactService: {mapped_error}")
            raise mapped_error

    def _create_s3_client(self) -> boto3.client:
        """Create optimized S3 client with error handling."""
        try:
            return self.connection_pool.get_client(
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                endpoint_url=self.endpoint_url,
            )
        except Exception as e:
            raise map_boto3_error(e, "create_client")

    def _validate_bucket_access(self) -> None:
        """Validate bucket exists and is accessible."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully validated access to bucket: {self.bucket_name}")

            # Security validation if enabled
            if self.enable_security_checks:
                security_status = self.security_manager.validate_bucket_security()
                if security_status["recommendations"]:
                    logger.warning(
                        f"Security recommendations for bucket {self.bucket_name}: "
                        f"{security_status['recommendations']}"
                    )

        except Exception as e:
            raise map_boto3_error(e, "validate_bucket_access")

    @asynccontextmanager
    async def _operation_context(
        self,
        operation_type: str,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
    ) -> Any:
        """Context manager for artifact operations with comprehensive error handling."""

        operation_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Input validation
            if self.enable_validation:
                self.validator.validate_artifact_params(
                    app_name, user_id, session_id, filename
                )

                # Sanitize inputs
                (
                    app_name,
                    user_id,
                    session_id,
                    filename,
                ) = self.validator.sanitize_inputs(
                    app_name, user_id, session_id, filename
                )

            # Acquire concurrency lock
            await self.concurrency_manager.acquire_operation_lock(
                app_name, user_id, session_id, filename, operation_type, operation_id
            )

            logger.debug(
                f"Started {operation_type} operation {operation_id} for "
                f"{app_name}/{user_id}/{session_id}/{filename}"
            )

            yield {
                "operation_id": operation_id,
                "sanitized_params": (app_name, user_id, session_id, filename),
            }

        except Exception as e:
            operation_time = time.time() - start_time
            logger.error(
                f"Operation {operation_type} failed after {operation_time:.2f}s: {e}"
            )
            raise

        finally:
            # Release concurrency lock
            try:
                await self.concurrency_manager.release_operation_lock(
                    app_name, user_id, session_id, filename, operation_id
                )
            except Exception as e:
                logger.warning(f"Failed to release operation lock: {e}")

            operation_time = time.time() - start_time
            logger.debug(
                f"Completed {operation_type} operation {operation_id} "
                f"in {operation_time:.2f}s"
            )

    async def save_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        artifact: types.Part,
    ) -> int:
        """Save artifact with comprehensive error handling and edge case management."""

        async def _save() -> int:
            async with self._operation_context(
                "save", app_name, user_id, session_id, filename
            ) as context:
                (
                    sanitized_app,
                    sanitized_user,
                    sanitized_session,
                    sanitized_file,
                ) = context["sanitized_params"]

                @self.write_circuit_breaker
                async def _save_with_protection() -> int:
                    return await self._save_artifact_impl(
                        sanitized_app,
                        sanitized_user,
                        sanitized_session,
                        sanitized_file,
                        artifact,
                    )

                result = await _save_with_protection()
                return cast(int, result)

        result = await with_retry(self.retry_config)(_save)()
        return cast(int, result)

    async def _save_artifact_impl(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        artifact: types.Part,
    ) -> int:
        """Implementation of save artifact with all error handling."""

        try:
            # Content validation
            if artifact.inline_data is None or artifact.inline_data.data is None:
                raise S3ValidationError(
                    message="Artifact has no inline data", error_code="NoInlineData"
                )

            content = artifact.inline_data.data
            mime_type = artifact.inline_data.mime_type or "application/octet-stream"

            if self.enable_validation:
                self.validator.validate_artifact_content(content, mime_type)

            # Large file handling
            content_size = len(content)
            await self.large_file_handler.validate_large_file_operation(
                content_size, "upload"
            )

            # Get existing versions to determine next version
            versions = await self.list_versions(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
            )
            version = 0 if not versions else max(versions) + 1

            # Generate secure object key
            object_key = self.security_manager.generate_secure_object_key(
                app_name, user_id, session_id, filename, version
            )

            # Validate object key
            if self.enable_security_checks:
                if not self.security_manager.validate_object_key(object_key):
                    raise S3ValidationError(
                        message="Generated object key failed security validation",
                        error_code="InvalidObjectKey",
                        context={"object_key": object_key},
                    )

            # Calculate content hash for integrity
            content_hash = None
            if self.enable_integrity_checks:
                content_hash = self.security_manager.calculate_content_hash(content)

            # Prepare metadata
            metadata = {
                "app-name": app_name,
                "user-id": user_id,
                "session-id": session_id,
                "filename": filename,
                "version": str(version),
                "upload-timestamp": str(int(time.time())),
                "content-size": str(content_size),
            }

            if content_hash:
                metadata["content-hash"] = content_hash

            # Handle large file upload
            if self.large_file_handler.should_use_multipart(content_size):
                await self.multipart_manager.upload_large_artifact(
                    object_key, content, mime_type, metadata
                )
            else:
                # Standard upload
                await self.connection_pool.execute_async(
                    self.s3_client.put_object,
                    Bucket=self.bucket_name,
                    Key=object_key,
                    Body=content,
                    ContentType=mime_type,
                    Metadata=metadata,
                )

            logger.info(
                f"Successfully saved artifact {filename} version {version} "
                f"({content_size} bytes) for {app_name}/{user_id}"
            )

            return version

        except Exception as e:
            mapped_error = map_boto3_error(e, "save_artifact")
            logger.error(f"Failed to save artifact {filename}: {mapped_error}")
            raise mapped_error

    async def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int] = None,
    ) -> Optional[types.Part]:
        """Load artifact with comprehensive error handling."""

        async def _load() -> Optional[types.Part]:
            async with self._operation_context(
                "load", app_name, user_id, session_id, filename
            ) as context:
                (
                    sanitized_app,
                    sanitized_user,
                    sanitized_session,
                    sanitized_file,
                ) = context["sanitized_params"]

                @self.read_circuit_breaker
                async def _load_with_protection() -> Optional[types.Part]:
                    return await self._load_artifact_impl(
                        sanitized_app,
                        sanitized_user,
                        sanitized_session,
                        sanitized_file,
                        version,
                    )

                result = await _load_with_protection()
                return cast(Optional[types.Part], result)

        result = await with_retry(self.retry_config)(_load)()
        return cast(Optional[types.Part], result)

    async def _load_artifact_impl(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int],
    ) -> Optional[types.Part]:
        """Implementation of load artifact with error handling."""

        try:
            # Determine version if not specified
            if version is None:
                versions = await self.list_versions(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                    filename=filename,
                )
                if not versions:
                    return None
                version = max(versions)

            # Version validation
            if self.enable_validation and version < 0:
                raise S3ValidationError(
                    message="Version must be non-negative",
                    error_code="InvalidVersion",
                    context={"version": version},
                )

            # Generate object key
            object_key = self.security_manager.generate_secure_object_key(
                app_name, user_id, session_id, filename, version
            )

            # Download from S3
            response = await self.connection_pool.execute_async(
                self.s3_client.get_object, Bucket=self.bucket_name, Key=object_key
            )

            # Read content
            content = response["Body"].read()
            content_type = response.get("ContentType", "application/octet-stream")
            metadata = response.get("Metadata", {})

            # Integrity verification
            if self.enable_integrity_checks and "content-hash" in metadata:
                expected_hash = metadata["content-hash"]
                if not self.security_manager.verify_content_integrity(
                    content, expected_hash
                ):
                    raise S3CorruptionError(
                        message="Content integrity verification failed",
                        error_code="IntegrityCheckFailed",
                        context={
                            "expected_hash": expected_hash,
                            "object_key": object_key,
                        },
                    )

            # Create and return types.Part
            # Decode bytes to text for text content types
            if content_type.startswith("text/"):
                text_content = content.decode("utf-8")
                part = types.Part(text=text_content)
            else:
                part = types.Part.from_bytes(data=content, mime_type=content_type)

            logger.debug(
                f"Successfully loaded artifact {filename} version {version} "
                f"({len(content)} bytes)"
            )

            return part

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None  # Artifact not found
            else:
                mapped_error = map_boto3_error(e, "load_artifact")
                logger.error(f"Failed to load artifact {filename}: {mapped_error}")
                raise mapped_error

        except Exception as e:
            mapped_error = map_boto3_error(e, "load_artifact")
            logger.error(f"Failed to load artifact {filename}: {mapped_error}")
            raise mapped_error

    async def list_artifact_keys(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> list[str]:
        """List all artifact filenames for a session."""
        try:
            # Input validation
            if self.enable_validation:
                self.validator.validate_artifact_params(
                    app_name, user_id, session_id, "dummy"
                )

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
            mapped_error = map_boto3_error(e, "list_artifact_keys")
            logger.error(f"Failed to list artifact keys: {mapped_error}")
            raise mapped_error

    async def _list_filenames_with_prefix(
        self, prefix: str, filenames: set[str]
    ) -> None:
        """Helper to list filenames with given prefix."""

        def _list_objects() -> None:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    # Extract filename: app/user/session/filename/version
                    key_parts = obj["Key"].split("/")
                    if len(key_parts) >= 5:
                        filename = key_parts[-2]  # Second to last is filename
                        filenames.add(filename)

        await self.connection_pool.execute_async(_list_objects)

    async def delete_artifact(
        self, *, app_name: str, user_id: str, session_id: str, filename: str
    ) -> None:
        """Delete all versions of an artifact."""
        async with self._operation_context(
            "delete", app_name, user_id, session_id, filename
        ) as context:
            app_name, user_id, session_id, filename = context["sanitized_params"]

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
                        object_key = self.security_manager.generate_secure_object_key(
                            app_name, user_id, session_id, filename, version
                        )
                        self.s3_client.delete_object(
                            Bucket=self.bucket_name, Key=object_key
                        )

                await self.connection_pool.execute_async(_delete_versions)

                logger.info(f"Deleted {len(versions)} versions of artifact {filename}")

            except Exception as e:
                mapped_error = map_boto3_error(e, "delete_artifact")
                logger.error(f"Failed to delete artifact {filename}: {mapped_error}")
                raise mapped_error

    async def list_versions(
        self, *, app_name: str, user_id: str, session_id: str, filename: str
    ) -> list[int]:
        """List all versions of an artifact."""
        try:
            # Input validation
            if self.enable_validation:
                self.validator.validate_artifact_params(
                    app_name, user_id, session_id, filename
                )

            def _list_versions() -> list[int]:
                # Use empty string for version to get prefix
                prefix = (
                    self.security_manager.generate_secure_object_key(
                        app_name, user_id, session_id, filename, 0
                    )
                    .rsplit("/", 1)[0]
                    .rstrip("/")
                )

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

            versions = cast(
                list[int], await self.connection_pool.execute_async(_list_versions)
            )

            logger.debug(f"Found {len(versions)} versions for artifact {filename}")
            return versions

        except Exception as e:
            mapped_error = map_boto3_error(e, "list_versions")
            logger.error(f"Failed to list versions for {filename}: {mapped_error}")
            raise mapped_error

    async def get_service_health(self) -> Dict[str, Any]:
        """Get comprehensive service health information."""
        return {
            "bucket_name": self.bucket_name,
            "region_name": self.region_name,
            "connection_pool_stats": self.connection_pool.get_stats(),
            "circuit_breaker_stats": {
                "read": self.read_circuit_breaker.get_stats(),
                "write": self.write_circuit_breaker.get_stats(),
            },
            "network_health": self.network_failure_handler.get_network_health_stats(),
            "feature_flags": {
                "validation": self.enable_validation,
                "security_checks": self.enable_security_checks,
                "integrity_checks": self.enable_integrity_checks,
            },
        }

    async def cleanup_resources(self) -> None:
        """Cleanup all resources and connections."""
        try:
            if hasattr(self, "connection_pool"):
                self.connection_pool.close()
            logger.info("S3ArtifactService resources cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")

    # Legacy methods for backward compatibility with the original implementation
    def _file_has_user_namespace(self, filename: str) -> bool:
        """Check if filename has user namespace prefix."""
        return filename.startswith("user:")

    def _get_object_key(
        self, app_name: str, user_id: str, session_id: str, filename: str, version: int
    ) -> str:
        """Construct S3 object key for artifact."""
        return self.security_manager.generate_secure_object_key(
            app_name, user_id, session_id, filename, version
        )
