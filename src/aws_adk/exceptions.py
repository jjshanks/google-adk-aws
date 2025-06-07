"""Enhanced exception hierarchy for comprehensive error handling."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class S3ArtifactError(Exception):
    """Base exception for S3 artifact operations with enhanced context."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.operation = operation
        self.context = context or {}
        self.cause = cause

        # Log error with full context
        logger.error(
            f"S3ArtifactError: {message} | Operation: {operation} | "
            f"Code: {error_code} | Context: {self.context}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error": self.__class__.__name__,
            "message": str(self),
            "error_code": self.error_code,
            "operation": self.operation,
            "context": self.context,
        }


class S3ConnectionError(S3ArtifactError):
    """Network and connection-related errors."""

    pass


class S3PermissionError(S3ArtifactError):
    """Access control and permission errors."""

    pass


class S3BucketError(S3ArtifactError):
    """Bucket-related operation errors."""

    pass


class S3ObjectError(S3ArtifactError):
    """Object-related operation errors."""

    pass


class S3ThrottleError(S3ArtifactError):
    """Rate limiting and throttling errors."""

    pass


class S3ArtifactNotFoundError(S3ArtifactError):
    """Artifact not found errors."""

    pass


class S3ArtifactVersionError(S3ArtifactError):
    """Version management errors."""

    pass


class S3ValidationError(S3ArtifactError):
    """Input validation and sanitization errors."""

    pass


class S3StorageQuotaError(S3ArtifactError):
    """Storage quota and capacity errors."""

    pass


class S3ConcurrencyError(S3ArtifactError):
    """Concurrent operation conflicts."""

    pass


class S3CorruptionError(S3ArtifactError):
    """Data corruption and integrity errors."""

    pass


# Error mapping for boto3 ClientError codes
BOTO3_ERROR_MAPPING = {
    "NoSuchBucket": (S3BucketError, "Bucket does not exist"),
    "BucketNotEmpty": (S3BucketError, "Bucket is not empty"),
    "NoSuchKey": (S3ArtifactNotFoundError, "Object not found"),
    "AccessDenied": (S3PermissionError, "Access denied"),
    "Forbidden": (S3PermissionError, "Operation forbidden"),
    "InvalidRequest": (S3ValidationError, "Invalid request"),
    "InvalidArgument": (S3ValidationError, "Invalid argument"),
    "Throttling": (S3ThrottleError, "Request throttled"),
    "RequestLimitExceeded": (S3ThrottleError, "Request limit exceeded"),
    "TooManyRequests": (S3ThrottleError, "Too many requests"),
    "SlowDown": (S3ThrottleError, "Slow down requests"),
    "ServiceUnavailable": (S3ConnectionError, "Service unavailable"),
    "InternalError": (S3ConnectionError, "Internal service error"),
    "RequestTimeout": (S3ConnectionError, "Request timeout"),
    "EntityTooLarge": (S3ValidationError, "Entity too large"),
    "InvalidObjectState": (S3ObjectError, "Invalid object state"),
    "ObjectLockConfigurationNotFoundError": (
        S3ObjectError,
        "Object lock not configured",
    ),
    "NotImplemented": (S3ArtifactError, "Feature not implemented"),
    "PreconditionFailed": (S3ConcurrencyError, "Precondition failed"),
    "QuotaExceeded": (S3StorageQuotaError, "Storage quota exceeded"),
}


def map_boto3_error(error: Exception, operation: str = "unknown") -> S3ArtifactError:
    """Map boto3 ClientError to appropriate S3ArtifactError subclass."""
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

    if isinstance(error, ClientError):
        error_code = error.response.get("Error", {}).get("Code", "Unknown")
        error_message = error.response.get("Error", {}).get("Message", str(error))

        exception_class, default_message = BOTO3_ERROR_MAPPING.get(
            error_code, (S3ArtifactError, "Unknown S3 error")
        )

        return exception_class(
            message=f"{default_message}: {error_message}",
            error_code=error_code,
            operation=operation,
            cause=error,
        )

    elif isinstance(error, NoCredentialsError):
        return S3ConnectionError(
            message="AWS credentials not found or invalid",
            error_code="NoCredentials",
            operation=operation,
            cause=error,
        )

    elif isinstance(error, BotoCoreError):
        return S3ConnectionError(
            message=f"AWS SDK error: {error}",
            error_code="BotoCoreError",
            operation=operation,
            cause=error,
        )

    else:
        return S3ArtifactError(
            message=f"Unexpected error: {error}", operation=operation, cause=error
        )
