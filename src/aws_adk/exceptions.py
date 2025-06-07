"""Enhanced exception hierarchy for comprehensive error handling."""

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _sanitize_context_for_logging(context: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize context dictionary for safe logging by masking sensitive information."""
    if not context:
        return {}

    # Fields that should be completely masked
    SENSITIVE_FIELDS = {
        "user_id",
        "user-id",
        "userId",
        "userID",
        "session_id",
        "session-id",
        "sessionId",
        "sessionID",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "password",
        "token",
        "secret",
        "key",
        "credential",
        "email",
        "phone",
        "ssn",
        "social_security_number",
    }

    # Fields that should be partially masked (show first/last few chars)
    PARTIALLY_MASKABLE_FIELDS = {
        "app_name",
        "app-name",
        "appName",
        "filename",
        "object_key",
        "bucket_name",
    }

    sanitized: Dict[str, Any] = {}

    for key, value in context.items():
        key_lower = key.lower().replace("-", "_").replace(" ", "_")

        if key_lower in SENSITIVE_FIELDS:
            # Completely mask sensitive fields
            sanitized[key] = "***MASKED***"
        elif (
            key_lower in PARTIALLY_MASKABLE_FIELDS
            and isinstance(value, str)
            and len(value) > 6
        ):
            # Partially mask - show first 3 and last 3 characters
            sanitized[key] = f"{value[:3]}***{value[-3:]}"
        elif isinstance(value, str):
            # Check if the string value looks like sensitive data
            if _is_sensitive_string_value(value):
                sanitized[key] = "***MASKED***"
            else:
                sanitized[key] = value
        elif isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized[key] = _sanitize_context_for_logging(value)
        else:
            # Keep non-string, non-dict values as-is (numbers, booleans, etc.)
            sanitized[key] = value

    return sanitized


def _is_sensitive_string_value(value: str) -> bool:
    """Check if a string value appears to contain sensitive information."""
    if not isinstance(value, str) or len(value) < 8:
        return False

    # Patterns that might indicate sensitive data
    sensitive_patterns = [
        r"^[A-Z0-9]{20,}$",  # AWS access key pattern
        r"^[A-Za-z0-9/+=]{40,}$",  # AWS secret key pattern
        r"^[a-f0-9]{32,}$",  # Hash-like strings
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email pattern
        r"\b\d{3}-?\d{2}-?\d{4}\b",  # SSN pattern
    ]

    for pattern in sensitive_patterns:
        if re.search(pattern, value):
            return True

    return False


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

        # Log error with sanitized context to avoid exposing sensitive data
        sanitized_context = _sanitize_context_for_logging(self.context)
        logger.error(
            f"S3ArtifactError: {message} | Operation: {operation} | "
            f"Code: {error_code} | Context: {sanitized_context}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error": self.__class__.__name__,
            "message": str(self),
            "error_code": self.error_code,
            "operation": self.operation,
            "context": _sanitize_context_for_logging(self.context),
        }

    def get_sanitized_context(self) -> Dict[str, Any]:
        """Get sanitized context for safe logging/display."""
        return _sanitize_context_for_logging(self.context)

    def get_raw_context(self) -> Dict[str, Any]:
        """Get raw context for debugging purposes. Use with caution."""
        return self.context.copy()


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
