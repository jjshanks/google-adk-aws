# Google ADK AWS Integrations - S3 Artifact Service Implementation - Phase 3

## Overview

Phase 3 focuses exclusively on **error handling and edge cases** as specified in the original roadmap. This phase transforms the S3ArtifactService into a production-ready, fault-tolerant system capable of gracefully handling all error conditions and edge cases encountered in real-world deployments.

**Duration**: 2-3 weeks
**Prerequisites**: Phase 2 completion (testing framework and performance optimization)
**Status**: Ready to begin following Phase 2 completion

## Phase 3 Objectives

Based on the original implementation plan Phase 3 scope:

1. **Comprehensive Error Handling**: Robust error recovery for all S3 operations
2. **Edge Case Management**: Handle large files, concurrent operations, network failures
3. **Retry Logic**: Exponential backoff and circuit breaker patterns
4. **Input Validation**: Sanitize and validate all inputs for security
5. **Graceful Degradation**: Maintain service availability under adverse conditions

## Implementation Roadmap

### Sub-Phase 3.1: Enhanced Exception Framework (Week 1, Days 1-3)
**Duration**: 3 days
**Focus**: Comprehensive exception hierarchy and error classification

#### 3.1.1 Exception Hierarchy Enhancement

**File**: `src/aws_adk/exceptions.py` (Update existing)
```python
"""Enhanced exception hierarchy for comprehensive error handling."""

from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class S3ArtifactError(Exception):
    """Base exception for S3 artifact operations with enhanced context."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
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
            "context": self.context
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
    "ObjectLockConfigurationNotFoundError": (S3ObjectError, "Object lock not configured"),
    "NotImplemented": (S3ArtifactError, "Feature not implemented"),
    "PreconditionFailed": (S3ConcurrencyError, "Precondition failed"),
    "QuotaExceeded": (S3StorageQuotaError, "Storage quota exceeded")
}


def map_boto3_error(error: Exception, operation: str = "unknown") -> S3ArtifactError:
    """Map boto3 ClientError to appropriate S3ArtifactError subclass."""
    from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError

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
            cause=error
        )

    elif isinstance(error, NoCredentialsError):
        return S3ConnectionError(
            message="AWS credentials not found or invalid",
            error_code="NoCredentials",
            operation=operation,
            cause=error
        )

    elif isinstance(error, BotoCoreError):
        return S3ConnectionError(
            message=f"AWS SDK error: {error}",
            error_code="BotoCoreError",
            operation=operation,
            cause=error
        )

    else:
        return S3ArtifactError(
            message=f"Unexpected error: {error}",
            operation=operation,
            cause=error
        )
```

#### 3.1.2 Error Context and Logging Enhancement

**File**: `src/aws_adk/retry_handler.py` (Update existing)
```python
"""Enhanced retry handler with comprehensive error context."""

import asyncio
import logging
import time
import random
from typing import Callable, Any, Optional, Type, Tuple, Dict
from functools import wraps
from dataclasses import dataclass

from .exceptions import (
    S3ArtifactError, S3ThrottleError, S3ConnectionError,
    map_boto3_error, S3ConcurrencyError
)

logger = logging.getLogger(__name__)


@dataclass
class RetryContext:
    """Context information for retry operations."""
    operation: str
    attempt: int
    total_attempts: int
    last_error: Optional[Exception]
    start_time: float
    delay: float

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


class RetryConfig:
    """Enhanced retry configuration with operation-specific settings."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_errors: Optional[Tuple[Type[Exception], ...]] = None,
        max_total_time: Optional[float] = None,
        backoff_strategy: str = "exponential"  # exponential, linear, fixed
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.max_total_time = max_total_time
        self.backoff_strategy = backoff_strategy

        self.retryable_errors = retryable_errors or (
            S3ConnectionError,
            S3ThrottleError,
            S3ConcurrencyError
        )

    def calculate_delay(self, attempt: int, context: RetryContext) -> float:
        """Calculate delay for next retry attempt."""
        if self.backoff_strategy == "exponential":
            delay = self.base_delay * (self.exponential_base ** attempt)
        elif self.backoff_strategy == "linear":
            delay = self.base_delay * (attempt + 1)
        else:  # fixed
            delay = self.base_delay

        # Apply max delay limit
        delay = min(delay, self.max_delay)

        # Apply jitter to prevent thundering herd
        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)

        # Check total time constraint
        if self.max_total_time:
            remaining_time = self.max_total_time - context.elapsed_time
            delay = min(delay, remaining_time)

        return max(0, delay)

    def should_retry(self, error: Exception, context: RetryContext) -> bool:
        """Determine if error should trigger retry."""
        # Check attempt limit
        if context.attempt >= self.max_attempts:
            return False

        # Check total time limit
        if self.max_total_time and context.elapsed_time >= self.max_total_time:
            return False

        # Check if error is retryable
        if not isinstance(error, self.retryable_errors):
            return False

        # Special handling for throttle errors - always retry with longer delay
        if isinstance(error, S3ThrottleError):
            return True

        return True


class CircuitBreaker:
    """Enhanced circuit breaker with state persistence and monitoring."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        success_threshold: int = 2,  # Successful calls needed to close circuit
        monitoring_window: float = 300.0  # 5 minutes
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.monitoring_window = monitoring_window

        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

        # Monitoring
        self.failure_history = []
        self.success_history = []

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Clean old history entries
            self._clean_history()

            # Check circuit state
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise S3ConnectionError(
                        message="Circuit breaker is OPEN - service unavailable",
                        error_code="CircuitBreakerOpen",
                        operation=func.__name__
                    )

            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result

            except self.expected_exception as e:
                self._on_failure(e)
                raise
            except Exception as e:
                # Unexpected errors don't affect circuit breaker
                logger.warning(f"Unexpected error in circuit breaker: {e}")
                raise

        return wrapper

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout

    def _on_success(self):
        """Handle successful operation."""
        self.success_count += 1
        self.success_history.append(time.time())

        if self.state == "HALF_OPEN":
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery")
        elif self.state == "CLOSED":
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self, error: Exception):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.failure_history.append((time.time(), str(error)))

        if self.state == "HALF_OPEN":
            # Return to OPEN state on any failure during half-open
            self.state = "OPEN"
            logger.warning("Circuit breaker returned to OPEN state")
        elif self.state == "CLOSED":
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(
                    f"Circuit breaker OPENED after {self.failure_count} failures"
                )

    def _clean_history(self):
        """Clean old entries from history."""
        current_time = time.time()
        cutoff_time = current_time - self.monitoring_window

        self.failure_history = [
            entry for entry in self.failure_history
            if entry[0] > cutoff_time
        ]

        self.success_history = [
            entry for entry in self.success_history
            if entry > cutoff_time
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "recent_failures": len(self.failure_history),
            "recent_successes": len(self.success_history),
            "last_failure_time": self.last_failure_time
        }


def with_retry(config: Optional[RetryConfig] = None):
    """Enhanced retry decorator with comprehensive error handling."""
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context = RetryContext(
                operation=func.__name__,
                attempt=0,
                total_attempts=config.max_attempts,
                last_error=None,
                start_time=time.time(),
                delay=0
            )

            while context.attempt < config.max_attempts:
                try:
                    logger.debug(
                        f"Executing {context.operation} (attempt {context.attempt + 1}/"
                        f"{context.total_attempts})"
                    )

                    result = await func(*args, **kwargs)

                    if context.attempt > 0:
                        logger.info(
                            f"Operation {context.operation} succeeded after "
                            f"{context.attempt + 1} attempts"
                        )

                    return result

                except Exception as e:
                    context.last_error = e
                    context.attempt += 1

                    # Map to appropriate exception type
                    mapped_error = map_boto3_error(e, context.operation)

                    # Check if we should retry
                    if not config.should_retry(mapped_error, context):
                        logger.error(
                            f"Operation {context.operation} failed permanently: {mapped_error}"
                        )
                        raise mapped_error

                    # Calculate delay for next attempt
                    context.delay = config.calculate_delay(context.attempt - 1, context)

                    if context.delay <= 0:
                        logger.error(
                            f"Retry timeout exceeded for {context.operation}"
                        )
                        break

                    logger.warning(
                        f"Attempt {context.attempt} failed for {context.operation}: "
                        f"{mapped_error}. Retrying in {context.delay:.2f}s"
                    )

                    await asyncio.sleep(context.delay)

            # All retry attempts exhausted
            final_error = map_boto3_error(context.last_error, context.operation)
            logger.error(
                f"Max retry attempts ({config.max_attempts}) exceeded for "
                f"{context.operation}: {final_error}"
            )
            raise final_error

        return wrapper
    return decorator
```

### Sub-Phase 3.2: Input Validation and Edge Cases (Week 1, Days 4-7)
**Duration**: 4 days
**Focus**: Comprehensive input validation and edge case handling

#### 3.2.1 Input Validation Framework

**File**: `src/aws_adk/validation.py` (New)
```python
"""Comprehensive input validation for S3 artifact operations."""

import re
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .exceptions import S3ValidationError

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Individual validation rule definition."""
    name: str
    validator: callable
    error_message: str
    is_critical: bool = True


class InputValidator:
    """Comprehensive input validation for artifact operations."""

    # S3 constraints
    MAX_OBJECT_KEY_LENGTH = 1024
    MAX_OBJECT_SIZE = 5 * 1024 * 1024 * 1024 * 1024  # 5TB
    MIN_OBJECT_SIZE = 0
    MAX_METADATA_SIZE = 2048
    MAX_METADATA_ENTRIES = 10

    # Application constraints
    MAX_APP_NAME_LENGTH = 100
    MAX_USER_ID_LENGTH = 100
    MAX_SESSION_ID_LENGTH = 100
    MAX_FILENAME_LENGTH = 255
    MAX_VERSION_NUMBER = 999999

    # Forbidden patterns
    DANGEROUS_PATTERNS = [
        r'\.\.',           # Path traversal
        r'\/\/',           # Double slashes
        r'\\\\',           # Double backslashes
        r'[\x00-\x1f]',    # Control characters
        r'[\x7f-\x9f]',    # Extended control characters
    ]

    # Valid character patterns
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._:-]+$')
    SAFE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.validation_rules = self._build_validation_rules()

    def _build_validation_rules(self) -> Dict[str, List[ValidationRule]]:
        """Build comprehensive validation rules."""
        return {
            "app_name": [
                ValidationRule(
                    "length",
                    lambda x: 1 <= len(str(x)) <= self.MAX_APP_NAME_LENGTH,
                    f"App name must be 1-{self.MAX_APP_NAME_LENGTH} characters"
                ),
                ValidationRule(
                    "pattern",
                    lambda x: self.SAFE_ID_PATTERN.match(str(x)),
                    "App name contains invalid characters"
                ),
                ValidationRule(
                    "not_empty",
                    lambda x: str(x).strip() != "",
                    "App name cannot be empty or whitespace"
                )
            ],
            "user_id": [
                ValidationRule(
                    "length",
                    lambda x: 1 <= len(str(x)) <= self.MAX_USER_ID_LENGTH,
                    f"User ID must be 1-{self.MAX_USER_ID_LENGTH} characters"
                ),
                ValidationRule(
                    "pattern",
                    lambda x: self.SAFE_ID_PATTERN.match(str(x)),
                    "User ID contains invalid characters"
                ),
                ValidationRule(
                    "not_reserved",
                    lambda x: str(x).lower() not in ["admin", "root", "system", "user"],
                    "User ID cannot be a reserved name"
                )
            ],
            "session_id": [
                ValidationRule(
                    "length",
                    lambda x: 1 <= len(str(x)) <= self.MAX_SESSION_ID_LENGTH,
                    f"Session ID must be 1-{self.MAX_SESSION_ID_LENGTH} characters"
                ),
                ValidationRule(
                    "pattern",
                    lambda x: self.SAFE_ID_PATTERN.match(str(x)),
                    "Session ID contains invalid characters"
                )
            ],
            "filename": [
                ValidationRule(
                    "length",
                    lambda x: 1 <= len(str(x)) <= self.MAX_FILENAME_LENGTH,
                    f"Filename must be 1-{self.MAX_FILENAME_LENGTH} characters"
                ),
                ValidationRule(
                    "safe_characters",
                    lambda x: self.SAFE_FILENAME_PATTERN.match(str(x)),
                    "Filename contains invalid characters"
                ),
                ValidationRule(
                    "no_dangerous_patterns",
                    lambda x: not any(re.search(p, str(x)) for p in self.DANGEROUS_PATTERNS),
                    "Filename contains dangerous patterns"
                ),
                ValidationRule(
                    "not_system_file",
                    lambda x: not str(x).lower().startswith(('.', '__', 'con', 'prn', 'aux')),
                    "Filename cannot be a system file"
                )
            ],
            "version": [
                ValidationRule(
                    "type",
                    lambda x: isinstance(x, int),
                    "Version must be an integer"
                ),
                ValidationRule(
                    "range",
                    lambda x: 0 <= x <= self.MAX_VERSION_NUMBER,
                    f"Version must be between 0 and {self.MAX_VERSION_NUMBER}"
                )
            ],
            "object_key": [
                ValidationRule(
                    "length",
                    lambda x: len(str(x)) <= self.MAX_OBJECT_KEY_LENGTH,
                    f"Object key exceeds maximum length of {self.MAX_OBJECT_KEY_LENGTH}"
                ),
                ValidationRule(
                    "no_path_traversal",
                    lambda x: not any(p in str(x) for p in ['../', '..\\']),
                    "Object key contains path traversal patterns"
                ),
                ValidationRule(
                    "valid_structure",
                    self._validate_object_key_structure,
                    "Object key does not follow expected structure"
                )
            ]
        }

    def validate_field(self, field_name: str, value: Any) -> List[str]:
        """Validate a single field and return list of error messages."""
        errors = []

        if field_name not in self.validation_rules:
            if self.strict_mode:
                errors.append(f"Unknown field: {field_name}")
            return errors

        rules = self.validation_rules[field_name]

        for rule in rules:
            try:
                if not rule.validator(value):
                    errors.append(rule.error_message)
                    if rule.is_critical and self.strict_mode:
                        break  # Stop on first critical error in strict mode

            except Exception as e:
                logger.warning(f"Validation rule {rule.name} failed: {e}")
                errors.append(f"Validation error for {field_name}: {rule.error_message}")

        return errors

    def validate_artifact_params(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int] = None
    ) -> None:
        """Validate all artifact operation parameters."""
        all_errors = []

        # Validate individual fields
        all_errors.extend(self.validate_field("app_name", app_name))
        all_errors.extend(self.validate_field("user_id", user_id))
        all_errors.extend(self.validate_field("session_id", session_id))
        all_errors.extend(self.validate_field("filename", filename))

        if version is not None:
            all_errors.extend(self.validate_field("version", version))

        # Validate composite object key
        object_key = self._construct_object_key(app_name, user_id, session_id, filename, version or 0)
        all_errors.extend(self.validate_field("object_key", object_key))

        # Cross-field validation
        all_errors.extend(self._validate_cross_fields(app_name, user_id, session_id, filename))

        if all_errors:
            raise S3ValidationError(
                message=f"Validation failed: {'; '.join(all_errors)}",
                error_code="ValidationFailed",
                context={
                    "app_name": app_name,
                    "user_id": user_id,
                    "session_id": session_id,
                    "filename": filename,
                    "version": version,
                    "errors": all_errors
                }
            )

    def validate_artifact_content(self, content: bytes, mime_type: str) -> None:
        """Validate artifact content and metadata."""
        errors = []

        # Size validation
        content_size = len(content)
        if content_size < self.MIN_OBJECT_SIZE:
            errors.append("Content cannot be empty")
        elif content_size > self.MAX_OBJECT_SIZE:
            errors.append(f"Content size ({content_size}) exceeds maximum ({self.MAX_OBJECT_SIZE})")

        # MIME type validation
        if not mime_type or not isinstance(mime_type, str):
            errors.append("MIME type is required and must be a string")
        elif len(mime_type) > 255:
            errors.append("MIME type too long")
        elif not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9!#$&\-\^_]*\/[a-zA-Z0-9][a-zA-Z0-9!#$&\-\^_]*$', mime_type):
            errors.append("Invalid MIME type format")

        # Content validation based on MIME type
        content_errors = self._validate_content_by_type(content, mime_type)
        errors.extend(content_errors)

        if errors:
            raise S3ValidationError(
                message=f"Content validation failed: {'; '.join(errors)}",
                error_code="ContentValidationFailed",
                context={
                    "content_size": content_size,
                    "mime_type": mime_type,
                    "errors": errors
                }
            )

    def sanitize_inputs(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str
    ) -> tuple[str, str, str, str]:
        """Sanitize inputs while preserving functionality."""

        def sanitize_id(value: str) -> str:
            """Sanitize ID fields."""
            # Remove dangerous characters, keep alphanumeric, dots, hyphens, underscores
            sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', str(value))
            # Ensure it starts with alphanumeric
            if sanitized and not sanitized[0].isalnum():
                sanitized = 'a' + sanitized
            return sanitized[:100]  # Truncate to max length

        def sanitize_filename(value: str) -> str:
            """Sanitize filename while preserving user: prefix if present."""
            if value.startswith("user:"):
                prefix = "user:"
                name_part = value[5:]
            else:
                prefix = ""
                name_part = value

            # Sanitize the name part
            sanitized = re.sub(r'[^a-zA-Z0-9._:-]', '_', name_part)
            return prefix + sanitized[:250]  # Leave room for prefix

        return (
            sanitize_id(app_name),
            sanitize_id(user_id),
            sanitize_id(session_id),
            sanitize_filename(filename)
        )

    def _validate_object_key_structure(self, object_key: str) -> bool:
        """Validate object key follows expected structure."""
        parts = object_key.split('/')

        # Minimum structure: app/user/session_or_user/filename/version
        if len(parts) < 5:
            return False

        # Check that version (last part) is numeric
        try:
            int(parts[-1])
        except ValueError:
            return False

        # Check middle part is either a valid session ID or "user"
        if len(parts) >= 5:
            scope_part = parts[2]
            return scope_part == "user" or self.SAFE_ID_PATTERN.match(scope_part)

        return True

    def _construct_object_key(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: int
    ) -> str:
        """Construct object key for validation."""
        if filename.startswith("user:"):
            return f"{app_name}/{user_id}/user/{filename}/{version}"
        else:
            return f"{app_name}/{user_id}/{session_id}/{filename}/{version}"

    def _validate_cross_fields(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str
    ) -> List[str]:
        """Validate relationships between fields."""
        errors = []

        # Check for conflicting user namespace usage
        if filename.startswith("user:") and session_id == "user":
            errors.append("Cannot use 'user' as session_id with user-namespaced filename")

        # Check for suspicious patterns
        if app_name == user_id == session_id:
            errors.append("App name, user ID, and session ID should not be identical")

        return errors

    def _validate_content_by_type(self, content: bytes, mime_type: str) -> List[str]:
        """Validate content based on MIME type."""
        errors = []

        try:
            # Text content validation
            if mime_type.startswith("text/"):
                try:
                    content.decode('utf-8')
                except UnicodeDecodeError:
                    errors.append("Text content must be valid UTF-8")

            # JSON validation
            elif mime_type == "application/json":
                import json
                try:
                    json.loads(content.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    errors.append("Invalid JSON content")

            # Image validation (basic header check)
            elif mime_type.startswith("image/"):
                if not self._is_valid_image_header(content, mime_type):
                    errors.append("Invalid image file header")

        except Exception as e:
            logger.warning(f"Content validation error: {e}")
            # Don't fail validation for content inspection errors

        return errors

    def _is_valid_image_header(self, content: bytes, mime_type: str) -> bool:
        """Basic image header validation."""
        if len(content) < 10:
            return False

        # Check common image format headers
        headers = {
            "image/jpeg": [b'\xff\xd8\xff'],
            "image/png": [b'\x89\x50\x4e\x47'],
            "image/gif": [b'GIF87a', b'GIF89a'],
            "image/webp": [b'RIFF'],
            "image/bmp": [b'BM']
        }

        expected_headers = headers.get(mime_type, [])
        if not expected_headers:
            return True  # Unknown format, skip validation

        return any(content.startswith(header) for header in expected_headers)


# Global validator instance
_global_validator: Optional[InputValidator] = None

def get_validator(strict_mode: bool = True) -> InputValidator:
    """Get global validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = InputValidator(strict_mode=strict_mode)
    return _global_validator
```

#### 3.2.2 Edge Case Handlers

**File**: `src/aws_adk/edge_case_handlers.py` (New)
```python
"""Handlers for edge cases in S3 artifact operations."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

from .exceptions import (
    S3ArtifactError, S3ConcurrencyError, S3ValidationError,
    S3StorageQuotaError, S3CorruptionError
)

logger = logging.getLogger(__name__)


@dataclass
class ConcurrencyContext:
    """Context for managing concurrent operations."""
    operation_id: str
    resource_key: str
    operation_type: str
    start_time: float
    max_duration: float = 300.0  # 5 minutes max


class ConcurrencyManager:
    """Manages concurrent access to artifacts with conflict resolution."""

    def __init__(self, max_concurrent_per_resource: int = 1):
        self.max_concurrent_per_resource = max_concurrent_per_resource
        self._active_operations: Dict[str, List[ConcurrencyContext]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_resource_key(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str
    ) -> str:
        """Generate unique resource key for locking."""
        if filename.startswith("user:"):
            return f"{app_name}:{user_id}:user:{filename}"
        else:
            return f"{app_name}:{user_id}:{session_id}:{filename}"

    async def acquire_operation_lock(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        operation_type: str,
        operation_id: str,
        timeout: float = 30.0
    ) -> None:
        """Acquire exclusive lock for artifact operation."""

        resource_key = self._get_resource_key(app_name, user_id, session_id, filename)

        # Get or create lock for this resource
        with self._global_lock:
            if resource_key not in self._locks:
                self._locks[resource_key] = asyncio.Lock()
            resource_lock = self._locks[resource_key]

        try:
            # Acquire lock with timeout
            await asyncio.wait_for(resource_lock.acquire(), timeout=timeout)

            # Check for conflicting operations
            await self._check_operation_conflicts(
                resource_key, operation_type, operation_id
            )

            # Register operation
            context = ConcurrencyContext(
                operation_id=operation_id,
                resource_key=resource_key,
                operation_type=operation_type,
                start_time=time.time()
            )

            if resource_key not in self._active_operations:
                self._active_operations[resource_key] = []
            self._active_operations[resource_key].append(context)

            logger.debug(f"Acquired lock for {operation_type} on {resource_key}")

        except asyncio.TimeoutError:
            raise S3ConcurrencyError(
                message=f"Timeout acquiring lock for {operation_type} on {filename}",
                error_code="ConcurrencyTimeout",
                operation=operation_type,
                context={
                    "resource_key": resource_key,
                    "timeout": timeout
                }
            )

    async def release_operation_lock(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        operation_id: str
    ) -> None:
        """Release operation lock."""

        resource_key = self._get_resource_key(app_name, user_id, session_id, filename)

        # Remove from active operations
        if resource_key in self._active_operations:
            self._active_operations[resource_key] = [
                ctx for ctx in self._active_operations[resource_key]
                if ctx.operation_id != operation_id
            ]

            if not self._active_operations[resource_key]:
                del self._active_operations[resource_key]

        # Release lock
        if resource_key in self._locks:
            self._locks[resource_key].release()
            logger.debug(f"Released lock for operation {operation_id} on {resource_key}")

    async def _check_operation_conflicts(
        self,
        resource_key: str,
        operation_type: str,
        operation_id: str
    ) -> None:
        """Check for conflicting operations on the same resource."""

        if resource_key not in self._active_operations:
            return

        active_ops = self._active_operations[resource_key]
        current_time = time.time()

        # Clean up stale operations
        active_ops = [
            ctx for ctx in active_ops
            if current_time - ctx.start_time < ctx.max_duration
        ]
        self._active_operations[resource_key] = active_ops

        # Check for conflicts
        conflicting_ops = {
            "write": ["write", "delete"],
            "delete": ["write", "read", "delete"],
            "read": ["write", "delete"]
        }

        conflicts = conflicting_ops.get(operation_type, [])

        for ctx in active_ops:
            if ctx.operation_type in conflicts:
                raise S3ConcurrencyError(
                    message=f"Conflicting {ctx.operation_type} operation in progress",
                    error_code="OperationConflict",
                    operation=operation_type,
                    context={
                        "conflicting_operation": ctx.operation_type,
                        "conflicting_operation_id": ctx.operation_id,
                        "resource_key": resource_key
                    }
                )


class LargeFileHandler:
    """Handles large file operations with special considerations."""

    # Size thresholds
    LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB
    HUGE_FILE_THRESHOLD = 1024 * 1024 * 1024  # 1GB
    MAX_MEMORY_USAGE = 500 * 1024 * 1024  # 500MB max in memory

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def should_use_streaming(self, content_size: int) -> bool:
        """Determine if streaming should be used for this file size."""
        return content_size > self.MAX_MEMORY_USAGE

    def should_use_multipart(self, content_size: int) -> bool:
        """Determine if multipart upload should be used."""
        return content_size > self.LARGE_FILE_THRESHOLD

    async def validate_large_file_operation(
        self,
        content_size: int,
        operation_type: str,
        available_memory: Optional[int] = None
    ) -> None:
        """Validate that large file operation can proceed."""

        # Check size limits
        if content_size > 5 * 1024 * 1024 * 1024 * 1024:  # 5TB S3 limit
            raise S3ValidationError(
                message=f"File size ({content_size}) exceeds S3 maximum (5TB)",
                error_code="FileTooLarge",
                context={"content_size": content_size}
            )

        # Check memory availability for large files
        if self.should_use_streaming(content_size):
            if available_memory and available_memory < self.MAX_MEMORY_USAGE:
                raise S3StorageQuotaError(
                    message="Insufficient memory for large file operation",
                    error_code="InsufficientMemory",
                    context={
                        "required_memory": self.MAX_MEMORY_USAGE,
                        "available_memory": available_memory
                    }
                )

        logger.info(
            f"Large file operation validated: {content_size} bytes, "
            f"streaming: {self.should_use_streaming(content_size)}, "
            f"multipart: {self.should_use_multipart(content_size)}"
        )

    async def process_large_upload(
        self,
        content: bytes,
        upload_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> Any:
        """Process large file upload with progress tracking."""

        content_size = len(content)

        if self.should_use_multipart(content_size):
            return await self._multipart_upload(content, upload_func, progress_callback)
        else:
            return await self._standard_upload(content, upload_func, progress_callback)

    async def _multipart_upload(
        self,
        content: bytes,
        upload_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> Any:
        """Handle multipart upload for very large files."""

        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        total_size = len(content)
        uploaded = 0

        try:
            for i in range(0, total_size, chunk_size):
                chunk = content[i:i + chunk_size]

                # Upload chunk
                result = await upload_func(chunk, i // chunk_size + 1)
                uploaded += len(chunk)

                # Progress callback
                if progress_callback:
                    progress_callback(uploaded, total_size)

                logger.debug(f"Uploaded chunk {i // chunk_size + 1}, {uploaded}/{total_size} bytes")

            return result

        except Exception as e:
            logger.error(f"Multipart upload failed at {uploaded}/{total_size} bytes: {e}")
            raise

    async def _standard_upload(
        self,
        content: bytes,
        upload_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> Any:
        """Handle standard upload with progress tracking."""

        if progress_callback:
            progress_callback(0, len(content))

        result = await upload_func(content)

        if progress_callback:
            progress_callback(len(content), len(content))

        return result


class CorruptionDetector:
    """Detects and handles data corruption during transfer."""

    def __init__(self):
        self.hash_algorithm = "sha256"

    async def verify_content_integrity(
        self,
        original_content: bytes,
        retrieved_content: bytes,
        expected_hash: Optional[str] = None
    ) -> None:
        """Verify content integrity after transfer."""

        # Size check
        if len(original_content) != len(retrieved_content):
            raise S3CorruptionError(
                message="Content size mismatch detected",
                error_code="SizeMismatch",
                context={
                    "original_size": len(original_content),
                    "retrieved_size": len(retrieved_content)
                }
            )

        # Hash verification
        if expected_hash:
            actual_hash = self._calculate_hash(retrieved_content)
            if actual_hash != expected_hash:
                raise S3CorruptionError(
                    message="Content hash mismatch detected",
                    error_code="HashMismatch",
                    context={
                        "expected_hash": expected_hash,
                        "actual_hash": actual_hash
                    }
                )

        # Byte-by-byte comparison for critical data
        if original_content != retrieved_content:
            raise S3CorruptionError(
                message="Content corruption detected during transfer",
                error_code="ContentCorruption"
            )

        logger.debug("Content integrity verification passed")

    def _calculate_hash(self, content: bytes) -> str:
        """Calculate content hash for verification."""
        import hashlib
        return hashlib.sha256(content).hexdigest()


class NetworkFailureHandler:
    """Handles network failures and connectivity issues."""

    def __init__(self):
        self.connection_pool_stats = {}
        self.failure_history = []

    async def handle_network_failure(
        self,
        error: Exception,
        operation: str,
        retry_count: int
    ) -> bool:
        """Handle network failure and determine if retry should proceed."""

        self.failure_history.append({
            "timestamp": time.time(),
            "error": str(error),
            "operation": operation,
            "retry_count": retry_count
        })

        # Clean old history (keep last hour)
        cutoff_time = time.time() - 3600
        self.failure_history = [
            entry for entry in self.failure_history
            if entry["timestamp"] > cutoff_time
        ]

        # Check failure rate
        recent_failures = len(self.failure_history)
        if recent_failures > 10:  # More than 10 failures in last hour
            logger.warning(f"High network failure rate detected: {recent_failures} failures")
            return False  # Don't retry

        # Analyze error type
        error_str = str(error).lower()

        # Temporary network issues - retry
        if any(term in error_str for term in [
            "timeout", "connection", "network", "dns", "socket"
        ]):
            logger.info(f"Network error detected, will retry: {error}")
            return True

        # Permanent issues - don't retry
        if any(term in error_str for term in [
            "authentication", "authorization", "forbidden", "invalid"
        ]):
            logger.error(f"Permanent error detected, will not retry: {error}")
            return False

        # Default: retry for unknown errors
        return True

    def get_network_health_stats(self) -> Dict[str, Any]:
        """Get network health statistics."""
        recent_failures = len(self.failure_history)

        return {
            "recent_failure_count": recent_failures,
            "connection_pool_stats": self.connection_pool_stats,
            "last_failure": self.failure_history[-1] if self.failure_history else None
        }


# Global instances
_concurrency_manager: Optional[ConcurrencyManager] = None
_large_file_handler: Optional[LargeFileHandler] = None
_corruption_detector: Optional[CorruptionDetector] = None
_network_failure_handler: Optional[NetworkFailureHandler] = None

def get_concurrency_manager() -> ConcurrencyManager:
    """Get global concurrency manager instance."""
    global _concurrency_manager
    if _concurrency_manager is None:
        _concurrency_manager = ConcurrencyManager()
    return _concurrency_manager

def get_large_file_handler() -> LargeFileHandler:
    """Get global large file handler instance."""
    global _large_file_handler
    if _large_file_handler is None:
        _large_file_handler = LargeFileHandler()
    return _large_file_handler

def get_corruption_detector() -> CorruptionDetector:
    """Get global corruption detector instance."""
    global _corruption_detector
    if _corruption_detector is None:
        _corruption_detector = CorruptionDetector()
    return _corruption_detector

def get_network_failure_handler() -> NetworkFailureHandler:
    """Get global network failure handler instance."""
    global _network_failure_handler
    if _network_failure_handler is None:
        _network_failure_handler = NetworkFailureHandler()
    return _network_failure_handler
```

### Sub-Phase 3.3: Enhanced S3ArtifactService Integration (Week 2, Days 1-4)
**Duration**: 4 days
**Focus**: Integrate all error handling and edge case management into the service

#### 3.3.1 Service Integration

**File**: `src/aws_adk/s3_artifact_service.py` (Major Update)
```python
"""Enhanced S3ArtifactService with comprehensive error handling and edge cases."""

import asyncio
import logging
import time
import uuid
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import boto3
from botocore.exceptions import ClientError
from google.adk.artifacts import BaseArtifactService
from google.genai import types

from .exceptions import S3ArtifactError, map_boto3_error
from .retry_handler import with_retry, RetryConfig, CircuitBreaker
from .validation import get_validator
from .edge_case_handlers import (
    get_concurrency_manager, get_large_file_handler,
    get_corruption_detector, get_network_failure_handler
)
from .connection_pool import get_connection_pool
from .batch_operations import S3BatchOperations, MultipartUploadManager
from .security import S3SecurityManager, AccessControlManager

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
        **kwargs
    ):
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
            backoff_strategy="exponential"
        )

        # Circuit breakers for different operation types
        self.read_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=30.0,
            expected_exception=S3ArtifactError
        )

        self.write_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=60.0,
            expected_exception=S3ArtifactError
        )

        # Initialize S3 client and components
        try:
            self.s3_client = self._create_s3_client()
            self.batch_operations = S3BatchOperations(
                self.s3_client, self.bucket_name
            )
            self.multipart_manager = MultipartUploadManager(
                self.s3_client, self.bucket_name
            )
            self.security_manager = S3SecurityManager(
                self.s3_client, self.bucket_name
            )
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
                endpoint_url=self.endpoint_url
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
        filename: str
    ):
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
                app_name, user_id, session_id, filename = self.validator.sanitize_inputs(
                    app_name, user_id, session_id, filename
                )

            # Acquire concurrency lock
            await self.concurrency_manager.acquire_operation_lock(
                app_name, user_id, session_id, filename,
                operation_type, operation_id
            )

            logger.debug(
                f"Started {operation_type} operation {operation_id} for "
                f"{app_name}/{user_id}/{session_id}/{filename}"
            )

            yield {
                "operation_id": operation_id,
                "sanitized_params": (app_name, user_id, session_id, filename)
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
                f"Completed {operation_type} operation {operation_id} in {operation_time:.2f}s"
            )

    @with_retry()
    async def save_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        artifact: types.Part
    ) -> int:
        """Save artifact with comprehensive error handling and edge case management."""

        async with self._operation_context(
            "save", app_name, user_id, session_id, filename
        ) as context:
            app_name, user_id, session_id, filename = context["sanitized_params"]

            @self.write_circuit_breaker
            async def _save_with_protection():
                return await self._save_artifact_impl(
                    app_name, user_id, session_id, filename, artifact
                )

            return await _save_with_protection()

    async def _save_artifact_impl(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        artifact: types.Part
    ) -> int:
        """Implementation of save artifact with all error handling."""

        try:
            # Content validation
            content = artifact.inline_data.data
            mime_type = artifact.inline_data.mime_type

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
                filename=filename
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
                        context={"object_key": object_key}
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
                "content-size": str(content_size)
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
                    Metadata=metadata
                )

            logger.info(
                f"Successfully saved artifact {filename} version {version} "
                f"({content_size} bytes) for {app_name}/{user_id}"
            )

            return version

        except Exception as e:
            mapped_error = map_boto3_error(e, "save_artifact")
            logger.error(
                f"Failed to save artifact {filename}: {mapped_error}"
            )
            raise mapped_error

    @with_retry()
    async def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int] = None
    ) -> Optional[types.Part]:
        """Load artifact with comprehensive error handling."""

        async with self._operation_context(
            "load", app_name, user_id, session_id, filename
        ) as context:
            app_name, user_id, session_id, filename = context["sanitized_params"]

            @self.read_circuit_breaker
            async def _load_with_protection():
                return await self._load_artifact_impl(
                    app_name, user_id, session_id, filename, version
                )

            return await _load_with_protection()

    async def _load_artifact_impl(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int]
    ) -> Optional[types.Part]:
        """Implementation of load artifact with error handling."""

        try:
            # Determine version if not specified
            if version is None:
                versions = await self.list_versions(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                    filename=filename
                )
                if not versions:
                    return None
                version = max(versions)

            # Version validation
            if self.enable_validation and version < 0:
                raise S3ValidationError(
                    message="Version must be non-negative",
                    error_code="InvalidVersion",
                    context={"version": version}
                )

            # Generate object key
            object_key = self.security_manager.generate_secure_object_key(
                app_name, user_id, session_id, filename, version
            )

            # Download from S3
            response = await self.connection_pool.execute_async(
                self.s3_client.get_object,
                Bucket=self.bucket_name,
                Key=object_key
            )

            # Read content
            content = response['Body'].read()
            content_type = response.get('ContentType', 'application/octet-stream')
            metadata = response.get('Metadata', {})

            # Integrity verification
            if self.enable_integrity_checks and 'content-hash' in metadata:
                expected_hash = metadata['content-hash']
                if not self.security_manager.verify_content_integrity(
                    content, expected_hash
                ):
                    raise S3CorruptionError(
                        message="Content integrity verification failed",
                        error_code="IntegrityCheckFailed",
                        context={
                            "expected_hash": expected_hash,
                            "object_key": object_key
                        }
                    )

            # Create and return types.Part
            part = types.Part.from_bytes(data=content, mime_type=content_type)

            logger.debug(
                f"Successfully loaded artifact {filename} version {version} "
                f"({len(content)} bytes)"
            )

            return part

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None  # Artifact not found
            else:
                mapped_error = map_boto3_error(e, "load_artifact")
                logger.error(f"Failed to load artifact {filename}: {mapped_error}")
                raise mapped_error

        except Exception as e:
            mapped_error = map_boto3_error(e, "load_artifact")
            logger.error(f"Failed to load artifact {filename}: {mapped_error}")
            raise mapped_error

    # Additional methods (list_artifact_keys, delete_artifact, list_versions)
    # would follow similar patterns with comprehensive error handling...

    async def get_service_health(self) -> Dict[str, Any]:
        """Get comprehensive service health information."""
        return {
            "bucket_name": self.bucket_name,
            "region_name": self.region_name,
            "connection_pool_stats": self.connection_pool.get_stats(),
            "circuit_breaker_stats": {
                "read": self.read_circuit_breaker.get_stats(),
                "write": self.write_circuit_breaker.get_stats()
            },
            "network_health": self.network_failure_handler.get_network_health_stats(),
            "feature_flags": {
                "validation": self.enable_validation,
                "security_checks": self.enable_security_checks,
                "integrity_checks": self.enable_integrity_checks
            }
        }

    async def cleanup_resources(self) -> None:
        """Cleanup all resources and connections."""
        try:
            if hasattr(self, 'connection_pool'):
                self.connection_pool.close()
            logger.info("S3ArtifactService resources cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")
```

## Phase 3 Testing Strategy

### Testing Error Scenarios
**File**: `tests/unit/test_error_handling.py` (New)
```python
"""Comprehensive tests for error handling and edge cases."""

import pytest
import asyncio
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError

from aws_adk import S3ArtifactService
from aws_adk.exceptions import *
from aws_adk.validation import InputValidator
from aws_adk.edge_case_handlers import ConcurrencyManager

# Comprehensive test cases for all error scenarios and edge cases
# Testing validation, concurrency, large files, network failures, etc.
```

## Success Criteria

1. **Robust Error Handling**: All boto3 errors properly mapped and handled
2. **Edge Case Coverage**: Large files, concurrent operations, network failures handled
3. **Input Validation**: All inputs sanitized and validated for security
4. **Graceful Degradation**: Service maintains availability under adverse conditions
5. **Comprehensive Testing**: >95% test coverage for error scenarios
6. **Production Readiness**: Zero unhandled exceptions in normal operations

## Implementation Timeline

| Sub-Phase | Duration | Focus Area |
|-----------|----------|------------|
| 3.1 | 3 days | Exception framework and error classification |
| 3.2 | 4 days | Input validation and edge case handlers |
| 3.3 | 4 days | Service integration and comprehensive testing |

**Total Duration**: 11 days (2.2 weeks)

This Phase 3 plan focuses exclusively on the error handling and edge cases scope from the original roadmap, ensuring the S3ArtifactService becomes truly production-ready with robust fault tolerance.
