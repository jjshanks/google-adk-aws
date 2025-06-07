"""Comprehensive input validation for S3 artifact operations."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .exceptions import S3ValidationError

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Individual validation rule definition."""

    name: str
    validator: Callable[[Any], bool]
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
        r"\.\.",  # Path traversal
        r"\/\/",  # Double slashes
        r"\\\\",  # Double backslashes
        r"[\x00-\x1f]",  # Control characters
        r"[\x7f-\x9f]",  # Extended control characters
    ]

    # Valid character patterns
    SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9._:-]+$")
    SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")

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
                    f"App name must be 1-{self.MAX_APP_NAME_LENGTH} characters",
                ),
                ValidationRule(
                    "pattern",
                    lambda x: bool(self.SAFE_ID_PATTERN.match(str(x))),
                    "App name contains invalid characters",
                ),
                ValidationRule(
                    "not_empty",
                    lambda x: str(x).strip() != "",
                    "App name cannot be empty or whitespace",
                ),
            ],
            "user_id": [
                ValidationRule(
                    "length",
                    lambda x: 1 <= len(str(x)) <= self.MAX_USER_ID_LENGTH,
                    f"User ID must be 1-{self.MAX_USER_ID_LENGTH} characters",
                ),
                ValidationRule(
                    "pattern",
                    lambda x: bool(self.SAFE_ID_PATTERN.match(str(x))),
                    "User ID contains invalid characters",
                ),
                ValidationRule(
                    "not_reserved",
                    lambda x: str(x).lower() not in ["admin", "root", "system", "user"],
                    "User ID cannot be a reserved name",
                ),
            ],
            "session_id": [
                ValidationRule(
                    "length",
                    lambda x: 1 <= len(str(x)) <= self.MAX_SESSION_ID_LENGTH,
                    f"Session ID must be 1-{self.MAX_SESSION_ID_LENGTH} characters",
                ),
                ValidationRule(
                    "pattern",
                    lambda x: bool(self.SAFE_ID_PATTERN.match(str(x))),
                    "Session ID contains invalid characters",
                ),
            ],
            "filename": [
                ValidationRule(
                    "length",
                    lambda x: 1 <= len(str(x)) <= self.MAX_FILENAME_LENGTH,
                    f"Filename must be 1-{self.MAX_FILENAME_LENGTH} characters",
                ),
                ValidationRule(
                    "safe_characters",
                    lambda x: bool(self.SAFE_FILENAME_PATTERN.match(str(x))),
                    "Filename contains invalid characters",
                ),
                ValidationRule(
                    "no_dangerous_patterns",
                    lambda x: not any(
                        re.search(p, str(x)) for p in self.DANGEROUS_PATTERNS
                    ),
                    "Filename contains dangerous patterns",
                ),
                ValidationRule(
                    "not_system_file",
                    lambda x: not str(x)
                    .lower()
                    .startswith((".", "__", "con", "prn", "aux")),
                    "Filename cannot be a system file",
                ),
            ],
            "version": [
                ValidationRule(
                    "type",
                    lambda x: isinstance(x, int),
                    "Version must be an integer",
                ),
                ValidationRule(
                    "range",
                    lambda x: 0 <= x <= self.MAX_VERSION_NUMBER,
                    f"Version must be between 0 and {self.MAX_VERSION_NUMBER}",
                ),
            ],
            "object_key": [
                ValidationRule(
                    "length",
                    lambda x: len(str(x)) <= self.MAX_OBJECT_KEY_LENGTH,
                    f"Object key exceeds maximum length of "
                    f"{self.MAX_OBJECT_KEY_LENGTH}",
                ),
                ValidationRule(
                    "no_path_traversal",
                    lambda x: not any(p in str(x) for p in ["../", "..\\"]),
                    "Object key contains path traversal patterns",
                ),
                ValidationRule(
                    "valid_structure",
                    self._validate_object_key_structure,
                    "Object key does not follow expected structure",
                ),
            ],
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
                errors.append(
                    f"Validation error for {field_name}: {rule.error_message}"
                )

        return errors

    def validate_artifact_params(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int] = None,
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
        object_key = self._construct_object_key(
            app_name, user_id, session_id, filename, version or 0
        )
        all_errors.extend(self.validate_field("object_key", object_key))

        # Cross-field validation
        all_errors.extend(
            self._validate_cross_fields(app_name, user_id, session_id, filename)
        )

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
                    "errors": all_errors,
                },
            )

    def validate_artifact_content(self, content: bytes, mime_type: str) -> None:
        """Validate artifact content and metadata."""
        errors = []

        # Size validation
        content_size = len(content)
        if content_size < self.MIN_OBJECT_SIZE:
            errors.append("Content cannot be empty")
        elif content_size > self.MAX_OBJECT_SIZE:
            errors.append(
                f"Content size ({content_size}) exceeds maximum "
                f"({self.MAX_OBJECT_SIZE})"
            )

        # MIME type validation
        if not mime_type or not isinstance(mime_type, str):
            errors.append("MIME type is required and must be a string")
        elif len(mime_type) > 255:
            errors.append("MIME type too long")
        elif not re.match(
            r"^[a-zA-Z0-9][a-zA-Z0-9!#$&\-\^_]*\/[a-zA-Z0-9][a-zA-Z0-9!#$&\-\^_]*$",
            mime_type,
        ):
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
                    "errors": errors,
                },
            )

    def sanitize_inputs(
        self, app_name: str, user_id: str, session_id: str, filename: str
    ) -> tuple[str, str, str, str]:
        """Sanitize inputs while preserving functionality."""

        def sanitize_id(value: str) -> str:
            """Sanitize ID fields."""
            # Remove dangerous characters, keep alphanumeric, dots, hyphens, underscores
            sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", str(value))
            # Ensure it starts with alphanumeric
            if sanitized and not sanitized[0].isalnum():
                sanitized = "a" + sanitized
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
            sanitized = re.sub(r"[^a-zA-Z0-9._:-]", "_", name_part)
            return prefix + sanitized[:250]  # Leave room for prefix

        return (
            sanitize_id(app_name),
            sanitize_id(user_id),
            sanitize_id(session_id),
            sanitize_filename(filename),
        )

    def _validate_object_key_structure(self, object_key: str) -> bool:
        """Validate object key follows expected structure."""
        parts = object_key.split("/")

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
            return scope_part == "user" or bool(self.SAFE_ID_PATTERN.match(scope_part))

        return True

    def _construct_object_key(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: int,
    ) -> str:
        """Construct object key for validation."""
        if filename.startswith("user:"):
            return f"{app_name}/{user_id}/user/{filename}/{version}"
        else:
            return f"{app_name}/{user_id}/{session_id}/{filename}/{version}"

    def _validate_cross_fields(
        self, app_name: str, user_id: str, session_id: str, filename: str
    ) -> List[str]:
        """Validate relationships between fields."""
        errors = []

        # Check for conflicting user namespace usage
        if filename.startswith("user:") and session_id == "user":
            errors.append(
                "Cannot use 'user' as session_id with user-namespaced filename"
            )

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
                    content.decode("utf-8")
                except UnicodeDecodeError:
                    errors.append("Text content must be valid UTF-8")

            # JSON validation
            elif mime_type == "application/json":
                import json

                try:
                    json.loads(content.decode("utf-8"))
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
            "image/jpeg": [b"\xff\xd8\xff"],
            "image/png": [b"\x89\x50\x4e\x47"],
            "image/gif": [b"GIF87a", b"GIF89a"],
            "image/webp": [b"RIFF"],
            "image/bmp": [b"BM"],
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
