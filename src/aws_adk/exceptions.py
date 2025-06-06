"""Enhanced exception hierarchy for S3 artifact operations."""

from typing import Any, Optional


class S3ArtifactError(Exception):
    """Base exception for S3 artifact operations."""

    def __init__(
        self, message: str, error_code: Optional[str] = None, **kwargs: Any
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.details = kwargs


class S3ConnectionError(S3ArtifactError):
    """Raised when S3 connection fails."""

    pass


class S3PermissionError(S3ArtifactError):
    """Raised when S3 permissions are insufficient."""

    pass


class S3BucketError(S3ArtifactError):
    """Raised when bucket-related operations fail."""

    pass


class S3ObjectError(S3ArtifactError):
    """Raised when object-related operations fail."""

    pass


class S3ThrottleError(S3ArtifactError):
    """Raised when S3 rate limiting occurs."""

    pass


class S3ArtifactNotFoundError(S3ArtifactError):
    """Raised when requested artifact is not found."""

    pass


class S3ArtifactVersionError(S3ArtifactError):
    """Raised when version-related operations fail."""

    pass
