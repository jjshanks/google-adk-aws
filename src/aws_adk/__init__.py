"""Google ADK AWS Integrations.

This package provides AWS service integrations for Google Agent Development Kit (ADK),
starting with an S3-based implementation of ADK's BaseArtifactService for artifact
storage and retrieval using Amazon S3 or S3-compatible services.
"""

from .exceptions import (
    S3ArtifactError,
    S3ArtifactNotFoundError,
    S3ArtifactVersionError,
    S3BucketError,
    S3ConcurrencyError,
    S3ConnectionError,
    S3ObjectError,
    S3PermissionError,
    S3ThrottleError,
    S3ValidationError,
)
from .retry_handler import RetryConfig
from .s3_artifact_service import S3ArtifactService
from .security import AccessControlManager, EncryptionManager, S3SecurityManager

__version__ = "0.1.0"
__author__ = "Joshua Shanks"
__email__ = "jjshanks@gmail.com"

__all__ = [
    "S3ArtifactService",
    "S3ArtifactError",
    "S3ConnectionError",
    "S3PermissionError",
    "S3BucketError",
    "S3ObjectError",
    "S3ThrottleError",
    "S3ArtifactNotFoundError",
    "S3ArtifactVersionError",
    "S3ConcurrencyError",
    "S3ValidationError",
    "RetryConfig",
    "S3SecurityManager",
    "AccessControlManager",
    "EncryptionManager",
    "__version__",
]
