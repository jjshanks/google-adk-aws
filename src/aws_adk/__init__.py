"""Google ADK AWS Integrations.

This package provides AWS service integrations for Google Agent Development Kit (ADK),
starting with an S3-based implementation of ADK's BaseArtifactService for artifact
storage and retrieval using Amazon S3 or S3-compatible services.
"""

from .s3_artifact_service import S3ArtifactService

__version__ = "0.1.0"
__author__ = "Joshua Shanks"
__email__ = "jjshanks@gmail.com"

__all__ = ["S3ArtifactService", "__version__"]
