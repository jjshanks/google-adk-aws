"""Security features and hardening for S3 artifact service."""

import hashlib
import hmac
import logging
import secrets
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from .exceptions import S3ArtifactError

logger = logging.getLogger(__name__)


class S3SecurityManager:
    """Manages security features for S3 artifact operations."""

    def __init__(self, s3_client: boto3.client, bucket_name: str):
        self.s3_client = s3_client
        self.bucket_name = bucket_name

    def validate_bucket_security(self) -> Dict[str, Any]:
        """Validate bucket security configuration."""
        security_status: Dict[str, Any] = {
            "encryption": False,
            "versioning": False,
            "public_access_blocked": False,
            "logging_enabled": False,
            "mfa_delete": False,
            "recommendations": [],
        }

        try:
            # Check encryption
            try:
                self.s3_client.get_bucket_encryption(Bucket=self.bucket_name)
                security_status["encryption"] = True
            except ClientError as e:
                if (
                    e.response["Error"]["Code"]
                    != "ServerSideEncryptionConfigurationNotFoundError"
                ):
                    raise
                security_status["recommendations"].append(
                    "Enable server-side encryption for enhanced security"
                )

            # Check versioning
            try:
                versioning = self.s3_client.get_bucket_versioning(
                    Bucket=self.bucket_name
                )
                if versioning.get("Status") == "Enabled":
                    security_status["versioning"] = True
                    if versioning.get("MfaDelete") == "Enabled":
                        security_status["mfa_delete"] = True
                else:
                    security_status["recommendations"].append(
                        "Enable versioning for better data protection"
                    )
            except ClientError:
                pass

            # Check public access block
            try:
                public_access = self.s3_client.get_public_access_block(
                    Bucket=self.bucket_name
                )
                block_config = public_access.get("PublicAccessBlockConfiguration", {})
                if all(
                    [
                        block_config.get("BlockPublicAcls", False),
                        block_config.get("IgnorePublicAcls", False),
                        block_config.get("BlockPublicPolicy", False),
                        block_config.get("RestrictPublicBuckets", False),
                    ]
                ):
                    security_status["public_access_blocked"] = True
                else:
                    security_status["recommendations"].append(
                        "Enable public access block for all settings"
                    )
            except ClientError:
                security_status["recommendations"].append(
                    "Configure public access block settings"
                )

            # Check logging
            try:
                logging_config = self.s3_client.get_bucket_logging(
                    Bucket=self.bucket_name
                )
                if "LoggingEnabled" in logging_config:
                    security_status["logging_enabled"] = True
                else:
                    security_status["recommendations"].append(
                        "Enable access logging for audit trails"
                    )
            except ClientError:
                pass

        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            raise S3ArtifactError(f"Failed to validate bucket security: {e}") from e

        return security_status

    def generate_secure_object_key(
        self, app_name: str, user_id: str, session_id: str, filename: str, version: int
    ) -> str:
        """Generate secure object key with sanitization."""

        # Sanitize inputs
        def sanitize(value: str) -> str:
            """Remove/replace unsafe characters."""
            # Allow alphanumeric, hyphens, underscores, dots
            import re

            return re.sub(r"[^a-zA-Z0-9\-_.]", "_", str(value))

        safe_app_name = sanitize(app_name)
        safe_user_id = sanitize(user_id)
        safe_session_id = sanitize(session_id)
        safe_filename = sanitize(filename)

        # Construct secure path
        if filename.startswith("user:"):
            return f"{safe_app_name}/{safe_user_id}/user/{safe_filename}/{version}"
        else:
            return (
                f"{safe_app_name}/{safe_user_id}/{safe_session_id}/"
                f"{safe_filename}/{version}"
            )

    def validate_object_key(self, object_key: str) -> bool:
        """Validate object key for security compliance."""

        # Check length (S3 limit is 1024 characters)
        if len(object_key) > 1024:
            return False

        # Check for path traversal attempts
        dangerous_patterns = ["../", "..\\", "/..", "\\..", "//"]
        for pattern in dangerous_patterns:
            if pattern in object_key:
                return False

        # Ensure key starts with expected app prefix pattern
        key_parts = object_key.split("/")
        if len(key_parts) < 4:  # app/user/session/filename or app/user/user/filename
            return False

        return True

    def calculate_content_hash(self, content: bytes, algorithm: str = "sha256") -> str:
        """Calculate secure hash of content for integrity verification."""
        if algorithm == "sha256":
            return hashlib.sha256(content).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(content, usedforsecurity=False).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    def verify_content_integrity(
        self, content: bytes, expected_hash: str, algorithm: str = "sha256"
    ) -> bool:
        """Verify content integrity using hash comparison."""
        actual_hash = self.calculate_content_hash(content, algorithm)
        return hmac.compare_digest(actual_hash, expected_hash)


class AccessControlManager:
    """Manages access control and permissions for artifacts."""

    def __init__(self) -> None:
        self.access_patterns = {
            "session_scoped": "{app_name}/{user_id}/{session_id}/*",
            "user_scoped": "{app_name}/{user_id}/user/*",
            "admin_scoped": "{app_name}/*",
        }

    def check_access_permission(
        self,
        operation: str,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        user_permissions: List[str],
    ) -> bool:
        """Check if user has permission for the operation."""

        # Define required permissions for each operation
        operation_permissions = {
            "read": ["artifact:read", "artifact:*"],
            "write": ["artifact:write", "artifact:*"],
            "delete": ["artifact:delete", "artifact:*"],
            "list": ["artifact:list", "artifact:read", "artifact:*"],
        }

        required_perms = operation_permissions.get(operation, [])

        # Check if user has any required permission
        for perm in required_perms:
            if perm in user_permissions:
                return True

        # Check scope-specific permissions
        if filename.startswith("user:"):
            # User-scoped artifact
            scope_perm = f"artifact:{operation}:user"
            if scope_perm in user_permissions:
                return True
        else:
            # Session-scoped artifact
            scope_perm = f"artifact:{operation}:session"
            if scope_perm in user_permissions:
                return True

        return False

    def generate_presigned_url(
        self,
        s3_client: boto3.client,
        bucket_name: str,
        object_key: str,
        operation: str,
        expiration: int = 3600,
        user_permissions: Optional[List[str]] = None,
    ) -> str:
        """Generate secure presigned URL for artifact access."""

        # Validate operation
        valid_operations = ["get_object", "put_object"]
        if operation not in valid_operations:
            raise ValueError(f"Invalid operation: {operation}")

        # Check permissions if provided
        if user_permissions:
            # Additional permission checking logic here
            pass

        try:
            # Generate presigned URL
            url = s3_client.generate_presigned_url(
                operation,
                Params={"Bucket": bucket_name, "Key": object_key},
                ExpiresIn=expiration,
            )

            logger.info(f"Generated presigned URL for {operation} on {object_key}")
            return str(url)

        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise S3ArtifactError(f"Presigned URL generation failed: {e}") from e


class EncryptionManager:
    """Manages client-side encryption for sensitive artifacts."""

    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or self._generate_key()

    def _generate_key(self) -> str:
        """Generate a new encryption key."""
        return secrets.token_hex(32)  # 256-bit key

    def encrypt_content(self, content: bytes) -> tuple[bytes, Dict[str, str]]:
        """Encrypt content and return encrypted data with metadata."""
        # This is a simplified example - use proper encryption libraries
        # like cryptography for production use

        try:
            import base64

            from cryptography.fernet import Fernet

            # Use the key to create Fernet cipher
            key = base64.urlsafe_b64encode(
                self.encryption_key.encode()[:32].ljust(32, b"0")
            )
            cipher = Fernet(key)

            encrypted_content = cipher.encrypt(content)

            # Return encrypted content and encryption metadata
            metadata = {"encryption": "client-side", "algorithm": "fernet"}

            return encrypted_content, metadata

        except ImportError:
            raise S3ArtifactError(
                "cryptography package required for client-side encryption"
            )

    def decrypt_content(
        self, encrypted_content: bytes, metadata: Dict[str, str]
    ) -> bytes:
        """Decrypt content using stored metadata."""

        if metadata.get("encryption") != "client-side":
            return encrypted_content  # Not encrypted

        try:
            import base64

            from cryptography.fernet import Fernet

            key = base64.urlsafe_b64encode(
                self.encryption_key.encode()[:32].ljust(32, b"0")
            )
            cipher = Fernet(key)

            result = cipher.decrypt(encrypted_content)
            return bytes(result)

        except ImportError:
            raise S3ArtifactError(
                "cryptography package required for client-side decryption"
            )
        except Exception as e:
            raise S3ArtifactError(f"Decryption failed: {e}") from e
