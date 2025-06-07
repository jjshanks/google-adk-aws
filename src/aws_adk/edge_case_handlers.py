"""Handlers for edge cases in S3 artifact operations."""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .exceptions import (
    S3ConcurrencyError,
    S3CorruptionError,
    S3StorageQuotaError,
    S3ValidationError,
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
        self, app_name: str, user_id: str, session_id: str, filename: str
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
        timeout: float = 30.0,
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
                start_time=time.time(),
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
                context={"resource_key": resource_key, "timeout": timeout},
            )

    async def release_operation_lock(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        operation_id: str,
    ) -> None:
        """Release operation lock."""

        resource_key = self._get_resource_key(app_name, user_id, session_id, filename)

        # Remove from active operations
        if resource_key in self._active_operations:
            self._active_operations[resource_key] = [
                ctx
                for ctx in self._active_operations[resource_key]
                if ctx.operation_id != operation_id
            ]

            if not self._active_operations[resource_key]:
                del self._active_operations[resource_key]

        # Release lock
        if resource_key in self._locks:
            self._locks[resource_key].release()
            logger.debug(
                f"Released lock for operation {operation_id} on {resource_key}"
            )

    async def _check_operation_conflicts(
        self, resource_key: str, operation_type: str, operation_id: str
    ) -> None:
        """Check for conflicting operations on the same resource."""

        if resource_key not in self._active_operations:
            return

        active_ops = self._active_operations[resource_key]
        current_time = time.time()

        # Clean up stale operations
        active_ops = [
            ctx
            for ctx in active_ops
            if current_time - ctx.start_time < ctx.max_duration
        ]
        self._active_operations[resource_key] = active_ops

        # Check for conflicts
        conflicting_ops = {
            "write": ["write", "delete"],
            "delete": ["write", "read", "delete"],
            "read": ["write", "delete"],
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
                        "resource_key": resource_key,
                    },
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
        available_memory: Optional[int] = None,
    ) -> None:
        """Validate that large file operation can proceed."""

        # Check size limits
        if content_size > 5 * 1024 * 1024 * 1024 * 1024:  # 5TB S3 limit
            raise S3ValidationError(
                message=f"File size ({content_size}) exceeds S3 maximum (5TB)",
                error_code="FileTooLarge",
                context={"content_size": content_size},
            )

        # Check memory availability for large files
        if self.should_use_streaming(content_size):
            if available_memory and available_memory < self.MAX_MEMORY_USAGE:
                raise S3StorageQuotaError(
                    message="Insufficient memory for large file operation",
                    error_code="InsufficientMemory",
                    context={
                        "required_memory": self.MAX_MEMORY_USAGE,
                        "available_memory": available_memory,
                    },
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
        progress_callback: Optional[Callable] = None,
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
        progress_callback: Optional[Callable] = None,
    ) -> Any:
        """Handle multipart upload for very large files."""

        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        total_size = len(content)
        uploaded = 0

        try:
            for i in range(0, total_size, chunk_size):
                chunk = content[i : i + chunk_size]

                # Upload chunk
                result = await upload_func(chunk, i // chunk_size + 1)
                uploaded += len(chunk)

                # Progress callback
                if progress_callback:
                    progress_callback(uploaded, total_size)

                logger.debug(
                    f"Uploaded chunk {i // chunk_size + 1}, "
                    f"{uploaded}/{total_size} bytes"
                )

            return result

        except Exception as e:
            logger.error(
                f"Multipart upload failed at {uploaded}/{total_size} bytes: {e}"
            )
            raise

    async def _standard_upload(
        self,
        content: bytes,
        upload_func: Callable,
        progress_callback: Optional[Callable] = None,
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

    def __init__(self) -> None:
        self.hash_algorithm = "sha256"

    async def verify_content_integrity(
        self,
        original_content: bytes,
        retrieved_content: bytes,
        expected_hash: Optional[str] = None,
    ) -> None:
        """Verify content integrity after transfer."""

        # Size check
        if len(original_content) != len(retrieved_content):
            raise S3CorruptionError(
                message="Content size mismatch detected",
                error_code="SizeMismatch",
                context={
                    "original_size": len(original_content),
                    "retrieved_size": len(retrieved_content),
                },
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
                        "actual_hash": actual_hash,
                    },
                )

        # Byte-by-byte comparison for critical data
        if original_content != retrieved_content:
            raise S3CorruptionError(
                message="Content corruption detected during transfer",
                error_code="ContentCorruption",
            )

        logger.debug("Content integrity verification passed")

    def _calculate_hash(self, content: bytes) -> str:
        """Calculate content hash for verification."""
        import hashlib

        return hashlib.sha256(content).hexdigest()


class NetworkFailureHandler:
    """Handles network failures and connectivity issues."""

    def __init__(self) -> None:
        self.connection_pool_stats: Dict[str, Any] = {}
        self.failure_history: List[Dict[str, Any]] = []

    async def handle_network_failure(
        self, error: Exception, operation: str, retry_count: int
    ) -> bool:
        """Handle network failure and determine if retry should proceed."""

        self.failure_history.append(
            {
                "timestamp": time.time(),
                "error": str(error),
                "operation": operation,
                "retry_count": retry_count,
            }
        )

        # Clean old history (keep last hour)
        cutoff_time = time.time() - 3600
        self.failure_history = [
            entry for entry in self.failure_history if entry["timestamp"] > cutoff_time
        ]

        # Check failure rate
        recent_failures = len(self.failure_history)
        if recent_failures > 10:  # More than 10 failures in last hour
            logger.warning(
                f"High network failure rate detected: {recent_failures} failures"
            )
            return False  # Don't retry

        # Analyze error type
        error_str = str(error).lower()

        # Temporary network issues - retry
        if any(
            term in error_str
            for term in ["timeout", "connection", "network", "dns", "socket"]
        ):
            logger.info(f"Network error detected, will retry: {error}")
            return True

        # Permanent issues - don't retry
        if any(
            term in error_str
            for term in ["authentication", "authorization", "forbidden", "invalid"]
        ):
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
            "last_failure": self.failure_history[-1] if self.failure_history else None,
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
