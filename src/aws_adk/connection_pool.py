"""S3 connection pool management for optimized performance."""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)


class S3ConnectionPool:
    """Manages pooled S3 client connections for better performance."""

    def __init__(
        self,
        max_pool_connections: int = 50,
        max_workers: int = 20,
        connect_timeout: int = 60,
        read_timeout: int = 60,
        retries_config: Optional[Dict[str, Any]] = None,
    ):
        self.max_pool_connections = max_pool_connections
        self.max_workers = max_workers
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

        # Default retries configuration
        self.retries_config = retries_config or {"max_attempts": 3, "mode": "adaptive"}

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="s3-pool"
        )

        # Client cache with connection reuse
        self._clients: Dict[str, boto3.client] = {}
        self._client_lock = threading.Lock()

        # Connection pool statistics
        self._stats = {
            "total_connections": 0,
            "active_connections": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def get_client(
        self,
        region_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> boto3.client:
        """Get or create optimized S3 client with connection pooling."""

        # Create cache key
        cache_key = f"{region_name}:{aws_access_key_id}:{endpoint_url}"

        with self._client_lock:
            if cache_key in self._clients:
                self._stats["cache_hits"] += 1
                return self._clients[cache_key]

            self._stats["cache_misses"] += 1

            # Create optimized boto3 config
            config = Config(
                region_name=region_name,
                retries=self.retries_config,
                max_pool_connections=self.max_pool_connections,
                connect_timeout=self.connect_timeout,
                read_timeout=self.read_timeout,
                # Enable signature_version for better performance
                signature_version="s3v4",
                # Use virtual hosted-style addressing
                addressing_style="virtual",
            )

            # Create session
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )

            # Create client with configuration
            client_kwargs = {"config": config}
            if endpoint_url:
                client_kwargs["endpoint_url"] = endpoint_url

            client = session.client("s3", **client_kwargs)

            # Cache the client
            self._clients[cache_key] = client
            self._stats["total_connections"] += 1

            logger.info(f"Created new S3 client for region {region_name}")
            return client

    async def execute_async(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute S3 operation asynchronously using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, func, *args, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._client_lock:
            stats = self._stats.copy()
            stats["active_connections"] = len(self._clients)
            return stats

    def close(self) -> None:
        """Close all connections and cleanup resources."""
        with self._client_lock:
            self._clients.clear()

        self._executor.shutdown(wait=True)
        logger.info("S3 connection pool closed")

    def __del__(self) -> None:
        """Cleanup on object destruction."""
        self.close()


# Global connection pool instance
_connection_pool: Optional[S3ConnectionPool] = None
_pool_lock = threading.Lock()


def get_connection_pool() -> S3ConnectionPool:
    """Get or create global S3 connection pool."""
    global _connection_pool

    with _pool_lock:
        if _connection_pool is None:
            _connection_pool = S3ConnectionPool()
        return _connection_pool


def close_connection_pool() -> None:
    """Close global connection pool."""
    global _connection_pool

    with _pool_lock:
        if _connection_pool is not None:
            _connection_pool.close()
            _connection_pool = None
