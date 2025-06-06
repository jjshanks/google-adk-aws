"""Batch operations for improved S3 performance."""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

import boto3

from .exceptions import S3ArtifactError

logger = logging.getLogger(__name__)


class S3BatchOperations:
    """Optimized batch operations for S3 artifacts."""

    def __init__(
        self, s3_client: boto3.client, bucket_name: str, max_concurrent: int = 10
    ):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def batch_delete(self, object_keys: List[str]) -> Dict[str, Any]:
        """Delete multiple objects in batches for efficiency."""
        if not object_keys:
            return {"deleted": [], "errors": []}

        results: Dict[str, List[Any]] = {"deleted": [], "errors": []}

        # S3 supports up to 1000 objects per delete request
        batch_size = 1000

        for i in range(0, len(object_keys), batch_size):
            batch = object_keys[i : i + batch_size]

            try:
                await self._delete_batch(batch, results)
            except Exception as e:
                logger.error(
                    f"Batch delete failed for batch starting at index {i}: {e}"
                )
                # Add all objects in failed batch to errors
                for key in batch:
                    results["errors"].append({"key": key, "error": str(e)})

        return results

    async def _delete_batch(
        self, object_keys: List[str], results: Dict[str, Any]
    ) -> None:
        """Delete a single batch of objects."""
        async with self._semaphore:
            # Prepare delete request
            delete_request = {
                "Objects": [{"Key": key} for key in object_keys],
                "Quiet": False,  # Return both successful and failed deletes
            }

            def _execute_delete() -> Any:
                return self.s3_client.delete_objects(
                    Bucket=self.bucket_name, Delete=delete_request
                )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _execute_delete)

            # Process results
            for deleted in response.get("Deleted", []):
                results["deleted"].append(deleted["Key"])

            for error in response.get("Errors", []):
                results["errors"].append(
                    {
                        "key": error["Key"],
                        "error": f"{error['Code']}: {error['Message']}",
                    }
                )

    async def batch_upload(
        self,
        upload_specs: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Upload multiple objects concurrently."""
        if not upload_specs:
            return {"uploaded": [], "errors": []}

        results: Dict[str, List[Any]] = {"uploaded": [], "errors": []}

        # Create upload tasks with concurrency control
        tasks = []
        for spec in upload_specs:
            task = self._upload_single(spec, results, progress_callback)
            tasks.append(task)

        # Execute uploads with limited concurrency
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def _upload_single(
        self,
        upload_spec: Dict[str, Any],
        results: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Upload a single object."""
        async with self._semaphore:
            try:

                def _execute_upload() -> Any:
                    return self.s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=upload_spec["key"],
                        Body=upload_spec["body"],
                        ContentType=upload_spec.get(
                            "content_type", "application/octet-stream"
                        ),
                        Metadata=upload_spec.get("metadata", {}),
                    )

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, _execute_upload)

                results["uploaded"].append(upload_spec["key"])

                if progress_callback:
                    progress_callback(upload_spec["key"], "uploaded")

            except Exception as e:
                logger.error(f"Upload failed for {upload_spec['key']}: {e}")
                results["errors"].append({"key": upload_spec["key"], "error": str(e)})

                if progress_callback:
                    progress_callback(upload_spec["key"], "error", str(e))

    async def batch_list_objects(
        self, prefixes: List[str], max_keys_per_prefix: int = 1000
    ) -> Dict[str, List[Dict[str, Any]]]:
        """List objects for multiple prefixes concurrently."""
        if not prefixes:
            return {}

        results = {}

        # Create list tasks
        tasks = []
        for prefix in prefixes:
            task = self._list_objects_for_prefix(prefix, max_keys_per_prefix)
            tasks.append((prefix, task))

        # Execute listing operations
        for prefix, task in tasks:
            try:
                objects = await task
                results[prefix] = objects
            except Exception as e:
                logger.error(f"List objects failed for prefix {prefix}: {e}")
                results[prefix] = []

        return results

    async def _list_objects_for_prefix(
        self, prefix: str, max_keys: int
    ) -> List[Dict[str, Any]]:
        """List objects for a single prefix."""
        async with self._semaphore:

            def _execute_list() -> List[Dict[str, Any]]:
                paginator = self.s3_client.get_paginator("list_objects_v2")
                objects = []

                for page in paginator.paginate(
                    Bucket=self.bucket_name, Prefix=prefix, MaxKeys=max_keys
                ):
                    objects.extend(page.get("Contents", []))

                return objects

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _execute_list)


class MultipartUploadManager:
    """Manages multipart uploads for large artifacts."""

    def __init__(self, s3_client: boto3.client, bucket_name: str):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.multipart_threshold = 100 * 1024 * 1024  # 100MB
        self.part_size = 10 * 1024 * 1024  # 10MB parts

    async def upload_large_artifact(
        self,
        object_key: str,
        data: bytes,
        content_type: str,
        metadata: Dict[str, str],
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """Upload large artifact using multipart upload if necessary."""

        if len(data) < self.multipart_threshold:
            # Use regular upload for smaller files
            return await self._regular_upload(object_key, data, content_type, metadata)
        else:
            # Use multipart upload for large files
            return await self._multipart_upload(
                object_key, data, content_type, metadata, progress_callback
            )

    async def _regular_upload(
        self, object_key: str, data: bytes, content_type: str, metadata: Dict[str, str]
    ) -> str:
        """Regular S3 upload for smaller files."""

        def _execute_upload() -> Any:
            return self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=data,
                ContentType=content_type,
                Metadata=metadata,
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _execute_upload)
        return str(response["ETag"])

    async def _multipart_upload(
        self,
        object_key: str,
        data: bytes,
        content_type: str,
        metadata: Dict[str, str],
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """Multipart upload for large files."""

        # Initiate multipart upload
        def _initiate_upload() -> Any:
            return self.s3_client.create_multipart_upload(
                Bucket=self.bucket_name,
                Key=object_key,
                ContentType=content_type,
                Metadata=metadata,
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _initiate_upload)
        upload_id = response["UploadId"]

        try:
            # Upload parts
            parts = await self._upload_parts(
                object_key, upload_id, data, progress_callback
            )

            # Complete multipart upload
            def _complete_upload() -> Any:
                return self.s3_client.complete_multipart_upload(
                    Bucket=self.bucket_name,
                    Key=object_key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )

            response = await loop.run_in_executor(None, _complete_upload)
            return str(response["ETag"])

        except Exception as e:
            # Abort multipart upload on error
            try:

                def _abort_upload() -> None:
                    self.s3_client.abort_multipart_upload(
                        Bucket=self.bucket_name, Key=object_key, UploadId=upload_id
                    )

                await loop.run_in_executor(None, _abort_upload)
            except Exception as abort_error:
                logger.error(f"Failed to abort multipart upload: {abort_error}")

            raise S3ArtifactError(f"Multipart upload failed: {e}") from e

    async def _upload_parts(
        self,
        object_key: str,
        upload_id: str,
        data: bytes,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Upload individual parts of multipart upload."""

        # Calculate parts
        total_size = len(data)
        num_parts = (total_size + self.part_size - 1) // self.part_size

        # Upload parts concurrently
        tasks = []
        for part_num in range(1, num_parts + 1):
            start = (part_num - 1) * self.part_size
            end = min(start + self.part_size, total_size)
            part_data = data[start:end]

            task = self._upload_part(
                object_key, upload_id, part_num, part_data, progress_callback
            )
            tasks.append(task)

        # Wait for all parts to complete
        parts = await asyncio.gather(*tasks)

        # Sort parts by part number
        return sorted(parts, key=lambda p: p["PartNumber"])

    async def _upload_part(
        self,
        object_key: str,
        upload_id: str,
        part_number: int,
        part_data: bytes,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Upload a single part."""

        def _execute_part_upload() -> Any:
            return self.s3_client.upload_part(
                Bucket=self.bucket_name,
                Key=object_key,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=part_data,
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _execute_part_upload)

        if progress_callback:
            progress_callback(part_number, len(part_data), "uploaded")

        return {"ETag": response["ETag"], "PartNumber": part_number}
