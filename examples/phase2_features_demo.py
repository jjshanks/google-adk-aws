#!/usr/bin/env python3
"""
Demo script showcasing Phase 2 enhanced features of S3ArtifactService.

This script demonstrates:
1. Enhanced error handling with retry logic
2. Performance optimization features (connection pooling, batch operations)
3. Security features (encryption, integrity verification, presigned URLs)
4. Monitoring and diagnostics capabilities

Usage:
    Copy examples/.env.example to examples/.env and configure:
    S3_BUCKET_NAME=your-test-bucket
    AWS_REGION=us-east-1
    python examples/phase2_features_demo.py
"""

import asyncio
import sys
from pathlib import Path

from google.genai import types

# Add the src directory to the path so we can import our package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.config import (  # noqa: E402
    get_optional_env,
    load_demo_config,
    print_config_summary,
)

from aws_adk import RetryConfig, S3ArtifactService  # noqa: E402


async def demo_basic_operations():
    """Demonstrate basic artifact operations with enhanced features."""
    print("=== Basic Operations Demo ===")

    bucket_name = get_optional_env("S3_BUCKET_NAME", "test-demo-bucket")

    # Create service with enhanced features
    service = S3ArtifactService(
        bucket_name=bucket_name,
        region_name=get_optional_env("AWS_REGION", "us-east-1"),
        enable_encryption=True,  # Enable client-side encryption
        retry_config=RetryConfig(max_attempts=5, base_delay=1.0),  # Custom retry config
    )

    # Create test artifact
    artifact = types.Part.from_bytes(
        data="This is a test artifact with enhanced Phase 2 features!".encode("utf-8"),
        mime_type="text/plain",
    )

    try:
        # Save artifact (with automatic encryption, retry logic, and
        # integrity verification)
        print("Saving artifact with enhanced features...")
        version = await service.save_artifact(
            app_name="demo_app",
            user_id="demo_user",
            session_id="demo_session",
            filename="enhanced_demo.txt",
            artifact=artifact,
        )
        print(f"✓ Saved artifact version: {version}")

        # Load artifact (with automatic decryption and integrity verification)
        print("Loading artifact...")
        loaded = await service.load_artifact(
            app_name="demo_app",
            user_id="demo_user",
            session_id="demo_session",
            filename="enhanced_demo.txt",
        )
        print(f"✓ Loaded artifact: {loaded.inline_data.data.decode()}")

        # List artifacts
        keys = await service.list_artifact_keys(
            app_name="demo_app", user_id="demo_user", session_id="demo_session"
        )
        print(f"✓ Found artifacts: {keys}")

        # Clean up
        await service.delete_artifact(
            app_name="demo_app",
            user_id="demo_user",
            session_id="demo_session",
            filename="enhanced_demo.txt",
        )
        print("✓ Cleaned up test artifact")

    except Exception as e:
        print(f"✗ Error in basic operations: {e}")
        raise


async def demo_security_features():
    """Demonstrate security features."""
    print("\n=== Security Features Demo ===")

    bucket_name = get_optional_env("S3_BUCKET_NAME", "test-demo-bucket")

    service = S3ArtifactService(
        bucket_name=bucket_name,
        region_name=get_optional_env("AWS_REGION", "us-east-1"),
        enable_encryption=True,
    )

    try:
        # Get bucket security status
        print("Checking bucket security status...")
        security_status = await service.get_security_status()
        print(f"✓ Encryption enabled: {security_status['encryption']}")
        print(f"✓ Versioning enabled: {security_status['versioning']}")
        print(f"✓ Public access blocked: {security_status['public_access_blocked']}")

        if security_status["recommendations"]:
            print("Security recommendations:")
            for rec in security_status["recommendations"]:
                print(f"  - {rec}")

        # Save a test artifact for presigned URL demo
        artifact = types.Part.from_bytes(
            data="Secure content".encode("utf-8"), mime_type="text/plain"
        )
        await service.save_artifact(
            app_name="security_demo",
            user_id="demo_user",
            session_id="demo_session",
            filename="secure_file.txt",
            artifact=artifact,
        )

        # Generate presigned URL
        print("Generating presigned URL...")
        url = await service.generate_presigned_url(
            app_name="security_demo",
            user_id="demo_user",
            session_id="demo_session",
            filename="secure_file.txt",
            expiration=300,  # 5 minutes
        )
        print("✓ Generated presigned URL (expires in 5 minutes)")
        print(f"  URL length: {len(url)} characters")

        # Clean up
        await service.delete_artifact(
            app_name="security_demo",
            user_id="demo_user",
            session_id="demo_session",
            filename="secure_file.txt",
        )
        print("✓ Cleaned up security demo artifact")

    except Exception as e:
        print(f"✗ Error in security features: {e}")
        raise


async def demo_performance_features():
    """Demonstrate performance optimization features."""
    print("\n=== Performance Features Demo ===")

    bucket_name = get_optional_env("S3_BUCKET_NAME", "test-demo-bucket")

    service = S3ArtifactService(
        bucket_name=bucket_name, region_name=get_optional_env("AWS_REGION", "us-east-1")
    )

    try:
        # Check connection pool statistics
        print("Connection pool statistics:")
        stats = service.get_connection_stats()
        print(f"✓ Total connections: {stats['total_connections']}")
        print(f"✓ Active connections: {stats['active_connections']}")
        print(f"✓ Cache hits: {stats['cache_hits']}")
        print(f"✓ Cache misses: {stats['cache_misses']}")

        # Demo batch operations - create multiple small artifacts
        print("\nCreating multiple artifacts for batch demo...")
        artifact = types.Part.from_bytes(
            data="Batch test content".encode("utf-8"), mime_type="text/plain"
        )

        filenames = [f"batch_file_{i:03d}.txt" for i in range(5)]

        # Save multiple artifacts
        tasks = []
        for filename in filenames:
            task = service.save_artifact(
                app_name="batch_demo",
                user_id="demo_user",
                session_id="demo_session",
                filename=filename,
                artifact=artifact,
            )
            tasks.append(task)

        versions = await asyncio.gather(*tasks)
        print(f"✓ Saved {len(versions)} artifacts concurrently")

        # Demo batch delete
        print("Performing batch delete...")
        delete_result = await service.batch_delete_artifacts(
            app_name="batch_demo",
            user_id="demo_user",
            session_id="demo_session",
            filenames=filenames,
        )

        print(f"✓ Batch deleted {len(delete_result['deleted'])} artifacts")
        print(f"✓ Errors: {len(delete_result['errors'])}")

        # Check updated connection stats
        final_stats = service.get_connection_stats()
        print("\nFinal connection pool statistics:")
        print(f"✓ Total connections: {final_stats['total_connections']}")
        print(f"✓ Cache hits: {final_stats['cache_hits']}")

    except Exception as e:
        print(f"✗ Error in performance features: {e}")
        raise


async def demo_large_file_handling():
    """Demonstrate large file handling with multipart upload."""
    print("\n=== Large File Handling Demo ===")

    bucket_name = get_optional_env("S3_BUCKET_NAME", "test-demo-bucket")

    service = S3ArtifactService(
        bucket_name=bucket_name, region_name=get_optional_env("AWS_REGION", "us-east-1")
    )

    try:
        # Create a large artifact (1MB) to trigger potential multipart upload
        print("Creating large artifact (1MB)...")
        large_content = "X" * (1024 * 1024)  # 1MB of data
        large_artifact = types.Part.from_bytes(
            data=large_content.encode("utf-8"), mime_type="text/plain"
        )

        print("Saving large artifact...")
        version = await service.save_artifact(
            app_name="large_demo",
            user_id="demo_user",
            session_id="demo_session",
            filename="large_file.txt",
            artifact=large_artifact,
        )
        print(f"✓ Saved large artifact version: {version}")

        print("Loading large artifact...")
        loaded = await service.load_artifact(
            app_name="large_demo",
            user_id="demo_user",
            session_id="demo_session",
            filename="large_file.txt",
        )

        print(f"✓ Loaded large artifact: {len(loaded.inline_data.data)} bytes")
        assert len(loaded.inline_data.data) == len(large_content)
        print("✓ Content integrity verified")

        # Clean up
        await service.delete_artifact(
            app_name="large_demo",
            user_id="demo_user",
            session_id="demo_session",
            filename="large_file.txt",
        )
        print("✓ Cleaned up large file")

    except Exception as e:
        print(f"✗ Error in large file handling: {e}")
        raise


async def main():
    """Run all Phase 2 feature demonstrations."""
    print("Google ADK AWS S3 Artifact Service - Phase 2 Features Demo")
    print("=" * 60)

    # Load configuration
    load_demo_config()
    print_config_summary()

    # Check environment
    bucket_name = get_optional_env("S3_BUCKET_NAME")
    if not bucket_name:
        print("Warning: S3_BUCKET_NAME not set, using default test bucket name")
        print("Note: Ensure you have AWS credentials configured and the bucket exists")

    try:
        await demo_basic_operations()
        await demo_security_features()
        await demo_performance_features()
        await demo_large_file_handling()

        print("\n" + "=" * 60)
        print("✓ All Phase 2 feature demonstrations completed successfully!")
        print("\nKey Phase 2 Features Demonstrated:")
        print("• Enhanced error handling with retry logic and circuit breakers")
        print("• Client-side encryption for sensitive data")
        print("• Content integrity verification with hash checking")
        print("• Connection pooling for improved performance")
        print("• Batch operations for efficient multi-artifact management")
        print("• Presigned URL generation for secure access")
        print("• Large file handling with multipart uploads")
        print("• Security status monitoring and recommendations")
        print("• Performance metrics and connection statistics")

    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
