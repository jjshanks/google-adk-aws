#!/usr/bin/env python3
"""
Demo script showcasing Phase 3 error handling and edge case features of
S3ArtifactService.

This script demonstrates:
1. Comprehensive error handling with automatic boto3 error mapping
2. Input validation and sanitization capabilities
3. Edge case management (large files, concurrency, network failures)
4. Service health monitoring and circuit breaker patterns
5. Content integrity verification and corruption detection

Usage:
    Copy examples/.env.example to examples/.env and configure:
    S3_BUCKET_NAME=your-test-bucket
    AWS_REGION=us-east-1
    python examples/phase3_error_handling_demo.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

from google.genai import types

# Add the src directory to the path so we can import our package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.config import (  # noqa: E402
    get_optional_env,
    load_demo_config,
    print_config_summary,
)

from aws_adk import (  # noqa: E402
    RetryConfig,
    S3ArtifactService,
    S3ConcurrencyError,
    S3ValidationError,
)


async def demo_comprehensive_error_handling():
    """Demonstrate comprehensive error handling capabilities."""
    print("=== Comprehensive Error Handling Demo ===")

    bucket_name = get_optional_env("S3_BUCKET_NAME", "test-demo-bucket")

    # Create service with all Phase 3 features enabled
    service = S3ArtifactService(
        bucket_name=bucket_name,
        region_name=get_optional_env("AWS_REGION", "us-east-1"),
        enable_validation=True,  # Input validation and sanitization
        enable_security_checks=True,  # Security validation
        enable_integrity_checks=True,  # Content integrity verification
        retry_config=RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=60.0,
            backoff_strategy="exponential",
        ),
    )

    print(f"✅ Created S3ArtifactService with bucket: {bucket_name}")

    # Test input validation and sanitization
    print("\n--- Input Validation & Sanitization ---")
    try:
        # Test with potentially dangerous input
        dangerous_filename = "../../../etc/passwd"

        print(f"Testing dangerous filename: {dangerous_filename}")

        # The service will sanitize this automatically
        artifact = types.Part.from_text("Test content", mime_type="text/plain")

        version = await service.save_artifact(
            app_name="demo-app",
            user_id="test-user",
            session_id="session-123",
            filename=dangerous_filename,  # This will be sanitized
            artifact=artifact,
        )
        print(f"✅ Input sanitized and artifact saved with version: {version}")

    except S3ValidationError as e:
        print(f"❌ Validation error (expected): {e.message}")
        print(f"   Context: {e.context}")

    # Test service health monitoring
    print("\n--- Service Health Monitoring ---")
    health = await service.get_service_health()
    print("✅ Service health check:")
    print(f"   Bucket: {health['bucket_name']}")
    print(f"   Region: {health['region_name']}")
    print(f"   Feature flags: {health['feature_flags']}")
    print(
        f"   Read circuit breaker: {health['circuit_breaker_stats']['read']['state']}"
    )
    print(
        f"   Write circuit breaker: {health['circuit_breaker_stats']['write']['state']}"
    )

    return service


async def demo_edge_case_handling():
    """Demonstrate edge case handling capabilities."""
    print("\n=== Edge Case Handling Demo ===")

    bucket_name = get_optional_env("S3_BUCKET_NAME", "test-demo-bucket")

    service = S3ArtifactService(
        bucket_name=bucket_name,
        enable_validation=True,
        enable_security_checks=True,
        enable_integrity_checks=True,
    )

    # Test large file handling
    print("\n--- Large File Handling ---")
    try:
        # Create a moderately large file (1MB)
        large_content = "A" * (1024 * 1024)  # 1MB of 'A's
        large_artifact = types.Part.from_text(large_content, mime_type="text/plain")

        print(f"Creating large artifact ({len(large_content)} bytes)...")

        version = await service.save_artifact(
            app_name="demo-app",
            user_id="test-user",
            session_id="session-123",
            filename="large-file.txt",
            artifact=large_artifact,
        )
        print(f"✅ Large file saved successfully with version: {version}")

        # Load it back and verify
        loaded_artifact = await service.load_artifact(
            app_name="demo-app",
            user_id="test-user",
            session_id="session-123",
            filename="large-file.txt",
            version=version,
        )

        if loaded_artifact and loaded_artifact.inline_data:
            loaded_size = len(loaded_artifact.inline_data.data)
            print(f"✅ Large file loaded successfully ({loaded_size} bytes)")

            # Verify content integrity
            if loaded_artifact.inline_data.data == large_content.encode():
                print("✅ Content integrity verified")
            else:
                print("❌ Content integrity failed")

    except Exception as e:
        print(f"❌ Large file handling error: {e}")

    # Test content validation
    print("\n--- Content Validation ---")
    try:
        # Test with invalid content type
        invalid_artifact = types.Part.from_bytes(
            b"invalid json content", mime_type="application/json"
        )

        await service.save_artifact(
            app_name="demo-app",
            user_id="test-user",
            session_id="session-123",
            filename="invalid.json",
            artifact=invalid_artifact,
        )
        print("❌ Should have failed validation")

    except S3ValidationError as e:
        print(f"✅ Content validation caught invalid JSON: {e.message}")

    # Test user namespace handling
    print("\n--- User Namespace Handling ---")
    try:
        user_artifact = types.Part.from_text("User data", mime_type="text/plain")

        version = await service.save_artifact(
            app_name="demo-app",
            user_id="test-user",
            session_id="session-123",
            filename="user:profile.txt",  # User-scoped
            artifact=user_artifact,
        )
        print(f"✅ User-scoped artifact saved with version: {version}")

    except Exception as e:
        print(f"❌ User namespace error: {e}")


async def demo_error_recovery():
    """Demonstrate error recovery and retry mechanisms."""
    print("\n=== Error Recovery & Retry Demo ===")

    bucket_name = get_optional_env("S3_BUCKET_NAME", "test-demo-bucket")

    # Create service with aggressive retry settings for demo
    service = S3ArtifactService(
        bucket_name=bucket_name,
        retry_config=RetryConfig(
            max_attempts=3,
            base_delay=0.5,
            max_delay=5.0,
            backoff_strategy="exponential",
        ),
        enable_validation=True,
    )

    print("--- Circuit Breaker & Retry Logic ---")

    # Simulate normal operations to show circuit breaker working
    try:
        artifact = types.Part.from_text("Recovery test", mime_type="text/plain")

        version = await service.save_artifact(
            app_name="demo-app",
            user_id="test-user",
            session_id="session-123",
            filename="recovery-test.txt",
            artifact=artifact,
        )
        print(f"✅ Artifact saved with retry protection, version: {version}")

        # Load it back
        loaded = await service.load_artifact(
            app_name="demo-app",
            user_id="test-user",
            session_id="session-123",
            filename="recovery-test.txt",
        )

        if loaded:
            print("✅ Artifact loaded with retry protection")
        else:
            print("❌ Failed to load artifact")

    except Exception as e:
        print(f"❌ Error recovery failed: {e}")

    # Check circuit breaker stats after operations
    health = await service.get_service_health()
    read_stats = health["circuit_breaker_stats"]["read"]
    write_stats = health["circuit_breaker_stats"]["write"]

    print("✅ Circuit breaker status after operations:")
    print(
        f"   Read CB - State: {read_stats['state']}, "
        f"Failures: {read_stats['failure_count']}"
    )
    print(
        f"   Write CB - State: {write_stats['state']}, "
        f"Failures: {write_stats['failure_count']}"
    )


async def demo_concurrency_handling():
    """Demonstrate concurrency control and conflict resolution."""
    print("\n=== Concurrency Control Demo ===")

    bucket_name = get_optional_env("S3_BUCKET_NAME", "test-demo-bucket")

    service = S3ArtifactService(
        bucket_name=bucket_name,
        enable_validation=True,
    )

    print("--- Concurrent Operations ---")

    # Create multiple tasks that operate on the same artifact
    async def save_task(task_id: int) -> Any:
        try:
            artifact = types.Part.from_text(
                f"Content from task {task_id}", mime_type="text/plain"
            )

            version = await service.save_artifact(
                app_name="demo-app",
                user_id="test-user",
                session_id="session-123",
                filename="concurrent-test.txt",
                artifact=artifact,
            )
            print(f"✅ Task {task_id} completed, version: {version}")
            return version

        except S3ConcurrencyError as e:
            print(f"⏱️ Task {task_id} handled concurrency: {e.message}")
            return None

        except Exception as e:
            print(f"❌ Task {task_id} failed: {e}")
            return None

    # Run multiple concurrent save operations
    print("Running 3 concurrent save operations...")
    tasks = [save_task(i) for i in range(1, 4)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful_saves = [r for r in results if isinstance(r, int)]
    print(f"✅ {len(successful_saves)} operations completed successfully")


async def demo_cleanup():
    """Clean up demo artifacts."""
    print("\n=== Cleanup Demo Artifacts ===")

    bucket_name = get_optional_env("S3_BUCKET_NAME", "test-demo-bucket")

    service = S3ArtifactService(bucket_name=bucket_name)

    try:
        # List all artifacts for our demo session
        artifacts = await service.list_artifact_keys(
            app_name="demo-app", user_id="test-user", session_id="session-123"
        )

        print(f"Found {len(artifacts)} demo artifacts to clean up")

        # Delete each artifact
        for filename in artifacts:
            try:
                await service.delete_artifact(
                    app_name="demo-app",
                    user_id="test-user",
                    session_id="session-123",
                    filename=filename,
                )
                print(f"✅ Deleted: {filename}")

            except Exception as e:
                print(f"❌ Failed to delete {filename}: {e}")

        # Cleanup service resources
        await service.cleanup_resources()
        print("✅ Service resources cleaned up")

    except Exception as e:
        print(f"❌ Cleanup error: {e}")


async def main():
    """Run all Phase 3 demos."""
    print("Phase 3 Error Handling & Edge Cases Demo")
    print("=" * 50)

    # Load configuration
    load_demo_config()
    print_config_summary()

    try:
        # Run all demo sections
        await demo_comprehensive_error_handling()
        await demo_edge_case_handling()
        await demo_error_recovery()
        await demo_concurrency_handling()

        print("\n" + "=" * 50)
        print("All Phase 3 demos completed successfully!")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")

    finally:
        # Always clean up
        await demo_cleanup()
        print("\nDemo complete!")


if __name__ == "__main__":
    # Load config first to check for required variables
    load_demo_config()

    # Check for required environment variables
    bucket_name = get_optional_env("S3_BUCKET_NAME")
    if not bucket_name:
        print("Error: S3_BUCKET_NAME environment variable is required")
        print("Please copy examples/.env.example to examples/.env and configure it")
        sys.exit(1)

    # Run the demo
    asyncio.run(main())
