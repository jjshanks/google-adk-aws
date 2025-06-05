# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone demonstration of S3 artifact operations."""

import asyncio
import logging
import os

from dotenv import load_dotenv
from google.genai import types

from aws_adk.s3_artifact_service import S3ArtifactService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_s3_artifacts():
    """Complete demonstration of S3 artifact operations."""

    # Check required environment variables
    bucket_name = os.getenv("S3_BUCKET_NAME")
    if not bucket_name:
        print("❌ Error: S3_BUCKET_NAME environment variable is required")
        print("Please create a .env file based on .env.example")
        return

    # Initialize S3 artifact service
    try:
        artifact_service = S3ArtifactService(
            bucket_name=bucket_name,
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        )
    except Exception as e:
        print(f"❌ Error initializing S3 service: {e}")
        print("Please check your AWS credentials and bucket configuration")
        return

    # Sample data to work with
    sample_data = {
        "report.txt": "This is a sample report with important findings.",
        "user:preferences.json": (
            '{"theme": "dark", "language": "en", "notifications": true}'
        ),
        "analysis.csv": (
            "date,value,category\n2024-01-01,100,A\n2024-01-02,150,B\n"
            "2024-01-03,200,A"
        ),
        "notes.md": (
            "# Meeting Notes\n\n- Discussed project timeline\n"
            "- Reviewed budget allocation\n- Next steps defined"
        ),
    }

    app_name = "s3_demo_app"
    user_id = "demo_user"
    session_id = "demo_session"

    try:
        print("🚀 S3 Artifact Service Demo")
        print("=" * 50)
        print(f"Bucket: {bucket_name}")
        print(f"Region: {os.getenv('AWS_REGION', 'us-east-1')}")
        print()

        # 1. Save multiple artifacts
        print("📁 1. Saving sample artifacts...")
        for filename, content in sample_data.items():
            artifact = types.Part(text=content)

            version = await artifact_service.save_artifact(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
                artifact=artifact,
            )
            scope = "👤 User" if filename.startswith("user:") else "📄 Session"
            print(f"  ✓ {scope}: {filename} → version {version}")

        print()

        # 2. List all artifacts
        print("📋 2. Listing all artifacts...")
        artifact_keys = await artifact_service.list_artifact_keys(
            app_name=app_name, user_id=user_id, session_id=session_id
        )

        session_files = [f for f in artifact_keys if not f.startswith("user:")]
        user_files = [f for f in artifact_keys if f.startswith("user:")]

        if session_files:
            print("  📄 Session-scoped files:")
            for f in session_files:
                print(f"     • {f}")

        if user_files:
            print("  👤 User-scoped files:")
            for f in user_files:
                print(f"     • {f}")

        print(f"  Total: {len(artifact_keys)} files")
        print()

        # 3. Load and preview artifacts
        print("📖 3. Loading artifact content...")
        for filename in artifact_keys:
            artifact = await artifact_service.load_artifact(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
            )
            if artifact:
                content = artifact.text
                preview = content[:60] + "..." if len(content) > 60 else content
                preview = preview.replace("\n", " ")
                print(f"  📄 {filename}: {preview}")
        print()

        # 4. Demonstrate versioning
        print("🔄 4. Testing versioning...")
        test_filename = "versioned_document.txt"

        # Save multiple versions
        for i in range(4):
            content = (
                f"Document version {i}\nUpdated at step {i}\n"
                f"Content: {'A' * (10 + i * 5)}"
            )
            artifact = types.Part(text=content)
            version = await artifact_service.save_artifact(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=test_filename,
                artifact=artifact,
            )
            print(f"  ✓ Created version {version} ({len(content)} chars)")

        # List versions
        versions = await artifact_service.list_versions(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=test_filename,
        )
        print(f"  📚 Available versions: {versions}")

        # Load specific versions
        print("  📖 Version comparison:")
        for v in [0, len(versions) - 1]:  # First and last versions
            artifact_v = await artifact_service.load_artifact(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=test_filename,
                version=v,
            )
            content_preview = artifact_v.text.split("\n")[0]  # First line only
            print(f"     Version {v}: {content_preview}")
        print()

        # 5. Cross-session artifact access
        print("🔄 5. Testing cross-session access...")
        different_session = "different_session_123"

        # List artifacts from different session (should include user files)
        cross_session_artifacts = await artifact_service.list_artifact_keys(
            app_name=app_name, user_id=user_id, session_id=different_session
        )

        user_artifacts = [f for f in cross_session_artifacts if f.startswith("user:")]
        print(f"  👤 User files available in other session: {user_artifacts}")

        if user_artifacts:
            # Load user file from different session
            user_file = user_artifacts[0]
            artifact = await artifact_service.load_artifact(
                app_name=app_name,
                user_id=user_id,
                session_id=different_session,
                filename=user_file,
            )
            if artifact:
                print(f"  ✓ Successfully loaded {user_file} from different session")
        print()

        # 6. Cleanup demonstration
        print("🧹 6. Cleanup operations...")
        cleanup_files = ["report.txt", "analysis.csv", "notes.md", test_filename]

        for filename in cleanup_files:
            try:
                await artifact_service.delete_artifact(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                    filename=filename,
                )
                print(f"  ✓ Deleted {filename}")
            except Exception as e:
                print(f"  ⚠️  Could not delete {filename}: {e}")

        # Final count
        final_artifacts = await artifact_service.list_artifact_keys(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        print(f"  📊 Artifacts remaining: {len(final_artifacts)}")

        print()
        print("✅ Demo completed successfully!")
        print("💡 Key takeaways:")
        print("   • Use 'user:' prefix for files that persist across sessions")
        print("   • Versioning is automatic and incremental")
        print("   • Different MIME types are supported automatically")
        print("   • Error handling provides clear feedback")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Error during demo: {e}")
        print(
            "Please check your AWS credentials, bucket permissions, and "
            "network connectivity."
        )


async def interactive_demo():
    """Interactive mode for testing S3 operations."""

    bucket_name = os.getenv("S3_BUCKET_NAME")
    if not bucket_name:
        print("❌ Error: S3_BUCKET_NAME environment variable is required")
        return

    try:
        artifact_service = S3ArtifactService(
            bucket_name=bucket_name,
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    except Exception as e:
        print(f"❌ Error initializing S3 service: {e}")
        return

    print("🎮 Interactive S3 Artifact Demo")
    print("=" * 40)
    print("Commands:")
    print("  save <filename> <content>")
    print("  load <filename> [version]")
    print("  list")
    print("  versions <filename>")
    print("  delete <filename>")
    print("  quit")
    print()

    app_name = "interactive_demo"
    user_id = "demo_user"
    session_id = "interactive_session"

    while True:
        try:
            command = input("📝 > ").strip().split(" ", 2)

            if not command or command[0] == "":
                continue

            cmd = command[0].lower()

            if cmd == "quit":
                print("👋 Goodbye!")
                break

            elif cmd == "save" and len(command) >= 3:
                filename = command[1]
                content = command[2]

                artifact = types.Part(text=content)
                version = await artifact_service.save_artifact(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                    filename=filename,
                    artifact=artifact,
                )
                print(f"✓ Saved {filename} as version {version}")

            elif cmd == "load" and len(command) >= 2:
                filename = command[1]
                version = int(command[2]) if len(command) > 2 else None

                artifact = await artifact_service.load_artifact(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                    filename=filename,
                    version=version,
                )

                if artifact:
                    print(f"📄 Content of {filename}:")
                    print(artifact.text)
                else:
                    print(f"❌ File {filename} not found")

            elif cmd == "list":
                artifacts = await artifact_service.list_artifact_keys(
                    app_name=app_name, user_id=user_id, session_id=session_id
                )
                if artifacts:
                    print(f"📁 Found {len(artifacts)} files:")
                    for f in artifacts:
                        print(f"  • {f}")
                else:
                    print("📁 No files found")

            elif cmd == "versions" and len(command) >= 2:
                filename = command[1]
                versions = await artifact_service.list_versions(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                    filename=filename,
                )
                if versions:
                    print(f"📚 Versions of {filename}: {versions}")
                else:
                    print(f"❌ No versions found for {filename}")

            elif cmd == "delete" and len(command) >= 2:
                filename = command[1]
                await artifact_service.delete_artifact(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                    filename=filename,
                )
                print(f"✓ Deleted {filename}")

            else:
                print("❌ Invalid command. Type 'quit' to exit.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Main entry point with mode selection."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await interactive_demo()
    else:
        await demonstrate_s3_artifacts()


if __name__ == "__main__":
    asyncio.run(main())
