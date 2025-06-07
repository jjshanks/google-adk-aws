#!/usr/bin/env python3
"""
S3-enabled agent implementation demonstrating integration with Google ADK.

This script demonstrates:
1. Creating an agent with S3 artifact storage capabilities
2. File management tools for saving, loading, listing, and versioning
3. User vs session-scoped file handling
4. Interactive agent conversation with persistent storage

Usage:
    Copy examples/.env.example to examples/.env and configure:
    S3_BUCKET_NAME=your-test-bucket
    AWS_REGION=us-east-1
    python examples/agent_integration_demo.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from google.adk import Agent
from google.genai import types
from google.genai.types import GenerateContentConfig

# Add the src directory to the path so we can import our package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.config import (  # noqa: E402
    get_optional_env,
    get_required_env,
    load_demo_config,
    print_config_summary,
)

from aws_adk import RetryConfig, S3ArtifactError, S3ArtifactService  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agent prompts and instructions
GLOBAL_INSTRUCTION = """
You are a helpful AI assistant that can save and retrieve files using S3 storage.
Always be helpful, accurate, and provide clear feedback about file operations.
"""

MAIN_INSTRUCTION = """
You are an S3 File Management Agent with the following capabilities:

Key Features:
- Save text content to files with automatic versioning
- Load previously saved files (latest version or specific version)
- Manage user-specific files that persist across sessions
- List all available files for the current session

File Scoping:
- Regular files (e.g., "report.txt"): Available only in current session
- User files (e.g., "user:preferences.json"): Persist across all sessions for the user

Guidelines:
1. Always validate file names and content before operations
2. Provide clear feedback about successful operations
3. Use appropriate MIME types (text/plain, application/json, etc.)
4. Suggest using "user:" prefix for files that should persist across sessions
5. When errors occur, explain what went wrong and suggest solutions

Response Format:
- For successful operations: Confirm what was done and provide relevant details
- For file retrieval: Show content preview or full content as appropriate
- For errors: Explain the issue and suggest next steps
- For questions: Provide helpful information about capabilities and usage

Remember: Use the save_user_data and load_user_data tools for all file operations.
"""


async def save_data_to_artifact(
    data: str,
    filename: str,
    artifact_service: S3ArtifactService,
    app_name: str = "s3_demo_app",
    user_id: str = "demo_user",
    session_id: str = "demo_session",
) -> Dict[str, Any]:
    """Save text data as an artifact to S3."""
    try:
        logger.info(f"Saving artifact: {filename} with {len(data)} characters")

        # Validate inputs
        if not data:
            return {"status": "error", "message": "Cannot save empty content"}

        if not filename or not isinstance(filename, str):
            return {"status": "error", "message": "Filename must be a non-empty string"}

        # Create artifact from text data
        artifact = types.Part(text=data)

        # Save to S3
        version = await artifact_service.save_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            artifact=artifact,
        )

        logger.info(f"Successfully saved {filename} as version {version}")
        return {
            "status": "success",
            "version": version,
            "message": f"Saved {filename} version {version}",
        }

    except S3ArtifactError as e:
        logger.error(f"S3 error saving artifact {filename}: {e}")
        return {"status": "error", "message": f"S3 operation failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error saving artifact {filename}: {e}")
        return {"status": "error", "message": f"Failed to save artifact: {str(e)}"}


async def load_artifact_data(
    filename: str,
    artifact_service: S3ArtifactService,
    version: Optional[int] = None,
    app_name: str = "s3_demo_app",
    user_id: str = "demo_user",
    session_id: str = "demo_session",
) -> Dict[str, Any]:
    """Load artifact data from S3."""
    try:
        logger.info(f"Loading artifact: {filename}, version: {version}")

        # Validate filename
        if not filename or not isinstance(filename, str):
            return {"status": "error", "message": "Filename must be a non-empty string"}

        # Load from S3
        artifact = await artifact_service.load_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            version=version,
        )

        if artifact is None:
            logger.debug(f"Artifact not found: {filename}")
            return {
                "status": "not_found",
                "message": f"Artifact '{filename}' not found",
            }

        # Extract content and metadata
        content = (
            artifact.text
            if hasattr(artifact, "text")
            else str(artifact.inline_data.data, "utf-8")
        )
        mime_type = artifact.inline_data.mime_type

        # If version wasn't specified, get the actual version loaded
        if version is None:
            versions = await artifact_service.list_versions(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
            )
            actual_version = max(versions) if versions else 0
        else:
            actual_version = version

        logger.info(f"Successfully loaded {filename} version {actual_version}")
        return {
            "status": "success",
            "content": content,
            "mime_type": mime_type,
            "version": actual_version,
            "message": f"Loaded {filename} version {actual_version}",
        }

    except S3ArtifactError as e:
        logger.error(f"S3 error loading artifact {filename}: {e}")
        return {"status": "error", "message": f"S3 operation failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error loading artifact {filename}: {e}")
        return {"status": "error", "message": f"Failed to load artifact: {str(e)}"}


async def list_artifact_files(
    artifact_service: S3ArtifactService,
    app_name: str = "s3_demo_app",
    user_id: str = "demo_user",
    session_id: str = "demo_session",
) -> Dict[str, Any]:
    """List all artifact files available for the current session."""
    try:
        logger.info("Listing available artifact files")

        # List all artifacts for the session
        artifact_keys = await artifact_service.list_artifact_keys(
            app_name=app_name, user_id=user_id, session_id=session_id
        )

        logger.info(f"Found {len(artifact_keys)} artifact files")
        return {
            "status": "success",
            "files": artifact_keys,
            "count": len(artifact_keys),
            "message": f"Found {len(artifact_keys)} artifact files",
        }

    except S3ArtifactError as e:
        logger.error(f"S3 error listing artifacts: {e}")
        return {"status": "error", "message": f"S3 operation failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error listing artifacts: {e}")
        return {"status": "error", "message": f"Failed to list artifacts: {str(e)}"}


async def get_artifact_versions(
    filename: str,
    artifact_service: S3ArtifactService,
    app_name: str = "s3_demo_app",
    user_id: str = "demo_user",
    session_id: str = "demo_session",
) -> Dict[str, Any]:
    """Get all available versions for a specific artifact."""
    try:
        logger.info(f"Getting versions for artifact: {filename}")

        # Validate filename
        if not filename or not isinstance(filename, str):
            return {"status": "error", "message": "Filename must be a non-empty string"}

        # List all versions
        versions = await artifact_service.list_versions(
            app_name=app_name, user_id=user_id, session_id=session_id, filename=filename
        )

        latest_version = max(versions) if versions else None

        logger.info(f"Found {len(versions)} versions for {filename}")
        return {
            "status": "success",
            "versions": versions,
            "count": len(versions),
            "latest": latest_version,
            "message": f"Found {len(versions)} versions for {filename}",
        }

    except S3ArtifactError as e:
        logger.error(f"S3 error getting versions for {filename}: {e}")
        return {"status": "error", "message": f"S3 operation failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error getting versions for {filename}: {e}")
        return {"status": "error", "message": f"Failed to get versions: {str(e)}"}


def create_s3_artifact_service() -> S3ArtifactService:
    """Create and configure S3ArtifactService from environment variables."""
    bucket_name = get_required_env("S3_BUCKET_NAME")

    # Enhanced configuration with Phase 2 features
    enable_encryption = (
        get_optional_env("S3_ENABLE_ENCRYPTION", "false").lower() == "true"
    )

    return S3ArtifactService(
        bucket_name=bucket_name,
        region_name=get_optional_env("AWS_REGION", "us-east-1"),
        aws_access_key_id=get_optional_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=get_optional_env("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=get_optional_env("AWS_SESSION_TOKEN"),
        endpoint_url=get_optional_env("S3_ENDPOINT_URL"),
        # Phase 2 enhancements
        enable_encryption=enable_encryption,
        encryption_key=get_optional_env("S3_ENCRYPTION_KEY"),
        retry_config=RetryConfig(
            max_attempts=int(get_optional_env("S3_RETRY_MAX_ATTEMPTS", "5")),
            base_delay=float(get_optional_env("S3_RETRY_BASE_DELAY", "1.0")),
            max_delay=float(get_optional_env("S3_RETRY_MAX_DELAY", "30.0")),
        ),
    )


def create_s3_agent() -> Agent:
    """Create an agent with S3 artifact capabilities."""
    # Initialize S3 artifact service
    artifact_service = create_s3_artifact_service()

    def save_user_data(content: str, filename: str) -> str:
        """
        Save text content to a file using S3 artifact storage.

        This tool saves text content to a file with automatic versioning.
        Files are stored in S3 and can be retrieved later in the same session
        or across sessions (if using 'user:' prefix).

        Args:
            content: The text content to save
            filename: Name for the file (use 'user:filename' for
                cross-session persistence)

        Returns:
            str: Confirmation message with version information

        Example:
            save_user_data("Hello, World!", "greeting.txt")
            save_user_data('{"theme": "dark"}', "user:preferences.json")
        """

        async def _save():
            result = await save_data_to_artifact(
                data=content, filename=filename, artifact_service=artifact_service
            )

            if result["status"] == "success":
                return f"✓ {result['message']}"
            else:
                return f"✗ Error: {result['message']}"

        return asyncio.run(_save())

    def load_user_data(filename: str, version: int = None) -> str:
        """
        Load text content from a previously saved file.

        This tool retrieves content from a file stored in S3 artifact storage.
        By default, loads the latest version unless a specific version is requested.

        Args:
            filename: Name of the file to load
            version: Specific version to load (optional, loads latest if not specified)

        Returns:
            str: File content or error message

        Example:
            load_user_data("greeting.txt")
            load_user_data("report.txt", version=2)
        """

        async def _load():
            result = await load_artifact_data(
                filename=filename, artifact_service=artifact_service, version=version
            )

            if result["status"] == "success":
                content = result["content"]
                version_info = (
                    f" (version {result['version']})" if "version" in result else ""
                )
                return f"Content of {filename}{version_info}:\\n\\n{content}"
            elif result["status"] == "not_found":
                return (
                    f"✗ File '{filename}' not found. Use list_user_files "
                    "to see available files."
                )
            else:
                return f"✗ Error loading file: {result['message']}"

        return asyncio.run(_load())

    def list_user_files() -> str:
        """
        List all files available in the current session.

        This tool shows all files that can be accessed, including both
        session-scoped files and user-scoped files (those with 'user:' prefix).

        Returns:
            str: Formatted list of available files
        """

        async def _list():
            result = await list_artifact_files(artifact_service=artifact_service)

            if result["status"] == "success":
                files = result["files"]
                if not files:
                    return "No files found. Use save_user_data to create some files!"

                output = f"Found {result['count']} files:\\n\\n"
                session_files = [f for f in files if not f.startswith("user:")]
                user_files = [f for f in files if f.startswith("user:")]

                if session_files:
                    output += "📄 Session files:\\n"
                    for f in session_files:
                        output += f"  • {f}\\n"

                if user_files:
                    if session_files:
                        output += "\\n"
                    output += "👤 User files (persist across sessions):\\n"
                    for f in user_files:
                        output += f"  • {f}\\n"

                return output
            else:
                return f"✗ Error listing files: {result['message']}"

        return asyncio.run(_list())

    def get_file_versions(filename: str) -> str:
        """
        Get version history for a specific file.

        This tool shows all available versions of a file, which is useful
        for understanding the change history and selecting specific versions to load.

        Args:
            filename: Name of the file to check versions for

        Returns:
            str: Formatted list of available versions
        """

        async def _versions():
            result = await get_artifact_versions(
                filename=filename, artifact_service=artifact_service
            )

            if result["status"] == "success":
                versions = result["versions"]
                if not versions:
                    return f"No versions found for '{filename}'. File may not exist."

                output = f"Version history for '{filename}':\\n\\n"
                for v in sorted(versions, reverse=True):
                    marker = " (latest)" if v == result["latest"] else ""
                    output += f"  • Version {v}{marker}\\n"

                return output
            else:
                return f"✗ Error getting versions: {result['message']}"

        return asyncio.run(_versions())

    # Create agent with S3 tools
    agent = Agent(
        name="s3_file_agent",
        model="gemini-2.0-flash-001",
        global_instruction=GLOBAL_INSTRUCTION,
        instruction=MAIN_INSTRUCTION,
        tools=[save_user_data, load_user_data, list_user_files, get_file_versions],
        artifact_service=artifact_service,
        generate_config=GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192,
        ),
    )

    return agent


async def main():
    """Interactive demo of the S3-enabled agent."""
    try:
        # Load configuration
        load_demo_config()
        print_config_summary()

        # Create the agent (for validation)
        create_s3_agent()

        print("S3 File Management Agent Demo")
        print("=" * 50)
        print("Available commands:")
        print("  • Ask me to save content: 'Save this text to myfile.txt'")
        print("  • Ask me to load files: 'Load the content of myfile.txt'")
        print("  • Ask me to list files: 'What files are available?'")
        print("  • Ask about versions: 'What versions exist for myfile.txt?'")
        print("  • Type 'quit' to exit")
        print()

        # Simple interactive loop (you could use ADK runners for more
        # sophisticated interaction)
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # For demo purposes, we'll simulate a simple agent response
            # In a real implementation, you'd use ADK runners for proper
            # conversation flow
            print(
                "Agent: I'm ready to help with file operations! "
                "(This is a simplified demo)"
            )
            print(
                "       Use the Google ADK runners for full conversation "
                "capabilities."
            )
            print()

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Error: {e}")
        print(
            "Make sure your AWS credentials and S3_BUCKET_NAME are "
            "configured correctly."
        )


if __name__ == "__main__":
    asyncio.run(main())
