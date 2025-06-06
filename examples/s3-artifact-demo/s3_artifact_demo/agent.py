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

"""S3-enabled agent implementation."""

import asyncio
import logging
import os

from dotenv import load_dotenv
from google.adk import Agent
from google.genai.types import GenerateContentConfig

from aws_adk import RetryConfig, S3ArtifactService

from .prompt import GLOBAL_INSTRUCTION, MAIN_INSTRUCTION
from .tools.file_tools import (
    get_artifact_versions,
    list_artifact_files,
    load_artifact_data,
    save_data_to_artifact,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_s3_artifact_service() -> S3ArtifactService:
    """Create and configure S3ArtifactService from environment variables.

    Returns:
        Configured S3ArtifactService instance with Phase 2 enhancements

    Raises:
        ValueError: If required environment variables are missing
    """
    bucket_name = os.getenv("S3_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable is required")

    # Enhanced configuration with Phase 2 features
    enable_encryption = os.getenv("S3_ENABLE_ENCRYPTION", "false").lower() == "true"

    return S3ArtifactService(
        bucket_name=bucket_name,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        # Phase 2 enhancements
        enable_encryption=enable_encryption,
        encryption_key=os.getenv("S3_ENCRYPTION_KEY"),
        retry_config=RetryConfig(
            max_attempts=int(os.getenv("S3_RETRY_MAX_ATTEMPTS", "5")),
            base_delay=float(os.getenv("S3_RETRY_BASE_DELAY", "1.0")),
            max_delay=float(os.getenv("S3_RETRY_MAX_DELAY", "30.0")),
        ),
    )


def create_s3_agent() -> Agent:
    """Create an agent with S3 artifact capabilities.

    Returns:
        Agent configured with S3 file management tools
    """
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
                return f"Content of {filename}{version_info}:\n\n{content}"
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

                output = f"Found {result['count']} files:\n\n"
                session_files = [f for f in files if not f.startswith("user:")]
                user_files = [f for f in files if f.startswith("user:")]

                if session_files:
                    output += "📄 Session files:\n"
                    for f in session_files:
                        output += f"  • {f}\n"

                if user_files:
                    if session_files:
                        output += "\n"
                    output += "👤 User files (persist across sessions):\n"
                    for f in user_files:
                        output += f"  • {f}\n"

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

                output = f"Version history for '{filename}':\n\n"
                for v in sorted(versions, reverse=True):
                    marker = " (latest)" if v == result["latest"] else ""
                    output += f"  • Version {v}{marker}\n"

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
        # Create the agent (for validation)
        create_s3_agent()

        print("🚀 S3 File Management Agent Demo")
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
                print("👋 Goodbye!")
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
        print(f"❌ Error: {e}")
        print(
            "Make sure your AWS credentials and S3_BUCKET_NAME are "
            "configured correctly."
        )


if __name__ == "__main__":
    asyncio.run(main())
