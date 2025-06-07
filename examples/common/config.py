"""Shared configuration module for all demo scripts."""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_demo_config(env_file: Optional[str] = None) -> None:
    """Load environment configuration for demos.

    Args:
        env_file: Optional path to .env file. If None, looks for .env in examples
            directory.
    """
    if env_file is None:
        # Look for .env file in examples directory
        examples_dir = Path(__file__).parent.parent
        env_file = examples_dir / ".env"

    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")
    else:
        print(f"No .env file found at {env_file}")
        print("Using environment variables from system")


def get_required_env(var_name: str) -> str:
    """Get a required environment variable, exiting if not found.

    Args:
        var_name: Name of the environment variable

    Returns:
        The environment variable value

    Raises:
        SystemExit: If the required variable is not found
    """
    value = os.environ.get(var_name)
    if not value:
        print(f"Error: Required environment variable {var_name} not found")
        print("Please check your .env file or system environment variables")
        sys.exit(1)
    return value


def get_optional_env(var_name: str, default: str = "") -> str:
    """Get an optional environment variable with a default value.

    Args:
        var_name: Name of the environment variable
        default: Default value if variable is not found

    Returns:
        The environment variable value or default
    """
    return os.environ.get(var_name, default)


def print_config_summary() -> None:
    """Print a summary of current configuration."""
    print("\n=== Demo Configuration Summary ===")
    print(f"S3 Bucket: {get_optional_env('S3_BUCKET_NAME', 'Not set')}")
    print(f"AWS Region: {get_optional_env('AWS_REGION', 'Not set')}")
    print(f"AWS Profile: {get_optional_env('AWS_PROFILE', 'default')}")
    print(
        f"S3 Endpoint URL: {get_optional_env('S3_ENDPOINT_URL', 'Not set (using AWS)')}"
    )
    print("=" * 35)
