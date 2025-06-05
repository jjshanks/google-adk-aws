# Google ADK AWS Integrations - S3 Artifact Service Implementation - Phase 1

## Overview

Phase 1 establishes the foundational AWS integrations package structure and implements the core S3ArtifactService class with basic CRUD operations. This phase creates a standalone, installable Python package that provides AWS service integrations for Google ADK.

**Duration**: 2-3 days
**Deliverables**: External package setup, core S3ArtifactService class with all BaseArtifactService methods

## Prerequisites

- Python 3.10+ development environment
- Git for version control
- AWS account with S3 access (for testing)
- Google ADK installed for interface compatibility verification

## Phase 1 Objectives

1. **Repository Setup**: Create complete external package structure
2. **Core Implementation**: Implement all BaseArtifactService abstract methods
3. **Basic Testing**: Ensure methods can be called without errors
4. **Package Structure**: Establish proper Python packaging with dependencies
5. **Integration Verification**: Confirm compatibility with ADK imports

## Task Breakdown

### Task 1.1: Repository Initialization (30 minutes)

#### 1.1.1 Create Repository Structure
```bash
# Initialize new repository
mkdir google-adk-aws
cd google-adk-aws
git init

# Create complete directory structure
mkdir -p src/adk_s3_artifacts
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p docs
mkdir -p examples
mkdir -p .github/workflows

# Create essential Python files
touch src/adk_s3_artifacts/__init__.py
touch src/adk_s3_artifacts/s3_artifact_service.py
touch src/adk_s3_artifacts/py.typed
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

# Create configuration files
touch pyproject.toml
touch README.md
touch LICENSE
touch .gitignore
```

#### 1.1.2 Git Configuration
```bash
# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.coverage
.pytest_cache/
.tox/
htmlcov/

# AWS
.aws/
*.pem

# OS
.DS_Store
Thumbs.db
EOF

# Initial commit
git add .
git commit -m "Initial repository structure"
```

### Task 1.2: Package Configuration (45 minutes)

#### 1.2.1 Create pyproject.toml
**File**: `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "google-adk-aws"
dynamic = ["version"]
description = "AWS service integrations for Google Agent Development Kit (ADK)"
readme = "README.md"
license = {file = "LICENSE"}
authors = [
    {name = "Joshua Shanks", email = "jjshanks@gmail.com"}
]
maintainers = [
    {name = "Joshua Shanks", email = "jjshanks@gmail.com"}
]
keywords = ["google", "adk", "aws", "s3", "artifacts", "agents", "integrations"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    "Topic :: Scientific/Engineering :: Artificial Intelligence"
]
requires-python = ">=3.9"
dependencies = [
    "google-adk>=1.2.0",
    "boto3>=1.34.0",
    "botocore>=1.34.0",
]

[project.urls]
Homepage = "https://github.com/jjshanks/google-adk-aws"
Repository = "https://github.com/jjshanks/google-adk-aws.git"
Issues = "https://github.com/jjshanks/google-adk-aws/issues"
Documentation = "https://github.com/jjshanks/google-adk-aws#readme"
Changelog = "https://github.com/jjshanks/google-adk-aws/blob/main/CHANGELOG.md"

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "moto[s3]>=4.2.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "ruff>=0.1.0"
]
test = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "moto[s3]>=4.2.0"
]

[tool.setuptools.dynamic]
version = {attr = "adk_s3_artifacts.__version__"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
adk_s3_artifacts = ["py.typed"]

# Development tools configuration
[tool.black]
line-length = 80
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  \.eggs
  | \.git
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 80
multi_line_output = 3

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=adk_s3_artifacts",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=90"
]
asyncio_mode = "auto"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow running tests"
]
```

#### 1.2.2 Create LICENSE
**File**: `LICENSE`
```
MIT License

Copyright (c) 2025 Joshua Shanks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

#### 1.2.3 Create Initial README
**File**: `README.md`
```markdown
# ADK S3 Artifacts

[![PyPI version](https://badge.fury.io/py/adk-s3-artifacts.svg)](https://badge.fury.io/py/adk-s3-artifacts)
[![Python versions](https://img.shields.io/pypi/pyversions/adk-s3-artifacts.svg)](https://pypi.org/project/adk-s3-artifacts/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

S3 Artifact Service implementation for Google Agent Development Kit (ADK).

## Features

- **Full BaseArtifactService Implementation**: Complete compatibility with ADK's artifact interface
- **AWS S3 Integration**: Production-ready S3 storage with proper authentication
- **Flexible Authentication**: Support for IAM roles, access keys, and custom endpoints
- **Version Management**: Automatic artifact versioning with history tracking
- **User Namespace Support**: Session-scoped and user-scoped artifact storage

## Quick Start

### Installation

```bash
pip install adk-s3-artifacts
```

### Basic Usage

```python
from adk_s3_artifacts import S3ArtifactService
from google.adk.agents import Agent

# Create S3 artifact service
artifact_service = S3ArtifactService(
    bucket_name="my-artifacts-bucket",
    region_name="us-west-2"
)

# Use with ADK agent
agent = Agent(
    name="document_processor",
    model="gemini-2.0-flash",
    artifact_service=artifact_service
)
```

## Development Status

🚧 **Phase 1 - Core Implementation** (Current)
- [x] Repository setup
- [x] Package configuration
- [ ] Core S3ArtifactService implementation
- [ ] Basic testing

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Task 1.3: Core Package Implementation (2 hours)

#### 1.3.1 Package __init__.py
**File**: `src/adk_s3_artifacts/__init__.py`
```python
"""S3 Artifact Service for Google Agent Development Kit (ADK).

This package provides an S3-based implementation of ADK's BaseArtifactService,
enabling artifact storage and retrieval using Amazon S3 or S3-compatible services.
"""

from .s3_artifact_service import S3ArtifactService

__version__ = "0.1.0"
__author__ = "Joshua Shanks"
__email__ = "jjshanks@gmail.com"

__all__ = ["S3ArtifactService"]
```

#### 1.3.2 Core S3ArtifactService Implementation
**File**: `src/adk_s3_artifacts/s3_artifact_service.py`
```python
"""S3 Artifact Service implementation for Google ADK."""

import asyncio
import logging
from typing import Optional, Any, Dict
from concurrent.futures import ThreadPoolExecutor

import boto3
import botocore.exceptions
from google.adk.artifacts import BaseArtifactService
from google.genai import types

logger = logging.getLogger(__name__)


class S3ArtifactError(Exception):
    """Base exception for S3 artifact operations."""
    pass


class S3ConnectionError(S3ArtifactError):
    """Raised when S3 connection fails."""
    pass


class S3PermissionError(S3ArtifactError):
    """Raised when S3 permissions are insufficient."""
    pass


class S3ArtifactService(BaseArtifactService):
    """S3-based implementation of ADK's BaseArtifactService.

    Provides artifact storage and retrieval using Amazon S3 or S3-compatible
    services. Supports multiple authentication methods and automatic versioning.

    Attributes:
        bucket_name: S3 bucket name for artifact storage
        region_name: AWS region for S3 operations
        s3_client: Boto3 S3 client instance
    """

    def __init__(
        self,
        bucket_name: str,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize S3ArtifactService.

        Args:
            bucket_name: Name of S3 bucket to use for storage
            region_name: AWS region name (default: us-east-1)
            aws_access_key_id: AWS access key ID (optional, uses IAM if not provided)
            aws_secret_access_key: AWS secret access key (optional)
            aws_session_token: AWS session token for temporary credentials (optional)
            endpoint_url: Custom S3 endpoint URL for S3-compatible services (optional)
            **kwargs: Additional arguments passed to boto3.Session

        Raises:
            S3ConnectionError: If S3 client initialization fails
            S3PermissionError: If bucket access is denied
        """
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.endpoint_url = endpoint_url

        # Store credentials for session creation
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.session_kwargs = kwargs

        # Initialize S3 client
        self.s3_client = self._create_s3_client()

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="s3-artifact")

        # Verify bucket access
        self._verify_bucket_access()

    def _create_s3_client(self) -> boto3.client:
        """Create and configure S3 client with authentication.

        Returns:
            Configured boto3 S3 client

        Raises:
            S3ConnectionError: If client creation fails
        """
        try:
            # Create session with provided credentials
            session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                region_name=self.region_name,
                **self.session_kwargs
            )

            # Configure client options
            client_config = {}
            if self.endpoint_url:
                client_config["endpoint_url"] = self.endpoint_url

            return session.client("s3", **client_config)

        except Exception as e:
            raise S3ConnectionError(f"Failed to create S3 client: {e}") from e

    def _verify_bucket_access(self) -> None:
        """Verify that the bucket exists and is accessible.

        Raises:
            S3PermissionError: If bucket access is denied
            S3ArtifactError: If bucket doesn't exist or other error
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully verified access to bucket: {self.bucket_name}")
        except botocore.exceptions.ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "403":
                raise S3PermissionError(
                    f"Access denied to bucket {self.bucket_name}. "
                    "Check IAM permissions."
                ) from e
            elif error_code == "404":
                raise S3ArtifactError(
                    f"Bucket {self.bucket_name} does not exist"
                ) from e
            else:
                raise S3ArtifactError(f"Failed to access bucket: {e}") from e
        except Exception as e:
            raise S3ArtifactError(f"Unexpected error accessing bucket: {e}") from e

    def _file_has_user_namespace(self, filename: str) -> bool:
        """Check if filename has user namespace prefix.

        Args:
            filename: The filename to check

        Returns:
            True if filename starts with "user:", False otherwise
        """
        return filename.startswith("user:")

    def _get_object_key(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: int,
    ) -> str:
        """Construct S3 object key for artifact.

        Args:
            app_name: Application name
            user_id: User identifier
            session_id: Session identifier
            filename: Artifact filename
            version: Artifact version number

        Returns:
            S3 object key string
        """
        if self._file_has_user_namespace(filename):
            return f"{app_name}/{user_id}/user/{filename}/{version}"
        return f"{app_name}/{user_id}/{session_id}/{filename}/{version}"

    async def save_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        artifact: types.Part,
    ) -> int:
        """Save artifact to S3 storage.

        Args:
            app_name: Application name
            user_id: User identifier
            session_id: Session identifier
            filename: Artifact filename
            artifact: Artifact data as types.Part

        Returns:
            Version number of saved artifact (starting from 0)

        Raises:
            S3ArtifactError: If save operation fails
        """
        try:
            # Get existing versions to determine next version
            existing_versions = await self.list_versions(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
            )
            version = 0 if not existing_versions else max(existing_versions) + 1

            # Construct object key
            object_key = self._get_object_key(
                app_name, user_id, session_id, filename, version
            )

            # Prepare metadata
            metadata = {
                "app-name": app_name,
                "user-id": user_id,
                "session-id": session_id,
                "filename": filename,
                "version": str(version),
            }

            # Upload to S3 asynchronously
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_key,
                    Body=artifact.inline_data.data,
                    ContentType=artifact.inline_data.mime_type,
                    Metadata=metadata,
                ),
            )

            logger.info(f"Saved artifact {filename} version {version} to {object_key}")
            return version

        except Exception as e:
            logger.error(f"Failed to save artifact {filename}: {e}")
            raise S3ArtifactError(f"Failed to save artifact: {e}") from e

    async def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int] = None,
    ) -> Optional[types.Part]:
        """Load artifact from S3 storage.

        Args:
            app_name: Application name
            user_id: User identifier
            session_id: Session identifier
            filename: Artifact filename
            version: Specific version to load (latest if None)

        Returns:
            Artifact as types.Part or None if not found

        Raises:
            S3ArtifactError: If load operation fails
        """
        try:
            # Determine version if not specified
            if version is None:
                existing_versions = await self.list_versions(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                    filename=filename,
                )
                if not existing_versions:
                    return None
                version = max(existing_versions)

            # Construct object key
            object_key = self._get_object_key(
                app_name, user_id, session_id, filename, version
            )

            # Download from S3 asynchronously
            response = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: self.s3_client.get_object(
                    Bucket=self.bucket_name, Key=object_key
                ),
            )

            # Read data and create types.Part
            data = response["Body"].read()
            content_type = response["ContentType"]

            logger.info(f"Loaded artifact {filename} version {version} from {object_key}")
            return types.Part.from_bytes(data=data, mime_type=content_type)

        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.debug(f"Artifact not found: {filename} version {version}")
                return None
            logger.error(f"Failed to load artifact {filename}: {e}")
            raise S3ArtifactError(f"Failed to load artifact: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error loading artifact {filename}: {e}")
            raise S3ArtifactError(f"Failed to load artifact: {e}") from e

    async def list_artifact_keys(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> list[str]:
        """List all artifact filenames for a session.

        Args:
            app_name: Application name
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Sorted list of artifact filenames

        Raises:
            S3ArtifactError: If list operation fails
        """
        try:
            filenames = set()

            # List session-scoped artifacts
            session_prefix = f"{app_name}/{user_id}/{session_id}/"
            await self._list_filenames_with_prefix(session_prefix, filenames)

            # List user-scoped artifacts
            user_prefix = f"{app_name}/{user_id}/user/"
            await self._list_filenames_with_prefix(user_prefix, filenames)

            result = sorted(list(filenames))
            logger.debug(f"Listed {len(result)} artifacts for session {session_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to list artifact keys: {e}")
            raise S3ArtifactError(f"Failed to list artifacts: {e}") from e

    async def _list_filenames_with_prefix(
        self, prefix: str, filenames: set[str]
    ) -> None:
        """Helper to list filenames with given prefix.

        Args:
            prefix: S3 key prefix to search
            filenames: Set to add found filenames to
        """
        def _list_objects():
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    # Extract filename: app/user/session/filename/version
                    key_parts = obj["Key"].split("/")
                    if len(key_parts) >= 5:
                        filename = key_parts[-2]  # Second to last is filename
                        filenames.add(filename)

        await asyncio.get_event_loop().run_in_executor(self._executor, _list_objects)

    async def delete_artifact(
        self, *, app_name: str, user_id: str, session_id: str, filename: str
    ) -> None:
        """Delete all versions of an artifact.

        Args:
            app_name: Application name
            user_id: User identifier
            session_id: Session identifier
            filename: Artifact filename to delete

        Raises:
            S3ArtifactError: If delete operation fails
        """
        try:
            # Get all versions to delete
            versions = await self.list_versions(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
            )

            if not versions:
                logger.debug(f"No versions found for artifact {filename}")
                return

            # Delete all versions
            def _delete_versions():
                for version in versions:
                    object_key = self._get_object_key(
                        app_name, user_id, session_id, filename, version
                    )
                    self.s3_client.delete_object(
                        Bucket=self.bucket_name, Key=object_key
                    )

            await asyncio.get_event_loop().run_in_executor(
                self._executor, _delete_versions
            )

            logger.info(f"Deleted {len(versions)} versions of artifact {filename}")

        except Exception as e:
            logger.error(f"Failed to delete artifact {filename}: {e}")
            raise S3ArtifactError(f"Failed to delete artifact: {e}") from e

    async def list_versions(
        self, *, app_name: str, user_id: str, session_id: str, filename: str
    ) -> list[int]:
        """List all versions of an artifact.

        Args:
            app_name: Application name
            user_id: User identifier
            session_id: Session identifier
            filename: Artifact filename

        Returns:
            Sorted list of version numbers

        Raises:
            S3ArtifactError: If list operation fails
        """
        try:
            def _list_versions():
                # Use empty string for version to get prefix
                prefix = self._get_object_key(
                    app_name, user_id, session_id, filename, ""
                )[:-1]  # Remove trailing slash

                paginator = self.s3_client.get_paginator("list_objects_v2")
                versions = []

                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        # Extract version from key: .../filename/version
                        key_parts = obj["Key"].split("/")
                        if len(key_parts) >= 5:
                            try:
                                version = int(key_parts[-1])
                                versions.append(version)
                            except ValueError:
                                # Skip invalid version numbers
                                continue

                return sorted(versions)

            versions = await asyncio.get_event_loop().run_in_executor(
                self._executor, _list_versions
            )

            logger.debug(f"Found {len(versions)} versions for artifact {filename}")
            return versions

        except Exception as e:
            logger.error(f"Failed to list versions for {filename}: {e}")
            raise S3ArtifactError(f"Failed to list versions: {e}") from e

    def __del__(self) -> None:
        """Cleanup resources on object destruction."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
```

### Task 1.4: Basic Verification (30 minutes)

#### 1.4.1 Create Basic Test
**File**: `tests/unit/test_s3_artifact_service_basic.py`
```python
"""Basic verification tests for S3ArtifactService."""

import pytest
from adk_s3_artifacts import S3ArtifactService


class TestS3ArtifactServiceBasic:
    """Basic tests to verify package structure and imports."""

    def test_import_s3_artifact_service(self):
        """Test that S3ArtifactService can be imported."""
        assert S3ArtifactService is not None

    def test_s3_artifact_service_initialization(self):
        """Test basic initialization (without real S3 connection)."""
        # This will fail bucket verification, but that's expected
        with pytest.raises(Exception):
            S3ArtifactService(bucket_name="test-bucket")

    def test_s3_artifact_service_has_required_methods(self):
        """Test that all required BaseArtifactService methods exist."""
        required_methods = [
            'save_artifact',
            'load_artifact',
            'list_artifact_keys',
            'delete_artifact',
            'list_versions'
        ]

        for method_name in required_methods:
            assert hasattr(S3ArtifactService, method_name)
            method = getattr(S3ArtifactService, method_name)
            assert callable(method)
```

#### 1.4.2 Verify Package Installation
```bash
# Install in development mode
pip install -e .

# Run basic tests
python -m pytest tests/unit/test_s3_artifact_service_basic.py -v

# Verify imports work
python -c "from adk_s3_artifacts import S3ArtifactService; print('Import successful')"
```

### Task 1.5: Documentation and Finalization (15 minutes)

#### 1.5.1 Update README with Current Status
Update the development status section in README.md:

```markdown
## Development Status

🚧 **Phase 1 - Core Implementation** (Complete)
- [x] Repository setup
- [x] Package configuration
- [x] Core S3ArtifactService implementation
- [x] Basic verification tests

✅ **Ready for Phase 2**: Error handling and comprehensive testing
```

#### 1.5.2 Commit Phase 1 Completion
```bash
git add .
git commit -m "Phase 1: Complete core S3ArtifactService implementation

- Implement all BaseArtifactService abstract methods
- Add proper S3 authentication and bucket verification
- Include comprehensive error handling with custom exceptions
- Add async support with ThreadPoolExecutor
- Create basic verification tests
- Configure complete Python package structure"

git tag v0.1.0-phase1
```

## Phase 1 Deliverables Checklist

- [ ] Repository structure created with all necessary directories
- [ ] Package configuration (pyproject.toml) with proper dependencies
- [ ] Core S3ArtifactService class implementing all BaseArtifactService methods:
  - [ ] `save_artifact()` - Save with versioning
  - [ ] `load_artifact()` - Load specific or latest version
  - [ ] `list_artifact_keys()` - List all filenames
  - [ ] `delete_artifact()` - Delete all versions
  - [ ] `list_versions()` - List version numbers
- [ ] Proper async implementation using ThreadPoolExecutor
- [ ] S3 authentication with multiple credential sources
- [ ] Custom exception classes for error handling
- [ ] Basic verification tests confirming imports and method existence
- [ ] Package can be installed with `pip install -e .`
- [ ] README documentation with usage examples
- [ ] MIT license file
- [ ] Git repository with proper .gitignore

## Success Criteria for Phase 1

1. **Package Structure**: Complete Python package that can be installed and imported
2. **Interface Compliance**: All BaseArtifactService methods implemented with correct signatures
3. **S3 Integration**: Working S3 client with authentication and bucket verification
4. **Error Handling**: Custom exceptions for S3-specific errors
5. **Basic Testing**: Verification tests pass confirming package integrity
6. **Documentation**: Clear README with installation and usage instructions

## Next Steps (Phase 2 Preview)

Phase 2 will focus on:
- Comprehensive error handling and edge cases
- Complete test suite with moto mocking
- Performance optimizations
- Integration testing with real S3 bucket
- Advanced features like multipart uploads

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `google-adk` is installed: `pip install google-adk`
2. **AWS Credentials**: Set up AWS credentials via environment variables or IAM roles
3. **Package Installation**: Use `pip install -e .` for development installation
4. **Version Conflicts**: Use virtual environment to isolate dependencies

### Required Environment Variables for Testing
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
export S3_TEST_BUCKET=your-test-bucket
```
