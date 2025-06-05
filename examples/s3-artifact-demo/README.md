# S3 Artifact Service Demo

This example demonstrates how to use the AWS ADK S3ArtifactService with Google ADK agents for persistent artifact storage.

## Overview

The S3ArtifactService provides:
- **Persistent Storage**: Save and retrieve artifacts using Amazon S3
- **Automatic Versioning**: Multiple versions of the same artifact
- **User Scoping**: Session-scoped vs user-scoped artifacts
- **Flexible Authentication**: IAM roles, credentials, or S3-compatible services

## Features Demonstrated

- Basic S3 artifact operations (save, load, list, delete)
- Version management with automatic incrementation
- User-scoped artifacts that persist across sessions
- Agent integration with custom tools
- Error handling for common S3 scenarios

## Prerequisites

1. **AWS Account** with S3 access
2. **S3 Bucket** for artifact storage
3. **AWS Credentials** (IAM role, access keys, or environment variables)
4. **Python 3.9+** with the required dependencies

## Quick Start

1. **Set up environment variables:**
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_REGION=us-east-1
   export S3_BUCKET_NAME=your-artifact-bucket
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Run the demo:**
   ```bash
   python s3_artifact_demo/demo.py
   ```

4. **Run the agent example:**
   ```bash
   python s3_artifact_demo/agent.py
   ```

## Project Structure

```
s3-artifact-demo/
├── README.md                   # This file
├── pyproject.toml             # Dependencies and metadata
├── .env.example               # Environment variables template
├── s3_artifact_demo/          # Main package
│   ├── __init__.py
│   ├── agent.py               # Agent with S3 artifact support
│   ├── demo.py                # Standalone S3 operations demo
│   ├── prompt.py              # Agent instructions
│   └── tools/                 # Custom tools
│       ├── __init__.py
│       └── file_tools.py      # S3 artifact tools
└── tests/                     # Unit and integration tests
    ├── __init__.py
    └── test_s3_artifacts.py    # Test suite
```

## Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# Required
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-artifact-bucket

# Optional: For S3-compatible services
S3_ENDPOINT_URL=https://your-s3-compatible-service.com
```

### IAM Permissions

Your AWS credentials need the following S3 permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name",
                "arn:aws:s3:::your-bucket-name/*"
            ]
        }
    ]
}
```

## Usage Examples

### Basic S3 Operations

```python
from aws_adk.s3_artifact_service import S3ArtifactService
from google.genai import types

# Initialize service
artifact_service = S3ArtifactService(
    bucket_name="my-bucket",
    region_name="us-east-1"
)

# Save an artifact
artifact = types.Part.from_text("Hello, World!")
version = await artifact_service.save_artifact(
    app_name="my_app",
    user_id="user123",
    session_id="session456",
    filename="greeting.txt",
    artifact=artifact
)

# Load the artifact
loaded = await artifact_service.load_artifact(
    app_name="my_app",
    user_id="user123",
    session_id="session456",
    filename="greeting.txt"
)
print(loaded.text)  # "Hello, World!"
```

### Agent Integration

```python
from google.adk import Agent
from s3_artifact_demo.agent import create_s3_agent

# Create agent with S3 support
agent = create_s3_agent()

# The agent now has tools for saving/loading files
```

## Key Concepts

### Artifact Scoping

- **Session-scoped**: `filename.txt` - Available only in the current session
- **User-scoped**: `user:filename.txt` - Available across all sessions for the user

### Versioning

- Automatic version incrementing (0, 1, 2, ...)
- Load latest version by default or specify version number
- List all available versions for any artifact

### Error Handling

- `S3ConnectionError`: Network or authentication issues
- `S3PermissionError`: Insufficient IAM permissions
- `S3ArtifactError`: General S3 operation failures

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=s3_artifact_demo --cov-report=html
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ADK Agent     │────│  S3ArtifactSvc   │────│   Amazon S3     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         │                        │                       │
    ┌────▼────┐              ┌────▼────┐            ┌────▼────┐
    │  Tools  │              │  boto3  │            │ Bucket  │
    └─────────┘              └─────────┘            └─────────┘
```

The S3ArtifactService acts as a bridge between Google ADK's artifact system and Amazon S3, providing:

1. **Async Operations**: All S3 operations run asynchronously
2. **Thread Pool**: Concurrent S3 requests using ThreadPoolExecutor
3. **Structured Keys**: Organized storage with app/user/session hierarchy
4. **Type Safety**: Proper type hints and error handling

## Common Use Cases

1. **Document Storage**: Save user-generated content, reports, or analysis results
2. **Configuration Management**: Store user preferences and settings
3. **Data Pipeline**: Intermediate results in multi-step processing
4. **Collaboration**: Share artifacts between sessions or users
5. **Backup and Recovery**: Persistent storage with versioning

## Troubleshooting

### Common Issues

1. **Bucket Access Denied**
   - Check IAM permissions
   - Verify bucket exists and region matches

2. **Connection Timeouts**
   - Check network connectivity
   - Verify AWS credentials

3. **Version Not Found**
   - List available versions first
   - Check if artifact exists in the session/user scope

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0.
