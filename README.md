# Google ADK AWS Integrations

[![PyPI version](https://badge.fury.io/py/google-adk-aws.svg)](https://badge.fury.io/py/google-adk-aws)
[![Python versions](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/google-adk-aws/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

AWS service integrations for Google Agent Development Kit (ADK).

## Requirements

- **Python 3.10 or higher**
- AWS account with S3 access
- Google ADK 1.2.0 or higher

## Features

### **S3 Artifact Service**
- **Full BaseArtifactService Implementation**: Complete compatibility with ADK's artifact interface
- **AWS S3 Integration**: Production-ready S3 storage with proper authentication
- **Flexible Authentication**: Support for IAM roles, access keys, and custom endpoints
- **Version Management**: Automatic artifact versioning with history tracking
- **User Namespace Support**: Session-scoped and user-scoped artifact storage

### **Future AWS Integrations** (Planned)
- DynamoDB for session state storage
- Lambda function execution
- CloudWatch for logging and monitoring
- SQS/SNS for messaging

## Quick Start

### Installation

```bash
pip install google-adk-aws
```

### Basic Usage

```python
from aws_adk import S3ArtifactService
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

## Examples

### S3 Artifact Service Demo

A complete example demonstrating S3 artifact operations with Google ADK is available in [`examples/s3-artifact-demo/`](examples/s3-artifact-demo/):

- **Standalone Demo**: [`demo.py`](examples/s3-artifact-demo/s3_artifact_demo/demo.py) - Shows basic S3 operations
- **Agent Integration**: [`agent.py`](examples/s3-artifact-demo/s3_artifact_demo/agent.py) - ADK agent with S3 tools
- **Interactive Mode**: Run `python demo.py --interactive` for hands-on testing

Features demonstrated:
- Basic save/load/delete operations
- Automatic versioning and version management
- User-scoped vs session-scoped artifacts
- Error handling and best practices
- Agent tool integration patterns

## Development

This project includes a comprehensive Makefile with common development targets:

```bash
# Install in development mode
make install-dev

# Run tests
make test-unit
make test-integration
make test-coverage

# Code quality
make format          # Format code with black and isort
make lint           # Check code style with ruff
make typecheck      # Run mypy type checking
make quality        # Run all quality checks

# Development workflow
make dev-test       # Format, lint, typecheck, and test
make ci            # Run full CI pipeline locally

# Cleanup
make clean         # Remove all build/test artifacts

# See all available targets
make help
```

## Development Status

✅ **Phase 1 - S3 Artifact Service** (Complete)
- [x] Repository setup and packaging
- [x] Core S3ArtifactService implementation
- [x] Full BaseArtifactService interface compliance
- [x] AWS S3 authentication and storage
- [x] Basic verification tests

🚧 **Phase 2 - Enhanced S3 Features** (Next)
- [ ] Comprehensive error handling
- [ ] Advanced testing with moto
- [ ] Performance optimizations
- [ ] Integration tests

🔮 **Future Phases - Additional AWS Services**
- [ ] DynamoDB integrations
- [ ] Lambda function utilities
- [ ] CloudWatch logging integration

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
