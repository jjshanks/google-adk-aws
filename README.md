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

### **S3 Artifact Service** (Production Ready)
- **Full BaseArtifactService Implementation**: Complete compatibility with ADK's artifact interface
- **Enterprise-Grade Reliability**: Advanced error handling, retry logic, and circuit breakers
- **Performance Optimized**: Connection pooling, batch operations, and multipart uploads
- **Security Hardened**: Client-side encryption, integrity verification, and presigned URLs
- **AWS S3 Integration**: Production-ready S3 storage with comprehensive authentication
- **Flexible Authentication**: Support for IAM roles, access keys, and custom endpoints
- **Version Management**: Automatic artifact versioning with history tracking
- **User Namespace Support**: Session-scoped and user-scoped artifact storage
- **Monitoring & Diagnostics**: Security assessments, performance metrics, and connection stats

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
from aws_adk import S3ArtifactService, RetryConfig
from google.adk.agents import Agent

# Create S3 artifact service with enhanced features
artifact_service = S3ArtifactService(
    bucket_name="my-artifacts-bucket",
    region_name="us-west-2",
    enable_encryption=True,  # Enable client-side encryption
    retry_config=RetryConfig(max_attempts=5, base_delay=1.0)
)

# Use with ADK agent
agent = Agent(
    name="document_processor",
    model="gemini-2.0-flash",
    artifact_service=artifact_service
)
```

### Advanced Features

```python
from aws_adk import S3ArtifactService, S3ArtifactError

# Production configuration
service = S3ArtifactService(
    bucket_name="production-artifacts",
    enable_encryption=True,
    encryption_key="your-encryption-key"  # Optional custom key
)

# Security monitoring
security_status = await service.get_security_status()
print(f"Encryption enabled: {security_status['encryption']}")

# Performance monitoring
stats = service.get_connection_stats()
print(f"Active connections: {stats['active_connections']}")

# Batch operations for efficiency
await service.batch_delete_artifacts(
    app_name="app", user_id="user", session_id="session",
    filenames=["file1.txt", "file2.txt", "file3.txt"]
)

# Secure temporary access
presigned_url = await service.generate_presigned_url(
    app_name="app", user_id="user", session_id="session",
    filename="document.pdf", expiration=3600
)
```

## Examples

### S3 Artifact Service Demos

Multiple comprehensive examples demonstrate the full capabilities:

#### Basic S3 Operations Demo
Available in [`examples/s3-artifact-demo/`](examples/s3-artifact-demo/):

- **Standalone Demo**: [`demo.py`](examples/s3-artifact-demo/s3_artifact_demo/demo.py) - Shows basic S3 operations
- **Agent Integration**: [`agent.py`](examples/s3-artifact-demo/s3_artifact_demo/agent.py) - ADK agent with S3 tools
- **Interactive Mode**: Run `python demo.py --interactive` for hands-on testing

#### Phase 2 Enhanced Features Demo
Available at [`examples/phase2_features_demo.py`](examples/phase2_features_demo.py):

- **Security Features**: Encryption, integrity verification, presigned URLs
- **Performance Features**: Connection pooling, batch operations, large file handling
- **Monitoring**: Security assessments and performance metrics
- **Error Handling**: Retry logic and circuit breaker demonstrations

Features demonstrated:
- Basic save/load/delete operations with enhanced reliability
- Automatic versioning and version management
- User-scoped vs session-scoped artifacts
- Client-side encryption for sensitive data
- Batch operations for improved efficiency
- Security monitoring and recommendations
- Performance optimization features
- Comprehensive error handling and recovery
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

✅ **Phase 2 - Enhanced S3 Features** (Complete)
- [x] Advanced error handling with retry logic and circuit breakers
- [x] Comprehensive testing framework with moto S3 mocking
- [x] Performance optimizations (connection pooling, batch operations)
- [x] Security features (encryption, integrity verification, presigned URLs)
- [x] Production monitoring and diagnostics
- [x] Integration tests with real S3 operations
- [x] Large file support with multipart uploads

🔮 **Future Phases - Additional AWS Services**
- [ ] DynamoDB integrations
- [ ] Lambda function utilities
- [ ] CloudWatch logging integration

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
