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
- **Comprehensive Error Handling**: Robust exception hierarchy with automatic boto3 error mapping
- **Edge Case Management**: Large file support, concurrency control, and network failure recovery
- **Input Validation & Security**: Sanitization, path traversal protection, and content validation
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
    enable_validation=True,  # Enable input validation and sanitization
    enable_security_checks=True,  # Enable security validation
    enable_integrity_checks=True,  # Enable content integrity verification
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
from aws_adk import S3ArtifactService, S3ValidationError, RetryConfig

# Production configuration with comprehensive error handling
service = S3ArtifactService(
    bucket_name="production-artifacts",
    enable_encryption=True,
    enable_validation=True,  # Input validation and sanitization
    enable_security_checks=True,  # Security validation
    enable_integrity_checks=True,  # Content integrity verification
    retry_config=RetryConfig(
        max_attempts=5,
        base_delay=1.0,
        max_delay=60.0,
        backoff_strategy="exponential"
    )
)

# Service health monitoring
health = await service.get_service_health()
print(f"Validation enabled: {health['feature_flags']['validation']}")
print(f"Circuit breaker status: {health['circuit_breaker_stats']['read']['state']}")

# Performance monitoring
stats = service.get_connection_stats()
print(f"Active connections: {stats['active_connections']}")

# Comprehensive error handling
try:
    version = await service.save_artifact(
        app_name="app", user_id="user", session_id="session",
        filename="document.pdf", artifact=artifact
    )
except S3ValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Error context: {e.context}")

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

#### Phase 3 Error Handling & Edge Cases Demo
Available at [`examples/phase3_error_handling_demo.py`](examples/phase3_error_handling_demo.py):

- **Robust Exception Framework**: Automatic boto3 error mapping and rich context
- **Input Validation**: Sanitization, security checks, and content validation
- **Edge Case Handling**: Large files, concurrency control, network failures
- **Service Health Monitoring**: Circuit breaker states and performance metrics

Features demonstrated:
- Basic save/load/delete operations with enhanced reliability
- Automatic versioning and version management
- User-scoped vs session-scoped artifacts
- Client-side encryption for sensitive data
- Batch operations for improved efficiency
- Security monitoring and recommendations
- Performance optimization features
- Comprehensive error handling and recovery
- Input validation and sanitization
- Large file handling with multipart uploads
- Concurrency control and conflict resolution
- Network failure detection and recovery
- Circuit breaker patterns for fault tolerance
- Content integrity verification
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

✅ **Phase 3 - Error Handling and Edge Cases** (Complete)
- [x] Comprehensive exception hierarchy with automatic boto3 error mapping
- [x] Input validation and sanitization framework with security checks
- [x] Edge case handlers for large files, concurrency, and network failures
- [x] Enhanced S3ArtifactService integration with all error handling
- [x] Circuit breaker patterns and retry logic with exponential backoff
- [x] Content integrity verification and corruption detection
- [x] Service health monitoring and resource cleanup
- [x] Production-ready fault tolerance and graceful degradation

🔮 **Future Phases - Additional AWS Services**
- [ ] DynamoDB integrations
- [ ] Lambda function utilities
- [ ] CloudWatch logging integration

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
