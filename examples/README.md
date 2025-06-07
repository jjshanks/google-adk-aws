# AWS ADK Examples

This directory contains demonstration scripts showing how to use the AWS ADK S3 Artifact Service.

## Setup

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Configure your environment variables in `.env`:**
   ```bash
   # Required
   S3_BUCKET_NAME=your-demo-bucket-name
   AWS_REGION=us-east-1

   # Optional (if not using AWS profiles or IAM roles)
   AWS_ACCESS_KEY_ID=your-access-key
   AWS_SECRET_ACCESS_KEY=your-secret-key
   AWS_SESSION_TOKEN=your-session-token

   # Optional features
   S3_ENABLE_ENCRYPTION=true
   S3_ENABLE_VALIDATION=true
   ```

3. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

## Available Demos

### Basic Operations
**File:** `basic_demo.py`

Demonstrates fundamental S3 artifact operations:
- Basic CRUD operations (save, load, list, delete)
- File versioning and content management
- Session vs user-scoped files
- Error handling and cleanup

**Usage:**
```bash
python examples/basic_demo.py
python examples/basic_demo.py --interactive  # Interactive mode
```

### Phase 2 Enhanced Features
**File:** `phase2_features_demo.py`

Showcases advanced features introduced in Phase 2:
- Enhanced error handling with retry logic
- Performance optimization (connection pooling, batch operations)
- Security features (encryption, integrity verification, presigned URLs)
- Monitoring and diagnostics capabilities

**Usage:**
```bash
python examples/phase2_features_demo.py
```

### Phase 3 Error Handling & Edge Cases
**File:** `phase3_error_handling_demo.py`

Demonstrates comprehensive error handling and edge case management:
- Automatic boto3 error mapping
- Input validation and sanitization
- Edge case management (large files, concurrency, network failures)
- Service health monitoring and circuit breaker patterns

**Usage:**
```bash
python examples/phase3_error_handling_demo.py
```

### Agent Integration
**File:** `agent_integration_demo.py`

Shows how to integrate S3 artifact storage with Google ADK agents:
- Creating agents with S3 artifact capabilities
- File management tools for AI agents
- Persistent storage across agent conversations
- Tool integration patterns

**Usage:**
```bash
python examples/agent_integration_demo.py
```

## Common Configuration

All demos use a unified configuration system:

- **Shared environment variables** defined in `.env`
- **Common configuration module** (`common/config.py`) for consistent setup
- **Standardized variable names** across all examples
- **Automatic dotenv loading** with clear error messages

## File Structure

```
examples/
├── .env.example              # Environment template
├── README.md                 # This file
├── common/                   # Shared utilities
│   ├── __init__.py
│   └── config.py            # Configuration management
├── basic_demo.py            # Basic operations demo
├── phase2_features_demo.py  # Phase 2 enhanced features
├── phase3_error_handling_demo.py  # Phase 3 error handling
└── agent_integration_demo.py # Agent integration example
```

## Environment Variables

### Required
- `S3_BUCKET_NAME`: S3 bucket for artifact storage

### Optional
- `AWS_REGION`: AWS region (default: us-east-1)
- `AWS_PROFILE`: AWS profile to use
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_SESSION_TOKEN`: AWS session token
- `S3_ENDPOINT_URL`: Custom S3 endpoint (for LocalStack, etc.)

### Feature Flags
- `S3_ENABLE_ENCRYPTION`: Enable S3 server-side encryption
- `S3_ENABLE_VALIDATION`: Enable content validation
- `S3_ENABLE_LOGGING`: Enable request logging

## Troubleshooting

### Common Issues

1. **Missing S3_BUCKET_NAME:**
   ```
   Error: Required environment variable S3_BUCKET_NAME not found
   ```
   Solution: Copy `.env.example` to `.env` and set your bucket name.

2. **AWS Credentials:**
   - Ensure your AWS credentials are configured via environment variables, AWS profiles, or IAM roles
   - Test with: `aws s3 ls` to verify credentials work

3. **Bucket Permissions:**
   - Ensure your credentials have read/write access to the specified bucket
   - Required permissions: `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`, `s3:DeleteObject`

4. **Import Errors:**
   - Install the package: `pip install -e .`
   - Ensure you're running from the project root directory

## Tips

- Start with `basic_demo.py` to verify your setup
- Use the `--interactive` flag with basic demo for hands-on testing
- Check the configuration summary printed at startup for debugging
- User-scoped files (prefix `user:`) persist across sessions
- All demos include cleanup to avoid leaving test artifacts
