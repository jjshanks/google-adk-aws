# BaseArtifactService Developer Guide

This guide provides comprehensive information for developers working with and creating new `BaseArtifactService` implementations in the Agent Development Kit (ADK). The ADK artifact system enables persistent storage and management of binary data with built-in versioning and namespacing capabilities.

## Overview

Artifacts in ADK represent binary data (like file content) identified by unique filenames within specific scopes. The `BaseArtifactService` provides an abstract interface for managing these artifacts with built-in versioning and namespacing capabilities.

## Architecture

### Core Concepts

- **Artifacts**: Binary data stored as `google.genai.types.Part` objects with associated MIME types
- **Versioning**: Each artifact save creates a new version, starting from 0
- **Namespacing**: Artifacts can be scoped to sessions or users
- **Hierarchical Storage**: Uses `app_name/user_id/session_id/filename` structure

### BaseArtifactService Interface

The abstract base class defines five core methods that must be implemented by any artifact service:

```python
class BaseArtifactService(ABC):
    async def save_artifact(self, *, app_name: str, user_id: str,
                           session_id: str, filename: str,
                           artifact: types.Part) -> int

    async def load_artifact(self, *, app_name: str, user_id: str,
                           session_id: str, filename: str,
                           version: Optional[int] = None) -> Optional[types.Part]

    async def list_artifact_keys(self, *, app_name: str, user_id: str,
                                session_id: str) -> list[str]

    async def delete_artifact(self, *, app_name: str, user_id: str,
                             session_id: str, filename: str) -> None

    async def list_versions(self, *, app_name: str, user_id: str,
                           session_id: str, filename: str) -> list[int]
```

## Built-in Implementations

### InMemoryArtifactService

**Characteristics**:
- In-memory storage using dictionary structure
- Inherits from both `BaseArtifactService` and `BaseModel` (Pydantic)
- Ephemeral storage (data lost on restart)
- Fast operations with no external dependencies
- Ideal for development and testing

**Key Implementation Details**:
- Stores artifacts in `dict[str, list[types.Part]]` structure
- Uses list index as version number
- Path construction: `_artifact_path()` method handles namespacing logic

### GcsArtifactService

**Characteristics**:
- Google Cloud Storage backend for persistent storage
- Production-ready with scalable storage
- Requires GCS bucket and appropriate permissions
- Blob naming pattern: `app_name/user_id/session_id/filename/version`

**Key Implementation Details**:
- Uses Google Cloud Storage client: `storage.Client(**kwargs)`
- Blob operations: `upload_from_string()`, `download_as_bytes()`
- Version discovery through blob prefix listing
- Content type preservation for artifacts

## Configuration Patterns

### Basic Setup

```python
from google.adk.artifacts import InMemoryArtifactService, GcsArtifactService
from google.adk.runners import Runner

# Development setup
artifact_service = InMemoryArtifactService()

# Production setup with GCS
artifact_service = GcsArtifactService(
    bucket_name="my-artifacts-bucket",
    project="my-gcp-project"
)

runner = Runner(
    app_name="my-app",
    agent=my_agent,
    artifact_service=artifact_service,
    session_service=session_service
)
```

### CLI Configuration

The ADK CLI automatically selects artifact service based on configuration:

```python
# Using environment variable or CLI argument
if artifact_storage_uri.startswith("gs://"):
    gcs_bucket = artifact_storage_uri.split("://")[1]
    artifact_service = GcsArtifactService(bucket_name=gcs_bucket)
else:
    artifact_service = InMemoryArtifactService()
```

## Integration with ADK Framework

### Context Integration

Artifact services integrate with ADK through several context layers:

**InvocationContext**: Contains the artifact service instance
```python
artifact_service: Optional[BaseArtifactService] = None
```

**CallbackContext**: Provides simplified artifact operations for agents
```python
async def load_artifact(self, filename: str, version: Optional[int] = None) -> Optional[types.Part]:
    return await self._invocation_context.artifact_service.load_artifact(...)

async def save_artifact(self, filename: str, artifact: types.Part) -> int:
    return await self._invocation_context.artifact_service.save_artifact(...)
```

**ToolContext**: Provides artifact operations for tools
```python
async def list_artifacts(self) -> list[str]:
    if self._invocation_context.artifact_service is None:
        raise ValueError('Artifact service is not initialized.')
    return await self._invocation_context.artifact_service.list_artifact_keys(...)
```

### Tool Integration

**LoadArtifactsTool**: Provides automatic artifact loading for LLM requests:
- Appends artifact instructions to prompts
- Loads artifact content when requested by model
- Available as `load_artifacts` in the tools module

### Code Execution Integration

The ADK automatically saves code execution output files as artifacts:

```python
for output_file in code_execution_result.output_files:
    version = await invocation_context.artifact_service.save_artifact(
        app_name=invocation_context.app_name,
        user_id=invocation_context.user_id,
        session_id=invocation_context.session.id,
        filename=output_file.name,
        artifact=types.Part.from_bytes(
            data=base64.b64decode(output_file.content),
            mime_type=output_file.mime_type,
        ),
    )
```

## Namespacing

### Session-Scoped Artifacts (Default)

```python
# Stored as: app_name/user_id/session_id/filename
await tool_context.save_artifact("report.pdf", pdf_artifact)
```

### User-Scoped Artifacts

```python
# Stored as: app_name/user_id/user/filename
# Accessible across all user sessions
await tool_context.save_artifact("user:profile.png", image_artifact)
```

Both implementations check for "user:" prefix using `_file_has_user_namespace()` method.

## Usage Examples

### Basic Artifact Operations

```python
from google.genai import types

# Save text artifact
async def save_text_artifact(tool_context: ToolContext, content: str):
    artifact = types.Part(
        inline_data=types.Blob(
            mime_type='text/plain',
            data=content.encode('utf-8')
        )
    )
    version = await tool_context.save_artifact('document.txt', artifact)
    return version

# Load artifact
async def load_artifact_content(tool_context: ToolContext):
    artifact = await tool_context.load_artifact('document.txt')
    if artifact:
        return artifact.inline_data.data.decode('utf-8')
    return None
```

### Image Generation Example

```python
await tool_context.save_artifact(
    'image.png',
    types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
)
```

## Creating Custom Implementations

### Implementation Requirements

When creating a new `BaseArtifactService` implementation:

1. **Inherit from BaseArtifactService**: Import and extend the abstract base class
2. **Implement all abstract methods**: All five methods must be implemented
3. **Handle versioning**: Maintain version numbering starting from 0
4. **Support namespacing**: Handle "user:" prefix for cross-session artifacts
5. **Error handling**: Handle missing artifacts gracefully (return None/empty lists)
6. **Async operations**: All methods are async for consistent interface

### Example Custom Implementation

```python
from google.adk.artifacts import BaseArtifactService
from google.genai import types
from typing import Optional
from typing_extensions import override

class CustomArtifactService(BaseArtifactService):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # Initialize your storage backend

    def _file_has_user_namespace(self, filename: str) -> bool:
        return filename.startswith("user:")

    def _build_path(self, app_name: str, user_id: str,
                   session_id: str, filename: str) -> str:
        if self._file_has_user_namespace(filename):
            return f"{app_name}/{user_id}/user/{filename}"
        return f"{app_name}/{user_id}/{session_id}/{filename}"

    @override
    async def save_artifact(self, *, app_name: str, user_id: str,
                           session_id: str, filename: str,
                           artifact: types.Part) -> int:
        # Get existing versions
        versions = await self.list_versions(
            app_name=app_name, user_id=user_id,
            session_id=session_id, filename=filename
        )
        version = 0 if not versions else max(versions) + 1

        # Store artifact with version
        path = self._build_path(app_name, user_id, session_id, filename)
        # Your storage logic here

        return version

    @override
    async def load_artifact(self, *, app_name: str, user_id: str,
                           session_id: str, filename: str,
                           version: Optional[int] = None) -> Optional[types.Part]:
        if version is None:
            versions = await self.list_versions(
                app_name=app_name, user_id=user_id,
                session_id=session_id, filename=filename
            )
            if not versions:
                return None
            version = max(versions)

        # Your retrieval logic here
        # Return types.Part or None if not found

    @override
    async def list_artifact_keys(self, *, app_name: str, user_id: str,
                                session_id: str) -> list[str]:
        # List both session and user-scoped artifacts
        # Return sorted list of filenames
        pass

    @override
    async def delete_artifact(self, *, app_name: str, user_id: str,
                             session_id: str, filename: str) -> None:
        # Delete all versions of the artifact
        pass

    @override
    async def list_versions(self, *, app_name: str, user_id: str,
                           session_id: str, filename: str) -> list[int]:
        # Return list of available version numbers
        pass
```

## Best Practices

### Development
- Use `InMemoryArtifactService` for development and testing
- Implement proper error handling for missing artifacts
- Test both session and user-scoped artifact scenarios
- Validate MIME types when creating artifacts

### Production
- Use `GcsArtifactService` or custom persistent storage
- Implement proper authentication and authorization
- Monitor storage costs and usage patterns
- Consider artifact cleanup policies for old versions
- Implement proper logging for debugging

### Performance
- Cache frequently accessed artifacts when appropriate
- Implement efficient listing operations for large artifact stores
- Consider pagination for list operations in custom implementations
- Monitor storage backend performance metrics

## API Endpoints

When using ADK's FastAPI deployment, artifacts are exposed through REST endpoints:

- `GET /apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}` - Download specific artifact
- `GET /apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts` - List all artifacts in session
- `DELETE /apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}` - Delete artifact

## Testing

### Unit Testing Patterns

For testing artifact services, follow these patterns:

- Test all CRUD operations (create, read, update, delete)
- Test versioning behavior (incremental version numbers)
- Test namespacing (session vs user scope)
- Test error conditions (missing artifacts, invalid parameters)
- Test concurrent access scenarios

### Integration Testing

- Test artifact service integration with agents and tools
- Test automatic code execution artifact saving
- Test artifact loading in LLM flows
- Verify proper cleanup and resource management

## Common Patterns

### Input Blob Handling

Automatic saving of input blobs when `save_input_blobs_as_artifacts=True`:

```python
if save_input_blobs_as_artifacts:
    for part in parts:
        if part.inline_data:
            filename = f"input_blob_{blob_counter}"
            await artifact_service.save_artifact(...)
            # Replace blob with filename placeholder
```

### Event Actions Integration

Artifact operations can trigger event actions for real-time updates:

```python
event_actions.artifact_delta[filename] = version
```

This enables UI components to react to artifact changes in real-time.

## Troubleshooting

### Common Issues

1. **Missing artifact service**: Ensure artifact service is properly configured in Runner
2. **Version conflicts**: Check version number handling in custom implementations
3. **Namespace confusion**: Verify "user:" prefix handling for cross-session artifacts
4. **MIME type mismatches**: Ensure proper content type handling
5. **Storage permissions**: Verify backend storage permissions (GCS bucket access)

### Debugging Tips

- Enable logging: `logging.getLogger("google_adk.artifacts")`
- Check artifact paths and version numbers
- Verify storage backend connectivity
- Test with simple artifacts before complex binary data
- Use InMemoryArtifactService for isolated testing

## Migration Guide

When switching between artifact service implementations:

1. **Development to Production**: Replace `InMemoryArtifactService` with `GcsArtifactService`
2. **Data Migration**: Implement custom migration scripts for existing artifacts
3. **Configuration Updates**: Update deployment configurations
4. **Testing**: Verify all artifact operations work with new implementation
5. **Monitoring**: Set up monitoring for the new storage backend

## Additional Resources

- [ADK Documentation](https://google.github.io/adk-docs/artifacts/) - Official artifact documentation
- [ADK Repository](https://github.com/google/adk-python) - Source code and examples
- ADK samples directory - Various artifact usage patterns

This guide provides the foundation for working with ADK's artifact system. The artifact service architecture enables flexible, scalable storage solutions that can be adapted to various deployment scenarios and storage backends.
