# ADK Agent Development Guide

This guide provides comprehensive instructions for creating example agents using the Agent Development Kit (ADK) based on the patterns and conventions established in this repository's Python examples.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup and Configuration](#setup-and-configuration)
3. [Agent Implementation](#agent-implementation)
4. [Tool Development](#tool-development)
5. [Sub-Agent Architecture](#sub-agent-architecture)
6. [Testing](#testing)
7. [Evaluation](#evaluation)
8. [Deployment](#deployment)
9. [Best Practices](#best-practices)
10. [Common Patterns](#common-patterns)

## Project Structure

### Standard Directory Layout

Every ADK agent example should follow this directory structure:

```
my-agent-example/
Γö£ΓöÇΓöÇ README.md                    # Comprehensive documentation
Γö£ΓöÇΓöÇ pyproject.toml              # Python dependencies and metadata
Γö£ΓöÇΓöÇ my_agent_example/           # Main package (snake_case)
Γöé   Γö£ΓöÇΓöÇ __init__.py            # Package initialization
Γöé   Γö£ΓöÇΓöÇ agent.py               # Agent definition and configuration
Γöé   Γö£ΓöÇΓöÇ prompt.py              # Agent instructions and prompts
Γöé   Γö£ΓöÇΓöÇ config.py              # Configuration settings (optional)
Γöé   Γö£ΓöÇΓöÇ tools/                 # Custom tools directory
Γöé   Γöé   Γö£ΓöÇΓöÇ __init__.py
Γöé   Γöé   ΓööΓöÇΓöÇ tools.py
Γöé   Γö£ΓöÇΓöÇ sub_agents/            # Sub-agent definitions (if needed)
Γöé   Γöé   Γö£ΓöÇΓöÇ __init__.py
Γöé   Γöé   ΓööΓöÇΓöÇ specialized_agent/
Γöé   Γöé       Γö£ΓöÇΓöÇ __init__.py
Γöé   Γöé       Γö£ΓöÇΓöÇ agent.py
Γöé   Γöé       ΓööΓöÇΓöÇ prompt.py
Γöé   Γö£ΓöÇΓöÇ shared_libraries/      # Common utilities
Γöé   Γöé   Γö£ΓöÇΓöÇ __init__.py
Γöé   Γöé   Γö£ΓöÇΓöÇ callbacks.py
Γöé   Γöé   Γö£ΓöÇΓöÇ types.py
Γöé   Γöé   ΓööΓöÇΓöÇ constants.py
Γöé   ΓööΓöÇΓöÇ entities/              # Data models (optional)
Γöé       Γö£ΓöÇΓöÇ __init__.py
Γöé       ΓööΓöÇΓöÇ models.py
Γö£ΓöÇΓöÇ deployment/                # Deployment configurations
Γöé   Γö£ΓöÇΓöÇ deploy.py
Γöé   ΓööΓöÇΓöÇ test_deployment.py
Γö£ΓöÇΓöÇ eval/                      # Evaluation framework
Γöé   Γö£ΓöÇΓöÇ __init__.py
Γöé   Γö£ΓöÇΓöÇ test_eval.py
Γöé   ΓööΓöÇΓöÇ data/
Γöé       Γö£ΓöÇΓöÇ example.test.json
Γöé       ΓööΓöÇΓöÇ test_config.json
Γö£ΓöÇΓöÇ tests/                     # Unit and integration tests
Γöé   Γö£ΓöÇΓöÇ __init__.py
Γöé   Γö£ΓöÇΓöÇ test_agents.py
Γöé   ΓööΓöÇΓöÇ unit/
Γöé       ΓööΓöÇΓöÇ test_tools.py
Γö£ΓöÇΓöÇ Dockerfile                 # Container configuration (optional)
ΓööΓöÇΓöÇ architecture.png           # Visual architecture diagram
```

### Naming Conventions

- **Directory names**: Use kebab-case (e.g., `my-agent-example`)
- **Python packages**: Use snake_case (e.g., `my_agent_example`)
- **Python files**: Use snake_case (e.g., `agent.py`, `tools.py`)
- **Class names**: Use PascalCase (e.g., `MyAgent`, `SpecializedTool`)
- **Function names**: Use snake_case (e.g., `process_data`, `handle_request`)

## Setup and Configuration

### pyproject.toml Configuration

Choose between two supported formats:

#### Option 1: PEP 621 Compliant (Recommended for new projects)

```toml
[project]
name = "my-agent-example"
version = "0.1.0"
description = "Brief description of your agent"
authors = [{ name = "Your Name", email = "your.email@example.com" }]
license = "Apache-2.0"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "google-adk>=1.0.0,<2.0.0",
    "google-cloud-aiplatform[agent-engines]>=1.93.0,<2.0.0",
    "pydantic>=2.0.0",
    # Add other dependencies as needed
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.25.0",
    "pytest-cov>=6.0.0",
    "pyink>=24.0.0",
    "pylint>=3.0.0",
]
eval = [
    "google-cloud-aiplatform[evaluation]>=1.93.0",
]

[tool.pytest.ini_options]
console_output_style = "progress"
addopts = "-vv -s"
testpaths = ["tests/", "eval/"]
markers = ["unit", "integration"]
log_level = "ERROR"
filterwarnings = ["ignore::UserWarning"]

[tool.pyink]
line-length = 80
pyink-indentation = 4
pyink-use-majority-quotes = true
```

#### Option 2: Poetry-based (Legacy support)

```toml
[tool.poetry]
name = "my-agent-example"
version = "0.1.0"
description = "Brief description of your agent"
authors = ["Your Name <your.email@example.com>"]
license = "Apache License 2.0"
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.11"
google-cloud-aiplatform = { extras = ["adk", "agent_engine"], version = "^1.93.0" }
google-adk = "^1.0.0"
pydantic = "^2.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.25.0"
pytest-cov = "^6.0.0"
pyink = "^24.0.0"
pylint = "^3.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### Required Dependencies

- **google-adk**: Core ADK functionality
- **google-cloud-aiplatform**: Vertex AI integration with ADK and agent engines
- **google-adk-aws**: AWS service integrations (S3 artifact storage, etc.)
- **pydantic**: Data validation and serialization

### Development Dependencies

- **pytest** & **pytest-asyncio**: Testing framework
- **pyink**: Google's Python formatter
- **pylint**: Code linting
- **pytest-cov**: Code coverage

## Agent Implementation

### Basic Agent Structure

Create your main agent in `my_agent_example/agent.py`:

```python
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

"""Main agent implementation for my-agent-example."""

import logging
from google.adk import Agent
from google.genai.types import GenerateContentConfig
from aws_adk import S3ArtifactService, RetryConfig

from .prompt import GLOBAL_INSTRUCTION, MAIN_INSTRUCTION
from .tools.tools import primary_tool, secondary_tool
from .shared_libraries.callbacks import before_tool_callback, after_tool_callback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure enhanced S3 artifact service
artifact_service = S3ArtifactService(
    bucket_name="my-agent-artifacts",
    region_name="us-west-2",
    enable_encryption=True,  # Enable client-side encryption
    retry_config=RetryConfig(max_attempts=5, base_delay=1.0)
)

# Agent configuration
root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="my_agent",
    global_instruction=GLOBAL_INSTRUCTION,
    instruction=MAIN_INSTRUCTION,
    tools=[primary_tool, secondary_tool],
    artifact_service=artifact_service,  # Enhanced S3 artifact storage
    generate_config=GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=8192,
        response_mime_type="application/json"  # If structured output needed
    ),
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)
```

### Prompt Definition

Create clear, specific prompts in `my_agent_example/prompt.py`:

```python
"""Agent prompts and instructions."""

GLOBAL_INSTRUCTION = """
You are a helpful AI assistant specialized in [your domain].
Always be precise, helpful, and follow the user's instructions carefully.
"""

MAIN_INSTRUCTION = """
Your primary role is to [specific role description].

Key capabilities:
- [Capability 1]
- [Capability 2]
- [Capability 3]

Guidelines:
1. Always validate inputs before processing
2. Provide clear, actionable responses
3. Use tools when appropriate to gather information
4. If uncertain, ask clarifying questions

Response format:
- For simple queries: Provide direct answers
- For complex tasks: Break down into steps
- For data requests: Use structured JSON format
"""
```

### Agent Types

#### Simple Agent Pattern
Use `google.adk.Agent` for straightforward single-purpose agents:

```python
from google.adk import Agent

simple_agent = Agent(
    name="simple_task_agent",
    model="gemini-2.0-flash-001",
    instruction="Handle specific simple tasks",
    tools=[tool1, tool2]
)
```

#### Hierarchical Agent Pattern
Use `google.adk.agents.LlmAgent` for coordination with sub-agents:

```python
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-pro-preview-05-06",
    description="Coordinates between specialized sub-agents",
    instruction=COORDINATOR_PROMPT,
    tools=[
        AgentTool(agent=specialist_agent_1),
        AgentTool(agent=specialist_agent_2),
    ],
    output_key="coordination_result"
)
```

## Tool Development

### Tool Implementation Pattern

Create tools in `my_agent_example/tools/tools.py`:

```python
"""Custom tools for the agent."""

import logging
from typing import Dict, Any, Optional
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

def primary_tool(
    input_param: str,
    optional_param: Optional[int] = None
) -> Dict[str, Any]:
    """
    Brief description of what this tool does.

    This tool performs [specific functionality] and returns [type of result].
    Use this tool when [specific use case].

    Args:
        input_param: Description of the required parameter
        optional_param: Description of optional parameter (default: None)

    Returns:
        dict: A dictionary containing:
            - status: "success" or "error"
            - result: The main result data
            - message: Human-readable description

    Example:
        >>> primary_tool("example input", 42)
        {
            "status": "success",
            "result": {"processed": "example input", "count": 42},
            "message": "Successfully processed input"
        }

    Raises:
        ValueError: If input_param is invalid
        RuntimeError: If processing fails
    """
    try:
        logger.info(f"Processing with primary_tool: {input_param}")

        # Validate inputs
        if not input_param or not isinstance(input_param, str):
            return {
                "status": "error",
                "result": None,
                "message": "Invalid input_param: must be a non-empty string"
            }

        # Process the input
        processed_result = _process_data(input_param, optional_param)

        return {
            "status": "success",
            "result": processed_result,
            "message": f"Successfully processed: {input_param}"
        }

    except Exception as e:
        logger.error(f"Error in primary_tool: {str(e)}")
        return {
            "status": "error",
            "result": None,
            "message": f"Processing failed: {str(e)}"
        }

def _process_data(data: str, param: Optional[int]) -> Dict[str, Any]:
    """Helper function for data processing."""
    # Implementation details
    return {"processed": data, "param": param}
```

### Tool Design Principles

1. **Comprehensive Docstrings**: Include description, parameters, return values, examples, and exceptions
2. **Consistent Return Format**: Use dictionaries with `status`, `result`, and `message` fields
3. **Error Handling**: Gracefully handle errors and return informative messages
4. **Input Validation**: Validate all inputs and provide clear error messages
5. **Logging**: Use structured logging for debugging and monitoring
6. **Type Hints**: Use proper type annotations for better code clarity

### Tool Context Usage

For tools that need access to agent context:

```python
from google.adk.tools import ToolContext

def context_aware_tool(param: str, context: ToolContext) -> Dict[str, Any]:
    """Tool that uses agent context information."""

    # Access session information
    session_id = context.session_id
    user_id = context.user_id

    # Access conversation history if needed
    # messages = context.get_conversation_history()

    return {
        "status": "success",
        "result": f"Processed {param} for user {user_id}",
        "message": "Context-aware processing completed"
    }
```

## Sub-Agent Architecture

### When to Use Sub-Agents

Use sub-agents for:
- **Specialized domains**: Different expertise areas (e.g., analysis, execution, research)
- **Complex workflows**: Multi-step processes requiring different approaches
- **Scalability**: Distributing workload across specialized components
- **Modularity**: Clean separation of concerns

### Sub-Agent Implementation

Create sub-agents in `my_agent_example/sub_agents/specialist/agent.py`:

```python
"""Specialized sub-agent implementation."""

from google.adk import Agent
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel

from .prompt import SPECIALIST_INSTRUCTION
from ..shared_libraries.types import SpecialistOutput

class SpecialistOutputModel(BaseModel):
    """Output schema for specialist agent."""
    analysis: str
    confidence: float
    recommendations: list[str]

specialist_agent = Agent(
    name="specialist",
    model="gemini-2.0-flash-001",
    description="Handles specialized analysis tasks",
    instruction=SPECIALIST_INSTRUCTION,
    tools=[specialized_tool_1, specialized_tool_2],
    output_schema=SpecialistOutputModel,
    output_key="specialist_result",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_config=GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json"
    )
)
```

### Sub-Agent Communication

Define clear output schemas for inter-agent communication:

```python
# In shared_libraries/types.py
from pydantic import BaseModel
from typing import List, Optional

class AgentOutput(BaseModel):
    """Base output schema for agent communication."""
    status: str
    data: dict
    confidence: Optional[float] = None
    next_steps: Optional[List[str]] = None

class SpecialistOutput(AgentOutput):
    """Specialized output for domain-specific agents."""
    analysis_type: str
    findings: List[str]
    recommendations: List[str]
```

## Testing

### Unit Tests

Create unit tests in `tests/unit/test_tools.py`:

```python
"""Unit tests for agent tools."""

import pytest
from unittest.mock import patch, MagicMock

from my_agent_example.tools.tools import primary_tool, secondary_tool

class TestPrimaryTool:
    """Test cases for primary_tool function."""

    def test_primary_tool_success(self):
        """Test successful execution of primary_tool."""
        result = primary_tool("test input", 42)

        assert result["status"] == "success"
        assert "result" in result
        assert result["message"] is not None

    def test_primary_tool_invalid_input(self):
        """Test primary_tool with invalid input."""
        result = primary_tool("", None)

        assert result["status"] == "error"
        assert "Invalid input_param" in result["message"]

    @patch('my_agent_example.tools.tools._process_data')
    def test_primary_tool_processing_error(self, mock_process):
        """Test primary_tool when processing fails."""
        mock_process.side_effect = RuntimeError("Processing failed")

        result = primary_tool("test input")

        assert result["status"] == "error"
        assert "Processing failed" in result["message"]

@pytest.mark.unit
class TestSecondaryTool:
    """Test cases for secondary_tool function."""

    def test_secondary_tool_basic(self):
        """Test basic functionality of secondary_tool."""
        # Implementation specific to your tool
        pass
```

### Integration Tests

Create integration tests in `tests/test_agents.py`:

```python
"""Integration tests for agent functionality."""

import pytest
from google.adk.runners import InMemoryRunner

from my_agent_example.agent import root_agent

@pytest.mark.asyncio
class TestAgentIntegration:
    """Integration tests for the main agent."""

    async def test_basic_agent_interaction(self):
        """Test basic agent conversation flow."""
        runner = InMemoryRunner(agent=root_agent)
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="test_user"
        )

        # Test basic interaction
        response = await session.send(
            query="Test query for the agent",
            session_id=session.session_id
        )

        assert response is not None
        assert len(response.candidates) > 0

        # Clean up
        await runner.session_service.delete_session(
            app_name=runner.app_name,
            session_id=session.session_id
        )

    async def test_tool_usage(self):
        """Test that agent correctly uses tools."""
        runner = InMemoryRunner(agent=root_agent)
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="test_user"
        )

        # Test query that should trigger tool usage
        response = await session.send(
            query="Please use the primary tool to process 'test data'",
            session_id=session.session_id
        )

        # Verify tool was called
        # Implementation depends on your specific tools and agent behavior
        assert response is not None

        # Clean up
        await runner.session_service.delete_session(
            app_name=runner.app_name,
            session_id=session.session_id
        )
```

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest -m unit

# Run with coverage
pytest --cov=my_agent_example

# Run specific test file
pytest tests/unit/test_tools.py

# Run with verbose output
pytest -vv
```

## Evaluation

### Evaluation Data Format

Create evaluation datasets in `eval/data/example.test.json`:

```json
[
  {
    "query": "Please analyze the provided data and give recommendations",
    "expected_tool_use": [
      {
        "tool_name": "primary_tool",
        "tool_input": {
          "input_param": "data analysis request"
        }
      }
    ],
    "reference": "The agent should use the primary tool to analyze data and provide structured recommendations based on the analysis results.",
    "metadata": {
      "category": "analysis",
      "difficulty": "medium",
      "expected_response_type": "structured"
    }
  },
  {
    "query": "What can you help me with?",
    "expected_tool_use": [],
    "reference": "The agent should describe its capabilities without using any tools, mentioning its ability to analyze data, provide recommendations, and use specialized tools.",
    "metadata": {
      "category": "capability_inquiry",
      "difficulty": "easy",
      "expected_response_type": "informational"
    }
  }
]
```

### Evaluation Configuration

Create evaluation criteria in `eval/data/test_config.json`:

```json
{
  "criteria": {
    "tool_trajectory_avg_score": 0.3,
    "response_match_score": 0.4,
    "response_quality_score": 0.3
  },
  "thresholds": {
    "tool_trajectory_avg_score": 0.7,
    "response_match_score": 0.6,
    "response_quality_score": 0.7
  }
}
```

### Evaluation Tests

Create evaluation runner in `eval/test_eval.py`:

```python
"""Evaluation tests for the agent."""

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

@pytest.mark.asyncio
class TestAgentEvaluation:
    """Evaluation test cases."""

    async def test_eval_basic_functionality(self):
        """Test agent evaluation on basic functionality."""
        await AgentEvaluator.evaluate(
            "my_agent_example",
            "eval/data/example.test.json",
            num_runs=1,
        )

    async def test_eval_with_config(self):
        """Test agent evaluation with custom configuration."""
        await AgentEvaluator.evaluate(
            "my_agent_example",
            "eval/data/example.test.json",
            config_file="eval/data/test_config.json",
            num_runs=3,
        )
```

### Running Evaluations

```bash
# Run evaluation tests
pytest eval/

# Run with specific number of iterations
pytest eval/test_eval.py::TestAgentEvaluation::test_eval_basic_functionality

# Run evaluation with detailed output
pytest eval/ -vv -s
```

## Deployment

### Deployment Script

Create deployment script in `deployment/deploy.py`:

```python
#!/usr/bin/env python3
"""Deployment script for my-agent-example."""

import argparse
import logging
import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

from my_agent_example.agent import root_agent

# Configuration
PROJECT_ID = "your-project-id"  # Replace with your project
LOCATION = "us-central1"
STAGING_BUCKET = f"gs://{PROJECT_ID}-adk-staging"
AGENT_WHL_FILE = "./dist/my_agent_example-0.1.0-py3-none-any.whl"

def setup_logging():
    """Configure logging for deployment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def deploy_agent(delete_existing: bool = False):
    """Deploy the agent to Vertex AI."""
    logger = logging.getLogger(__name__)

    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    if delete_existing:
        try:
            logger.info("Deleting existing agent deployment...")
            agent_engines.delete("my-agent-example")
            logger.info("Successfully deleted existing deployment")
        except Exception as e:
            logger.warning(f"Could not delete existing deployment: {e}")

    # Create ADK application
    logger.info("Creating ADK application...")
    app = AdkApp(agent=root_agent, enable_tracing=False)

    # Deploy to Vertex AI
    logger.info("Deploying to Vertex AI Agent Engines...")
    remote_app = agent_engines.create(
        app,
        requirements=[AGENT_WHL_FILE],
        extra_packages=[AGENT_WHL_FILE],
        staging_bucket=STAGING_BUCKET,
        display_name="My Agent Example",
        description="Example agent demonstrating ADK patterns",
    )

    logger.info(f"Successfully deployed agent: {remote_app.resource_name}")
    return remote_app

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy my-agent-example")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete existing deployment before creating new one"
    )

    args = parser.parse_args()
    setup_logging()

    try:
        deploy_agent(delete_existing=args.delete)
    except Exception as e:
        logging.error(f"Deployment failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

### Deployment Testing

Create deployment test in `deployment/test_deployment.py`:

```python
"""Test deployment functionality."""

import pytest
import asyncio
from google.adk.runners import CloudRunner

@pytest.mark.asyncio
async def test_deployed_agent():
    """Test that deployed agent responds correctly."""
    # This test assumes agent is already deployed
    runner = CloudRunner(
        agent_engine_name="my-agent-example",
        project_id="your-project-id",
        location="us-central1"
    )

    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="test_user"
    )

    try:
        response = await session.send(
            query="Hello, can you help me?",
            session_id=session.session_id
        )

        assert response is not None
        assert len(response.candidates) > 0
        assert response.candidates[0].content is not None

    finally:
        await runner.session_service.delete_session(
            app_name=runner.app_name,
            session_id=session.session_id
        )
```

### Build and Deploy Commands

```bash
# Build the package
python -m build

# Deploy with deletion of existing
python deployment/deploy.py --delete

# Test deployment
pytest deployment/test_deployment.py
```

## Best Practices

### Code Quality

1. **Include Apache 2.0 license headers** in all Python files:
```python
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ...
```

2. **Use type hints** consistently:
```python
from typing import Dict, List, Optional, Union
from pydantic import BaseModel

def process_data(
    input_data: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Union[Dict[str, Any], None]:
    """Process input data with optional configuration."""
    pass
```

3. **Implement proper logging**:
```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.info("Starting processing")
    try:
        # Process data
        logger.debug("Processing step completed")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
```

4. **Use structured error handling**:
```python
def safe_operation(data: str) -> Dict[str, Any]:
    """Safely perform operation with structured error handling."""
    try:
        result = process_data(data)
        return {"status": "success", "result": result}
    except ValueError as e:
        return {"status": "error", "error_type": "validation", "message": str(e)}
    except Exception as e:
        return {"status": "error", "error_type": "processing", "message": str(e)}
```

### Agent Design

1. **Clear and specific prompts**: Make instructions unambiguous and actionable
2. **Appropriate model selection**: Use `gemini-2.0-flash-001` for most cases, `gemini-2.5-pro-preview-05-06` for complex reasoning
3. **Structured outputs**: Use Pydantic models for consistent data formats
4. **Temperature settings**: Use 0.1 for deterministic responses, higher for creative tasks

### Tool Design

1. **Comprehensive documentation**: Include examples, error cases, and usage scenarios
2. **Consistent interfaces**: Use similar parameter patterns across tools
3. **Graceful degradation**: Handle errors without breaking the agent flow
4. **Input validation**: Validate all parameters and provide clear error messages

### Testing Strategy

1. **Unit tests for tools**: Test individual tool functions in isolation
2. **Integration tests for agents**: Test complete agent workflows
3. **Evaluation tests**: Use structured datasets to measure agent performance
4. **Mock external dependencies**: Use pytest mocks for external services

## Common Patterns

### Configuration Management

```python
# config.py
from pydantic_settings import BaseSettings
from typing import Optional

class AgentConfig(BaseSettings):
    """Agent configuration settings."""

    project_id: str
    location: str = "us-central1"
    model_name: str = "gemini-2.0-flash-001"
    temperature: float = 0.1
    max_tokens: int = 8192

    # Optional API keys
    api_key: Optional[str] = None

    class Config:
        env_file = ".env"
        env_prefix = "AGENT_"

# Usage
config = AgentConfig()
```

### Callback Functions

```python
# shared_libraries/callbacks.py
import logging
from google.adk.tools import ToolContext
from typing import Any, Dict

logger = logging.getLogger(__name__)

def before_tool_callback(
    tool_name: str,
    tool_input: Dict[str, Any],
    context: ToolContext
) -> None:
    """Called before tool execution."""
    logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

def after_tool_callback(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Any,
    context: ToolContext
) -> None:
    """Called after tool execution."""
    logger.info(f"Tool {tool_name} completed with output type: {type(tool_output)}")

def before_agent_callback(
    agent_name: str,
    user_input: str,
    context: ToolContext
) -> None:
    """Called before agent processing."""
    logger.info(f"Agent {agent_name} processing: {user_input[:100]}...")
```

### State Management

```python
# For sub-agents with output_key
from pydantic import BaseModel

class AgentState(BaseModel):
    """Shared state between agents."""
    current_step: str
    processed_data: dict
    confidence_score: float
    next_actions: list[str]

# Agent with state output
stateful_agent = Agent(
    name="stateful_agent",
    instruction="Process data and maintain state",
    output_schema=AgentState,
    output_key="agent_state"
)
```

### Error Handling Patterns

```python
from enum import Enum
from typing import Union
from pydantic import BaseModel

class ErrorType(Enum):
    VALIDATION = "validation"
    PROCESSING = "processing"
    EXTERNAL_API = "external_api"
    CONFIGURATION = "configuration"

class ErrorResponse(BaseModel):
    status: str = "error"
    error_type: ErrorType
    message: str
    details: dict = {}

class SuccessResponse(BaseModel):
    status: str = "success"
    result: dict
    message: str = ""

def handle_operation(data: str) -> Union[SuccessResponse, ErrorResponse]:
    """Standard error handling pattern."""
    try:
        # Validate input
        if not data:
            return ErrorResponse(
                error_type=ErrorType.VALIDATION,
                message="Input data cannot be empty"
            )

        # Process data
        result = process_data(data)
        return SuccessResponse(
            result=result,
            message="Operation completed successfully"
        )

    except ValidationError as e:
        return ErrorResponse(
            error_type=ErrorType.VALIDATION,
            message=str(e),
            details={"validation_errors": e.errors()}
        )
    except ExternalAPIError as e:
        return ErrorResponse(
            error_type=ErrorType.EXTERNAL_API,
            message="External service unavailable",
            details={"api_error": str(e)}
        )
    except Exception as e:
        return ErrorResponse(
            error_type=ErrorType.PROCESSING,
            message="Unexpected error occurred",
            details={"exception": str(e)}
        )
```

## Quick Start Checklist

When creating a new agent example:

- [ ] Create directory structure following the standard layout
- [ ] Set up `pyproject.toml` with required dependencies
- [ ] Implement main agent in `agent.py`
- [ ] Define clear prompts in `prompt.py`
- [ ] Create tools in `tools/tools.py` with proper documentation
- [ ] Add unit tests for all tools
- [ ] Add integration tests for agent functionality
- [ ] Create evaluation datasets and tests
- [ ] Implement deployment script
- [ ] Write comprehensive README with architecture diagram
- [ ] Add license headers to all Python files
- [ ] Test the complete workflow from development to deployment


This guide provides a comprehensive foundation for creating high-quality ADK agent examples that follow established patterns and best practices. For specific implementation details, refer to the existing examples in the repository.
