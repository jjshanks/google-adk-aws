"""Shared test configuration and fixtures."""

import asyncio
import os
from typing import Any, Generator

import pytest


def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring real AWS resources"
    )
    config.addinivalue_line("markers", "slow: Slow running tests (>10 seconds)")
    config.addinivalue_line("markers", "unit: Fast unit tests")


def pytest_collection_modifyitems(config, items) -> None:  # type: ignore
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session")  # type: ignore[misc]
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(autouse=True)  # type: ignore[misc]
def clean_environment() -> Generator[None, None, None]:
    """Clean environment variables between tests."""
    original_env = os.environ.copy()
    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
