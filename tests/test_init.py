"""Test the main package."""

import aws_adk


def test_version() -> None:
    """Test that version is defined."""
    assert hasattr(aws_adk, "__version__")
    assert isinstance(aws_adk.__version__, str)