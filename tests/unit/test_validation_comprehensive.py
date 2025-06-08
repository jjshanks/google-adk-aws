"""Comprehensive unit tests for input validation and edge cases."""

# mypy: ignore-errors

import asyncio
from typing import Any, Dict

import pytest

from aws_adk.exceptions import S3ValidationError
from aws_adk.validation import InputValidator
from tests.utils import ValidationTester


@pytest.mark.unit
@pytest.mark.validation
class TestInputValidationComprehensive:
    """Comprehensive input validation testing."""

    @pytest.fixture
    def validator(self) -> InputValidator:
        """Input validator instance."""
        return InputValidator()

    @pytest.fixture
    def validation_test_data(self) -> Dict[str, Dict[str, Any]]:
        """Comprehensive validation test data."""

        return {
            "valid_cases": {
                "basic": {
                    "app_name": "test-app",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "document.txt",
                    "version": 0,
                },
                "with_special_chars": {
                    "app_name": "test-app_v2",
                    "user_id": "user-123",
                    "session_id": "session_456",
                    "filename": "my-document_v2.txt",
                    "version": 5,
                },
                "alphanumeric": {
                    "app_name": "test123",
                    "user_id": "user456",
                    "session_id": "session789",
                    "filename": "document123.txt",
                    "version": 0,
                },
            },
            "invalid_cases": {
                "empty_app_name": {
                    "app_name": "",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "document.txt",
                    "version": 0,
                },
                "whitespace_app_name": {
                    "app_name": "   ",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "document.txt",
                    "version": 0,
                },
                "too_long_app_name": {
                    "app_name": "x" * 256,
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "document.txt",
                    "version": 0,
                },
                "path_traversal": {
                    "app_name": "test-app",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "../../../etc/passwd",
                    "version": 0,
                },
                "null_bytes": {
                    "app_name": "test-app",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "test\x00.txt",
                    "version": 0,
                },
                "negative_version": {
                    "app_name": "test-app",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "document.txt",
                    "version": -1,
                },
                "huge_version": {
                    "app_name": "test-app",
                    "user_id": "user123",
                    "session_id": "session456",
                    "filename": "document.txt",
                    "version": 2**31,
                },
            },
        }

    def test_all_validation_rules(
        self, validator: InputValidator, validation_test_data: Dict[str, Dict[str, Any]]
    ) -> None:
        """Test all validation rules comprehensively."""

        # Create a wrapper function for artifact parameter validation
        def validate_artifact_params_wrapper(**kwargs):
            return validator.validate_artifact_params(
                app_name=kwargs["app_name"],
                user_id=kwargs["user_id"],
                session_id=kwargs["session_id"],
                filename=kwargs["filename"],
                version=kwargs.get("version"),
            )

        results = ValidationTester.test_all_validation_rules(
            validate_artifact_params_wrapper,
            validation_test_data["valid_cases"],
            validation_test_data["invalid_cases"],
        )

        # All valid cases should pass
        assert (
            results["valid_failed"] == 0
        ), f"Valid cases failed: {results['failures']}"

        # All invalid cases should be caught
        assert (
            results["invalid_missed"] == 0
        ), f"Invalid cases missed: {results['failures']}"

        # Verify comprehensive coverage
        assert results["valid_passed"] == len(validation_test_data["valid_cases"])
        assert results["invalid_caught"] == len(validation_test_data["invalid_cases"])

    @pytest.mark.parametrize(
        "field,invalid_value,expected_error",
        [
            ("app_name", "", "cannot be empty"),
            ("app_name", "x" * 256, "too long"),
            ("user_id", None, "required"),
            ("filename", "../test.txt", "path traversal"),
            ("filename", "test\x00.txt", "null bytes"),
            ("version", -1, "cannot be negative"),
        ],
    )
    def test_specific_validation_errors(
        self,
        validator: InputValidator,
        field: str,
        invalid_value: Any,
        expected_error: str,
    ) -> None:
        """Test specific validation error messages."""
        test_data = {
            "app_name": "test-app",
            "user_id": "user123",
            "session_id": "session456",
            "filename": "test.txt",
            "version": 0,
        }
        test_data[field] = invalid_value

        with pytest.raises(S3ValidationError) as exc_info:
            validator.validate_artifact_params(**test_data)

        assert expected_error.lower() in str(exc_info.value).lower()

    def test_filename_validation_edge_cases(self, validator: InputValidator) -> None:
        """Test filename validation edge cases."""
        edge_cases = {
            "very_long_filename": "x" * 1000 + ".txt",
            "only_extension": ".txt",
            "no_extension": "filename",
            "multiple_dots": "file.name.with.dots.txt",
            "unicode_filename": "файл.txt",
            "mixed_case": "FileNAME.TXT",
        }

        for case_name, filename in edge_cases.items():
            try:
                validator.validate_artifact_params(
                    app_name="test-app",
                    user_id="user123",
                    session_id="session456",
                    filename=filename,
                    version=0,
                )
                # If we get here, validation passed
                assert (
                    len(filename) <= 255
                ), f"Long filename {case_name} should have failed"
                assert not filename.startswith(
                    "."
                ), f"Extension-only {case_name} should have failed"
            except S3ValidationError:
                # Validation failed as expected for problematic cases
                assert (
                    len(filename) > 255
                    or filename.startswith(".")
                    or any(
                        char in filename for char in ["<", ">", ":", '"', "|", "?", "*"]
                    )
                )

    def test_metadata_validation_edge_cases(self, validator: InputValidator) -> None:
        """Test metadata validation edge cases."""
        # Test various metadata scenarios
        test_cases = [
            # Empty metadata should be fine
            {},
            # Small metadata should be fine
            {"key1": "value1", "key2": "value2"},
            # Metadata with special characters
            {"app-version": "1.0.0", "user_type": "premium"},
        ]

        for metadata in test_cases:
            # Should not raise an exception
            validator.validate_metadata(metadata)

        # Test large metadata that should fail
        large_metadata = {"key" + str(i): "value" * 100 for i in range(100)}

        with pytest.raises(S3ValidationError) as exc_info:
            validator.validate_metadata(large_metadata)

        assert "metadata too large" in str(exc_info.value).lower()

    def test_concurrent_validation(self, validator: InputValidator) -> None:
        """Test validation under concurrent access."""

        async def validate_concurrently():
            tasks = []
            for i in range(50):
                task = asyncio.create_task(
                    asyncio.to_thread(
                        validator.validate_artifact_params,
                        app_name=f"app-{i}",
                        user_id=f"user-{i}",
                        session_id=f"session-{i}",
                        filename=f"file-{i}.txt",
                        version=i,
                    )
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"Concurrent validation failed: {exceptions}"

        asyncio.run(validate_concurrently())

    def test_cross_field_validation(self, validator: InputValidator) -> None:
        """Test cross-field validation rules."""
        # Test case where app_name, user_id, and session_id are identical
        with pytest.raises(S3ValidationError) as exc_info:
            validator.validate_artifact_params(
                app_name="test",
                user_id="test",
                session_id="test",
                filename="file.txt",
                version=0,
            )

        assert "identical" in str(exc_info.value).lower()

    def test_version_validation_edge_cases(self, validator: InputValidator) -> None:
        """Test version validation edge cases."""
        # Valid versions
        valid_versions = [0, 1, 100, 999999]
        for version in valid_versions:
            validator.validate_artifact_params(
                app_name="test-app",
                user_id="test-user",
                session_id="test-session",
                filename="test.txt",
                version=version,
            )

        # Invalid versions
        invalid_versions = [-1, -100, 2**32, float("inf")]
        for version in invalid_versions:
            with pytest.raises(S3ValidationError):
                validator.validate_artifact_params(
                    app_name="test-app",
                    user_id="test-user",
                    session_id="test-session",
                    filename="test.txt",
                    version=version,
                )

    def test_object_key_validation(self, validator: InputValidator) -> None:
        """Test object key validation."""
        # Test object key generation and validation
        valid_params = {
            "app_name": "test-app",
            "user_id": "test-user",
            "session_id": "test-session",
            "filename": "test.txt",
            "version": 0,
        }

        # Should not raise an exception
        validator.validate_artifact_params(**valid_params)

        # Test with parameters that would create invalid object key
        invalid_key_params = {
            "app_name": "test/app",  # Contains slash
            "user_id": "test-user",
            "session_id": "test-session",
            "filename": "test.txt",
            "version": 0,
        }

        with pytest.raises(S3ValidationError) as exc_info:
            validator.validate_artifact_params(**invalid_key_params)

        assert "invalid" in str(exc_info.value).lower()

    def test_performance_validation(self, validator: InputValidator) -> None:
        """Test validation performance under load."""
        import time

        start_time = time.time()

        # Run 1000 validations
        for i in range(1000):
            validator.validate_artifact_params(
                app_name=f"app-{i % 100}",
                user_id=f"user-{i % 100}",
                session_id=f"session-{i % 100}",
                filename=f"file-{i % 100}.txt",
                version=i % 10,
            )

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete 1000 validations in under 1 second
        assert (
            total_time < 1.0
        ), f"Validation too slow: {total_time:.3f}s for 1000 validations"

        # Average validation should be under 1ms
        avg_time = total_time / 1000
        assert avg_time < 0.001, f"Average validation too slow: {avg_time:.6f}s"


@pytest.mark.unit
@pytest.mark.validation
class TestValidationErrorMessages:
    """Test validation error message clarity and usefulness."""

    @pytest.fixture
    def validator(self) -> InputValidator:
        """Input validator instance."""
        return InputValidator()

    def test_error_message_contains_field_name(self, validator: InputValidator) -> None:
        """Test that error messages contain the field name."""
        with pytest.raises(S3ValidationError) as exc_info:
            validator.validate_artifact_params(
                app_name="",  # Empty app_name
                user_id="test-user",
                session_id="test-session",
                filename="test.txt",
                version=0,
            )

        error_message = str(exc_info.value)
        assert (
            "app_name" in error_message.lower() or "app name" in error_message.lower()
        )

    def test_error_message_contains_validation_rule(
        self, validator: InputValidator
    ) -> None:
        """Test that error messages explain what validation rule failed."""
        with pytest.raises(S3ValidationError) as exc_info:
            validator.validate_artifact_params(
                app_name="x" * 300,  # Too long
                user_id="test-user",
                session_id="test-session",
                filename="test.txt",
                version=0,
            )

        error_message = str(exc_info.value)
        assert any(
            keyword in error_message.lower()
            for keyword in ["too long", "length", "maximum"]
        )

    def test_error_context_preservation(self, validator: InputValidator) -> None:
        """Test that validation errors preserve context."""
        with pytest.raises(S3ValidationError) as exc_info:
            validator.validate_artifact_params(
                app_name="test-app",
                user_id="test-user",
                session_id="test-session",
                filename="../invalid.txt",  # Path traversal
                version=0,
            )

        error = exc_info.value
        assert error.context is not None
        assert "filename" in error.context
        assert error.context["filename"] == "../invalid.txt"

    def test_multiple_validation_errors(self, validator: InputValidator) -> None:
        """Test handling of multiple validation errors."""
        with pytest.raises(S3ValidationError) as exc_info:
            validator.validate_artifact_params(
                app_name="",  # Empty
                user_id="",  # Also empty
                session_id="test-session",
                filename="../invalid.txt",  # Path traversal
                version=-1,  # Negative
            )

        error_message = str(exc_info.value)
        # Should mention multiple errors
        assert (
            ";" in error_message
            or "," in error_message
            or "and" in error_message.lower()
        )
