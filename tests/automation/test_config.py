"""Test configuration management and environment setup."""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TestEnvironment:
    """Test environment configuration."""

    name: str
    python_version: str
    dependencies: List[str]
    environment_vars: Dict[str, str]
    markers: List[str]
    timeout: int = 300


@dataclass
class TestConfiguration:
    """Comprehensive test configuration."""

    project_root: Path
    test_environments: List[TestEnvironment]
    coverage_threshold: float = 95.0
    performance_threshold: Dict[str, float] = None
    quality_gates: Dict[str, any] = None

    def __post_init__(self):
        if self.performance_threshold is None:
            self.performance_threshold = {
                "save_artifact_max_time": 0.1,
                "load_artifact_max_time": 0.05,
                "list_operations_max_time": 0.02,
                "validation_max_time": 0.001,
            }

        if self.quality_gates is None:
            self.quality_gates = {
                "min_success_rate": 0.95,
                "min_coverage": 95.0,
                "max_failed_tests": 0,
                "min_error_scenarios": 50,
                "min_edge_cases": 30,
                "performance_required": True,
            }


class TestConfigurationManager:
    """Manages test configuration and environment setup."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config = self._load_default_configuration()

    def _load_default_configuration(self) -> TestConfiguration:
        """Load default test configuration."""
        return TestConfiguration(
            project_root=self.project_root,
            test_environments=[
                TestEnvironment(
                    name="unit-core",
                    python_version="3.11",
                    dependencies=["pytest", "pytest-asyncio", "pytest-cov"],
                    environment_vars={},
                    markers=["unit", "not slow", "not performance"],
                    timeout=300,
                ),
                TestEnvironment(
                    name="unit-error-handling",
                    python_version="3.11",
                    dependencies=["pytest", "pytest-asyncio", "moto"],
                    environment_vars={},
                    markers=["unit", "error_handling"],
                    timeout=600,
                ),
                TestEnvironment(
                    name="unit-edge-cases",
                    python_version="3.11",
                    dependencies=["pytest", "pytest-asyncio", "moto"],
                    environment_vars={},
                    markers=["unit", "edge_cases"],
                    timeout=900,
                ),
                TestEnvironment(
                    name="performance",
                    python_version="3.11",
                    dependencies=["pytest", "pytest-asyncio", "pytest-benchmark"],
                    environment_vars={},
                    markers=["performance"],
                    timeout=1800,
                ),
                TestEnvironment(
                    name="integration",
                    python_version="3.11",
                    dependencies=["pytest", "pytest-asyncio", "boto3", "moto"],
                    environment_vars={
                        "AWS_ACCESS_KEY_ID": "test",
                        "AWS_SECRET_ACCESS_KEY": "test",
                        "AWS_DEFAULT_REGION": "us-east-1",
                        "S3_ENDPOINT_URL": "http://localhost:4566",
                    },
                    markers=["integration"],
                    timeout=1200,
                ),
            ],
        )

    def setup_environment(self, env_name: str) -> bool:
        """Setup test environment."""
        env = self._get_environment(env_name)
        if not env:
            print(f"❌ Environment '{env_name}' not found")
            return False

        print(f"🔧 Setting up environment: {env.name}")

        # Check Python version
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if current_version != env.python_version:
            print(
                f"⚠️  Python version mismatch: {current_version} != {env.python_version}"
            )

        # Install dependencies
        if env.dependencies:
            print(f"📦 Installing dependencies: {', '.join(env.dependencies)}")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + env.dependencies,
                    check=True,
                    capture_output=True,
                )
                print("✅ Dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install dependencies: {e}")
                return False

        # Set environment variables
        for key, value in env.environment_vars.items():
            os.environ[key] = value
            print(f"🌍 Set {key}={value}")

        return True

    def _get_environment(self, name: str) -> Optional[TestEnvironment]:
        """Get environment by name."""
        for env in self.config.test_environments:
            if env.name == name:
                return env
        return None

    def validate_environment(self, env_name: str) -> bool:
        """Validate test environment readiness."""
        env = self._get_environment(env_name)
        if not env:
            return False

        print(f"🔍 Validating environment: {env.name}")

        # Check if required tools are available
        required_tools = ["pytest"]
        if "integration" in env.name:
            required_tools.extend(["curl"])  # For LocalStack health checks

        for tool in required_tools:
            try:
                subprocess.run(
                    [tool, "--version" if tool != "curl" else "--help"],
                    check=True,
                    capture_output=True,
                )
                print(f"✅ {tool} available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"❌ {tool} not available")
                return False

        # Validate special requirements
        if env.name == "integration":
            return self._validate_integration_environment()

        return True

    def _validate_integration_environment(self) -> bool:
        """Validate integration test environment."""
        # Check if LocalStack is running
        try:
            subprocess.run(
                ["curl", "-f", "http://localhost:4566/_localstack/health"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            print("✅ LocalStack is running")
            return True
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            print("❌ LocalStack not available")
            print("   Start with: docker run --rm -p 4566:4566 localstack/localstack")
            return False

    def get_pytest_args(self, env_name: str) -> List[str]:
        """Get pytest arguments for environment."""
        env = self._get_environment(env_name)
        if not env:
            return []

        args = [
            "--tb=short",
            "--strict-markers",
            "--strict-config",
            "-v",
            f"--timeout={env.timeout}",
        ]

        # Add markers
        if env.markers:
            marker_expr = " and ".join(env.markers)
            args.extend(["-m", marker_expr])

        # Add coverage for unit tests
        if "unit" in env.name:
            args.extend(
                [
                    "--cov=aws_adk",
                    "--cov-report=term-missing",
                    f"--cov-fail-under={self.config.coverage_threshold}",
                ]
            )

        # Add performance benchmarking
        if env.name == "performance":
            args.extend(["--benchmark-only"])

        return args

    def generate_test_matrix(self) -> List[Dict[str, str]]:
        """Generate test matrix for CI/CD."""
        matrix = []

        for env in self.config.test_environments:
            matrix.append(
                {
                    "name": env.name,
                    "python-version": env.python_version,
                    "markers": " and ".join(env.markers),
                    "timeout": str(env.timeout),
                }
            )

        return matrix

    def check_quality_gates(self, test_results: Dict[str, any]) -> bool:
        """Check if test results meet quality gates."""
        gates = self.config.quality_gates

        print("🚦 Checking quality gates...")

        # Check success rate
        success_rate = test_results.get("success_rate", 0.0)
        if success_rate < gates["min_success_rate"]:
            print(
                f"❌ Success rate {success_rate:.1%} < {gates['min_success_rate']:.1%}"
            )
            return False
        print(f"✅ Success rate: {success_rate:.1%}")

        # Check coverage
        coverage = test_results.get("coverage", 0.0)
        if coverage < gates["min_coverage"]:
            print(f"❌ Coverage {coverage:.1f}% < {gates['min_coverage']:.1f}%")
            return False
        print(f"✅ Coverage: {coverage:.1f}%")

        # Check failed tests
        failed_tests = test_results.get("failed", 0)
        if failed_tests > gates["max_failed_tests"]:
            print(f"❌ Failed tests {failed_tests} > {gates['max_failed_tests']}")
            return False
        print(f"✅ Failed tests: {failed_tests}")

        # Check error scenarios
        error_scenarios = test_results.get("error_scenarios_tested", 0)
        if error_scenarios < gates["min_error_scenarios"]:
            print(
                f"❌ Error scenarios {error_scenarios} < {gates['min_error_scenarios']}"
            )
            return False
        print(f"✅ Error scenarios: {error_scenarios}")

        # Check edge cases
        edge_cases = test_results.get("edge_cases_tested", 0)
        if edge_cases < gates["min_edge_cases"]:
            print(f"❌ Edge cases {edge_cases} < {gates['min_edge_cases']}")
            return False
        print(f"✅ Edge cases: {edge_cases}")

        # Check performance validation
        if gates["performance_required"]:
            performance_validated = test_results.get("performance_validated", False)
            if not performance_validated:
                print("❌ Performance validation required but not found")
                return False
            print("✅ Performance validated")

        print("🎉 All quality gates passed!")
        return True

    def save_configuration(self, filepath: Path) -> None:
        """Save configuration to file."""
        import json
        from dataclasses import asdict

        config_dict = asdict(self.config)
        # Convert Path to string for JSON serialization
        config_dict["project_root"] = str(self.config.project_root)

        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"💾 Configuration saved to {filepath}")

    def load_configuration(self, filepath: Path) -> None:
        """Load configuration from file."""
        import json

        with open(filepath, "r") as f:
            config_dict = json.load(f)

        # Convert string back to Path
        config_dict["project_root"] = Path(config_dict["project_root"])

        # Recreate configuration object
        self.config = TestConfiguration(**config_dict)

        print(f"📁 Configuration loaded from {filepath}")


def main():
    """Main CLI for test configuration management."""
    import argparse

    parser = argparse.ArgumentParser(description="Test configuration management")
    parser.add_argument(
        "command",
        choices=["setup", "validate", "matrix", "save", "load"],
        help="Configuration command",
    )
    parser.add_argument(
        "--environment",
        "-e",
        help="Environment name",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Configuration file path",
    )

    args = parser.parse_args()

    manager = TestConfigurationManager()

    if args.command == "setup":
        if not args.environment:
            print("❌ Environment name required for setup")
            sys.exit(1)

        if not manager.setup_environment(args.environment):
            sys.exit(1)

    elif args.command == "validate":
        if not args.environment:
            print("❌ Environment name required for validation")
            sys.exit(1)

        if not manager.validate_environment(args.environment):
            sys.exit(1)

    elif args.command == "matrix":
        matrix = manager.generate_test_matrix()
        import json

        print(json.dumps(matrix, indent=2))

    elif args.command == "save":
        if not args.file:
            args.file = Path("test_config.json")
        manager.save_configuration(args.file)

    elif args.command == "load":
        if not args.file:
            print("❌ Configuration file required for load")
            sys.exit(1)
        manager.load_configuration(args.file)


if __name__ == "__main__":
    main()
