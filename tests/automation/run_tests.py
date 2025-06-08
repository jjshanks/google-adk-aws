#!/usr/bin/env python3
"""Local test orchestration script for development."""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.automation.test_runner import ComprehensiveTestRunner


def run_quick_tests() -> int:
    """Run quick unit tests for development."""
    print("Running quick unit tests...")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/",
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--disable-warnings",
        "-m",
        "not slow and not performance",
        "--maxfail=5",
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    return result.returncode


def run_coverage_tests() -> int:
    """Run tests with coverage reporting."""
    print("Running tests with coverage...")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/",
        "--cov=aws_adk",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--cov-fail-under=90",
        "-v",
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)

    if result.returncode == 0:
        print("\n✅ Coverage report generated:")
        print("  HTML: htmlcov/index.html")
        print("  XML: coverage.xml")

    return result.returncode


def run_performance_tests() -> int:
    """Run performance tests."""
    print("Running performance tests...")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/test_performance_comprehensive.py",
        "-v",
        "--tb=short",
        "-m",
        "performance",
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    return result.returncode


def run_integration_tests() -> int:
    """Run integration tests."""
    print("Running integration tests...")

    # Check if LocalStack is available
    try:
        subprocess.run(
            ["curl", "-f", "http://localhost:4566/_localstack/health"],
            check=True,
            capture_output=True,
        )
        print("✅ LocalStack detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  LocalStack not available, skipping integration tests")
        print("   To run integration tests, start LocalStack:")
        print("   docker run --rm -p 4566:4566 localstack/localstack")
        return 0

    cmd = [sys.executable, "-m", "pytest", "tests/integration/", "-v", "--tb=short"]

    # Set LocalStack environment
    env = os.environ.copy()
    env.update(
        {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": "us-east-1",
            "S3_ENDPOINT_URL": "http://localhost:4566",
        }
    )

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent, env=env)
    return result.returncode


def run_security_scan() -> int:
    """Run security scans."""
    print("Running security scans...")

    exit_code = 0

    # Run Bandit for security issues
    try:
        print("  Running Bandit security scan...")
        cmd = ["bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"]
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
        if result.returncode != 0:
            print("  ⚠️  Bandit found security issues")
            exit_code = 1
        else:
            print("  ✅ No security issues found by Bandit")
    except FileNotFoundError:
        print("  ⚠️  Bandit not installed, skipping security scan")
        print("     Install with: pip install bandit")

    # Run Safety for dependency vulnerabilities
    try:
        print("  Running Safety dependency scan...")
        cmd = ["safety", "check", "--json", "--output", "safety-report.json"]
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
        if result.returncode != 0:
            print("  ⚠️  Safety found vulnerable dependencies")
            exit_code = 1
        else:
            print("  ✅ No vulnerable dependencies found")
    except FileNotFoundError:
        print("  ⚠️  Safety not installed, skipping dependency scan")
        print("     Install with: pip install safety")

    return exit_code


async def run_comprehensive_tests() -> int:
    """Run comprehensive test suite."""
    print("Running comprehensive test suite...")

    project_root = Path(__file__).parent.parent.parent
    runner = ComprehensiveTestRunner(project_root)

    try:
        report = await runner.run_comprehensive_tests()
        summary = report["summary"]

        print("\n📊 Test Summary:")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        print(f"   Coverage: {summary['average_coverage']:.1f}%")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Duration: {summary['overall_duration']:.1f}s")

        if report["recommendations"]:
            print("\n📋 Recommendations:")
            for rec in report["recommendations"]:
                print(f"   - {rec}")

        return 0 if summary["failed"] == 0 else 1

    except Exception as e:
        print(f"❌ Comprehensive test execution failed: {e}")
        return 1


def run_lint_and_format() -> int:
    """Run linting and formatting checks."""
    print("Running linting and formatting...")

    exit_code = 0
    project_root = Path(__file__).parent.parent.parent

    # Run ruff linting
    try:
        print("  Running ruff linting...")
        result = subprocess.run(["ruff", "check", "src/", "tests/"], cwd=project_root)
        if result.returncode != 0:
            exit_code = 1
    except FileNotFoundError:
        print("  ⚠️  ruff not installed")

    # Run ruff formatting
    try:
        print("  Running ruff formatting...")
        result = subprocess.run(
            ["ruff", "format", "--check", "src/", "tests/"], cwd=project_root
        )
        if result.returncode != 0:
            exit_code = 1
    except FileNotFoundError:
        print("  ⚠️  ruff not installed")

    # Run mypy type checking
    try:
        print("  Running mypy type checking...")
        result = subprocess.run(["mypy", "src/"], cwd=project_root)
        if result.returncode != 0:
            exit_code = 1
    except FileNotFoundError:
        print("  ⚠️  mypy not installed")

    return exit_code


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test orchestration for S3ArtifactService"
    )
    parser.add_argument(
        "command",
        choices=[
            "quick",
            "coverage",
            "performance",
            "integration",
            "security",
            "comprehensive",
            "lint",
            "all",
        ],
        help="Test command to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        os.environ["PYTEST_VERBOSE"] = "1"

    exit_code = 0

    if args.command == "quick":
        exit_code = run_quick_tests()
    elif args.command == "coverage":
        exit_code = run_coverage_tests()
    elif args.command == "performance":
        exit_code = run_performance_tests()
    elif args.command == "integration":
        exit_code = run_integration_tests()
    elif args.command == "security":
        exit_code = run_security_scan()
    elif args.command == "comprehensive":
        exit_code = asyncio.run(run_comprehensive_tests())
    elif args.command == "lint":
        exit_code = run_lint_and_format()
    elif args.command == "all":
        print("🚀 Running all test categories...")

        # Run in order of speed/importance
        commands = [
            ("Lint & Format", run_lint_and_format),
            ("Quick Tests", run_quick_tests),
            ("Coverage Tests", run_coverage_tests),
            ("Security Scan", run_security_scan),
            ("Performance Tests", run_performance_tests),
            ("Integration Tests", run_integration_tests),
        ]

        results = {}
        for name, func in commands:
            print(f"\n{'='*60}")
            print(f"Running {name}")
            print("=" * 60)

            if asyncio.iscoroutinefunction(func):
                result = asyncio.run(func())
            else:
                result = func()

            results[name] = result
            if result != 0:
                print(f"❌ {name} failed")
                exit_code = 1
            else:
                print(f"✅ {name} passed")

        # Final summary
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print("=" * 60)
        for name, result in results.items():
            status = "✅ PASS" if result == 0 else "❌ FAIL"
            print(f"{name:20} {status}")

        if exit_code == 0:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n💥 Some tests failed (exit code: {exit_code})")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
