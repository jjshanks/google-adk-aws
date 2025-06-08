"""Automated test runner with comprehensive reporting."""

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from tests.utils import TestMetricsCollector

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class TestSuite:
    """Test suite configuration."""

    name: str
    pattern: str
    markers: List[str]
    timeout: int
    parallel: bool = True


@dataclass
class TestResult:
    """Test execution result."""

    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration: float
    coverage_percentage: float
    error_scenarios_tested: int
    edge_cases_tested: int


class ComprehensiveTestRunner:
    """Comprehensive test runner with reporting."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results: List[TestResult] = []
        self.metrics_collector = TestMetricsCollector()

    def define_test_suites(self) -> List[TestSuite]:
        """Define all test suites for comprehensive testing."""
        return [
            TestSuite(
                name="Unit Tests - Core",
                pattern="tests/unit/test_s3_artifact_service_*.py",
                markers=["unit", "not slow"],
                timeout=300,
                parallel=True,
            ),
            TestSuite(
                name="Unit Tests - Error Handling",
                pattern="tests/unit/test_*error*.py",
                markers=["unit", "error_handling"],
                timeout=600,
                parallel=True,
            ),
            TestSuite(
                name="Unit Tests - Edge Cases",
                pattern="tests/unit/test_*edge*.py",
                markers=["unit", "edge_cases"],
                timeout=900,
                parallel=True,
            ),
            TestSuite(
                name="Unit Tests - Validation",
                pattern="tests/unit/test_*validation*.py",
                markers=["unit", "validation"],
                timeout=300,
                parallel=True,
            ),
            TestSuite(
                name="Performance Tests",
                pattern="tests/unit/test_*performance*.py",
                markers=["performance"],
                timeout=1800,
                parallel=False,  # Performance tests run sequentially
            ),
            TestSuite(
                name="Integration Tests",
                pattern="tests/integration/test_*.py",
                markers=["integration"],
                timeout=1200,
                parallel=True,
            ),
            TestSuite(
                name="Slow Tests",
                pattern="tests/**/*test*.py",
                markers=["slow"],
                timeout=3600,
                parallel=True,
            ),
        ]

    async def run_test_suite(self, suite: TestSuite) -> TestResult:
        """Run a single test suite and collect results."""
        print(f"Running test suite: {suite.name}")

        # Build pytest command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--tb=short",
            "--strict-markers",
            "--strict-config",
            "-v",
            f"--timeout={suite.timeout}",
            "--cov=aws_adk",
            "--cov-report=json",
            "--json-report",
            (
                f"--json-report-file="
                f"test_results_{suite.name.replace(' ', '_').lower()}.json"
            ),
        ]

        # Add markers
        if suite.markers:
            marker_expr = " and ".join(suite.markers)
            cmd.extend(["-m", marker_expr])

        # Add parallelization if enabled
        if suite.parallel:
            cmd.extend(["-n", "auto"])

        # Add test pattern
        cmd.append(suite.pattern)

        start_time = time.time()

        try:
            # Run tests
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=suite.timeout,
            )

            duration = time.time() - start_time

            # Parse results
            test_result = self._parse_test_results(suite, result, duration)

            return test_result

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"Test suite {suite.name} timed out after {duration:.2f}s")

            return TestResult(
                suite_name=suite.name,
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                duration=duration,
                coverage_percentage=0.0,
                error_scenarios_tested=0,
                edge_cases_tested=0,
            )

    def _parse_test_results(
        self, suite: TestSuite, result: subprocess.CompletedProcess, duration: float
    ) -> TestResult:
        """Parse pytest results."""
        try:
            # Try to load JSON report
            json_file = f"test_results_{suite.name.replace(' ', '_').lower()}.json"
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    json_data = json.load(f)

                summary = json_data.get("summary", {})

                return TestResult(
                    suite_name=suite.name,
                    total_tests=summary.get("total", 0),
                    passed=summary.get("passed", 0),
                    failed=summary.get("failed", 0),
                    skipped=summary.get("skipped", 0),
                    duration=duration,
                    coverage_percentage=self._extract_coverage_percentage(),
                    error_scenarios_tested=self._count_error_scenarios(json_data),
                    edge_cases_tested=self._count_edge_cases(json_data),
                )
        except Exception as e:
            print(f"Error parsing results for {suite.name}: {e}")

        # Fallback parsing from stdout
        return self._parse_stdout_results(suite, result.stdout, duration)

    def _parse_stdout_results(
        self, suite: TestSuite, stdout: str, duration: float
    ) -> TestResult:
        """Parse results from pytest stdout."""
        lines = stdout.split("\n")

        # Look for summary line like "= 10 passed, 2 failed, 1 skipped in 5.23s ="
        summary_line = None
        for line in lines:
            if " passed" in line or " failed" in line:
                if line.startswith("=") and line.endswith("="):
                    summary_line = line.strip("= ")
                    break

        passed = failed = skipped = total = 0

        if summary_line:
            parts = summary_line.split(",")
            for part in parts:
                part = part.strip()
                if " passed" in part:
                    passed = int(part.split()[0])
                elif " failed" in part:
                    failed = int(part.split()[0])
                elif " skipped" in part:
                    skipped = int(part.split()[0])

            total = passed + failed + skipped

        return TestResult(
            suite_name=suite.name,
            total_tests=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration=duration,
            coverage_percentage=0.0,  # Cannot extract from stdout
            error_scenarios_tested=0,
            edge_cases_tested=0,
        )

    def _extract_coverage_percentage(self) -> float:
        """Extract coverage percentage from coverage report."""
        try:
            if os.path.exists("coverage.json"):
                with open("coverage.json", "r") as f:
                    coverage_data = json.load(f)
                totals = coverage_data.get("totals", {})
                return float(totals.get("percent_covered", 0.0))
        except Exception:
            pass
        return 0.0

    def _count_error_scenarios(self, json_data: Dict) -> int:
        """Count error scenarios tested."""
        count = 0
        for test in json_data.get("tests", []):
            if any(
                marker in test.get("keywords", [])
                for marker in ["error_handling", "error"]
            ):
                count += 1
        return count

    def _count_edge_cases(self, json_data: Dict) -> int:
        """Count edge cases tested."""
        count = 0
        for test in json_data.get("tests", []):
            if any(
                marker in test.get("keywords", []) for marker in ["edge_cases", "edge"]
            ):
                count += 1
        return count

    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all test suites comprehensively."""
        print("Starting comprehensive test execution...")

        test_suites = self.define_test_suites()
        overall_start = time.time()

        # Run test suites
        for suite in test_suites:
            result = await self.run_test_suite(suite)
            self.test_results.append(result)

            # Print immediate results
            print(f"\nResults for {suite.name}:")
            print(f"  Total: {result.total_tests}")
            print(f"  Passed: {result.passed}")
            print(f"  Failed: {result.failed}")
            print(f"  Skipped: {result.skipped}")
            print(f"  Duration: {result.duration:.2f}s")
            print(f"  Coverage: {result.coverage_percentage:.1f}%")

        overall_duration = time.time() - overall_start

        # Generate comprehensive report
        report = self.generate_comprehensive_report(overall_duration)

        # Save report
        await self.save_reports(report)

        return report

    def generate_comprehensive_report(self, overall_duration: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = sum(r.total_tests for r in self.test_results)
        total_passed = sum(r.passed for r in self.test_results)
        total_failed = sum(r.failed for r in self.test_results)
        total_skipped = sum(r.skipped for r in self.test_results)

        # Calculate averages
        avg_coverage = (
            sum(
                r.coverage_percentage
                for r in self.test_results
                if r.coverage_percentage > 0
            )
            / len([r for r in self.test_results if r.coverage_percentage > 0])
            if any(r.coverage_percentage > 0 for r in self.test_results)
            else 0.0
        )

        total_error_scenarios = sum(r.error_scenarios_tested for r in self.test_results)
        total_edge_cases = sum(r.edge_cases_tested for r in self.test_results)

        return {
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
                "success_rate": total_passed / max(total_tests, 1),
                "overall_duration": overall_duration,
                "average_coverage": avg_coverage,
                "error_scenarios_tested": total_error_scenarios,
                "edge_cases_tested": total_edge_cases,
            },
            "suite_results": [asdict(result) for result in self.test_results],
            "quality_metrics": {
                "test_categories_covered": len(self.test_results),
                "comprehensive_coverage": avg_coverage >= 95.0,
                "error_handling_coverage": total_error_scenarios >= 50,
                "edge_case_coverage": total_edge_cases >= 30,
                "performance_validated": any(
                    "Performance" in r.suite_name for r in self.test_results
                ),
            },
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        total_failed = sum(r.failed for r in self.test_results)

        if total_failed > 0:
            recommendations.append(
                f"Address {total_failed} failing tests before production deployment"
            )

        avg_coverage = (
            sum(
                r.coverage_percentage
                for r in self.test_results
                if r.coverage_percentage > 0
            )
            / len([r for r in self.test_results if r.coverage_percentage > 0])
            if any(r.coverage_percentage > 0 for r in self.test_results)
            else 0.0
        )

        if avg_coverage < 95.0:
            recommendations.append(
                f"Increase test coverage from {avg_coverage:.1f}% to at least 95%"
            )

        error_scenarios = sum(r.error_scenarios_tested for r in self.test_results)
        if error_scenarios < 50:
            recommendations.append(
                f"Add more error scenario tests "
                f"(current: {error_scenarios}, target: 50+)"
            )

        edge_cases = sum(r.edge_cases_tested for r in self.test_results)
        if edge_cases < 30:
            recommendations.append(
                f"Add more edge case tests (current: {edge_cases}, target: 30+)"
            )

        return recommendations

    async def save_reports(self, report: Dict[str, Any]) -> None:
        """Save comprehensive reports."""
        # Save JSON report
        with open("comprehensive_test_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Save HTML report
        html_report = self._generate_html_report(report)
        with open("comprehensive_test_report.html", "w") as f:
            f.write(html_report)

        print("\nReports saved:")
        print("  JSON: comprehensive_test_report.json")
        print("  HTML: comprehensive_test_report.html")

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report."""
        summary = report["summary"]
        quality_metrics = report["quality_metrics"]

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>S3ArtifactService Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            border: 1px solid #dee2e6;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .danger {{ color: #dc3545; }}
        .suite-results {{
            margin-top: 30px;
        }}
        .suite {{
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 6px;
            overflow: hidden;
        }}
        .suite-header {{
            background: #007bff;
            color: white;
            padding: 15px;
            font-weight: bold;
        }}
        .suite-body {{
            padding: 15px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }}
        .recommendations {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 15px;
            margin-top: 30px;
        }}
        .recommendations h3 {{
            color: #856404;
            margin-top: 0;
        }}
        .recommendations ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .quality-indicators {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 30px 0;
        }}
        .indicator {{
            padding: 10px;
            border-radius: 6px;
            text-align: center;
            font-weight: bold;
        }}
        .indicator.pass {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .indicator.fail {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>S3ArtifactService Comprehensive Test Report</h1>
            <p>Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="summary">
            <div class="metric">
                <div class="metric-value {
            "success"
            if summary["success_rate"] >= 0.95
            else "warning"
            if summary["success_rate"] >= 0.8
            else "danger"
        }">{summary["success_rate"]:.1%}</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary["total_tests"]}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value success">{summary["passed"]}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric">
                <div class="metric-value {
            "danger" if summary["failed"] > 0 else "success"
        }">{summary["failed"]}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary["skipped"]}</div>
                <div class="metric-label">Skipped</div>
            </div>
            <div class="metric">
                <div class="metric-value {
            "success"
            if summary["average_coverage"] >= 95
            else "warning"
            if summary["average_coverage"] >= 80
            else "danger"
        }">{summary["average_coverage"]:.1f}%</div>
                <div class="metric-label">Coverage</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary["overall_duration"]:.1f}s</div>
                <div class="metric-label">Duration</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary["error_scenarios_tested"]}</div>
                <div class="metric-label">Error Scenarios</div>
            </div>
        </div>

        <div class="quality-indicators">
            <div class="indicator {
            "pass" if quality_metrics["comprehensive_coverage"] else "fail"
        }">
                {
            "✓" if quality_metrics["comprehensive_coverage"] else "✗"
        } Comprehensive Coverage (95%+)
            </div>
            <div class="indicator {
            "pass" if quality_metrics["error_handling_coverage"] else "fail"
        }">
                {
            "✓" if quality_metrics["error_handling_coverage"] else "✗"
        } Error Handling Coverage (50+ scenarios)
            </div>
            <div class="indicator {
            "pass" if quality_metrics["edge_case_coverage"] else "fail"
        }">
                {
            "✓" if quality_metrics["edge_case_coverage"] else "✗"
        } Edge Case Coverage (30+ cases)
            </div>
            <div class="indicator {
            "pass" if quality_metrics["performance_validated"] else "fail"
        }">
                {
            "✓" if quality_metrics["performance_validated"] else "✗"
        } Performance Validated
            </div>
        </div>

        <div class="suite-results">
            <h2>Test Suite Results</h2>
"""

        for result in report["suite_results"]:
            status_class = (
                "success"
                if result["failed"] == 0
                else "warning"
                if result["failed"] <= 2
                else "danger"
            )
            html += f"""
            <div class="suite">
                <div class="suite-header {status_class}">
                    {result["suite_name"]}
                </div>
                <div class="suite-body">
                    <div>Total: {result["total_tests"]}</div>
                    <div>Passed: {result["passed"]}</div>
                    <div>Failed: {result["failed"]}</div>
                    <div>Skipped: {result["skipped"]}</div>
                    <div>Duration: {result["duration"]:.2f}s</div>
                    <div>Coverage: {result["coverage_percentage"]:.1f}%</div>
                </div>
            </div>
"""

        if report["recommendations"]:
            html += """
        <div class="recommendations">
            <h3>Recommendations</h3>
            <ul>
"""
            for rec in report["recommendations"]:
                html += f"                <li>{rec}</li>\n"

            html += """
            </ul>
        </div>
"""

        html += """
        </div>
    </div>
</body>
</html>
"""
        return html


async def main() -> None:
    """Main test runner entry point."""
    project_root = Path(__file__).parent.parent.parent
    runner = ComprehensiveTestRunner(project_root)

    try:
        report = await runner.run_comprehensive_tests()

        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEST EXECUTION COMPLETE")
        print("=" * 60)

        summary = report["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Average Coverage: {summary['average_coverage']:.1f}%")
        print(f"Overall Duration: {summary['overall_duration']:.2f}s")

        if report["recommendations"]:
            print("\nRecommendations:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")

        # Exit with appropriate code
        exit_code = 0 if summary["failed"] == 0 else 1
        sys.exit(exit_code)

    except Exception as e:
        print(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
