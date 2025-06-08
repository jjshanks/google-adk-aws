"""Test report generation and analysis utilities."""

import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


@dataclass
class TestMetrics:
    """Test execution metrics."""

    timestamp: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration: float
    coverage_percentage: float
    error_scenarios_tested: int
    edge_cases_tested: int
    success_rate: float

    @classmethod
    def from_report(cls, report: Dict[str, Any]) -> "TestMetrics":
        """Create metrics from test report."""
        summary = report["summary"]
        return cls(
            timestamp=datetime.now().isoformat(),
            total_tests=summary["total_tests"],
            passed=summary["passed"],
            failed=summary["failed"],
            skipped=summary["skipped"],
            duration=summary["overall_duration"],
            coverage_percentage=summary["average_coverage"],
            error_scenarios_tested=summary["error_scenarios_tested"],
            edge_cases_tested=summary["edge_cases_tested"],
            success_rate=summary["success_rate"],
        )


class TestReportGenerator:
    """Generate comprehensive test reports and analytics."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)

    def generate_executive_summary(self, report: Dict[str, Any]) -> str:
        """Generate executive summary report."""
        summary = report["summary"]
        quality_metrics = report["quality_metrics"]

        # Calculate health score
        health_score = self._calculate_health_score(summary, quality_metrics)

        executive_summary = f"""
# S3ArtifactService Test Executive Summary

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overall Health Score: {health_score:.0f}/100

### Key Metrics
- **Success Rate:** {summary["success_rate"]:.1%}
- **Test Coverage:** {summary["average_coverage"]:.1f}%
- **Total Tests Executed:** {summary["total_tests"]}
- **Execution Time:** {summary["overall_duration"]:.1f} seconds

### Quality Indicators
- ✅ Comprehensive Coverage: {
    "Yes" if quality_metrics["comprehensive_coverage"] else "No"
}
- ✅ Error Handling: {
    "Robust" if quality_metrics["error_handling_coverage"] else "Needs Improvement"
}
- ✅ Edge Case Testing: {
    "Complete" if quality_metrics["edge_case_coverage"] else "Incomplete"
}
- ✅ Performance Validated: {
    "Yes" if quality_metrics["performance_validated"] else "No"
}

### Risk Assessment
"""

        # Risk assessment
        risks = []
        if summary["failed"] > 0:
            risks.append(f"🔴 **HIGH RISK:** {summary['failed']} tests are failing")

        if summary["average_coverage"] < 90:
            risks.append(
                f"🟡 **MEDIUM RISK:** Test coverage below 90% "
                f"({summary['average_coverage']:.1f}%)"
            )

        if summary["error_scenarios_tested"] < 30:
            risks.append(
                f"🟡 **MEDIUM RISK:** Limited error scenario coverage "
                f"({summary['error_scenarios_tested']} scenarios)"
            )

        if not risks:
            executive_summary += (
                "🟢 **LOW RISK:** All quality metrics are within "
                "acceptable thresholds.\n"
            )
        else:
            executive_summary += "\n".join(risks) + "\n"

        # Recommendations
        if report["recommendations"]:
            executive_summary += "\n### Immediate Actions Required\n"
            for i, rec in enumerate(report["recommendations"], 1):
                executive_summary += f"{i}. {rec}\n"

        executive_summary += f"""
### Production Readiness
{
    "🟢 **READY FOR PRODUCTION**"
    if health_score >= 90 and summary["failed"] == 0
    else "🔴 **NOT READY FOR PRODUCTION**"
}

*This summary provides a high-level overview. See detailed reports for "
"technical analysis.*
"""

        return executive_summary

    def _calculate_health_score(
        self, summary: Dict[str, Any], quality_metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall health score (0-100)."""
        score = 0.0

        # Success rate (40 points)
        score += summary["success_rate"] * 40

        # Coverage (25 points)
        score += min(summary["average_coverage"] / 100, 1.0) * 25

        # Quality metrics (35 points)
        quality_score = 0
        if quality_metrics["comprehensive_coverage"]:
            quality_score += 10
        if quality_metrics["error_handling_coverage"]:
            quality_score += 10
        if quality_metrics["edge_case_coverage"]:
            quality_score += 10
        if quality_metrics["performance_validated"]:
            quality_score += 5

        score += quality_score

        return min(score, 100.0)

    def generate_trend_analysis(self, historical_reports: List[Dict[str, Any]]) -> str:
        """Generate trend analysis from historical test reports."""
        if len(historical_reports) < 2:
            return "Insufficient data for trend analysis (minimum 2 reports required)."

        metrics = [TestMetrics.from_report(report) for report in historical_reports]

        # Calculate trends
        success_rates = [m.success_rate for m in metrics]
        coverages = [m.coverage_percentage for m in metrics]
        durations = [m.duration for m in metrics]

        trend_analysis = f"""
# Test Trend Analysis

**Reports Analyzed:** {len(historical_reports)}
**Time Period:** {metrics[0].timestamp} to {metrics[-1].timestamp}

## Trends

### Success Rate Trend
- **Current:** {success_rates[-1]:.1%}
- **Previous:** {success_rates[-2]:.1%}
- **Change:** {(success_rates[-1] - success_rates[-2]) * 100:+.1f} percentage points
- **Trend:** {
    "📈 Improving" if success_rates[-1] > success_rates[-2]
    else "📉 Declining" if success_rates[-1] < success_rates[-2]
    else "📊 Stable"
}

### Coverage Trend
- **Current:** {coverages[-1]:.1f}%
- **Previous:** {coverages[-2]:.1f}%
- **Change:** {coverages[-1] - coverages[-2]:+.1f} percentage points
- **Trend:** {
    "📈 Improving" if coverages[-1] > coverages[-2]
    else "📉 Declining" if coverages[-1] < coverages[-2]
    else "📊 Stable"
}

### Performance Trend
- **Current Duration:** {durations[-1]:.1f}s
- **Previous Duration:** {durations[-2]:.1f}s
- **Change:** {durations[-1] - durations[-2]:+.1f}s
- **Trend:** {
    "📈 Slower" if durations[-1] > durations[-2]
    else "📉 Faster" if durations[-1] < durations[-2]
    else "📊 Stable"
}

## Statistical Summary
- **Average Success Rate:** {statistics.mean(success_rates):.1%}
- **Success Rate Std Dev:** {
    statistics.stdev(success_rates) if len(success_rates) > 1 else 0:.3f
}
- **Average Coverage:** {statistics.mean(coverages):.1f}%
- **Coverage Std Dev:** {statistics.stdev(coverages) if len(coverages) > 1 else 0:.1f}%
"""

        return trend_analysis

    def generate_performance_report(self, report: Dict[str, Any]) -> str:
        """Generate detailed performance analysis report."""
        # Extract performance data from suite results
        performance_suites = [
            suite
            for suite in report["suite_results"]
            if "Performance" in suite["suite_name"]
        ]

        if not performance_suites:
            return "No performance test data available."

        performance_report = """
# Performance Analysis Report

## Performance Test Results

"""

        for suite in performance_suites:
            performance_report += f"""
### {suite["suite_name"]}
- **Tests Executed:** {suite["total_tests"]}
- **Success Rate:** {(suite["passed"] / max(suite["total_tests"], 1)):.1%}
- **Execution Time:** {suite["duration"]:.2f}s
- **Average Time per Test:** {suite["duration"] / max(suite["total_tests"], 1):.3f}s

"""

        # Performance recommendations
        total_perf_duration = sum(suite["duration"] for suite in performance_suites)
        performance_report += f"""
## Performance Summary
- **Total Performance Test Duration:** {total_perf_duration:.2f}s
- **Performance Test Coverage:** {len(performance_suites)} test suite(s)

## Performance Recommendations
"""

        if total_perf_duration > 300:  # 5 minutes
            performance_report += (
                "- ⚠️ Performance tests are taking longer than 5 minutes - "
                "consider optimization\n"
            )

        if len(performance_suites) < 1:
            performance_report += (
                "- ⚠️ Limited performance test coverage - "
                "consider adding more performance tests\n"
            )

        return performance_report

    def generate_coverage_analysis(self, coverage_file: Path) -> str:
        """Generate detailed coverage analysis."""
        if not coverage_file.exists():
            return "Coverage data not available."

        try:
            with open(coverage_file, "r") as f:
                coverage_data = json.load(f)

            files = coverage_data.get("files", {})
            totals = coverage_data.get("totals", {})

            coverage_analysis = f"""
# Code Coverage Analysis

## Overall Coverage
- **Total Coverage:** {totals.get("percent_covered", 0):.1f}%
- **Lines Covered:** {totals.get("covered_lines", 0)}
- **Total Lines:** {totals.get("num_statements", 0)}
- **Missing Lines:** {totals.get("missing_lines", 0)}

## File-by-File Analysis

"""

            # Sort files by coverage percentage
            file_coverage = []
            for filepath, data in files.items():
                coverage_pct = data.get("summary", {}).get("percent_covered", 0)
                file_coverage.append((filepath, coverage_pct, data))

            file_coverage.sort(key=lambda x: x[1])  # Sort by coverage percentage

            # Show files with low coverage
            low_coverage_files = [f for f in file_coverage if f[1] < 90]
            if low_coverage_files:
                coverage_analysis += "### Files Needing Attention (< 90% coverage)\n\n"
                for filepath, coverage_pct, _ in low_coverage_files:
                    coverage_analysis += f"- **{filepath}:** {coverage_pct:.1f}%\n"
                coverage_analysis += "\n"

            # Show top performing files
            high_coverage_files = [f for f in file_coverage if f[1] >= 95][-5:]  # Top 5
            if high_coverage_files:
                coverage_analysis += "### Well-Tested Files (≥ 95% coverage)\n\n"
                for filepath, coverage_pct, _ in reversed(high_coverage_files):
                    coverage_analysis += f"- **{filepath}:** {coverage_pct:.1f}%\n"

            return coverage_analysis

        except Exception as e:
            return f"Error analyzing coverage data: {e}"

    def generate_charts(self, historical_data: List[TestMetrics]) -> None:
        """Generate trend charts."""
        if len(historical_data) < 2:
            return

        # Create charts directory
        charts_dir = self.reports_dir / "charts"
        charts_dir.mkdir(exist_ok=True)

        # Prepare data
        timestamps = [datetime.fromisoformat(m.timestamp) for m in historical_data]
        success_rates = [m.success_rate * 100 for m in historical_data]
        coverages = [m.coverage_percentage for m in historical_data]
        durations = [m.duration for m in historical_data]

        # Success rate trend
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(timestamps, success_rates, marker="o", color="green")
        plt.title("Success Rate Trend")
        plt.ylabel("Success Rate (%)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Coverage trend
        plt.subplot(2, 2, 2)
        plt.plot(timestamps, coverages, marker="o", color="blue")
        plt.title("Coverage Trend")
        plt.ylabel("Coverage (%)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Duration trend
        plt.subplot(2, 2, 3)
        plt.plot(timestamps, durations, marker="o", color="orange")
        plt.title("Execution Duration Trend")
        plt.ylabel("Duration (seconds)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Test count trend
        test_counts = [m.total_tests for m in historical_data]
        plt.subplot(2, 2, 4)
        plt.plot(timestamps, test_counts, marker="o", color="purple")
        plt.title("Test Count Trend")
        plt.ylabel("Number of Tests")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(charts_dir / "test_trends.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Charts saved to {charts_dir}")

    def save_historical_data(self, metrics: TestMetrics) -> None:
        """Save test metrics to historical data."""
        history_file = self.reports_dir / "test_history.json"

        # Load existing history
        history = []
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    history_data = json.load(f)
                    history = [TestMetrics(**item) for item in history_data]
            except Exception:
                pass  # Start fresh if file is corrupted

        # Add new metrics
        history.append(metrics)

        # Keep only last 100 entries
        history = history[-100:]

        # Save back to file
        with open(history_file, "w") as f:
            json.dump([metrics.__dict__ for metrics in history], f, indent=2)

        print(f"📝 Historical data saved ({len(history)} entries)")

    def generate_comprehensive_report(self, test_report: Dict[str, Any]) -> None:
        """Generate all reports from test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save test metrics to history
        metrics = TestMetrics.from_report(test_report)
        self.save_historical_data(metrics)

        # Load historical data for trends
        history_file = self.reports_dir / "test_history.json"
        historical_data = []
        if history_file.exists():
            with open(history_file, "r") as f:
                history_data = json.load(f)
                historical_data = [TestMetrics(**item) for item in history_data]

        # Generate executive summary
        exec_summary = self.generate_executive_summary(test_report)
        with open(self.reports_dir / f"executive_summary_{timestamp}.md", "w") as f:
            f.write(exec_summary)

        # Generate trend analysis
        if len(historical_data) >= 2:
            trend_analysis = self.generate_trend_analysis(
                [metrics.__dict__ for metrics in historical_data[-10:]]
            )
            with open(self.reports_dir / f"trend_analysis_{timestamp}.md", "w") as f:
                f.write(trend_analysis)

        # Generate performance report
        perf_report = self.generate_performance_report(test_report)
        with open(self.reports_dir / f"performance_report_{timestamp}.md", "w") as f:
            f.write(perf_report)

        # Generate coverage analysis
        coverage_file = self.project_root / "coverage.json"
        if coverage_file.exists():
            coverage_analysis = self.generate_coverage_analysis(coverage_file)
            with open(self.reports_dir / f"coverage_analysis_{timestamp}.md", "w") as f:
                f.write(coverage_analysis)

        # Generate charts
        try:
            if len(historical_data) >= 2:
                self.generate_charts(historical_data[-20:])  # Last 20 data points
        except ImportError:
            print("⚠️ matplotlib not available, skipping chart generation")

        print(f"📊 Comprehensive reports generated in {self.reports_dir}")
        print(f"📋 Executive summary: executive_summary_{timestamp}.md")


def main() -> int:
    """CLI for report generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Test report generator")
    parser.add_argument("report_file", type=Path, help="Path to test report JSON file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_reports"),
        help="Output directory for reports",
    )

    args = parser.parse_args()

    if not args.report_file.exists():
        print(f"❌ Report file not found: {args.report_file}")
        return 1

    try:
        with open(args.report_file, "r") as f:
            test_report = json.load(f)

        generator = TestReportGenerator(args.output_dir.parent)
        generator.generate_comprehensive_report(test_report)

        print("✅ Report generation complete!")
        return 0

    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return 1


if __name__ == "__main__":
    main()
