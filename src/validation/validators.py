"""Validation functions using Great Expectations."""

import great_expectations as gx
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json

from src.validation.suite_builder import build_expectation_suite, ExpectationLevel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Results from validation run."""

    level: str
    success: bool
    total_expectations: int
    passed_expectations: int
    failed_expectations: int
    warnings: int
    errors: List[str]
    validation_time: datetime
    report_path: Optional[Path] = None

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        if self.total_expectations == 0:
            return 0.0
        return (self.passed_expectations / self.total_expectations) * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level,
            "success": self.success,
            "total_expectations": self.total_expectations,
            "passed_expectations": self.passed_expectations,
            "failed_expectations": self.failed_expectations,
            "warnings": self.warnings,
            "pass_rate": self.pass_rate,
            "errors": self.errors,
            "validation_time": self.validation_time.isoformat(),
            "report_path": str(self.report_path) if self.report_path else None,
        }


def run_validation(
    df: pd.DataFrame,
    suite: gx.core.ExpectationSuite,
    batch_name: str = "telemetry_batch",
) -> Tuple[bool, Dict]:
    """Run validation on DataFrame using expectation suite.

    Args:
        df: DataFrame to validate
        suite: ExpectationSuite to use
        batch_name: Name for this validation batch

    Returns:
        (success: bool, results: dict) tuple
    """
    logger.info(f"Running validation: {suite.name}")
    logger.info(f"  DataFrame shape: {df.shape}")
    logger.info(f"  Expectations: {len(suite.expectations)}")

    # Create a simple validation without full context
    results = {
        "success": True,
        "statistics": {
            "evaluated_expectations": 0,
            "successful_expectations": 0,
            "unsuccessful_expectations": 0,
        },
        "results": [],
    }

    # Manually run each expectation
    for expectation in suite.expectations:
        try:
            # Run the expectation
            result = expectation.validate(df)

            results["statistics"]["evaluated_expectations"] += 1

            if result.success:
                results["statistics"]["successful_expectations"] += 1
            else:
                results["statistics"]["unsuccessful_expectations"] += 1
                results["success"] = False

            results["results"].append({
                "success": result.success,
                "expectation_type": type(expectation).__name__,
                "expectation_config": {
                    "kwargs": expectation.configuration.kwargs,
                    "meta": {},
                },
                "result": result.result if hasattr(result, "result") else {},
            })

        except Exception as e:
            logger.warning(f"Expectation failed with error: {e}")
            results["statistics"]["evaluated_expectations"] += 1
            results["statistics"]["unsuccessful_expectations"] += 1
            results["success"] = False

            results["results"].append({
                "success": False,
                "expectation_type": type(expectation).__name__,
                "error": str(e),
            })

    # Extract summary
    success = results["success"]
    statistics = results["statistics"]

    passed = statistics["successful_expectations"]
    failed = statistics["unsuccessful_expectations"]
    total = statistics["evaluated_expectations"]

    logger.info(f"  Validation {'PASSED' if success else 'FAILED'}")
    logger.info(f"  Passed: {passed}/{total}")
    logger.info(f"  Failed: {failed}/{total}")

    return success, results


def extract_errors_and_warnings(results: Dict) -> Tuple[List[str], List[str]]:
    """Extract error and warning messages from validation results.

    Args:
        results: GE validation results dict

    Returns:
        (errors, warnings) tuple of lists
    """
    errors = []
    warnings = []

    for result in results.get("results", []):
        if not result["success"]:
            expectation_type = result.get("expectation_type", "Unknown")
            error_msg = result.get("error", "Validation failed")

            msg = f"{expectation_type}: {error_msg}"

            # For now, treat all as warnings
            warnings.append(msg)

    return errors, warnings


def validate_raw_curated(
    df: pd.DataFrame,
    context_root: Path,
    event_name: str,
    output_dir: Path,
) -> ValidationResult:
    """Validate raw_curated data layer.

    Args:
        df: DataFrame with raw curated data (long format)
        context_root: Path to GE context
        event_name: Event identifier
        output_dir: Output directory for reports

    Returns:
        ValidationResult with validation summary
    """
    logger.info("="*60)
    logger.info("VALIDATION: RAW_CURATED")
    logger.info("="*60)

    # Build expectation suite
    suite = build_expectation_suite(ExpectationLevel.RAW_CURATED)

    # Run validation
    success, results = run_validation(
        df=df,
        suite=suite,
        batch_name=f"{event_name}_raw_curated",
    )

    # Extract errors and warnings
    errors, warnings = extract_errors_and_warnings(results)

    # Create validation result
    validation_result = ValidationResult(
        level="raw_curated",
        success=success,
        total_expectations=results["statistics"]["evaluated_expectations"],
        passed_expectations=results["statistics"]["successful_expectations"],
        failed_expectations=results["statistics"]["unsuccessful_expectations"],
        warnings=len(warnings),
        errors=errors + warnings,
        validation_time=datetime.now(),
    )

    # Save report
    report_path = output_dir / event_name / "validation_raw_curated.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(validation_result.to_dict(), f, indent=2)

    validation_result.report_path = report_path

    logger.info(f"\n  Validation report saved to: {report_path}")
    logger.info(f"  Pass rate: {validation_result.pass_rate:.1f}%")
    logger.info(f"  Warnings: {validation_result.warnings}")
    logger.info(f"  Errors: {len(errors)}")
    logger.info("="*60 + "\n")

    return validation_result


def validate_refined(
    df: pd.DataFrame,
    context_root: Path,
    event_name: str,
    output_dir: Path,
) -> ValidationResult:
    """Validate refined data layer.

    Args:
        df: DataFrame with refined data (time-corrected, lap-repaired)
        context_root: Path to GE context
        event_name: Event identifier
        output_dir: Output directory for reports

    Returns:
        ValidationResult with validation summary
    """
    logger.info("="*60)
    logger.info("VALIDATION: REFINED")
    logger.info("="*60)

    # Build expectation suite
    suite = build_expectation_suite(ExpectationLevel.REFINED)

    # Run validation
    success, results = run_validation(
        df=df,
        suite=suite,
        batch_name=f"{event_name}_refined",
    )

    # Extract errors and warnings
    errors, warnings = extract_errors_and_warnings(results)

    # Create validation result
    validation_result = ValidationResult(
        level="refined",
        success=success,
        total_expectations=results["statistics"]["evaluated_expectations"],
        passed_expectations=results["statistics"]["successful_expectations"],
        failed_expectations=results["statistics"]["unsuccessful_expectations"],
        warnings=len(warnings),
        errors=errors + warnings,
        validation_time=datetime.now(),
    )

    # Save report
    report_path = output_dir / event_name / "validation_refined.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(validation_result.to_dict(), f, indent=2)

    validation_result.report_path = report_path

    logger.info(f"\n  Validation report saved to: {report_path}")
    logger.info(f"  Pass rate: {validation_result.pass_rate:.1f}%")
    logger.info(f"  Warnings: {validation_result.warnings}")
    logger.info(f"  Errors: {len(errors)}")
    logger.info("="*60 + "\n")

    return validation_result


def validate_simulation_ready(
    df: pd.DataFrame,
    context_root: Path,
    event_name: str,
    output_dir: Path,
) -> ValidationResult:
    """Validate simulation-ready data.

    Args:
        df: DataFrame with simulation-ready data (wide format, synchronized)
        context_root: Path to GE context
        event_name: Event identifier
        output_dir: Output directory for reports

    Returns:
        ValidationResult with validation summary
    """
    logger.info("="*60)
    logger.info("VALIDATION: SIMULATION_READY")
    logger.info("="*60)

    # Build expectation suite
    suite = build_expectation_suite(ExpectationLevel.SIMULATION_READY)

    # Run validation
    success, results = run_validation(
        df=df,
        suite=suite,
        batch_name=f"{event_name}_simulation_ready",
    )

    # Extract errors and warnings
    errors, warnings = extract_errors_and_warnings(results)

    # Create validation result
    validation_result = ValidationResult(
        level="simulation_ready",
        success=success,
        total_expectations=results["statistics"]["evaluated_expectations"],
        passed_expectations=results["statistics"]["successful_expectations"],
        failed_expectations=results["statistics"]["unsuccessful_expectations"],
        warnings=len(warnings),
        errors=errors + warnings,
        validation_time=datetime.now(),
    )

    # Save report
    report_path = output_dir / event_name / "validation_simulation_ready.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(validation_result.to_dict(), f, indent=2)

    validation_result.report_path = report_path

    logger.info(f"\n  Validation report saved to: {report_path}")
    logger.info(f"  Pass rate: {validation_result.pass_rate:.1f}%")
    logger.info(f"  Warnings: {validation_result.warnings}")
    logger.info(f"  Errors: {len(errors)}")
    logger.info("="*60 + "\n")

    return validation_result
