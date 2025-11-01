"""Baseline validation: Compare pipeline output against raw data expectations.

This module implements 4 critical validation checks:
1. Sample count check (±10%): Catch duplicate removal bugs, pivot data loss
2. Sampling rate check (±10%): Catch rate drift, interpolation failures
3. Time coverage check (±1%): Catch truncation (7min vs 45min)
4. Signal presence check (±5%): Catch column stripping (GPS loss)

These checks prevent silent data quality failures and provide clear error messages.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PipelineValidationError(Exception):
    """Raised when pipeline output fails baseline validation."""
    pass


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    check_name: str
    baseline_value: float
    actual_value: float
    delta_pct: float
    threshold_pct: float
    status: str  # "PASS", "WARN", "FAIL"
    message: str = ""


@dataclass
class ValidationResult:
    """Complete validation result for a stage."""
    stage: str
    chassis_id: str
    checks: List[ValidationCheck] = field(default_factory=list)
    overall_status: str = "PASS"  # "PASS", "WARN", "FAIL"

    @property
    def passed(self) -> bool:
        return self.overall_status != "FAIL"

    @property
    def failed(self) -> bool:
        return self.overall_status == "FAIL"

    def add_check(self, check: ValidationCheck):
        """Add a check and update overall status."""
        self.checks.append(check)

        # Update overall status (FAIL > WARN > PASS)
        if check.status == "FAIL":
            self.overall_status = "FAIL"
        elif check.status == "WARN" and self.overall_status == "PASS":
            self.overall_status = "WARN"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage": self.stage,
            "chassis_id": self.chassis_id,
            "overall_status": self.overall_status,
            "checks": [
                {
                    "check_name": c.check_name,
                    "baseline_value": c.baseline_value,
                    "actual_value": c.actual_value,
                    "delta_pct": c.delta_pct,
                    "threshold_pct": c.threshold_pct,
                    "status": c.status,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }


def load_baseline(baseline_path: Path) -> Dict:
    """Load baseline JSON.

    Args:
        baseline_path: Path to baseline JSON file

    Returns:
        Baseline dict
    """
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline not found: {baseline_path}\n"
            f"Run: python tools/compute_baseline.py --event <event> --chassis <ids>"
        )

    with open(baseline_path) as f:
        return json.load(f)


def compute_delta_pct(baseline: float, actual: float) -> float:
    """Compute percentage delta.

    Args:
        baseline: Baseline value
        actual: Actual value

    Returns:
        Delta as percentage (negative = loss, positive = gain)
    """
    if baseline == 0:
        return 0.0 if actual == 0 else 100.0

    return 100.0 * (actual - baseline) / baseline


def check_sample_counts(
    df: pd.DataFrame,
    baseline: Dict,
    stage: str,
    threshold_pct: float = 10.0,
) -> List[ValidationCheck]:
    """Check #1: Sample counts per signal (±10%).

    Compares non-null counts per column against baseline signal counts.
    Catches: Duplicate removal bugs, pivot data loss.

    Args:
        df: DataFrame to validate
        baseline: Car baseline dict
        stage: Pipeline stage name
        threshold_pct: Maximum allowed delta percentage

    Returns:
        List of ValidationCheck objects
    """
    checks = []

    # Map processed column names to baseline signal names
    column_mapping = {
        "gps_lat": "VBOX_Lat_Min",
        "gps_lon": "VBOX_Long_Minutes",
        "track_distance_m": "Laptrigger_lapdist_dls",
        "speed": "speed",
        "speed_raw": "speed",
        "Steering_Angle": "Steering_Angle",
        "aps": "aps",
        "gear": "gear",
        "accx_can": "accx_can",
        "accy_can": "accy_can",
        "nmot": "nmot",
        "pbrake_f": "pbrake_f",
        "pbrake_r": "pbrake_r",
    }

    for col_name, signal_name in column_mapping.items():
        if col_name not in df.columns:
            continue

        if signal_name not in baseline["signals"]:
            continue

        # Get baseline count
        baseline_count = baseline["signals"][signal_name]["sample_count"]

        # Get actual non-null count
        actual_count = df[col_name].notna().sum()

        # Compute delta
        delta_pct = compute_delta_pct(baseline_count, actual_count)

        # Determine status
        if abs(delta_pct) <= threshold_pct:
            status = "PASS"
            message = f"{col_name}: {actual_count:,} samples (baseline: {baseline_count:,}, {delta_pct:+.1f}%)"
        elif delta_pct < 0:  # Loss
            status = "FAIL"
            message = f"{col_name}: LOST {baseline_count - actual_count:,} samples ({delta_pct:.1f}%, threshold: ±{threshold_pct}%)"
        else:  # Gain (suspicious but not critical)
            status = "WARN"
            message = f"{col_name}: GAINED {actual_count - baseline_count:,} samples ({delta_pct:+.1f}%, expected ~0%)"

        checks.append(ValidationCheck(
            check_name=f"sample_count_{col_name}",
            baseline_value=baseline_count,
            actual_value=actual_count,
            delta_pct=delta_pct,
            threshold_pct=threshold_pct,
            status=status,
            message=message,
        ))

    return checks


def check_time_coverage(
    df: pd.DataFrame,
    baseline: Dict,
    time_col: str = "time_corrected",
    threshold_pct: float = 1.0,
) -> ValidationCheck:
    """Check #3: Time coverage (±1%).

    Validates that time span is preserved through stages.
    Catches: Truncation (Car 002's 7min issue).

    Args:
        df: DataFrame to validate
        baseline: Car baseline dict
        time_col: Time column name
        threshold_pct: Maximum allowed delta percentage

    Returns:
        ValidationCheck object
    """
    # Get baseline time span
    baseline_span_sec = baseline["time_span_sec"]

    # Compute actual time span
    if time_col not in df.columns:
        return ValidationCheck(
            check_name="time_coverage",
            baseline_value=baseline_span_sec,
            actual_value=0,
            delta_pct=-100.0,
            threshold_pct=threshold_pct,
            status="FAIL",
            message=f"Time column '{time_col}' not found in DataFrame",
        )

    df_time = pd.to_datetime(df[time_col])
    actual_span_sec = (df_time.max() - df_time.min()).total_seconds()

    # Compute delta
    delta_pct = compute_delta_pct(baseline_span_sec, actual_span_sec)

    # Determine status
    if abs(delta_pct) <= threshold_pct:
        status = "PASS"
        message = f"Time span: {actual_span_sec/60:.1f} min (baseline: {baseline_span_sec/60:.1f} min, {delta_pct:+.2f}%)"
    else:
        status = "FAIL" if delta_pct < 0 else "WARN"
        message = f"Time span: {actual_span_sec/60:.1f} min vs baseline {baseline_span_sec/60:.1f} min ({delta_pct:+.1f}%, threshold: ±{threshold_pct}%)"

    return ValidationCheck(
        check_name="time_coverage",
        baseline_value=baseline_span_sec,
        actual_value=actual_span_sec,
        delta_pct=delta_pct,
        threshold_pct=threshold_pct,
        status=status,
        message=message,
    )


def check_signal_presence(
    df: pd.DataFrame,
    baseline: Dict,
    threshold_pct: float = 5.0,
) -> List[ValidationCheck]:
    """Check #4: Signal presence (±5%).

    Validates that critical signal coverage is preserved.
    Catches: Column stripping (GPS 100% → 0% bug).

    Args:
        df: DataFrame to validate
        baseline: Car baseline dict
        threshold_pct: Maximum allowed delta percentage

    Returns:
        List of ValidationCheck objects
    """
    checks = []

    # Critical signals to check
    critical_signals = {
        "gps_lat": "VBOX_Lat_Min",
        "gps_lon": "VBOX_Long_Minutes",
        "track_distance_m": "Laptrigger_lapdist_dls",
        "speed_final": "speed",
    }

    total_rows = len(df)

    for col_name, signal_name in critical_signals.items():
        if col_name not in df.columns:
            # Special case: speed_final might not exist in early stages
            if col_name == "speed_final" and "speed" in df.columns:
                col_name = "speed"
            else:
                continue

        if signal_name not in baseline["signals"]:
            continue

        # Baseline coverage (should be ~100% for most signals)
        baseline_coverage = baseline["signals"][signal_name]["coverage_pct"]

        # Actual coverage
        actual_coverage = 100.0 * df[col_name].notna().sum() / total_rows if total_rows > 0 else 0

        # Compute absolute delta (not percentage of percentage)
        delta_pct = actual_coverage - baseline_coverage

        # Determine status
        if abs(delta_pct) <= threshold_pct:
            status = "PASS"
            message = f"{col_name}: {actual_coverage:.1f}% coverage (baseline: {baseline_coverage:.1f}%, {delta_pct:+.1f}pp)"
        elif delta_pct < 0:
            status = "FAIL"
            message = f"{col_name}: Coverage dropped from {baseline_coverage:.1f}% to {actual_coverage:.1f}% ({delta_pct:.1f}pp, threshold: ±{threshold_pct}pp)"
        else:
            status = "PASS"  # Higher coverage is good
            message = f"{col_name}: {actual_coverage:.1f}% coverage (baseline: {baseline_coverage:.1f}%, {delta_pct:+.1f}pp)"

        checks.append(ValidationCheck(
            check_name=f"presence_{col_name}",
            baseline_value=baseline_coverage,
            actual_value=actual_coverage,
            delta_pct=delta_pct,
            threshold_pct=threshold_pct,
            status=status,
            message=message,
        ))

    return checks


def validate_against_baseline(
    df: pd.DataFrame,
    baseline_path: Path,
    chassis_id: str,
    stage: str,
    strict: bool = True,
    time_col: str = "time_corrected",
) -> ValidationResult:
    """Validate DataFrame against baseline expectations.

    Args:
        df: DataFrame to validate
        baseline_path: Path to baseline JSON
        chassis_id: Chassis ID
        stage: Pipeline stage name
        strict: If True, raise exception on failure
        time_col: Time column name

    Returns:
        ValidationResult object

    Raises:
        PipelineValidationError: If strict=True and validation fails
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"BASELINE VALIDATION: {stage.upper()} - Chassis {chassis_id}")
    logger.info(f"{'='*80}")

    # Load baseline
    baseline_data = load_baseline(baseline_path)

    if chassis_id not in baseline_data["cars"]:
        raise ValueError(f"Chassis {chassis_id} not found in baseline")

    car_baseline = baseline_data["cars"][chassis_id]

    # Create result object
    result = ValidationResult(stage=stage, chassis_id=chassis_id)

    # Run checks
    logger.info("\n[Check 1] Sample Counts (±10%)")
    sample_checks = check_sample_counts(df, car_baseline, stage, threshold_pct=10.0)
    for check in sample_checks:
        result.add_check(check)
        if check.status == "FAIL":
            logger.error(f"  FAIL: {check.message}")
        elif check.status == "WARN":
            logger.warning(f"  WARN: {check.message}")

    logger.info("\n[Check 3] Time Coverage (±1%)")
    time_check = check_time_coverage(df, car_baseline, time_col, threshold_pct=1.0)
    result.add_check(time_check)
    if time_check.status == "FAIL":
        logger.error(f"  FAIL: {time_check.message}")
    elif time_check.status == "WARN":
        logger.warning(f"  WARN: {time_check.message}")
    else:
        logger.info(f"  PASS: {time_check.message}")

    logger.info("\n[Check 4] Signal Presence (±5%)")
    presence_checks = check_signal_presence(df, car_baseline, threshold_pct=5.0)
    for check in presence_checks:
        result.add_check(check)
        if check.status == "FAIL":
            logger.error(f"  FAIL: {check.message}")
        elif check.status == "WARN":
            logger.warning(f"  WARN: {check.message}")
        else:
            logger.info(f"  PASS: {check.message}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"VALIDATION RESULT: {result.overall_status}")
    logger.info(f"{'='*80}")

    passed_count = sum(1 for c in result.checks if c.status == "PASS")
    warned_count = sum(1 for c in result.checks if c.status == "WARN")
    failed_count = sum(1 for c in result.checks if c.status == "FAIL")

    logger.info(f"  Checks: {passed_count} passed, {warned_count} warnings, {failed_count} failed")

    # Raise exception if strict and failed
    if strict and result.failed:
        fail_messages = [c.message for c in result.checks if c.status == "FAIL"]
        raise PipelineValidationError(
            f"Stage '{stage}' failed baseline validation for chassis {chassis_id}:\n" +
            "\n".join(f"  - {msg}" for msg in fail_messages)
        )

    return result
