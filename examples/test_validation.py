"""Test Great Expectations validation suite on Barber R1 data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.validation import (
    validate_raw_curated,
    validate_refined,
    ValidationResult,
)
from src.utils.logging_utils import get_logger

# Setup logging
logger = get_logger("test_validation")


def load_test_data(data_path: Path, chassis_id: str) -> pd.DataFrame:
    """Load processed data for a single chassis.

    Args:
        data_path: Path to processed data directory
        chassis_id: Chassis ID to load

    Returns:
        DataFrame with telemetry data
    """
    # Look for refined parquet files
    parquet_files = list(
        data_path.glob(f"barber_r1/chassis_id={chassis_id}/segment_id=*/telemetry_name=*/*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for chassis {chassis_id} in {data_path}"
        )

    logger.info(f"Found {len(parquet_files)} parquet files for chassis {chassis_id}")

    # Load all files
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)

    # Concatenate
    df = pd.concat(dfs, ignore_index=True)

    logger.info(f"Loaded {len(df):,} rows for chassis {chassis_id}")

    return df


def test_raw_curated_validation():
    """Test validation on raw_curated data."""
    logger.info("\n" + "="*60)
    logger.info("TEST: RAW_CURATED VALIDATION")
    logger.info("="*60 + "\n")

    # Load data
    data_path = Path("data/processed")
    chassis_id = "010"

    try:
        df = load_test_data(data_path, chassis_id)
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        logger.info("Skipping raw_curated validation test")
        return None

    # Ensure required columns exist
    required_cols = [
        "timestamp",
        "meta_time",
        "vehicle_id",
        "chassis_id",
        "car_no",
        "lap",
        "telemetry_name",
        "telemetry_value",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        logger.info("Adding placeholder columns for testing...")
        for col in missing_cols:
            if col == "vehicle_id":
                df[col] = f"GR86-{chassis_id}-16"
            elif col == "car_no":
                df[col] = "16"
            elif col in ["timestamp", "meta_time"]:
                df[col] = pd.Timestamp.now()
            else:
                df[col] = None

    # Run validation
    context_root = Path("great_expectations")
    output_dir = Path("data/reports")

    result = validate_raw_curated(
        df=df,
        context_root=context_root,
        event_name="barber_r1_test",
        output_dir=output_dir,
    )

    # Display results
    logger.info("\n" + "="*60)
    logger.info("RAW_CURATED VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Pass rate: {result.pass_rate:.1f}%")
    logger.info(f"  Passed: {result.passed_expectations}/{result.total_expectations}")
    logger.info(f"  Failed: {result.failed_expectations}")
    logger.info(f"  Warnings: {result.warnings}")

    if result.errors:
        logger.info("\n  Errors/Warnings:")
        for i, error in enumerate(result.errors[:5], 1):
            logger.info(f"    {i}. {error}")
        if len(result.errors) > 5:
            logger.info(f"    ... and {len(result.errors) - 5} more")

    logger.info(f"\n  Report: {result.report_path}")
    logger.info("="*60 + "\n")

    return result


def test_refined_validation():
    """Test validation on refined data."""
    logger.info("\n" + "="*60)
    logger.info("TEST: REFINED VALIDATION")
    logger.info("="*60 + "\n")

    # Load data
    data_path = Path("data/processed")
    chassis_id = "010"

    try:
        df = load_test_data(data_path, chassis_id)
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        logger.info("Skipping refined validation test")
        return None

    # Ensure refined columns exist
    if "time_corrected" not in df.columns:
        logger.warning("time_corrected column missing, creating from timestamp")
        df["time_corrected"] = pd.to_datetime(df.get("timestamp", pd.Timestamp.now()))

    if "lap_repaired" not in df.columns:
        logger.warning("lap_repaired column missing, using lap column")
        df["lap_repaired"] = df.get("lap", 1)

    if "segment_id" not in df.columns:
        logger.warning("segment_id column missing, setting to 0")
        df["segment_id"] = 0

    # Run validation
    context_root = Path("great_expectations")
    output_dir = Path("data/reports")

    result = validate_refined(
        df=df,
        context_root=context_root,
        event_name="barber_r1_test",
        output_dir=output_dir,
    )

    # Display results
    logger.info("\n" + "="*60)
    logger.info("REFINED VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Pass rate: {result.pass_rate:.1f}%")
    logger.info(f"  Passed: {result.passed_expectations}/{result.total_expectations}")
    logger.info(f"  Failed: {result.failed_expectations}")
    logger.info(f"  Warnings: {result.warnings}")

    if result.errors:
        logger.info("\n  Errors/Warnings:")
        for i, error in enumerate(result.errors[:5], 1):
            logger.info(f"    {i}. {error}")
        if len(result.errors) > 5:
            logger.info(f"    ... and {len(result.errors) - 5} more")

    logger.info(f"\n  Report: {result.report_path}")
    logger.info("="*60 + "\n")

    return result


if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("GREAT EXPECTATIONS VALIDATION TEST")
    logger.info("="*60 + "\n")

    # Test raw_curated validation
    result1 = test_raw_curated_validation()

    # Test refined validation
    result2 = test_refined_validation()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION TEST SUMMARY")
    logger.info("="*60)

    if result1:
        logger.info(f"  RAW_CURATED: {'✅ PASS' if result1.success else '❌ FAIL'} ({result1.pass_rate:.1f}%)")
    else:
        logger.info("  RAW_CURATED: ⚠️ SKIPPED")

    if result2:
        logger.info(f"  REFINED: {'✅ PASS' if result2.success else '❌ FAIL'} ({result2.pass_rate:.1f}%)")
    else:
        logger.info("  REFINED: ⚠️ SKIPPED")

    logger.info("="*60 + "\n")
