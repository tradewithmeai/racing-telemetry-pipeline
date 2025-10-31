"""Test time synchronization on Barber R1 data."""

import sys
from pathlib import Path
import polars as pl
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.transform.time_sync import (
    process_car_time_sync,
    align_to_global_time,
    save_drift_calibrations,
)
from src.utils.logging_utils import get_logger
from src.utils.time_utils import detect_backwards_time

logger = get_logger(__name__)


def load_car_data(chassis_id: str, event_name: str = "barber_r1") -> pd.DataFrame:
    """Load all telemetry for one car from raw_curated partitions.

    Args:
        chassis_id: Chassis identifier (e.g., "002", "004")
        event_name: Event name

    Returns:
        DataFrame with timestamp, meta_time, telemetry_name, telemetry_value
    """
    logger.info(f"Loading data for chassis {chassis_id}")

    base_path = Path("data/processed") / event_name / "raw_curated" / f"event={event_name}" / f"chassis_id={chassis_id}"

    if not base_path.exists():
        raise FileNotFoundError(f"No data found for chassis {chassis_id} at {base_path}")

    # Load all telemetry signals
    all_dfs = []
    for signal_dir in base_path.iterdir():
        if signal_dir.is_dir():
            signal_name = signal_dir.name.replace("telemetry_name=", "")

            # Read all parquet files in this signal partition
            parquet_files = list(signal_dir.glob("*.parquet"))

            for pq_file in parquet_files:
                df_signal = pl.read_parquet(pq_file).to_pandas()
                all_dfs.append(df_signal)

    if not all_dfs:
        raise ValueError(f"No data loaded for chassis {chassis_id}")

    # Combine all signals
    df = pd.concat(all_dfs, ignore_index=True)

    logger.info(f"  Loaded {len(df):,} rows across {len(all_dfs)} signal partitions")

    return df


def analyze_car_before_after(
    chassis_id: str,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    calibrations,
) -> dict:
    """Analyze time sync results for one car.

    Args:
        chassis_id: Chassis identifier
        df_before: Original data
        df_after: Corrected data
        calibrations: List of DriftCalibration objects

    Returns:
        Dict with analysis results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Analysis: {chassis_id}")
    logger.info(f"{'='*60}")

    # Before stats
    backwards_before = detect_backwards_time(df_before["timestamp"]).sum()
    time_span_before = (df_before["timestamp"].max() - df_before["timestamp"].min()).total_seconds()

    # After stats
    backwards_after = detect_backwards_time(df_after["time_corrected"]).sum()
    time_span_after = (df_after["time_corrected"].max() - df_after["time_corrected"].min()).total_seconds()

    # Calibration stats
    valid_calibrations = [c for c in calibrations if c.is_valid]
    avg_drift = sum(c.drift_sec for c in valid_calibrations) / len(valid_calibrations) if valid_calibrations else 0.0
    avg_std = sum(c.drift_std for c in valid_calibrations) / len(valid_calibrations) if valid_calibrations else 0.0

    logger.info(f"\n[BEFORE Correction]")
    logger.info(f"  Backwards timestamps: {backwards_before:,}")
    logger.info(f"  Time span: {time_span_before:.1f} seconds")

    logger.info(f"\n[AFTER Correction]")
    logger.info(f"  Backwards timestamps: {backwards_after:,}")
    logger.info(f"  Time span: {time_span_after:.1f} seconds")
    if backwards_before > 0:
        logger.info(f"  Backwards reduction: {backwards_before - backwards_after:,} ({100*(backwards_before - backwards_after)/backwards_before:.1f}% eliminated)")
    else:
        logger.info(f"  Backwards reduction: 0 (no backwards timestamps present)")

    logger.info(f"\n[Drift Calibrations]")
    logger.info(f"  Total windows: {len(calibrations)}")
    logger.info(f"  Valid calibrations: {len(valid_calibrations)}/{len(calibrations)}")
    logger.info(f"  Average drift: {avg_drift:.3f} ± {avg_std:.3f} seconds")

    # Quality breakdown
    quality_counts = {}
    for c in calibrations:
        quality_counts[c.quality_level] = quality_counts.get(c.quality_level, 0) + 1

    logger.info(f"\n[Calibration Quality]")
    for quality, count in sorted(quality_counts.items()):
        logger.info(f"  {quality}: {count} windows")

    # Clock steps
    step_count = sum(c.step_detected for c in calibrations)
    if step_count > 0:
        logger.warning(f"  [WARNING] Clock steps detected: {step_count}")

    return {
        "chassis_id": chassis_id,
        "backwards_before": backwards_before,
        "backwards_after": backwards_after,
        "backwards_eliminated_pct": 100 * (backwards_before - backwards_after) / backwards_before if backwards_before > 0 else 0,
        "total_calibrations": len(calibrations),
        "valid_calibrations": len(valid_calibrations),
        "avg_drift_sec": avg_drift,
        "avg_drift_std": avg_std,
        "quality_counts": quality_counts,
        "clock_steps": step_count,
    }


def test_time_sync(chassis_ids: list, event_name: str = "barber_r1"):
    """Test time synchronization on multiple cars.

    Args:
        chassis_ids: List of chassis identifiers to test
        event_name: Event name
    """
    logger.info(f"\n{'#'*80}")
    logger.info(f"# Time Synchronization Test: {event_name}")
    logger.info(f"# Testing {len(chassis_ids)} vehicles")
    logger.info(f"{'#'*80}\n")

    results = {}
    processed_dfs = {}
    all_calibrations = {}

    for chassis_id in chassis_ids:
        try:
            # Load data
            df = load_car_data(chassis_id, event_name)

            # Store original for comparison
            df_before = df.copy()

            # Run time sync pipeline
            df_after, calibrations = process_car_time_sync(
                df,
                chassis_id=f"GR86-{chassis_id}",
                window_minutes=5,
                gap_threshold_sec=2.0,
                method="median",
            )

            # Analyze results
            result = analyze_car_before_after(
                chassis_id, df_before, df_after, calibrations
            )
            results[chassis_id] = result
            processed_dfs[chassis_id] = df_after
            all_calibrations[chassis_id] = calibrations

        except Exception as e:
            logger.error(f"Failed to process chassis {chassis_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save calibrations
    if all_calibrations:
        output_path = Path("data/processed") / event_name
        save_drift_calibrations(all_calibrations, output_path, event_name)

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY: Time Synchronization Test")
    logger.info(f"{'='*80}\n")

    for chassis_id, result in results.items():
        logger.info(f"{chassis_id}:")
        logger.info(f"  Backwards eliminated: {result['backwards_eliminated_pct']:.1f}%")
        logger.info(f"  Valid calibrations: {result['valid_calibrations']}/{result['total_calibrations']}")
        logger.info(f"  Avg drift: {result['avg_drift_sec']:.3f}s ± {result['avg_drift_std']:.3f}s")

        if result['backwards_after'] == 0:
            logger.info(f"  [PERFECT] All backwards timestamps eliminated")
        elif result['backwards_eliminated_pct'] >= 99:
            logger.info(f"  [EXCELLENT] 99%+ backwards eliminated")
        elif result['backwards_eliminated_pct'] >= 90:
            logger.info(f"  [GOOD] 90%+ backwards eliminated")
        else:
            logger.warning(f"  [NEEDS REVIEW] <90% backwards eliminated")
        logger.info("")

    return results, processed_dfs, all_calibrations


if __name__ == "__main__":
    # Test on 4 cars:
    # - 2 "clean" cars (004, 013) with 0 backwards timestamps
    # - 2 "problematic" cars (010, 002) with ~60k backwards timestamps each

    test_chassis = [
        "004",  # Clean car
        "013",  # Clean car
        "010",  # Problematic car (60k backwards)
        "002",  # Problematic car (59k backwards)
    ]

    results, processed_dfs, calibrations = test_time_sync(test_chassis)

    logger.info("\n[SUCCESS] Time synchronization test complete!")
    logger.info(f"Processed {len(results)} vehicles")
    if results:
        logger.info(f"Drift calibrations saved to: data/processed/barber_r1/drift_calibration.parquet")
