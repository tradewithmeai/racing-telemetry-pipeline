"""Test pivot transformation on Barber R1 data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.transform.pivot import (
    pivot_to_wide_format,
    save_wide_format,
    save_pivot_stats,
    PivotStats,
)
from src.utils.logging_utils import get_logger

# Setup logging
logger = get_logger("test_pivot")


def load_refined_data(data_path: Path, chassis_id: str) -> pd.DataFrame:
    """Load refined (time-corrected, lap-repaired) data for a single chassis.

    Args:
        data_path: Path to processed data directory
        chassis_id: Chassis ID to load

    Returns:
        DataFrame with refined telemetry data (long format)
    """
    # Look for refined parquet files
    parquet_pattern = f"barber_r1/chassis_id={chassis_id}/segment_id=*/telemetry_name=*/*.parquet"
    parquet_files = list(data_path.glob(parquet_pattern))

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


def test_pivot_single_car():
    """Test pivot on a single car's data."""
    logger.info("\n" + "="*60)
    logger.info("TEST: PIVOT SINGLE CAR")
    logger.info("="*60 + "\n")

    # Load data
    data_path = Path("data/processed")
    chassis_id = "010"

    try:
        df_long = load_refined_data(data_path, chassis_id)
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        logger.info("Skipping pivot test")
        return None

    # Display sample of long format
    logger.info("\nLong format sample (first 10 rows):")
    sample_cols = ["time_corrected", "chassis_id", "telemetry_name", "telemetry_value"]
    sample_cols = [col for col in sample_cols if col in df_long.columns]
    logger.info(f"\n{df_long[sample_cols].head(10).to_string()}")

    # Run pivot
    df_wide, stats = pivot_to_wide_format(
        df=df_long,
        chassis_id=chassis_id,
        time_col="time_corrected",
        signal_col="telemetry_name",
        value_col="telemetry_value",
        preserve_cols=["chassis_id", "car_no", "lap_repaired", "segment_id"],
    )

    # Display results
    logger.info("\n" + "="*60)
    logger.info("PIVOT RESULTS")
    logger.info("="*60)
    logger.info(f"  Chassis: {stats.chassis_id}")
    logger.info(f"  Rows: {stats.rows_before:,} → {stats.rows_after:,}")
    logger.info(f"  Signals: {stats.signals_before} → {stats.signals_after}")
    logger.info(f"  Coverage: {stats.coverage_pct:.1f}%")

    if stats.missing_signals:
        logger.info(f"  Missing signals: {stats.missing_signals}")

    # Display sample of wide format
    logger.info("\nWide format sample (first 5 rows):")
    logger.info(f"Columns: {list(df_wide.columns)}")
    logger.info(f"\n{df_wide.head(5).to_string()}")

    # Display signal columns
    signal_cols = [col for col in df_wide.columns if col not in ["time_corrected", "chassis_id", "car_no", "lap_repaired", "segment_id"]]
    logger.info(f"\nSignal columns ({len(signal_cols)}):")
    for i, col in enumerate(signal_cols, 1):
        non_null = df_wide[col].notna().sum()
        coverage = (non_null / len(df_wide)) * 100
        logger.info(f"  {i:2d}. {col:25s} - {non_null:6,} values ({coverage:5.1f}% coverage)")

    # Save results
    output_path = Path("data/processed")
    output_file = save_wide_format(
        df=df_wide,
        output_path=output_path,
        event_name="barber_r1_test",
        chassis_id=chassis_id,
    )

    logger.info(f"\n  Wide format saved to: {output_file}")

    # Save stats
    stats_dict = {chassis_id: stats}
    save_pivot_stats(
        stats_by_car=stats_dict,
        output_path=output_path,
        event_name="barber_r1_test",
    )

    logger.info("="*60 + "\n")

    return df_wide, stats


def test_pivot_multiple_cars():
    """Test pivot on multiple cars."""
    logger.info("\n" + "="*60)
    logger.info("TEST: PIVOT MULTIPLE CARS")
    logger.info("="*60 + "\n")

    data_path = Path("data/processed")
    test_chassis = ["010", "002"]

    all_stats = {}

    for chassis_id in test_chassis:
        logger.info(f"\nProcessing chassis {chassis_id}...")

        try:
            df_long = load_refined_data(data_path, chassis_id)

            df_wide, stats = pivot_to_wide_format(
                df=df_long,
                chassis_id=chassis_id,
                time_col="time_corrected",
                signal_col="telemetry_name",
                value_col="telemetry_value",
            )

            all_stats[chassis_id] = stats

            # Save
            output_path = Path("data/processed")
            save_wide_format(
                df=df_wide,
                output_path=output_path,
                event_name="barber_r1_test",
                chassis_id=chassis_id,
            )

        except FileNotFoundError as e:
            logger.error(f"Data not found for chassis {chassis_id}: {e}")
            continue

    # Save combined stats
    if all_stats:
        save_pivot_stats(
            stats_by_car=all_stats,
            output_path=Path("data/processed"),
            event_name="barber_r1_test",
        )

        # Summary
        logger.info("\n" + "="*60)
        logger.info("MULTI-CAR PIVOT SUMMARY")
        logger.info("="*60)
        logger.info(f"  Cars processed: {len(all_stats)}")

        for chassis_id, stats in all_stats.items():
            logger.info(f"\n  {chassis_id}:")
            logger.info(f"    Rows: {stats.rows_before:,} → {stats.rows_after:,}")
            logger.info(f"    Signals: {stats.signals_after}")
            logger.info(f"    Coverage: {stats.coverage_pct:.1f}%")

        logger.info("="*60 + "\n")

    return all_stats


if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("PIVOT TRANSFORMATION TEST")
    logger.info("="*60 + "\n")

    # Test single car
    result1 = test_pivot_single_car()

    # Test multiple cars
    result2 = test_pivot_multiple_cars()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("PIVOT TEST SUMMARY")
    logger.info("="*60)

    if result1:
        logger.info("  Single car pivot: PASS")
    else:
        logger.info("  Single car pivot: SKIP (no data)")

    if result2:
        logger.info(f"  Multi-car pivot: PASS ({len(result2)} cars)")
    else:
        logger.info("  Multi-car pivot: SKIP (no data)")

    logger.info("="*60 + "\n")
