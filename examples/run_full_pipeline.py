"""Complete end-to-end pipeline: Raw data → Synchronized multi-car frames.

This script runs all pipeline stages in sequence:
1. Load raw_curated data (from ingestion)
2. Time synchronization (drift correction)
3. Lap repair (boundary detection)
4. Position normalization (GPS/track)
5. Pivot (long → wide format)
6. Resample (uniform 20Hz grid per car)
7. Multi-car synchronization
8. Validation

Usage:
    python examples/run_full_pipeline.py --chassis 010 002 --event barber_r1
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from typing import Dict, List

from src.transform import (
    # Time sync
    apply_drift_correction,
    windowed_drift_calibration,
    # Lap repair
    repair_laps,
    # Position
    normalize_position_data,
    # Pivot
    pivot_to_wide_format,
    save_pivot_stats,
    # Resample
    resample_to_time_grid,
    save_resample_stats,
    # Sync
    synchronize_multi_car,
    save_synchronized_data,
    save_sync_stats,
)
from src.validation import validate_simulation_ready
from src.utils.logging_utils import get_logger

logger = get_logger("full_pipeline")


def load_raw_curated_data(data_path: Path, event_name: str, chassis_id: str) -> pd.DataFrame:
    """Load raw curated data for a single chassis.

    Args:
        data_path: Path to processed data directory
        event_name: Event name (e.g., 'barber_r1')
        chassis_id: Chassis ID to load

    Returns:
        DataFrame with raw telemetry data (long format)
    """
    logger.info(f"Loading raw curated data for chassis {chassis_id}...")

    # Correct path to raw_curated data
    base_path = data_path / event_name / "raw_curated" / f"event={event_name}"
    parquet_pattern = f"chassis_id={chassis_id}/telemetry_name=*/*.parquet"

    parquet_files = list(base_path.glob(parquet_pattern))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for chassis {chassis_id} at {base_path / parquet_pattern}"
        )

    logger.info(f"  Found {len(parquet_files)} parquet files")

    # Load all files
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)

    # Concatenate
    df = pd.concat(dfs, ignore_index=True)

    logger.info(f"  Loaded {len(df):,} rows")

    return df


def apply_time_sync(df: pd.DataFrame, chassis_id: str) -> pd.DataFrame:
    """Apply time synchronization (drift correction).

    Args:
        df: Raw telemetry DataFrame with timestamp and meta_time columns
        chassis_id: Vehicle identifier

    Returns:
        DataFrame with time_corrected column
    """
    logger.info(f"\n[STAGE 1: TIME SYNC] {chassis_id}")
    logger.info("="*60)

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    if not pd.api.types.is_datetime64_any_dtype(df['meta_time']):
        df['meta_time'] = pd.to_datetime(df['meta_time'])

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Compute drift calibration (expects df with timestamp and meta_time columns)
    calibrations = windowed_drift_calibration(
        df=df,
        chassis_id=chassis_id,
        window_minutes=5,
        method='median',
    )

    logger.info(f"  Calibrated {len(calibrations)} time windows")

    # Apply drift correction
    df_corrected = apply_drift_correction(
        df=df,
        calibrations=calibrations,
    )

    logger.info(f"  Applied drift correction to {len(df_corrected):,} rows")

    return df_corrected


def apply_lap_repair(df: pd.DataFrame, chassis_id: str, event: str = "barber") -> pd.DataFrame:
    """Apply lap repair (detect boundaries, assign lap numbers).

    Args:
        df: Time-corrected telemetry DataFrame
        chassis_id: Vehicle identifier
        event: Event name (e.g., 'barber')

    Returns:
        DataFrame with lap_repaired column
    """
    logger.info(f"\n[STAGE 2: LAP REPAIR] {chassis_id}")
    logger.info("="*60)

    # Repair laps
    df_repaired, boundaries = repair_laps(
        df=df,
        chassis_id=chassis_id,
        event=event,
        track_length_m=3700.0,
        min_lap_duration_sec=85.0,
        max_lap_duration_sec=300.0,
    )

    logger.info(f"  Detected {len(boundaries)} lap boundaries")
    logger.info(f"  Repaired {len(df_repaired):,} rows")

    return df_repaired


def apply_position_normalization(df: pd.DataFrame, chassis_id: str) -> pd.DataFrame:
    """Apply position normalization (GPS, track distance).

    Args:
        df: Lap-repaired telemetry DataFrame
        chassis_id: Vehicle identifier

    Returns:
        DataFrame with normalized position columns
    """
    logger.info(f"\n[STAGE 3: POSITION NORMALIZATION] {chassis_id}")
    logger.info("="*60)

    # Normalize position data
    df_normalized, quality = normalize_position_data(
        df=df,
        chassis_id=chassis_id,
        circuit='barber',
        max_jump_meters=500.0,
        interpolate=True,
    )

    logger.info(f"  Quality score: {quality.quality_score:.2f}")
    logger.info(f"  GPS coverage: {quality.gps_coverage_pct:.1f}%")

    return df_normalized


def run_pipeline(
    chassis_ids: List[str],
    event_name: str = "barber_r1",
    data_path: Path = Path("data/processed"),
    output_event_name: str = "barber_r1_pipeline",
    skip_validation: bool = False,
):
    """Run complete end-to-end pipeline for multiple cars.

    Args:
        chassis_ids: List of chassis IDs to process
        event_name: Input event name
        data_path: Path to processed data directory
        output_event_name: Output event name for results
        skip_validation: Skip validation stage (GX 1.x has API compatibility issues)
    """
    logger.info("\n" + "="*60)
    logger.info("FULL PIPELINE: END-TO-END PROCESSING")
    logger.info("="*60)
    logger.info(f"  Event: {event_name}")
    logger.info(f"  Cars: {chassis_ids}")
    logger.info(f"  Output: {output_event_name}")
    logger.info("="*60 + "\n")

    # Track statistics
    pivot_stats = {}
    resample_stats = {}
    dfs_resampled = {}

    # Process each car
    for chassis_id in chassis_ids:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# PROCESSING CHASSIS {chassis_id}")
        logger.info(f"{'#'*60}\n")

        try:
            # Load raw curated data
            df = load_raw_curated_data(data_path, event_name, chassis_id)

            # Stage 1: Time sync
            df = apply_time_sync(df, chassis_id)

            # Stage 2: Lap repair
            df = apply_lap_repair(df, chassis_id)

            # Stage 3: Position normalization
            df = apply_position_normalization(df, chassis_id)

            # Stage 4: Pivot (long → wide format)
            logger.info(f"\n[STAGE 4: PIVOT] {chassis_id}")
            logger.info("="*60)

            df_wide, pivot_stat = pivot_to_wide_format(
                df=df,
                chassis_id=chassis_id,
                time_col='time_corrected',
                signal_col='telemetry_name',
                value_col='telemetry_value',
                preserve_cols=['chassis_id', 'car_no', 'lap_repaired', 'segment_id'],
            )

            pivot_stats[chassis_id] = pivot_stat

            # Stage 5: Resample to 20Hz
            logger.info(f"\n[STAGE 5: RESAMPLE] {chassis_id}")
            logger.info("="*60)

            df_resampled, resample_stat = resample_to_time_grid(
                df=df_wide,
                chassis_id=chassis_id,
                time_col='time_corrected',
                freq_hz=20.0,
                ffill_limit_sec=0.2,
                max_gap_sec=2.0,
            )

            resample_stats[chassis_id] = resample_stat
            dfs_resampled[chassis_id] = df_resampled

            logger.info(f"✅ Successfully processed chassis {chassis_id}")

        except Exception as e:
            logger.error(f"❌ Failed to process chassis {chassis_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Stage 6: Multi-car synchronization
    if len(dfs_resampled) == 0:
        logger.error("\n❌ No cars processed successfully. Exiting.")
        return

    logger.info(f"\n{'#'*60}")
    logger.info("# STAGE 6: MULTI-CAR SYNCHRONIZATION")
    logger.info(f"{'#'*60}\n")

    df_sync, sync_stats_obj, coverage_by_car = synchronize_multi_car(
        dfs_by_car=dfs_resampled,
        event_name=output_event_name,
        time_col='time_corrected',
        freq_hz=20.0,
        ffill_limit_sec=0.2,
    )

    logger.info(f"✅ Synchronized {len(dfs_resampled)} cars")
    logger.info(f"   Total frames: {len(df_sync):,}")
    logger.info(f"   Time span: {sync_stats_obj.time_span_sec:.1f} seconds")

    # Save synchronized data
    output_path = data_path
    output_file = save_synchronized_data(
        df=df_sync,
        output_path=output_path,
        event_name=output_event_name,
        partitioned=False,
    )

    # Save statistics
    save_pivot_stats(pivot_stats, output_path, output_event_name)
    save_resample_stats(resample_stats, output_path, output_event_name)
    save_sync_stats(sync_stats_obj, coverage_by_car, output_path, output_event_name)

    # Stage 7: Validation (optional)
    validation_result = None
    if not skip_validation:
        logger.info(f"\n{'#'*60}")
        logger.info("# STAGE 7: VALIDATION")
        logger.info(f"{'#'*60}\n")

        try:
            validation_result = validate_simulation_ready(
                df=df_sync,
                context_root=Path("great_expectations"),
                event_name=output_event_name,
                output_dir=Path("data/reports"),
            )
        except Exception as e:
            logger.warning(f"⚠️  Validation failed: {e}")
            logger.warning("   Continuing without validation (use --skip-validation to suppress this warning)")
    else:
        logger.info(f"\n{'#'*60}")
        logger.info("# STAGE 7: VALIDATION (SKIPPED)")
        logger.info(f"{'#'*60}\n")

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*60)
    logger.info(f"✅ Processed {len(dfs_resampled)} cars")
    logger.info(f"✅ Generated {len(df_sync):,} synchronized frames")
    if validation_result:
        logger.info(f"✅ Validation: {validation_result.pass_rate:.1f}% pass rate")
    logger.info(f"\nOutputs:")
    logger.info(f"  Synchronized data: {output_file}")
    if validation_result:
        logger.info(f"  Validation report: {validation_result.report_path}")
    logger.info(f"  Statistics:")
    logger.info(f"    - Pivot: {output_path / output_event_name / 'pivot_stats.parquet'}")
    logger.info(f"    - Resample: {output_path / output_event_name / 'resample_stats.parquet'}")
    logger.info(f"    - Sync: {output_path / output_event_name / 'sync_stats.parquet'}")
    logger.info("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run full telemetry processing pipeline")

    parser.add_argument(
        '--chassis',
        nargs='+',
        default=['010', '002'],
        help='Chassis IDs to process (default: 010 002)',
    )

    parser.add_argument(
        '--event',
        default='barber_r1',
        help='Event name (default: barber_r1)',
    )

    parser.add_argument(
        '--output',
        default='barber_r1_pipeline',
        help='Output event name (default: barber_r1_pipeline)',
    )

    parser.add_argument(
        '--data-path',
        type=Path,
        default=Path('data/processed'),
        help='Path to processed data directory (default: data/processed)',
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation stage (recommended until GX 1.x API issues are resolved)',
    )

    args = parser.parse_args()

    # Run pipeline
    run_pipeline(
        chassis_ids=args.chassis,
        event_name=args.event,
        data_path=args.data_path,
        output_event_name=args.output,
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    main()
