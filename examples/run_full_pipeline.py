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
    resample_with_interpolation,
    save_resample_stats,
    # Sync
    synchronize_multi_car,
    save_synchronized_data,
    save_sync_stats,
)
from src.validation.validators import validate_simulation_ready
from src.validation.baseline_validator import validate_against_baseline, PipelineValidationError
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
    strict_baseline: bool = True,
):
    """Run complete end-to-end pipeline for multiple cars.

    Args:
        chassis_ids: List of chassis IDs to process
        event_name: Input event name
        data_path: Path to processed data directory
        output_event_name: Output event name for results
        skip_validation: Skip GX validation stage (API compatibility issues)
        strict_baseline: Fail hard on baseline validation failures (default: True)
    """
    logger.info("\n" + "="*60)
    logger.info("FULL PIPELINE: END-TO-END PROCESSING")
    logger.info("="*60)
    logger.info(f"  Event: {event_name}")
    logger.info(f"  Cars: {chassis_ids}")
    logger.info(f"  Output: {output_event_name}")
    logger.info(f"  Baseline validation: {'STRICT' if strict_baseline else 'WARN-ONLY'}")
    logger.info("="*60 + "\n")

    # Load baseline once for all cars
    baseline_path = Path("data/baselines") / f"{event_name}_baseline.json"
    if not baseline_path.exists():
        logger.warning(f"Baseline not found: {baseline_path}")
        logger.warning("Run: python tools/compute_baseline.py to generate baseline")
        logger.warning("Continuing without baseline validation...")
        baseline_path = None

    # Track statistics
    pivot_stats = {}
    resample_stats = {}
    dfs_resampled = {}
    validation_results = {}

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

            # Stage 3: Pivot (long → wide format) - MUST RUN BEFORE POSITION NORMALIZATION
            logger.info(f"\n[STAGE 3: PIVOT] {chassis_id}")
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

            # Baseline validation: Post-pivot
            if baseline_path:
                try:
                    validate_against_baseline(
                        df=df_wide,
                        baseline_path=baseline_path,
                        chassis_id=chassis_id,
                        stage="pivot",
                        strict=strict_baseline,
                        time_col='time_corrected',
                    )
                except PipelineValidationError as e:
                    logger.error(f"Baseline validation failed: {e}")
                    if strict_baseline:
                        raise

            # Stage 4: Position normalization (now on wide format with GPS columns)
            df_wide = apply_position_normalization(df_wide, chassis_id)

            # Stage 5: Resample to 20Hz with interpolation and derived speed
            logger.info(f"\n[STAGE 5: RESAMPLE] {chassis_id}")
            logger.info("="*60)

            df_resampled, resample_stat = resample_with_interpolation(
                df=df_wide,
                chassis_id=chassis_id,
                time_col='time_corrected',
                freq_hz=20.0,
                interpolation_method='linear',
                max_gap_sec=2.0,
            )

            resample_stats[chassis_id] = resample_stat
            dfs_resampled[chassis_id] = df_resampled

            # Baseline validation: Post-resample
            if baseline_path:
                try:
                    result = validate_against_baseline(
                        df=df_resampled,
                        baseline_path=baseline_path,
                        chassis_id=chassis_id,
                        stage="resample",
                        strict=strict_baseline,
                        time_col='time_corrected',
                    )
                    validation_results[chassis_id] = result
                except PipelineValidationError as e:
                    logger.error(f"Baseline validation failed: {e}")
                    if strict_baseline:
                        raise

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

    # Save baseline validation results
    if validation_results:
        import json
        validation_report_path = output_path / output_event_name / "baseline_validation.json"
        validation_report_path.parent.mkdir(parents=True, exist_ok=True)

        validation_report = {
            "event_name": output_event_name,
            "generated_at": datetime.now().isoformat(),
            "overall_status": "PASS" if all(v.passed for v in validation_results.values()) else "FAIL",
            "cars": {cid: v.to_dict() for cid, v in validation_results.items()},
        }

        with open(validation_report_path, "w") as f:
            json.dump(validation_report, f, indent=2)

        logger.info(f"\n  Baseline validation report: {validation_report_path}")

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
        help='Skip GX validation stage (recommended until GX 1.x API issues are resolved)',
    )

    parser.add_argument(
        '--no-strict-baseline',
        action='store_true',
        help='Disable strict baseline validation (warn only, do not fail)',
    )

    args = parser.parse_args()

    # Run pipeline
    run_pipeline(
        chassis_ids=args.chassis,
        event_name=args.event,
        data_path=args.data_path,
        output_event_name=args.output,
        skip_validation=args.skip_validation,
        strict_baseline=not args.no_strict_baseline,
    )


if __name__ == "__main__":
    main()
