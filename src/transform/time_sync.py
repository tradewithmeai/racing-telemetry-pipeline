"""Time synchronization and global alignment for multi-car telemetry."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

from src.utils.logging_utils import get_logger
from src.utils.time_utils import detect_backwards_time, detect_time_gaps, segment_by_gaps
from src.schemas.metadata import DriftCalibration as DriftCalibrationSchema
from .drift import windowed_drift_calibration, DriftCalibration, apply_segmented_drift_correction

logger = get_logger(__name__)


def compute_per_car_drift(
    df: pd.DataFrame,
    chassis_id: str,
    window_minutes: int = 5,
    method: str = "median",
) -> List[DriftCalibration]:
    """Compute drift calibration for a single car.

    Args:
        df: DataFrame with timestamp and meta_time columns for one car
        chassis_id: Vehicle identifier
        window_minutes: Window size for drift estimation
        method: 'median' (robust) or 'mean'

    Returns:
        List of DriftCalibration objects
    """
    logger.info(f"Computing drift calibration for {chassis_id}")

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute windowed drift
    calibrations = windowed_drift_calibration(
        df, chassis_id=chassis_id, window_minutes=window_minutes, method=method
    )

    return calibrations


def detect_backwards_segments(
    df: pd.DataFrame,
    chassis_id: str,
    tolerance_sec: float = 0.001,
) -> pd.DataFrame:
    """Detect backwards timestamps and assign segment IDs.

    Args:
        df: DataFrame with timestamp column for one car (must be sorted by index)
        chassis_id: Vehicle identifier
        tolerance_sec: Tolerance for clock jitter

    Returns:
        DataFrame with 'segment_id' and 'is_backwards' columns added
    """
    logger.info(f"Detecting backwards timestamps for {chassis_id}")

    df = df.copy()

    # Detect backwards points
    is_backwards = detect_backwards_time(df["timestamp"], tolerance_sec=tolerance_sec)

    # Segment at backwards points
    # Each backwards event starts a new segment
    segment_id = is_backwards.cumsum()

    df["segment_id"] = segment_id
    df["is_backwards"] = is_backwards

    # Log statistics
    num_backwards = is_backwards.sum()
    num_segments = segment_id.max() + 1

    logger.info(f"  {chassis_id}: {num_backwards} backwards events")
    logger.info(f"  {chassis_id}: {num_segments} time segments")

    if num_backwards > 0:
        logger.warning(
            f"  {chassis_id}: Data segmented into {num_segments} parts due to backwards timestamps"
        )

    return df


def apply_drift_correction(
    df: pd.DataFrame,
    calibrations: List[DriftCalibration],
) -> pd.DataFrame:
    """Apply drift corrections to create time_corrected column.

    Args:
        df: DataFrame with timestamp column
        calibrations: List of calibrations per segment

    Returns:
        DataFrame with time_corrected column
    """
    df = apply_segmented_drift_correction(df, calibrations)
    return df


def align_to_global_time(
    dfs_by_car: Dict[str, pd.DataFrame],
    reference_car: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], datetime]:
    """Align all cars to a global time reference.

    Args:
        dfs_by_car: Dict mapping chassis_id -> DataFrame (with time_corrected)
        reference_car: Optional chassis_id to use as time reference
                       (if None, uses earliest time_corrected across all cars)

    Returns:
        (aligned_dfs_by_car, global_session_start) tuple
    """
    logger.info("Aligning cars to global time reference")

    if len(dfs_by_car) == 0:
        logger.warning("No data to align")
        return {}, datetime.now()

    # Find global session start
    if reference_car and reference_car in dfs_by_car:
        session_start = dfs_by_car[reference_car]["time_corrected"].min()
        logger.info(f"Using {reference_car} as time reference")
    else:
        # Use earliest time across all cars
        all_starts = [df["time_corrected"].min() for df in dfs_by_car.values()]
        session_start = min(all_starts)
        logger.info(f"Using earliest time as reference: {session_start}")

    # Add time_global column to each car
    aligned_dfs = {}
    for chassis_id, df in dfs_by_car.items():
        df = df.copy()
        df["time_global"] = df["time_corrected"]  # Already corrected, just rename
        df["session_start"] = session_start
        aligned_dfs[chassis_id] = df

        logger.info(
            f"  {chassis_id}: {len(df):,} rows, "
            f"time range {df['time_global'].min()} to {df['time_global'].max()}"
        )

    return aligned_dfs, session_start


def segment_by_gaps_and_backwards(
    df: pd.DataFrame,
    chassis_id: str,
    gap_threshold_sec: float = 2.0,
    backwards_tolerance_sec: float = 0.001,
) -> pd.DataFrame:
    """Segment data by both time gaps and backwards timestamps.

    Args:
        df: DataFrame with timestamp column
        chassis_id: Vehicle identifier
        gap_threshold_sec: Threshold for time gaps
        backwards_tolerance_sec: Tolerance for backwards detection

    Returns:
        DataFrame with segment_id column
    """
    logger.info(f"Segmenting {chassis_id} by gaps and backwards timestamps")

    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Detect backwards points
    is_backwards = detect_backwards_time(df["timestamp"], tolerance_sec=backwards_tolerance_sec)

    # Detect gaps
    is_gap = detect_time_gaps(df["timestamp"], gap_threshold_sec=gap_threshold_sec)

    # Segment at either backwards or gap
    is_segment_boundary = is_backwards | is_gap
    segment_id = is_segment_boundary.cumsum()

    df["segment_id"] = segment_id
    df["is_backwards"] = is_backwards
    df["is_gap"] = is_gap

    num_backwards = is_backwards.sum()
    num_gaps = is_gap.sum()
    num_segments = segment_id.max() + 1

    logger.info(f"  {chassis_id}: {num_backwards} backwards events, {num_gaps} gaps")
    logger.info(f"  {chassis_id}: {num_segments} segments created")

    return df


def process_car_time_sync(
    df: pd.DataFrame,
    chassis_id: str,
    window_minutes: int = 5,
    gap_threshold_sec: float = 2.0,
    method: str = "median",
) -> Tuple[pd.DataFrame, List[DriftCalibration]]:
    """Complete time synchronization pipeline for one car.

    Args:
        df: DataFrame with timestamp and meta_time columns
        chassis_id: Vehicle identifier
        window_minutes: Window size for drift estimation
        gap_threshold_sec: Threshold for time gaps
        method: Drift estimation method

    Returns:
        (processed_df, calibrations) tuple
    """
    logger.info(f"={'='*60}")
    logger.info(f"Time Sync Pipeline: {chassis_id}")
    logger.info(f"={'='*60}")

    # Step 1: Segment by backwards and gaps
    df = segment_by_gaps_and_backwards(
        df, chassis_id=chassis_id, gap_threshold_sec=gap_threshold_sec
    )

    # Step 2: Compute drift calibrations
    calibrations = compute_per_car_drift(
        df, chassis_id=chassis_id, window_minutes=window_minutes, method=method
    )

    # Step 3: Apply drift corrections
    if len(calibrations) > 0:
        df = apply_drift_correction(df, calibrations)
        logger.info(f"  Applied drift corrections: time_corrected column added")
    else:
        logger.warning(f"  No calibrations available, using original timestamps")
        df["time_corrected"] = df["timestamp"]

    # Step 4: Validate monotonicity
    is_still_backwards = detect_backwards_time(df["time_corrected"], tolerance_sec=0.001)
    remaining_backwards = is_still_backwards.sum()

    if remaining_backwards == 0:
        logger.info(f"  [SUCCESS] Time correction successful: 0 backwards timestamps remaining")
    else:
        logger.warning(
            f"  [WARNING] {remaining_backwards} backwards timestamps remain after correction"
        )

    logger.info(f"{'='*60}")

    return df, calibrations


def save_drift_calibrations(
    calibrations_by_car: Dict[str, List[DriftCalibration]],
    output_path: Path,
    event_name: str,
) -> None:
    """Save drift calibrations to Parquet.

    Args:
        calibrations_by_car: Dict mapping chassis_id -> list of calibrations
        output_path: Output directory
        event_name: Event identifier
    """
    logger.info(f"Saving drift calibrations to {output_path}")

    all_calibrations = []
    for chassis_id, calibrations in calibrations_by_car.items():
        for calib in calibrations:
            all_calibrations.append(
                {
                    "event": event_name,
                    "chassis_id": calib.chassis_id,
                    "segment_id": calib.segment_id,
                    "window_start": calib.window_start,
                    "window_end": calib.window_end,
                    "drift_sec": calib.drift_sec,
                    "drift_std": calib.drift_std,
                    "step_detected": calib.step_detected,
                    "samples": calib.samples,
                    "method": calib.method,
                    "quality": calib.quality_level,
                    "is_valid": calib.is_valid,
                }
            )

    df = pd.DataFrame(all_calibrations)

    # Save to Parquet
    output_file = output_path / event_name / "drift_calibration.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, compression="snappy", index=False)

    logger.info(f"  Saved {len(all_calibrations)} calibration records")
    logger.info(f"  Output: {output_file}")
