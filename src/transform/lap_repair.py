"""Deterministic lap boundary detection and repair."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.logging_utils import get_logger
from src.schemas.metadata import LapBoundary, LapBoundaryReason

logger = get_logger(__name__)

# Sentinel value used in raw data for invalid laps
LAP_SENTINEL = 32768


def detect_lapdist_resets(
    lapdist: pd.Series,
    reset_threshold: float = -100.0,
    min_lapdist: float = 50.0,
) -> pd.Series:
    """Detect lap boundaries from track distance resets.

    Args:
        lapdist: Laptrigger_lapdist_dls series (track distance in meters)
        reset_threshold: Negative change threshold for reset detection (m)
        min_lapdist: Minimum distance before reset is valid (m)

    Returns:
        Boolean series where True indicates a lap boundary
    """
    # Compute distance change
    lapdist_diff = lapdist.diff()

    # Reset detected when:
    # 1. Distance drops significantly (negative change < threshold)
    # 2. Previous distance was substantial (car completed lap)
    is_reset = (lapdist_diff < reset_threshold) & (lapdist.shift(1) > min_lapdist)

    return is_reset


def detect_lap_increments(lap: pd.Series) -> pd.Series:
    """Detect lap boundaries from lap number increments.

    Args:
        lap: Lap number series

    Returns:
        Boolean series where True indicates a lap boundary
    """
    # Lap incremented when lap[i] > lap[i-1]
    lap_diff = lap.diff()
    is_increment = lap_diff > 0

    return is_increment


def fix_lap_sentinels(lap: pd.Series, sentinel: int = LAP_SENTINEL) -> pd.Series:
    """Replace sentinel lap values with NaN.

    Args:
        lap: Lap number series
        sentinel: Sentinel value to replace (default 32768)

    Returns:
        Lap series with sentinels replaced
    """
    lap_fixed = lap.copy()
    lap_fixed[lap_fixed == sentinel] = np.nan
    return lap_fixed


def validate_lap_duration(
    boundaries: List[Tuple[int, datetime]],
    min_duration_sec: float = 60.0,
    max_duration_sec: float = 300.0,
) -> List[Tuple[int, datetime, bool, str]]:
    """Validate lap durations and flag invalid laps.

    Args:
        boundaries: List of (lap_number, boundary_time) tuples
        min_duration_sec: Minimum valid lap duration
        max_duration_sec: Maximum valid lap duration

    Returns:
        List of (lap_number, boundary_time, is_valid, reason) tuples
    """
    validated = []

    for i, (lap_num, boundary_time) in enumerate(boundaries):
        if i == 0:
            # First boundary - no previous lap to validate
            validated.append((lap_num, boundary_time, True, "first_lap"))
            continue

        # Compute duration since previous boundary
        prev_lap_num, prev_boundary_time = boundaries[i - 1]
        duration_sec = (boundary_time - prev_boundary_time).total_seconds()

        # Validate duration
        if duration_sec < min_duration_sec:
            reason = f"too_short ({duration_sec:.1f}s < {min_duration_sec:.1f}s)"
            is_valid = False
        elif duration_sec > max_duration_sec:
            reason = f"too_long ({duration_sec:.1f}s > {max_duration_sec:.1f}s)"
            is_valid = False
        else:
            reason = f"valid ({duration_sec:.1f}s)"
            is_valid = True

        validated.append((lap_num, boundary_time, is_valid, reason))

    return validated


def detect_lap_boundaries(
    df: pd.DataFrame,
    chassis_id: str,
    event: str,
    track_length_m: float = 3700.0,
    min_lap_duration_sec: float = 90.0,
    max_lap_duration_sec: float = 300.0,
    reset_threshold: float = -100.0,
) -> List[LapBoundary]:
    """Detect lap boundaries using multiple signals with deterministic logic.

    Combines:
    1. Laptrigger_lapdist_dls resets (primary)
    2. Lap number increments (secondary)
    3. Duration validation (filter)

    Args:
        df: DataFrame with time_corrected, lap, Laptrigger_lapdist_dls columns
        chassis_id: Vehicle identifier
        event: Event name
        track_length_m: Track length in meters
        min_lap_duration_sec: Minimum valid lap duration
        max_lap_duration_sec: Maximum valid lap duration
        reset_threshold: Distance drop threshold for reset detection

    Returns:
        List of LapBoundary objects with audit trail
    """
    logger.info(f"Detecting lap boundaries for {chassis_id}")

    # Ensure sorted by time
    df = df.sort_values("time_corrected").reset_index(drop=True)

    # Fix lap sentinels
    df["lap_fixed"] = fix_lap_sentinels(df["lap"])

    # Detect boundaries from multiple sources
    boundaries_lapdist = pd.Series(False, index=df.index)
    boundaries_lapnum = pd.Series(False, index=df.index)

    # 1. Track distance resets
    if "Laptrigger_lapdist_dls" in df.columns:
        boundaries_lapdist = detect_lapdist_resets(
            df["Laptrigger_lapdist_dls"],
            reset_threshold=reset_threshold,
            min_lapdist=track_length_m * 0.8,  # Require 80% of track completed
        )

    # 2. Lap number increments
    if "lap_fixed" in df.columns:
        boundaries_lapnum = detect_lap_increments(df["lap_fixed"])

    # Combine: Boundary detected if EITHER lapdist reset OR lap increment
    # Priority: lapdist > lapnum (lapdist is more reliable)
    boundaries_combined = boundaries_lapdist | boundaries_lapnum

    # Extract boundary indices
    boundary_indices = df[boundaries_combined].index.tolist()

    logger.info(f"  Found {len(boundary_indices)} candidate boundaries")
    logger.info(f"    - From lapdist resets: {boundaries_lapdist.sum()}")
    logger.info(f"    - From lap increments: {boundaries_lapnum.sum()}")

    if len(boundary_indices) == 0:
        logger.warning(f"  No lap boundaries detected for {chassis_id}")
        return []

    # Build boundary list with timestamps
    boundary_list = []
    for idx in boundary_indices:
        boundary_time = df.loc[idx, "time_corrected"]
        lap_num_after = df.loc[idx, "lap_fixed"]

        # Lap number before boundary
        if idx > 0:
            lap_num_before = df.loc[idx - 1, "lap_fixed"]
        else:
            lap_num_before = None

        boundary_list.append((lap_num_after, boundary_time, lap_num_before, idx))

    # Validate lap durations
    validated_boundaries = []
    for i, (lap_after, boundary_time, lap_before, idx) in enumerate(boundary_list):
        if i == 0:
            # First boundary
            reason = LapBoundaryReason.LAP_INCREMENT
            confidence = 0.9
            is_valid = True
        else:
            # Compute duration since previous boundary
            prev_lap, prev_time, _, _ = boundary_list[i - 1]
            duration_sec = (boundary_time - prev_time).total_seconds()

            # Determine reason code
            if boundaries_lapdist.iloc[idx]:
                reason = LapBoundaryReason.LAPDIST_RESET
                confidence = 1.0
            elif boundaries_lapnum.iloc[idx]:
                reason = LapBoundaryReason.LAP_INCREMENT
                confidence = 0.8
            else:
                reason = LapBoundaryReason.UNKNOWN
                confidence = 0.5

            # Validate duration
            if duration_sec < min_lap_duration_sec:
                logger.warning(
                    f"  Lap {lap_after}: Duration too short ({duration_sec:.1f}s < {min_lap_duration_sec:.1f}s)"
                )
                confidence *= 0.5  # Reduce confidence for short laps
                is_valid = False
            elif duration_sec > max_lap_duration_sec:
                logger.warning(
                    f"  Lap {lap_after}: Duration too long ({duration_sec:.1f}s > {max_lap_duration_sec:.1f}s)"
                )
                # Long laps may be valid (pit stops, yellow flags)
                confidence *= 0.7
                is_valid = True
            else:
                is_valid = True

        # Create LapBoundary object
        validated_boundaries.append(
            LapBoundary(
                event=event,
                chassis_id=chassis_id,
                boundary_time=boundary_time,
                pre_lap=int(lap_before) if pd.notna(lap_before) else None,
                post_lap=int(lap_after) if pd.notna(lap_after) else i + 1,
                reason=reason,
                confidence=confidence,
            )
        )

    logger.info(f"  Validated {len(validated_boundaries)} lap boundaries")

    return validated_boundaries


def assign_lap_numbers(
    df: pd.DataFrame,
    boundaries: List[LapBoundary],
    fill_method: str = "forward",
) -> pd.DataFrame:
    """Assign lap numbers based on detected boundaries.

    Args:
        df: DataFrame with time_corrected column
        boundaries: List of LapBoundary objects
        fill_method: 'forward' or 'nearest'

    Returns:
        DataFrame with lap_repaired column
    """
    logger.info("Assigning lap numbers from boundaries")

    df = df.copy()
    df["lap_repaired"] = np.nan

    if len(boundaries) == 0:
        logger.warning("No boundaries provided - lap_repaired will be NaN")
        return df

    # Sort boundaries by time
    boundaries_sorted = sorted(boundaries, key=lambda b: b.boundary_time)

    # Assign lap numbers
    for i, boundary in enumerate(boundaries_sorted):
        lap_num = boundary.post_lap

        # Find rows after this boundary and before next boundary
        if i < len(boundaries_sorted) - 1:
            next_boundary = boundaries_sorted[i + 1]
            mask = (df["time_corrected"] >= boundary.boundary_time) & (
                df["time_corrected"] < next_boundary.boundary_time
            )
        else:
            # Last boundary - assign to all remaining rows
            mask = df["time_corrected"] >= boundary.boundary_time

        df.loc[mask, "lap_repaired"] = lap_num

    # Handle rows before first boundary
    first_boundary = boundaries_sorted[0]
    mask_before_first = df["time_corrected"] < first_boundary.boundary_time
    if mask_before_first.sum() > 0:
        if fill_method == "forward":
            # Assign to lap 1
            df.loc[mask_before_first, "lap_repaired"] = 1
        else:
            # Leave as NaN
            pass

    num_assigned = df["lap_repaired"].notna().sum()
    logger.info(f"  Assigned lap numbers to {num_assigned:,} / {len(df):,} rows")

    return df


def repair_laps(
    df: pd.DataFrame,
    chassis_id: str,
    event: str,
    track_length_m: float = 3700.0,
    min_lap_duration_sec: float = 90.0,
    max_lap_duration_sec: float = 300.0,
) -> Tuple[pd.DataFrame, List[LapBoundary]]:
    """Complete lap repair pipeline.

    Args:
        df: DataFrame with time_corrected, lap, Laptrigger_lapdist_dls columns
        chassis_id: Vehicle identifier
        event: Event name
        track_length_m: Track length in meters
        min_lap_duration_sec: Minimum valid lap duration
        max_lap_duration_sec: Maximum valid lap duration

    Returns:
        (df_with_lap_repaired, boundaries) tuple
    """
    logger.info(f"={'='*60}")
    logger.info(f"Lap Repair Pipeline: {chassis_id}")
    logger.info(f"={'='*60}")

    # Step 1: Detect boundaries
    boundaries = detect_lap_boundaries(
        df,
        chassis_id=chassis_id,
        event=event,
        track_length_m=track_length_m,
        min_lap_duration_sec=min_lap_duration_sec,
        max_lap_duration_sec=max_lap_duration_sec,
    )

    # Step 2: Assign lap numbers
    df = assign_lap_numbers(df, boundaries)

    # Step 3: Validate results
    num_laps = df["lap_repaired"].nunique()
    num_rows_with_laps = df["lap_repaired"].notna().sum()

    logger.info(f"\n  Lap Repair Summary:")
    logger.info(f"    Total boundaries: {len(boundaries)}")
    logger.info(f"    Unique laps: {num_laps}")
    logger.info(f"    Rows with laps: {num_rows_with_laps:,} / {len(df):,}")

    if num_rows_with_laps < len(df):
        logger.warning(
            f"    {len(df) - num_rows_with_laps:,} rows without lap assignment"
        )

    logger.info(f"{'='*60}\n")

    return df, boundaries


def save_lap_boundaries(
    boundaries_by_car: Dict[str, List[LapBoundary]],
    output_path: Path,
    event_name: str,
) -> None:
    """Save lap boundaries to Parquet.

    Args:
        boundaries_by_car: Dict mapping chassis_id -> list of boundaries
        output_path: Output directory
        event_name: Event identifier
    """
    logger.info(f"Saving lap boundaries to {output_path}")

    all_boundaries = []
    for chassis_id, boundaries in boundaries_by_car.items():
        for boundary in boundaries:
            all_boundaries.append(
                {
                    "event": boundary.event,
                    "chassis_id": boundary.chassis_id,
                    "boundary_time": boundary.boundary_time,
                    "pre_lap": boundary.pre_lap,
                    "post_lap": boundary.post_lap,
                    "reason": boundary.reason.value,
                    "confidence": boundary.confidence,
                }
            )

    if len(all_boundaries) == 0:
        logger.warning("No lap boundaries to save")
        return

    df = pd.DataFrame(all_boundaries)

    # Save to Parquet
    output_file = output_path / event_name / "lap_boundaries.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, compression="snappy", index=False)

    logger.info(f"  Saved {len(all_boundaries)} boundary records")
    logger.info(f"  Output: {output_file}")
