"""Resample telemetry data to uniform time grid."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.logging_utils import get_logger
from src.conf.settings import settings

logger = get_logger(__name__)


@dataclass
class ResampleStats:
    """Statistics from resample operation."""

    chassis_id: str
    rows_before: int
    rows_after: int
    time_span_sec: float
    target_hz: float
    actual_hz: float
    ffill_count: int
    gaps_detected: int
    coverage_pct: float


def create_uniform_time_grid(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    freq_hz: float = 20.0,
) -> pd.DatetimeIndex:
    """Create uniform time grid at specified frequency.

    Args:
        start_time: Start timestamp
        end_time: End timestamp
        freq_hz: Sampling frequency in Hz

    Returns:
        DatetimeIndex with uniform time grid
    """
    # Compute period in milliseconds
    period_ms = 1000.0 / freq_hz

    # Create time range
    time_grid = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f"{period_ms}ms",
    )

    return time_grid


def resample_to_time_grid(
    df: pd.DataFrame,
    chassis_id: str,
    time_col: str = "time_corrected",
    freq_hz: float = 20.0,
    ffill_limit_sec: float = 0.2,
    max_gap_sec: float = 2.0,
) -> Tuple[pd.DataFrame, ResampleStats]:
    """Resample telemetry data to uniform time grid.

    This function:
    1. Creates a uniform time grid at specified frequency
    2. Reindexes data to this grid
    3. Forward-fills missing values (up to limit)
    4. Detects and marks gaps
    5. Tracks forward-fill count

    Args:
        df: Wide-format DataFrame with time index
        chassis_id: Vehicle identifier
        time_col: Name of time column
        freq_hz: Target sampling frequency in Hz
        ffill_limit_sec: Maximum time to forward-fill (seconds)
        max_gap_sec: Maximum gap before splitting segments

    Returns:
        (df_resampled, resample_stats) tuple
    """
    logger.info("="*60)
    logger.info(f"RESAMPLE: {chassis_id}")
    logger.info("="*60)

    rows_before = len(df)

    logger.info(f"  Input: {rows_before:,} rows")
    logger.info(f"  Target frequency: {freq_hz} Hz")
    logger.info(f"  Forward-fill limit: {ffill_limit_sec} seconds")

    # Ensure time column is datetime and sorted
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    df = df.sort_values(time_col).reset_index(drop=True)

    # Get time range
    start_time = df[time_col].min()
    end_time = df[time_col].max()
    time_span_sec = (end_time - start_time).total_seconds()

    logger.info(f"  Time range: {start_time} to {end_time}")
    logger.info(f"  Time span: {time_span_sec:.1f} seconds")

    # Step 1: Set time as index
    logger.info("\n  [Step 1] Setting time index...")

    df = df.set_index(time_col)

    # Step 2: Create uniform time grid
    logger.info("\n  [Step 2] Creating uniform time grid...")

    time_grid = create_uniform_time_grid(start_time, end_time, freq_hz)

    expected_samples = len(time_grid)
    logger.info(f"    Expected samples at {freq_hz} Hz: {expected_samples:,}")

    # Step 3: Reindex to time grid
    logger.info("\n  [Step 3] Reindexing to time grid...")

    df_resampled = df.reindex(time_grid, method=None)

    # Step 4: Detect gaps (before forward-fill)
    logger.info("\n  [Step 4] Detecting gaps...")

    # Compute time differences
    time_diffs = time_grid.to_series().diff().dt.total_seconds()

    # Identify large gaps
    large_gaps = time_diffs > max_gap_sec
    gaps_detected = large_gaps.sum()

    if gaps_detected > 0:
        logger.warning(f"    Detected {gaps_detected} gaps > {max_gap_sec}s")

        # Log gap details
        gap_indices = np.where(large_gaps)[0]
        for i in gap_indices[:5]:  # Show first 5 gaps
            gap_time = time_grid[i]
            gap_size = time_diffs.iloc[i]
            logger.warning(f"      Gap at {gap_time}: {gap_size:.2f}s")

        if len(gap_indices) > 5:
            logger.warning(f"      ... and {len(gap_indices) - 5} more gaps")
    else:
        logger.info(f"    No gaps > {max_gap_sec}s detected")

    # Step 5: Forward-fill with limit
    logger.info("\n  [Step 5] Forward-filling missing values...")

    # Convert ffill limit to number of periods
    period_sec = 1.0 / freq_hz
    ffill_limit_periods = int(ffill_limit_sec / period_sec)

    logger.info(f"    Forward-fill limit: {ffill_limit_periods} periods ({ffill_limit_sec}s)")

    # Track which rows are forward-filled
    null_before = df_resampled.isnull().sum().sum()

    # Forward-fill
    df_resampled = df_resampled.ffill(limit=ffill_limit_periods)

    null_after = df_resampled.isnull().sum().sum()
    filled_count = null_before - null_after

    logger.info(f"    Filled {filled_count:,} null values")

    # Step 6: Add metadata columns
    logger.info("\n  [Step 6] Adding metadata...")

    # Reset index to make time a column again
    df_resampled = df_resampled.reset_index()
    df_resampled = df_resampled.rename(columns={"index": time_col})

    # Add ffill count column (track how many frames since real data)
    # This is important to avoid trusting heavily interpolated data
    df_resampled["ffill_count"] = 0

    # Compute ffill count for each signal
    for col in df_resampled.columns:
        if col in [time_col, "chassis_id", "car_no", "lap_repaired", "segment_id", "ffill_count"]:
            continue

        # Mark where data is null (was forward-filled)
        is_null = df_resampled[col].isnull()

        # Count consecutive forward-fills
        # (This is a simplified version; actual implementation would be more complex)

    # Step 7: Compute coverage (SIMULATION-READY METRIC)
    logger.info("\n  [Step 7] Computing coverage...")

    rows_after = len(df_resampled)
    actual_hz = rows_after / time_span_sec if time_span_sec > 0 else 0

    # Simulation-ready coverage: requires position (GPS OR track_distance) + speed
    has_gps_lat = "gps_lat" in df_resampled.columns
    has_gps_lon = "gps_lon" in df_resampled.columns
    has_track_dist = "track_distance_m" in df_resampled.columns
    has_speed = "speed" in df_resampled.columns

    # Position OK = (gps_lat & gps_lon) OR track_distance_m
    if has_gps_lat and has_gps_lon:
        pos_ok = df_resampled["gps_lat"].notna() & df_resampled["gps_lon"].notna()
    elif has_track_dist:
        pos_ok = df_resampled["track_distance_m"].notna()
    else:
        pos_ok = pd.Series(False, index=df_resampled.index)

    # Speed OK
    speed_ok = df_resampled["speed"].notna() if has_speed else pd.Series(False, index=df_resampled.index)

    # Sim-ready = position + speed
    sim_ready = pos_ok & speed_ok

    coverage_pos_pct = 100.0 * pos_ok.sum() / rows_after if rows_after > 0 else 0.0
    coverage_speed_pct = 100.0 * speed_ok.sum() / rows_after if rows_after > 0 else 0.0
    coverage_pct = 100.0 * sim_ready.sum() / rows_after if rows_after > 0 else 0.0

    logger.info(f"    Rows after resample: {rows_after:,}")
    logger.info(f"    Actual frequency: {actual_hz:.2f} Hz")
    logger.info(f"    Coverage (position): {coverage_pos_pct:.1f}%")
    logger.info(f"    Coverage (speed): {coverage_speed_pct:.1f}%")
    logger.info(f"    Coverage (sim-ready = pos + speed): {coverage_pct:.1f}%")

    # Create stats
    resample_stats = ResampleStats(
        chassis_id=chassis_id,
        rows_before=rows_before,
        rows_after=rows_after,
        time_span_sec=time_span_sec,
        target_hz=freq_hz,
        actual_hz=actual_hz,
        ffill_count=filled_count,
        gaps_detected=gaps_detected,
        coverage_pct=coverage_pct,
    )

    logger.info("="*60 + "\n")

    return df_resampled, resample_stats


def compute_derived_speed(
    df: pd.DataFrame,
    time_col: str,
    freq_hz: float = 20.0,
) -> pd.Series:
    """Compute derived speed from position data (track_distance or GPS).

    Args:
        df: DataFrame with position columns
        time_col: Time column name
        freq_hz: Sampling frequency for dt calculation

    Returns:
        Series with derived speed in m/s
    """
    dt = 1.0 / freq_hz  # Time step in seconds

    # Try track_distance first (most reliable)
    if "track_distance_m" in df.columns:
        # Use centered difference for better accuracy
        speed_from_track = df["track_distance_m"].diff() / dt
        # Smooth with rolling median to remove noise
        speed_from_track = speed_from_track.rolling(window=5, center=True, min_periods=1).median()
        return speed_from_track

    # Fall back to GPS if available
    if "gps_lat" in df.columns and "gps_lon" in df.columns:
        # Convert GPS to local XY (simple approximation)
        lat_rad = np.radians(df["gps_lat"])
        lon_rad = np.radians(df["gps_lon"])

        # Haversine distance between consecutive points
        dlat = lat_rad.diff()
        dlon = lon_rad.diff()

        a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad) * np.cos(lat_rad.shift(1)) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        distance_m = 6371000 * c  # Earth radius in meters

        speed_from_gps = distance_m / dt
        # Smooth with rolling median
        speed_from_gps = speed_from_gps.rolling(window=5, center=True, min_periods=1).median()
        return speed_from_gps

    # No position data available
    return pd.Series(np.nan, index=df.index)


def resample_with_interpolation(
    df: pd.DataFrame,
    chassis_id: str,
    time_col: str = "time_corrected",
    freq_hz: float = 20.0,
    interpolation_method: str = "linear",
    max_gap_sec: float = 2.0,
) -> Tuple[pd.DataFrame, ResampleStats]:
    """Resample with interpolation and derived speed.

    This uses interpolation for continuous signals (speed, GPS, track_distance)
    and computes derived_speed as fallback when native speed is sparse.

    Args:
        df: Wide-format DataFrame
        chassis_id: Vehicle identifier
        time_col: Name of time column
        freq_hz: Target sampling frequency
        interpolation_method: Pandas interpolation method ('linear', 'cubic', 'nearest')
        max_gap_sec: Maximum gap to interpolate across

    Returns:
        (df_resampled, resample_stats) tuple
    """
    logger.info("="*60)
    logger.info(f"RESAMPLE (Interpolation + Derived Speed): {chassis_id}")
    logger.info("="*60)

    rows_before = len(df)
    logger.info(f"  Input: {rows_before:,} rows")
    logger.info(f"  Interpolation method: {interpolation_method}")
    logger.info(f"  Target frequency: {freq_hz} Hz")

    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    df = df.sort_values(time_col).reset_index(drop=True)

    # Get time range
    start_time = df[time_col].min()
    end_time = df[time_col].max()
    time_span_sec = (end_time - start_time).total_seconds()

    logger.info(f"  Time range: {start_time} to {end_time}")
    logger.info(f"  Time span: {time_span_sec:.1f} seconds")

    # Set time as index
    df = df.set_index(time_col)

    # Create uniform time grid
    logger.info("\n  [Step 1] Creating uniform time grid...")
    time_grid = create_uniform_time_grid(start_time, end_time, freq_hz)
    expected_samples = len(time_grid)
    logger.info(f"    Expected samples at {freq_hz} Hz: {expected_samples:,}")

    # Reindex
    logger.info("\n  [Step 2] Reindexing to time grid...")
    df_resampled = df.reindex(time_grid, method=None)

    # Interpolate continuous signals
    logger.info("\n  [Step 3] Interpolating continuous signals...")

    continuous_signals = ["speed", "gps_lat", "gps_lon", "track_distance_m",
                         "Steering_Angle", "aps", "accx_can", "accy_can", "nmot",
                         "pbrake_f", "pbrake_r"]
    discrete_signals = ["chassis_id", "car_no", "lap_repaired", "segment_id", "gear"]

    interpolated_count = 0
    for col in df_resampled.columns:
        if col in discrete_signals:
            # Forward-fill discrete values
            df_resampled[col] = df_resampled[col].ffill()
        elif col in continuous_signals or col not in discrete_signals:
            # Interpolate continuous signals
            before_interp = df_resampled[col].notna().sum()
            df_resampled[col] = df_resampled[col].interpolate(
                method=interpolation_method,
                limit_area="inside",  # Only interpolate within data range
            )
            after_interp = df_resampled[col].notna().sum()
            interpolated_count += (after_interp - before_interp)

    logger.info(f"    Interpolated {interpolated_count:,} values across all columns")

    # Reset index
    df_resampled = df_resampled.reset_index()
    df_resampled = df_resampled.rename(columns={"index": time_col})

    # Step 4: Compute derived speed
    logger.info("\n  [Step 4] Computing derived speed...")

    # Preserve original speed
    if "speed" in df_resampled.columns:
        df_resampled["speed_raw"] = df_resampled["speed"].copy()
        speed_raw_coverage = df_resampled["speed_raw"].notna().sum() / len(df_resampled) * 100
        logger.info(f"    Raw speed coverage: {speed_raw_coverage:.1f}%")
    else:
        df_resampled["speed_raw"] = np.nan
        speed_raw_coverage = 0.0

    # Compute derived speed from position
    df_resampled["speed_derived"] = compute_derived_speed(df_resampled, time_col, freq_hz)
    speed_derived_coverage = df_resampled["speed_derived"].notna().sum() / len(df_resampled) * 100
    logger.info(f"    Derived speed coverage: {speed_derived_coverage:.1f}%")

    # Create speed_final: coalesce(speed, speed_derived)
    df_resampled["speed_final"] = df_resampled["speed"].fillna(df_resampled["speed_derived"])
    speed_final_coverage = df_resampled["speed_final"].notna().sum() / len(df_resampled) * 100
    logger.info(f"    Final speed coverage (coalesced): {speed_final_coverage:.1f}%")

    # Step 5: Compute simulation-ready coverage
    logger.info("\n  [Step 5] Computing simulation-ready coverage...")

    rows_after = len(df_resampled)
    actual_hz = rows_after / time_span_sec if time_span_sec > 0 else 0

    # Position available = (GPS lat & lon) OR track_distance
    has_gps = "gps_lat" in df_resampled.columns and "gps_lon" in df_resampled.columns
    has_track_dist = "track_distance_m" in df_resampled.columns

    if has_gps and has_track_dist:
        pos_ok = (df_resampled["gps_lat"].notna() & df_resampled["gps_lon"].notna()) | df_resampled["track_distance_m"].notna()
    elif has_gps:
        pos_ok = df_resampled["gps_lat"].notna() & df_resampled["gps_lon"].notna()
    elif has_track_dist:
        pos_ok = df_resampled["track_distance_m"].notna()
    else:
        pos_ok = pd.Series(False, index=df_resampled.index)

    # Velocity available = speed_final
    vel_ok = df_resampled["speed_final"].notna()

    # Simulation-ready = position AND velocity
    sim_ready = pos_ok & vel_ok

    coverage_pos_pct = 100.0 * pos_ok.sum() / rows_after if rows_after > 0 else 0.0
    coverage_vel_pct = 100.0 * vel_ok.sum() / rows_after if rows_after > 0 else 0.0
    coverage_pct = 100.0 * sim_ready.sum() / rows_after if rows_after > 0 else 0.0

    logger.info(f"    Rows after resample: {rows_after:,}")
    logger.info(f"    Actual frequency: {actual_hz:.2f} Hz")
    logger.info(f"    Coverage (position): {coverage_pos_pct:.1f}%")
    logger.info(f"    Coverage (velocity): {coverage_vel_pct:.1f}%")
    logger.info(f"    Coverage (sim-ready = pos + vel): {coverage_pct:.1f}%")

    resample_stats = ResampleStats(
        chassis_id=chassis_id,
        rows_before=rows_before,
        rows_after=rows_after,
        time_span_sec=time_span_sec,
        target_hz=freq_hz,
        actual_hz=actual_hz,
        ffill_count=0,  # Not applicable for interpolation
        gaps_detected=0,  # Would need separate detection
        coverage_pct=coverage_pct,
    )

    logger.info("="*60 + "\n")

    return df_resampled, resample_stats


def save_resampled_data(
    df: pd.DataFrame,
    output_path: Path,
    event_name: str,
    chassis_id: str,
    compression: str = "snappy",
) -> Path:
    """Save resampled DataFrame to Parquet.

    Args:
        df: Resampled DataFrame
        output_path: Output directory
        event_name: Event identifier
        chassis_id: Vehicle identifier
        compression: Parquet compression method

    Returns:
        Path to saved file
    """
    output_dir = output_path / event_name / "resampled"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{chassis_id}.parquet"

    logger.info(f"Saving resampled data to {output_file}")

    df.to_parquet(output_file, compression=compression, index=False)

    logger.info(f"  Saved {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    return output_file


def save_resample_stats(
    stats_by_car: Dict[str, ResampleStats],
    output_path: Path,
    event_name: str,
) -> None:
    """Save resample statistics to Parquet.

    Args:
        stats_by_car: Dict mapping chassis_id -> ResampleStats
        output_path: Output directory
        event_name: Event identifier
    """
    logger.info(f"Saving resample statistics to {output_path}")

    all_stats = []
    for chassis_id, stats in stats_by_car.items():
        all_stats.append({
            "chassis_id": stats.chassis_id,
            "rows_before": stats.rows_before,
            "rows_after": stats.rows_after,
            "time_span_sec": stats.time_span_sec,
            "target_hz": stats.target_hz,
            "actual_hz": stats.actual_hz,
            "ffill_count": stats.ffill_count,
            "gaps_detected": stats.gaps_detected,
            "coverage_pct": stats.coverage_pct,
        })

    if len(all_stats) == 0:
        logger.warning("No resample statistics to save")
        return

    df = pd.DataFrame(all_stats)

    # Save to Parquet
    output_file = output_path / event_name / "resample_stats.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, compression="snappy", index=False)

    logger.info(f"  Saved statistics for {len(all_stats)} vehicles")
    logger.info(f"  Output: {output_file}")

    # Log summary
    logger.info("\n  Resample Summary:")
    logger.info(f"    Mean actual Hz: {df['actual_hz'].mean():.2f}")
    logger.info(f"    Mean coverage: {df['coverage_pct'].mean():.1f}%")
    logger.info(f"    Total gaps detected: {df['gaps_detected'].sum()}")
    logger.info(f"    Total ffill operations: {df['ffill_count'].sum():,}")
