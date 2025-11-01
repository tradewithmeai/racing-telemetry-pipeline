"""Multi-car global time synchronization for race replay."""

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
class SyncStats:
    """Statistics from multi-car synchronization."""

    event_name: str
    num_cars: int
    global_start_time: pd.Timestamp
    global_end_time: pd.Timestamp
    time_span_sec: float
    target_hz: float
    total_frames: int
    cars_per_frame_mean: float
    cars_per_frame_min: int
    cars_per_frame_max: int
    coverage_pct: float


@dataclass
class CarCoverage:
    """Coverage statistics for a single car."""

    chassis_id: str
    frames_present: int
    frames_total: int
    coverage_pct: float
    first_frame_time: pd.Timestamp
    last_frame_time: pd.Timestamp


def determine_global_time_range(
    dfs_by_car: Dict[str, pd.DataFrame],
    time_col: str = "time_corrected",
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Determine global time range across all cars.

    Args:
        dfs_by_car: Dict mapping chassis_id -> DataFrame
        time_col: Name of time column

    Returns:
        (global_start, global_end) tuple
    """
    logger.info("Determining global time range...")

    all_starts = []
    all_ends = []

    for chassis_id, df in dfs_by_car.items():
        if time_col not in df.columns:
            logger.warning(f"  {chassis_id}: Missing {time_col} column, skipping")
            continue

        start = df[time_col].min()
        end = df[time_col].max()

        all_starts.append(start)
        all_ends.append(end)

        logger.info(f"  {chassis_id}: {start} to {end}")

    if not all_starts:
        raise ValueError("No valid time ranges found in any car data")

    global_start = min(all_starts)
    global_end = max(all_ends)

    logger.info(f"\n  Global time range:")
    logger.info(f"    Start: {global_start}")
    logger.info(f"    End: {global_end}")
    logger.info(f"    Span: {(global_end - global_start).total_seconds():.1f} seconds")

    return global_start, global_end


def create_global_time_grid(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    freq_hz: float = 20.0,
) -> pd.DataFrame:
    """Create global time grid for all cars.

    Args:
        start_time: Global start timestamp
        end_time: Global end timestamp
        freq_hz: Sampling frequency in Hz

    Returns:
        DataFrame with time_global column
    """
    logger.info(f"Creating global time grid at {freq_hz} Hz...")

    # Compute period in milliseconds
    period_ms = 1000.0 / freq_hz

    # Create time range
    time_grid = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f"{period_ms}ms",
    )

    df_grid = pd.DataFrame({"time_global": time_grid})

    logger.info(f"  Created {len(df_grid):,} time points")

    return df_grid


def align_car_to_global_grid(
    df_car: pd.DataFrame,
    df_global_grid: pd.DataFrame,
    chassis_id: str,
    time_col: str = "time_corrected",
    ffill_limit_sec: float = 0.2,
) -> pd.DataFrame:
    """Align single car's data to global time grid.

    Args:
        df_car: Car's telemetry data (already resampled to ~20Hz)
        df_global_grid: Global time grid DataFrame
        chassis_id: Vehicle identifier
        time_col: Name of car's time column
        ffill_limit_sec: Maximum forward-fill time

    Returns:
        DataFrame with car's data aligned to global grid
    """
    logger.info(f"  Aligning {chassis_id} to global grid...")

    # Rename car's time column to match global
    df_car = df_car.copy()

    # Validate time column has non-null values before proceeding
    if time_col in df_car.columns and df_car[time_col].notna().any():
        df_car["time_global"] = df_car[time_col]
    else:
        logger.warning(
            f"    {chassis_id}: No valid {time_col} values found. "
            f"Car data will not be synchronized to global grid."
        )
        # Return empty aligned dataframe with same structure
        df_aligned = df_global_grid.copy()
        df_aligned["chassis_id"] = chassis_id
        logger.info(f"    Coverage: 0/{len(df_aligned):,} frames (0.0%)")
        return df_aligned

    # Merge with global grid (left join to keep all global times)
    df_aligned = pd.merge(
        df_global_grid,
        df_car,
        on="time_global",
        how="left",
    )

    # Forward-fill chassis_id (so we know which car this is)
    if "chassis_id" not in df_aligned.columns or df_aligned["chassis_id"].isnull().all():
        df_aligned["chassis_id"] = chassis_id

    # Count presence using speed_final (fallback to speed if not available)
    if "speed_final" in df_aligned.columns:
        frames_present = df_aligned["speed_final"].notna().sum()
    elif "speed" in df_aligned.columns:
        frames_present = df_aligned["speed"].notna().sum()
    else:
        frames_present = 0

    frames_total = len(df_aligned)
    coverage = (frames_present / frames_total * 100) if frames_total > 0 else 0

    logger.info(f"    Coverage: {frames_present:,}/{frames_total:,} frames ({coverage:.1f}%)")

    return df_aligned


def synchronize_multi_car(
    dfs_by_car: Dict[str, pd.DataFrame],
    event_name: str,
    time_col: str = "time_corrected",
    freq_hz: float = 20.0,
    ffill_limit_sec: float = 0.2,
) -> Tuple[pd.DataFrame, SyncStats, Dict[str, CarCoverage]]:
    """Synchronize multiple cars to global time grid.

    This creates a unified DataFrame with all cars aligned to the same
    time axis, suitable for multi-car race replay and analysis.

    Args:
        dfs_by_car: Dict mapping chassis_id -> DataFrame (wide format, resampled)
        event_name: Event identifier
        time_col: Name of time column in each car's data
        freq_hz: Target sampling frequency
        ffill_limit_sec: Maximum forward-fill time

    Returns:
        (df_synchronized, sync_stats, coverage_by_car) tuple
    """
    logger.info("="*60)
    logger.info(f"MULTI-CAR SYNCHRONIZATION: {event_name}")
    logger.info("="*60)
    logger.info(f"  Cars: {len(dfs_by_car)}")
    logger.info(f"  Target frequency: {freq_hz} Hz")

    # Step 1: Determine global time range
    logger.info("\n  [Step 1] Determining global time range...")

    global_start, global_end = determine_global_time_range(dfs_by_car, time_col)
    time_span_sec = (global_end - global_start).total_seconds()

    # Step 2: Create global time grid
    logger.info("\n  [Step 2] Creating global time grid...")

    df_global_grid = create_global_time_grid(global_start, global_end, freq_hz)
    total_frames = len(df_global_grid)

    # Step 3: Align each car to global grid
    logger.info("\n  [Step 3] Aligning cars to global grid...")

    aligned_dfs = []
    coverage_by_car = {}

    for chassis_id, df_car in dfs_by_car.items():
        logger.info(f"\n  Processing {chassis_id}...")

        # Align to global grid
        df_aligned = align_car_to_global_grid(
            df_car=df_car,
            df_global_grid=df_global_grid,
            chassis_id=chassis_id,
            time_col=time_col,
            ffill_limit_sec=ffill_limit_sec,
        )

        aligned_dfs.append(df_aligned)

        # Track coverage using speed_final (fallback to speed if not available)
        if "speed_final" in df_aligned.columns:
            frames_present = df_aligned["speed_final"].notna().sum()
        elif "speed" in df_aligned.columns:
            frames_present = df_aligned["speed"].notna().sum()
        else:
            frames_present = 0

        coverage_pct = (frames_present / total_frames * 100) if total_frames > 0 else 0

        # Find first and last frame times
        present_mask = df_aligned["speed"].notna() if "speed" in df_aligned.columns else pd.Series([False] * len(df_aligned))

        if present_mask.any():
            first_frame = df_aligned.loc[present_mask, "time_global"].min()
            last_frame = df_aligned.loc[present_mask, "time_global"].max()
        else:
            first_frame = global_start
            last_frame = global_start

        coverage_by_car[chassis_id] = CarCoverage(
            chassis_id=chassis_id,
            frames_present=frames_present,
            frames_total=total_frames,
            coverage_pct=coverage_pct,
            first_frame_time=first_frame,
            last_frame_time=last_frame,
        )

    # Step 4: Concatenate all cars
    logger.info("\n  [Step 4] Concatenating all cars...")

    df_synchronized = pd.concat(aligned_dfs, ignore_index=True)

    logger.info(f"    Total rows: {len(df_synchronized):,}")
    logger.info(f"    Total columns: {len(df_synchronized.columns)}")

    # Step 5: Compute multi-car statistics
    logger.info("\n  [Step 5] Computing statistics...")

    # Cars per frame
    cars_per_frame = df_synchronized.groupby("time_global")["chassis_id"].nunique()

    cars_per_frame_mean = cars_per_frame.mean()
    cars_per_frame_min = cars_per_frame.min()
    cars_per_frame_max = cars_per_frame.max()

    logger.info(f"    Cars per frame: {cars_per_frame_mean:.1f} (min: {cars_per_frame_min}, max: {cars_per_frame_max})")

    # Overall coverage (% of frames with at least one car)
    frames_with_data = (cars_per_frame > 0).sum()
    coverage_pct = (frames_with_data / total_frames * 100) if total_frames > 0 else 0

    logger.info(f"    Coverage: {frames_with_data:,}/{total_frames:,} frames ({coverage_pct:.1f}%)")

    # Create sync stats
    sync_stats = SyncStats(
        event_name=event_name,
        num_cars=len(dfs_by_car),
        global_start_time=global_start,
        global_end_time=global_end,
        time_span_sec=time_span_sec,
        target_hz=freq_hz,
        total_frames=total_frames,
        cars_per_frame_mean=cars_per_frame_mean,
        cars_per_frame_min=cars_per_frame_min,
        cars_per_frame_max=cars_per_frame_max,
        coverage_pct=coverage_pct,
    )

    logger.info("="*60 + "\n")

    return df_synchronized, sync_stats, coverage_by_car


def save_synchronized_data(
    df: pd.DataFrame,
    output_path: Path,
    event_name: str,
    compression: str = "snappy",
    partitioned: bool = False,
) -> Path:
    """Save synchronized multi-car data to Parquet.

    Args:
        df: Synchronized DataFrame with all cars
        output_path: Output directory
        event_name: Event identifier
        compression: Parquet compression method
        partitioned: Whether to partition by chassis_id

    Returns:
        Path to saved file/directory
    """
    output_dir = output_path / event_name / "synchronized"
    output_dir.mkdir(parents=True, exist_ok=True)

    if partitioned:
        # Save partitioned by chassis_id
        logger.info(f"Saving synchronized data (partitioned) to {output_dir}")

        df.to_parquet(
            output_dir,
            partition_cols=["chassis_id"],
            compression=compression,
            index=False,
        )

        logger.info(f"  Saved {len(df):,} rows, partitioned by chassis_id")

        return output_dir
    else:
        # Save as single file
        output_file = output_dir / "multi_car_frames.parquet"

        logger.info(f"Saving synchronized data to {output_file}")

        df.to_parquet(output_file, compression=compression, index=False)

        logger.info(f"  Saved {len(df):,} rows, {len(df.columns)} columns")
        logger.info(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        return output_file


def save_sync_stats(
    sync_stats: SyncStats,
    coverage_by_car: Dict[str, CarCoverage],
    output_path: Path,
    event_name: str,
) -> None:
    """Save synchronization statistics.

    Args:
        sync_stats: Overall sync statistics
        coverage_by_car: Per-car coverage statistics
        output_path: Output directory
        event_name: Event identifier
    """
    logger.info(f"Saving sync statistics to {output_path}")

    # Save overall stats
    stats_dict = {
        "event_name": sync_stats.event_name,
        "num_cars": sync_stats.num_cars,
        "global_start_time": sync_stats.global_start_time.isoformat(),
        "global_end_time": sync_stats.global_end_time.isoformat(),
        "time_span_sec": sync_stats.time_span_sec,
        "target_hz": sync_stats.target_hz,
        "total_frames": sync_stats.total_frames,
        "cars_per_frame_mean": sync_stats.cars_per_frame_mean,
        "cars_per_frame_min": sync_stats.cars_per_frame_min,
        "cars_per_frame_max": sync_stats.cars_per_frame_max,
        "coverage_pct": sync_stats.coverage_pct,
    }

    df_stats = pd.DataFrame([stats_dict])

    output_file = output_path / event_name / "sync_stats.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_stats.to_parquet(output_file, compression="snappy", index=False)

    logger.info(f"  Saved overall stats to: {output_file}")

    # Save per-car coverage
    coverage_data = []
    for chassis_id, coverage in coverage_by_car.items():
        coverage_data.append({
            "chassis_id": coverage.chassis_id,
            "frames_present": coverage.frames_present,
            "frames_total": coverage.frames_total,
            "coverage_pct": coverage.coverage_pct,
            "first_frame_time": coverage.first_frame_time.isoformat(),
            "last_frame_time": coverage.last_frame_time.isoformat(),
        })

    df_coverage = pd.DataFrame(coverage_data)

    output_file_coverage = output_path / event_name / "car_coverage.parquet"
    df_coverage.to_parquet(output_file_coverage, compression="snappy", index=False)

    logger.info(f"  Saved per-car coverage to: {output_file_coverage}")

    # Log summary
    logger.info("\n  Sync Summary:")
    logger.info(f"    Total cars: {sync_stats.num_cars}")
    logger.info(f"    Total frames: {sync_stats.total_frames:,}")
    logger.info(f"    Mean cars/frame: {sync_stats.cars_per_frame_mean:.1f}")
    logger.info(f"    Overall coverage: {sync_stats.coverage_pct:.1f}%")
    logger.info(f"    Mean per-car coverage: {df_coverage['coverage_pct'].mean():.1f}%")
