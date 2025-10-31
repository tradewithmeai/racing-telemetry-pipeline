"""Pivot telemetry data from long format to wide format."""

import pandas as pd
import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PivotStats:
    """Statistics from pivot operation."""

    chassis_id: str
    rows_before: int
    rows_after: int
    signals_before: int
    signals_after: int
    missing_signals: List[str]
    coverage_pct: float


def identify_signal_columns(df: pd.DataFrame) -> List[str]:
    """Identify all unique telemetry signal names in long-format data.

    Args:
        df: DataFrame in long format with telemetry_name column

    Returns:
        List of unique signal names
    """
    if "telemetry_name" not in df.columns:
        raise ValueError("DataFrame must have 'telemetry_name' column for pivot")

    signals = df["telemetry_name"].unique().tolist()
    signals = [s for s in signals if pd.notna(s)]  # Remove NaN

    logger.info(f"  Identified {len(signals)} unique signals")

    return sorted(signals)


def pivot_to_wide_format(
    df: pd.DataFrame,
    chassis_id: str,
    time_col: str = "time_corrected",
    signal_col: str = "telemetry_name",
    value_col: str = "telemetry_value",
    preserve_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, PivotStats]:
    """Pivot telemetry data from long format to wide format.

    Converts:
        | time_corrected | chassis_id | telemetry_name | telemetry_value |
        |----------------|------------|----------------|-----------------|
        | 12:00:00       | 010        | speed          | 45.5            |
        | 12:00:00       | 010        | aps            | 0.8             |

    To:
        | time_corrected | chassis_id | speed | aps  | ... |
        |----------------|------------|-------|------|-----|
        | 12:00:00       | 010        | 45.5  | 0.8  | ... |

    Args:
        df: DataFrame in long format
        chassis_id: Vehicle identifier
        time_col: Name of time column to group by
        signal_col: Name of column containing signal names
        value_col: Name of column containing signal values
        preserve_cols: Additional columns to preserve (e.g., lap, segment_id)

    Returns:
        (df_wide, pivot_stats) tuple
    """
    logger.info("="*60)
    logger.info(f"PIVOT: {chassis_id}")
    logger.info("="*60)

    rows_before = len(df)
    signals_before = df[signal_col].nunique()

    logger.info(f"  Input: {rows_before:,} rows, {signals_before} unique signals")
    logger.info(f"  Time column: {time_col}")

    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        logger.info(f"  Converting {time_col} to datetime...")
        df[time_col] = pd.to_datetime(df[time_col])

    # Default preserve columns
    if preserve_cols is None:
        preserve_cols = ["chassis_id", "car_no", "lap_repaired", "segment_id"]

    # Filter to only columns that exist
    preserve_cols = [col for col in preserve_cols if col in df.columns]

    logger.info(f"  Preserving columns: {preserve_cols}")

    # Identify all signals
    all_signals = identify_signal_columns(df)

    # Step 1: Pivot the data
    logger.info("\n  [Step 1] Pivoting to wide format...")

    # Group by time + preserve cols, then pivot
    index_cols = [time_col] + preserve_cols

    # Create pivot table
    df_wide = df.pivot_table(
        index=index_cols,
        columns=signal_col,
        values=value_col,
        aggfunc="first",  # Take first value if duplicates
    ).reset_index()

    # Flatten column names (remove MultiIndex if present)
    df_wide.columns.name = None

    rows_after = len(df_wide)
    signals_after = len([col for col in df_wide.columns if col not in index_cols])

    logger.info(f"  Output: {rows_after:,} rows, {signals_after} signals as columns")

    # Step 2: Identify missing signals
    missing_signals = [sig for sig in all_signals if sig not in df_wide.columns]

    if missing_signals:
        logger.warning(f"  Missing signals: {missing_signals}")

    # Step 3: Rename position columns if present
    logger.info("\n  [Step 2] Processing position data...")

    position_renames = {
        "VBOX_Lat_Min": "gps_lat",
        "VBOX_Long_Minutes": "gps_lon",
        "Laptrigger_lapdist_dls": "track_distance_m",
    }

    for old_name, new_name in position_renames.items():
        if old_name in df_wide.columns:
            logger.info(f"    Renaming: {old_name} → {new_name}")
            df_wide = df_wide.rename(columns={old_name: new_name})

    # Step 4: Normalize units for known signals
    logger.info("\n  [Step 3] Normalizing signal units...")

    # APS: Convert from percentage to 0-1
    if "aps" in df_wide.columns:
        df_wide["aps_raw"] = df_wide["aps"].copy()
        df_wide["aps"] = df_wide["aps"] / 100.0
        logger.info("    aps: % → [0, 1]")

    # Speed: Detect units and convert to m/s
    if "speed" in df_wide.columns:
        df_wide["speed_raw"] = df_wide["speed"].copy()

        # Detect units based on typical racing speeds
        mean_speed = df_wide["speed"].mean()

        if mean_speed > 150:  # Likely km/h
            df_wide["speed"] = df_wide["speed"] / 3.6
            logger.info(f"    speed: km/h → m/s (mean: {mean_speed:.1f} → {df_wide['speed'].mean():.1f})")
        elif mean_speed > 50:  # Likely m/s already
            logger.info(f"    speed: already in m/s (mean: {mean_speed:.1f})")
        else:  # Uncertain
            logger.warning(f"    speed: unknown units (mean: {mean_speed:.1f}), leaving as-is")

    # Step 5: Compute coverage
    logger.info("\n  [Step 4] Computing signal coverage...")

    coverage_pct = (signals_after / len(all_signals)) * 100 if len(all_signals) > 0 else 0

    logger.info(f"    Coverage: {signals_after}/{len(all_signals)} signals ({coverage_pct:.1f}%)")

    # Create stats
    pivot_stats = PivotStats(
        chassis_id=chassis_id,
        rows_before=rows_before,
        rows_after=rows_after,
        signals_before=signals_before,
        signals_after=signals_after,
        missing_signals=missing_signals,
        coverage_pct=coverage_pct,
    )

    logger.info("="*60 + "\n")

    return df_wide, pivot_stats


def pivot_to_wide_format_polars(
    df: pl.DataFrame,
    chassis_id: str,
    time_col: str = "time_corrected",
    signal_col: str = "telemetry_name",
    value_col: str = "telemetry_value",
    preserve_cols: Optional[List[str]] = None,
) -> Tuple[pl.DataFrame, PivotStats]:
    """Pivot telemetry data using Polars (faster for large datasets).

    Args:
        df: Polars DataFrame in long format
        chassis_id: Vehicle identifier
        time_col: Name of time column to group by
        signal_col: Name of column containing signal names
        value_col: Name of column containing signal values
        preserve_cols: Additional columns to preserve

    Returns:
        (df_wide, pivot_stats) tuple
    """
    logger.info("="*60)
    logger.info(f"PIVOT (Polars): {chassis_id}")
    logger.info("="*60)

    rows_before = len(df)
    signals_before = df[signal_col].n_unique()

    logger.info(f"  Input: {rows_before:,} rows, {signals_before} unique signals")

    # Default preserve columns
    if preserve_cols is None:
        preserve_cols = ["chassis_id", "car_no", "lap_repaired", "segment_id"]

    # Filter to existing columns
    preserve_cols = [col for col in preserve_cols if col in df.columns]

    logger.info(f"  Preserving columns: {preserve_cols}")

    # Pivot using Polars
    logger.info("\n  [Step 1] Pivoting to wide format...")

    index_cols = [time_col] + preserve_cols

    df_wide = df.pivot(
        values=value_col,
        index=index_cols,
        columns=signal_col,
        aggregate_function="first",
    )

    rows_after = len(df_wide)
    signals_after = len(df_wide.columns) - len(index_cols)

    logger.info(f"  Output: {rows_after:,} rows, {signals_after} signals as columns")

    # Identify all signals
    all_signals = df[signal_col].unique().to_list()
    all_signals = [s for s in all_signals if s is not None]

    missing_signals = [sig for sig in all_signals if sig not in df_wide.columns]

    if missing_signals:
        logger.warning(f"  Missing signals: {missing_signals}")

    # Compute coverage
    coverage_pct = (signals_after / len(all_signals)) * 100 if len(all_signals) > 0 else 0

    pivot_stats = PivotStats(
        chassis_id=chassis_id,
        rows_before=rows_before,
        rows_after=rows_after,
        signals_before=signals_before,
        signals_after=signals_after,
        missing_signals=missing_signals,
        coverage_pct=coverage_pct,
    )

    logger.info("="*60 + "\n")

    return df_wide, pivot_stats


def save_wide_format(
    df: pd.DataFrame,
    output_path: Path,
    event_name: str,
    chassis_id: str,
    compression: str = "snappy",
) -> Path:
    """Save wide-format DataFrame to Parquet.

    Args:
        df: Wide-format DataFrame
        output_path: Output directory
        event_name: Event identifier
        chassis_id: Vehicle identifier
        compression: Parquet compression method

    Returns:
        Path to saved file
    """
    output_dir = output_path / event_name / "wide_format"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{chassis_id}.parquet"

    logger.info(f"Saving wide format data to {output_file}")

    df.to_parquet(output_file, compression=compression, index=False)

    logger.info(f"  Saved {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    return output_file


def save_pivot_stats(
    stats_by_car: Dict[str, PivotStats],
    output_path: Path,
    event_name: str,
) -> None:
    """Save pivot statistics to Parquet.

    Args:
        stats_by_car: Dict mapping chassis_id -> PivotStats
        output_path: Output directory
        event_name: Event identifier
    """
    logger.info(f"Saving pivot statistics to {output_path}")

    all_stats = []
    for chassis_id, stats in stats_by_car.items():
        all_stats.append({
            "chassis_id": stats.chassis_id,
            "rows_before": stats.rows_before,
            "rows_after": stats.rows_after,
            "signals_before": stats.signals_before,
            "signals_after": stats.signals_after,
            "missing_signals": ",".join(stats.missing_signals),
            "coverage_pct": stats.coverage_pct,
        })

    if len(all_stats) == 0:
        logger.warning("No pivot statistics to save")
        return

    df = pd.DataFrame(all_stats)

    # Save to Parquet
    output_file = output_path / event_name / "pivot_stats.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, compression="snappy", index=False)

    logger.info(f"  Saved statistics for {len(all_stats)} vehicles")
    logger.info(f"  Output: {output_file}")

    # Log summary
    logger.info("\n  Pivot Summary:")
    logger.info(f"    Mean coverage: {df['coverage_pct'].mean():.1f}%")
    logger.info(f"    Mean signals: {df['signals_after'].mean():.1f}")
    logger.info(f"    Total rows after pivot: {df['rows_after'].sum():,}")
