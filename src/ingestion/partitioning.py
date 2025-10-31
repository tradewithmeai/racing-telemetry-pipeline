"""Partitioned Parquet writing for efficient storage and querying."""

import pandas as pd
import polars as pl
from pathlib import Path
from typing import Union, List, Optional
from src.utils.logging_utils import get_logger
from src.utils.io_utils import ensure_dir

logger = get_logger(__name__)

DataFrame = Union[pd.DataFrame, pl.DataFrame]


def write_partitioned_parquet(
    df: DataFrame,
    output_dir: str | Path,
    partition_cols: List[str],
    compression: str = "snappy",
    max_rows_per_file: int = 1_000_000,
) -> int:
    """Write DataFrame to partitioned Parquet.

    Args:
        df: Input DataFrame
        output_dir: Output directory
        partition_cols: Columns to partition by
        compression: Compression codec
        max_rows_per_file: Maximum rows per file

    Returns:
        Number of files written
    """
    output_path = Path(output_dir)
    ensure_dir(output_path)

    if isinstance(df, pd.DataFrame):
        return _write_pandas_partitioned(
            df, output_path, partition_cols, compression, max_rows_per_file
        )
    elif isinstance(df, pl.DataFrame):
        return _write_polars_partitioned(
            df, output_path, partition_cols, compression
        )
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def _write_pandas_partitioned(
    df: pd.DataFrame,
    output_path: Path,
    partition_cols: List[str],
    compression: str,
    max_rows_per_file: int,
) -> int:
    """Write pandas DataFrame with partitioning."""
    # Use manual partitioning for reliability
    # PyArrow's automatic partitioning has complex parameter interactions
    return _write_manual_partitioned(df, output_path, partition_cols, compression)


def _write_polars_partitioned(
    df: pl.DataFrame,
    output_path: Path,
    partition_cols: List[str],
    compression: str,
) -> int:
    """Write polars DataFrame with partitioning."""
    # Polars doesn't have built-in partitioning, use manual
    return _write_manual_partitioned(
        df.to_pandas(), output_path, partition_cols, compression
    )


def _write_manual_partitioned(
    df: pd.DataFrame,
    output_path: Path,
    partition_cols: List[str],
    compression: str,
) -> int:
    """Manual partitioning by grouping."""
    files_written = 0

    # Group by partition columns
    if len(partition_cols) == 0:
        # No partitioning, write single file
        file_path = output_path / "data.parquet"
        ensure_dir(file_path.parent)
        df.to_parquet(file_path, compression=compression, index=False)
        files_written = 1

    else:
        # Partition manually
        for group_keys, group_df in df.groupby(partition_cols):
            # Ensure group_keys is tuple
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)

            # Create partition directory
            partition_parts = [
                f"{col}={val}" for col, val in zip(partition_cols, group_keys)
            ]
            partition_dir = output_path / "/".join(partition_parts)
            ensure_dir(partition_dir)

            # Write file
            file_path = partition_dir / f"part-{files_written:05d}.parquet"
            group_df.to_parquet(file_path, compression=compression, index=False)
            files_written += 1

    logger.info(f"Wrote {files_written} partitioned files to {output_path}")
    return files_written


def write_raw_curated(
    df: DataFrame,
    event_name: str,
    output_base: str | Path,
    compression: str = "snappy",
) -> int:
    """Write DataFrame to raw_curated layer with standard partitioning.

    Partitions by: event -> chassis -> signal

    Args:
        df: DataFrame with columns: chassis_id, telemetry_name, etc.
        event_name: Event identifier
        output_base: Base output directory
        compression: Compression codec

    Returns:
        Number of files written
    """
    # Add event column if not present
    if isinstance(df, pd.DataFrame):
        if "event" not in df.columns:
            df = df.copy()  # Avoid SettingWithCopyWarning
            df["event"] = event_name
    elif isinstance(df, pl.DataFrame):
        if "event" not in df.columns:
            df = df.with_columns(pl.lit(event_name).alias("event"))

    # Output path
    output_dir = Path(output_base) / event_name / "raw_curated"

    # Partition by event, chassis, signal
    partition_cols = ["event", "chassis_id", "telemetry_name"]

    logger.info(f"Writing raw_curated data to {output_dir}")
    logger.info(f"Partitioning by: {partition_cols}")

    num_files = write_partitioned_parquet(
        df, output_dir, partition_cols, compression=compression
    )

    return num_files


def write_refined(
    df: DataFrame,
    event_name: str,
    output_base: str | Path,
    segment_id: Optional[int] = None,
    compression: str = "snappy",
) -> int:
    """Write DataFrame to refined layer with segmentation.

    Partitions by: event -> chassis -> segment -> signal

    Args:
        df: DataFrame with time-corrected data
        event_name: Event identifier
        output_base: Base output directory
        segment_id: Optional segment ID for segmented data
        compression: Compression codec

    Returns:
        Number of files written
    """
    # Add metadata columns
    if isinstance(df, pd.DataFrame):
        if "event" not in df.columns:
            df["event"] = event_name
        if segment_id is not None and "segment_id" not in df.columns:
            df["segment_id"] = segment_id
    elif isinstance(df, pl.DataFrame):
        if "event" not in df.columns:
            df = df.with_columns(pl.lit(event_name).alias("event"))
        if segment_id is not None and "segment_id" not in df.columns:
            df = df.with_columns(pl.lit(segment_id).alias("segment_id"))

    # Output path
    output_dir = Path(output_base) / event_name / "refined"

    # Partition columns
    partition_cols = ["event", "chassis_id"]
    if segment_id is not None:
        partition_cols.append("segment_id")
    partition_cols.append("telemetry_name")

    logger.info(f"Writing refined data to {output_dir}")
    logger.info(f"Partitioning by: {partition_cols}")

    num_files = write_partitioned_parquet(
        df, output_dir, partition_cols, compression=compression
    )

    return num_files
