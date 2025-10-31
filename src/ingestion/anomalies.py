"""Anomaly detection for telemetry ingestion."""

import pandas as pd
import polars as pl
from typing import Dict, List, Tuple, Union
from src.schemas.raw import VehicleIdentity
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

DataFrame = Union[pd.DataFrame, pl.DataFrame]


def extract_vehicle_identity(vehicle_id: str) -> Dict[str, str]:
    """Extract chassis_id and car_no from vehicle_id.

    Args:
        vehicle_id: Vehicle ID string (e.g., "GR86-004-78")

    Returns:
        Dict with vehicle_id, chassis_id, car_no
    """
    try:
        identity = VehicleIdentity.from_vehicle_id(vehicle_id)
        return {
            "vehicle_id": identity.vehicle_id,
            "chassis_id": identity.chassis_id,
            "car_no": identity.car_no,
        }
    except ValueError as e:
        logger.warning(f"Failed to parse vehicle_id '{vehicle_id}': {e}")
        return {
            "vehicle_id": vehicle_id,
            "chassis_id": "UNKNOWN",
            "car_no": "UNKNOWN",
        }


def detect_duplicates(
    df: DataFrame,
    subset: List[str] = ["vehicle_id", "timestamp", "telemetry_name"],
    keep: str = "last",
) -> Tuple[DataFrame, int]:
    """Detect and optionally remove duplicate rows.

    Args:
        df: Input DataFrame
        subset: Columns to check for duplicates
        keep: Which duplicates to keep ('first', 'last', or False to mark all)

    Returns:
        (cleaned_df, num_duplicates) tuple
    """
    if isinstance(df, pd.DataFrame):
        # Pandas
        duplicated = df.duplicated(subset=subset, keep=False)
        num_duplicates = duplicated.sum()

        if num_duplicates > 0:
            logger.warning(
                f"Found {num_duplicates} duplicate rows based on {subset}. "
                f"Keeping '{keep}' occurrence."
            )

        if keep != False:
            df_clean = df.drop_duplicates(subset=subset, keep=keep)
        else:
            df_clean = df

        return df_clean, num_duplicates

    elif isinstance(df, pl.DataFrame):
        # Polars
        df_with_dup = df.with_columns(
            pl.struct(subset).is_duplicated().alias("_is_duplicate")
        )
        num_duplicates = df_with_dup.filter(pl.col("_is_duplicate")).height

        if num_duplicates > 0:
            logger.warning(
                f"Found {num_duplicates} duplicate rows based on {subset}. "
                f"Keeping '{keep}' occurrence."
            )

        if keep == "last":
            df_clean = df.unique(subset=subset, keep="last")
        elif keep == "first":
            df_clean = df.unique(subset=subset, keep="first")
        else:
            df_clean = df

        return df_clean, num_duplicates

    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def detect_backwards_time_per_car(
    df: DataFrame, tolerance_sec: float = 0.001
) -> Dict[str, int]:
    """Detect backwards-moving timestamps per car.

    Args:
        df: DataFrame with 'vehicle_id' and 'timestamp' columns
        tolerance_sec: Tolerance for small clock jitter

    Returns:
        Dict mapping vehicle_id to count of backwards timestamps
    """
    backwards_counts = {}

    if isinstance(df, pd.DataFrame):
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Group by vehicle
        for vehicle_id, group in df.groupby("vehicle_id"):
            # Sort by original index to preserve data order
            group_sorted = group.sort_index()

            # Compute time diff
            time_diff = group_sorted["timestamp"].diff().dt.total_seconds()

            # Count backwards
            backwards = (time_diff < -tolerance_sec).sum()

            if backwards > 0:
                backwards_counts[vehicle_id] = backwards

    elif isinstance(df, pl.DataFrame):
        # Polars version
        for vehicle_id in df["vehicle_id"].unique():
            vehicle_df = df.filter(pl.col("vehicle_id") == vehicle_id)

            # Compute time diff
            time_diff = vehicle_df.select(
                pl.col("timestamp").cast(pl.Datetime).diff().dt.total_seconds()
            )

            # Count backwards
            backwards = time_diff.filter(pl.col("timestamp") < -tolerance_sec).height

            if backwards > 0:
                backwards_counts[vehicle_id] = backwards

    if backwards_counts:
        total = sum(backwards_counts.values())
        logger.error(
            f"Detected {total} backwards timestamps across {len(backwards_counts)} cars: "
            f"{backwards_counts}"
        )

    return backwards_counts


def detect_gaps_per_car(
    df: DataFrame, gap_threshold_sec: float = 2.0
) -> Dict[str, List[Dict]]:
    """Detect large time gaps per car.

    Args:
        df: DataFrame with 'vehicle_id' and 'timestamp' columns
        gap_threshold_sec: Threshold for flagging gaps

    Returns:
        Dict mapping vehicle_id to list of gaps (each with start_time, gap_sec)
    """
    gaps_by_car = {}

    if isinstance(df, pd.DataFrame):
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Group by vehicle
        for vehicle_id, group in df.groupby("vehicle_id"):
            group_sorted = group.sort_values("timestamp")

            # Compute time diff
            time_diff = group_sorted["timestamp"].diff().dt.total_seconds()

            # Find gaps
            gap_mask = time_diff > gap_threshold_sec
            gap_indices = gap_mask[gap_mask].index

            if len(gap_indices) > 0:
                gaps = []
                for idx in gap_indices:
                    gap_sec = time_diff.loc[idx]
                    start_time = group_sorted.loc[idx, "timestamp"]
                    gaps.append(
                        {
                            "start_time": start_time.isoformat(),
                            "gap_sec": float(gap_sec),
                        }
                    )

                gaps_by_car[vehicle_id] = gaps

    elif isinstance(df, pl.DataFrame):
        # Polars version
        for vehicle_id in df["vehicle_id"].unique():
            vehicle_df = df.filter(pl.col("vehicle_id") == vehicle_id).sort("timestamp")

            # Compute time diff
            vehicle_df = vehicle_df.with_columns(
                pl.col("timestamp").cast(pl.Datetime).diff().dt.total_seconds().alias("time_diff")
            )

            # Find gaps
            gap_rows = vehicle_df.filter(pl.col("time_diff") > gap_threshold_sec)

            if gap_rows.height > 0:
                gaps = []
                for row in gap_rows.iter_rows(named=True):
                    gaps.append(
                        {
                            "start_time": row["timestamp"].isoformat(),
                            "gap_sec": float(row["time_diff"]),
                        }
                    )

                gaps_by_car[vehicle_id] = gaps

    if gaps_by_car:
        total_gaps = sum(len(gaps) for gaps in gaps_by_car.values())
        logger.warning(
            f"Detected {total_gaps} time gaps >{gap_threshold_sec}s "
            f"across {len(gaps_by_car)} cars"
        )

    return gaps_by_car


def validate_schema(df: DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns.

    Args:
        df: Input DataFrame
        required_columns: List of required column names

    Returns:
        True if valid, raises ValueError if not
    """
    if isinstance(df, pd.DataFrame):
        missing = set(required_columns) - set(df.columns)
    elif isinstance(df, pl.DataFrame):
        missing = set(required_columns) - set(df.columns)
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Schema validation passed. Found all {len(required_columns)} required columns.")
    return True
