"""Time manipulation and drift computation utilities."""

from datetime import datetime, timedelta
from typing import Union, Tuple, List, Optional
import pandas as pd
import numpy as np


def parse_timestamp(ts: Union[str, datetime, pd.Timestamp]) -> datetime:
    """Parse timestamp to datetime.

    Args:
        ts: Timestamp in various formats

    Returns:
        datetime object
    """
    if isinstance(ts, datetime):
        return ts
    elif isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime()
    elif isinstance(ts, str):
        return pd.to_datetime(ts).to_pydatetime()
    else:
        raise ValueError(f"Cannot parse timestamp of type {type(ts)}: {ts}")


def compute_drift(
    timestamp: pd.Series, meta_time: pd.Series, method: str = "median"
) -> Tuple[float, float]:
    """Compute time drift between ECU and receiver clocks.

    Args:
        timestamp: ECU timestamps
        meta_time: Receiver timestamps
        method: 'median' (robust) or 'mean'

    Returns:
        (drift_sec, drift_std) tuple
    """
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(timestamp):
        timestamp = pd.to_datetime(timestamp)
    if not pd.api.types.is_datetime64_any_dtype(meta_time):
        meta_time = pd.to_datetime(meta_time)

    # Compute drift in seconds
    drift = (meta_time - timestamp).dt.total_seconds()

    # Remove NaN
    drift = drift.dropna()

    if len(drift) == 0:
        return 0.0, 0.0

    # Compute statistics
    if method == "median":
        drift_sec = float(drift.median())
    elif method == "mean":
        drift_sec = float(drift.mean())
    else:
        raise ValueError(f"Unknown method: {method}")

    drift_std = float(drift.std())

    return drift_sec, drift_std


def detect_backwards_time(
    timestamp: pd.Series, tolerance_sec: float = 0.001
) -> pd.Series:
    """Detect backwards-moving timestamps.

    Args:
        timestamp: Timestamp series (must be sorted by original index)
        tolerance_sec: Allow small backwards movements (clock jitter)

    Returns:
        Boolean series indicating backwards time points
    """
    if not pd.api.types.is_datetime64_any_dtype(timestamp):
        timestamp = pd.to_datetime(timestamp)

    # Compute time delta from previous
    time_diff = timestamp.diff().dt.total_seconds()

    # Backwards if diff < -tolerance
    is_backwards = time_diff < -tolerance_sec

    return is_backwards


def detect_time_gaps(
    timestamp: pd.Series, gap_threshold_sec: float = 2.0
) -> pd.Series:
    """Detect large time gaps in timestamp series.

    Args:
        timestamp: Timestamp series (must be sorted)
        gap_threshold_sec: Threshold for flagging large gaps

    Returns:
        Boolean series indicating points after large gaps
    """
    if not pd.api.types.is_datetime64_any_dtype(timestamp):
        timestamp = pd.to_datetime(timestamp)

    # Compute time delta from previous
    time_diff = timestamp.diff().dt.total_seconds()

    # Gap if diff > threshold
    is_gap = time_diff > gap_threshold_sec

    return is_gap


def apply_drift_correction(
    timestamp: pd.Series, drift_sec: float
) -> pd.Series:
    """Apply drift correction to timestamps.

    Args:
        timestamp: ECU timestamps
        drift_sec: Drift in seconds to add

    Returns:
        Corrected timestamps
    """
    if not pd.api.types.is_datetime64_any_dtype(timestamp):
        timestamp = pd.to_datetime(timestamp)

    corrected = timestamp + pd.Timedelta(seconds=drift_sec)
    return corrected


def segment_by_gaps(
    timestamp: pd.Series, gap_threshold_sec: float = 2.0
) -> pd.Series:
    """Assign segment IDs based on time gaps.

    Args:
        timestamp: Timestamp series
        gap_threshold_sec: Threshold for starting new segment

    Returns:
        Integer series with segment IDs (0, 1, 2, ...)
    """
    gaps = detect_time_gaps(timestamp, gap_threshold_sec)

    # Cumulative sum of gaps gives segment ID
    segment_id = gaps.cumsum()

    return segment_id


def windowed_drift_estimation(
    timestamp: pd.Series,
    meta_time: pd.Series,
    window_minutes: int = 5,
    method: str = "median",
) -> pd.DataFrame:
    """Compute drift in rolling time windows.

    Args:
        timestamp: ECU timestamps
        meta_time: Receiver timestamps
        window_minutes: Window size in minutes
        method: 'median' or 'mean'

    Returns:
        DataFrame with columns: window_start, window_end, drift_sec, drift_std, samples
    """
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(timestamp):
        timestamp = pd.to_datetime(timestamp)
    if not pd.api.types.is_datetime64_any_dtype(meta_time):
        meta_time = pd.to_datetime(meta_time)

    # Create dataframe
    df = pd.DataFrame({"timestamp": timestamp, "meta_time": meta_time})
    df = df.dropna()

    # Compute drift
    df["drift"] = (df["meta_time"] - df["timestamp"]).dt.total_seconds()

    # Set timestamp as index for resampling
    df = df.set_index("timestamp")

    # Resample to windows
    window_str = f"{window_minutes}min"  # min = minutes

    windows = []
    for window_start, window_df in df.resample(window_str):
        if len(window_df) == 0:
            continue

        if method == "median":
            drift_sec = window_df["drift"].median()
        else:
            drift_sec = window_df["drift"].mean()

        drift_std = window_df["drift"].std()
        samples = len(window_df)

        window_end = window_start + pd.Timedelta(minutes=window_minutes)

        windows.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                "drift_sec": drift_sec,
                "drift_std": drift_std,
                "samples": samples,
            }
        )

    return pd.DataFrame(windows)


def detect_clock_steps(
    drift_windows: pd.DataFrame, spike_threshold: float = 3.0
) -> pd.Series:
    """Detect sudden clock step changes from windowed drift.

    Args:
        drift_windows: DataFrame from windowed_drift_estimation
        spike_threshold: Multiplier for median drift_std

    Returns:
        Boolean series indicating windows with step changes
    """
    if len(drift_windows) == 0:
        return pd.Series(dtype=bool)

    median_std = drift_windows["drift_std"].median()

    # Step detected if drift_std > threshold * median_std
    is_step = drift_windows["drift_std"] > (spike_threshold * median_std)

    return is_step
