"""Drift calibration with step detection and robust estimation."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.utils.logging_utils import get_logger
from src.utils.time_utils import compute_drift, windowed_drift_estimation, detect_clock_steps

logger = get_logger(__name__)


@dataclass
class DriftCalibration:
    """Drift calibration result for a time segment."""

    chassis_id: str
    segment_id: int
    window_start: datetime
    window_end: datetime
    drift_sec: float
    drift_std: float
    step_detected: bool
    samples: int
    method: str = "median"

    @property
    def is_valid(self) -> bool:
        """Check if calibration is valid."""
        return self.samples > 10 and self.drift_std < 30.0

    @property
    def quality_level(self) -> str:
        """Get quality assessment."""
        if self.drift_std < 1.0:
            return "EXCELLENT"
        elif self.drift_std < 5.0:
            return "GOOD"
        elif self.drift_std < 10.0:
            return "ACCEPTABLE"
        elif self.drift_std < 30.0:
            return "POOR"
        else:
            return "INVALID"


def windowed_drift_calibration(
    df: pd.DataFrame,
    chassis_id: str,
    window_minutes: int = 5,
    method: str = "median",
    spike_threshold: float = 3.0,
) -> List[DriftCalibration]:
    """Compute drift calibration in rolling windows with step detection.

    Args:
        df: DataFrame with timestamp and meta_time columns (sorted by index)
        chassis_id: Vehicle identifier
        window_minutes: Window size in minutes
        method: 'median' (robust) or 'mean'
        spike_threshold: Multiplier for step detection

    Returns:
        List of DriftCalibration objects per window
    """
    logger.info(f"Computing windowed drift for {chassis_id} (window={window_minutes}min)")

    # Ensure datetime types
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not pd.api.types.is_datetime64_any_dtype(df["meta_time"]):
        df = df.copy()
        df["meta_time"] = pd.to_datetime(df["meta_time"])

    # Use existing utility function
    drift_windows = windowed_drift_estimation(
        df["timestamp"], df["meta_time"], window_minutes=window_minutes, method=method
    )

    if len(drift_windows) == 0:
        logger.warning(f"No drift windows computed for {chassis_id}")
        return []

    # Detect clock steps
    step_flags = detect_clock_steps(drift_windows, spike_threshold=spike_threshold)

    # Convert to DriftCalibration objects
    calibrations = []
    for idx, row in drift_windows.iterrows():
        calib = DriftCalibration(
            chassis_id=chassis_id,
            segment_id=idx,  # Will be updated later with proper segment IDs
            window_start=row["window_start"],
            window_end=row["window_end"],
            drift_sec=float(row["drift_sec"]),
            drift_std=float(row["drift_std"]),
            step_detected=bool(step_flags.iloc[idx]) if idx < len(step_flags) else False,
            samples=int(row["samples"]),
            method=method,
        )
        calibrations.append(calib)

    # Log statistics
    valid_calibrations = [c for c in calibrations if c.is_valid]
    step_count = sum(c.step_detected for c in calibrations)

    logger.info(f"  {chassis_id}: {len(calibrations)} windows")
    logger.info(f"  Valid calibrations: {len(valid_calibrations)}/{len(calibrations)}")
    logger.info(f"  Clock steps detected: {step_count}")

    if valid_calibrations:
        avg_drift = np.mean([c.drift_sec for c in valid_calibrations])
        avg_std = np.mean([c.drift_std for c in valid_calibrations])
        logger.info(f"  Average drift: {avg_drift:.3f}s ± {avg_std:.3f}s")

    return calibrations


def apply_segmented_drift_correction(
    df: pd.DataFrame,
    calibrations: List[DriftCalibration],
) -> pd.DataFrame:
    """Apply drift corrections per segment.

    Args:
        df: DataFrame with timestamp column
        calibrations: List of calibrations (one per segment)

    Returns:
        DataFrame with time_corrected column added
    """
    df = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Initialize time_corrected
    df["time_corrected"] = df["timestamp"]

    # Apply corrections per window
    for calib in calibrations:
        # Find rows in this window
        mask = (df["timestamp"] >= calib.window_start) & (df["timestamp"] < calib.window_end)

        if calib.is_valid:
            # Apply drift correction
            df.loc[mask, "time_corrected"] = df.loc[mask, "timestamp"] + pd.Timedelta(
                seconds=calib.drift_sec
            )
            df.loc[mask, "segment_id"] = calib.segment_id
        else:
            logger.warning(
                f"Invalid calibration for segment {calib.segment_id} "
                f"(std={calib.drift_std:.2f}s, samples={calib.samples})"
            )
            # Keep original timestamp
            df.loc[mask, "segment_id"] = calib.segment_id

    return df


def compute_overall_drift(
    calibrations: List[DriftCalibration],
    method: str = "median",
) -> Tuple[float, float]:
    """Compute overall drift statistics from calibrations.

    Args:
        calibrations: List of calibrations
        method: 'median' or 'mean'

    Returns:
        (overall_drift_sec, overall_drift_std) tuple
    """
    valid_calibrations = [c for c in calibrations if c.is_valid]

    if len(valid_calibrations) == 0:
        logger.warning("No valid calibrations to compute overall drift")
        return 0.0, 0.0

    drifts = [c.drift_sec for c in valid_calibrations]

    if method == "median":
        overall_drift = float(np.median(drifts))
    else:
        overall_drift = float(np.mean(drifts))

    overall_std = float(np.std(drifts))

    return overall_drift, overall_std


def robust_huber_mean(values: np.ndarray, delta: float = 1.5) -> Tuple[float, float]:
    """Compute Huber mean (robust to outliers).

    Args:
        values: Array of values
        delta: Threshold for outlier detection (in standard deviations)

    Returns:
        (huber_mean, huber_std) tuple
    """
    if len(values) == 0:
        return 0.0, 0.0

    # Iterative Huber estimation
    median = np.median(values)
    mad = np.median(np.abs(values - median))  # Median Absolute Deviation
    sigma = 1.4826 * mad  # Robust scale estimate

    if sigma == 0:
        return float(median), 0.0

    # Weight function
    residuals = (values - median) / sigma
    weights = np.where(np.abs(residuals) <= delta, 1.0, delta / np.abs(residuals))

    # Weighted mean
    huber_mean = np.sum(weights * values) / np.sum(weights)
    huber_std = np.sqrt(np.sum(weights * (values - huber_mean) ** 2) / np.sum(weights))

    return float(huber_mean), float(huber_std)


def compare_drift_methods(
    df: pd.DataFrame,
) -> dict:
    """Compare different drift estimation methods.

    Args:
        df: DataFrame with timestamp and meta_time columns

    Returns:
        Dict with results from different methods
    """
    # Compute drift
    drift_series = (pd.to_datetime(df["meta_time"]) - pd.to_datetime(df["timestamp"])).dt.total_seconds()
    drift_series = drift_series.dropna()

    if len(drift_series) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "huber": 0.0,
            "std": 0.0,
        }

    results = {
        "mean": float(drift_series.mean()),
        "median": float(drift_series.median()),
        "std": float(drift_series.std()),
        "samples": len(drift_series),
    }

    # Huber mean
    huber_mean, huber_std = robust_huber_mean(drift_series.values)
    results["huber_mean"] = huber_mean
    results["huber_std"] = huber_std

    return results
