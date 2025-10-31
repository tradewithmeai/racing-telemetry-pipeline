"""Position data normalization and validation."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import yaml

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GPSBounds:
    """GPS coordinate bounds for a circuit."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass
class PositionQuality:
    """Position data quality metrics."""

    chassis_id: str
    total_rows: int
    gps_coverage_pct: float
    lapdist_coverage_pct: float
    outliers_detected: int
    out_of_bounds: int
    interpolated_points: int
    quality_score: float  # 0-1


def load_circuit_params(circuit_name: str) -> dict:
    """Load circuit parameters from YAML config.

    Args:
        circuit_name: Name of circuit (e.g., 'barber')

    Returns:
        Dict with circuit parameters
    """
    config_path = Path(__file__).parent.parent / "conf" / "circuit_params.yaml"

    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    if circuit_name not in params:
        raise ValueError(f"Circuit '{circuit_name}' not found in circuit_params.yaml")

    return params[circuit_name]


def convert_gps_minutes_to_degrees(
    lat_minutes: pd.Series, lon_minutes: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """Convert GPS coordinates from VBOX DDMM.MMMM format to decimal degrees.

    VBOX GPS data is stored as DDMM.MMMM:
    - Example: 3352.9380 = 33 degrees + 52.9380 minutes
    - Convert to: 33 + (52.9380 / 60) = 33.882 degrees

    For negative (Western/Southern):
    - Example: -8637.1820 = -86 degrees - 37.1820 minutes
    - Convert to: -86 - (37.1820 / 60) = -86.620 degrees

    Args:
        lat_minutes: Latitude in VBOX format (DDMM.MMMM)
        lon_minutes: Longitude in VBOX format (DDMM.MMMM or -DDMM.MMMM)

    Returns:
        (lat_degrees, lon_degrees) tuple in decimal degrees
    """
    # Extract degrees (integer part divided by 100)
    lat_deg = np.floor(lat_minutes / 100.0)
    lon_deg = np.floor(lon_minutes / 100.0)

    # Extract minutes (remainder after removing degrees)
    lat_min = lat_minutes - (lat_deg * 100.0)
    lon_min = lon_minutes - (lon_deg * 100.0)

    # Convert to decimal degrees: DD + (MM.MMMM / 60)
    lat_degrees = lat_deg + (lat_min / 60.0)
    lon_degrees = lon_deg + (lon_min / 60.0)

    return lat_degrees, lon_degrees


def validate_gps_bounds(
    lat: pd.Series,
    lon: pd.Series,
    bounds: GPSBounds,
    tolerance: float = 0.01,
) -> Tuple[pd.Series, pd.Series]:
    """Validate GPS coordinates against track bounds.

    Args:
        lat: Latitude series (degrees)
        lon: Longitude series (degrees)
        bounds: GPS bounds for circuit
        tolerance: Additional tolerance beyond bounds (degrees)

    Returns:
        (is_valid_lat, is_valid_lon) boolean series
    """
    # Check latitude bounds
    is_valid_lat = (lat >= bounds.lat_min - tolerance) & (
        lat <= bounds.lat_max + tolerance
    )

    # Check longitude bounds
    is_valid_lon = (lon >= bounds.lon_min - tolerance) & (
        lon <= bounds.lon_max + tolerance
    )

    return is_valid_lat, is_valid_lon


def detect_gps_outliers(
    lat: pd.Series,
    lon: pd.Series,
    max_jump_meters: float = 500.0,
) -> pd.Series:
    """Detect GPS outliers based on jump distance.

    Args:
        lat: Latitude series (degrees)
        lon: Longitude series (degrees)
        max_jump_meters: Maximum allowed jump distance between consecutive points

    Returns:
        Boolean series where True indicates outlier
    """
    # Compute distances between consecutive points
    # Using haversine formula approximation for small distances
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    dlat = lat_rad.diff()
    dlon = lon_rad.diff()

    # Simplified distance calculation (good for small distances)
    # Distance in meters ≈ sqrt((dlat * 111000)^2 + (dlon * 111000 * cos(lat))^2)
    lat_m = dlat * 111000  # 1 degree latitude ≈ 111 km
    lon_m = dlon * 111000 * np.cos(lat_rad)  # Adjust for latitude

    distance_m = np.sqrt(lat_m**2 + lon_m**2)

    # Flag points with jumps > threshold
    is_outlier = distance_m > max_jump_meters

    # First point is never an outlier (no previous point to compare)
    is_outlier.iloc[0] = False

    return is_outlier


def interpolate_position(
    df: pd.DataFrame,
    time_col: str = "time_corrected",
    lat_col: str = "gps_lat",
    lon_col: str = "gps_lon",
    method: str = "linear",
    max_gap_sec: float = 2.0,
) -> pd.DataFrame:
    """Interpolate sparse GPS position data.

    Args:
        df: DataFrame with time and position columns
        time_col: Name of time column
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        method: Interpolation method ('linear', 'cubic', 'nearest')
        max_gap_sec: Maximum gap to interpolate across (seconds)

    Returns:
        DataFrame with interpolated position columns
    """
    df = df.copy()

    # Ensure sorted by time
    df = df.sort_values(time_col).reset_index(drop=True)

    # Compute time gaps
    time_diff = df[time_col].diff().dt.total_seconds()

    # Mark gaps > threshold
    large_gaps = time_diff > max_gap_sec

    # Interpolate latitude and longitude
    # But only within segments (don't interpolate across large gaps)
    for i, has_gap in enumerate(large_gaps):
        if i == 0:
            continue

        if has_gap:
            # Insert NaN to prevent interpolation across gap
            df.loc[i, lat_col] = np.nan
            df.loc[i, lon_col] = np.nan

    # Perform interpolation
    df[lat_col] = df[lat_col].interpolate(method=method, limit_direction="both")
    df[lon_col] = df[lon_col].interpolate(method=method, limit_direction="both")

    return df


def normalize_position_data(
    df: pd.DataFrame,
    chassis_id: str,
    circuit: str = "barber",
    max_jump_meters: float = 500.0,
    interpolate: bool = True,
) -> Tuple[pd.DataFrame, PositionQuality]:
    """Complete position normalization pipeline.

    Args:
        df: DataFrame with VBOX_Lat_Min, VBOX_Long_Minutes, Laptrigger_lapdist_dls
        chassis_id: Vehicle identifier
        circuit: Circuit name for bounds lookup
        max_jump_meters: Maximum allowed GPS jump
        interpolate: Whether to interpolate sparse GPS data

    Returns:
        (df_normalized, position_quality) tuple
    """
    logger.info(f"{'='*60}")
    logger.info(f"Position Normalization: {chassis_id}")
    logger.info(f"{'='*60}")

    df = df.copy()
    total_rows = len(df)

    # Load circuit parameters
    circuit_params = load_circuit_params(circuit)
    gps_bounds = GPSBounds(**circuit_params["gps_bounds"])

    logger.info(f"  Circuit: {circuit_params['full_name']}")
    logger.info(
        f"  GPS Bounds: Lat [{gps_bounds.lat_min}, {gps_bounds.lat_max}], "
        f"Lon [{gps_bounds.lon_min}, {gps_bounds.lon_max}]"
    )

    # Step 1: Extract GPS data (already in decimal degrees despite misleading column names)
    if "VBOX_Lat_Min" in df.columns and "VBOX_Long_Minutes" in df.columns:
        logger.info("\n  [Step 1] Extracting GPS data (already in decimal degrees)...")

        # Preserve raw values
        df["gps_lat_raw"] = df["VBOX_Lat_Min"]
        df["gps_lon_raw"] = df["VBOX_Long_Minutes"]

        # GPS data is already in decimal degrees (column names are misleading)
        df["gps_lat"] = df["VBOX_Lat_Min"]
        df["gps_lon"] = df["VBOX_Long_Minutes"]

        gps_coverage = df["gps_lat"].notna().sum() / total_rows * 100
        logger.info(f"    GPS coverage: {gps_coverage:.1f}%")
        logger.info(
            f"    Lat range: {df['gps_lat'].min():.6f} to {df['gps_lat'].max():.6f}"
        )
        logger.info(
            f"    Lon range: {df['gps_lon'].min():.6f} to {df['gps_lon'].max():.6f}"
        )
    else:
        logger.warning("  No GPS columns found (VBOX_Lat_Min, VBOX_Long_Minutes)")
        df["gps_lat"] = np.nan
        df["gps_lon"] = np.nan
        df["gps_lat_raw"] = np.nan
        df["gps_lon_raw"] = np.nan
        gps_coverage = 0.0

    # Step 2: Validate GPS bounds
    if gps_coverage > 0:
        logger.info("\n  [Step 2] Validating GPS bounds...")

        is_valid_lat, is_valid_lon = validate_gps_bounds(
            df["gps_lat"], df["gps_lon"], gps_bounds
        )

        out_of_bounds_lat = (~is_valid_lat & df["gps_lat"].notna()).sum()
        out_of_bounds_lon = (~is_valid_lon & df["gps_lon"].notna()).sum()
        out_of_bounds_total = (
            (~is_valid_lat | ~is_valid_lon) & df["gps_lat"].notna()
        ).sum()

        if out_of_bounds_total > 0:
            logger.warning(
                f"    Out of bounds: {out_of_bounds_total} points ({out_of_bounds_total / df['gps_lat'].notna().sum() * 100:.1f}%)"
            )
            logger.warning(f"      Latitude: {out_of_bounds_lat} points")
            logger.warning(f"      Longitude: {out_of_bounds_lon} points")

            # Mark out-of-bounds points as NaN
            df.loc[~is_valid_lat, "gps_lat"] = np.nan
            df.loc[~is_valid_lon, "gps_lon"] = np.nan
        else:
            logger.info(f"    All GPS points within bounds")

        out_of_bounds = out_of_bounds_total
    else:
        out_of_bounds = 0

    # Step 3: Detect outliers
    if gps_coverage > 0:
        logger.info("\n  [Step 3] Detecting GPS outliers...")

        is_outlier = detect_gps_outliers(
            df["gps_lat"], df["gps_lon"], max_jump_meters=max_jump_meters
        )

        outliers_detected = is_outlier.sum()

        if outliers_detected > 0:
            logger.warning(
                f"    Outliers detected: {outliers_detected} ({outliers_detected / total_rows * 100:.2f}%)"
            )

            # Mark outliers as NaN
            df.loc[is_outlier, "gps_lat"] = np.nan
            df.loc[is_outlier, "gps_lon"] = np.nan
        else:
            logger.info(f"    No outliers detected (jump threshold: {max_jump_meters}m)")
    else:
        outliers_detected = 0

    # Step 4: Interpolate sparse GPS data
    interpolated_points = 0
    if gps_coverage > 0 and interpolate:
        logger.info("\n  [Step 4] Interpolating sparse GPS data...")

        gps_before = df["gps_lat"].notna().sum()

        df = interpolate_position(
            df,
            time_col="time_corrected",
            lat_col="gps_lat",
            lon_col="gps_lon",
            method="linear",
            max_gap_sec=2.0,
        )

        gps_after = df["gps_lat"].notna().sum()
        interpolated_points = gps_after - gps_before

        if interpolated_points > 0:
            logger.info(
                f"    Interpolated {interpolated_points} points ({interpolated_points / total_rows * 100:.1f}%)"
            )
            logger.info(
                f"    GPS coverage after interpolation: {gps_after / total_rows * 100:.1f}%"
            )
        else:
            logger.info("    No interpolation needed (full coverage)")

    # Step 5: Normalize track distance
    if "Laptrigger_lapdist_dls" in df.columns:
        logger.info("\n  [Step 5] Normalizing track distance...")

        # Preserve raw value
        df["track_distance_raw"] = df["Laptrigger_lapdist_dls"]

        # Normalize to meters (already in meters, just rename)
        df["track_distance_m"] = df["Laptrigger_lapdist_dls"]

        lapdist_coverage = df["track_distance_m"].notna().sum() / total_rows * 100
        logger.info(f"    Track distance coverage: {lapdist_coverage:.1f}%")

        if lapdist_coverage > 0:
            logger.info(
                f"    Range: {df['track_distance_m'].min():.1f}m to {df['track_distance_m'].max():.1f}m"
            )
    else:
        logger.warning("  No track distance column found (Laptrigger_lapdist_dls)")
        df["track_distance_m"] = np.nan
        df["track_distance_raw"] = np.nan
        lapdist_coverage = 0.0

    # Step 6: Compute quality score
    logger.info("\n  [Step 6] Computing position quality score...")

    quality_score = compute_position_quality_score(
        gps_coverage=gps_coverage,
        lapdist_coverage=lapdist_coverage,
        outliers_pct=outliers_detected / total_rows * 100 if total_rows > 0 else 0,
        out_of_bounds_pct=out_of_bounds / total_rows * 100 if total_rows > 0 else 0,
    )

    logger.info(f"    Quality score: {quality_score:.2f} / 1.00")

    position_quality = PositionQuality(
        chassis_id=chassis_id,
        total_rows=total_rows,
        gps_coverage_pct=gps_coverage,
        lapdist_coverage_pct=lapdist_coverage,
        outliers_detected=outliers_detected,
        out_of_bounds=out_of_bounds,
        interpolated_points=interpolated_points,
        quality_score=quality_score,
    )

    logger.info(f"{'='*60}\n")

    return df, position_quality


def compute_position_quality_score(
    gps_coverage: float,
    lapdist_coverage: float,
    outliers_pct: float,
    out_of_bounds_pct: float,
) -> float:
    """Compute overall position quality score (0-1).

    Args:
        gps_coverage: GPS coverage percentage
        lapdist_coverage: Track distance coverage percentage
        outliers_pct: Outlier percentage
        out_of_bounds_pct: Out of bounds percentage

    Returns:
        Quality score (0-1, higher is better)
    """
    # Weighted scoring:
    # - GPS coverage: 40%
    # - Track distance coverage: 30%
    # - Outlier penalty: 15%
    # - Out of bounds penalty: 15%

    gps_score = min(gps_coverage / 100.0, 1.0) * 0.4
    lapdist_score = min(lapdist_coverage / 100.0, 1.0) * 0.3
    outlier_penalty = min(outliers_pct / 10.0, 1.0) * 0.15  # Max 10% outliers
    bounds_penalty = min(out_of_bounds_pct / 5.0, 1.0) * 0.15  # Max 5% out of bounds

    quality_score = gps_score + lapdist_score - outlier_penalty - bounds_penalty

    # Clamp to [0, 1]
    quality_score = max(0.0, min(1.0, quality_score))

    return quality_score


def save_position_quality_report(
    quality_by_car: Dict[str, PositionQuality],
    output_path: Path,
    event_name: str,
) -> None:
    """Save position quality report to Parquet.

    Args:
        quality_by_car: Dict mapping chassis_id -> PositionQuality
        output_path: Output directory
        event_name: Event identifier
    """
    logger.info(f"Saving position quality report to {output_path}")

    all_quality = []
    for chassis_id, quality in quality_by_car.items():
        all_quality.append(
            {
                "chassis_id": quality.chassis_id,
                "total_rows": quality.total_rows,
                "gps_coverage_pct": quality.gps_coverage_pct,
                "lapdist_coverage_pct": quality.lapdist_coverage_pct,
                "outliers_detected": quality.outliers_detected,
                "out_of_bounds": quality.out_of_bounds,
                "interpolated_points": quality.interpolated_points,
                "quality_score": quality.quality_score,
            }
        )

    if len(all_quality) == 0:
        logger.warning("No position quality data to save")
        return

    df = pd.DataFrame(all_quality)

    # Save to Parquet
    output_file = output_path / event_name / "position_quality.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, compression="snappy", index=False)

    logger.info(f"  Saved quality report for {len(all_quality)} vehicles")
    logger.info(f"  Output: {output_file}")

    # Log summary statistics
    logger.info("\n  Quality Summary:")
    logger.info(f"    Mean GPS coverage: {df['gps_coverage_pct'].mean():.1f}%")
    logger.info(
        f"    Mean lapdist coverage: {df['lapdist_coverage_pct'].mean():.1f}%"
    )
    logger.info(f"    Mean quality score: {df['quality_score'].mean():.2f}")
    logger.info(f"    Total outliers: {df['outliers_detected'].sum()}")
    logger.info(f"    Total out of bounds: {df['out_of_bounds'].sum()}")
