"""Simple data loader for ribbon-based race replay."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import config

logger = logging.getLogger(__name__)


def load_ribbons(ribbons_file: Path) -> Dict:
    """Load ribbon polylines from JSON.

    Returns:
        dict with 'meta' and 'ribbons' list
    """
    with open(ribbons_file, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data['ribbons'])} ribbons")
    return data


def load_telemetry_simple(parquet_path: Path, car_ids: List[str]) -> pd.DataFrame:
    """Load telemetry data for specified cars.

    Returns:
        DataFrame with columns: time_global, chassis_id, track_distance_m, ...
    """
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Filter to requested cars
    df = df[df['chassis_id'].isin(car_ids)]
    logger.info(f"Filtered to {len(df):,} rows for cars: {car_ids}")

    return df


def build_gps_to_distance_converter(gps_ref_file: Path):
    """Build a GPS-to-track-distance converter using GPS centerline reference.

    Args:
        gps_ref_file: Path to GPS centerline reference JSON

    Returns:
        Function that converts (lat, lon) to track_distance_m
    """
    # Load GPS centerline reference
    with open(gps_ref_file, 'r') as f:
        gps_ref = json.load(f)

    centerline_lat = np.array(gps_ref['centerline_lat'])
    centerline_lon = np.array(gps_ref['centerline_lon'])
    centerline_distances = np.array(gps_ref['centerline_distances'])
    track_length_m = gps_ref['track_length_m']

    # Build KD-tree on GPS centerline (lat, lon)
    gps_centerline = np.column_stack([centerline_lat, centerline_lon])
    tree = cKDTree(gps_centerline)

    logger.info(f"Built GPS converter: {len(centerline_lat)} centerline points, {track_length_m:.2f}m track length")

    def gps_to_track_distance(lat: float, lon: float) -> Optional[float]:
        """Convert GPS lat/lon to track distance.

        Args:
            lat: GPS latitude (degrees)
            lon: GPS longitude (degrees)

        Returns:
            track_distance_m or None if invalid
        """
        if pd.isna(lat) or pd.isna(lon):
            return None

        # Find nearest centerline point
        _, nearest_idx = tree.query([lat, lon])

        # Return corresponding track distance
        return centerline_distances[nearest_idx]

    return gps_to_track_distance


def map_distance_to_ribbon_xy(
    distance_m: float,
    ribbon_data: Dict,
    track_length_m: float
) -> tuple:
    """Map track_distance_m to (x, y) on a ribbon polyline.

    Simple parametric mapping: distance → ribbon point via linear interpolation.

    Args:
        distance_m: Distance along track (meters)
        ribbon_data: Single ribbon dict with 'xy' list
        track_length_m: Total track length

    Returns:
        (x, y) tuple in meters
    """
    # Normalize distance to [0, 1]
    t = (distance_m % track_length_m) / track_length_m

    # Get ribbon points
    points = np.array(ribbon_data['xy'])
    n_points = len(points)

    # Map t to point index
    idx_float = t * (n_points - 1)
    idx = int(idx_float)
    frac = idx_float - idx

    # Linear interpolation between points
    if idx >= n_points - 1:
        return tuple(points[-1])

    p1 = points[idx]
    p2 = points[idx + 1]

    x = p1[0] * (1 - frac) + p2[0] * frac
    y = p1[1] * (1 - frac) + p2[1] * frac

    return (float(x), float(y))


def prepare_trajectories(
    df: pd.DataFrame,
    ribbons_data: Dict,
    car_ribbon_map: Dict[str, str],
    use_gps_fallback: bool = True
) -> Dict:
    """Prepare trajectory data for visualization.

    Args:
        df: Telemetry dataframe
        ribbons_data: Ribbon polylines
        car_ribbon_map: car_id -> ribbon_name mapping
        use_gps_fallback: If True, use GPS to compute track_distance when missing

    Returns:
        Dict with trajectories for each car
    """
    # Get track length from centerline
    center_ribbon = next(r for r in ribbons_data['ribbons'] if r['name'] == 'center')
    points = np.array(center_ribbon['xy'])
    diffs = np.diff(points, axis=0)
    track_length_m = np.sum(np.linalg.norm(diffs, axis=1))

    logger.info(f"Track length: {track_length_m:.2f}m")

    # Build GPS converter if fallback enabled
    gps_converter = None
    if use_gps_fallback:
        gps_ref_file = Path(__file__).parent / 'assets/gps_centerline_reference.json'
        if gps_ref_file.exists():
            gps_converter = build_gps_to_distance_converter(gps_ref_file)
            logger.info("GPS fallback enabled")
        else:
            logger.warning(f"GPS reference file not found: {gps_ref_file}")

    # Get unique timeline
    unique_times = df['time_global'].sort_values().unique()
    frame_count = len(unique_times)

    logger.info(f"Computing trajectories for {frame_count:,} frames...")

    # Build ribbon lookup
    ribbon_lookup = {r['name']: r for r in ribbons_data['ribbons']}

    trajectories = {}
    car_ids = df['chassis_id'].unique()

    for car_id in car_ids:
        car_data = df[df['chassis_id'] == car_id]

        # Get ribbon assignment
        ribbon_name = car_ribbon_map.get(car_id, 'center')
        if ribbon_name not in ribbon_lookup:
            logger.warning(f"Ribbon '{ribbon_name}' not found for car {car_id}, using center")
            ribbon_name = 'center'

        ribbon = ribbon_lookup[ribbon_name]

        # Index by global time
        car_indexed = car_data.set_index('time_global').reindex(unique_times)

        # Check if car has any positioning data
        has_track_dist = 'track_distance_m' in car_indexed.columns
        has_gps = ('gps_lat' in car_indexed.columns and 'gps_lon' in car_indexed.columns)

        if not has_track_dist and not has_gps:
            logger.warning(f"Car {car_id}: No track_distance_m or GPS data, skipping")
            continue

        # Map to ribbon coordinates with GPS fallback
        x_coords = []
        y_coords = []
        gps_fallback_count = 0

        for idx in range(len(car_indexed)):
            dist = car_indexed['track_distance_m'].iloc[idx] if has_track_dist else np.nan

            # Try GPS fallback if track distance missing
            if pd.isna(dist) and has_gps and gps_converter is not None:
                lat = car_indexed['gps_lat'].iloc[idx]
                lon = car_indexed['gps_lon'].iloc[idx]
                dist = gps_converter(lat, lon)
                if dist is not None:
                    gps_fallback_count += 1

            # Map to XY
            if pd.isna(dist):
                x_coords.append(np.nan)
                y_coords.append(np.nan)
            else:
                x, y = map_distance_to_ribbon_xy(dist, ribbon, track_length_m)
                x_coords.append(x)
                y_coords.append(y)

        x_arr = np.array(x_coords)
        y_arr = np.array(y_coords)

        # Count valid positions
        valid_count = (~np.isnan(x_arr)).sum()
        logger.info(f"  {car_id}: {valid_count:,}/{frame_count:,} valid positions ({valid_count/frame_count*100:.1f}%) on {ribbon_name}")
        if gps_fallback_count > 0:
            logger.info(f"    - GPS fallback used for {gps_fallback_count:,} frames ({gps_fallback_count/frame_count*100:.1f}%)")

        trajectories[car_id] = {
            'x': x_arr.tolist(),
            'y': y_arr.tolist(),
            'ribbon': ribbon_name,
            'color': config.CAR_COLORS.get(car_id, '#888888'),
            'car_no': car_indexed['car_no'].iloc[0] if 'car_no' in car_indexed.columns else car_id,
        }

    return {
        'trajectories': trajectories,
        'frame_count': frame_count,
        'car_ids': list(trajectories.keys()),
    }
