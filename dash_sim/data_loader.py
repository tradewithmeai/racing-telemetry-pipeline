"""Data loading and preprocessing for telemetry visualization."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from centerline_transform import CenterlineTransformer
from track_processing import Track
import config

logger = logging.getLogger(__name__)


class TelemetryDataLoader:
    """Load and preprocess telemetry data for visualization."""

    def __init__(self, parquet_path: Path, car_ids: List[str]):
        """Initialize data loader.

        Args:
            parquet_path: Path to synchronized multi-car parquet file
            car_ids: List of chassis IDs to load
        """
        self.parquet_path = parquet_path
        self.car_ids = car_ids
        self.df = None
        self.transformer = None
        self.trajectories = {}
        self.frame_count = 0
        self.time_labels = []

    def load_parquet(self) -> pd.DataFrame:
        """Load parquet file and filter to selected cars.

        Returns:
            DataFrame with telemetry data
        """
        logger.info(f"Loading parquet: {self.parquet_path}")

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        # Load data
        df = pd.read_parquet(self.parquet_path)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

        # Validate required columns
        required_cols = ['time_global', 'chassis_id', 'gps_lat', 'gps_lon']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Filter to selected cars
        df = df[df['chassis_id'].isin(self.car_ids)].copy()
        logger.info(f"Filtered to {len(df):,} rows for cars: {self.car_ids}")

        if len(df) == 0:
            raise ValueError(f"No data found for cars: {self.car_ids}")

        # Sort by time
        df = df.sort_values('time_global').reset_index(drop=True)

        # Convert time_global to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['time_global']):
            df['time_global'] = pd.to_datetime(df['time_global'])

        self.df = df
        return df

    def compute_trajectories(self, transformer: CenterlineTransformer, track: Track) -> Dict:
        """Compute centerline-based trajectories for all cars.

        Uses track_distance_m from telemetry for accurate positioning.

        Args:
            transformer: Centerline coordinate transformer
            track: Track object with ribbon assignments

        Returns:
            Dict with trajectory data for each car
        """
        logger.info("Computing trajectories using centerline...")

        self.transformer = transformer
        trajectories = {}

        # Get unique time points (global timeline)
        unique_times = self.df['time_global'].unique()
        self.frame_count = len(unique_times)
        self.time_labels = [pd.Timestamp(t).isoformat() for t in unique_times]

        logger.info(f"Total frames: {self.frame_count:,}")

        # Process each car
        for car_id in self.car_ids:
            car_data = self.df[self.df['chassis_id'] == car_id].copy()

            if len(car_data) == 0:
                logger.warning(f"No data for car {car_id}")
                continue

            # Create frame index aligned with global timeline
            car_indexed = car_data.set_index('time_global').reindex(unique_times)

            # Get ribbon assignment for this car
            ribbon_name = track.get_ribbon_for_car(car_id)
            logger.debug(f"  Car {car_id} assigned to ribbon: {ribbon_name}")

            # Check if this car has any position data at all
            has_track_dist = 'track_distance_m' in car_indexed.columns and not car_indexed['track_distance_m'].isna().all()
            has_gps = ('gps_lat' in car_indexed.columns and 'gps_lon' in car_indexed.columns and
                       not (car_indexed['gps_lat'].isna().all() or car_indexed['gps_lon'].isna().all()))

            if not has_track_dist and not has_gps:
                logger.warning(f"  Car {car_id}: No position data available (skipping)")
                continue

            # Use hybrid approach: prefer track_distance_m, fallback to GPS per-row
            x_coords = []
            y_coords = []

            # Get both data sources if available
            distances = car_indexed['track_distance_m'].values if has_track_dist else [np.nan] * len(car_indexed)
            lats = car_indexed['gps_lat'].values if has_gps else [np.nan] * len(car_indexed)
            lons = car_indexed['gps_lon'].values if has_gps else [np.nan] * len(car_indexed)

            # Process each frame, using track_distance if available, GPS otherwise
            used_track_dist = 0
            used_gps = 0

            for dist, lat, lon in zip(distances, lats, lons):
                # Prefer track_distance_m (more accurate)
                if not pd.isna(dist):
                    x, y = transformer.distance_to_xy(dist, ribbon_name)
                    x_coords.append(x)
                    y_coords.append(y)
                    used_track_dist += 1
                # Fallback to GPS
                elif not pd.isna(lat) and not pd.isna(lon):
                    x, y = transformer.gps_to_xy(lat, lon, ribbon_name)
                    x_coords.append(x)
                    y_coords.append(y)
                    used_gps += 1
                # No position data for this frame
                else:
                    x_coords.append(np.nan)
                    y_coords.append(np.nan)

            x_meters = np.array(x_coords)
            y_meters = np.array(y_coords)

            logger.info(f"  Car {car_id}: Used track_distance for {used_track_dist} frames, GPS for {used_gps} frames")

            # Extract telemetry (for future hover tooltips)
            # Prefer speed_final (derived + interpolated), fallback to speed (raw)
            if 'speed_final' in car_indexed.columns:
                speed = car_indexed['speed_final'].values
            elif 'speed' in car_indexed.columns:
                speed = car_indexed['speed'].values
            else:
                speed = np.full(len(car_indexed), np.nan)

            gear = car_indexed['gear'].values if 'gear' in car_indexed.columns else np.full(len(car_indexed), np.nan)
            aps = car_indexed['aps'].values if 'aps' in car_indexed.columns else np.full(len(car_indexed), np.nan)
            lap = car_indexed['lap_repaired'].values if 'lap_repaired' in car_indexed.columns else np.full(len(car_indexed), np.nan)

            # Store as lists (JSON-serializable for dcc.Store)
            trajectories[car_id] = {
                'x': x_meters.tolist(),
                'y': y_meters.tolist(),
                'speed': speed.tolist(),
                'gear': gear.tolist(),
                'aps': aps.tolist(),
                'lap': lap.tolist(),
                'color': config.CAR_COLORS.get(car_id, '#888888'),
                'car_no': car_indexed['car_no'].iloc[0] if 'car_no' in car_indexed.columns else car_id,
                'ribbon': ribbon_name,
            }

            # Count valid positions (non-NaN)
            valid_count = (~np.isnan(x_meters)).sum()
            logger.info(f"  {car_id}: {valid_count:,}/{self.frame_count:,} valid positions "
                       f"({valid_count/self.frame_count*100:.1f}%) on {ribbon_name}")

        self.trajectories = trajectories
        return trajectories

    def get_store_data(self) -> Dict:
        """Get data formatted for dcc.Store (JSON-serializable).

        Returns:
            Dict with all trajectory data and metadata
        """
        if not self.trajectories:
            raise ValueError("Trajectories not computed. Call compute_trajectories first.")

        # Only return cars that have valid trajectories (some may be skipped)
        valid_car_ids = [car_id for car_id in self.car_ids if car_id in self.trajectories]

        return {
            'trajectories': self.trajectories,
            'frame_count': self.frame_count,
            'time_labels': self.time_labels,
            'car_ids': valid_car_ids,
        }

    def get_gps_bounds(self) -> Tuple[float, float, float, float]:
        """Get GPS bounds from loaded data.

        Returns:
            (lat_min, lat_max, lon_min, lon_max)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_parquet first.")

        # Get all valid GPS coordinates
        lats = self.df['gps_lat'].dropna()
        lons = self.df['gps_lon'].dropna()

        if len(lats) == 0 or len(lons) == 0:
            raise ValueError("No valid GPS coordinates in data")

        return (
            float(lats.min()),
            float(lats.max()),
            float(lons.min()),
            float(lons.max())
        )


def load_and_prepare_data(
    parquet_path: Path,
    track_name: str,
    car_ids: List[str]
) -> Tuple[Dict, CenterlineTransformer, Track]:
    """Load telemetry data and prepare for visualization using centerline.

    Convenience function that handles the full data loading pipeline.

    Args:
        parquet_path: Path to synchronized parquet file
        track_name: Name of track (e.g., 'barber')
        car_ids: List of chassis IDs to load

    Returns:
        (store_data, transformer, track) tuple
    """
    # Load track
    track_dir = config.TRACKS_DIR / track_name
    track = Track(track_dir)

    # Ensure track is processed (auto-generate if missing)
    if track.ensure_processed():
        logger.warning(f"Track '{track_name}' was not processed, auto-generated files")

    # Load track geometry
    track.load_geometry()

    # Load telemetry data
    loader = TelemetryDataLoader(parquet_path, car_ids)
    loader.load_parquet()

    # Get GPS bounds from telemetry data for calibration
    lat_min, lat_max, lon_min, lon_max = loader.get_gps_bounds()
    logger.info(f"GPS bounds from telemetry: lat=[{lat_min:.6f}, {lat_max:.6f}], lon=[{lon_min:.6f}, {lon_max:.6f}]")

    # Initialize centerline transformer with GPS calibration
    transformer = CenterlineTransformer(track, gps_bounds=(lat_min, lat_max, lon_min, lon_max))

    # Compute trajectories using centerline
    loader.compute_trajectories(transformer, track)

    # Get data for dcc.Store
    store_data = loader.get_store_data()

    return store_data, transformer, track
