"""Data loading and preprocessing for telemetry visualization."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from coordinate_transform import TrackCoordinateTransformer
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

    def compute_trajectories(self, transformer: TrackCoordinateTransformer) -> Dict:
        """Compute pixel trajectories for all cars.

        Args:
            transformer: Coordinate transformer (GPS → pixels)

        Returns:
            Dict with trajectory data for each car
        """
        logger.info("Computing trajectories...")

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
            # For each global time point, get the car's data (may be NaN if car not present)
            car_indexed = car_data.set_index('time_global').reindex(unique_times)

            # Extract GPS coordinates
            lats = car_indexed['gps_lat'].values
            lons = car_indexed['gps_lon'].values

            # Transform to pixel coordinates
            x_pixels, y_pixels = transformer.transform(lats, lons)

            # Extract telemetry (for future hover tooltips)
            speed = car_indexed['speed'].values if 'speed' in car_indexed.columns else np.full(len(car_indexed), np.nan)
            gear = car_indexed['gear'].values if 'gear' in car_indexed.columns else np.full(len(car_indexed), np.nan)
            aps = car_indexed['aps'].values if 'aps' in car_indexed.columns else np.full(len(car_indexed), np.nan)
            lap = car_indexed['lap_repaired'].values if 'lap_repaired' in car_indexed.columns else np.full(len(car_indexed), np.nan)

            # Store as lists (JSON-serializable for dcc.Store)
            trajectories[car_id] = {
                'x': x_pixels.tolist(),
                'y': y_pixels.tolist(),
                'speed': speed.tolist(),
                'gear': gear.tolist(),
                'aps': aps.tolist(),
                'lap': lap.tolist(),
                'color': config.CAR_COLORS.get(car_id, '#888888'),
                'car_no': car_indexed['car_no'].iloc[0] if 'car_no' in car_indexed.columns else car_id,
            }

            # Count valid positions (non-NaN)
            valid_count = (~np.isnan(x_pixels)).sum()
            logger.info(f"  {car_id}: {valid_count:,}/{self.frame_count:,} valid positions "
                       f"({valid_count/self.frame_count*100:.1f}%)")

        self.trajectories = trajectories
        return trajectories

    def get_store_data(self) -> Dict:
        """Get data formatted for dcc.Store (JSON-serializable).

        Returns:
            Dict with all trajectory data and metadata
        """
        if not self.trajectories:
            raise ValueError("Trajectories not computed. Call compute_trajectories first.")

        return {
            'trajectories': self.trajectories,
            'frame_count': self.frame_count,
            'time_labels': self.time_labels,
            'car_ids': self.car_ids,
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
    track_image_path: Path,
    car_ids: List[str]
) -> Tuple[Dict, TrackCoordinateTransformer]:
    """Load telemetry data and prepare for visualization.

    Convenience function that handles the full data loading pipeline.

    Args:
        parquet_path: Path to synchronized parquet file
        track_image_path: Path to track image
        car_ids: List of chassis IDs to load

    Returns:
        (store_data, transformer) tuple
    """
    # Load data
    loader = TelemetryDataLoader(parquet_path, car_ids)
    loader.load_parquet()

    # Get GPS bounds from data
    lat_min, lat_max, lon_min, lon_max = loader.get_gps_bounds()

    # Initialize transformer
    transformer = TrackCoordinateTransformer(
        track_image_path,
        gps_bounds=(lat_min, lat_max, lon_min, lon_max),
        padding=config.TRACK_PADDING
    )

    # Set bounds and compute transform
    transformer.set_bounds_from_data(
        loader.df['gps_lat'].values,
        loader.df['gps_lon'].values
    )

    # Compute trajectories
    loader.compute_trajectories(transformer)

    # Get data for dcc.Store
    store_data = loader.get_store_data()

    return store_data, transformer
