"""Coordinate transformation utilities for GPS to pixel mapping."""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TrackCoordinateTransformer:
    """Transform GPS (lat/lon) to track image pixel coordinates.

    Uses affine transformation to map GPS coordinates to pixel space
    while preserving aspect ratio and handling track rotation.
    """

    def __init__(
        self,
        track_image_path: Path,
        gps_bounds: Optional[Tuple[float, float, float, float]] = None,
        padding: float = 0.05
    ):
        """Initialize coordinate transformer.

        Args:
            track_image_path: Path to track background image
            gps_bounds: Optional (lat_min, lat_max, lon_min, lon_max) bounds.
                       If None, will be set later from data.
            padding: Fraction of image to use as padding (default 0.05 = 5%)
        """
        self.track_image_path = track_image_path
        self.padding = padding

        # Load image dimensions
        if not track_image_path.exists():
            raise FileNotFoundError(f"Track image not found: {track_image_path}")

        with Image.open(track_image_path) as img:
            self.img_width, self.img_height = img.size

        logger.info(f"Track image: {self.img_width}x{self.img_height} pixels")

        # GPS bounds (will be set from data if not provided)
        self.gps_bounds = gps_bounds
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = (
            gps_bounds if gps_bounds else (None, None, None, None)
        )

        # Transformation parameters (computed after bounds are set)
        self.scale_x = None
        self.scale_y = None
        self.offset_x = None
        self.offset_y = None
        self.transform_ready = False

    def set_bounds_from_data(self, lats: np.ndarray, lons: np.ndarray):
        """Set GPS bounds from actual data.

        Args:
            lats: Array of latitude values
            lons: Array of longitude values
        """
        # Filter out NaN values
        valid_lats = lats[~np.isnan(lats)]
        valid_lons = lons[~np.isnan(lons)]

        if len(valid_lats) == 0 or len(valid_lons) == 0:
            raise ValueError("No valid GPS coordinates found in data")

        self.lat_min = float(np.min(valid_lats))
        self.lat_max = float(np.max(valid_lats))
        self.lon_min = float(np.min(valid_lons))
        self.lon_max = float(np.max(valid_lons))

        logger.info(f"GPS bounds: lat=[{self.lat_min:.6f}, {self.lat_max:.6f}], "
                   f"lon=[{self.lon_min:.6f}, {self.lon_max:.6f}]")

        # Compute transformation
        self._compute_transform()

    def _compute_transform(self):
        """Compute affine transformation parameters.

        Maps GPS rectangle to pixel rectangle while preserving aspect ratio.
        """
        if None in (self.lat_min, self.lat_max, self.lon_min, self.lon_max):
            raise ValueError("GPS bounds not set. Call set_bounds_from_data first.")

        # GPS data range
        lat_range = self.lat_max - self.lat_min
        lon_range = self.lon_max - self.lon_min

        if lat_range == 0 or lon_range == 0:
            raise ValueError("GPS range is zero - cannot compute transformation")

        # Available pixel space (with padding)
        padding_px_x = self.img_width * self.padding
        padding_px_y = self.img_height * self.padding
        available_width = self.img_width - 2 * padding_px_x
        available_height = self.img_height - 2 * padding_px_y

        # Compute scale to fit GPS data into image while preserving aspect ratio
        # Note: Longitude (x-axis), Latitude (y-axis in GPS, but y increases downward in images)
        scale_x = available_width / lon_range
        scale_y = available_height / lat_range

        # Use the smaller scale to ensure everything fits
        scale = min(scale_x, scale_y)

        self.scale_x = scale
        self.scale_y = -scale  # Negative because image y increases downward, GPS lat increases upward

        # Center the data in the image
        scaled_width = lon_range * scale
        scaled_height = lat_range * scale

        self.offset_x = padding_px_x + (available_width - scaled_width) / 2 - self.lon_min * scale
        self.offset_y = self.img_height - padding_px_y - (available_height - scaled_height) / 2 + self.lat_min * scale

        self.transform_ready = True

        logger.info(f"Transform computed: scale=({self.scale_x:.2f}, {self.scale_y:.2f}), "
                   f"offset=({self.offset_x:.2f}, {self.offset_y:.2f})")

    def transform(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform GPS coordinates to pixel coordinates.

        Args:
            lat: Latitude values (numpy array or scalar)
            lon: Longitude values (numpy array or scalar)

        Returns:
            (x, y) pixel coordinates as numpy arrays
        """
        if not self.transform_ready:
            raise ValueError("Transform not ready. Call set_bounds_from_data first.")

        x = lon * self.scale_x + self.offset_x
        y = lat * self.scale_y + self.offset_y

        return x, y

    def inverse_transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform pixel coordinates back to GPS coordinates.

        Useful for debugging and interactive features.

        Args:
            x: X pixel coordinates
            y: Y pixel coordinates

        Returns:
            (lat, lon) GPS coordinates
        """
        if not self.transform_ready:
            raise ValueError("Transform not ready. Call set_bounds_from_data first.")

        lon = (x - self.offset_x) / self.scale_x
        lat = (y - self.offset_y) / self.scale_y

        return lat, lon

    def get_image_dimensions(self) -> Tuple[int, int]:
        """Get track image dimensions.

        Returns:
            (width, height) in pixels
        """
        return self.img_width, self.img_height

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get GPS bounds.

        Returns:
            (lat_min, lat_max, lon_min, lon_max)
        """
        return self.lat_min, self.lat_max, self.lon_min, self.lon_max
