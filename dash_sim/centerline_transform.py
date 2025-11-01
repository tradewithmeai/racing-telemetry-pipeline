"""Centerline-based coordinate transformation for track positioning.

Transforms telemetry data (GPS or track_distance_m) to centerline coordinates
with ribbon offset support for visual car separation.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


class CenterlineTransformer:
    """Transform GPS/distance to centerline coordinates with ribbon offsets.

    This replaces the old GPS→pixel affine transformation with a more accurate
    centerline-based positioning system.
    """

    def __init__(self, track, gps_bounds: Optional[Tuple[float, float, float, float]] = None):
        """Initialize transformer from Track object.

        Args:
            track: Track object with loaded geometry
            gps_bounds: (lat_min, lat_max, lon_min, lon_max) for GPS calibration
        """
        self.track = track
        track.load_geometry()  # Ensure geometry is loaded

        self.centerline = track.centerline  # Nx2 array (x, y) in meters
        self.ribbons = track.ribbons
        self.ribbon_dict = {r['name']: np.array(r['xy']) for r in track.ribbons['ribbons']}

        # Compute cumulative distance along centerline
        self._build_distance_map()

        # Build KDTree for GPS projection (if needed)
        self._build_kdtree()

        # GPS calibration (optional)
        self.gps_calibrated = False
        if gps_bounds is not None:
            self.calibrate_gps(gps_bounds)

        logger.info(f"CenterlineTransformer initialized:")
        logger.info(f"  Centerline: {len(self.centerline)} points")
        logger.info(f"  Total track length: {self.total_length:.2f}m")
        logger.info(f"  Ribbons available: {list(self.ribbon_dict.keys())}")
        logger.info(f"  GPS calibrated: {self.gps_calibrated}")

    def _build_distance_map(self):
        """Compute cumulative distance along centerline for parametric positioning."""
        # Compute distances between consecutive points
        diff = np.diff(self.centerline, axis=0)
        segment_lengths = np.linalg.norm(diff, axis=1)

        # Cumulative distance (arc length parameter)
        self.cumulative_distance = np.concatenate([[0], np.cumsum(segment_lengths)])
        self.total_length = self.cumulative_distance[-1]

        logger.debug(f"Distance map built: {self.total_length:.2f}m total")

    def _build_kdtree(self):
        """Build KDTree for fast GPS nearest-neighbor search."""
        self.kdtree = KDTree(self.centerline)
        logger.debug(f"KDTree built with {len(self.centerline)} points")

    def distance_to_xy(
        self,
        distance_m: float,
        ribbon_name: str = 'center'
    ) -> Tuple[float, float]:
        """Map track_distance_m to (x, y) on specified ribbon.

        Args:
            distance_m: Distance along track in meters
            ribbon_name: Name of ribbon (e.g., 'center', 'left_1.37m')

        Returns:
            (x, y) coordinates in meters
        """
        # Handle wrap-around for lap distance
        distance_normalized = distance_m % self.total_length

        # Find interpolation indices
        idx = np.searchsorted(self.cumulative_distance, distance_normalized)

        if idx == 0:
            return tuple(self.centerline[0])
        elif idx >= len(self.centerline):
            return tuple(self.centerline[-1])

        # Linear interpolation between points
        t = (distance_normalized - self.cumulative_distance[idx-1]) / (
            self.cumulative_distance[idx] - self.cumulative_distance[idx-1]
        )

        point_on_centerline = (
            self.centerline[idx-1] * (1 - t) +
            self.centerline[idx] * t
        )

        # If using a ribbon offset, get position on that ribbon instead
        if ribbon_name != 'center' and ribbon_name in self.ribbon_dict:
            # For now, use centerline position (ribbon offset would require
            # more complex parametric mapping). This is a simplified version.
            # Full implementation would map distance → ribbon point with normals
            return tuple(point_on_centerline)

        return tuple(point_on_centerline)

    def calibrate_gps(self, gps_bounds: Tuple[float, float, float, float]):
        """Calibrate GPS→centerline transformation from actual GPS data.

        Creates a simple affine transform that maps GPS bounds to centerline bounds.

        Args:
            gps_bounds: (lat_min, lat_max, lon_min, lon_max) from telemetry data
        """
        lat_min, lat_max, lon_min, lon_max = gps_bounds

        # Get centerline bounds
        x_min = self.centerline[:, 0].min()
        x_max = self.centerline[:, 0].max()
        y_min = self.centerline[:, 1].min()
        y_max = self.centerline[:, 1].max()

        # Compute GPS reference point (center of GPS bounds)
        self._ref_lat = (lat_min + lat_max) / 2
        self._ref_lon = (lon_min + lon_max) / 2

        # Compute scale factors (meters per degree)
        # Longitude scale depends on latitude
        self._lon_scale = 111320.0 * np.cos(np.radians(self._ref_lat))
        self._lat_scale = 111320.0

        # Compute centerline reference (center of centerline bounds)
        self._ref_x = (x_min + x_max) / 2
        self._ref_y = (y_min + y_max) / 2

        # Compute GPS→meters transform offsets to align centers
        # When GPS is at reference, result should be centerline reference
        self.gps_calibrated = True

        logger.info(f"GPS calibration complete:")
        logger.info(f"  GPS reference: ({self._ref_lat:.6f}, {self._ref_lon:.6f})")
        logger.info(f"  Centerline reference: ({self._ref_x:.2f}, {self._ref_y:.2f})m")
        logger.info(f"  Scales: lon={self._lon_scale:.2f} m/deg, lat={self._lat_scale:.2f} m/deg")

    def gps_to_xy(
        self,
        lat: float,
        lon: float,
        ribbon_name: str = 'center'
    ) -> Tuple[float, float]:
        """Map GPS coordinates to nearest centerline point.

        Args:
            lat: Latitude
            lon: Longitude
            ribbon_name: Name of ribbon (currently uses centerline for projection)

        Returns:
            (x, y) coordinates in meters

        Note:
            If GPS is not calibrated, uses simple projection.
            For best accuracy, call calibrate_gps() first with telemetry GPS bounds.
        """
        if not self.gps_calibrated:
            logger.debug("GPS not calibrated, using simple projection")
            # Use simple equirectangular projection centered on Barber
            self._ref_lat = 33.533
            self._ref_lon = -86.619
            self._lon_scale = 111320.0 * np.cos(np.radians(self._ref_lat))
            self._lat_scale = 111320.0
            self._ref_x = 0.0
            self._ref_y = 0.0

        # Convert GPS to meters using equirectangular projection
        lat_offset_m = (lat - self._ref_lat) * self._lat_scale
        lon_offset_m = (lon - self._ref_lon) * self._lon_scale

        # Adjust to centerline coordinate system
        x_meters = self._ref_x + lon_offset_m
        y_meters = self._ref_y + lat_offset_m

        # Find nearest centerline point using KDTree
        query_point = np.array([[x_meters, y_meters]])
        dist, idx = self.kdtree.query(query_point, k=1)

        # Return nearest centerline point
        nearest_point = self.centerline[idx[0]]

        return tuple(nearest_point)

    def get_ribbon_xy(self, ribbon_name: str) -> np.ndarray:
        """Get full ribbon polyline coordinates.

        Args:
            ribbon_name: Name of ribbon

        Returns:
            Nx2 array of (x, y) coordinates
        """
        if ribbon_name not in self.ribbon_dict:
            logger.warning(f"Ribbon '{ribbon_name}' not found, using center")
            ribbon_name = 'center'

        return self.ribbon_dict[ribbon_name]

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of centerline for visualization scaling.

        Returns:
            (x_min, x_max, y_min, y_max)
        """
        x_min = self.centerline[:, 0].min()
        x_max = self.centerline[:, 0].max()
        y_min = self.centerline[:, 1].min()
        y_max = self.centerline[:, 1].max()

        return x_min, x_max, y_min, y_max
