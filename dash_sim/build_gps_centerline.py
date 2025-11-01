"""Build GPS centerline reference from Car 010 telemetry data.

This creates a GPS centerline by extracting Car 010's GPS trace and mapping it
to the track centerline for distance calculations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
import json

# Barber Motorsports Park reference point (approximate track center)
TRACK_CENTER_LAT = 33.532614
TRACK_CENTER_LON = -86.619427
METERS_PER_DEGREE_LAT = 111000  # Approximate meters per degree latitude
METERS_PER_DEGREE_LON = 91000   # Approximate meters per degree longitude at 34°N


def latlon_to_xy(lat, lon):
    """Convert lat/lon to XY meters relative to track center."""
    x = (lon - TRACK_CENTER_LON) * METERS_PER_DEGREE_LON
    y = (lat - TRACK_CENTER_LAT) * METERS_PER_DEGREE_LAT
    return x, y


def build_gps_reference():
    """Build GPS centerline reference from Car 010 data."""
    # Get paths relative to script location
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    # Load synchronized data
    df = pd.read_parquet(base_dir / 'data/processed/barber_r1_pipeline/synchronized/multi_car_frames.parquet')

    # Get Car 010 data with valid GPS
    car_010 = df[df['chassis_id'] == '010'].copy()
    car_010 = car_010[car_010['gps_lat'].notna() & car_010['gps_lon'].notna()]

    print(f"Car 010: {len(car_010)} rows with valid GPS")

    # Convert GPS to XY
    car_010['x_gps'], car_010['y_gps'] = zip(*[latlon_to_xy(lat, lon)
                                                  for lat, lon in zip(car_010['gps_lat'], car_010['gps_lon'])])

    # Load centerline
    centerline_xy = np.load(script_dir / 'assets/track_centerline.npy')
    print(f"Centerline: {len(centerline_xy)} points")

    # Compute distance along centerline
    diffs = np.diff(centerline_xy, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    centerline_distances = np.concatenate([[0], np.cumsum(segment_lengths)])
    track_length = centerline_distances[-1]
    print(f"Track length: {track_length:.2f}m")

    # Build KD-tree on centerline XY
    tree = cKDTree(centerline_xy)

    # Map each GPS point to nearest centerline point
    gps_xy = np.column_stack([car_010['x_gps'].values, car_010['y_gps'].values])
    distances_to_centerline, nearest_indices = tree.query(gps_xy)

    # Get track distances for GPS points
    car_010['computed_track_distance'] = centerline_distances[nearest_indices]
    car_010['distance_to_centerline'] = distances_to_centerline

    print(f"\nGPS to centerline mapping:")
    print(f"  Mean distance to centerline: {distances_to_centerline.mean():.2f}m")
    print(f"  Max distance to centerline: {distances_to_centerline.max():.2f}m")

    # Now build GPS reference: for each centerline point, find nearest GPS lat/lon
    gps_tree = cKDTree(gps_xy)
    _, gps_nearest_indices = gps_tree.query(centerline_xy)

    # Create GPS centerline
    gps_centerline_lat = car_010['gps_lat'].iloc[gps_nearest_indices].values
    gps_centerline_lon = car_010['gps_lon'].iloc[gps_nearest_indices].values

    # Save GPS centerline reference
    gps_reference = {
        'centerline_lat': gps_centerline_lat.tolist(),
        'centerline_lon': gps_centerline_lon.tolist(),
        'centerline_distances': centerline_distances.tolist(),
        'track_length_m': float(track_length),
        'track_center_lat': TRACK_CENTER_LAT,
        'track_center_lon': TRACK_CENTER_LON,
        'meters_per_degree_lat': METERS_PER_DEGREE_LAT,
        'meters_per_degree_lon': METERS_PER_DEGREE_LON,
    }

    output_file = script_dir / 'assets/gps_centerline_reference.json'
    with open(output_file, 'w') as f:
        json.dump(gps_reference, f, indent=2)

    print(f"\nSaved GPS reference to: {output_file}")
    print(f"  {len(gps_centerline_lat)} GPS centerline points")
    print(f"  Lat range: [{gps_centerline_lat.min():.6f}, {gps_centerline_lat.max():.6f}]")
    print(f"  Lon range: [{gps_centerline_lon.min():.6f}, {gps_centerline_lon.max():.6f}]")


if __name__ == '__main__':
    build_gps_reference()
