"""Generate parallel offset ribbons from track centerline."""

import logging
from typing import Dict, List
import numpy as np
from shapely import LineString
from shapely import offset_curve

logger = logging.getLogger(__name__)


def to_xy_list(geom) -> List[List[float]]:
    """Convert Shapely geometry to list of [x, y] coordinates.

    Handles LineString and MultiLineString (concatenates parts).

    Args:
        geom: Shapely LineString or MultiLineString

    Returns:
        List of [x, y] coordinate pairs
    """
    if geom.is_empty:
        return []

    if geom.geom_type == "LineString":
        return [[float(x), float(y)] for x, y in np.asarray(geom.coords)]

    # Handle MultiLineString (can occur on tight bends)
    if geom.geom_type == "MultiLineString":
        coords = []
        for part in geom.geoms:
            coords.extend([[float(x), float(y)] for x, y in np.asarray(part.coords)])
        return coords

    logger.warning(f"Unexpected geometry type: {geom.geom_type}")
    return []


def generate_ribbons(
    centerline: np.ndarray,
    ribbons_per_side: int = 5,
    max_half_width_m: float = 6.86,
    join_style: str = 'round',
    quad_segs: int = 12
) -> Dict:
    """Generate parallel offset ribbons from centerline.

    Creates N=2*ribbons_per_side+1 polylines (center + offsets on each side).
    Offsets are evenly spaced within [-max_half_width, +max_half_width].

    Args:
        centerline: Nx2 array of (x, y) coordinates in meters
        ribbons_per_side: Number of ribbons on each side (default 5)
        max_half_width_m: Maximum offset distance (half track width)
        join_style: Join style for offset ('round', 'mitre', 'bevel')
        quad_segs: Number of segments for round joins (higher = smoother)

    Returns:
        dict with:
            'meta': metadata about ribbon generation
            'ribbons': list of ribbon dicts with 'name', 'offset_m', 'xy'
    """
    logger.info(f"Generating ribbons ({ribbons_per_side} per side)...")

    # Create Shapely LineString from centerline
    line = LineString(centerline.tolist())

    # Build offset distances
    if ribbons_per_side < 1:
        offsets = []
    else:
        step = max_half_width_m / ribbons_per_side
        # Positive = left, negative = right (Shapely convention)
        offsets = [i * step for i in range(1, ribbons_per_side + 1)]

    ribbons = []

    # Add centerline first
    ribbons.append({
        'name': 'center',
        'offset_m': 0.0,
        'xy': centerline.astype(float).tolist()
    })
    logger.info(f"  Added centerline ({len(centerline)} points)")

    # Generate offset ribbons
    for d in offsets:
        try:
            # Left offset (positive)
            left = offset_curve(line, d, quad_segs=quad_segs, join_style=join_style)
            left_xy = to_xy_list(left)

            if left_xy:
                ribbons.append({
                    'name': f'left_{d:.2f}m',
                    'offset_m': d,
                    'xy': left_xy
                })
                logger.info(f"  Added left_{d:.2f}m ({len(left_xy)} points)")
            else:
                logger.warning(f"  Empty geometry for left_{d:.2f}m")

            # Right offset (negative)
            right = offset_curve(line, -d, quad_segs=quad_segs, join_style=join_style)
            right_xy = to_xy_list(right)

            if right_xy:
                ribbons.append({
                    'name': f'right_{d:.2f}m',
                    'offset_m': -d,
                    'xy': right_xy
                })
                logger.info(f"  Added right_{d:.2f}m ({len(right_xy)} points)")
            else:
                logger.warning(f"  Empty geometry for right_{d:.2f}m")

        except Exception as e:
            logger.error(f"  Failed to generate offset {d:.2f}m: {e}")

    # Create metadata
    meta = {
        'ribbons_per_side': ribbons_per_side,
        'max_half_width_m': max_half_width_m,
        'spacing_m': (max_half_width_m / ribbons_per_side) if ribbons_per_side else 0.0,
        'join_style': join_style,
        'quad_segs': quad_segs,
        'total_ribbons': len(ribbons),
    }

    logger.info(f"Generated {len(ribbons)} ribbons total")

    return {
        'meta': meta,
        'ribbons': ribbons
    }
