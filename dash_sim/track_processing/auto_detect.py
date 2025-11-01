"""Automatic detection of track parameters from SVG metadata."""

import logging
from pathlib import Path
from typing import Dict, Optional
import xml.etree.ElementTree as ET
from svgpathtools import svg2paths
import numpy as np

logger = logging.getLogger(__name__)


def detect_track_metadata(svg_path: Path) -> Dict:
    """Extract metadata from SVG track map.

    Attempts to auto-detect:
    - Track length (from path distance)
    - Track direction (from signed area)
    - SVG viewBox dimensions
    - Any embedded metadata/attributes

    Args:
        svg_path: Path to SVG file

    Returns:
        dict with detected parameters:
        {
            'length_svg_units': float,
            'direction': 'clockwise' | 'counter-clockwise',
            'viewbox': {'width': float, 'height': float, 'x': float, 'y': float},
            'detected': bool,  # True if metadata found
        }
    """
    metadata = {
        'length_svg_units': None,
        'direction': None,
        'viewbox': None,
        'detected': False,
    }

    try:
        # Parse SVG paths
        paths, attributes = svg2paths(str(svg_path))

        if not paths:
            logger.warning(f"No paths found in {svg_path}")
            return metadata

        # Sample all paths to get total length in SVG units
        total_length = 0.0
        all_points_x = []
        all_points_y = []

        for path in paths:
            total_length += path.length()

            # Sample points for direction detection
            ts = np.linspace(0, 1, 1000)
            points = [path.point(t) for t in ts]
            all_points_x.extend([p.real for p in points])
            all_points_y.extend([p.imag for p in points])

        metadata['length_svg_units'] = total_length

        # Detect direction using signed area (shoelace formula)
        x = np.array(all_points_x)
        y = np.array(all_points_y)
        signed_area = np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)) / 2.0

        # Positive area = counter-clockwise, Negative = clockwise
        metadata['direction'] = 'counter-clockwise' if signed_area > 0 else 'clockwise'

        # Try to get viewBox from SVG root
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            viewbox_str = root.get('viewBox')

            if viewbox_str:
                vb_parts = [float(v) for v in viewbox_str.split()]
                if len(vb_parts) == 4:
                    metadata['viewbox'] = {
                        'x': vb_parts[0],
                        'y': vb_parts[1],
                        'width': vb_parts[2],
                        'height': vb_parts[3]
                    }
        except Exception as e:
            logger.debug(f"Could not parse viewBox: {e}")

        metadata['detected'] = True
        logger.info(f"Auto-detected track metadata:")
        logger.info(f"  Length (SVG units): {total_length:.2f}")
        logger.info(f"  Direction: {metadata['direction']}")
        if metadata['viewbox']:
            logger.info(f"  ViewBox: {metadata['viewbox']}")

    except Exception as e:
        logger.error(f"Failed to auto-detect metadata from {svg_path}: {e}")

    return metadata


def estimate_track_length(svg_path: Path) -> Optional[float]:
    """Quick estimate of track length from SVG path.

    Args:
        svg_path: Path to SVG file

    Returns:
        Estimated length in SVG units, or None if failed
    """
    try:
        paths, _ = svg2paths(str(svg_path))
        return sum(p.length() for p in paths) if paths else None
    except Exception as e:
        logger.error(f"Failed to estimate length: {e}")
        return None
