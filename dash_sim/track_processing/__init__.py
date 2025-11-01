"""Track processing library for SVG-based race track geometry extraction.

This package provides utilities for converting SVG track maps into scaled
centerlines and parallel ribbon offsets for use in race replay visualization.
"""

from .svg_extractor import extract_centerline
from .ribbon_generator import generate_ribbons
from .track_loader import Track

__all__ = [
    'extract_centerline',
    'generate_ribbons',
    'Track',
]

__version__ = '1.0.0'
