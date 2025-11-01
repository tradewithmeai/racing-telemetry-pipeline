"""SVG centerline extraction with scaling and orientation control."""

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from svgpathtools import svg2paths
from scipy.interpolate import splprep, splev

from .auto_detect import detect_track_metadata

logger = logging.getLogger(__name__)


def chord_length_param(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute chord-length parameterization of a curve.

    Args:
        x: X coordinates
        y: Y coordinates

    Returns:
        s: Cumulative arc length array
        total_length: Total curve length
    """
    d = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.insert(np.cumsum(d), 0, 0.0)
    return s, s[-1]


def resample_equal_arclen(
    x: np.ndarray,
    y: np.ndarray,
    n_pts: int,
    smoothing: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Resample curve to equal arc-length intervals.

    Args:
        x: X coordinates
        y: Y coordinates
        n_pts: Number of points in resampled curve
        smoothing: Spline smoothing factor (0 = exact interpolation)

    Returns:
        x_new: Resampled X coordinates
        y_new: Resampled Y coordinates
        total_length: Total arc length before resampling
    """
    s, L = chord_length_param(x, y)

    # Create equal-spacing arc length array
    s_new = np.linspace(0, L, n_pts)

    # Fit parametric spline x(s), y(s)
    tck, _ = splprep([x, y], u=s, s=smoothing, k=3)
    x_new, y_new = splev(s_new, tck)

    return np.asarray(x_new), np.asarray(y_new), L


def scale_to_length(
    x: np.ndarray,
    y: np.ndarray,
    target_len: float
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Scale curve to target physical length.

    Args:
        x: X coordinates
        y: Y coordinates
        target_len: Target length in meters

    Returns:
        x_scaled: Scaled X coordinates
        y_scaled: Scaled Y coordinates
        original_length: Length before scaling
        scale_factor: Applied scale factor
    """
    _, L = chord_length_param(x, y)

    if L == 0:
        raise RuntimeError("Zero-length curve cannot be scaled")

    scale = target_len / L

    # Center at origin and scale
    x_centered = x - x.mean()
    y_centered = y - y.mean()

    x_scaled = x_centered * scale
    y_scaled = y_centered * scale

    return x_scaled, y_scaled, L, scale


def ensure_direction(
    x: np.ndarray,
    y: np.ndarray,
    target_direction: str = 'clockwise'
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Ensure curve has specified direction.

    Args:
        x: X coordinates
        y: Y coordinates
        target_direction: 'clockwise' or 'counter-clockwise'

    Returns:
        x: X coordinates (possibly reversed)
        y: Y coordinates (possibly reversed)
        action: Description of action taken
    """
    # Calculate signed area (shoelace formula)
    signed_area = np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)) / 2.0

    # Positive area = counter-clockwise, Negative = clockwise
    current_direction = 'counter-clockwise' if signed_area > 0 else 'clockwise'

    if current_direction != target_direction:
        return x[::-1], y[::-1], f"flipped_to_{target_direction}"
    else:
        return x, y, f"{target_direction}_ok"


def sample_svg_paths(
    svg_path: Path,
    samples_per_seg: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample all paths from SVG into concatenated point arrays.

    Args:
        svg_path: Path to SVG file
        samples_per_seg: Number of samples per continuous path segment

    Returns:
        x: X coordinates array
        y: Y coordinates array
    """
    paths, _ = svg2paths(str(svg_path))

    if not paths:
        raise RuntimeError(f"No paths found in SVG: {svg_path}")

    xs, ys = [], []

    for path in paths:
        # Skip degenerate paths (zero or near-zero length)
        try:
            path_length = path.length()
            if path_length < 1e-6:
                logger.debug(f"Skipping degenerate path with length {path_length}")
                continue
        except:
            logger.warning(f"Could not compute length for path, skipping")
            continue

        # Sample parameter t ∈ [0,1] along each continuous Path
        ts = np.linspace(0, 1, samples_per_seg)

        try:
            pts = [path.point(t) for t in ts]
            xs.extend([z.real for z in pts])
            ys.extend([z.imag for z in pts])
        except Exception as e:
            logger.warning(f"Error sampling path: {e}, skipping")
            continue

    if not xs:
        raise RuntimeError(f"No valid paths found in SVG: {svg_path}")

    x_arr = np.asarray(xs)
    y_arr = np.asarray(ys)

    # Remove duplicate consecutive points (can cause spline fitting issues)
    points = np.column_stack([x_arr, y_arr])
    diff = np.diff(points, axis=0)
    distances = np.linalg.norm(diff, axis=1)
    non_duplicate_mask = np.concatenate([[True], distances > 1e-6])

    x_clean = x_arr[non_duplicate_mask]
    y_clean = y_arr[non_duplicate_mask]

    logger.info(f"Sampled {len(x_arr)} points, kept {len(x_clean)} after deduplication")

    return x_clean, y_clean


def extract_centerline(
    svg_path: Path,
    target_length_m: Optional[float] = None,
    samples: int = 5000,
    smoothing: float = 0.0,
    ensure_direction_val: str = 'clockwise',
    samples_per_seg: int = 4000
) -> Tuple[np.ndarray, dict]:
    """Extract scaled centerline from SVG track map.

    This is the main entry point for SVG → centerline conversion.

    Args:
        svg_path: Path to SVG file
        target_length_m: Target length in meters (None = auto-detect from SVG)
        samples: Number of equally-spaced points in output
        smoothing: Spline smoothing factor (0 = exact interpolation)
        ensure_direction_val: 'clockwise' or 'counter-clockwise'
        samples_per_seg: Initial samples per path segment

    Returns:
        centerline: Nx2 array of (x, y) coordinates in meters
        metadata: dict with processing metadata
    """
    logger.info(f"Extracting centerline from {svg_path}")

    # Auto-detect metadata if target length not provided
    auto_meta = detect_track_metadata(svg_path)

    if target_length_m is None:
        if auto_meta['length_svg_units']:
            logger.warning(
                f"No target length provided. Using SVG units: "
                f"{auto_meta['length_svg_units']:.2f}. "
                f"Set target_length_m for accurate scaling."
            )
            target_length_m = auto_meta['length_svg_units']
        else:
            raise ValueError(
                "Could not auto-detect length and no target_length_m provided"
            )

    # Sample SVG paths
    logger.info(f"Sampling SVG paths ({samples_per_seg} samples/segment)...")
    x_raw, y_raw = sample_svg_paths(svg_path, samples_per_seg=samples_per_seg)

    # Resample to equal arc length
    logger.info(f"Resampling to {samples} equally-spaced points...")
    x_rs, y_rs, L_svg_units = resample_equal_arclen(
        x_raw, y_raw, samples, smoothing=smoothing
    )

    # Scale to target length
    logger.info(f"Scaling to target length: {target_length_m:.2f}m...")
    x_m, y_m, L_before, scale = scale_to_length(x_rs, y_rs, target_length_m)

    # Ensure correct direction
    logger.info(f"Ensuring {ensure_direction_val} orientation...")
    x_final, y_final, orient_note = ensure_direction(
        x_m, y_m, target_direction=ensure_direction_val
    )

    # Compute final length for verification
    final_length = np.linalg.norm(np.diff(np.stack([x_final, y_final], axis=1), axis=0), axis=1).sum()

    metadata = {
        'source_svg': str(svg_path),
        'target_length_m': target_length_m,
        'actual_length_m': float(final_length),
        'pre_scale_length_units': L_svg_units,
        'scale_factor': scale,
        'samples': samples,
        'smoothing': smoothing,
        'orientation': orient_note,
        'direction_target': ensure_direction_val,
        'auto_detected': auto_meta,
    }

    logger.info(f"Centerline extracted:")
    logger.info(f"  Target length: {target_length_m:.2f}m")
    logger.info(f"  Actual length: {final_length:.2f}m")
    logger.info(f"  Scale factor: {scale:.6f}")
    logger.info(f"  Orientation: {orient_note}")

    # Return as Nx2 array
    centerline = np.stack([x_final, y_final], axis=1).astype(np.float32)

    return centerline, metadata
