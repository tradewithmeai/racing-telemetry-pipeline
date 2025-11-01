"""Extract ONLY the main centerline from Barber SVG.

This script identifies and extracts the main track centerline (the thick black line)
from the Barber Motorsports Park SVG, ignoring corner numbers and gray sections.
"""

from pathlib import Path
import numpy as np
from svgpathtools import svg2paths
from scipy.interpolate import splprep, splev

# Target track length from config
TARGET_LENGTH_M = 3830.0  # Barber official length

def extract_main_centerline_only(svg_path: Path, n_samples: int = 5000):
    """Extract only the main track centerline path from SVG.

    Args:
        svg_path: Path to Barber SVG
        n_samples: Number of points to sample

    Returns:
        centerline_xy: Nx2 array of (x, y) in meters
    """
    paths, attributes = svg2paths(str(svg_path))

    # Find the main centerline (longest path with black stroke)
    # Based on analysis: Path 1 or 10, length ~5487, black stroke width 15-17
    main_path = None
    max_length = 0

    for path, attr in zip(paths, attributes):
        length = path.length()
        style = attr.get('style', '')

        # Look for the main track: long path with black or white stroke, width 15-17
        if length > 5000 and ('stroke:black' in style or 'stroke:white' in style):
            if 'stroke-width:15' in style or 'stroke-width:17' in style:
                if length > max_length:
                    max_length = length
                    main_path = path
                    print(f"Found main centerline: length={length:.2f}, style={style[:100]}...")

    if main_path is None:
        raise RuntimeError("Could not find main centerline path in SVG")

    # Sample the path uniformly
    ts = np.linspace(0, 1, n_samples * 2)  # Oversample then smooth
    pts = [main_path.point(t) for t in ts]
    x_raw = np.array([z.real for z in pts])
    y_raw = np.array([z.imag for z in pts])

    # Remove consecutive duplicates
    diff = np.sqrt(np.diff(x_raw)**2 + np.diff(y_raw)**2)
    non_dup_mask = np.concatenate([[True], diff > 1e-6])
    x_clean = x_raw[non_dup_mask]
    y_clean = y_raw[non_dup_mask]

    print(f"Sampled {len(x_raw)} points, kept {len(x_clean)} after deduplication")

    # Compute arc length for scaling
    diff = np.sqrt(np.diff(x_clean)**2 + np.diff(y_clean)**2)
    arc_length_svg = np.sum(diff)
    print(f"SVG arc length: {arc_length_svg:.2f} units")

    # Scale to target length
    scale_factor = TARGET_LENGTH_M / arc_length_svg
    print(f"Scale factor: {scale_factor:.6f}")

    # Center and scale
    x_centered = x_clean - x_clean.mean()
    y_centered = y_clean - y_clean.mean()

    x_scaled = x_centered * scale_factor
    y_scaled = y_centered * scale_factor

    # Flip Y-axis (SVG coordinate system is inverted)
    y_scaled = -y_scaled

    # Resample to uniform spacing using spline
    s_param = np.concatenate([[0], np.cumsum(diff)]) * scale_factor
    s_uniform = np.linspace(0, s_param[-1], n_samples)

    tck, _ = splprep([x_scaled, y_scaled], u=s_param, s=0, k=3)
    x_final, y_final = splev(s_uniform, tck)

    centerline = np.column_stack([x_final, y_final])

    # Verify final length
    final_diff = np.sqrt(np.diff(centerline[:, 0])**2 + np.diff(centerline[:, 1])**2)
    final_length = np.sum(final_diff)
    print(f"Final centerline length: {final_length:.2f}m (target: {TARGET_LENGTH_M}m)")

    return centerline


if __name__ == "__main__":
    svg_file = Path("dash_sim/assets/Barber_Motorsports_Park.svg")
    output_file = Path("dash_sim/assets/track_centerline.npy")

    print("Extracting main centerline from Barber SVG...")
    print(f"Input: {svg_file}")

    centerline = extract_main_centerline_only(svg_file)

    np.save(output_file, centerline)
    print(f"\nSaved centerline to: {output_file}")
    print(f"Shape: {centerline.shape}")
    print(f"X range: [{centerline[:, 0].min():.2f}, {centerline[:, 0].max():.2f}]")
    print(f"Y range: [{centerline[:, 1].min():.2f}, {centerline[:, 1].max():.2f}]")
