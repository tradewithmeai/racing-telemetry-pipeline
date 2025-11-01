"""Generate parallel offset ribbons from track centerline.

Simple module for creating left/right offset polylines using Shapely.
"""

from pathlib import Path
import json
import numpy as np
from shapely import LineString
from shapely import offset_curve


def build_ribbons(
    centerline_xy_npy: str = "dash_sim/assets/track_centerline.npy",
    out_json: str = "dash_sim/assets/track_ribbons.json",
    ribbons_per_side: int = 5,
    max_half_width_m: float = 6.86,   # Barber ~45 ft / 2
    join_style: str = "round",
    quad_segs: int = 8
):
    """
    Returns N=2*ribbons_per_side+1 polylines (center + offsets).
    Offsets are evenly spaced within [-max_half_width, +max_half_width],
    excluding 0 for centerline (which we include explicitly).

    Args:
        centerline_xy_npy: Path to centerline numpy file (Nx2 array, meters)
        out_json: Output JSON path
        ribbons_per_side: Number of ribbons on each side (default 5)
        max_half_width_m: Maximum offset distance in meters
        join_style: 'round', 'mitre', or 'bevel' (default 'round')
        quad_segs: Number of segments for round joins (higher = smoother)

    Outputs:
        JSON file with structure: {"meta": {...}, "ribbons": [{name, offset_m, xy}, ...]}
    """
    arr = np.load(centerline_xy_npy)  # shape (N,2), metres, clockwise
    line = LineString(arr.tolist())

    # Build distances symmetrically (e.g., for K=5 → ±1.37, ±2.74, ... up to 6.86)
    # Use equal spacing across the half-width.
    if ribbons_per_side < 1:
        offsets = []
    else:
        step = max_half_width_m / ribbons_per_side
        # positive = left, negative = right (per Shapely docs)
        offsets = [i*step for i in range(1, ribbons_per_side+1)]

    def to_xy(geom):
        """Convert Shapely geometry to list of [x,y] coordinates."""
        if geom.is_empty:
            return []
        if geom.geom_type == "LineString":
            return list(map(list, np.asarray(geom.coords)))
        # Occasionally MultiLineString can occur on tight bends; concatenate parts
        if geom.geom_type == "MultiLineString":
            return [list(map(float, c)) for part in geom.geoms for c in np.asarray(part.coords)]
        return []

    ribbons = []
    # Centerline first
    ribbons.append({"name": "center", "offset_m": 0.0, "xy": arr.astype(float).tolist()})

    for d in offsets:
        left = offset_curve(line,  d, quad_segs=quad_segs, join_style=join_style)
        right= offset_curve(line, -d, quad_segs=quad_segs, join_style=join_style)
        ribbons.append({"name": f"left_{d:.2f}m",  "offset_m":  d, "xy": to_xy(left)})
        ribbons.append({"name": f"right_{d:.2f}m", "offset_m": -d, "xy": to_xy(right)})

    meta = {
        "ribbons_per_side": ribbons_per_side,
        "max_half_width_m": max_half_width_m,
        "spacing_m": (max_half_width_m / ribbons_per_side) if ribbons_per_side else 0.0,
        "join_style": join_style,
        "quad_segs": quad_segs,
        "source_centerline": centerline_xy_npy
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({"meta": meta, "ribbons": ribbons}, f, indent=2)

    print(f"Wrote {out_json} with {len(ribbons)} polylines.")
    print(f"  Center + {ribbons_per_side} left + {ribbons_per_side} right = {len(ribbons)} total")
    print(f"  Spacing: {meta['spacing_m']:.2f}m")
    print(f"  Max offset: ±{max_half_width_m:.2f}m")


if __name__ == "__main__":
    # Get paths relative to this script
    script_dir = Path(__file__).parent
    centerline_path = script_dir / "assets/track_centerline.npy"
    output_path = script_dir / "assets/track_ribbons.json"

    build_ribbons(
        centerline_xy_npy=str(centerline_path),
        out_json=str(output_path),
        ribbons_per_side=5,          # start with 5 per side (11 ribbons total)
        max_half_width_m=6.86,       # from 45 ft width
        join_style="round",
        quad_segs=12                 # slightly smoother corners
    )
