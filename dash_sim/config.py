"""Configuration for race replay dashboard."""

from pathlib import Path

# Data paths
BASE_DIR = Path(__file__).parent.parent
PARQUET_PATH = BASE_DIR / "data/processed/barber_r1_pipeline/synchronized/multi_car_frames.parquet"

# Track configuration
CURRENT_TRACK = "barber"  # Current track name
TRACKS_DIR = Path(__file__).parent / "tracks"  # Directory containing track definitions

# Track files (simple ribbon system) - using assets versions that work correctly
CENTERLINE_FILE = Path(__file__).parent / "assets/track_centerline.npy"
RIBBONS_FILE = Path(__file__).parent / "assets/track_ribbons.json"

# Cars to display (chassis IDs)
DEFAULT_CARS = ["010", "002"]  # Start with 2, easy to expand to all 20

# All available cars for selection
ALL_CARS = ["010", "002", "004", "006", "013", "015", "016", "022", "025", "026",
            "030", "033", "036", "038", "040", "047", "049", "060", "063", "065"]

# Performance settings
TARGET_FPS = 20  # Match telemetry sampling rate
TICK_INTERVAL_MS = int(1000 / TARGET_FPS)  # 50ms for 20Hz
FRAME_WINDOW_SIZE = 1000  # Pre-load N frames around playhead (future optimization)

# Visual settings - Car colors
# Generate distinct colors for all cars
CAR_COLORS = {
    "010": "#FF0000",  # Red
    "002": "#0000FF",  # Blue
    "004": "#00FF00",  # Green
    "006": "#FF00FF",  # Magenta
    "013": "#FFFF00",  # Yellow
    "015": "#00FFFF",  # Cyan
    "016": "#FF8800",  # Orange
    "022": "#8800FF",  # Purple
    "025": "#FF0088",  # Pink
    "026": "#00FF88",  # Spring Green
    "030": "#88FF00",  # Lime
    "033": "#0088FF",  # Sky Blue
    "036": "#FF8888",  # Light Red
    "038": "#88FF88",  # Light Green
    "040": "#8888FF",  # Light Blue
    "047": "#FFFF88",  # Light Yellow
    "049": "#FF88FF",  # Light Magenta
    "060": "#88FFFF",  # Light Cyan
    "063": "#888888",  # Gray
    "065": "#FFFFFF",  # White
}

CAR_MARKER_SIZE = 12
CAR_MARKER_OPACITY = 0.8

# Ribbon configuration (from offset_ribbons.py)
RIBBONS_PER_SIDE = 5           # 5 per side (11 total with center)
MAX_HALF_WIDTH_M = 6.86        # Barber track width ~45 ft / 2

# Default ribbon assignment per car
# If not specified, cars are auto-assigned to distribute across ribbons
DEFAULT_RIBBON_BY_CAR = {
    "010": "right_1.37m",  # Example: car 010 on first right ribbon
    "002": "left_1.37m",   # Example: car 002 on first left ribbon
    # Add more as needed, or leave empty for auto-assignment
}

# Debug mode
DEBUG = True  # Shows frame counter, FPS, coordinate grid overlay
SHOW_TELEMETRY_VALUES = False  # Show speed/gear on hover (Phase 3)

# App settings
APP_TITLE = "Race Replay Dashboard - Barber Motorsports Park"
APP_PORT = 8050
APP_HOST = "127.0.0.1"
