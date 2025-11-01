# Race Replay Dashboard

Real-time visualization of multi-car racing telemetry with synchronized playback on a 2D track map.

## Features

- **Real-time replay** of multi-car race telemetry at 20Hz
- **Interactive controls**: Play/Pause, speed adjustment (0.5x - 4x), timeline scrubbing
- **Track visualization**: Cars rendered as colored dots on overhead track image
- **Client-side animation**: Smooth playback without server lag
- **Modular architecture**: Easy to extend with telemetry overlays, 3D views, analysis tools

## Setup

### Prerequisites

- Python 3.10+
- Track image file (overhead view of Barber Motorsports Park)
- Processed telemetry data from the data pipeline

### Installation

```bash
# Navigate to dash_sim directory
cd dash_sim

# Install dependencies
pip install -r requirements.txt
```

### Track Image Setup

Place your track image (PNG or JPG) at:
```
dash_sim/assets/track.png
```

The image should be an overhead/satellite view of Barber Motorsports Park. The coordinate transformer will automatically map GPS coordinates to pixel positions.

## Usage

### Basic Usage

```bash
# Default: uses config.py settings
python app.py

# Override cars and parquet path
python app.py --cars 010 002 --parquet ../data/processed/barber_r1_pipeline/synchronized/multi_car_frames.parquet

# Test with single car
python app.py --cars 010
```

Then open your browser to: **http://127.0.0.1:8050**

### Configuration

Edit `config.py` to customize:

```python
# Select which cars to display (start with 2, expand to all 20)
DEFAULT_CARS = ["010", "002"]

# Adjust animation speed
TARGET_FPS = 20  # Matches telemetry sampling rate

# Customize car colors
CAR_COLORS = {
    "010": "#FF0000",  # Red
    "002": "#0000FF",  # Blue
    # ... add more
}

# Performance tuning
FRAME_WINDOW_SIZE = 1000  # Future: load frames in windows

# Debug mode
DEBUG = True  # Shows FPS counter, frame numbers
```

### Data Requirements

The dashboard expects synchronized multi-car telemetry from the pipeline:

**Input file**: `../data/processed/barber_r1_pipeline/synchronized/multi_car_frames.parquet`

**Required columns**:
- `time_global` (datetime) - Global timeline synchronized across all cars
- `chassis_id` (str) - Car identifier (e.g., "010", "002")
- `car_no` (str) - Car number for display
- `gps_lat` (float) - Latitude in decimal degrees
- `gps_lon` (float) - Longitude in decimal degrees

**Optional columns** (recommended for best experience):
- `track_distance_m` (float) - Distance along track centerline in meters (preferred for positioning)
- `speed_final` (float) - Velocity in m/s (derived + interpolated, preferred over `speed`)
- `speed` (float) - Raw velocity in m/s (fallback if `speed_final` not available)
- `gear`, `aps`, `lap_repaired` - Additional telemetry for hover tooltips (Phase 3)

**Column priority**:
- **Position**: Uses `track_distance_m` if available (centerline-snapped), falls back to `gps_lat/gps_lon`
- **Speed**: Uses `speed_final` if available (98.8% coverage with interpolation), falls back to `speed`

## Architecture

### Modules

- **`config.py`** - Centralized configuration (paths, colors, performance settings)
- **`coordinate_transform.py`** - GPS to pixel coordinate transformation with auto-bounds detection
- **`data_loader.py`** - Load parquet, preprocess, compute trajectories
- **`app.py`** - Main Dash application with layout and callbacks

### Data Flow

1. **Startup**: Load parquet → Extract GPS bounds → Initialize coordinate transformer
2. **Transform**: Convert all GPS coordinates to pixel positions → Store in `dcc.Store`
3. **Animation**: Client-side JS callback advances frame index → Updates car positions at 20Hz
4. **Controls**: Server-side callbacks handle play/pause/seek/speed changes

### Performance

- **Client-side animation** (JavaScript) for smooth playback without server roundtrips
- **Plotly ScatterGL** for hardware-accelerated rendering
- **Pre-computed trajectories** stored as arrays (no per-frame computation)
- **Frame windowing architecture** ready for scaling to 20+ cars over 100k+ frames

## Controls

- **Play Button (▶)**: Start replay from current position
- **Pause Button (⏸)**: Pause replay
- **Speed Dropdown**: Adjust playback speed (0.5x, 1x, 2x, 4x)
- **Timeline Slider**: Seek to any point in the race (pauses playback)
- **Frame Info**: Shows current frame number and elapsed time

## Extending the Dashboard

### Phase 2: Multi-Car Scaling

To display all 20 cars:

```python
# config.py
DEFAULT_CARS = ["010", "002", "004", "006", "013", "015", "016", "022",
                "025", "026", "030", "033", "036", "038", "040", "047",
                "049", "060", "063", "065"]
```

If performance degrades with 20 cars, implement frame windowing in `data_loader.py`.

### Phase 3: Telemetry Overlays

Add hover tooltips or side panel displaying:
- Current speed
- Gear position
- Throttle position (APS)
- Brake pressure
- Steering angle

Data is already loaded in `store-trajectories['trajectories'][car_id]` dict.

### Phase 4: Advanced Features

**Planned additions**:
- Lap markers and current lap display
- Sector timing comparison
- Car selection multiselect (toggle which cars to show)
- Side-by-side comparison mode
- Export view as PNG/video
- 3D track visualization (separate module)
- Track centreline snapping for smoother visualization

## Troubleshooting

### Track image not found

Ensure `assets/track.png` exists. Check `config.TRACK_IMAGE` path.

### No GPS data

Some cars may not have GPS coordinates. The dashboard handles this gracefully by showing cars only when valid GPS is available.

### Slow performance

- Reduce `DEFAULT_CARS` to 2-5 cars
- Implement frame windowing (load only ±500 frames around playhead)
- Check browser console for JavaScript errors

### Cars not moving

- Verify `gps_lat` and `gps_lon` columns exist in parquet file
- Check coordinate transformer logs for GPS bounds
- Enable `DEBUG = True` in config to see frame updates

## Development

### Debug Mode

Set `DEBUG = True` in `config.py` to enable:
- Console logging for data loading and transformation
- FPS monitoring (future)
- Coordinate grid overlay (future)

### Testing Coordinate Transformation

```python
from coordinate_transform import TrackCoordinateTransformer

transformer = TrackCoordinateTransformer(
    track_image_path=Path("assets/track.png"),
    gps_bounds=(33.5, 33.6, -86.65, -86.55),  # Barber track approx bounds
    padding=0.05
)

# Test forward transform
x, y = transformer.transform(lat=33.55, lon=-86.60)
print(f"GPS (33.55, -86.60) → Pixel ({x:.1f}, {y:.1f})")

# Test inverse transform
lat, lon = transformer.inverse_transform(x, y)
print(f"Pixel ({x:.1f}, {y:.1f}) → GPS ({lat:.6f}, {lon:.6f})")
```

## License

Part of Toyota Gazoo Racing Motorsport telemetry analysis pipeline.

## Credits

Built with Dash, Plotly, and Python for high-performance race data visualization.
