# Track Maps

This directory contains track map images and metadata for circuit visualization.

## Directory Structure

```
track_maps/
├── README.md
├── {track_name}/
│   ├── metadata.json      # Track parameters and GPS bounds
│   ├── map.svg            # Vector track map (preferred)
│   └── map.png            # Raster track map (alternative)
```

## Metadata Schema

Each track directory must contain a `metadata.json` file with the following structure:

```json
{
  "track_name": "Track Full Name",
  "track_length_m": 3700.0,
  "location": "City, State/Country",
  "gps_bounds": {
    "lat_min": 33.50,
    "lat_max": 33.60,
    "lon_min": -86.65,
    "lon_max": -86.55
  },
  "start_finish_gps": {
    "lat": 33.5543,
    "lon": -86.6197
  },
  "map_image_path": "track_maps/{track_name}/map.svg",
  "map_width_px": 1920,
  "map_height_px": 1080,
  "reference_lap_time_sec": 95.0
}
```

## Adding a New Track

1. **Create Track Directory**
   ```bash
   mkdir track_maps/{track_name}
   ```

2. **Add Track Map Image**
   - Preferred: SVG vector format (`map.svg`)
   - Alternative: High-resolution PNG (`map.png`)
   - Image should show track layout with clear orientation

3. **Create metadata.json**
   - Copy template from existing track
   - Update all fields:
     - `track_name`: Official track name
     - `track_length_m`: Total lap distance in meters
     - `gps_bounds`: Min/max latitude/longitude covering the track
     - `start_finish_gps`: GPS coordinates of start/finish line
     - `reference_lap_time_sec`: Typical/target lap time for the series

4. **Determine GPS Bounds**

   From telemetry data:
   ```python
   # Extract GPS from VBOX signals
   lat_values = telemetry[telemetry.telemetry_name == 'VBOX_Lat_Min'].telemetry_value / 60
   lon_values = telemetry[telemetry.telemetry_name == 'VBOX_Long_Minutes'].telemetry_value / 60

   gps_bounds = {
       "lat_min": float(lat_values.min()),
       "lat_max": float(lat_values.max()),
       "lon_min": float(lon_values.min()),
       "lon_max": float(lon_values.max())
   }
   ```

5. **Update Circuit Params**
   Add entry to `src/conf/circuit_params.yaml`:
   ```yaml
   {track_name}:
     full_name: "Track Full Name"
     location: "City, State/Country"
     track_length_m: 3700.0
     reference_lap_time_sec: 95.0
     min_lap_time_sec: 85.0
     max_lap_time_sec: 180.0
     gps_bounds:
       lat_min: 33.50
       lat_max: 33.60
       lon_min: -86.65
       lon_max: -86.55
     start_finish:
       latitude: 33.5543
       longitude: -86.6197
   ```

## GPS to Map Pixel Mapping

For visualization systems consuming this data:

### Converting GPS to Image Coordinates

```python
def gps_to_pixel(lat, lon, gps_bounds, map_width_px, map_height_px):
    """Map GPS coordinates to pixel position on track map image."""

    # Normalize to [0, 1]
    x_norm = (lon - gps_bounds['lon_min']) / (gps_bounds['lon_max'] - gps_bounds['lon_min'])
    y_norm = (lat - gps_bounds['lat_min']) / (gps_bounds['lat_max'] - gps_bounds['lat_min'])

    # Map to pixels (note: image Y is inverted)
    x_px = x_norm * map_width_px
    y_px = (1 - y_norm) * map_height_px  # Invert Y for image coordinates

    return x_px, y_px
```

### Using Track Position Instead

If using `track_position` (normalized 0-1 around track):
- Map to centerline path on track image
- Requires additional centerline coordinates or spline definition
- More accurate than GPS for replay visualization

## Existing Tracks

### Barber Motorsports Park
- **Length**: 3.7 km (2.3 miles)
- **Type**: Technical road course
- **Turns**: 17
- **Lap Time**: ~95 seconds (GR Cup)
- **GPS Bounds**: 33.50-33.60 N, -86.65 to -86.55 W

## Notes

- SVG maps are preferred as they scale without quality loss
- GPS bounds should include small buffer beyond actual track edges
- Reference lap times help validate lap repair logic
- Track length is critical for normalizing `Laptrigger_lapdist_dls`
