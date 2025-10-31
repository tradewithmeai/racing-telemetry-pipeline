# Track Map Placeholder

Place your Barber Motorsports Park track map image here.

## Recommended Formats

- **map.svg** - Vector format (preferred for scalability)
- **map.png** - Raster format (high resolution, e.g., 1920x1080 or higher)

## Where to Get Track Maps

1. **Official Track Website**: Many tracks provide downloadable maps
2. **Racing Simulators**: iRacing, Assetto Corsa often have accurate track layouts
3. **Google Maps**: Satellite view can be screenshot and traced
4. **Track Day Organizations**: Often provide corner maps for drivers

## Map Requirements

- Clear track layout with visible turns
- Proper orientation (North-up preferred, or note rotation)
- Start/finish line marked
- Scale/reference points helpful for GPS alignment

## After Adding Map

Update `metadata.json`:
```json
{
  "map_image_path": "track_maps/barber/map.svg",
  "map_width_px": 1920,
  "map_height_px": 1080
}
```

## GPS Calibration

If the map has known GPS coordinates at specific points, add them to metadata for better alignment:

```json
{
  "calibration_points": [
    {"pixel": [960, 540], "gps": {"lat": 33.5543, "lon": -86.6197}, "label": "Start/Finish"},
    {"pixel": [1200, 300], "gps": {"lat": 33.558, "lon": -86.615}, "label": "Turn 1"}
  ]
}
```
