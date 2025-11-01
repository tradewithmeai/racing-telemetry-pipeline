"""Interactive track calibration tool.

This tool helps you manually set GPS coordinates for the track image corners
to correctly map telemetry positions onto the circuit map.
"""

import config
from coordinate_transform import TrackCoordinateTransformer
from PIL import Image
import numpy as np

print("=" * 60)
print("TRACK CALIBRATION TOOL")
print("=" * 60)

# Load track image
img = Image.open(config.TRACK_IMAGE)
width, height = img.size
print(f"\nTrack image: {width}x{height} pixels")

# Show current GPS bounds from telemetry data
print(f"\nTelemetry GPS bounds:")
print(f"  Latitude:  33.529335 to 33.535892")
print(f"  Longitude: -86.624321 to -86.614532")

print("\n" + "=" * 60)
print("MANUAL CALIBRATION")
print("=" * 60)
print("\nYou need to determine the GPS coordinates of the track image corners.")
print("Look at the PDF circuit map and find GPS coordinates or use Google Maps.")
print("\nBarber Motorsports Park approximate bounds:")
print("  Top-left corner:     Lat ~33.537, Lon ~-86.625")
print("  Bottom-right corner: Lat ~33.528, Lon ~-86.613")

print("\nTo calibrate:")
print("1. Open the track PDF and look for GPS coordinates")
print("2. Or use Google Maps satellite view of Barber Motorsports Park")
print("3. Identify the GPS coordinates at the corners of your track image")
print("4. Update these values in config.py:")
print("\nGPS_LAT_MIN = 33.528  # Bottom latitude")
print("GPS_LAT_MAX = 33.537  # Top latitude")
print("GPS_LON_MIN = -86.625 # Left longitude")
print("GPS_LON_MAX = -86.613 # Right longitude")

print("\n" + "=" * 60)
print("QUICK FIX - Using telemetry bounds with padding")
print("=" * 60)

# Add padding to telemetry bounds
lat_range = 33.535892 - 33.529335
lon_range = -86.614532 - (-86.624321)
padding = 0.1  # 10% padding

lat_min = 33.529335 - lat_range * padding
lat_max = 33.535892 + lat_range * padding
lon_min = -86.624321 - lon_range * padding
lon_max = -86.614532 + lon_range * padding

print(f"\nSuggested bounds (telemetry + 10% padding):")
print(f"GPS_LAT_MIN = {lat_min:.6f}")
print(f"GPS_LAT_MAX = {lat_max:.6f}")
print(f"GPS_LON_MIN = {lon_min:.6f}")
print(f"GPS_LON_MAX = {lon_max:.6f}")

print("\nAdd these to config.py and restart the dashboard.")
print("If the car is still off-track, you'll need to manually calibrate using the PDF/Google Maps.")
