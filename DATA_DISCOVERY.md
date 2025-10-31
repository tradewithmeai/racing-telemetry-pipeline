# Data Discovery Report

## Dataset: Barber Motorsports Park

### Files Identified

**Telemetry Data:**
- `R1_barber_telemetry_data.csv` - Race 1 telemetry (11.5M rows, 20 cars)
- `R2_barber_telemetry_data.csv` - Race 2 telemetry

**Lap Timing:**
- `R1_barber_lap_start.csv` - Lap start times
- `R1_barber_lap_end.csv` - Lap end times
- `R1_barber_lap_time.csv` - Lap time summary
- (R2 equivalents)

**Results/Analysis:**
- Various official results and analysis CSVs

### Telemetry CSV Schema (R1 sample)

| Column | Example Value | Description |
|--------|--------------|-------------|
| `expire_at` | (empty) | Data expiration timestamp |
| `lap` | 2 | Current lap number |
| `meta_event` | I_R06_2025-09-07 | Event identifier |
| `meta_session` | R1 | Session name (Race 1) |
| `meta_source` | kafka:gr-raw | Data source |
| `meta_time` | 2025-09-06T18:40:41.926Z | **Receiver UTC timestamp** |
| `original_vehicle_id` | GR86-002-000 | Original vehicle ID |
| `outing` | 0 | Outing number |
| `telemetry_name` | speed, aps, gear, etc. | **Signal name** |
| `telemetry_value` | 100, 7206, 3.5, etc. | **Signal value** |
| `timestamp` | 2025-09-05T00:28:20.593Z | **ECU timestamp** |
| `vehicle_id` | GR86-002-000 | **Vehicle identifier** |
| `vehicle_number` | 0 | Car number (extracted) |

### Vehicle Identifiers (20 cars in R1)

```
GR86-002-000  (chassis: 002, car_no: 000)
GR86-004-78   (chassis: 004, car_no: 78)
GR86-006-7    (chassis: 006, car_no: 7)
GR86-010-16   (chassis: 010, car_no: 16)
GR86-013-80   (chassis: 013, car_no: 80)
GR86-015-31   (chassis: 015, car_no: 31)
GR86-016-55   (chassis: 016, car_no: 55)
GR86-022-13   (chassis: 022, car_no: 13)
GR86-025-47   (chassis: 025, car_no: 47)
GR86-026-72   (chassis: 026, car_no: 72)
GR86-030-18   (chassis: 030, car_no: 18)
GR86-033-46   (chassis: 033, car_no: 46)
GR86-036-98   (chassis: 036, car_no: 98)
GR86-038-93   (chassis: 038, car_no: 93)
GR86-040-3    (chassis: 040, car_no: 3)
GR86-047-21   (chassis: 047, car_no: 21)
GR86-049-88   (chassis: 049, car_no: 88)
GR86-060-2    (chassis: 060, car_no: 2)
GR86-063-113  (chassis: 063, car_no: 113)
GR86-065-5    (chassis: 065, car_no: 5)
```

### Available Telemetry Signals (12 total)

| Signal Name | Description | Units | Notes |
|-------------|-------------|-------|-------|
| `speed` | Vehicle speed | Unknown (likely km/h or m/s) | **CRITICAL** |
| `Steering_Angle` | Steering wheel angle | degrees | **CRITICAL** |
| `aps` | Accelerator pedal position | % (0-100) | **CRITICAL** |
| `pbrake_f` | Front brake pressure | bar/psi | **CRITICAL** |
| `pbrake_r` | Rear brake pressure | bar/psi | **CRITICAL** |
| `gear` | Current gear | integer | **CRITICAL** (-1=R, 0=N, 1-6) |
| `nmot` | Engine speed | rpm | **CRITICAL** |
| `accx_can` | Longitudinal acceleration | g or m/s² | From CAN bus |
| `accy_can` | Lateral acceleration | g or m/s² | From CAN bus |
| **`VBOX_Long_Minutes`** | GPS Longitude | **minutes** | **POSITION - convert to degrees** |
| **`VBOX_Lat_Min`** | GPS Latitude | **minutes** | **POSITION - convert to degrees** |
| **`Laptrigger_lapdist_dls`** | Track distance | meters (absolute) | **POSITION - track location** |

### Position Data Availability ✅

**GPS Coordinates:**
- `VBOX_Long_Minutes` → Convert to degrees by dividing by 60
  - Example: -86.61963653564453 minutes → -1.44366 degrees
- `VBOX_Lat_Min` → Convert to degrees by dividing by 60
  - Example: 33.532588958740234 minutes → 0.55888 degrees

**Track Distance:**
- `Laptrigger_lapdist_dls` provides absolute distance around track in meters
  - Example value: 3715 meters
  - Need track length to normalize to [0, 1] or detect lap wraps

### Data Contract Compliance

✅ **Required columns present:**
- `timestamp` (ECU)
- `meta_time` (receiver UTC)
- `vehicle_id` (format: GR86-{chassis}-{car_no})
- `lap` (lap number)
- `telemetry_name` (signal name)
- `telemetry_value` (signal value)

✅ **All critical signals present:**
- speed, Steering_Angle, aps, pbrake_f/r, gear, lap
- **Plus**: GPS (lat/lon) and track_distance

✅ **Multi-car data:**
- 20 cars in Race 1
- Sufficient for synchronized multi-car simulation

### Identified Issues & Handling

1. **Car number "000"**: At least one car (chassis 002) has car_no=000
   - **Solution**: Use chassis_id as canonical key

2. **GPS in Minutes format**: VBOX outputs lat/lon in decimal minutes
   - **Solution**: Divide by 60 to convert to degrees
   - **Keep both**: Store _raw and converted values

3. **Unknown speed units**: Need to detect if km/h, m/s, or mph
   - **Solution**: Analyze typical racing speeds at Barber (max ~180 km/h)
   - Convert to m/s for simulation

4. **Track distance absolute**: Laptrigger_lapdist_dls is absolute meters
   - **Solution**: Detect wraps at lap boundaries
   - Normalize by track length (Barber ≈ 3,700m)

5. **Lap value**: Need to check for sentinel value 32768
   - **Solution**: Lap repair logic will handle

### Next Steps

1. ✅ Create Pydantic schemas matching this structure
2. ✅ Add GPS conversion utilities (minutes → degrees)
3. ✅ Detect speed units (analyze value ranges)
4. ✅ Implement chassis_id / car_no extraction
5. ✅ Build track metadata for Barber (length ≈ 3,700m, GPS bounds)

### Barber Motorsports Park Track Info

- **Location**: Leeds, Alabama, USA
- **Track Length**: ~3.7 km (2.3 miles) = 3,700 meters
- **Configuration**: Road course, 17 turns
- **GPS Bounds** (approximate from data):
  - Longitude: -86.6° to -86.5° (in degrees)
  - Latitude: 33.5° to 33.6° (in degrees)
