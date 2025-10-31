# Racing Telemetry Data Validation & Simulation Pipeline

Production-ready GPU-accelerated pipeline for processing multi-GB racing telemetry datasets into synchronized, validated simulation-ready data.

## Overview

This system processes raw racing telemetry CSVs through a robust validation and transformation pipeline to produce:
- **Multi-car synchronized frames** on a global time grid for race replay simulation
- **Validated position data** (GPS or track distance) for map visualization
- **Per-lap performance features** for analysis and leaderboards
- **Quality reports** indicating data readiness for simulation

### Key Features

- **GPU-Accelerated**: RAPIDS cuDF/Dask for processing multi-GB datasets efficiently
- **Rock-Solid Validation**: Great Expectations suite with explicit fail/warn policies
- **Time Synchronization**: Robust drift correction and multi-car global time alignment
- **Deterministic**: Bit-identical outputs on re-runs; version-pinned dependencies
- **Production-Ready**: Comprehensive error handling, logging, and test coverage

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11+ (for GPU mode) or CPU fallback
- 16GB+ RAM recommended for large datasets

### Installation

```bash
# GPU mode (recommended for large datasets)
make setup-gpu

# CPU mode (fallback)
make setup-cpu

# Development dependencies
make dev
```

### Running the Pipeline

```bash
# Run end-to-end pipeline on sample data
make flow

# Or run directly with Prefect
python flows/e2e_pipeline.py --event barber --raw-path "data/raw/*.csv"
```

## Project Structure

```
├── src/
│   ├── conf/               # Configuration and settings
│   ├── schemas/            # Pydantic data contracts
│   ├── utils/              # Utilities (time, IO, logging)
│   ├── ingestion/          # CSV ingestion and partitioning
│   ├── validation/         # Great Expectations validation suite
│   ├── transform/          # Time correction, lap repair, normalization
│   ├── features/           # Per-lap feature computation
│   └── sim/                # Simulation dataset builder
├── flows/                  # Prefect orchestration flows
├── tests/                  # Unit and integration tests
├── data/
│   ├── raw/                # Input CSV files
│   ├── processed/          # Intermediate Parquet (raw_curated, refined)
│   ├── simulation/         # Final synchronized multi-car frames
│   └── reports/            # Validation HTML reports
├── logs/                   # Pipeline execution logs
├── track_maps/             # Track map images and metadata
└── great_expectations/     # GE configuration and checkpoints
```

## Data Contracts

### Raw CSV Schema (Input)

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | ISO8601/epoch | ECU timestamp |
| `meta_time` | ISO8601 | Receiver UTC timestamp |
| `vehicle_id` | string | Format: `GR86-{chassis}-{car_no}` |
| `lap` | int | Lap number (32768 = invalid) |
| `telemetry_name` | string | Signal name |
| `telemetry_value` | float | Signal value |

### Derived Identifiers

- **chassis_id**: Middle token of `vehicle_id` (e.g., `GR86-004-78` → `004`)
- **car_no**: Last token (e.g., `78`); may be `000` if unassigned
- **time_corrected**: `timestamp + drift_sec` (per-car clock correction)
- **time_global**: Synchronized to session reference time

### Simulation Frame Schema (Output)

Wide format table partitioned by event and time bucket:

| Column | Unit | Description |
|--------|------|-------------|
| `time_global` | UTC | Session-synchronized timestamp |
| `chassis_id` | - | Canonical car identifier |
| `car_no` | - | Car number (may change mid-season) |
| `lap` | int | Repaired lap number |
| `speed` | m/s | Vehicle speed (normalized) |
| `speed_raw` | original | Original speed value |
| `Steering_Angle` | degrees | Steering position |
| `aps` | 0-1 | Accelerator pedal (normalized) |
| `aps_raw` | % | Original throttle percentage |
| `pbrake_f` | bar/psi | Front brake pressure |
| `pbrake_r` | bar/psi | Rear brake pressure |
| `gear` | int | Current gear (-1=reverse, 0=neutral) |
| `nmot` | rpm | Engine speed |
| `accx_can` | m/s² | Longitudinal acceleration |
| `accy_can` | m/s² | Lateral acceleration |
| `latitude` | degrees | GPS latitude (if available) |
| `longitude` | degrees | GPS longitude (if available) |
| `track_position` | 0-1 or m | Normalized track distance (if available) |

### Required Signals for Simulation

Minimum set for visualization:
- `time_corrected`, `speed`, `Steering_Angle`, `aps`, `pbrake_f` OR `pbrake_r`, `gear`, `lap`
- **Position**: `latitude`+`longitude` OR `track_position`

## Validation Policy

### FAIL Conditions (Pipeline Halts)

- Missing required columns or invalid types
- Negative speed values
- Gear outside range [-1, 10]
- Brake pressures < 0
- >5% NaN rate in critical signals (speed, aps, steering, position)
- Clock drift std >30 seconds
- Backwards timestamp detected
- Missing position data for >10% of session
- <3 cars with valid overlapping data

### WARN Conditions (Logged, Pipeline Continues)

- Clock drift std 10-30 seconds
- 1-5% NaN rate in critical signals
- Time gaps 0.5-2.0 seconds
- Position data sparse but interpolable (<10% gaps)
- Missing non-critical signals

See validation reports in `data/reports/<event>/index.html`

## Units & Conversions

All normalized values preserve original `_raw` columns:

| Signal | Input Unit | Output Unit | Conversion |
|--------|-----------|-------------|------------|
| Speed | varies | m/s | Detect and convert |
| APS | % | 0-1 | ÷ 100 |
| VBOX GPS | Minutes | Degrees | ÷ 60 |
| Track Position | varies | 0-1 or meters | Normalize by track length |

## Data Quality Features

### Time Correction
- **Windowed drift estimation**: 5-min rolling windows with median estimator
- **Step detection**: Flags clock jumps (drift_std spike >3× median)
- **Segmentation**: Splits data at step changes for independent calibration

### Lap Repair
- Replace lap=32768 (invalid) with NaN
- Detect boundaries from `Laptrigger_lapdist_dls` resets
- Enforce minimum lap duration (configurable per circuit)
- Log all inferred boundaries with reason codes

### Position Handling
- GPS: VBOX lat/lon conversion, outlier smoothing, bounds validation
- Track distance: Normalize to [0,1], handle wraps at lap boundaries
- Interpolation: Fill sparse position data up to 20Hz (respects gap limits)

### Multi-Car Synchronization
- Global time grid: All cars resampled to identical timestamps
- Forward-fill limit: 0.2s (prevents phantom continuity across gaps)
- Gap handling: Segments with >2s gaps recorded as dropped windows

## Output Artifacts

### For Simulation System

1. **`data/simulation/<event>/multi_car_frames.parquet`**
   - Synchronized telemetry on global time grid
   - All cars in single table (or per-car partitions)

2. **`data/simulation/<event>/simulation_metadata.json`**
   - Time range, car list, signal inventory, sample rate

3. **`track_maps/<event>/metadata.json`**
   - Track length, GPS bounds, map image path

4. **`data/simulation/<event>/per_lap_features.parquet`**
   - Lap summaries for leaderboards

5. **`data/reports/<event>/simulation_readiness_report.json`**
   - Coverage analysis, quality scores per car

### For Auditing

- `chassis_car_mapping.parquet`: Car number changes over time
- `drift_calibration.parquet`: Time correction parameters
- `lap_boundaries.parquet`: Inferred lap transitions with reasons
- `dropped_windows.parquet`: Segments excluded due to gaps

## Testing

```bash
# Run full test suite
make test

# Run specific tests
pytest tests/test_time_sync.py -v
pytest tests/test_lap_repair.py -k "test_determinism"
```

### Critical Test Coverage

- Multi-car time synchronization to global grid
- Position interpolation without phantom jumps
- Lap repair determinism (same input → same boundaries)
- Gap handling (no forward-fill across >2s gaps)
- Simulation dataset schema validation
- Coverage report accuracy

## Configuration

### `conf/settings.py`

- GPU toggle (fallback to CPU if unavailable)
- Chunk sizes for memory management
- Resample frequency (default: 20Hz)
- Forward-fill limit (default: 0.2s)
- Max gap threshold (default: 2.0s)

### `conf/validation_policy.yaml`

- Fail vs warn thresholds per check
- Critical signal definitions
- Range constraints per signal type

### `conf/circuit_params.yaml`

- Track lengths (meters)
- Expected lap times (for sanity checks)
- GPS bounds per circuit
- Start/finish coordinates

## Development

```bash
# Format code
make format

# Lint
make lint

# Clean caches
make clean
```

## Troubleshooting

### GPU Not Detected

If RAPIDS fails to load:
```bash
# Verify CUDA installation
nvidia-smi

# Reinstall CPU mode
make setup-cpu
```

### Out of Memory

Reduce chunk sizes in `conf/settings.py`:
```python
CHUNK_SIZE_ROWS = 500_000  # Default: 1_000_000
```

### Validation Failures

Check HTML report in `data/reports/<event>/index.html` for specific failures.
Adjust severity in `conf/validation_policy.yaml` if needed.

## Architecture Decisions

### Why Global Time Synchronization?

Multi-car race simulation requires all cars on the same time axis. Per-car drift correction + global resampling ensures:
- Frame-accurate synchronization for replay
- No clock skew between cars
- Deterministic temporal alignment

### Why Preserve `_raw` Columns?

Auditing and debugging require original values. Normalized units improve consistency, but raw values enable:
- Tracing conversion errors
- Validating source data quality
- Comparing with external tools

### Why Segment on >2s Gaps?

Forward-filling across pit stops or session breaks creates misleading continuity. Segmentation preserves data integrity by:
- Not interpolating across true gaps
- Recording dropped windows for transparency
- Enabling per-segment analysis

## License

Internal Toyota Gazoo Racing use only.

## Support

For issues or questions, contact the data engineering team.
