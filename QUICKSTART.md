# Quick Start Guide: Running the Telemetry Pipeline

This guide walks you through running the complete telemetry processing pipeline on your Barber Motorsports Park Race 1 data.

---

## Prerequisites

- **Python 3.10+** installed
- **16GB+ RAM** (processing 1.5GB CSV with 11.5M rows)
- **GPU optional** (RAPIDS cuDF) - will auto-fallback to CPU if not available

---

## Step 1: Set Up Python Environment

### Option A: Using pip (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies (CPU mode)
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Option B: Using Makefile

```bash
# CPU mode (most compatible)
make setup-cpu

# GPU mode (if you have NVIDIA GPU + CUDA)
make setup-gpu

# Development mode (includes testing tools)
make dev
```

---

## Step 2: Move Data to Raw Directory

Your telemetry data is currently in `barber-motorsports-park/barber/`. Let's organize it:

```bash
# Copy telemetry CSV to raw data directory
cp "barber-motorsports-park/barber/R1_barber_telemetry_data.csv" data/raw/

# Verify file is there
ls -lh data/raw/
```

**Expected output**: You should see `R1_barber_telemetry_data.csv` (~1.5GB)

---

## Step 3: Run Stage-by-Stage Pipeline

### Stage 0: Ingestion (Already Tested ✅)

The ingestion has already been run based on the processed data in `data/processed/barber_r1/`.

To re-run ingestion:

```bash
python test_ingestion.py
```

**What it does**:
- Reads CSV in chunks (memory-efficient)
- Detects anomalies (duplicates, backwards timestamps, gaps)
- Parses vehicle IDs (chassis_id, car_no)
- Writes partitioned Parquet files

**Output**: `data/processed/barber_r1/` with partitioned data

---

### Stage 1: Time Synchronization (Already Tested ✅)

```bash
python test_time_sync.py
```

**What it does**:
- Detects per-car clock drift
- Corrects backwards timestamps
- Applies drift calibration
- Creates `time_corrected` column

**Output**:
- `data/processed/barber_r1/barber_r1/drift_calibration.parquet`
- Updated telemetry data with `time_corrected` column

**Expected Results**: 99.999% backwards timestamp elimination (see `TIME_SYNC_TEST_RESULTS.md`)

---

### Stage 2: Lap Repair (Already Tested ✅)

```bash
python test_lap_repair.py
```

**What it does**:
- Detects lap boundaries from lap number increments
- Validates lap durations (85s - 300s)
- Replaces sentinel values (32768)
- Creates deterministic lap assignments

**Output**:
- `data/processed/barber_r1/barber_r1/lap_boundaries.parquet`
- Updated data with `lap_repaired` column

**Expected Results**: 28 lap boundaries detected (see `LAP_REPAIR_TEST_RESULTS.md`)

---

### Stage 3: Position Normalization (Already Tested ✅)

```bash
python test_position_normalization.py
```

**What it does**:
- Converts GPS from VBOX format (already in degrees)
- Validates GPS bounds for Barber track
- Detects outliers (jumps >500m)
- Normalizes track distance

**Output**:
- `data/processed/barber_r1/barber_r1/position_quality.parquet`
- Updated data with `gps_lat`, `gps_lon`, `track_distance_m` columns

---

### Stage 4: Pivot Transformation (NEW)

Now let's run the new stages we just built:

```bash
python examples/test_pivot.py
```

**What it does**:
- Converts long format → wide format
- Each signal becomes a column (speed, aps, gear, etc.)
- Normalizes units (aps → [0,1], speed → m/s)
- Renames position columns

**Output**:
- `data/processed/barber_r1_test/wide_format/{chassis_id}.parquet`
- `data/processed/barber_r1_test/pivot_stats.parquet`

**Expected**: ~12 signals converted to columns per car

---

### Stage 5: Resample to 20Hz Grid

Create a test script to run resampling:

```bash
# Create test script
cat > test_resample.py << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.transform import resample_to_time_grid, save_resampled_data, save_resample_stats
from src.utils.logging_utils import get_logger

logger = get_logger("test_resample")

# Load wide-format data from pivot stage
chassis_ids = ["010", "002"]
output_path = Path("data/processed")

all_stats = {}

for chassis_id in chassis_ids:
    logger.info(f"\nResampling chassis {chassis_id}...")

    # Load wide-format data
    wide_file = output_path / "barber_r1_test" / "wide_format" / f"{chassis_id}.parquet"

    if not wide_file.exists():
        logger.error(f"Wide format data not found: {wide_file}")
        continue

    df = pd.read_parquet(wide_file)

    # Resample to 20Hz
    df_resampled, stats = resample_to_time_grid(
        df=df,
        chassis_id=chassis_id,
        time_col="time_corrected",
        freq_hz=20.0,
        ffill_limit_sec=0.2,
        max_gap_sec=2.0,
    )

    # Save
    save_resampled_data(df_resampled, output_path, "barber_r1_test", chassis_id)
    all_stats[chassis_id] = stats

# Save stats
save_resample_stats(all_stats, output_path, "barber_r1_test")

logger.info("\nResample complete!")
EOF

# Run it
python test_resample.py
```

**What it does**:
- Creates uniform 20Hz time grid for each car
- Forward-fills gaps (max 0.2s)
- Detects large gaps (>2s)
- Tracks coverage statistics

**Output**:
- `data/processed/barber_r1_test/resampled/{chassis_id}.parquet`
- `data/processed/barber_r1_test/resample_stats.parquet`

---

### Stage 6: Multi-Car Synchronization

Create test script for synchronization:

```bash
# Create test script
cat > test_sync.py << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.transform import synchronize_multi_car, save_synchronized_data, save_sync_stats
from src.utils.logging_utils import get_logger

logger = get_logger("test_sync")

# Load resampled data for all cars
chassis_ids = ["010", "002"]
output_path = Path("data/processed")

dfs_by_car = {}

for chassis_id in chassis_ids:
    resampled_file = output_path / "barber_r1_test" / "resampled" / f"{chassis_id}.parquet"

    if not resampled_file.exists():
        logger.error(f"Resampled data not found: {resampled_file}")
        continue

    df = pd.read_parquet(resampled_file)
    dfs_by_car[chassis_id] = df
    logger.info(f"Loaded {chassis_id}: {len(df):,} rows")

# Synchronize all cars to global time grid
df_sync, sync_stats, coverage_by_car = synchronize_multi_car(
    dfs_by_car=dfs_by_car,
    event_name="barber_r1_test",
    time_col="time_corrected",
    freq_hz=20.0,
    ffill_limit_sec=0.2,
)

# Save synchronized data
save_synchronized_data(
    df=df_sync,
    output_path=output_path,
    event_name="barber_r1_test",
    partitioned=False,
)

# Save stats
save_sync_stats(sync_stats, coverage_by_car, output_path, "barber_r1_test")

logger.info("\n✅ Multi-car synchronization complete!")
logger.info(f"   Total frames: {sync_stats.total_frames:,}")
logger.info(f"   Cars per frame (mean): {sync_stats.cars_per_frame_mean:.1f}")
logger.info(f"   Coverage: {sync_stats.coverage_pct:.1f}%")
EOF

# Run it
python test_sync.py
```

**What it does**:
- Determines global time range across all cars
- Aligns each car to unified time grid
- Creates multi-car synchronized dataset
- Tracks per-car coverage

**Output**:
- `data/processed/barber_r1_test/synchronized/multi_car_frames.parquet`
- `data/processed/barber_r1_test/sync_stats.parquet`
- `data/processed/barber_r1_test/car_coverage.parquet`

---

### Stage 7: Validation

Run Great Expectations validation on synchronized data:

```bash
python examples/test_validation.py
```

**What it does**:
- Validates schema compliance
- Checks signal ranges (speed ≥ 0, gear -1 to 10, etc.)
- Verifies minimum car count (≥3)
- Checks coverage thresholds

**Output**:
- `data/reports/barber_r1_test/validation_*.json`

---

## Step 4: Complete End-to-End Test

Run all stages on a subset of cars:

```bash
# Create complete pipeline test
cat > test_complete_pipeline.py << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.transform import (
    pivot_to_wide_format,
    resample_to_time_grid,
    synchronize_multi_car,
    save_synchronized_data,
)
from src.validation import validate_simulation_ready
from src.utils.logging_utils import get_logger

logger = get_logger("complete_pipeline")

# Test with 2 cars
chassis_ids = ["010", "002"]
data_path = Path("data/processed")
event_name = "barber_r1_complete"

# Load refined data (time-corrected, lap-repaired, position-normalized)
dfs_by_car = {}

for chassis_id in chassis_ids:
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {chassis_id}")
    logger.info(f"{'='*60}")

    # Load refined data (you'll need to point to actual processed data)
    parquet_files = list(data_path.glob(f"barber_r1/chassis_id={chassis_id}/segment_id=*/telemetry_name=*/*.parquet"))

    if not parquet_files:
        logger.error(f"No data found for {chassis_id}")
        continue

    df_long = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    logger.info(f"Loaded {len(df_long):,} rows (long format)")

    # PIVOT
    df_wide, _ = pivot_to_wide_format(df_long, chassis_id)
    logger.info(f"Pivoted to {len(df_wide):,} rows (wide format)")

    # RESAMPLE
    df_resampled, _ = resample_to_time_grid(df_wide, chassis_id)
    logger.info(f"Resampled to {len(df_resampled):,} rows at 20Hz")

    dfs_by_car[chassis_id] = df_resampled

# SYNCHRONIZE
df_sync, sync_stats, coverage = synchronize_multi_car(dfs_by_car, event_name)
logger.info(f"\n✅ Synchronized {len(dfs_by_car)} cars: {len(df_sync):,} total rows")

# SAVE
output_file = save_synchronized_data(df_sync, data_path, event_name)
logger.info(f"✅ Saved to: {output_file}")

# VALIDATE
result = validate_simulation_ready(
    df=df_sync,
    context_root=Path("great_expectations"),
    event_name=event_name,
    output_dir=Path("data/reports"),
)

logger.info(f"\n{'='*60}")
logger.info(f"PIPELINE COMPLETE")
logger.info(f"{'='*60}")
logger.info(f"Pass rate: {result.pass_rate:.1f}%")
logger.info(f"Report: {result.report_path}")
EOF

# Run complete pipeline
python test_complete_pipeline.py
```

---

## Step 5: Check Outputs

### View Synchronized Data

```python
import pandas as pd

# Load synchronized multi-car data
df = pd.read_parquet("data/processed/barber_r1_complete/synchronized/multi_car_frames.parquet")

print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"Cars: {df['chassis_id'].nunique()}")
print(f"Time range: {df['time_global'].min()} to {df['time_global'].max()}")

# Sample data
print("\nSample (first 10 rows):")
print(df.head(10))

# Check car coverage
print("\nCar coverage:")
print(df.groupby('chassis_id')['speed'].count())
```

### View Statistics

```python
import pandas as pd

# Sync stats
sync_stats = pd.read_parquet("data/processed/barber_r1_complete/sync_stats.parquet")
print("Sync Statistics:")
print(sync_stats)

# Per-car coverage
coverage = pd.read_parquet("data/processed/barber_r1_complete/car_coverage.parquet")
print("\nPer-car Coverage:")
print(coverage)
```

---

## Troubleshooting

### Out of Memory Error

**Solution**: Reduce chunk size in processing:

```python
# In your script, before loading data:
import os
os.environ["POLARS_MAX_THREADS"] = "4"

# Or reduce chunk size
CHUNK_SIZE = 500_000  # Instead of 1_000_000
```

### GPU Not Available

The pipeline auto-detects GPU and falls back to CPU. To force CPU mode:

```python
from src.conf.settings import settings
settings.use_gpu = False
```

### Data Not Found Errors

Make sure you've run the previous stages first:
1. Ingestion → processed data exists
2. Time sync → `time_corrected` column exists
3. Lap repair → `lap_repaired` column exists
4. Then run pivot/resample/sync

---

## Expected Processing Times

On a typical system (16GB RAM, no GPU):

| Stage | Cars | Rows | Time |
|-------|------|------|------|
| Ingestion | 20 | 11.5M | ~5 min |
| Time Sync | 4 | 670K | ~2 sec |
| Lap Repair | 2 | 72K | ~17 sec |
| Position | 2 | 72K | ~5 sec |
| Pivot | 2 | 72K | ~3 sec |
| Resample | 2 | 72K | ~5 sec |
| Sync | 2 | varies | ~2 sec |

**Full pipeline (2 cars)**: ~30 seconds

---

## Next Steps

Once you have synchronized data:

1. **Visualization**: Load into plotting tools (Plotly, Matplotlib)
2. **Analysis**: Compute per-lap features, compare drivers
3. **Simulation**: Feed into race replay system
4. **Export**: Convert to other formats (JSON, CSV) if needed

---

## Support

For issues:
- Check `logs/` directory for detailed logs
- Review test result documents (`TIME_SYNC_TEST_RESULTS.md`, etc.)
- See GitHub issues: https://github.com/tradewithmeai/racing-telemetry-pipeline/issues
