"""Test position normalization on Barber R1 data."""

import sys
from pathlib import Path
import polars as pl
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.transform.time_sync import process_car_time_sync
from src.transform.lap_repair import repair_laps
from src.transform.position import normalize_position_data, save_position_quality_report
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_car_data_wide(chassis_id: str, event_name: str = "barber_r1") -> pd.DataFrame:
    """Load all telemetry for one car and pivot to wide format.

    Args:
        chassis_id: Chassis identifier (e.g., "002", "004")
        event_name: Event name

    Returns:
        DataFrame with columns: timestamp, meta_time, lap, speed, aps, Laptrigger_lapdist_dls, etc.
    """
    logger.info(f"Loading data for chassis {chassis_id}")

    base_path = (
        Path("data/processed")
        / event_name
        / "raw_curated"
        / f"event={event_name}"
        / f"chassis_id={chassis_id}"
    )

    if not base_path.exists():
        raise FileNotFoundError(f"No data found for chassis {chassis_id} at {base_path}")

    # Load all telemetry signals
    all_dfs = []
    for signal_dir in base_path.iterdir():
        if signal_dir.is_dir():
            signal_name = signal_dir.name.replace("telemetry_name=", "")

            # Read all parquet files in this signal partition
            parquet_files = list(signal_dir.glob("*.parquet"))

            for pq_file in parquet_files:
                df_signal = pl.read_parquet(pq_file).to_pandas()
                df_signal["telemetry_name"] = signal_name
                all_dfs.append(df_signal)

    if not all_dfs:
        raise ValueError(f"No data loaded for chassis {chassis_id}")

    # Combine all signals
    df_long = pd.concat(all_dfs, ignore_index=True)

    logger.info(f"  Loaded {len(df_long):,} rows (long format)")

    # Pivot to wide format
    df_wide = df_long.pivot_table(
        index=["timestamp", "meta_time"],
        columns="telemetry_name",
        values="telemetry_value",
        aggfunc="first",
    ).reset_index()

    # Also get lap number
    if "lap" in df_long.columns:
        lap_series = (
            df_long[df_long["telemetry_name"] == df_long["telemetry_name"].iloc[0]]
            .set_index(["timestamp", "meta_time"])["lap"]
        )
        df_wide["lap"] = df_wide.set_index(["timestamp", "meta_time"]).index.map(
            lap_series
        )

    logger.info(f"  Pivoted to wide format: {len(df_wide):,} rows x {len(df_wide.columns)} columns")

    return df_wide


def test_full_pipeline_on_car(
    chassis_id: str,
    event_name: str = "barber_r1",
    circuit: str = "barber",
):
    """Test complete pipeline: time sync + lap repair + position normalization.

    Args:
        chassis_id: Chassis identifier
        event_name: Event name
        circuit: Circuit name
    """
    logger.info(f"\n{'#'*80}")
    logger.info(f"# Full Pipeline Test: {chassis_id}")
    logger.info(f"{'#'*80}\n")

    # Step 1: Load data
    df = load_car_data_wide(chassis_id, event_name)

    # Step 2: Time synchronization
    logger.info("\n[Step 1] Running time synchronization...")
    df, calibrations = process_car_time_sync(
        df,
        chassis_id=f"GR86-{chassis_id}",
        window_minutes=5,
        gap_threshold_sec=2.0,
        method="median",
    )

    # Step 3: Lap repair
    logger.info("\n[Step 2] Running lap repair...")
    df, boundaries = repair_laps(
        df,
        chassis_id=f"GR86-{chassis_id}",
        event=event_name,
        track_length_m=3700.0,
        min_lap_duration_sec=85.0,
        max_lap_duration_sec=300.0,
    )

    # Step 4: Position normalization
    logger.info("\n[Step 3] Running position normalization...")
    df, position_quality = normalize_position_data(
        df,
        chassis_id=f"GR86-{chassis_id}",
        circuit=circuit,
        max_jump_meters=500.0,
        interpolate=True,
    )

    # Step 5: Analyze results
    logger.info("\n[Step 4] Analyzing position normalization results...")
    analyze_position_data(df, position_quality, chassis_id)

    return df, position_quality


def analyze_position_data(df: pd.DataFrame, quality: object, chassis_id: str):
    """Analyze position normalization results.

    Args:
        df: DataFrame with position columns
        quality: PositionQuality object
        chassis_id: Chassis identifier
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Position Analysis: {chassis_id}")
    logger.info(f"{'='*60}\n")

    # GPS data
    if "gps_lat" in df.columns and "gps_lon" in df.columns:
        logger.info("[GPS Data]")
        logger.info(f"  Coverage: {quality.gps_coverage_pct:.1f}%")

        if quality.gps_coverage_pct > 0:
            logger.info(f"  Latitude range: {df['gps_lat'].min():.6f} to {df['gps_lat'].max():.6f}")
            logger.info(f"  Longitude range: {df['gps_lon'].min():.6f} to {df['gps_lon'].max():.6f}")
            logger.info(f"  Outliers detected: {quality.outliers_detected}")
            logger.info(f"  Out of bounds: {quality.out_of_bounds}")
            logger.info(f"  Interpolated points: {quality.interpolated_points}")

    # Track distance
    if "track_distance_m" in df.columns:
        logger.info(f"\n[Track Distance]")
        logger.info(f"  Coverage: {quality.lapdist_coverage_pct:.1f}%")

        if quality.lapdist_coverage_pct > 0:
            logger.info(
                f"  Range: {df['track_distance_m'].min():.1f}m to {df['track_distance_m'].max():.1f}m"
            )

    # Quality score
    logger.info(f"\n[Position Quality]")
    logger.info(f"  Overall score: {quality.quality_score:.2f} / 1.00")

    if quality.quality_score >= 0.8:
        logger.info(f"  Rating: EXCELLENT")
    elif quality.quality_score >= 0.6:
        logger.info(f"  Rating: GOOD")
    elif quality.quality_score >= 0.4:
        logger.info(f"  Rating: ACCEPTABLE")
    else:
        logger.warning(f"  Rating: POOR")

    # Check for raw columns preservation
    logger.info(f"\n[Raw Data Preservation]")
    raw_cols = [c for c in df.columns if c.endswith("_raw")]
    logger.info(f"  Raw columns preserved: {len(raw_cols)}")
    if raw_cols:
        for col in raw_cols:
            logger.info(f"    - {col}")

    # Final data structure
    logger.info(f"\n[Output Data Structure]")
    logger.info(f"  Total rows: {len(df):,}")
    logger.info(f"  Total columns: {len(df.columns)}")

    # List key columns
    key_cols = [
        "time_corrected",
        "lap_repaired",
        "gps_lat",
        "gps_lon",
        "track_distance_m",
        "speed",
        "aps",
    ]
    available_key_cols = [c for c in key_cols if c in df.columns]
    logger.info(f"  Key columns present: {', '.join(available_key_cols)}")

    logger.info(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Test on same cars as before
    test_chassis = [
        "010",  # Problematic car with lots of data
        "002",  # Another problematic car
    ]

    all_results = {}
    all_quality = {}

    for chassis_id in test_chassis:
        try:
            df, quality = test_full_pipeline_on_car(chassis_id, circuit="barber")
            all_results[chassis_id] = df
            all_quality[chassis_id] = quality
        except Exception as e:
            logger.error(f"Failed to process chassis {chassis_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save position quality report
    if all_quality:
        output_path = Path("data/processed/barber_r1")
        save_position_quality_report(all_quality, output_path, "barber_r1")

    logger.info("\n[SUCCESS] Position normalization test complete!")
    logger.info(f"Processed {len(all_results)} vehicles")

    if all_quality:
        avg_quality = sum(q.quality_score for q in all_quality.values()) / len(all_quality)
        logger.info(f"Average position quality score: {avg_quality:.2f}")
