"""Test lap repair on Barber R1 data."""

import sys
from pathlib import Path
import polars as pl
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.transform.time_sync import process_car_time_sync
from src.transform.lap_repair import repair_laps, save_lap_boundaries
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

    # Also get lap number (if present as a column in the original data)
    if "lap" in df_long.columns:
        lap_series = (
            df_long[df_long["telemetry_name"] == df_long["telemetry_name"].iloc[0]]
            .set_index(["timestamp", "meta_time"])["lap"]
        )
        df_wide["lap"] = df_wide.set_index(["timestamp", "meta_time"]).index.map(
            lap_series
        )

    logger.info(f"  Pivoted to wide format: {len(df_wide):,} rows x {len(df_wide.columns)} columns")
    logger.info(f"  Available signals: {', '.join([c for c in df_wide.columns if c not in ['timestamp', 'meta_time', 'lap']])}")

    return df_wide


def test_lap_repair_on_car(
    chassis_id: str,
    event_name: str = "barber_r1",
    track_length_m: float = 3700.0,
    min_lap_duration_sec: float = 85.0,
    max_lap_duration_sec: float = 300.0,
):
    """Test lap repair pipeline on one car.

    Args:
        chassis_id: Chassis identifier
        event_name: Event name
        track_length_m: Track length in meters
        min_lap_duration_sec: Minimum valid lap duration
        max_lap_duration_sec: Maximum valid lap duration
    """
    logger.info(f"\n{'#'*80}")
    logger.info(f"# Lap Repair Test: {chassis_id}")
    logger.info(f"{'#'*80}\n")

    # Step 1: Load data
    df = load_car_data_wide(chassis_id, event_name)

    # Step 2: Run time sync (if not already done)
    logger.info("\n[Step 1] Running time synchronization...")
    df_with_time, calibrations = process_car_time_sync(
        df,
        chassis_id=f"GR86-{chassis_id}",
        window_minutes=5,
        gap_threshold_sec=2.0,
        method="median",
    )

    # Step 3: Run lap repair
    logger.info("\n[Step 2] Running lap repair...")
    df_with_laps, boundaries = repair_laps(
        df_with_time,
        chassis_id=f"GR86-{chassis_id}",
        event=event_name,
        track_length_m=track_length_m,
        min_lap_duration_sec=min_lap_duration_sec,
        max_lap_duration_sec=max_lap_duration_sec,
    )

    # Step 4: Analyze results
    logger.info("\n[Step 3] Analyzing lap repair results...")
    analyze_lap_repair(df_with_laps, boundaries, chassis_id)

    return df_with_laps, boundaries


def analyze_lap_repair(df: pd.DataFrame, boundaries, chassis_id: str):
    """Analyze lap repair results.

    Args:
        df: DataFrame with lap_repaired column
        boundaries: List of LapBoundary objects
        chassis_id: Chassis identifier
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Lap Repair Analysis: {chassis_id}")
    logger.info(f"{'='*60}\n")

    # Original lap stats
    if "lap" in df.columns:
        orig_laps = df["lap"].dropna()
        logger.info(f"[Original Lap Data]")
        logger.info(f"  Rows with lap data: {len(orig_laps):,} / {len(df):,}")
        logger.info(f"  Unique laps: {orig_laps.nunique()}")
        logger.info(f"  Lap range: {orig_laps.min():.0f} - {orig_laps.max():.0f}")

        # Check for sentinels
        num_sentinels = (df["lap"] == 32768).sum()
        if num_sentinels > 0:
            logger.warning(f"  Sentinel values (32768): {num_sentinels:,}")

    # Repaired lap stats
    repaired_laps = df["lap_repaired"].dropna()
    logger.info(f"\n[Repaired Lap Data]")
    logger.info(f"  Rows with lap data: {len(repaired_laps):,} / {len(df):,}")
    logger.info(f"  Unique laps: {repaired_laps.nunique()}")
    if len(repaired_laps) > 0:
        logger.info(f"  Lap range: {repaired_laps.min():.0f} - {repaired_laps.max():.0f}")

    # Boundary stats
    logger.info(f"\n[Lap Boundaries]")
    logger.info(f"  Total boundaries: {len(boundaries)}")

    if len(boundaries) > 0:
        # Reason breakdown
        reason_counts = {}
        for b in boundaries:
            reason_counts[b.reason.value] = reason_counts.get(b.reason.value, 0) + 1

        logger.info(f"  Reasons:")
        for reason, count in sorted(reason_counts.items()):
            logger.info(f"    {reason}: {count}")

        # Confidence stats
        confidences = [b.confidence for b in boundaries]
        avg_confidence = sum(confidences) / len(confidences)
        logger.info(f"  Average confidence: {avg_confidence:.2f}")

        # Duration stats (if multiple boundaries)
        if len(boundaries) > 1:
            durations = []
            for i in range(1, len(boundaries)):
                duration = (
                    boundaries[i].boundary_time - boundaries[i - 1].boundary_time
                ).total_seconds()
                durations.append(duration)

            logger.info(f"\n[Lap Durations]")
            logger.info(f"  Mean: {sum(durations) / len(durations):.1f}s")
            logger.info(f"  Min: {min(durations):.1f}s")
            logger.info(f"  Max: {max(durations):.1f}s")

            # Check for outliers
            short_laps = [d for d in durations if d < 85.0]
            long_laps = [d for d in durations if d > 200.0]

            if short_laps:
                logger.warning(f"  Short laps (<85s): {len(short_laps)}")
            if long_laps:
                logger.warning(f"  Long laps (>200s): {len(long_laps)}")

    # Check for Laptrigger_lapdist_dls availability
    if "Laptrigger_lapdist_dls" in df.columns:
        lapdist = df["Laptrigger_lapdist_dls"].dropna()
        logger.info(f"\n[Track Distance Data]")
        logger.info(f"  Rows with lapdist: {len(lapdist):,} / {len(df):,}")
        if len(lapdist) > 0:
            logger.info(f"  Range: {lapdist.min():.1f}m - {lapdist.max():.1f}m")

            # Count resets
            lapdist_diff = lapdist.diff()
            resets = (lapdist_diff < -100).sum()
            logger.info(f"  Detected resets: {resets}")
    else:
        logger.warning(f"  No Laptrigger_lapdist_dls column found")

    logger.info(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Test on same cars as time sync test
    test_chassis = [
        "010",  # Problematic car with lots of data
        "002",  # Another problematic car
    ]

    all_results = {}
    all_boundaries = {}

    for chassis_id in test_chassis:
        try:
            df, boundaries = test_lap_repair_on_car(
                chassis_id,
                track_length_m=3700.0,  # Barber track length
                min_lap_duration_sec=85.0,  # Barber minimum lap ~90s
                max_lap_duration_sec=300.0,  # 5 minutes max (includes pit stops)
            )
            all_results[chassis_id] = df
            all_boundaries[chassis_id] = boundaries
        except Exception as e:
            logger.error(f"Failed to process chassis {chassis_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save boundaries
    if all_boundaries:
        output_path = Path("data/processed/barber_r1")
        save_lap_boundaries(all_boundaries, output_path, "barber_r1")

    logger.info("\n[SUCCESS] Lap repair test complete!")
    logger.info(f"Processed {len(all_results)} vehicles")
    if all_boundaries:
        total_boundaries = sum(len(b) for b in all_boundaries.values())
        logger.info(f"Detected {total_boundaries} lap boundaries total")
