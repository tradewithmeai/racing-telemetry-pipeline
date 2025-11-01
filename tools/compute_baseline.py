"""Compute baseline expectations from raw curated data.

This tool analyzes the raw curated data (post-ingestion, pre-pipeline) to
establish exact expectations for data quality validation. The baseline is
used to detect data loss, rate drift, truncation, and silent column stripping.

Usage:
    python tools/compute_baseline.py --event barber_r1 --chassis 010 002

Output:
    data/baselines/barber_r1_baseline.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import json
import argparse
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def analyze_signal_baseline(
    df_signal: pd.DataFrame,
    signal_name: str,
    time_col: str = "timestamp",
) -> Dict:
    """Analyze a single signal to compute baseline statistics.

    Args:
        df_signal: DataFrame for one signal (long format)
        signal_name: Name of the signal
        time_col: Time column name

    Returns:
        Dict with baseline stats
    """
    # Convert timestamp to datetime
    if not pd.api.types.is_datetime64_any_dtype(df_signal[time_col]):
        df_signal[time_col] = pd.to_datetime(df_signal[time_col])

    df_signal = df_signal.sort_values(time_col)

    # Basic counts
    sample_count = len(df_signal)
    non_null_count = df_signal["telemetry_value"].notna().sum()

    # Time range
    time_start = df_signal[time_col].min()
    time_end = df_signal[time_col].max()
    time_span_sec = (time_end - time_start).total_seconds()

    # Sampling rate (average Hz)
    hz_avg = sample_count / time_span_sec if time_span_sec > 0 else 0

    # Coverage percentage
    coverage_pct = 100.0 * non_null_count / sample_count if sample_count > 0 else 0

    return {
        "signal_name": signal_name,
        "sample_count": int(sample_count),
        "non_null_count": int(non_null_count),
        "coverage_pct": round(coverage_pct, 2),
        "time_start": time_start.isoformat(),
        "time_end": time_end.isoformat(),
        "time_span_sec": round(time_span_sec, 2),
        "hz_avg": round(hz_avg, 2),
    }


def compute_car_baseline(
    data_path: Path,
    event_name: str,
    chassis_id: str,
) -> Dict:
    """Compute baseline for a single car.

    Args:
        data_path: Path to processed data directory
        event_name: Event name (e.g., 'barber_r1')
        chassis_id: Chassis ID (e.g., '010')

    Returns:
        Dict with car baseline stats
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Computing baseline for chassis {chassis_id}")
    logger.info(f"{'='*80}")

    # Find all signal parquet files
    base_path = data_path / event_name / "raw_curated" / f"event={event_name}"
    parquet_pattern = f"chassis_id={chassis_id}/telemetry_name=*/*.parquet"

    parquet_files = list(base_path.glob(parquet_pattern))

    if not parquet_files:
        raise FileNotFoundError(
            f"No raw curated data found for chassis {chassis_id} at {base_path}"
        )

    logger.info(f"  Found {len(parquet_files)} signal files")

    # Group files by signal
    signals_by_name = {}
    for file_path in parquet_files:
        # Extract signal name from path
        signal_dir = file_path.parent.name
        signal_name = signal_dir.replace("telemetry_name=", "")

        if signal_name not in signals_by_name:
            signals_by_name[signal_name] = []

        signals_by_name[signal_name].append(file_path)

    logger.info(f"  Unique signals: {len(signals_by_name)}")

    # Analyze each signal
    signal_baselines = {}
    total_samples = 0
    car_time_start = None
    car_time_end = None

    for signal_name, signal_files in sorted(signals_by_name.items()):
        logger.info(f"\n  Analyzing signal: {signal_name}")

        # Load all files for this signal
        dfs = []
        for file_path in signal_files:
            df = pd.read_parquet(file_path)
            dfs.append(df)

        df_signal = pd.concat(dfs, ignore_index=True)
        logger.info(f"    Loaded {len(df_signal):,} samples")

        # Analyze signal
        signal_baseline = analyze_signal_baseline(df_signal, signal_name)
        signal_baselines[signal_name] = signal_baseline

        total_samples += signal_baseline["sample_count"]

        # Track overall time range
        signal_start = pd.to_datetime(signal_baseline["time_start"])
        signal_end = pd.to_datetime(signal_baseline["time_end"])

        if car_time_start is None or signal_start < car_time_start:
            car_time_start = signal_start
        if car_time_end is None or signal_end > car_time_end:
            car_time_end = signal_end

        logger.info(f"    Coverage: {signal_baseline['coverage_pct']:.1f}%")
        logger.info(f"    Avg Hz: {signal_baseline['hz_avg']:.2f}")

    # Compute overall stats
    car_time_span_sec = (car_time_end - car_time_start).total_seconds()

    car_baseline = {
        "chassis_id": chassis_id,
        "total_samples": int(total_samples),
        "unique_signals": len(signal_baselines),
        "time_start": car_time_start.isoformat(),
        "time_end": car_time_end.isoformat(),
        "time_span_sec": round(car_time_span_sec, 2),
        "signals": signal_baselines,
    }

    logger.info(f"\n  Summary:")
    logger.info(f"    Total samples: {total_samples:,}")
    logger.info(f"    Unique signals: {len(signal_baselines)}")
    logger.info(f"    Time span: {car_time_span_sec / 60:.1f} minutes")

    return car_baseline


def compute_baseline(
    data_path: Path,
    event_name: str,
    chassis_ids: List[str],
    output_path: Path = None,
) -> Dict:
    """Compute baseline for multiple cars.

    Args:
        data_path: Path to processed data directory
        event_name: Event name
        chassis_ids: List of chassis IDs
        output_path: Optional custom output path

    Returns:
        Dict with complete baseline
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"BASELINE COMPUTATION: {event_name}")
    logger.info(f"{'='*80}")
    logger.info(f"  Cars: {chassis_ids}")
    logger.info(f"{'='*80}")

    # Compute baseline for each car
    cars_baselines = {}

    for chassis_id in chassis_ids:
        try:
            car_baseline = compute_car_baseline(data_path, event_name, chassis_id)
            cars_baselines[chassis_id] = car_baseline
        except Exception as e:
            logger.error(f"Failed to compute baseline for chassis {chassis_id}: {e}")
            continue

    # Create complete baseline
    baseline = {
        "event_name": event_name,
        "generated_at": datetime.now().isoformat(),
        "data_path": str(data_path),
        "chassis_ids": chassis_ids,
        "cars": cars_baselines,
    }

    # Save baseline
    if output_path is None:
        output_path = Path("data/baselines") / f"{event_name}_baseline.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info(f"BASELINE SAVED: {output_path}")
    logger.info(f"{'='*80}")
    logger.info(f"  Cars analyzed: {len(cars_baselines)}")
    logger.info(f"  Total signals: {sum(b['unique_signals'] for b in cars_baselines.values())}")
    logger.info(f"{'='*80}\n")

    return baseline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute baseline expectations from raw curated data"
    )

    parser.add_argument(
        "--event",
        default="barber_r1",
        help="Event name (default: barber_r1)",
    )

    parser.add_argument(
        "--chassis",
        nargs="+",
        default=["010", "002"],
        help="Chassis IDs to process (default: 010 002)",
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed data directory (default: data/processed)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Custom output path for baseline JSON",
    )

    args = parser.parse_args()

    # Compute baseline
    baseline = compute_baseline(
        data_path=args.data_path,
        event_name=args.event,
        chassis_ids=args.chassis,
        output_path=args.output,
    )

    # Print summary
    print("\n" + "="*80)
    print("BASELINE COMPUTATION COMPLETE")
    print("="*80)
    print(f"\nEvent: {baseline['event_name']}")
    print(f"Cars: {len(baseline['cars'])}")
    print(f"\nPer-car summary:")

    for chassis_id, car_baseline in baseline['cars'].items():
        print(f"\n  Chassis {chassis_id}:")
        print(f"    Samples: {car_baseline['total_samples']:,}")
        print(f"    Signals: {car_baseline['unique_signals']}")
        print(f"    Time span: {car_baseline['time_span_sec'] / 60:.1f} minutes")
        print(f"    Date range: {car_baseline['time_start']} to {car_baseline['time_end']}")


if __name__ == "__main__":
    main()
