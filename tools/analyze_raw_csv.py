"""Analyze raw CSV telemetry data BEFORE processing.

This tool inspects the original CSV files to understand:
- What columns are present
- How much data exists per car
- What the sampling rates are
- Whether GPS columns exist
- Time ranges and data quality

Use this to set baseline expectations BEFORE running the pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import argparse


def analyze_raw_csv(
    csv_path: Path,
    sample_size: int = 100000,
) -> Dict:
    """Analyze a single raw CSV file.

    Args:
        csv_path: Path to raw CSV file
        sample_size: Number of rows to sample for detailed analysis

    Returns:
        Dict with analysis results
    """
    print(f"\nAnalyzing: {csv_path.name}")
    print("=" * 80)

    # Read header to get column names
    df_header = pd.read_csv(csv_path, nrows=0)
    columns = df_header.columns.tolist()

    print(f"Total columns: {len(columns)}")
    print(f"\nColumns present:")
    for col in columns:
        print(f"  - {col}")

    # Read full file for complete analysis
    print(f"\nReading full CSV...")
    df = pd.read_csv(csv_path)

    total_rows = len(df)
    print(f"Total rows: {total_rows:,}")

    # Check required columns
    required_cols = {
        'timestamp': 'timestamp',
        'vehicle_id': 'vehicle_id',
        'chassis_id': 'chassis_id',
        'car_no': 'car_no',
        'lap': 'lap',
        'signal_name': 'telemetry_name',
        'signal_value': 'telemetry_value',
    }

    print(f"\nRequired columns check:")
    for name, col in required_cols.items():
        if col in columns:
            print(f"  [OK] {name}: '{col}' present")
        else:
            print(f"  [X] {name}: '{col}' MISSING")

    # GPS column detection
    gps_columns = []
    gps_keywords = ['gps', 'lat', 'lon', 'vbox', 'latitude', 'longitude']

    print(f"\nGPS-related columns:")
    for col in columns:
        if any(keyword in col.lower() for keyword in gps_keywords):
            gps_columns.append(col)
            print(f"  - {col}")

    if not gps_columns:
        print("  [!] WARNING: No GPS columns detected!")

    # Analyze by chassis_id
    print(f"\n{'='*80}")
    print("PER-CAR ANALYSIS")
    print("=" * 80)

    chassis_ids = df['chassis_id'].unique()
    print(f"\nCars found: {len(chassis_ids)}")

    car_stats = {}

    for chassis_id in sorted(chassis_ids):
        if pd.isna(chassis_id):
            continue

        car_df = df[df['chassis_id'] == chassis_id].copy()

        print(f"\n{'='*80}")
        print(f"CHASSIS: {chassis_id}")
        print("=" * 80)

        # Basic counts
        row_count = len(car_df)
        print(f"Rows: {row_count:,} ({row_count/total_rows*100:.1f}% of total)")

        # Time range
        if 'timestamp' in car_df.columns:
            car_df['timestamp'] = pd.to_datetime(car_df['timestamp'])
            min_time = car_df['timestamp'].min()
            max_time = car_df['timestamp'].max()
            time_span_sec = (max_time - min_time).total_seconds()
            time_span_min = time_span_sec / 60

            print(f"\nTime range:")
            print(f"  Start: {min_time}")
            print(f"  End:   {max_time}")
            print(f"  Span:  {time_span_min:.1f} minutes ({time_span_sec:.0f} seconds)")

        # Unique signals
        if 'telemetry_name' in car_df.columns:
            signals = car_df['telemetry_name'].unique()
            print(f"\nUnique signals: {len(signals)}")

            # Signal breakdown
            signal_counts = car_df['telemetry_name'].value_counts()
            print(f"\nSignal counts:")
            for signal, count in signal_counts.items():
                pct = count / row_count * 100
                print(f"  {signal:30s}: {count:8,} ({pct:5.1f}%)")

        # GPS analysis
        if gps_columns:
            print(f"\nGPS Data Analysis:")
            for gps_col in gps_columns:
                if gps_col in car_df.columns:
                    # Check if GPS data exists in long format
                    gps_rows = car_df[car_df['telemetry_name'] == gps_col]
                    if len(gps_rows) > 0:
                        non_null = gps_rows['telemetry_value'].notna().sum()
                        coverage = non_null / len(gps_rows) * 100 if len(gps_rows) > 0 else 0
                        print(f"  {gps_col:30s}: {non_null:8,} values ({coverage:5.1f}% coverage)")

                        # Show sample values
                        sample_vals = gps_rows['telemetry_value'].dropna().head(5).tolist()
                        if sample_vals:
                            print(f"    Sample values: {sample_vals}")

        # Laps
        if 'lap' in car_df.columns:
            unique_laps = car_df['lap'].nunique()
            min_lap = car_df['lap'].min()
            max_lap = car_df['lap'].max()
            print(f"\nLaps:")
            print(f"  Unique laps: {unique_laps}")
            print(f"  Lap range: {min_lap} to {max_lap}")

        # Estimated sampling rate
        if 'timestamp' in car_df.columns and row_count > 0:
            hz_estimate = row_count / time_span_sec if time_span_sec > 0 else 0
            print(f"\nEstimated sampling rate: {hz_estimate:.1f} Hz (total samples / time span)")
            print(f"  Note: This is aggregate rate across all signals")

        # Store stats
        car_stats[chassis_id] = {
            'row_count': row_count,
            'time_span_sec': time_span_sec if 'timestamp' in car_df.columns else None,
            'unique_signals': len(signals) if 'telemetry_name' in car_df.columns else 0,
            'unique_laps': unique_laps if 'lap' in car_df.columns else 0,
        }

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)

    print(f"\nTotal rows: {total_rows:,}")
    print(f"Total cars: {len(chassis_ids)}")
    print(f"Total columns: {len(columns)}")
    print(f"GPS columns found: {len(gps_columns)}")

    if gps_columns:
        print(f"\nGPS columns:")
        for col in gps_columns:
            print(f"  - {col}")
    else:
        print(f"\n[!] WARNING: No GPS columns detected!")
        print(f"    This will result in 0% GPS coverage in processed data.")

    return {
        'csv_path': str(csv_path),
        'total_rows': total_rows,
        'total_columns': len(columns),
        'columns': columns,
        'gps_columns': gps_columns,
        'chassis_ids': sorted([c for c in chassis_ids if not pd.isna(c)]),
        'car_stats': car_stats,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze raw CSV telemetry data')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to raw CSV file',
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100000,
        help='Number of rows to sample for detailed analysis (default: 100000)',
    )

    args = parser.parse_args()

    csv_path = Path(args.input)

    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        return

    results = analyze_raw_csv(csv_path, sample_size=args.sample_size)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    # Key findings
    print(f"\nKey Findings:")
    print(f"  - Total rows: {results['total_rows']:,}")
    print(f"  - Cars: {len(results['chassis_ids'])}")
    print(f"  - GPS columns: {len(results['gps_columns'])}")

    if not results['gps_columns']:
        print(f"\n[!] CRITICAL: No GPS columns found in raw CSV!")
        print(f"    Expected columns like: VBOX_Lat_Min, VBOX_Long_Minutes")
        print(f"    Without GPS, position-based simulation will fail.")


if __name__ == '__main__':
    main()
