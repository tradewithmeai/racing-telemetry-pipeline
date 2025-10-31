"""Ingestion module for telemetry data processing."""

from .ingest import ingest_csv, ChunkedCSVReader
from .partitioning import write_partitioned_parquet
from .anomalies import (
    detect_duplicates,
    detect_backwards_time_per_car,
    detect_gaps_per_car,
    extract_vehicle_identity,
)

__all__ = [
    "ingest_csv",
    "ChunkedCSVReader",
    "write_partitioned_parquet",
    "detect_duplicates",
    "detect_backwards_time_per_car",
    "detect_gaps_per_car",
    "extract_vehicle_identity",
]
