"""CSV ingestion with GPU acceleration and chunked reading."""

import pandas as pd
import polars as pl
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
#from tqdm import tqdm

from src.conf.settings import settings
from src.schemas.raw import RawCSVSchema, VehicleIdentity
from src.utils.logging_utils import get_logger
from src.utils.io_utils import ensure_dir, compute_file_hash, save_json
from .anomalies import (
    detect_duplicates,
    detect_backwards_time_per_car,
    detect_gaps_per_car,
    validate_schema,
    extract_vehicle_identity,
)
from .partitioning import write_raw_curated

logger = get_logger(__name__)


@dataclass
class IngestionStats:
    """Statistics from ingestion process."""

    file_path: str
    file_size_mb: float
    input_hash: str
    total_rows: int
    rows_processed: int
    rows_dropped_duplicates: int
    unique_vehicles: int
    unique_signals: int
    backwards_time_count: Dict[str, int]
    gaps_count: int
    processing_time_sec: float
    timestamp: str


class ChunkedCSVReader:
    """Chunked CSV reader with GPU/CPU support."""

    def __init__(
        self,
        file_path: str | Path,
        chunk_size: int = 1_000_000,
        use_gpu: bool = True,
    ):
        """Initialize chunked reader.

        Args:
            file_path: Path to CSV file
            chunk_size: Number of rows per chunk
            use_gpu: Whether to use GPU (cuDF) if available
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.use_gpu = use_gpu and settings.use_gpu

        # Try to use GPU
        if self.use_gpu:
            try:
                import cudf

                self.backend = "cudf"
                logger.info("Using cuDF (GPU) for CSV reading")
            except ImportError:
                self.backend = "polars"
                logger.warning("cuDF not available, falling back to Polars")
                self.use_gpu = False
        else:
            self.backend = "polars"
            logger.info("Using Polars (CPU) for CSV reading")

    def read_chunks(self) -> Iterator[pd.DataFrame]:
        """Read CSV in chunks.

        Yields:
            DataFrame chunks (pandas format for consistency)
        """
        if self.backend == "cudf":
            yield from self._read_cudf_chunks()
        else:
            yield from self._read_polars_chunks()

    def _read_cudf_chunks(self) -> Iterator[pd.DataFrame]:
        """Read using cuDF in chunks."""
        import cudf

        # cuDF doesn't have built-in chunking, so we use skiprows
        total_rows = None
        rows_read = 0

        while True:
            try:
                chunk = cudf.read_csv(
                    self.file_path,
                    skiprows=range(1, rows_read + 1) if rows_read > 0 else None,
                    nrows=self.chunk_size,
                )

                if len(chunk) == 0:
                    break

                rows_read += len(chunk)

                # Convert to pandas for consistency
                yield chunk.to_pandas()

                if len(chunk) < self.chunk_size:
                    # Last chunk
                    break

            except Exception as e:
                logger.error(f"Error reading chunk at row {rows_read}: {e}")
                break

    def _read_polars_chunks(self) -> Iterator[pd.DataFrame]:
        """Read using Polars lazy scan."""
        # Use lazy reading for memory efficiency
        try:
            df_lazy = pl.scan_csv(self.file_path)

            # Process in chunks
            total_rows = 0
            offset = 0

            while True:
                chunk = df_lazy.slice(offset, self.chunk_size).collect()

                if chunk.height == 0:
                    break

                offset += chunk.height
                total_rows += chunk.height

                # Convert to pandas for consistency
                yield chunk.to_pandas()

                if chunk.height < self.chunk_size:
                    # Last chunk
                    break

        except Exception as e:
            logger.error(f"Error reading with Polars: {e}. Falling back to pandas.")
            # Fallback to pandas
            yield from pd.read_csv(self.file_path, chunksize=self.chunk_size)


def ingest_csv(
    file_path: str | Path,
    output_dir: str | Path,
    event_name: str,
    chunk_size: int = 1_000_000,
    use_gpu: bool = True,
    detect_anomalies: bool = True,
) -> IngestionStats:
    """Ingest CSV telemetry file with anomaly detection.

    Args:
        file_path: Path to input CSV
        output_dir: Output directory for raw_curated data
        event_name: Event identifier
        chunk_size: Rows per chunk
        use_gpu: Use GPU acceleration if available
        detect_anomalies: Run anomaly detection

    Returns:
        IngestionStats with processing summary
    """
    start_time = datetime.now()
    file_path = Path(file_path)

    logger.info(f"Starting ingestion of {file_path}")
    logger.info(f"File size: {file_path.stat().st_size / (1024**2):.2f} MB")

    # Compute input hash for reproducibility
    input_hash = compute_file_hash(file_path)
    logger.info(f"Input file hash: {input_hash}")

    # Create reader
    reader = ChunkedCSVReader(file_path, chunk_size=chunk_size, use_gpu=use_gpu)

    # Expected schema
    schema = RawCSVSchema()
    required_cols = schema.required_columns

    # Statistics
    total_rows = 0
    rows_processed = 0
    rows_dropped = 0
    all_vehicles = set()
    all_signals = set()
    all_backwards_time = {}
    all_gaps = {}

    # Process chunks
    chunks_processed = 0

    for chunk_df in reader.read_chunks():
        chunks_processed += 1
        total_rows += len(chunk_df)

        # Log progress
        if chunks_processed % 5 == 0 or chunks_processed == 1:
            logger.info(f"Processing chunk {chunks_processed}: {total_rows:,} rows processed so far")

        # Validate schema
        if chunks_processed == 1:
            validate_schema(chunk_df, required_cols)

        # Make a copy to avoid SettingWithCopyWarning
        chunk_df = chunk_df.copy()

        # Convert timestamp columns to datetime
        chunk_df["timestamp"] = pd.to_datetime(chunk_df["timestamp"])
        chunk_df["meta_time"] = pd.to_datetime(chunk_df["meta_time"])

        # Detect anomalies
        if detect_anomalies:
            # Duplicates
            chunk_df, num_dups = detect_duplicates(chunk_df)
            rows_dropped += num_dups

            # Backwards time
            backwards_counts = detect_backwards_time_per_car(chunk_df)
            for vid, count in backwards_counts.items():
                all_backwards_time[vid] = all_backwards_time.get(vid, 0) + count

            # Gaps
            gaps = detect_gaps_per_car(chunk_df, gap_threshold_sec=2.0)
            for vid, gap_list in gaps.items():
                if vid not in all_gaps:
                    all_gaps[vid] = []
                all_gaps[vid].extend(gap_list)

        # Extract vehicle identities
        unique_vehicles = chunk_df["vehicle_id"].unique()
        for vid in unique_vehicles:
            identity = extract_vehicle_identity(vid)
            chunk_df.loc[chunk_df["vehicle_id"] == vid, "chassis_id"] = identity["chassis_id"]
            chunk_df.loc[chunk_df["vehicle_id"] == vid, "car_no"] = identity["car_no"]
            all_vehicles.add(identity["chassis_id"])

        # Track signals
        unique_signals = chunk_df["telemetry_name"].unique()
        all_signals.update(unique_signals)

        rows_processed += len(chunk_df)

        # Write chunk to partitioned Parquet
        write_raw_curated(
            chunk_df,
            event_name=event_name,
            output_base=output_dir,
            compression="snappy",
        )

    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds()

    # Create stats
    stats = IngestionStats(
        file_path=str(file_path),
        file_size_mb=file_path.stat().st_size / (1024**2),
        input_hash=input_hash,
        total_rows=total_rows,
        rows_processed=rows_processed,
        rows_dropped_duplicates=rows_dropped,
        unique_vehicles=len(all_vehicles),
        unique_signals=len(all_signals),
        backwards_time_count=all_backwards_time,
        gaps_count=sum(len(gaps) for gaps in all_gaps.values()),
        processing_time_sec=processing_time,
        timestamp=datetime.now().isoformat(),
    )

    logger.info("="*60)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total rows: {total_rows:,}")
    logger.info(f"  Rows processed: {rows_processed:,}")
    logger.info(f"  Duplicates dropped: {rows_dropped:,}")
    logger.info(f"  Unique vehicles: {len(all_vehicles)}")
    logger.info(f"  Unique signals: {len(all_signals)}")
    logger.info(f"  Backwards time events: {len(all_backwards_time)}")
    logger.info(f"  Time gaps: {stats.gaps_count}")
    logger.info(f"  Processing time: {processing_time:.2f}s")
    logger.info(f"  Throughput: {total_rows/processing_time:,.0f} rows/sec")
    logger.info("="*60)

    # Save stats
    stats_path = Path(output_dir) / event_name / "ingestion_stats.json"
    ensure_dir(stats_path.parent)
    save_json(stats.__dict__, stats_path)

    return stats
