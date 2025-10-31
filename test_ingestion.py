"""Test ingestion system on Barber R1 telemetry data."""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.ingest import ingest_csv
from src.utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger(
    "test_ingestion",
    log_level="INFO",
    log_file=f"test_ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    log_dir="logs",
)


def main():
    """Run ingestion test on Barber R1 data."""

    logger.info("="*80)
    logger.info("TELEMETRY INGESTION TEST - BARBER R1")
    logger.info("="*80)

    # Input file
    input_file = Path("barber-motorsports-park/barber/R1_barber_telemetry_data.csv")

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Expected location: barber-motorsports-park/barber/R1_barber_telemetry_data.csv")
        logger.info("Please ensure the Barber data directory is in the project root")
        return 1

    # Output directory
    output_dir = Path("data/processed")

    # File info
    file_size_mb = input_file.stat().st_size / (1024**2)

    logger.info(f"Input file: {input_file}")
    logger.info(f"File size: {file_size_mb:.2f} MB")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)

    # Configuration
    chunk_size = 500_000  # Smaller chunks for testing (500K rows)
    use_gpu = True  # Will auto-fallback to CPU if no GPU available

    logger.info("Configuration:")
    logger.info(f"  - Chunk size: {chunk_size:,} rows")
    logger.info(f"  - GPU mode: {use_gpu} (auto-fallback to CPU)")
    logger.info(f"  - Anomaly detection: Enabled")
    logger.info("="*80)

    # Run ingestion
    try:
        logger.info("Starting ingestion...")
        logger.info("")

        stats = ingest_csv(
            file_path=input_file,
            output_dir=output_dir,
            event_name="barber_r1",
            chunk_size=chunk_size,
            use_gpu=use_gpu,
            detect_anomalies=True,
        )

        logger.info("")
        logger.info("="*80)
        logger.info("INGESTION COMPLETED SUCCESSFULLY!")
        logger.info("="*80)

        # Detailed statistics
        logger.info("")
        logger.info("Processing Statistics:")
        logger.info(f"  Input file hash: {stats.input_hash}")
        logger.info(f"  Total rows read: {stats.total_rows:,}")
        logger.info(f"  Rows processed: {stats.rows_processed:,}")
        logger.info(f"  Duplicates removed: {stats.rows_dropped_duplicates:,}")
        logger.info(f"  Unique vehicles: {stats.unique_vehicles}")
        logger.info(f"  Unique signals: {stats.unique_signals}")
        logger.info("")

        logger.info("Performance:")
        logger.info(f"  Processing time: {stats.processing_time_sec:.2f} seconds")
        throughput = stats.total_rows / stats.processing_time_sec
        logger.info(f"  Throughput: {throughput:,.0f} rows/second")
        logger.info(f"  Speed: {file_size_mb / stats.processing_time_sec:.2f} MB/s")
        logger.info("")

        # Data quality issues
        if stats.backwards_time_count:
            logger.warning("Data Quality Issues Detected:")
            logger.warning(f"  Backwards timestamps found in {len(stats.backwards_time_count)} vehicles:")
            for vid, count in sorted(stats.backwards_time_count.items()):
                logger.warning(f"    - {vid}: {count} backwards timestamps")
            logger.warning("")

        if stats.gaps_count > 0:
            logger.warning(f"  Time gaps detected: {stats.gaps_count} total gaps >2s")
            logger.warning("  (Data will be segmented at these gaps)")
            logger.warning("")

        if not stats.backwards_time_count and stats.gaps_count == 0:
            logger.info("Data Quality: EXCELLENT (no anomalies detected)")
            logger.info("")

        # Output files
        logger.info("Output:")
        output_path = output_dir / "barber_r1"
        logger.info(f"  - Raw curated Parquet: {output_path / 'raw_curated'}")
        logger.info(f"  - Ingestion stats: {output_path / 'ingestion_stats.json'}")
        logger.info("")

        # Check output files
        raw_curated_dir = output_path / "raw_curated"
        if raw_curated_dir.exists():
            parquet_files = list(raw_curated_dir.rglob("*.parquet"))
            logger.info(f"  Parquet files written: {len(parquet_files)}")

            if parquet_files:
                # Show sample file structure
                logger.info("")
                logger.info("  Sample partition structure:")
                for i, pfile in enumerate(parquet_files[:5]):
                    relative_path = pfile.relative_to(output_dir)
                    logger.info(f"    {relative_path}")
                if len(parquet_files) > 5:
                    logger.info(f"    ... and {len(parquet_files) - 5} more files")

        logger.info("")
        logger.info("="*80)
        logger.info("TEST PASSED ✓")
        logger.info("="*80)
        return 0

    except Exception as e:
        logger.error("")
        logger.error("="*80)
        logger.error("INGESTION FAILED!")
        logger.error("="*80)
        logger.error(f"Error: {e}")
        logger.error("", exc_info=True)
        logger.error("")
        logger.error("="*80)
        logger.error("TEST FAILED ✗")
        logger.error("="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
