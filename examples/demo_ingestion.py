"""Demo script for testing telemetry ingestion."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.ingest import ingest_csv
from src.utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger(
    "demo_ingestion",
    log_level="INFO",
    log_file="demo_ingestion.log",
    log_dir="logs",
)


def main():
    """Run ingestion demo on Barber data."""

    # Input file
    input_file = Path("barber-motorsports-park/barber/R1_barber_telemetry_data.csv")

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please ensure the Barber data is in the project directory")
        return

    # Output directory
    output_dir = Path("data/processed")

    logger.info("="*60)
    logger.info("Telemetry Ingestion Demo")
    logger.info("="*60)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("="*60)

    # Run ingestion
    try:
        stats = ingest_csv(
            file_path=input_file,
            output_dir=output_dir,
            event_name="barber_r1",
            chunk_size=1_000_000,  # 1M rows per chunk
            use_gpu=True,  # Will auto-fallback to CPU if no GPU
            detect_anomalies=True,
        )

        logger.info("")
        logger.info("Ingestion completed successfully!")
        logger.info(f"Stats saved to: {output_dir}/barber_r1/ingestion_stats.json")

        # Print key stats
        logger.info("")
        logger.info("Key Statistics:")
        logger.info(f"  - Total rows: {stats.total_rows:,}")
        logger.info(f"  - Unique vehicles: {stats.unique_vehicles}")
        logger.info(f"  - Unique signals: {stats.unique_signals}")
        logger.info(f"  - Processing time: {stats.processing_time_sec:.2f}s")
        logger.info(f"  - Throughput: {stats.total_rows/stats.processing_time_sec:,.0f} rows/sec")

        if stats.backwards_time_count:
            logger.warning(f"  - Backwards time events detected: {stats.backwards_time_count}")

        if stats.gaps_count > 0:
            logger.warning(f"  - Time gaps detected: {stats.gaps_count}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
