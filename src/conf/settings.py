"""Configuration settings for telemetry processing pipeline."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
    """Global pipeline settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # GPU Configuration
    use_gpu: bool = True
    gpu_device_id: int = 0

    # Processing Configuration
    chunk_size_rows: int = 1_000_000
    max_workers: int = 4

    # Resampling
    resample_hz: float = 20.0
    ffill_limit_sec: float = 0.2
    max_gap_sec: float = 2.0

    # Time Correction
    drift_window_min: int = 5  # Minutes for windowed drift estimation
    drift_std_spike_threshold: float = 3.0  # Multiplier for step detection

    # Lap Repair
    min_lap_duration_sec: float = 60.0  # Minimum valid lap time
    lap_invalid_sentinel: int = 32768

    # Position
    gps_outlier_threshold_m: float = 500.0  # Max jump between frames

    # Data Paths
    data_root: str = "data"
    raw_data_path: str = os.path.join(data_root, "raw")
    processed_data_path: str = os.path.join(data_root, "processed")
    simulation_data_path: str = os.path.join(data_root, "simulation")
    reports_path: str = os.path.join(data_root, "reports")
    logs_path: str = "logs"

    # Track Maps
    track_maps_path: str = "track_maps"

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Determinism
    random_seed: int = 42

    # Pipeline Options
    enable_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"

    # Great Expectations
    ge_context_root: str = "great_expectations"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Auto-detect GPU availability
        if self.use_gpu:
            try:
                import cudf  # noqa: F401

                print("GPU (RAPIDS cuDF) detected and enabled.")
            except ImportError:
                print("Warning: GPU requested but RAPIDS not available. Falling back to CPU.")
                self.use_gpu = False


# Global settings instance
settings = Settings()
