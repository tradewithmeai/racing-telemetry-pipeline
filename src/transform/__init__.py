"""Transformation modules for telemetry processing."""

from .time_sync import (
    compute_per_car_drift,
    detect_backwards_segments,
    apply_drift_correction,
    align_to_global_time,
)
from .drift import (
    windowed_drift_calibration,
    detect_clock_steps,
    DriftCalibration,
)
from .lap_repair import (
    detect_lap_boundaries,
    assign_lap_numbers,
    repair_laps,
    fix_lap_sentinels,
)
from .position import (
    normalize_position_data,
    convert_gps_minutes_to_degrees,
    validate_gps_bounds,
    detect_gps_outliers,
    interpolate_position,
    PositionQuality,
)
from .pivot import (
    pivot_to_wide_format,
    pivot_to_wide_format_polars,
    save_wide_format,
    save_pivot_stats,
    PivotStats,
)
from .resample import (
    resample_to_time_grid,
    resample_with_interpolation,
    create_uniform_time_grid,
    save_resampled_data,
    save_resample_stats,
    ResampleStats,
)
from .sync import (
    synchronize_multi_car,
    determine_global_time_range,
    create_global_time_grid,
    align_car_to_global_grid,
    save_synchronized_data,
    save_sync_stats,
    SyncStats,
    CarCoverage,
)

__all__ = [
    # Time sync
    "compute_per_car_drift",
    "detect_backwards_segments",
    "apply_drift_correction",
    "align_to_global_time",
    # Drift calibration
    "windowed_drift_calibration",
    "detect_clock_steps",
    "DriftCalibration",
    # Lap repair
    "detect_lap_boundaries",
    "assign_lap_numbers",
    "repair_laps",
    "fix_lap_sentinels",
    # Position normalization
    "normalize_position_data",
    "convert_gps_minutes_to_degrees",
    "validate_gps_bounds",
    "detect_gps_outliers",
    "interpolate_position",
    "PositionQuality",
    # Pivot transformation
    "pivot_to_wide_format",
    "pivot_to_wide_format_polars",
    "save_wide_format",
    "save_pivot_stats",
    "PivotStats",
    # Resample transformation
    "resample_to_time_grid",
    "resample_with_interpolation",
    "create_uniform_time_grid",
    "save_resampled_data",
    "save_resample_stats",
    "ResampleStats",
    # Multi-car synchronization
    "synchronize_multi_car",
    "determine_global_time_range",
    "create_global_time_grid",
    "align_car_to_global_grid",
    "save_synchronized_data",
    "save_sync_stats",
    "SyncStats",
    "CarCoverage",
]
