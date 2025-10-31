"""Per-lap feature schemas."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class LapFeatures(BaseModel):
    """Computed features for a single lap."""

    # Identity
    event: str = Field(..., description="Event name")
    chassis_id: str = Field(..., description="Chassis ID")
    car_no: str = Field(..., description="Car number")
    lap: int = Field(..., description="Lap number", gt=0)

    # Timing
    lap_start_time: Optional[datetime] = Field(None, description="Lap start timestamp (global)")
    lap_end_time: Optional[datetime] = Field(None, description="Lap end timestamp (global)")
    lap_time: Optional[float] = Field(None, description="Lap time in seconds", gt=0)

    # Speed metrics
    avg_speed: Optional[float] = Field(None, description="Average speed (m/s)", ge=0)
    max_speed: Optional[float] = Field(None, description="Maximum speed (m/s)", ge=0)
    min_speed: Optional[float] = Field(None, description="Minimum speed (m/s)", ge=0)

    # Throttle/brake usage
    throttle_pct_time: Optional[float] = Field(
        None, description="% of lap on throttle (aps > 0.1)", ge=0, le=100
    )
    full_throttle_pct_time: Optional[float] = Field(
        None, description="% of lap at full throttle (aps > 0.9)", ge=0, le=100
    )
    brake_pct_time: Optional[float] = Field(
        None, description="% of lap on brakes (brake > threshold)", ge=0, le=100
    )

    # Engine
    avg_rpm: Optional[float] = Field(None, description="Average RPM", ge=0)
    max_rpm: Optional[float] = Field(None, description="Maximum RPM", ge=0)

    # Gearbox
    gear_changes: Optional[int] = Field(None, description="Number of gear changes", ge=0)

    # Distance
    distance_covered: Optional[float] = Field(
        None, description="Distance covered in meters (from GPS or track)", ge=0
    )

    # Data quality
    frame_count: Optional[int] = Field(None, description="Number of frames in lap", gt=0)
    nan_pct: Optional[float] = Field(
        None, description="% of frames with NaN in critical signals", ge=0, le=100
    )

    class Config:
        json_schema_extra = {
            "example": {
                "event": "barber",
                "chassis_id": "004",
                "car_no": "78",
                "lap": 5,
                "lap_start_time": "2025-09-05T12:00:00.000Z",
                "lap_end_time": "2025-09-05T12:01:35.250Z",
                "lap_time": 95.25,
                "avg_speed": 38.8,
                "max_speed": 58.3,
                "throttle_pct_time": 62.5,
                "full_throttle_pct_time": 35.2,
                "brake_pct_time": 18.7,
                "avg_rpm": 5800,
                "max_rpm": 7800,
                "gear_changes": 42,
                "distance_covered": 3700,
                "frame_count": 1905,
                "nan_pct": 0.5,
            }
        }


class PerLapSummary(BaseModel):
    """Summary statistics across all laps for a car."""

    # Identity
    event: str = Field(..., description="Event name")
    chassis_id: str = Field(..., description="Chassis ID")
    car_no: str = Field(..., description="Car number")

    # Session summary
    total_laps: int = Field(..., description="Total valid laps", ge=0)
    total_distance: Optional[float] = Field(None, description="Total distance (meters)", ge=0)
    total_time: Optional[float] = Field(None, description="Total session time (seconds)", ge=0)

    # Best laps
    best_lap_time: Optional[float] = Field(None, description="Best lap time (seconds)", gt=0)
    best_lap_number: Optional[int] = Field(None, description="Lap number of best lap", gt=0)

    # Average metrics
    avg_lap_time: Optional[float] = Field(None, description="Average lap time (seconds)", gt=0)
    avg_speed_overall: Optional[float] = Field(
        None, description="Average speed across session (m/s)", ge=0
    )

    # Consistency
    lap_time_std: Optional[float] = Field(
        None, description="Lap time standard deviation", ge=0
    )

    # Data quality
    valid_lap_pct: Optional[float] = Field(
        None, description="% of laps with complete data", ge=0, le=100
    )

    class Config:
        json_schema_extra = {
            "example": {
                "event": "barber",
                "chassis_id": "004",
                "car_no": "78",
                "total_laps": 20,
                "total_distance": 74000,
                "total_time": 1920.5,
                "best_lap_time": 94.8,
                "best_lap_number": 12,
                "avg_lap_time": 96.0,
                "avg_speed_overall": 38.5,
                "lap_time_std": 1.2,
                "valid_lap_pct": 95.0,
            }
        }
