"""Metadata schemas for pipeline artifacts."""

from datetime import datetime
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from enum import Enum


class LapBoundaryReason(str, Enum):
    """Reason codes for lap boundary detection."""

    LAPDIST_RESET = "lapdist_reset"  # Laptrigger_lapdist_dls dropped (wrap)
    TIME_GAP = "time_gap"  # Large time gap (>threshold)
    LAP_INCREMENT = "lap_increment"  # Lap number incremented
    MIN_DURATION = "min_duration"  # Minimum lap duration enforced
    MANUAL = "manual"  # Manually specified
    UNKNOWN = "unknown"  # Could not determine


class DriftCalibration(BaseModel):
    """Time drift calibration parameters for a car in a time segment."""

    # Identity
    event: str = Field(..., description="Event name")
    chassis_id: str = Field(..., description="Chassis ID")
    segment_id: int = Field(..., description="Segment ID (for clock step changes)", ge=0)

    # Time window
    window_start: datetime = Field(..., description="Start of calibration window")
    window_end: datetime = Field(..., description="End of calibration window")

    # Drift parameters
    drift_sec: float = Field(..., description="Mean drift in seconds (meta_time - timestamp)")
    drift_std: float = Field(..., description="Standard deviation of drift", ge=0)

    # Detection
    step_detected: bool = Field(
        False, description="Whether a clock step change was detected"
    )
    samples: int = Field(..., description="Number of samples used for calibration", gt=0)

    # Quality flags
    is_valid: bool = Field(True, description="Whether calibration is considered valid")
    warning: Optional[str] = Field(None, description="Warning message if drift_std is high")

    class Config:
        json_schema_extra = {
            "example": {
                "event": "barber",
                "chassis_id": "004",
                "segment_id": 0,
                "window_start": "2025-09-05T12:00:00Z",
                "window_end": "2025-09-05T12:05:00Z",
                "drift_sec": 2.5,
                "drift_std": 0.3,
                "step_detected": False,
                "samples": 15000,
                "is_valid": True,
            }
        }


class LapBoundary(BaseModel):
    """Detected lap boundary with audit trail."""

    # Identity
    event: str = Field(..., description="Event name")
    chassis_id: str = Field(..., description="Chassis ID")
    boundary_time: datetime = Field(..., description="Timestamp of lap boundary (global)")

    # Lap info
    pre_lap: Optional[int] = Field(None, description="Lap number before boundary")
    post_lap: int = Field(..., description="Lap number after boundary", gt=0)

    # Detection
    reason: LapBoundaryReason = Field(..., description="Reason code for boundary detection")
    confidence: float = Field(
        ..., description="Confidence score [0, 1]", ge=0, le=1
    )

    # Supporting data
    lapdist_value: Optional[float] = Field(
        None, description="Laptrigger_lapdist_dls value at boundary"
    )
    time_gap_sec: Optional[float] = Field(
        None, description="Time gap before this point (seconds)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "event": "barber",
                "chassis_id": "004",
                "boundary_time": "2025-09-05T12:01:35.250Z",
                "pre_lap": 4,
                "post_lap": 5,
                "reason": "lapdist_reset",
                "confidence": 0.95,
                "lapdist_value": 3715.5,
            }
        }


class TrackMetadata(BaseModel):
    """Track/circuit metadata for position mapping."""

    # Track info
    track_name: str = Field(..., description="Track name")
    track_length_m: float = Field(..., description="Track length in meters", gt=0)
    location: Optional[str] = Field(None, description="Track location (city, state/country)")

    # GPS bounds (for mapping to image)
    gps_bounds: Optional[Dict[str, float]] = Field(
        None,
        description="GPS bounding box",
        json_schema_extra={
            "example": {
                "lat_min": 33.5,
                "lat_max": 33.6,
                "lon_min": -86.65,
                "lon_max": -86.55,
            }
        },
    )

    # Start/finish
    start_finish_gps: Optional[Dict[str, float]] = Field(
        None,
        description="Start/finish line GPS",
        json_schema_extra={"example": {"lat": 33.5543, "lon": -86.6197}},
    )

    # Map image
    map_image_path: Optional[str] = Field(None, description="Path to track map image (SVG/PNG)")
    map_width_px: Optional[int] = Field(None, description="Map image width in pixels")
    map_height_px: Optional[int] = Field(None, description="Map image height in pixels")

    # Reference times (for validation)
    reference_lap_time_sec: Optional[float] = Field(
        None, description="Typical/reference lap time (seconds)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "track_name": "Barber Motorsports Park",
                "track_length_m": 3700,
                "location": "Leeds, Alabama, USA",
                "gps_bounds": {
                    "lat_min": 33.5,
                    "lat_max": 33.6,
                    "lon_min": -86.65,
                    "lon_max": -86.55,
                },
                "start_finish_gps": {"lat": 33.5543, "lon": -86.6197},
                "map_image_path": "track_maps/barber/map.svg",
                "reference_lap_time_sec": 95.0,
            }
        }


class EventMetadata(BaseModel):
    """Event-level metadata."""

    event_name: str = Field(..., description="Event identifier")
    session_name: str = Field(..., description="Session name (e.g., R1, R2)")
    track: str = Field(..., description="Track name")

    # Time range
    session_start: datetime = Field(..., description="Session start (global reference)")
    session_end: datetime = Field(..., description="Session end (global)")

    # Cars
    car_count: int = Field(..., description="Number of cars with data", ge=0)
    chassis_ids: List[str] = Field(..., description="List of chassis IDs")

    class Config:
        json_schema_extra = {
            "example": {
                "event_name": "barber",
                "session_name": "R1",
                "track": "Barber Motorsports Park",
                "session_start": "2025-09-05T12:00:00Z",
                "session_end": "2025-09-05T13:30:00Z",
                "car_count": 20,
                "chassis_ids": ["002", "004", "006"],
            }
        }


class SimulationMetadata(BaseModel):
    """Metadata for simulation-ready dataset."""

    # Event info
    event: str = Field(..., description="Event name")
    session: str = Field(..., description="Session name")

    # Time range
    time_range_start: datetime = Field(..., description="First timestamp with data")
    time_range_end: datetime = Field(..., description="Last timestamp with data")
    duration_sec: float = Field(..., description="Total session duration (seconds)", ge=0)

    # Cars
    cars_available: List[str] = Field(..., description="List of chassis IDs with data")
    car_count: int = Field(..., description="Number of cars", ge=0)

    # Data coverage
    full_coverage_windows: Optional[List[Dict[str, datetime]]] = Field(
        None,
        description="Time windows with >=3 cars present",
        json_schema_extra={
            "example": [{"start": "2025-09-05T12:00:00Z", "end": "2025-09-05T12:30:00Z"}]
        },
    )

    # Signals
    signal_list: List[str] = Field(..., description="Available telemetry signals")
    signal_coverage: Dict[str, float] = Field(
        ...,
        description="Per-signal coverage % across all cars",
        json_schema_extra={"example": {"speed": 99.5, "aps": 98.7, "gear": 99.9}},
    )

    # Position
    position_type: str = Field(
        ..., description="Position data type available", json_schema_extra={"enum": ["gps", "track_dist", "both", "none"]}
    )

    # Quality
    position_quality: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Position quality per car",
        json_schema_extra={
            "example": {"004": {"coverage_pct": 99.2, "outlier_count": 3}}
        },
    )

    # Sampling
    sample_rate_hz: float = Field(..., description="Resampling frequency (Hz)", gt=0)
    total_frames: int = Field(..., description="Total number of frames", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "event": "barber",
                "session": "R1",
                "time_range_start": "2025-09-05T12:00:00Z",
                "time_range_end": "2025-09-05T13:30:00Z",
                "duration_sec": 5400,
                "cars_available": ["004", "006", "010"],
                "car_count": 20,
                "signal_list": ["speed", "aps", "gear"],
                "signal_coverage": {"speed": 99.5, "aps": 98.7},
                "position_type": "both",
                "position_quality": {"004": {"coverage_pct": 99.2}},
                "sample_rate_hz": 20.0,
                "total_frames": 108000,
            }
        }


class ChassisCarMapping(BaseModel):
    """Mapping of chassis to car number over time (handles mid-season changes)."""

    chassis_id: str = Field(..., description="Chassis ID (canonical)")
    car_no: str = Field(..., description="Car number at this time")
    first_seen: datetime = Field(..., description="First timestamp with this pairing")
    last_seen: datetime = Field(..., description="Last timestamp with this pairing")
    event: Optional[str] = Field(None, description="Event where this pairing was observed")

    class Config:
        json_schema_extra = {
            "example": {
                "chassis_id": "004",
                "car_no": "78",
                "first_seen": "2025-09-05T12:00:00Z",
                "last_seen": "2025-09-05T13:30:00Z",
                "event": "barber",
            }
        }
