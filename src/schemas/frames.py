"""Telemetry frame schemas for wide-format time-indexed data."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class PositionData(BaseModel):
    """Vehicle position from GPS or track distance."""

    # GPS position (if available)
    latitude: Optional[float] = Field(None, description="GPS latitude in degrees", ge=-90, le=90)
    longitude: Optional[float] = Field(
        None, description="GPS longitude in degrees", ge=-180, le=180
    )
    latitude_raw: Optional[float] = Field(
        None, description="GPS latitude in minutes (original VBOX format)"
    )
    longitude_raw: Optional[float] = Field(
        None, description="GPS longitude in minutes (original VBOX format)"
    )

    # Track position (if available)
    track_position: Optional[float] = Field(
        None, description="Normalized track position [0, 1] or meters", ge=0
    )
    track_position_raw: Optional[float] = Field(
        None, description="Raw track distance from Laptrigger_lapdist_dls"
    )

    @property
    def has_gps(self) -> bool:
        """Check if GPS coordinates are available."""
        return self.latitude is not None and self.longitude is not None

    @property
    def has_track_position(self) -> bool:
        """Check if track position is available."""
        return self.track_position is not None

    @property
    def has_any_position(self) -> bool:
        """Check if any position data is available."""
        return self.has_gps or self.has_track_position


class TelemetryFrame(BaseModel):
    """Single telemetry frame (one timestep for one car).

    Wide format with all signals as columns.
    Time-indexed by time_corrected or time_global.
    """

    # Time
    time_corrected: datetime = Field(..., description="ECU timestamp + drift correction")
    time_global: Optional[datetime] = Field(
        None, description="Session-synchronized global timestamp"
    )
    timestamp_raw: Optional[datetime] = Field(None, description="Original ECU timestamp")
    meta_time_raw: Optional[datetime] = Field(None, description="Original receiver timestamp")

    # Identity
    chassis_id: str = Field(..., description="Chassis ID (canonical)")
    car_no: str = Field(..., description="Car number")
    lap: int = Field(..., description="Lap number (repaired)")
    lap_raw: Optional[int] = Field(None, description="Original lap value (may be 32768)")

    # Critical signals (required for simulation)
    speed: Optional[float] = Field(None, description="Vehicle speed in m/s", ge=0)
    speed_raw: Optional[float] = Field(None, description="Original speed value")

    Steering_Angle: Optional[float] = Field(None, description="Steering angle in degrees")

    aps: Optional[float] = Field(None, description="Throttle position [0, 1]", ge=0, le=1)
    aps_raw: Optional[float] = Field(None, description="Original throttle % [0, 100]")

    pbrake_f: Optional[float] = Field(None, description="Front brake pressure (bar/psi)", ge=0)
    pbrake_r: Optional[float] = Field(None, description="Rear brake pressure (bar/psi)", ge=0)

    gear: Optional[int] = Field(None, description="Current gear", ge=-1, le=10)

    # Additional signals
    nmot: Optional[float] = Field(None, description="Engine speed (RPM)", ge=0)
    accx_can: Optional[float] = Field(None, description="Longitudinal acceleration (m/s²)")
    accy_can: Optional[float] = Field(None, description="Lateral acceleration (m/s²)")

    # Position (embedded)
    position: Optional[PositionData] = Field(None, description="GPS and/or track position")

    # Metadata
    segment_id: Optional[int] = Field(None, description="Time segment ID (for drift step changes)")
    ffill_count: Optional[int] = Field(
        None, description="Number of forward-filled frames since last real data"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "time_corrected": "2025-09-05T00:28:20.593Z",
                "time_global": "2025-09-05T12:00:00.000Z",
                "chassis_id": "004",
                "car_no": "78",
                "lap": 2,
                "speed": 45.5,
                "Steering_Angle": 2.6,
                "aps": 1.0,
                "pbrake_f": 0.0,
                "pbrake_r": 0.0,
                "gear": 3,
                "nmot": 7206,
                "accx_can": 0.346,
                "accy_can": -0.074,
                "position": {
                    "latitude": 33.5543,
                    "longitude": -86.6197,
                    "track_position": 0.87,
                },
            }
        }


class SimulationFrame(BaseModel):
    """Simulation-ready frame for multi-car synchronized playback.

    Flattened structure with all required fields for visualization.
    """

    # Time (global synchronized)
    time_global: datetime = Field(..., description="Session-synchronized timestamp")

    # Identity
    chassis_id: str = Field(..., description="Chassis ID")
    car_no: str = Field(..., description="Car number")
    lap: int = Field(..., description="Lap number")

    # Position (required)
    latitude: Optional[float] = Field(None, description="GPS latitude (degrees)")
    longitude: Optional[float] = Field(None, description="GPS longitude (degrees)")
    track_position: Optional[float] = Field(None, description="Track position [0, 1]")

    # Telemetry (normalized units)
    speed: float = Field(..., description="Speed (m/s)", ge=0)
    steering: float = Field(..., description="Steering angle (degrees)")
    throttle: float = Field(..., description="Throttle position [0, 1]", ge=0, le=1)
    brake_front: float = Field(..., description="Front brake pressure", ge=0)
    brake_rear: float = Field(..., description="Rear brake pressure", ge=0)
    gear: int = Field(..., description="Current gear", ge=-1, le=10)
    rpm: float = Field(..., description="Engine speed (RPM)", ge=0)
    accel_x: Optional[float] = Field(None, description="Longitudinal accel (m/s²)")
    accel_y: Optional[float] = Field(None, description="Lateral accel (m/s²)")

    class Config:
        json_schema_extra = {
            "example": {
                "time_global": "2025-09-05T12:00:00.000Z",
                "chassis_id": "004",
                "car_no": "78",
                "lap": 2,
                "latitude": 33.5543,
                "longitude": -86.6197,
                "track_position": 0.87,
                "speed": 45.5,
                "steering": 2.6,
                "throttle": 1.0,
                "brake_front": 0.0,
                "brake_rear": 0.0,
                "gear": 3,
                "rpm": 7206,
                "accel_x": 0.346,
                "accel_y": -0.074,
            }
        }


class RequiredSignals(BaseModel):
    """Definition of required signals for simulation readiness."""

    minimal: list[str] = Field(
        default=[
            "time_corrected",
            "speed",
            "Steering_Angle",
            "aps",
            "gear",
            "lap",
        ],
        description="Absolute minimum signals needed",
    )

    brakes: list[str] = Field(
        default=["pbrake_f", "pbrake_r"],
        description="At least one brake signal required",
    )

    position: list[str] = Field(
        default=["latitude_longitude", "track_position"],
        description="At least one position type required",
    )

    recommended: list[str] = Field(
        default=[
            "nmot",
            "accx_can",
            "accy_can",
        ],
        description="Recommended but not strictly required",
    )
