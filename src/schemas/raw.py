"""Raw telemetry data schemas."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import re


class VehicleIdentity(BaseModel):
    """Parsed vehicle identity from vehicle_id string.

    Format: GR86-{chassis_id}-{car_no}
    Example: GR86-004-78 → chassis_id=004, car_no=78
    """

    vehicle_id: str = Field(..., description="Full vehicle identifier")
    chassis_id: str = Field(..., description="Chassis ID (canonical key)")
    car_no: str = Field(..., description="Car number (may be 000 if unassigned)")

    @classmethod
    def from_vehicle_id(cls, vehicle_id: str) -> "VehicleIdentity":
        """Parse vehicle_id string into components.

        Args:
            vehicle_id: String like "GR86-004-78"

        Returns:
            VehicleIdentity with parsed fields

        Raises:
            ValueError: If vehicle_id doesn't match expected format
        """
        pattern = r"GR86-(\d+)-(\d+)"
        match = re.match(pattern, vehicle_id)
        if not match:
            raise ValueError(
                f"Invalid vehicle_id format: {vehicle_id}. "
                f"Expected format: GR86-{{chassis}}-{{car_no}}"
            )

        chassis_id = match.group(1)
        car_no = match.group(2)

        return cls(vehicle_id=vehicle_id, chassis_id=chassis_id, car_no=car_no)


class RawTelemetryRow(BaseModel):
    """Raw telemetry CSV row schema.

    Matches the actual CSV structure from Barber data.
    """

    # Core telemetry fields
    timestamp: datetime = Field(..., description="ECU timestamp (ISO8601)")
    meta_time: datetime = Field(..., description="Receiver UTC timestamp (ISO8601)")
    vehicle_id: str = Field(..., description="Vehicle identifier (GR86-XXX-YYY)")
    lap: int = Field(..., description="Lap number (32768 = invalid)")
    telemetry_name: str = Field(..., description="Signal name")
    telemetry_value: float = Field(..., description="Signal value")

    # Metadata fields (optional in processing)
    meta_event: Optional[str] = Field(None, description="Event identifier")
    meta_session: Optional[str] = Field(None, description="Session name (e.g., R1, R2)")
    meta_source: Optional[str] = Field(None, description="Data source (e.g., kafka:gr-raw)")
    outing: Optional[int] = Field(None, description="Outing number")
    vehicle_number: Optional[int] = Field(None, description="Extracted car number")
    original_vehicle_id: Optional[str] = Field(None, description="Original vehicle ID")
    expire_at: Optional[datetime] = Field(None, description="Data expiration timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-09-05T00:28:20.593Z",
                "meta_time": "2025-09-06T18:40:41.926Z",
                "vehicle_id": "GR86-004-78",
                "lap": 2,
                "telemetry_name": "speed",
                "telemetry_value": 45.5,
                "meta_event": "I_R06_2025-09-07",
                "meta_session": "R1",
            }
        }


class RawCSVSchema(BaseModel):
    """Expected CSV column schema for validation."""

    required_columns: list[str] = Field(
        default=[
            "timestamp",
            "meta_time",
            "vehicle_id",
            "lap",
            "telemetry_name",
            "telemetry_value",
        ],
        description="Columns that must be present",
    )

    optional_columns: list[str] = Field(
        default=[
            "meta_event",
            "meta_session",
            "meta_source",
            "outing",
            "vehicle_number",
            "original_vehicle_id",
            "expire_at",
        ],
        description="Columns that may be present",
    )

    @property
    def all_columns(self) -> list[str]:
        """All recognized columns."""
        return self.required_columns + self.optional_columns
