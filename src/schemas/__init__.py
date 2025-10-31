"""Data schemas and contracts for telemetry processing."""

from .raw import RawTelemetryRow, VehicleIdentity
from .frames import TelemetryFrame, SimulationFrame, PositionData
from .features import LapFeatures, PerLapSummary
from .metadata import (
    DriftCalibration,
    LapBoundary,
    TrackMetadata,
    EventMetadata,
    SimulationMetadata,
    ChassisCarMapping,
)
from .validation import ValidationPolicy, SignalRange, CheckSeverity

__all__ = [
    # Raw data
    "RawTelemetryRow",
    "VehicleIdentity",
    # Frames
    "TelemetryFrame",
    "SimulationFrame",
    "PositionData",
    # Features
    "LapFeatures",
    "PerLapSummary",
    # Metadata
    "DriftCalibration",
    "LapBoundary",
    "TrackMetadata",
    "EventMetadata",
    "SimulationMetadata",
    "ChassisCarMapping",
    # Validation
    "ValidationPolicy",
    "SignalRange",
    "CheckSeverity",
]
