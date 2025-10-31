"""Validation policy and threshold schemas."""

from typing import Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum


class CheckSeverity(str, Enum):
    """Severity levels for validation checks."""

    FAIL = "fail"  # Pipeline halts
    WARN = "warn"  # Logged, pipeline continues
    INFO = "info"  # Informational only


class SignalRange(BaseModel):
    """Expected range for a telemetry signal."""

    signal_name: str = Field(..., description="Signal name")
    min_value: Optional[float] = Field(None, description="Minimum valid value")
    max_value: Optional[float] = Field(None, description="Maximum valid value")
    unit: Optional[str] = Field(None, description="Signal unit")

    # Severity for violations
    severity_below_min: CheckSeverity = Field(
        CheckSeverity.FAIL, description="Action when value < min"
    )
    severity_above_max: CheckSeverity = Field(
        CheckSeverity.FAIL, description="Action when value > max"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "signal_name": "speed",
                "min_value": 0.0,
                "max_value": 100.0,
                "unit": "m/s",
                "severity_below_min": "fail",
                "severity_above_max": "warn",
            }
        }


class ValidationPolicy(BaseModel):
    """Comprehensive validation policy with fail/warn thresholds."""

    # Schema validation
    require_all_columns: bool = Field(
        True, description="Fail if required columns missing"
    )

    # Signal ranges
    signal_ranges: Dict[str, SignalRange] = Field(
        default_factory=lambda: {
            "speed": SignalRange(
                signal_name="speed",
                min_value=0.0,
                max_value=None,
                unit="m/s",
                severity_below_min=CheckSeverity.FAIL,
            ),
            "aps": SignalRange(
                signal_name="aps",
                min_value=0.0,
                max_value=1.0,
                unit="0-1",
                severity_below_min=CheckSeverity.FAIL,
                severity_above_max=CheckSeverity.FAIL,
            ),
            "gear": SignalRange(
                signal_name="gear",
                min_value=-1,
                max_value=10,
                severity_below_min=CheckSeverity.FAIL,
                severity_above_max=CheckSeverity.FAIL,
            ),
            "pbrake_f": SignalRange(
                signal_name="pbrake_f",
                min_value=0.0,
                max_value=None,
                severity_below_min=CheckSeverity.FAIL,
            ),
            "pbrake_r": SignalRange(
                signal_name="pbrake_r",
                min_value=0.0,
                max_value=None,
                severity_below_min=CheckSeverity.FAIL,
            ),
        },
        description="Per-signal range constraints",
    )

    # NaN thresholds (% of frames)
    nan_critical_warn_pct: float = Field(1.0, description="Warn if >1% NaN in critical signals")
    nan_critical_fail_pct: float = Field(5.0, description="Fail if >5% NaN in critical signals")

    # Critical signals list
    critical_signals: list[str] = Field(
        default=["speed", "Steering_Angle", "aps", "gear"],
        description="Signals that must meet strict quality",
    )

    # Time drift thresholds (seconds)
    drift_std_warn_sec: float = Field(10.0, description="Warn if drift_std > 10s")
    drift_std_fail_sec: float = Field(30.0, description="Fail if drift_std > 30s")

    # Time gap thresholds (seconds)
    time_gap_warn_sec: float = Field(0.5, description="Warn if time gap > 0.5s")
    time_gap_fail_sec: float = Field(2.0, description="Fail if time gap > 2.0s")

    # Position data quality
    position_missing_fail_pct: float = Field(
        10.0, description="Fail if position missing for >10% of time"
    )

    gps_jump_threshold_m: float = Field(
        500.0, description="Fail if GPS jumps >500m between frames"
    )

    # Multi-car requirements
    min_cars_for_simulation: int = Field(
        3, description="Fail if <3 cars have valid data"
    )

    # Lap validation
    lap_invalid_sentinel: int = Field(32768, description="Lap value indicating invalid data")
    lap_invalid_fail_pct: float = Field(
        10.0, description="Fail if >10% rows have lap=32768 post-repair"
    )

    # Monotonicity
    require_monotonic_time: bool = Field(
        True, description="Fail if backwards timestamps detected"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "require_all_columns": True,
                "nan_critical_warn_pct": 1.0,
                "nan_critical_fail_pct": 5.0,
                "critical_signals": ["speed", "aps", "gear"],
                "drift_std_warn_sec": 10.0,
                "drift_std_fail_sec": 30.0,
                "time_gap_warn_sec": 0.5,
                "time_gap_fail_sec": 2.0,
                "position_missing_fail_pct": 10.0,
                "min_cars_for_simulation": 3,
            }
        }
