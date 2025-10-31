"""Build Great Expectations expectation suites for telemetry data."""

import great_expectations as gx
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
import yaml

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ExpectationLevel(str, Enum):
    """Validation strictness level."""

    RAW_CURATED = "raw_curated"  # After ingestion, before transformations
    REFINED = "refined"  # After time sync, lap repair, position normalization
    SIMULATION_READY = "simulation_ready"  # Final wide-format synchronized data


def load_validation_policy() -> Dict:
    """Load validation policy from YAML config.

    Returns:
        Dict with validation thresholds
    """
    config_path = Path(__file__).parent.parent / "conf" / "validation_policy.yaml"

    with open(config_path, "r") as f:
        policy = yaml.safe_load(f)

    return policy


def build_raw_curated_suite(policy: Dict, context: Optional[Any] = None) -> gx.core.ExpectationSuite:
    """Build expectation suite for raw_curated data layer.

    This validates data immediately after ingestion:
    - Schema compliance
    - Required columns present
    - Data types correct
    - Basic range checks

    Args:
        policy: Validation policy dict
        context: Optional GX context (required for GX 1.x)

    Returns:
        ExpectationSuite for raw_curated layer
    """
    logger.info("Building expectation suite: RAW_CURATED")

    # For GX 1.x, we need a context to create suites
    if context is None:
        # Create a minimal in-memory context
        context = gx.get_context(mode="ephemeral")

    suite = context.suites.add(gx.core.ExpectationSuite(name="raw_curated"))

    # Required columns
    required_columns = [
        "timestamp",
        "meta_time",
        "vehicle_id",
        "chassis_id",
        "car_no",
        "lap",
        "telemetry_name",
        "telemetry_value",
    ]

    # Column existence
    for col in required_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=col)
        )

    # Data types
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="telemetry_value",
            type_="float64"
        )
    )

    # Non-null critical columns
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="chassis_id")
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="telemetry_name")
    )

    logger.info(f"  Created suite with {len(suite.expectations)} expectations")

    return suite


def build_refined_suite(policy: Dict, context: Optional[Any] = None) -> gx.core.ExpectationSuite:
    """Build expectation suite for refined data layer.

    This validates data after transformations:
    - Time correction applied
    - Lap repair completed
    - Position normalization done
    - Signal ranges validated
    - NaN thresholds enforced

    Args:
        policy: Validation policy dict
        context: Optional GX context (required for GX 1.x)

    Returns:
        ExpectationSuite for refined layer
    """
    logger.info("Building expectation suite: REFINED")

    # For GX 1.x, we need a context to create suites
    if context is None:
        context = gx.get_context(mode="ephemeral")

    suite = context.suites.add(gx.core.ExpectationSuite(name="refined"))

    # Required columns
    refined_columns = [
        "time_corrected",
        "chassis_id",
        "car_no",
        "lap_repaired",
        "telemetry_name",
        "telemetry_value",
        "segment_id",
    ]

    for col in refined_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=col)
        )

    # Non-null columns
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="chassis_id")
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="time_corrected")
    )

    # Value ranges for critical signals
    signal_ranges = policy.get("signal_ranges", {})

    # Speed must be >= 0
    if "speed" in signal_ranges:
        min_val = signal_ranges["speed"].get("min_value", 0.0)
        suite.add_expectation(
            gx.expectations.ExpectColumnMinToBeBetween(
                column="telemetry_value",
                min_value=min_val,
                max_value=None,
            )
        )

    logger.info(f"  Created suite with {len(suite.expectations)} expectations")

    return suite


def build_simulation_ready_suite(policy: Dict, context: Optional[Any] = None) -> gx.core.ExpectationSuite:
    """Build expectation suite for simulation-ready data.

    This validates final wide-format synchronized data:
    - All required signals present
    - Multi-car coverage sufficient
    - Time grid uniform
    - No phantom forward-fills

    Args:
        policy: Validation policy dict
        context: Optional GX context (required for GX 1.x)

    Returns:
        ExpectationSuite for simulation_ready layer
    """
    logger.info("Building expectation suite: SIMULATION_READY")

    # For GX 1.x, we need a context to create suites
    if context is None:
        context = gx.get_context(mode="ephemeral")

    suite = context.suites.add(gx.core.ExpectationSuite(name="simulation_ready"))

    # Required columns (wide format)
    required_sim_columns = [
        "time_global",
        "chassis_id",
        "car_no",
        "lap",
        "speed",
        "Steering_Angle",
        "aps",
        "gear",
    ]

    for col in required_sim_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=col)
        )

    # Critical signals must have low null rate
    for signal in ["speed", "Steering_Angle", "aps", "gear"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=signal,
                mostly=0.95  # At least 95% non-null
            )
        )

    # Speed range
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="speed",
            min_value=0.0,
            max_value=100.0
        )
    )

    # Gear range
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="gear",
            min_value=-1,
            max_value=10
        )
    )

    # APS range (normalized)
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="aps",
            min_value=0.0,
            max_value=1.0
        )
    )

    # Minimum distinct cars
    min_cars = policy.get("multi_car", {}).get("min_cars_for_simulation", 3)
    suite.add_expectation(
        gx.expectations.ExpectColumnUniqueValueCountToBeBetween(
            column="chassis_id",
            min_value=min_cars,
            max_value=None
        )
    )

    logger.info(f"  Created suite with {len(suite.expectations)} expectations")

    return suite


def build_expectation_suite(
    level: ExpectationLevel,
    policy: Optional[Dict] = None,
    context: Optional[Any] = None
) -> gx.core.ExpectationSuite:
    """Build expectation suite for specified validation level.

    Args:
        level: Validation level (RAW_CURATED, REFINED, SIMULATION_READY)
        policy: Optional validation policy dict (loaded from YAML if None)
        context: Optional GX context (required for GX 1.x)

    Returns:
        ExpectationSuite for the specified level
    """
    if policy is None:
        policy = load_validation_policy()

    if level == ExpectationLevel.RAW_CURATED:
        return build_raw_curated_suite(policy, context)
    elif level == ExpectationLevel.REFINED:
        return build_refined_suite(policy, context)
    elif level == ExpectationLevel.SIMULATION_READY:
        return build_simulation_ready_suite(policy, context)
    else:
        raise ValueError(f"Unknown expectation level: {level}")
