"""Custom Great Expectations for multi-car telemetry validation."""

import pandas as pd
from typing import Dict, Any, Optional
from great_expectations.core import ExpectationValidationResult
from great_expectations.expectations.expectation import Expectation
from great_expectations.render import RenderedStringTemplateContent

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ExpectPerCarMinimumCoverage(Expectation):
    """Expectation that validates minimum data coverage per car.

    This ensures each car in the synchronized dataset has sufficient data
    for simulation purposes (e.g., > 10% of frames with any data).
    """

    metric_dependencies = tuple()

    def __init__(self, min_coverage_pct: float = 10.0, **kwargs):
        """Initialize per-car coverage expectation.

        Args:
            min_coverage_pct: Minimum percentage of frames with data per car
        """
        super().__init__(**kwargs)
        self.min_coverage_pct = min_coverage_pct

    def _validate(
        self,
        metrics: Dict[str, Any],
        runtime_configuration: Optional[dict] = None,
        execution_engine: Optional[Any] = None,
    ):
        """Validate per-car coverage in synchronized telemetry data.

        Args:
            metrics: Metrics dict (not used, we compute directly)
            runtime_configuration: Runtime config
            execution_engine: Execution engine (contains the DataFrame)

        Returns:
            ExpectationValidationResult
        """
        # Get the DataFrame from execution engine
        if execution_engine is None:
            return ExpectationValidationResult(
                success=False,
                result={"error": "No execution engine provided"}
            )

        # Access the DataFrame through the execution engine
        # For GX 1.x, this might be different depending on the engine type
        try:
            df = execution_engine.batch_manager.active_batch.data.dataframe
        except AttributeError:
            # Fallback for different GX versions
            try:
                df = execution_engine.active_batch_data.dataframe
            except AttributeError:
                return ExpectationValidationResult(
                    success=False,
                    result={"error": "Could not access DataFrame from execution engine"}
                )

        if not isinstance(df, pd.DataFrame):
            return ExpectationValidationResult(
                success=False,
                result={"error": "Data is not a pandas DataFrame"}
            )

        # Check required columns
        if 'chassis_id' not in df.columns:
            return ExpectationValidationResult(
                success=False,
                result={"error": "Column 'chassis_id' not found"}
            )

        # Calculate coverage per car
        total_frames = len(df['time_global'].unique()) if 'time_global' in df.columns else len(df)

        car_coverage = {}
        failed_cars = []

        for chassis_id in df['chassis_id'].unique():
            if pd.isna(chassis_id):
                continue

            car_data = df[df['chassis_id'] == chassis_id]

            # Count frames with ANY non-null data (excluding chassis_id, time_global)
            data_columns = [col for col in car_data.columns
                          if col not in ['chassis_id', 'time_global', 'car_no']]

            if len(data_columns) == 0:
                frames_with_data = 0
            else:
                # A frame has data if ANY telemetry column is non-null
                frames_with_data = car_data[data_columns].notna().any(axis=1).sum()

            coverage_pct = (frames_with_data / total_frames * 100) if total_frames > 0 else 0.0

            car_coverage[chassis_id] = {
                'frames_with_data': int(frames_with_data),
                'total_frames': int(total_frames),
                'coverage_pct': float(coverage_pct)
            }

            if coverage_pct < self.min_coverage_pct:
                failed_cars.append(chassis_id)

        success = len(failed_cars) == 0

        result = {
            'observed_value': car_coverage,
            'element_count': len(car_coverage),
            'missing_count': 0,
            'missing_percent': 0.0,
            'unexpected_count': len(failed_cars),
            'unexpected_percent': (len(failed_cars) / len(car_coverage) * 100) if car_coverage else 0.0,
            'unexpected_list': failed_cars,
            'partial_unexpected_list': failed_cars[:10],  # First 10 failures
        }

        return ExpectationValidationResult(
            success=success,
            result=result
        )


class ExpectPerCarMinimumDuration(Expectation):
    """Expectation that validates minimum race duration per car.

    This ensures each car has data spanning a minimum time period
    (e.g., > 30 minutes of the race).
    """

    metric_dependencies = tuple()

    def __init__(self, min_duration_sec: float = 1800.0, **kwargs):
        """Initialize per-car duration expectation.

        Args:
            min_duration_sec: Minimum duration in seconds per car
        """
        super().__init__(**kwargs)
        self.min_duration_sec = min_duration_sec

    def _validate(
        self,
        metrics: Dict[str, Any],
        runtime_configuration: Optional[dict] = None,
        execution_engine: Optional[Any] = None,
    ):
        """Validate per-car duration in synchronized telemetry data."""
        # Get DataFrame
        if execution_engine is None:
            return ExpectationValidationResult(
                success=False,
                result={"error": "No execution engine provided"}
            )

        try:
            df = execution_engine.batch_manager.active_batch.data.dataframe
        except AttributeError:
            try:
                df = execution_engine.active_batch_data.dataframe
            except AttributeError:
                return ExpectationValidationResult(
                    success=False,
                    result={"error": "Could not access DataFrame"}
                )

        # Check required columns
        if 'chassis_id' not in df.columns or 'time_global' not in df.columns:
            return ExpectationValidationResult(
                success=False,
                result={"error": "Required columns not found"}
            )

        # Calculate duration per car
        car_durations = {}
        failed_cars = []

        for chassis_id in df['chassis_id'].unique():
            if pd.isna(chassis_id):
                continue

            car_data = df[df['chassis_id'] == chassis_id]

            # Get time range where car has ANY data
            data_columns = [col for col in car_data.columns
                          if col not in ['chassis_id', 'time_global', 'car_no']]

            if len(data_columns) > 0:
                has_data_mask = car_data[data_columns].notna().any(axis=1)
                car_with_data = car_data[has_data_mask]

                if len(car_with_data) > 0:
                    min_time = car_with_data['time_global'].min()
                    max_time = car_with_data['time_global'].max()
                    duration_sec = (max_time - min_time).total_seconds()
                else:
                    duration_sec = 0.0
            else:
                duration_sec = 0.0

            car_durations[chassis_id] = {
                'duration_sec': float(duration_sec),
                'duration_min': float(duration_sec / 60)
            }

            if duration_sec < self.min_duration_sec:
                failed_cars.append(chassis_id)

        success = len(failed_cars) == 0

        result = {
            'observed_value': car_durations,
            'element_count': len(car_durations),
            'unexpected_count': len(failed_cars),
            'unexpected_list': failed_cars,
        }

        return ExpectationValidationResult(
            success=success,
            result=result
        )
