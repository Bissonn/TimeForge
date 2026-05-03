"""
Integration pytest suite for SGOLD TimeSeriesDataset._prepare_data()

This suite tests the missing value handling in the context of your
actual SGOLD project structure.

Run with:
    pytest test_sgold_integration_pytest.py -v
    pytest test_sgold_integration_pytest.py -v -s  # with output
"""

import pandas as pd
import numpy as np
import logging
import pytest
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Mock SGOLD Dataset Class
# ============================================================================

class MockSGOLDDataset:
    """Simplified mock of TimeSeriesDataset for testing _prepare_data"""

    def __init__(self, config: Dict[str, Any], name: str = "test_dataset"):
        self.config = config
        self.name = name
        self.date_column = config.get("date_column", "date")
        self.freq = config.get("freq", None)
        self.target_columns = config.get("target_columns", ["target"])
        self.past_covariates = config.get("past_covariates", [])
        self.future_covariates = config.get("future_covariates", [])
        self.columns = []
        self._original_series = None

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simplified time feature generation"""
        if isinstance(df.index, pd.DatetimeIndex):
            df['month'] = df.index.month
            df['day'] = df.index.day
        return df

    def _apply_differencing(self, df: pd.DataFrame, diff_config: dict) -> pd.DataFrame:
        """Simplified differencing (not tested here)"""
        return df

    def _prepare_data_old(self, df: pd.DataFrame) -> pd.DataFrame:
        """OLD implementation - current simple dropna"""
        if df.empty:
            raise ValueError("Input DataFrame for preparation cannot be empty.")

        if isinstance(df.index, pd.DatetimeIndex):
            logger.debug("DataFrame already has a DatetimeIndex. Using it.")
        elif self.date_column in df.columns:
            df[self.date_column] = pd.to_datetime(df[self.date_column])
            df = df.set_index(self.date_column)
        else:
            logger.warning(f"No DatetimeIndex or '{self.date_column}' column found.")
            if not isinstance(df.index, pd.RangeIndex):
                df.index = pd.RangeIndex(len(df))

        if isinstance(df.index, pd.DatetimeIndex):
            if self.freq:
                df = df.asfreq(self.freq)
            else:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq:
                    self.freq = inferred_freq
                    logger.info(f"Inferred frequency: {self.freq}")
                    df = df.asfreq(self.freq)

        df = self._add_time_features(df)

        all_cols = self.target_columns + self.past_covariates + self.future_covariates
        for col in df.columns:
            if col not in all_cols:
                all_cols.append(col)
        seen = set()
        self.columns = [c for c in all_cols if not (c in seen or seen.add(c))]

        self._original_series = df.copy()

        dataset_cfg = self.config.get("datasets", {}).get(self.name, {})
        diff_config = dataset_cfg.get("differencing", {})
        if diff_config.get("enabled", False):
            df = self._apply_differencing(df, diff_config)

        missing = [col for col in self.target_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Required target columns not found: {missing}")

        df = df[self.columns].copy()

        # === OLD MISSING VALUE HANDLING ===
        if df.isna().any().any():
            logger.warning("NaN values detected in data. Dropping rows with NaNs.")
            df = df.dropna()

        if np.any(np.isinf(df.select_dtypes(include=np.number).values)):
            raise ValueError("Data contains infinite values.")

        if df.empty:
            raise ValueError("DataFrame is empty after processing.")

        return df

    def _prepare_data_new(self, df: pd.DataFrame) -> pd.DataFrame:
        """NEW implementation - with improved missing value handling"""
        if df.empty:
            raise ValueError("Input DataFrame for preparation cannot be empty.")

        if isinstance(df.index, pd.DatetimeIndex):
            logger.debug("DataFrame already has a DatetimeIndex. Using it.")
        elif self.date_column in df.columns:
            df[self.date_column] = pd.to_datetime(df[self.date_column])
            df = df.set_index(self.date_column)
        else:
            logger.warning(f"No DatetimeIndex or '{self.date_column}' column found.")
            if not isinstance(df.index, pd.RangeIndex):
                df.index = pd.RangeIndex(len(df))

        if isinstance(df.index, pd.DatetimeIndex):
            if self.freq:
                df = df.asfreq(self.freq)
            else:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq:
                    self.freq = inferred_freq
                    logger.info(f"Inferred frequency: {self.freq}")
                    df = df.asfreq(self.freq)

        df = self._add_time_features(df)

        all_cols = self.target_columns + self.past_covariates + self.future_covariates
        for col in df.columns:
            if col not in all_cols:
                all_cols.append(col)
        seen = set()
        self.columns = [c for c in all_cols if not (c in seen or seen.add(c))]

        self._original_series = df.copy()

        dataset_cfg = self.config.get("datasets", {}).get(self.name, {})
        diff_config = dataset_cfg.get("differencing", {})
        if diff_config.get("enabled", False):
            df = self._apply_differencing(df, diff_config)

        missing = [col for col in self.target_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Required target columns not found: {missing}")

        df = df[self.columns].copy()

        # === NEW MISSING VALUE HANDLING ===
        if df.isna().any().any():
            missing_count = df.isna().sum().sum()
            total_values = df.shape[0] * df.shape[1]
            missing_pct = (missing_count / total_values) * 100

            logger.warning(
                f"NaN values detected: {missing_count}/{total_values} "
                f"({missing_pct:.2f}%) values missing"
            )

            has_datetime_index = isinstance(df.index, pd.DatetimeIndex)

            if has_datetime_index:
                df_filled = df.ffill(limit=3)
                nans_after_ffill = df_filled.isna().sum().sum()

                if nans_after_ffill > 0:
                    logger.info(
                        f"After forward fill: {nans_after_ffill} NaNs remaining. "
                        f"Applying linear interpolation..."
                    )
                    df_filled = df_filled.interpolate(
                        method='linear',
                        limit_direction='both',
                        axis=0
                    )
                    nans_after_interp = df_filled.isna().sum().sum()

                    if nans_after_interp > 0:
                        rows_before = len(df_filled)
                        df_filled = df_filled.dropna()
                        rows_dropped = rows_before - len(df_filled)

                        if rows_dropped > 0:
                            logger.warning(
                                f"After imputation, {nans_after_interp} NaNs remained. "
                                f"Dropped {rows_dropped} rows."
                            )
                    else:
                        logger.info("All NaNs successfully imputed via interpolation.")
                else:
                    logger.info("All NaNs successfully imputed via forward fill.")

                df = df_filled
            else:
                logger.warning("No DatetimeIndex. Using simple dropna().")
                rows_before = len(df)
                df = df.dropna()
                rows_dropped = rows_before - len(df)
                if rows_dropped > 0:
                    logger.warning(f"Dropped {rows_dropped} rows with NaNs.")

        if np.any(np.isinf(df.select_dtypes(include=np.number).values)):
            raise ValueError("Data contains infinite values.")

        if df.empty:
            raise ValueError("DataFrame is empty after processing.")

        return df


# ============================================================================
# Fixtures - Synthetic Datasets
# ============================================================================

@pytest.fixture
def ett_config():
    """Config for ETT-like dataset"""
    return {
        "date_column": "date",
        "freq": "h",
        "target_columns": ["OT"],
        "past_covariates": ["HUFL", "HULL"],
        "datasets": {}
    }


@pytest.fixture
def weather_config():
    """Config for Weather-like dataset"""
    return {
        "date_column": "date",
        "freq": "10min",
        "target_columns": ["WetBulbCelsius"],
        "past_covariates": ["DewPointCelsius", "Pressure"],
        "datasets": {}
    }


@pytest.fixture
def sensor_config():
    """Config for sensor data"""
    return {
        "date_column": "date",
        "freq": "h",
        "target_columns": ["temperature"],
        "past_covariates": ["humidity"],
        "datasets": {}
    }


@pytest.fixture
def clean_ett_data():
    """Clean ETT-like data (hourly, no missing values)"""
    dates = pd.date_range('2020-01-01', periods=100, freq='h')
    return pd.DataFrame({
        'date': dates,
        'OT': np.random.randn(100),
        'HUFL': np.random.randn(100),
        'HULL': np.random.randn(100),
    })


@pytest.fixture
def weather_with_gaps():
    """Weather-like data with some missing values"""
    dates = pd.date_range('2020-01-01', periods=100, freq='10min')
    df = pd.DataFrame({
        'date': dates,
        'WetBulbCelsius': np.random.randn(100) * 5 + 20,
        'DewPointCelsius': np.random.randn(100) * 5 + 15,
        'Pressure': np.random.randn(100) * 10 + 1013,
    })
    # Introduce some random gaps
    df.loc[10:12, 'WetBulbCelsius'] = np.nan
    df.loc[45, 'Pressure'] = np.nan
    df.loc[78:79, 'DewPointCelsius'] = np.nan
    return df


@pytest.fixture
def sensor_failure_data():
    """Sensor failure - long gap"""
    dates = pd.date_range('2020-01-01', periods=100, freq='h')
    df = pd.DataFrame({
        'date': dates,
        'temperature': np.random.randn(100) * 5 + 20,
        'humidity': np.random.randn(100) * 10 + 60,
    })
    # Simulate sensor failure for 10 hours
    df.loc[40:50, 'temperature'] = np.nan
    return df


# ============================================================================
# Integration Tests
# ============================================================================

class TestCleanETTDataset:
    """Tests for clean ETT-like benchmark dataset"""

    def test_no_data_loss_on_clean_data_old(self, clean_ett_data, ett_config):
        """OLD method should not modify clean data"""
        dataset = MockSGOLDDataset(ett_config)
        result = dataset._prepare_data_old(clean_ett_data.copy())
        assert len(result) == 100
        assert result.isna().sum().sum() == 0

    def test_no_data_loss_on_clean_data_new(self, clean_ett_data, ett_config):
        """NEW method should not modify clean data"""
        dataset = MockSGOLDDataset(ett_config)
        result = dataset._prepare_data_new(clean_ett_data.copy())
        assert len(result) == 100
        assert result.isna().sum().sum() == 0

    def test_time_features_added(self, clean_ett_data, ett_config):
        """Time features should be added"""
        dataset = MockSGOLDDataset(ett_config)
        result = dataset._prepare_data_new(clean_ett_data.copy())
        assert 'month' in result.columns
        assert 'day' in result.columns


class TestWeatherDatasetWithGaps:
    """Tests for Weather-like dataset with missing values"""

    def test_old_method_drops_rows(self, weather_with_gaps, weather_config):
        """OLD method drops rows with NaNs"""
        original_nans = weather_with_gaps.isna().sum().sum()
        assert original_nans == 6  # Verify fixture

        dataset = MockSGOLDDataset(weather_config)
        result = dataset._prepare_data_old(weather_with_gaps.copy())
        assert len(result) < 100
        assert result.isna().sum().sum() == 0

    def test_new_method_preserves_data(self, weather_with_gaps, weather_config):
        """NEW method preserves data via imputation"""
        dataset = MockSGOLDDataset(weather_config)
        result = dataset._prepare_data_new(weather_with_gaps.copy())
        assert len(result) == 100
        assert result.isna().sum().sum() == 0

    def test_improvement_percentage(self, weather_with_gaps, weather_config):
        """Calculate improvement percentage"""
        dataset_old = MockSGOLDDataset(weather_config)
        dataset_new = MockSGOLDDataset(weather_config)

        result_old = dataset_old._prepare_data_old(weather_with_gaps.copy())
        result_new = dataset_new._prepare_data_new(weather_with_gaps.copy())

        rows_saved = len(result_new) - len(result_old)
        improvement_pct = (rows_saved / 100) * 100

        assert rows_saved == 6
        assert improvement_pct == 6.0


class TestSensorFailureScenario:
    """Tests for sensor failure with long gaps"""

    def test_old_method_significant_loss(self, sensor_failure_data, sensor_config):
        """OLD method loses significant data"""
        dataset = MockSGOLDDataset(sensor_config)
        result = dataset._prepare_data_old(sensor_failure_data.copy())
        rows_dropped = 100 - len(result)
        assert rows_dropped == 11

    def test_new_method_recovers_data(self, sensor_failure_data, sensor_config):
        """NEW method recovers data via interpolation"""
        dataset = MockSGOLDDataset(sensor_config)
        result = dataset._prepare_data_new(sensor_failure_data.copy())
        assert len(result) == 100
        assert result.isna().sum().sum() == 0

    def test_interpolated_values_reasonable(self, sensor_failure_data, sensor_config):
        """Interpolated values should be reasonable (not extreme)"""
        dataset = MockSGOLDDataset(sensor_config)
        result = dataset._prepare_data_new(sensor_failure_data.copy())

        # Check that interpolated region (rows 40-50) has reasonable values
        interpolated_region = result.iloc[40:51]['temperature']
        assert interpolated_region.min() > 0  # Temperature shouldn't be negative
        assert interpolated_region.max() < 50  # Should be in reasonable range


class TestDataIntegrity:
    """Tests for data integrity across all scenarios"""

    @pytest.mark.parametrize("fixture_name,config_name", [
        ("clean_ett_data", "ett_config"),
        ("weather_with_gaps", "weather_config"),
        ("sensor_failure_data", "sensor_config"),
    ])
    def test_no_nans_in_final_output(self, fixture_name, config_name, request):
        """All outputs should have zero NaNs"""
        data = request.getfixturevalue(fixture_name)
        config = request.getfixturevalue(config_name)

        dataset = MockSGOLDDataset(config)
        result = dataset._prepare_data_new(data.copy())

        assert result.isna().sum().sum() == 0, f"NaNs found in {fixture_name}"

    @pytest.mark.parametrize("fixture_name,config_name", [
        ("clean_ett_data", "ett_config"),
        ("weather_with_gaps", "weather_config"),
        ("sensor_failure_data", "sensor_config"),
    ])
    def test_datetime_index_preserved(self, fixture_name, config_name, request):
        """DatetimeIndex should be preserved"""
        data = request.getfixturevalue(fixture_name)
        config = request.getfixturevalue(config_name)

        dataset = MockSGOLDDataset(config)
        result = dataset._prepare_data_new(data.copy())

        assert isinstance(result.index, pd.DatetimeIndex)

    def test_column_order_consistency(self, weather_with_gaps, weather_config):
        """Column order should be consistent"""
        dataset = MockSGOLDDataset(weather_config)
        result = dataset._prepare_data_new(weather_with_gaps.copy())

        # Target columns should come first
        assert result.columns[0] == 'WetBulbCelsius'


# ============================================================================
# Summary Statistics
# ============================================================================

class TestSummaryMetrics:
    """Calculate and display summary metrics"""

    def test_overall_improvement_clean_data(self, clean_ett_data, ett_config):
        """Clean data: 0% improvement expected"""
        dataset_old = MockSGOLDDataset(ett_config)
        dataset_new = MockSGOLDDataset(ett_config)

        result_old = dataset_old._prepare_data_old(clean_ett_data.copy())
        result_new = dataset_new._prepare_data_new(clean_ett_data.copy())

        assert len(result_old) == len(result_new)

    def test_overall_improvement_weather(self, weather_with_gaps, weather_config):
        """Weather data: ~6% improvement expected"""
        dataset_old = MockSGOLDDataset(weather_config)
        dataset_new = MockSGOLDDataset(weather_config)

        result_old = dataset_old._prepare_data_old(weather_with_gaps.copy())
        result_new = dataset_new._prepare_data_new(weather_with_gaps.copy())

        improvement = (len(result_new) - len(result_old)) / 100 * 100
        assert improvement == pytest.approx(6.0, abs=0.1)

    def test_overall_improvement_sensor(self, sensor_failure_data, sensor_config):
        """Sensor failure: ~11% improvement expected"""
        dataset_old = MockSGOLDDataset(sensor_config)
        dataset_new = MockSGOLDDataset(sensor_config)

        result_old = dataset_old._prepare_data_old(sensor_failure_data.copy())
        result_new = dataset_new._prepare_data_new(sensor_failure_data.copy())

        improvement = (len(result_new) - len(result_old)) / 100 * 100
        assert improvement == pytest.approx(11.0, abs=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])