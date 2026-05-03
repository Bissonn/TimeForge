"""
Comprehensive pytest suite for missing value handling in time series data.
Tests various edge cases and scenarios.

Run with:
    pytest test_missing_value_handling_pytest.py -v
    pytest test_missing_value_handling_pytest.py -v -s  # with print output
    pytest test_missing_value_handling_pytest.py -v --tb=short  # short traceback
"""

import pandas as pd
import numpy as np
import logging
import pytest
from typing import Tuple, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Core Functions Under Test
# ============================================================================

def handle_missing_values_new(df: pd.DataFrame) -> pd.DataFrame:
    """
    NEW IMPLEMENTATION - bulletproof version with tiered approach
    """
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
            # Tier 1: Forward fill with limit
            df_filled = df.ffill(limit=3)
            nans_after_ffill = df_filled.isna().sum().sum()

            if nans_after_ffill > 0:
                logger.info(
                    f"After forward fill: {nans_after_ffill} NaNs remaining. "
                    f"Applying linear interpolation..."
                )
                # Tier 2: Interpolation
                df_filled = df_filled.interpolate(
                    method='linear',
                    limit_direction='both',
                    axis=0
                )
                nans_after_interp = df_filled.isna().sum().sum()

                if nans_after_interp > 0:
                    # Tier 3: Dropna if necessary
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
            # Fallback for non-temporal data
            logger.warning(
                "No DatetimeIndex found. Using simple dropna() without imputation."
            )
            rows_before = len(df)
            df = df.dropna()
            rows_dropped = rows_before - len(df)
            if rows_dropped > 0:
                logger.warning(f"Dropped {rows_dropped} rows with NaNs.")

    return df


def handle_missing_values_old(df: pd.DataFrame) -> pd.DataFrame:
    """
    OLD IMPLEMENTATION - simple dropna
    """
    if df.isna().any().any():
        logger.warning("NaN values detected in data. Dropping rows with NaNs.")
        df = df.dropna()
    return df


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def clean_data():
    """Clean data with no missing values"""
    return pd.DataFrame({
        'target': [1, 2, 3, 4, 5],
        'exog': [10, 20, 30, 40, 50]
    }, index=pd.date_range('2020-01-01', periods=5, freq='D'))


@pytest.fixture
def short_gaps():
    """Small gaps that should be filled by ffill(limit=3)"""
    return pd.DataFrame({
        'target': [1, np.nan, np.nan, 4, 5, 6],
        'exog': [10, 20, np.nan, 40, 50, 60]
    }, index=pd.date_range('2020-01-01', periods=6, freq='D'))


@pytest.fixture
def long_gaps():
    """Long gaps that need interpolation after ffill fails"""
    return pd.DataFrame({
        'target': [1, np.nan, np.nan, np.nan, np.nan, 6, 7],
        'exog': [10, 20, 30, 40, 50, 60, 70]
    }, index=pd.date_range('2020-01-01', periods=7, freq='D'))


@pytest.fixture
def all_nan_column():
    """Entire column is NaN - should be dropped"""
    return pd.DataFrame({
        'target': [1, 2, 3, 4, 5],
        'exog': [np.nan, np.nan, np.nan, np.nan, np.nan]
    }, index=pd.date_range('2020-01-01', periods=5, freq='D'))


@pytest.fixture
def leading_nans():
    """NaNs at the beginning"""
    return pd.DataFrame({
        'target': [np.nan, np.nan, 3, 4, 5],
        'exog': [np.nan, 20, 30, 40, 50]
    }, index=pd.date_range('2020-01-01', periods=5, freq='D'))


@pytest.fixture
def trailing_nans():
    """NaNs at the end"""
    return pd.DataFrame({
        'target': [1, 2, 3, np.nan, np.nan],
        'exog': [10, 20, 30, np.nan, np.nan]
    }, index=pd.date_range('2020-01-01', periods=5, freq='D'))


@pytest.fixture
def scattered_nans():
    """Single NaNs scattered throughout"""
    return pd.DataFrame({
        'target': [1, np.nan, 3, np.nan, 5, np.nan, 7],
        'exog': [10, np.nan, 30, 40, np.nan, 60, 70]
    }, index=pd.date_range('2020-01-01', periods=7, freq='D'))


@pytest.fixture
def no_datetime_index():
    """Data without DatetimeIndex"""
    return pd.DataFrame({
        'target': [1, np.nan, 3, 4, 5],
        'exog': [10, 20, np.nan, 40, 50]
    })


@pytest.fixture
def entire_row_nan():
    """One row where all values are NaN"""
    return pd.DataFrame({
        'target': [1, np.nan, 3, 4, 5],
        'exog': [10, np.nan, 30, 40, 50]
    }, index=pd.date_range('2020-01-01', periods=5, freq='D'))


@pytest.fixture
def complex_pattern():
    """Realistic scenario with various gap lengths"""
    return pd.DataFrame({
        'target': [np.nan, 2, np.nan, np.nan, 5, 6, np.nan, np.nan, np.nan, 10],
        'exog': [10, np.nan, 30, np.nan, np.nan, 60, 70, 80, np.nan, 100]
    }, index=pd.date_range('2020-01-01', periods=10, freq='D'))


# ============================================================================
# Test Classes
# ============================================================================

class TestCleanData:
    """Tests for clean data (baseline)"""

    def test_no_nans_old(self, clean_data):
        """OLD method should not modify clean data"""
        result = handle_missing_values_old(clean_data.copy())
        assert result.shape == clean_data.shape
        assert result.isna().sum().sum() == 0
        pd.testing.assert_frame_equal(result, clean_data)

    def test_no_nans_new(self, clean_data):
        """NEW method should not modify clean data"""
        result = handle_missing_values_new(clean_data.copy())
        assert result.shape == clean_data.shape
        assert result.isna().sum().sum() == 0
        pd.testing.assert_frame_equal(result, clean_data)


class TestShortGaps:
    """Tests for short gaps (≤3 timesteps)"""

    def test_short_gaps_old_drops_rows(self, short_gaps):
        """OLD method drops rows with NaNs"""
        original_len = len(short_gaps)
        result = handle_missing_values_old(short_gaps.copy())
        assert len(result) < original_len
        assert result.isna().sum().sum() == 0

    def test_short_gaps_new_preserves_data(self, short_gaps):
        """NEW method preserves all rows via forward fill"""
        result = handle_missing_values_new(short_gaps.copy())
        assert result.shape == short_gaps.shape
        assert result.isna().sum().sum() == 0

    def test_short_gaps_improvement(self, short_gaps):
        """NEW method should preserve more data than OLD"""
        result_old = handle_missing_values_old(short_gaps.copy())
        result_new = handle_missing_values_new(short_gaps.copy())
        assert len(result_new) > len(result_old)


class TestLongGaps:
    """Tests for long gaps (>3 timesteps)"""

    def test_long_gaps_old_drops_many_rows(self, long_gaps):
        """OLD method drops many rows"""
        original_len = len(long_gaps)
        result = handle_missing_values_old(long_gaps.copy())
        rows_dropped = original_len - len(result)
        assert rows_dropped >= 4

    def test_long_gaps_new_uses_interpolation(self, long_gaps):
        """NEW method uses interpolation for long gaps"""
        result = handle_missing_values_new(long_gaps.copy())
        # Should preserve all rows
        assert result.shape == long_gaps.shape
        assert result.isna().sum().sum() == 0

    def test_long_gaps_improvement(self, long_gaps):
        """NEW method should save significant data"""
        result_old = handle_missing_values_old(long_gaps.copy())
        result_new = handle_missing_values_new(long_gaps.copy())
        rows_saved = len(result_new) - len(result_old)
        assert rows_saved >= 3


class TestAllNaNColumn:
    """Tests for entire column being NaN"""

    def test_all_nan_column_old(self, all_nan_column):
        """OLD method drops all rows"""
        result = handle_missing_values_old(all_nan_column.copy())
        assert len(result) == 0

    def test_all_nan_column_new(self, all_nan_column):
        """NEW method should also drop all rows (can't impute all-NaN column)"""
        result = handle_missing_values_new(all_nan_column.copy())
        assert len(result) == 0

    def test_all_nan_column_same_behavior(self, all_nan_column):
        """Both methods should behave the same for all-NaN column"""
        result_old = handle_missing_values_old(all_nan_column.copy())
        result_new = handle_missing_values_new(all_nan_column.copy())
        assert len(result_old) == len(result_new)


class TestLeadingTrailingNaNs:
    """Tests for NaNs at boundaries"""

    def test_leading_nans_new_handles_via_interpolation(self, leading_nans):
        """NEW method handles leading NaNs via interpolation"""
        result = handle_missing_values_new(leading_nans.copy())
        assert result.shape == leading_nans.shape
        assert result.isna().sum().sum() == 0

    def test_trailing_nans_new_handles_via_interpolation(self, trailing_nans):
        """NEW method handles trailing NaNs via interpolation"""
        result = handle_missing_values_new(trailing_nans.copy())
        assert result.shape == trailing_nans.shape
        assert result.isna().sum().sum() == 0

    def test_boundary_nans_improvement(self, leading_nans, trailing_nans):
        """NEW method preserves data at boundaries"""
        result_old_leading = handle_missing_values_old(leading_nans.copy())
        result_new_leading = handle_missing_values_new(leading_nans.copy())
        assert len(result_new_leading) > len(result_old_leading)

        result_old_trailing = handle_missing_values_old(trailing_nans.copy())
        result_new_trailing = handle_missing_values_new(trailing_nans.copy())
        assert len(result_new_trailing) > len(result_old_trailing)


class TestScatteredNaNs:
    """Tests for scattered single NaNs"""

    def test_scattered_nans_easy_to_fill(self, scattered_nans):
        """Single scattered NaNs should be easily filled"""
        result = handle_missing_values_new(scattered_nans.copy())
        assert result.shape == scattered_nans.shape
        assert result.isna().sum().sum() == 0

    def test_scattered_nans_improvement(self, scattered_nans):
        """NEW method should save all scattered NaN rows"""
        result_old = handle_missing_values_old(scattered_nans.copy())
        result_new = handle_missing_values_new(scattered_nans.copy())
        assert len(result_new) > len(result_old)


class TestNoDatetimeIndex:
    """Tests for data without DatetimeIndex"""

    def test_no_datetime_index_falls_back_to_dropna(self, no_datetime_index):
        """NEW method should fall back to dropna for non-temporal data"""
        result_old = handle_missing_values_old(no_datetime_index.copy())
        result_new = handle_missing_values_new(no_datetime_index.copy())
        # Should behave the same
        assert len(result_old) == len(result_new)
        assert result_old.isna().sum().sum() == 0
        assert result_new.isna().sum().sum() == 0


class TestComplexPattern:
    """Tests for realistic complex patterns"""

    def test_complex_pattern_significant_improvement(self, complex_pattern):
        """NEW method should save significant data on complex patterns"""
        original_len = len(complex_pattern)
        result_old = handle_missing_values_old(complex_pattern.copy())
        result_new = handle_missing_values_new(complex_pattern.copy())

        rows_saved = len(result_new) - len(result_old)
        improvement_pct = (rows_saved / original_len) * 100

        # Should save at least 50% of data
        assert improvement_pct >= 50
        assert result_new.isna().sum().sum() == 0


class TestDataIntegrity:
    """Tests for data integrity guarantees"""

    @pytest.mark.parametrize("fixture_name", [
        "clean_data",
        "short_gaps",
        "long_gaps",
        "leading_nans",
        "trailing_nans",
        "scattered_nans",
        "complex_pattern"
    ])
    def test_no_nans_in_output(self, fixture_name, request):
        """All outputs should have zero NaNs (except all_nan_column)"""
        df = request.getfixturevalue(fixture_name)
        result = handle_missing_values_new(df.copy())
        assert result.isna().sum().sum() == 0, f"NaNs found in output for {fixture_name}"

    def test_no_infinite_values(self, complex_pattern):
        """Output should not contain infinite values"""
        result = handle_missing_values_new(complex_pattern.copy())
        numeric_cols = result.select_dtypes(include=np.number)
        assert not np.any(np.isinf(numeric_cols.values))

    def test_column_names_preserved(self, short_gaps):
        """Column names should be preserved"""
        result = handle_missing_values_new(short_gaps.copy())
        assert list(result.columns) == list(short_gaps.columns)

    def test_datetime_index_preserved(self, short_gaps):
        """DatetimeIndex should be preserved"""
        result = handle_missing_values_new(short_gaps.copy())
        assert isinstance(result.index, pd.DatetimeIndex)


# ============================================================================
# Summary Statistics
# ============================================================================

@pytest.fixture(scope="session")
def test_results():
    """Collect results across all tests for summary"""
    return {
        "rows_saved": [],
        "tests_with_improvement": 0,
        "total_tests": 0
    }


def pytest_sessionfinish(session, exitstatus):
    """Print summary after all tests"""
    print("\n" + "=" * 80)
    print("MISSING VALUE HANDLING TEST SUMMARY")
    print("=" * 80)
    print(f"Exit status: {'PASSED' if exitstatus == 0 else 'FAILED'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Allow running with: python test_missing_value_handling_pytest.py
    pytest.main([__file__, "-v", "-s"])