"""
REAL integration tests for  TimeSeriesDataset._prepare_data()

These tests import YOUR actual dataset.py code and test it!
This is what you should run BEFORE and AFTER implementing the fix.

Usage:
    # Before fix - some tests SHOULD FAIL
    pytest test_dataset_nan_handling_REAL.py -v

    # After fix - all tests SHOULD PASS
    pytest test_dataset_nan_handling_REAL.py -v
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add your project to path - ADJUST THIS TO YOUR STRUCTURE
# Option 1: If you have src/ structure
sys.path.insert(0, str(Path(__file__).parent / 'src'))
# Option 2: If utils is at root
# sys.path.insert(0, str(Path(__file__).parent))

# Try to import - will fail if path is wrong
try:
    from utils.dataset import TimeSeriesDataset

    DATASET_AVAILABLE = True
except ImportError as e:
    DATASET_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Skip all tests if import failed
pytestmark = pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason=f"Cannot import TimeSeriesDataset: {IMPORT_ERROR if not DATASET_AVAILABLE else ''}"
)


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture
def basic_config():
    return {
        "datasets": {
            "test": {
                "date_column": "date",
                "freq": "D",
                "target_columns": ["target"],
                "past_covariates": ["exog"],
                "future_covariates": [],
            }
        }
    }

# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def clean_data():
    """Clean data - no NaNs - should work fine"""
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    return pd.DataFrame({
        'date': dates,
        'target': np.random.randn(20),
        'exog': np.random.randn(20)
    })


@pytest.fixture
def data_with_short_gap():
    """
    Data with 2 consecutive NaNs (short gap)

    CURRENT BEHAVIOR: dropna() will drop 2 rows → 18 rows remain
    DESIRED BEHAVIOR: ffill(limit=3) will preserve all → 20 rows remain
    """
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'target': np.random.randn(20),
        'exog': np.random.randn(20)
    })
    # Add short gap
    df.loc[5:6, 'target'] = np.nan
    return df


@pytest.fixture
def data_with_scattered_nans():
    """
    Data with scattered single NaNs

    CURRENT BEHAVIOR: dropna() will drop ~5 rows → 15 rows remain
    DESIRED BEHAVIOR: ffill/interpolate preserves all → 20 rows remain
    """
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'target': np.random.randn(20),
        'exog': np.random.randn(20)
    })
    # Scatter some NaNs
    df.loc[3, 'target'] = np.nan
    df.loc[8, 'exog'] = np.nan
    df.loc[12, 'target'] = np.nan
    df.loc[15, 'exog'] = np.nan
    df.loc[18, 'target'] = np.nan
    return df


@pytest.fixture
def data_with_long_gap():
    """
    Data with long gap (5 consecutive NaNs)

    CURRENT BEHAVIOR: dropna() will drop 5 rows → 15 rows remain
    DESIRED BEHAVIOR: interpolate preserves all → 20 rows remain
    """
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'target': np.random.randn(20),
        'exog': np.random.randn(20)
    })
    # Add long gap
    df.loc[8:12, 'target'] = np.nan
    return df


# ============================================================================
# Tests - Current Behavior (WILL FAIL with current dropna implementation)
# ============================================================================

class TestCurrentNaNHandling:
    """
    These tests document CURRENT BEHAVIOR with simple dropna()
    They SHOULD FAIL if you want to preserve more data!
    """

    def test_clean_data_preserved(self, clean_data, basic_config):
        """Clean data should be fully preserved - this SHOULD PASS"""
        dataset = TimeSeriesDataset(dataset_name="test", config=basic_config, num_features=1, data=clean_data,
                                    columns=["target"], past_covariates=["exog"])
        assert dataset.total_length == 20, "Clean data should not lose any rows"

    def test_short_gap_preserved(self, data_with_short_gap, basic_config):
        """
        Short gaps should be preserved via ffill
        """
        original_nans = data_with_short_gap.isna().sum().sum()
        assert original_nans > 0, "Test data should have NaNs"

        dataset = TimeSeriesDataset(dataset_name="test", config=basic_config, num_features=1, data=data_with_short_gap,
                                    columns=["target"], past_covariates=["exog"])

        # After NEW implementation, this should pass
        assert dataset.total_length == 20, f"Expected 20 rows, got {len(dataset.data)}"
        assert dataset.series.isna().sum().sum() == 0, "Should have no NaNs after processing"

    def test_scattered_nans_preserved(self, data_with_scattered_nans, basic_config):
        """
        Scattered NaNs should be filled
        """
        dataset = TimeSeriesDataset(dataset_name="test", config=basic_config, num_features=1, data=data_with_scattered_nans,
                                    columns=["target"], past_covariates=["exog"])

        assert dataset.total_length == 20, f"Expected 20 rows, got {len(dataset.data)}"
        assert dataset.series.isna().sum().sum() == 0

    def test_long_gap_handled(self, data_with_long_gap, basic_config):
        """
        Long gaps should be interpolated
        """
        dataset = TimeSeriesDataset(dataset_name="test", config=basic_config, num_features=1, data=data_with_long_gap,
                                    columns=["target"], past_covariates=["exog"])

        assert dataset.total_length == 20, f"Expected 20 rows, got {len(dataset.data)}"
        assert dataset.series.isna().sum().sum() == 0


class TestDataIntegrity:
    """
    These tests check data integrity - should ALWAYS pass
    """

    def test_no_nans_in_final_output(self, data_with_short_gap, basic_config):
        """Final output should NEVER have NaNs"""
        dataset = TimeSeriesDataset(dataset_name="test", config=basic_config, num_features=1, data=data_with_short_gap,
                                    columns=["target"], past_covariates=["exog"])
        assert dataset.series.isna().sum().sum() == 0, "Final data must not contain NaNs"

    def test_no_infinite_values(self, data_with_short_gap, basic_config):
        """Final output should NEVER have infinite values"""
        dataset = TimeSeriesDataset(dataset_name="test", config=basic_config, num_features=1, data=data_with_short_gap,
                                    columns=["target"], past_covariates=["exog"])
        numeric_cols = dataset.series.select_dtypes(include=np.number)
        assert not np.any(np.isinf(numeric_cols.values)), "Final data must not contain inf"

    def test_datetime_index_preserved(self, clean_data, basic_config):
        """DatetimeIndex should be preserved"""
        dataset = TimeSeriesDataset(dataset_name="test", config=basic_config, num_features=1, data=clean_data,
                                    columns=["target"], past_covariates=["exog"])
        assert isinstance(dataset.series.index, pd.DatetimeIndex)


class TestDataLoss:
    """
    These tests measure data loss - higher numbers = more data lost
    With current dropna(), these show significant loss
    """

    def test_short_gap_data_loss(self, data_with_short_gap, basic_config):
        """Measure data loss for short gaps"""
        original_len = len(data_with_short_gap)
        dataset = TimeSeriesDataset(dataset_name="test", config=basic_config, num_features=1, data=data_with_short_gap,
                                    columns=["target"], past_covariates=["exog"])
        final_len = dataset.total_length

        loss = original_len - final_len
        loss_pct = (loss / original_len) * 100

        print(f"\nData loss for short gap: {loss} rows ({loss_pct:.1f}%)")

        # Current dropna: will lose 2 rows (10%)
        # NEW method: will lose 0 rows (0%)
        # This test just reports - doesn't fail

    def test_scattered_nans_data_loss(self, data_with_scattered_nans, basic_config):
        """Measure data loss for scattered NaNs"""
        original_len = len(data_with_scattered_nans)
        dataset = TimeSeriesDataset(dataset_name="test", config=basic_config, num_features=1, data=data_with_scattered_nans,
                                    columns=["target"], past_covariates=["exog"])
        final_len = dataset.total_length

        loss = original_len - final_len
        loss_pct = (loss / original_len) * 100

        print(f"\nData loss for scattered NaNs: {loss} rows ({loss_pct:.1f}%)")

    def test_long_gap_data_loss(self, data_with_long_gap, basic_config):
        """Measure data loss for long gap"""
        original_len = len(data_with_long_gap)
        dataset = TimeSeriesDataset(dataset_name="test", config=basic_config, num_features=1, data=data_with_long_gap,
                                    columns=["target"], past_covariates=["exog"])
        final_len = dataset.total_length

        loss = original_len - final_len
        loss_pct = (loss / original_len) * 100

        print(f"\nData loss for long gap: {loss} rows ({loss_pct:.1f}%)")


# ============================================================================
# Summary Report
# ============================================================================

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print summary after all tests"""
    print("\n" + "=" * 70)
    print("SUMMARY: NaN Handling Tests")
    print("=" * 70)

    if not DATASET_AVAILABLE:
        print("⚠️  Could not import TimeSeriesDataset - tests skipped")
        print(f"Error: {IMPORT_ERROR}")
        print("\nTo fix:")
        print("1. Adjust sys.path in this file to point to your project")
        print("2. Make sure utils/dataset.py exists")
        return

    print("\n📊 Expected Results:")
    print("\nBEFORE implementing fix:")
    print("  ❌ test_short_gap_preserved - XFAIL (expected)")
    print("  ❌ test_scattered_nans_preserved - XFAIL (expected)")
    print("  ❌ test_long_gap_handled - XFAIL (expected)")
    print("  ✅ test_*_data_loss - PASS but shows data loss")
    print("  ✅ test_no_nans_in_final_output - PASS (dropna works)")

    print("\nAFTER implementing fix:")
    print("  ✅ test_short_gap_preserved - PASS")
    print("  ✅ test_scattered_nans_preserved - PASS")
    print("  ✅ test_long_gap_handled - PASS")
    print("  ✅ test_*_data_loss - PASS with 0% loss")
    print("  ✅ test_no_nans_in_final_output - PASS")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])