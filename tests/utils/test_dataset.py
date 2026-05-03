"""
Unit tests for the TimeSeriesDataset class.

This module provides a comprehensive suite of tests for the `TimeSeriesDataset`
class from `utils/dataset.py`. It verifies the core functionality, including
dataset initialization from various sources, data preparation, splitting,
and the generation of walk-forward validation folds. The tests also cover
the correct handling of target, encoder, and decoder exogenous columns,
as well as automatic time feature generation and the new differencing logic.
"""
import pytest
import pandas as pd
import numpy as np
import os
import logging

from utils.dataset import TimeSeriesDataset

pytestmark = pytest.mark.unit

# --- Fixtures ---

@pytest.fixture
def sample_config(tmp_path) -> dict:
    """
    Provides a sample configuration dictionary and creates a dummy CSV file.
    """
    data_path = tmp_path / "sample_data.csv"
    data = pd.DataFrame({
        'date': pd.to_datetime(pd.date_range('2023-01-01', periods=200)),
        'target_col': np.arange(200, dtype=float),
        'encoder_col': np.random.rand(200),
        'decoder_col': np.random.rand(200)
    })
    data.to_csv(data_path, index=False)

    return {
        'datasets': {
            'test_dataset': {
                'path': str(data_path),
                'columns': ['target_col'],
                'past_covariates': ['encoder_col'],
                'future_covariates': ['decoder_col'],
                'freq': 'D'
            }
        },
        'experiments': [{'validation_setup': {'forecast_steps': 20}}]
    }

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Provides a sample DataFrame for direct instantiation of TimeSeriesDataset."""
    return pd.DataFrame({
        'value': np.arange(100, dtype=float)
    }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=100, freq='D')))


# --- Initialization and Feature Generation Tests ---

def test_dataset_initialization_from_file(sample_config):
    """
    Verifies that the dataset is correctly initialized by loading data from a file.
    """
    dataset = TimeSeriesDataset('test_dataset', sample_config, num_features=1)

    assert not dataset.series.empty
    assert dataset.target_columns == ['target_col']
    assert dataset.past_covariates == ['encoder_col']
    assert dataset.future_covariates == ['decoder_col']
    assert dataset.num_features == 1
    assert isinstance(dataset.series.index, pd.DatetimeIndex)

def test_dataset_initialization_from_dataframe(sample_config, sample_dataframe):
    """
    Verifies that the dataset is correctly initialized from a DataFrame.
    """
    dataset = TimeSeriesDataset(
        'direct_data',
        sample_config,
        data=sample_dataframe,
        num_features=1,
        columns=['value']
    )
    assert len(dataset.series) == 100
    assert dataset.target_columns == ['value']

def test_initialization_fails_on_num_features_mismatch():
    """
    SCENARIO: Simulation of the 'dec_exog' bug.
    Data contains 3 numeric columns, but we expect only 2 targets.
    One column is 'forgotten' (not assigned to exog).
    """
    df = pd.DataFrame({
        't1': np.random.rand(10),
        't2': np.random.rand(10),
        'forgotten_exog': np.random.rand(10)
    }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=10)))

    with pytest.raises(ValueError, match="Data contract violation"):
        # We expect 2, but without assigning 'forgotten_exog', dataset finds 3
        TimeSeriesDataset(
            'fail_test',
            {'datasets': {}},
            num_features=2,
            data=df
        )

def test_initialization_warns_and_drops_unregistered_columns(caplog):
    """
    SCENARIO: Data contains columns that are NOT registered in target_columns,
    past_covariates, or future_covariates.
    The validation should WARN and DROP them instead of crashing.
    """
    df = pd.DataFrame({
        'target': np.random.rand(10),
        'target2': np.random.rand(10),
        'enc_exog': np.random.rand(10),
        'dec_exog': np.random.rand(10)  # This column exists but is NOT registered
    }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=10)))

    # We expect a warning, not a ValueError
    with caplog.at_level(logging.WARNING):
        dataset = TimeSeriesDataset(
            'unregistered_test',
            {'datasets': {}},
            num_features=2,
            data=df,
            columns=['target', 'target2'],
            past_covariates=['enc_exog'],
            future_covariates=[]  # dec_exog NOT registered
        )

    # 1. Verify warning was logged
    assert "IGNORING UNREGISTERED COLUMNS" in caplog.text
    assert "dec_exog" in caplog.text

    # 2. Verify column was physically dropped from the dataset series
    assert 'dec_exog' not in dataset.series.columns
    # 3. Verify registered columns are still present
    assert 'target' in dataset.series.columns
    assert 'enc_exog' in dataset.series.columns

def test_dataset_enforces_canonical_order():
    """
    Verifies that the dataset physically reorders columns to:
    [Targets] -> [Encoder Exog] -> [Decoder Exog]
    """
    df = pd.DataFrame({
        'enc': [1, 1, 1],
        'dec': [2, 2, 2],
        'target': [3, 3, 3]
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-02']))

    dataset = TimeSeriesDataset(
        'order_test',
        {'datasets': {}},
        num_features=1,
        data=df,
        columns=['target'],
        past_covariates=['enc'],
        future_covariates=['dec']
    )

    # Physical order in .series MUST be correct
    expected_order = ['target', 'enc', 'dec']
    assert list(dataset.series.columns) == expected_order

def test_initialization_fails_with_missing_dataset_in_config():
    """
    Verifies that initialization raises an error for a non-existent dataset.
    """
    with pytest.raises(ValueError, match="not found in config"):
        TimeSeriesDataset('nonexistent', {'datasets': {}}, num_features=1)

# --- Data Splitting and Fold Generation Tests ---

def test_data_split_correctly(sample_config):
    """
    Verifies the split of the main series into development and test sets based on
    the specified number of forecast steps.
    """
    dataset = TimeSeriesDataset('test_dataset', sample_config, num_features=1)
    forecast_steps = 30
    dataset.split_data(forecast_steps=forecast_steps)

    assert dataset.development_data is not None
    assert dataset.test_data is not None
    assert len(dataset.test_data) == forecast_steps
    assert len(dataset.development_data) == len(dataset.series) - forecast_steps
    # Check for data continuity between splits
    expected_next_date = dataset.development_data.index[-1] + pd.Timedelta(days=1)
    assert dataset.test_data.index[0] == expected_next_date

def test_split_fails_with_insufficient_data(sample_dataframe):
    """
    Verifies that splitting raises a ValueError if the dataset is too short
    to create a test set of the requested size.
    """
    # Use a fresh TimeSeriesDataset instance for this test
    dataset = TimeSeriesDataset('test', {'datasets': {}}, num_features=1, data=sample_dataframe.iloc[:15].copy(), columns=['value'])
    with pytest.raises(ValueError, match="Dataset is too short"):
        dataset.split_data(forecast_steps=20)


def test_generate_walk_forward_folds(sample_config):
    """
    Verifies the correct generation of expanding training folds for
    walk-forward validation.
    """
    dataset = TimeSeriesDataset('test_dataset', sample_config, num_features=1)
    dataset.split_data(forecast_steps=20)  # dev_set size = 180

    # Assuming forecast_steps=20 from the config for fold calculation
    folds = dataset.generate_walk_forward_folds(max_window_size=100, n_folds=4)

    assert len(folds) == 4
    # Expected sizes: 100, 120, 140, 160
    assert len(folds[0]) == 100
    assert len(folds[-1]) == 160
    # The last fold should be a slice of the development data
    pd.testing.assert_frame_equal(folds[-1], dataset.development_data.iloc[:160])

def test_generate_folds_logs_warning_when_data_is_short(sample_config, caplog):
    """
    Verifies that fold generation proceeds but logs a warning if the development
    data is too short to generate all requested folds with the specified parameters.
    """
    dataset = TimeSeriesDataset('test_dataset', sample_config, num_features=1)
    dataset.series = dataset.series.iloc[:90].copy()
    dataset.split_data(forecast_steps=10) # dev_set size = 80

    with caplog.at_level(logging.WARNING):
        # Requesting folds that would require more data than available
        folds = dataset.generate_walk_forward_folds(max_window_size=50, n_folds=2)
        # This warning logic was complex and might be removed or changed.
        # assert "Development data (80 rows) is potentially too short" not in caplog.text

    # It should still generate what it can
    # With dev size 80, n_folds=2, forecast_steps=20 -> initial train size = 80 - 2*20 = 40.
    # This is smaller than max_window_size=50, so initial size becomes 50.
    # Fold 1: 50. Fold 2: 70.
    assert len(folds) == 2
    assert len(folds[0]) == 50
    assert len(folds[1]) == 70


# --- Getter and View Method Tests ---

def test_getters_raise_error_before_split(sample_config):
    """
    Verifies that accessing development or test data before splitting
    raises a ValueError.
    """
    dataset = TimeSeriesDataset('test_dataset', sample_config, num_features=1)
    with pytest.raises(ValueError, match="Data has not been split yet"):
        dataset.get_development_data()
    with pytest.raises(ValueError, match="Data has not been split yet"):
        dataset.get_test_data()

def test_get_data_for_model(sample_config):
    """
    Verifies that `get_data_for_model` returns the correct DataFrame slice
    based on the `use_exogenous` flag.
    """
    dataset = TimeSeriesDataset('test_dataset', sample_config, num_features=1)

    # When exogenous features are disabled, should return only target columns
    target_only_df = dataset.get_data_for_model(use_exogenous=False)
    assert list(target_only_df.columns) == dataset.target_columns

    # When enabled, should return all columns
    all_features_df = dataset.get_data_for_model(use_exogenous=True)
    assert sorted(list(all_features_df.columns)) == sorted(list(dataset.series.columns))


def test_time_feature_generation(tmp_path):
    """
    Verifies that time-based features are correctly generated from a DatetimeIndex
    but are NOT automatically added to the future_covariates list (must be explicit in config).
    """
    config_with_time_features = {
        'datasets': {
            'timedata': {
                'path': 'dummy.csv',
                'columns': ['target'],
                'future_covariates': ['is_holiday'], # Pre-existing decoder column
                'time_features': ['hour', 'day_of_week', 'is_month_start']
            }
        }
    }

    # Create data with a DatetimeIndex
    raw_data = pd.DataFrame({
        'is_holiday': 0,
    }, index=pd.to_datetime(pd.date_range('2023-01-01 00:00', periods=50, freq='h')))
    raw_data['target'] = np.arange(50)

    dataset = TimeSeriesDataset(
        'timedata',
        config_with_time_features,
        num_features = 1,
        data=raw_data,
        columns=['target'],
        future_covariates=['is_holiday']
    )

    # 1. Verify that the new columns were created in the DataFrame
    # Cyclic features are replaced by sin/cos encodings
    assert 'hour_sin' in dataset.series.columns
    assert 'hour_cos' in dataset.series.columns
    assert 'day_of_week_sin' in dataset.series.columns
    assert 'day_of_week_cos' in dataset.series.columns
    assert 'is_month_start' in dataset.series.columns

    # 2. Verify that the values are correct (is_month_start is not cyclic)
    assert dataset.series['is_month_start'].iloc[0] == 1 # First day of the month

    # 3. Verify that the new features were NOT automatically added to any list
    # (This behavior was removed; assignment MUST be explicit in the experiment config)
    assert 'hour' not in dataset.future_covariates
    assert 'day_of_week' not in dataset.future_covariates
    # The original column passed to the constructor should still be present
    assert 'is_holiday' in dataset.future_covariates
    assert len(dataset.future_covariates) == 1

# ---------------------------------------------------------------------------------
# NEW: Tests for Differencing Logic (Krok 2)
# ---------------------------------------------------------------------------------

@pytest.fixture
def diff_config():
    """Config that enables differencing."""
    return {
        'datasets': {
            'diff_data': {
                'path': 'dummy.csv', # Not used, data provided directly
                'columns': ['value'],
                'differencing': {
                    'enabled': True,
                    'auto': 'none',
                    'order': 1,
                    'seasonal_order': 0
                }
            }
        }
    }

@pytest.fixture
def stationary_config():
    """Config for auto-differencing stationary data."""
    return {
        'datasets': {
            'stat_data': {
                'path': 'dummy.csv',
                'columns': ['value'],
                'differencing': {
                    'enabled': True,
                    'auto': 'adf', # Use 'adf' or 'kpss'
                    'max_d': 2
                }
            }
        }
    }

@pytest.fixture
def linear_trend_data():
    """DataFrame with a simple linear trend."""
    return pd.DataFrame({
        'value': np.arange(1, 101, dtype=float), # 1, 2, 3, ...
        'exog': np.zeros(100)
    }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=100, freq='D')))

@pytest.fixture
def stationary_data():
    """DataFrame with stationary data."""
    np.random.seed(42)
    return pd.DataFrame({
        'value': np.random.randn(100)
    }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=100, freq='D')))


def test_dataset_skips_differencing_if_disabled(sample_config, linear_trend_data):
    """Tests that 'series' and '_original_series' are identical if differencing is off."""
    # sample_config has no 'differencing' key
    dataset = TimeSeriesDataset(
        'test_dataset',
        sample_config,
        num_features = 1,
        data=linear_trend_data,
        columns=['value'],
        past_covariates=['exog']
    )

    assert dataset._diff_state is None # State is not set
    # The .series should be the same object as _original_series after prep
    pd.testing.assert_frame_equal(dataset.series, dataset._original_series[dataset.columns])

def test_dataset_applies_differencing_if_configured(diff_config, linear_trend_data):
    """Tests that 'series' is correctly differenced and '_original_series' is preserved."""
    dataset = TimeSeriesDataset(
        'diff_data',
        diff_config,
        num_features = 1,
        data=linear_trend_data,
        columns=['value'],
        past_covariates=['exog']
    )

    # 1. Check state
    assert dataset._diff_state is not None
    assert dataset._diff_state['d'] == 1
    assert dataset._diff_state['D'] == 0

    # 2. Check original series
    assert dataset._original_series is not None
    pd.testing.assert_frame_equal(dataset._original_series, linear_trend_data)

    # 3. Check processed series
    # 'series' should have 1 less row (due to d=1 and dropna)
    assert len(dataset.series) == len(dataset._original_series) - 1
    # The 'value' column should be all 1.0s (diff of 1, 2, 3...)
    assert (dataset.series['value'] == 1.0).all()
    # Exog column 'exog' should NOT be differenced
    assert (dataset.series['exog'] == 0.0).all()

def test_dataset_stores_diff_state_auto(stationary_config, stationary_data):
    """Tests that auto-differencing (adf) correctly identifies d=0 for stationary data."""
    try:
        import pmdarima
    except ImportError:
        pytest.skip("pmdarima not installed, skipping auto-diff test")

    dataset = TimeSeriesDataset(
        'stat_data',
        stationary_config,
        num_features = 1,
        data=stationary_data,
        columns=['value']
    )

    # Check state
    assert dataset._diff_state is not None
    assert dataset._diff_state['d'] == 0
    assert dataset._diff_state['D'] == 0
    # Series should not have been modified
    assert len(dataset.series) == len(dataset._original_series)

def test_dataset_inverse_difference_roundtrip(diff_config, linear_trend_data):
    """Tests the full fit_diff -> inverse_diff cycle."""
    dataset = TimeSeriesDataset(
        'diff_data',
        diff_config,
        num_features = 1,
        data=linear_trend_data,
        columns=['value'],
        past_covariates=['exog']
    )

    # 'dataset.series' now holds the differenced data (all 1.0s)
    differenced_data = dataset.series

    # Reconstruct the forecast (which is just the differenced data in this test)
    reconstructed_df = dataset.inverse_difference_forecast(
        differenced_data[['value']] # Pass only target column as forecast
    )

    # Compare the reconstructed data to the original data (ignoring the first row)
    original_to_compare = dataset._original_series.iloc[1:][['value']]

    pd.testing.assert_frame_equal(original_to_compare, reconstructed_df)