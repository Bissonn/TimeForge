"""
Unit tests for the experiment management utility.

This module provides a comprehensive suite of tests for the classes in
`utils.experiment_manager`, which are central to the object-oriented
refactoring of the experiment configuration and data preparation pipeline.
It ensures that `DataSpec`, `ExperimentRun`, and `DataProvider` are
instantiated correctly and interact as expected, including the logic
for configuration overrides.
"""
import pytest
from core.experiment_manager import ExperimentRun, DataSpec, DataProvider
from utils.dataset import TimeSeriesDataset
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit

# --- Fixtures for Configuration and Data ---

@pytest.fixture
def mock_model_config() -> dict:
    """
    Provides a sample model configuration dictionary as found within
    an experiment block in `config.yaml`.
    """
    return {
        'name': 'transformer',
        'use_exogenous': True,
    }

@pytest.fixture
def mock_dataset_config() -> dict:
    """
    Provides a sample dataset configuration dictionary as found in the
    top-level `datasets` block in `config.yaml`. This represents the "global" config.
    """
    return {
        'path': 'dummy/path.csv',
        'columns': ['target_1'],
        'past_covariates': ['global_enc_exog'],
        'future_covariates': ['global_dec_exog'],
        'time_features': ['hour', 'day_of_week']
    }

@pytest.fixture
def mock_validation_config() -> dict:
    """Provides a sample validation setup dictionary."""
    return {
        'forecast_steps': 10,
        'n_folds': 3,
        'max_window_size': 50
    }

@pytest.fixture
def mock_full_dataset(mock_dataset_config) -> TimeSeriesDataset:
    """
    Creates a full `TimeSeriesDataset` instance that mimics one loaded
    from a file, containing all potential columns.

    NEW (Step 4): This mock now includes both .series (processed)
    and ._original_series (raw) with different values to test the switch.
    """
    # The DataFrame column names must exactly match the names defined
    # in the mock_dataset_config fixture to ensure consistency.
    data = pd.DataFrame({
        'target_1': np.array([10, 20, 30, 40, 50], dtype=float),
        'global_enc_exog': np.random.rand(5),
        'global_dec_exog': np.random.rand(5),
    })

    # The config dictionary passed to TimeSeriesDataset still needs the 'datasets' key
    # for other internal lookups.
    full_config = {"datasets": {"test_data": mock_dataset_config}}

    # We use a MagicMock to easily add the _original_series attribute
    mock_ds = MagicMock(spec=TimeSeriesDataset)
    mock_ds.name = "test_data"
    mock_ds.config = full_config
    mock_ds.freq = 'D'

    # --- Define the two data versions ---
    # Processed (e.g., differenced: [10, 10, 10, 10])
    mock_ds.series = data.copy()
    mock_ds.series['target_1'] = data['target_1'].diff().fillna(0) # [10, 10, 10, 10, 10]

    # Raw
    mock_ds._original_series = data.copy() # [10, 20, 30, 40, 50]

    mock_ds.target_columns = mock_dataset_config['columns']
    mock_ds.past_covariates = mock_dataset_config['past_covariates']
    mock_ds.future_covariates = mock_dataset_config['future_covariates']

    mock_ds._diff_state = {'d': 1, 'D': 0, 's': 1} # Mock that differencing happened

    return mock_ds

# --- Test Cases for Each Class ---

class TestDataSpec:
    """Tests the DataSpec class, which acts as a data contract."""

    def test_instantiation_with_all_params(self):
        """
        Verifies that a DataSpec object correctly assigns all provided
        parameters to its attributes upon creation.
        """
        spec = DataSpec(
            target_columns=['t1', 't2'],
            past_covariates=['enc1'],
            future_covariates=['dec1'],
            time_features_to_generate=['hour']
        )
        assert spec.target_columns == ['t1', 't2']
        assert spec.past_covariates == ['enc1']
        assert spec.future_covariates == ['dec1']
        assert spec.time_features_to_generate == ['hour']

    def test_instantiation_with_missing_optional_params(self):
        """
        Verifies that a DataSpec object correctly defaults its optional
        attributes to empty lists when they are not provided.
        """
        spec = DataSpec(target_columns=['t1'])
        assert spec.target_columns == ['t1']
        assert spec.past_covariates == []
        assert spec.future_covariates == []
        assert spec.time_features_to_generate == []

class TestExperimentRun:
    """Tests the ExperimentRun class, focusing on configuration resolution."""

    def test_instantiation(self, mock_model_config, mock_dataset_config, mock_validation_config):
        """
        Verifies that an ExperimentRun object correctly extracts and stores
        core attributes from the raw configuration dictionaries.
        """
        run = ExperimentRun(
            model_config=mock_model_config,
            dataset_config=mock_dataset_config,
            validation_config=mock_validation_config,
            dataset_name="test_data"
        )
        assert run.model_name == 'transformer'
        assert run.dataset_name == 'test_data'
        assert run.model_params == mock_model_config

    def test_get_data_spec_uses_global_config_as_fallback(self, mock_model_config, mock_dataset_config, mock_validation_config):
        """
        Verifies that `get_data_spec()` correctly falls back to the global
        dataset configuration when no local overrides for exogenous columns
        are present in the model configuration.
        """
        # Ensure the model config does NOT have local overrides
        model_config_no_override = mock_model_config.copy()

        run = ExperimentRun(
            model_config=model_config_no_override,
            dataset_config=mock_dataset_config,
            validation_config=mock_validation_config,
            dataset_name="test_data"
        )
        spec = run.get_data_spec()

        assert spec.past_covariates == mock_dataset_config['past_covariates']
        assert spec.future_covariates == mock_dataset_config['future_covariates']
        assert spec.time_features_to_generate == mock_dataset_config['time_features']

    def test_get_data_spec_uses_local_override(self, mock_model_config, mock_dataset_config, mock_validation_config):
        """
        Verifies that `get_data_spec()` correctly uses the local `exog_*_columns`
        lists from the model configuration, ignoring the global ones.
        """
        model_config_with_override = mock_model_config.copy()
        model_config_with_override['past_covariates'] = ['local_encoder_col']
        model_config_with_override['future_covariates'] = ['local_decoder_col']

        run = ExperimentRun(
            model_config=model_config_with_override,
            dataset_config=mock_dataset_config,
            validation_config=mock_validation_config,
            dataset_name="test_data"
        )
        spec = run.get_data_spec()

        assert spec.past_covariates == ['local_encoder_col']
        assert spec.future_covariates == ['local_decoder_col']
        assert spec.past_covariates != mock_dataset_config['past_covariates']

    def test_get_data_spec_handles_empty_list_override(self, mock_model_config, mock_dataset_config, mock_validation_config):
        """
        Verifies that providing an empty list as a local override correctly
        disables that type of exogenous feature for the run.
        """
        model_config_with_empty_override = mock_model_config.copy()
        model_config_with_empty_override['past_covariates'] = []

        run = ExperimentRun(
            model_config=model_config_with_empty_override,
            dataset_config=mock_dataset_config,
            validation_config=mock_validation_config,
            dataset_name="test_data"
        )
        spec = run.get_data_spec()

        assert spec.past_covariates == []
        assert spec.future_covariates == mock_dataset_config['future_covariates']

    def test_get_data_spec_handles_use_exogenous_false(self, mock_model_config, mock_dataset_config, mock_validation_config):
        """
        Verifies that if `use_exogenous` is false, all exogenous column
        lists are empty, regardless of any global or local definitions.
        """
        model_config_disabled = mock_model_config.copy()
        model_config_disabled['use_exogenous'] = False
        model_config_disabled['past_covariates'] = ['local_encoder_col']

        run = ExperimentRun(
            model_config=model_config_disabled,
            dataset_config=mock_dataset_config,
            validation_config=mock_validation_config,
            dataset_name="test_data"
        )
        spec = run.get_data_spec()

        assert spec.past_covariates == []
        assert spec.future_covariates == []


class TestDataProvider:
    """Tests the DataProvider class, which is responsible for preparing the run_dataset."""

    # We patch 'core.experiment_manager.TimeSeriesDataset' to intercept the creation of the *new* dataset
    @patch('core.experiment_manager.TimeSeriesDataset')
    def test_prepare_run_dataset_selects_correct_columns(self, mock_ts_constructor, mock_full_dataset):
        """
        Verifies that the DataProvider creates a new TimeSeriesDataset containing
        only the specific subset of columns requested by the DataSpec.
        """
        spec = DataSpec(
            target_columns=['target_1'],
            past_covariates=[], # Request no encoder columns for this run
            future_covariates=['global_dec_exog']
        )
        model_config = {'name': 'test_model', 'use_raw_data_source': False}
        provider = DataProvider(mock_full_dataset)

        # Action: Pass the processed series index as the override mask
        run_dataset = provider.prepare_run_dataset(spec, model_config, data_override=mock_full_dataset.series)

        # 1. Verify the correct constructor was called on the new dataset
        mock_ts_constructor.assert_called_once()
        call_kwargs = mock_ts_constructor.call_args.kwargs

        # 2. Verify the data passed to the constructor is the correct slice
        assert sorted(list(call_kwargs['data'].columns)) == ['global_dec_exog', 'target_1']
        assert call_kwargs['columns'] == ['target_1']
        assert call_kwargs['past_covariates'] == []
        assert call_kwargs['future_covariates'] == ['global_dec_exog']

        # 3. Verify the differencing state was passed through
        assert run_dataset._diff_state == mock_full_dataset._diff_state
        assert run_dataset._original_series is mock_full_dataset._original_series


    @patch('core.experiment_manager.TimeSeriesDataset')
    def test_dataprovider_provides_processed_data_by_default(self, mock_ts_constructor, mock_full_dataset):
        """
        Tests that DataProvider uses the processed `.series` by default
        (when 'use_raw_data_source' is false or absent).
        """
        spec = DataSpec(target_columns=['target_1'])
        model_config = {'name': 'test_model', 'use_raw_data_source': False}
        provider = DataProvider(mock_full_dataset)

        provider.prepare_run_dataset(spec, model_config, data_override=mock_full_dataset.series)

        # Check the 'data' argument passed to the TimeSeriesDataset constructor
        call_kwargs = mock_ts_constructor.call_args.kwargs
        passed_data_df = call_kwargs['data']

        # Check if the data came from .series (where mean is 10)
        assert passed_data_df['target_1'].mean() == pytest.approx(8.0)
        assert passed_data_df['target_1'].mean() != mock_full_dataset._original_series['target_1'].mean()

    @patch('core.experiment_manager.TimeSeriesDataset')
    def test_dataprovider_provides_raw_data_when_flag_is_true(self, mock_ts_constructor, mock_full_dataset):
        """
        Tests that DataProvider uses the raw `._original_series` when
        'use_raw_data_source' is True.
        """
        spec = DataSpec(target_columns=['target_1'])
        model_config = {'name': 'test_model', 'use_raw_data_source': True}
        provider = DataProvider(mock_full_dataset)

        # Pass the original series index as the override mask
        provider.prepare_run_dataset(spec, model_config, data_override=mock_full_dataset._original_series)

        # Check the 'data' argument passed to the TimeSeriesDataset constructor
        call_kwargs = mock_ts_constructor.call_args.kwargs
        passed_data_df = call_kwargs['data']

        # Check if the data came from ._original_series (where mean is 30)
        assert passed_data_df['target_1'].mean() == pytest.approx(30.0)
        assert passed_data_df['target_1'].mean() != mock_full_dataset.series['target_1'].mean()


    @patch('core.experiment_manager.TimeSeriesDataset')
    def test_prepare_run_dataset_with_all_columns(self, mock_ts_constructor, mock_full_dataset):
        """
        Verifies that the DataProvider correctly includes all column types when
        the DataSpec requests them all.
        """
        spec = DataSpec(
            target_columns=['target_1'],
            past_covariates=['global_enc_exog'],
            future_covariates=['global_dec_exog']
        )
        model_config = {'name': 'test_model', 'use_raw_data_source': False}
        provider = DataProvider(mock_full_dataset)

        provider.prepare_run_dataset(spec, model_config, data_override=mock_full_dataset.series)

        call_kwargs = mock_ts_constructor.call_args.kwargs

        assert sorted(list(call_kwargs['data'].columns)) == ['global_dec_exog', 'global_enc_exog', 'target_1']
        assert call_kwargs['past_covariates'] == ['global_enc_exog']
        assert call_kwargs['future_covariates'] == ['global_dec_exog']


    def test_prepare_run_dataset_raises_on_missing_columns(self, mock_full_dataset):
        """
        Verifies that the DataProvider raises a ValueError if the DataSpec
        requests a column that does not exist in the source dataset.
        """
        spec = DataSpec(
            target_columns=['target_1'],
            past_covariates=['non_existent_column']
        )
        model_config = {'name': 'test_model', 'use_raw_data_source': False}
        provider = DataProvider(mock_full_dataset)

        with pytest.raises(ValueError, match="Columns .* not available in the source data"):
            provider.prepare_run_dataset(spec, model_config, data_override=mock_full_dataset.series)