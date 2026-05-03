# tests/utils/test_preprocessor_full_context.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from utils.preprocessor import Preprocessor

# --- Fixtures reproducing configuration from config.yaml ---

@pytest.fixture
def full_config_for_default_experiment():
    """Reproduces relevant parts of config.yaml for 'default' experiment."""
    # Simulating data file path, assuming it exists
    # In a real test, tmp_path can be used as before
    dummy_data_path = "data/national_illness.csv"

    return {
        'datasets': {
            'illness': { # 'illness' dataset configuration
                'path': dummy_data_path,
                'columns': ['ILITOTAL'],
                # No exog_columns in this definition in config.yaml
            }
        },
        'models': {
            'lstm_iterative': { # Global configuration for 'lstm_iterative' model
                'hidden_size': 128,
                'num_layers': 2,
                'optimize': True,
                'optimization': {
                    'method': 'optuna',
                    'n_trials': 20,
                    'params': {
                        'hidden_size': [64, 128, 256],
                        'num_layers': [1, 2, 3],
                        'learning_rate': {'min': 0.0001, 'max': 0.01}
                    }
                },
                'preprocessing': { # <--- Global preprocessing configuration
                    'preprocessing_groups': [{
                        'name': 'default_target_pipeline',
                        'apply_to': '__targets__',
                        'pipeline': {
                            'scaling': {
                                'enabled': True,
                                'method': 'standard'
                            }
                        }
                    }]
                }
            }
            # ... inne modele
        },
        'experiments': [{ # Definition of 'default' experiment
            'name': 'default',
            'description': 'Default.',
            'dataset': 'illness',
            'models': [
                {'name': 'lstm_iterative'}  # <--- Reference to model
                # No local 'preprocessing' configuration for the model in this experiment
            ],
            'validation_setup': {
                'forecast_steps': 30,
                'n_folds': 3,
                'window_size': 90,
                'evaluation_metric': 'mse'
            }
        }]
        # ... other experiments
    }

@pytest.fixture
def effective_model_params_default_lstm(full_config_for_default_experiment):
    """
    Simulates merging global model configuration with configuration
    of the model in 'default' experiment (no changes here).
    """
    exp_config = next(exp for exp in full_config_for_default_experiment['experiments'] if exp['name'] == 'default')
    model_ref_in_exp = next(m for m in exp_config['models'] if m['name'] == 'lstm_iterative')
    base_model_config = full_config_for_default_experiment['models']['lstm_iterative'].copy()

    # In this case model_ref_in_exp is just {'name': 'lstm_iterative'},
    # so update() won't change anything significant besides adding/overwriting 'name'.
    base_model_config.update(model_ref_in_exp)
    return base_model_config

@pytest.fixture
def mock_run_dataset_for_default_lstm(full_config_for_default_experiment):
    """
    Mocks a TimeSeriesDataset instance as it would be prepared for
    lstm_iterative in the 'default' experiment (only 'ILITOTAL').
    """
    mock_dataset = MagicMock()
    dataset_cfg = full_config_for_default_experiment['datasets']['illness']

    mock_dataset.target_columns = dataset_cfg['columns'] # ['ILITOTAL']
    mock_dataset.past_covariates = dataset_cfg.get('past_covariates', []) # []
    mock_dataset.future_covariates = dataset_cfg.get('future_covariates', []) # []
    mock_dataset.columns = mock_dataset.target_columns + mock_dataset.past_covariates + mock_dataset.future_covariates
    mock_dataset.name = "illness"
    mock_dataset.config = full_config_for_default_experiment  # Passing full conf.
    mock_dataset.freq = 'W-TUE'  # From logs

    return mock_dataset

@pytest.fixture
def sample_illness_data_for_test():
    """Provides a sample DataFrame mimicking the 'illness' data."""
    np.random.seed(42)
    data = np.linspace(1000, 70000, 100) + np.random.randn(100) * 5000
    index = pd.date_range(start='2020-01-01', periods=100, freq='W-TUE')
    return pd.DataFrame({'ILITOTAL': data}, index=index)

# --- Test Case ---

def test_preprocessor_in_default_lstm_context(
    effective_model_params_default_lstm,
    mock_run_dataset_for_default_lstm,
    sample_illness_data_for_test
):
    """
    Tests Preprocessor using the fully resolved configuration context for
    the 'default' experiment and 'lstm_iterative' model.
    """
    # --- Simulation of input data for Preprocessor.__init__ ---
    # Preprocessing configuration taken from effective model parameters
    preprocessing_config = effective_model_params_default_lstm.get('preprocessing', {})
    # Target columns from run_dataset mock
    target_cols = mock_run_dataset_for_default_lstm.target_columns
    # Exogenous columns from run_dataset mock (empty)
    exog_cols = mock_run_dataset_for_default_lstm.past_covariates + mock_run_dataset_for_default_lstm.future_covariates
    # Data for transformation
    data_to_transform = sample_illness_data_for_test

    # --- 1. Preprocessor Initialization Test ---
    print("\n--- Testing Preprocessor Initialization ---")
    print(f"Config passed to Preprocessor: {preprocessing_config}")
    print(f"Target columns: {target_cols}")
    print(f"Exog columns: {exog_cols}")
    try:
        preprocessor = Preprocessor(
            config=preprocessing_config,
            target_columns=target_cols,
            exog_columns=exog_cols
        )
    except Exception as e:
        pytest.fail(f"Preprocessor initialization failed.\nError: {e}")

    # Check if `column_pipelines` was correctly populated
    assert 'ILITOTAL' in preprocessor.column_pipelines, \
        f"Pipeline for 'ILITOTAL' not created. column_pipelines: {preprocessor.column_pipelines}"
    pipeline = preprocessor.column_pipelines['ILITOTAL']
    assert 'scaling' in pipeline, "Scaling step missing in pipeline."
    assert pipeline['scaling']['enabled'] is True, "Scaling step not enabled."
    assert pipeline['scaling']['method'] == 'standard', "Scaling method not 'standard'."
    print("Preprocessor Initialization successful.")

    # --- 2. fit_transform Test ---
    print("\n--- Testing Preprocessor fit_transform ---")
    try:
        transformed_data = preprocessor.fit_transform(data_to_transform.copy())
    except Exception as e:
         pytest.fail(f"Preprocessor fit_transform failed.\nError: {e}")

    # Check if data was scaled
    original_mean = data_to_transform['ILITOTAL'].mean()
    transformed_mean = transformed_data['ILITOTAL'].mean()
    original_std = data_to_transform['ILITOTAL'].std()
    transformed_std = transformed_data['ILITOTAL'].std(ddof=0)

    print(f"Original Mean: {original_mean:.4f}, Transformed Mean: {transformed_mean:.4f}")
    print(f"Original Std Dev: {original_std:.4f}, Transformed Std Dev: {transformed_std:.4f}")

    assert not np.isclose(original_mean, transformed_mean), "Mean did not change."
    assert transformed_mean == pytest.approx(0.0, abs=1e-6), "Mean not close to 0."
    assert transformed_std == pytest.approx(1.0, abs=1e-6), "Std Dev not close to 1."
    assert preprocessor.pipeline_states['ILITOTAL'].get('scaler') is not None, "Scaler state not saved."
    print("Preprocessor fit_transform successful.")

    # --- 3. (Optional) inverse_transform Test ---
    print("\n--- Testing Preprocessor inverse_transform ---")
    try:
         preprocessor._full_raw_data_context = data_to_transform.copy()
         inverted_data = preprocessor.inverse_transforms(transformed_data)
    except Exception as e:
         pytest.fail(f"Preprocessor inverse_transforms failed.\nError: {e}")

    # Compare inverted data with original
    pd.testing.assert_frame_equal(
         data_to_transform.loc[inverted_data.index],
         inverted_data,
         check_dtype=False,
         rtol=1e-5
    )
    print("Preprocessor inverse_transform successful.")
    print("\nTest test_preprocessor_in_default_lstm_context PASSED!")