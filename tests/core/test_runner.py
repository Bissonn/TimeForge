"""
Unit tests for the ExperimentRunner class.
"""
import pytest
from unittest.mock import MagicMock, patch, ANY
from argparse import Namespace

from core.runner import ExperimentRunner
from utils.dataset import TimeSeriesDataset
import pandas as pd

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_configs():
    """Provides mock global and experiment configurations."""
    global_config = {
        'datasets': {
            'my_data': {'path': 'dummy.csv', 'columns': ['value']}
        },
        'models': {
            # Added 'type' as required by new validation logic
            'test_model': {'param': 1, 'type': 'test_type'}
        }
    }
    experiment_config = {
        'name': 'test_experiment',
        'dataset': 'my_data',
        'models': [{'name': 'test_model', 'param_exp': 2}], # Has experiment override
        'validation_setup': {
            'forecast_steps': 10,
            'window_size': 20,
            'n_folds': 3
        }
    }
    return global_config, experiment_config


@pytest.fixture
def mock_args():
    """Provides mock command-line arguments."""
    return Namespace(
        config_path='config.yaml',
        optimize=False,
        evaluate=True,
        train_final=False,
        run_id=None,
        no_visualization=False,
        force_defaults=False
    )

@patch('core.runner.ModelTrainer')
@patch('core.runner.TimeSeriesDataset')
@patch('core.runner.DataProvider')
@patch('core.runner.get_model_config')
@patch('core.runner.am') # Mock the artifact manager
def test_runner_calls_evaluate_by_default(
    mock_am, mock_get_model_config, mock_data_provider_cls, mock_ts_dataset_cls, mock_model_trainer_cls,
    mock_configs, mock_args
):
    """Tests that the runner calls ModelTrainer.evaluate() when no mode is specified."""
    global_config, experiment_config = mock_configs

    # Mock instances returned by the classes
    mock_full_dataset = MagicMock(spec=TimeSeriesDataset)
    mock_full_dataset.series = pd.DataFrame({'value': range(100)}) # Mock series data
    mock_ts_dataset_cls.return_value = mock_full_dataset

    mock_get_model_config.return_value = global_config['models']['test_model']

    mock_run_dataset = MagicMock()
    mock_data_provider_instance = MagicMock()
    mock_data_provider_instance.prepare_run_dataset.return_value = mock_run_dataset
    mock_data_provider_cls.return_value = mock_data_provider_instance

    mock_trainer_instance = MagicMock()
    mock_model_trainer_cls.return_value = mock_trainer_instance

    mock_am.get_run_id.return_value = "eval_run_id"
    mock_am.get_run_path.return_value = "/fake/path/test_model_eval_run_id"

    # --- Action ---
    # Args are set to default (evaluate=True)
    runner = ExperimentRunner(experiment_config, global_config, mock_args)
    runner.run()

    # --- Assertions ---
    # 1. Data pipeline is correctly called
    mock_ts_dataset_cls.assert_called_once_with('my_data', global_config, num_features=1)
    assert mock_data_provider_instance.prepare_run_dataset.call_count == 1

    # 2. Correct dataset and parameters are passed to ModelTrainer
    # We use kwargs because Runner instantiates ModelTrainer with keyword arguments
    mock_model_trainer_cls.assert_called_once()
    call_kwargs = mock_model_trainer_cls.call_args.kwargs

    assert call_kwargs['run_dataset'] is mock_run_dataset # run_dataset (full)
    assert call_kwargs['model_name'] == 'test_model'
    assert call_kwargs['model_type'] == 'test_type' # New check

    # 3. Model config includes experiment-level overrides
    # Note: 'type' is preserved in base_model_config but not necessarily in the final dict passed if not explicitly checked,
    # but here we check the merged config passed to trainer.
    expected_model_config = {'param': 1, 'name': 'test_model', 'param_exp': 2, 'type': 'test_type'}
    assert call_kwargs['model_config'] == expected_model_config

    # 4. ModelTrainer evaluate is called
    mock_trainer_instance.optimize.assert_not_called()
    mock_trainer_instance.evaluate.assert_called_once_with(folds=ANY, visualize=True)


@patch('core.runner.ModelTrainer')
@patch('core.runner.TimeSeriesDataset')
@patch('core.runner.DataProvider')
@patch('core.runner.get_model_config')
@patch('core.runner.am') # Mock the artifact manager
def test_runner_calls_optimize(
    mock_am, mock_get_model_config, mock_data_provider_cls, mock_ts_dataset_cls, mock_model_trainer_cls,
    mock_configs, mock_args
):
    """Tests that the runner calls ModelTrainer.optimize() when --optimize is true."""
    global_config, experiment_config = mock_configs
    global_config['models']['test_model']['optimize'] = True
    mock_args.optimize = True
    mock_args.evaluate = False # This would be set by argparse

    mock_full_dataset = MagicMock(spec=TimeSeriesDataset)
    mock_full_dataset.series = pd.DataFrame({'value': range(100)})
    mock_ts_dataset_cls.return_value = mock_full_dataset

    mock_get_model_config.return_value = global_config['models']['test_model']

    mock_run_dataset = MagicMock()
    mock_data_provider_instance = MagicMock()
    mock_data_provider_instance.prepare_run_dataset.return_value = mock_run_dataset
    mock_data_provider_cls.return_value = mock_data_provider_instance

    mock_trainer_instance = MagicMock()
    mock_model_trainer_cls.return_value = mock_trainer_instance

    runner = ExperimentRunner(experiment_config, global_config, mock_args)
    runner.run()

    assert mock_data_provider_instance.prepare_run_dataset.call_count == 1

    # 1. Correct parameters passed to ModelTrainer
    mock_model_trainer_cls.assert_called_once()
    call_kwargs = mock_model_trainer_cls.call_args.kwargs

    assert call_kwargs['run_dataset'] is mock_run_dataset
    assert call_kwargs['model_name'] == 'test_model'
    assert call_kwargs['model_type'] == 'test_type'

    # 2. ModelTrainer is initialized and its `optimize` method is called
    mock_trainer_instance.optimize.assert_called_once()
    mock_trainer_instance.evaluate.assert_not_called()