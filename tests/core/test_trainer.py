"""
Unit tests for the ModelTrainer class.
"""
import pytest
import os
from unittest.mock import MagicMock, patch, call
from models.base import TSForecaster
from utils.dataset import TimeSeriesDataset
import pandas as pd
import numpy as np

from core.trainer import ModelTrainer

pytestmark = pytest.mark.unit

# --- Reusable Side Effect for predict() ---
def get_predict_side_effect():
    """Returns a fresh list of mock predictions with correct indices."""
    # Fold 1 prediction index (starts day 50)
    pred_index_1 = pd.date_range('2020-01-01', periods=10, freq='D') + pd.Timedelta(days=50) # 2020-02-20
    # Fold 2 prediction index (starts day 60)
    pred_index_2 = pd.date_range('2020-01-01', periods=10, freq='D') + pd.Timedelta(days=60) # 2020-03-01

    return [
        pd.DataFrame({'value': [i+0.1 for i in range(50, 60)]}, index=pred_index_1),
        pd.DataFrame({'value': [i+0.1 for i in range(60, 70)]}, index=pred_index_2)
    ]

@pytest.fixture
def mock_model_template(mock_dataset):
    """
    Mock TSForecaster template.
    """
    model = MagicMock(spec=TSForecaster)
    model.optimize_hyperparameters.return_value = ({'p': 1}, 0.5)

    model.target_columns = ['value']
    model.__class__.__name__ = "MockForecaster"
    model.model_params = {}
    model.preprocessor = MagicMock()
    model.preprocessor.target_columns = ['value']

    # This patch is *only* for the ModelTrainer constructor
    with patch('core.trainer.ModelFactory.create') as mock_create:
        mock_create.return_value = model
        yield model


@pytest.fixture
def mock_dataset():
    """
    Mock TimeSeriesDataset.
    """
    dataset = MagicMock(spec=TimeSeriesDataset)

    # Create a real DataFrame with a DatetimeIndex
    index = pd.date_range('2020-01-01', periods=100, freq='D')
    real_series = pd.DataFrame({'value': range(100)}, index=index)

    dataset.series = real_series
    dataset.target_columns = ['value']
    dataset.columns = ['value']

    # --- Mocks for Step 5 ---
    dataset._diff_state = {'d': 1, 'D': 0, 's': 1}
    dataset._original_series = real_series.copy()
    dataset._original_series['value'] = np.cumsum(real_series['value'])
    dataset.inverse_difference_forecast = MagicMock(
        side_effect=lambda df: df * 10
    )

    # Mock prepare_for_evaluation to call inverse_difference_forecast
    def mock_prepare_for_evaluation(y_true, y_pred, model_used_raw=False):
        """Mock that mimics real prepare_for_evaluation behavior."""
        if model_used_raw:
            return y_true, y_pred

        was_differenced = (
            dataset._diff_state is not None and
            (dataset._diff_state.get('d', 0) > 0 or dataset._diff_state.get('D', 0) > 0)
        )

        if not was_differenced:
            return y_true, y_pred

        # Call the mocked inverse_difference_forecast
        y_pred_transformed = dataset.inverse_difference_forecast(y_pred)

        # Get actuals from original series
        y_true_original = dataset._original_series.loc[y_true.index][dataset.target_columns]

        return y_true_original, y_pred_transformed

    dataset.prepare_for_evaluation = MagicMock(side_effect=mock_prepare_for_evaluation)
    # --- End Mocks for Step 5 ---

    # Fold 1: train=0-49, test=50-59 (Index: 2020-02-20 to 2020-02-29)
    # Fold 2: train=0-59, test=60-69 (Index: 2020-03-01 to 2020-03-10)
    fold_1_train = real_series.iloc[:50]
    fold_1_test = real_series.iloc[50:60]
    fold_2_train = real_series.iloc[:60]
    fold_2_test = real_series.iloc[60:70]

    # Store mock folds for retrieval in tests
    dataset.mock_folds = [
        (fold_1_train, fold_1_test),
        (fold_2_train, fold_2_test)
    ]

    yield dataset


@patch('utils.dataset.TimeSeriesDataset.generate_sequential_folds')
@patch('core.trainer.save_json')
def test_model_trainer_optimize(mock_save_json, mock_static_folds, mock_model_template, mock_dataset, tmp_path):
    """
    Tests that ModelTrainer.optimize correctly calls the model's HPO method
    and then saves the results to the specified run_path.
    """
    mock_static_folds.return_value = mock_dataset.mock_folds

    model_config = {'optimize': True, 'optimization': {'method': 'grid'}}
    validation_params = {'n_folds': 2, 'forecast_steps': 10, 'window_size': 50}
    run_path = str(tmp_path)

    trainer = ModelTrainer(
        model_name="mock_model",
        model_type="mock_type",  # Added model_type
        run_dataset=mock_dataset,
        model_config=model_config,
        validation_params=validation_params,
        run_path=run_path,
        experiment_name="test_exp"
    )

    mock_folds = TimeSeriesDataset.generate_sequential_folds(mock_dataset.series, 2, 10)

    best_params, best_score = trainer.optimize(folds=mock_folds)

    mock_model_template.optimize_hyperparameters.assert_called_once_with(
        dataset=mock_dataset,
        model_config=model_config,
        validation_params=validation_params,
        folds=mock_folds
    )
    assert best_score == 0.5
    assert best_params == {'p': 1}
    mock_save_json.assert_called_once_with(best_params, run_path, "best_params.json")


@patch('utils.dataset.TimeSeriesDataset.generate_sequential_folds')
@patch('core.trainer.ModelFactory.create')
@patch('core.trainer.find_latest_hpo_run_id')
@patch('core.trainer.load_json')
@patch('core.trainer.save_json')
@patch('core.trainer.calculate_metrics')
@patch('core.trainer.Visualizer.plot_predictions')
def test_model_trainer_evaluate(
    mock_visualizer, mock_calculate_metrics, mock_save_json, mock_load_json, mock_find_latest_run,
    mock_factory_create,
    mock_static_folds,
    mock_model_template, mock_dataset, tmp_path
):
    """
    Tests the full backtesting evaluation loop.
    """
    # --- Configure mocks *inside* the test ---
    mock_static_folds.return_value = mock_dataset.mock_folds
    mock_preds = get_predict_side_effect()
    mock_model_template._fit_and_evaluate_fold.side_effect = [(0.1, mock_preds[0],{}), (0.2, mock_preds[1],{})]
    mock_factory_create.return_value = mock_model_template
    # ---

    model_config = {}
    validation_params = {'n_folds': 2, 'forecast_steps': 10, 'window_size': 50, 'evaluation_metric': 'mse'}
    run_path = str(tmp_path)

    mock_find_latest_run.return_value = "fake_run_id"
    mock_load_json.return_value = {'p': 1}
    mock_calculate_metrics.return_value = {'mse': 0.1, 'mae': 0.2}

    # Updated call with model_type argument
    trainer = ModelTrainer(
        "mock_model", "mock_type", mock_dataset, model_config, validation_params,
        run_path, "test_exp", run_id_to_load=None
    )

    mock_folds = TimeSeriesDataset.generate_sequential_folds(mock_dataset.series, 2, 10)

    trainer.evaluate(folds=mock_folds, visualize=True)

    mock_find_latest_run.assert_called_once()
    mock_load_json.assert_called_once()
    assert mock_model_template._fit_and_evaluate_fold.call_count == 2

    assert mock_dataset.inverse_difference_forecast.call_count == 2

    assert mock_calculate_metrics.call_count == 2

    # We save a richer metrics dict (including mae_std, per_channel_agg, etc.).
    # The test only needs to verify:
    # - save_json is called once
    # - it writes to the correct path/filename
    # - the top-level 'mae' matches the mocked calculate_metrics output.

    mock_save_json.assert_called_once()
    args, kwargs = mock_save_json.call_args
    saved_metrics, saved_run_path, saved_filename = args

    assert saved_run_path == run_path
    assert saved_filename == "backtest_metrics.json"
    assert saved_metrics["mae"] == pytest.approx(0.2)

    assert mock_visualizer.call_count == 2

class TestInverseDifferencingLogic:

    @pytest.fixture
    def mock_trainer(self, mock_dataset, tmp_path):
        """
        Creates a ModelTrainer instance with mocked dependencies for logic testing.
        """
        validation_params = {'n_folds': 2, 'forecast_steps': 10, 'window_size': 50}
        model_config = {}

        mock_model = MagicMock(spec=TSForecaster)

        # This patch is *only* for the ModelTrainer constructor
        with patch('core.trainer.ModelFactory.create') as mock_create_in_fixture:
            mock_create_in_fixture.return_value = mock_model

            # Updated call with model_type argument
            trainer = ModelTrainer(
                "mock_model", "mock_type", mock_dataset, model_config, validation_params,
                str(tmp_path), "test_exp", run_id_to_load="fake_run_id"
            )
            trainer.model_template = mock_model
            yield trainer, mock_dataset, mock_model


    @patch('utils.dataset.TimeSeriesDataset.generate_sequential_folds')
    @patch('core.trainer.ModelFactory.create')
    @patch('core.trainer.load_json', return_value={'p': 1})
    @patch('core.trainer.calculate_metrics', return_value={'mse': 0.1})
    def test_trainer_applies_inverse_differencing_by_default(self, mock_calc, mock_load, mock_factory_create, mock_static_folds, mock_trainer):
        """
        Tests that `evaluate` CALLS inverse_difference_forecast
        when use_raw_data_source=False and _diff_state exists.
        """
        trainer, mock_dataset, mock_model = mock_trainer

        # --- Configure mocks *inside* the test ---
        mock_factory_create.return_value = mock_model
        mock_static_folds.return_value = mock_dataset.mock_folds # Use 2 folds
        mock_preds = get_predict_side_effect()
        mock_model._fit_and_evaluate_fold.side_effect = [(0.1, mock_preds[0],{}), (0.1, mock_preds[1],{})]        # ---

        mock_dataset._diff_state = {'d': 1, 'D': 0, 's': 1} # Enable differencing
        trainer.model_config = {'use_raw_data_source': False} # Default

        folds = TimeSeriesDataset.generate_sequential_folds(mock_dataset.series, 2, 10) # Request 2 folds

        trainer.evaluate(folds=folds, visualize=False)

        assert mock_dataset.inverse_difference_forecast.call_count == 2
        assert mock_calc.call_count == 2 # Use mock_calc

    @patch('utils.dataset.TimeSeriesDataset.generate_sequential_folds')
    @patch('core.trainer.ModelFactory.create')
    @patch('core.trainer.load_json', return_value={'p': 1})
    @patch('core.trainer.calculate_metrics', return_value={'mse': 0.1})
    def test_trainer_skips_inverse_differencing_for_raw_model(self, mock_calc, mock_load, mock_factory_create, mock_static_folds, mock_trainer):
        """
        Tests that `evaluate` SKIPS inverse_difference_forecast
        when use_raw_data_source=True, even if _diff_state exists.
        """
        trainer, mock_dataset, mock_model = mock_trainer

        # --- Configure mocks *inside* the test ---
        mock_factory_create.return_value = mock_model
        mock_static_folds.return_value = mock_dataset.mock_folds # Use 2 folds
        mock_preds = get_predict_side_effect()
        mock_model._fit_and_evaluate_fold.side_effect = [(0.1, mock_preds[0],{}), (0.1, mock_preds[1],{})]
        # ---

        mock_dataset._diff_state = {'d': 1, 'D': 0, 's': 1} # Enable differencing
        trainer.model_config = {'use_raw_data_source': True} # Override

        folds = TimeSeriesDataset.generate_sequential_folds(mock_dataset.series, 2, 10) # Request 2 folds

        trainer.evaluate(folds=folds, visualize=False)

        # Key assertion: inverse_difference_forecast was NOT called
        mock_dataset.inverse_difference_forecast.assert_not_called()

        assert mock_calc.call_count == 2 # Use mock_calc

    @patch('utils.dataset.TimeSeriesDataset.generate_sequential_folds')
    @patch('core.trainer.ModelFactory.create')
    @patch('core.trainer.load_json', return_value={'p': 1})
    @patch('core.trainer.calculate_metrics', return_value={'mse': 0.1})
    def test_trainer_skips_inverse_differencing_if_data_was_not_differenced(self, mock_calc, mock_load, mock_factory_create, mock_static_folds, mock_trainer):
        """
        Tests that `evaluate` SKIPS inverse_difference_forecast
        when _diff_state is None.
        """
        trainer, mock_dataset, mock_model = mock_trainer

        # --- Configure mocks *inside* the test ---
        mock_factory_create.return_value = mock_model
        mock_static_folds.return_value = mock_dataset.mock_folds # Use 2 folds
        mock_preds = get_predict_side_effect()
        mock_model._fit_and_evaluate_fold.side_effect = [(0.1, mock_preds[0],{}), (0.1, mock_preds[1],{})]
        # ---

        mock_dataset._diff_state = None # Disable differencing
        trainer.model_config = {'use_raw_data_source': False} # Default

        folds = TimeSeriesDataset.generate_sequential_folds(mock_dataset.series, 2, 10) # Request 2 folds

        trainer.evaluate(folds=folds, visualize=False)

        # Key assertion: inverse_difference_forecast was NOT called
        mock_dataset.inverse_difference_forecast.assert_not_called()

        assert mock_calc.call_count == 2 # Use mock_calc