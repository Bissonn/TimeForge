"""Module for logging training-related events in the forecasting framework.

This module provides functions to log the start and completion of model training,
hyperparameter optimization results, and trial failures, ensuring consistent logging
across models like ARIMA, VAR, LSTM, and Transformer.
"""
import pytest
import logging
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

from utils.logging_utils import (
    log_training_start,
    log_training_success,
    log_trial_failure,
    log_best_hyperparams,
)
# Mock the classes that the logging functions expect as type hints
from models.base import NeuralTSForecaster, StatTSForecaster
from utils.dataset import TimeSeriesDataset


# --- Test Setup (Mocks) ---

def _get_mock_dataset():
    mock_ds = MagicMock(spec=TimeSeriesDataset)
    mock_ds.target_columns = ['target']
    mock_ds.past_covariates = []
    mock_ds.future_covariates = []
    mock_ds.columns = mock_ds.target_columns + mock_ds.past_covariates + mock_ds.future_covariates
    return mock_ds

# Create simple mock classes that inherit from the required types.
# They don't need any logic; they exist solely for type validation in the tests.
class MockNeuralForecaster(NeuralTSForecaster):
    def __init__(self, run_context=None):
        # Pass run_context to super().__init__ as it is now mandatory
        super().__init__(
            model_params={},
            num_features=1,
            forecast_steps=1,
            window_size=10,
            dataset=_get_mock_dataset(),
            run_context=run_context
        )
        self.model_name = "MockNeuralForecaster"

    def _fit_and_evaluate_fold(self, *args, **kwargs): pass
    def fit(self, *args, **kwargs): return 0.0, {}
    def predict(self, *args, **kwargs): pass
    def _train_model(self, *args, **kwargs): pass
    def _has_decoder(self): return False


class MockStatForecaster(StatTSForecaster):
    def __init__(self, run_context=None):
        # Pass run_context to super().__init__
        super().__init__(
            model_params={},
            num_features=1,
            forecast_steps=1,
            window_size=10,
            dataset=_get_mock_dataset(),
            run_context=run_context
        )
        self.model_name = "MockStatForecaster"

    def _fit_and_evaluate_fold(self, *args, **kwargs): pass
    def fit(self, *args, **kwargs): return 0.0, {}
    def predict(self, *args, **kwargs): pass
    def generate_predictions(self, dataset): return self.predict()
    def get_valid_params(self): return {}


# --- Tests for `log_training_start` ---

def test_log_training_start_success(caplog, base_context):
    """Tests that the function correctly logs the start of a training process."""
    # Inject base_context fixture
    model_mock = MockNeuralForecaster(run_context=base_context)

    with caplog.at_level(logging.INFO):
        log_training_start("test_model", model_mock)

    assert "test_model" in caplog.text
    assert "Starting training" in caplog.text or "Training model" in caplog.text

def test_log_training_start_empty_model_name_fails(base_context):
    """Tests that the function raises a ValueError for an empty model name."""
    # Inject base_context fixture
    model_mock = MockStatForecaster(run_context=base_context)

    with pytest.raises(ValueError, match="model_name cannot be empty"):
        log_training_start("", model_mock)

# --- Tests for `log_training_success` ---

def test_log_training_success(caplog):
    """Tests that the function correctly logs a successful training completion."""
    with caplog.at_level(logging.INFO):
        log_training_success("test_model", 0.12345, 10)
    assert "test_model" in caplog.text
    assert "0.12345" in caplog.text

@pytest.mark.parametrize("args, error_msg", [
    (("", 0.1, 10), "model_name cannot be empty."),
    (("test", -0.1, 10), "val_loss must be a non-negative number."),
    (("test", 0.1, -1), "best_epoch must be a non-negative integer."),
    (("test", 0.1, 1.5), "best_epoch must be a non-negative integer."),
])
def test_log_training_success_invalid_inputs_fail(args, error_msg):
    """Tests that the function raises ValueErrors for various invalid inputs."""
    with pytest.raises(ValueError, match=error_msg):
        log_training_success(*args)

# --- Tests for `log_trial_failure` ---

def test_log_trial_failure(caplog):
    """Tests that the function correctly logs a failed optimization trial."""
    params = {'lr': 0.1}
    exception = ValueError("Test error")
    with caplog.at_level(logging.WARNING):
        # Removed extra arg '1' (trial_number) to match signature (model_name, params, error)
        log_trial_failure("test_model", params, exception)

    assert "test_model" in caplog.text
    assert "Trial failed" in caplog.text or "Hyperparameter optimization trial failed" in caplog.text
    assert "Test error" in caplog.text

@pytest.mark.parametrize("args, error_msg", [
    (("", {'lr': 0.1}, ValueError()), "model_name cannot be empty."),
    (("test", "not_a_dict", ValueError()), "hyperparameters must be a dictionary."),
])
def test_log_trial_failure_invalid_inputs_fail(args, error_msg):
    """Tests that the function raises ValueErrors for various invalid inputs."""
    with pytest.raises(ValueError, match=error_msg):
        log_trial_failure(*args)

# --- Tests for `log_best_hyperparams` ---

def test_log_best_hyperparams(caplog):
    """Tests that the function correctly logs the best found hyperparameters."""
    params = {'lr': 0.01, 'layers': 2}
    with caplog.at_level(logging.INFO):
        # Swapped args order to match signature (model_name, method, params, loss)
        log_best_hyperparams("test_model", "optuna", params, 0.05)

    assert "test_model" in caplog.text
    assert "optuna" in caplog.text
    assert "0.05" in caplog.text

@pytest.mark.parametrize("args, error_msg", [
    (("", "optuna", {}, 0.1), "model_name cannot be empty."),
    (("test", "", {}, 0.1), "method cannot be empty."),
    (("test", "optuna", "not_a_dict", 0.1), "hyperparameters must be a dictionary."),
    (("test", "optuna", {}, -0.1), "loss must be a non-negative number."),
])
def test_log_best_hyperparams_invalid_inputs_fail(args, error_msg):
    """Tests that the function raises ValueErrors for various invalid inputs."""
    with pytest.raises(ValueError, match=error_msg):
        log_best_hyperparams(*args)
