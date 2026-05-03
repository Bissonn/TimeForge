"""Module for logging training-related events in the forecasting framework.

This module provides functions to log the start and completion of model training,
hyperparameter optimization results, and trial failures, ensuring consistent logging
across models like ARIMA, VAR, LSTM, and Transformer.
"""

import logging
from typing import Dict, Union, Any

from models.base import NeuralTSForecaster, StatTSForecaster

logger = logging.getLogger(__name__)


def log_training_start(model_name: str, model_object: Union[NeuralTSForecaster, StatTSForecaster]) -> None:
    """
    Log the start of model training.

    Args:
        model_name: Name of the model (e.g., 'arima', 'lstm_direct').
        model_object: Instance of the model being trained.

    Raises:
        ValueError: If model_name is empty or model_object is invalid.
    """
    if not model_name:
        raise ValueError("model_name cannot be empty.")

    logger.info(f"[{model_name}] Starting training with model: {type(model_object).__name__}")


def log_training_success(model_name: str, val_loss: float, best_epoch: int) -> None:
    """
    Log the successful completion of model training.

    Args:
        model_name: Name of the model (e.g., 'arima', 'lstm_direct').
        val_loss: Best validation loss achieved during training.
        best_epoch: Epoch at which the best validation loss was achieved.

    Raises:
        ValueError: If model_name is empty, val_loss is invalid, or best_epoch is negative.
    """
    if not model_name:
        raise ValueError("model_name cannot be empty.")
    if not isinstance(val_loss, (int, float)) or val_loss < 0:
        raise ValueError("val_loss must be a non-negative number.")
    if not isinstance(best_epoch, int) or best_epoch < 0:
        raise ValueError("best_epoch must be a non-negative integer.")

    logger.info(f"[{model_name}] Training completed. Best val_loss: {float(val_loss):.6f}, at epoch {best_epoch}")


def log_trial_failure(model_name: str, hyperparameters: Dict, exception: Exception) -> None:
    """
    Log a failed hyperparameter optimization trial.

    Args:
        model_name: Name of the model (e.g., 'arima', 'lstm_direct').
        hyperparameters: Dictionary of hyperparameters used in the trial.
        exception: Exception that caused the trial to fail.

    Raises:
        ValueError: If model_name is empty or hyperparameters is not a dictionary.
    """
    if not model_name:
        raise ValueError("model_name cannot be empty.")
    if not isinstance(hyperparameters, Dict):
        raise ValueError("hyperparameters must be a dictionary.")

    logger.warning(
        f"[{model_name}] Trial failed with hyperparameters={hyperparameters}: {str(exception)}",
        exc_info=True
    )


def log_best_hyperparams(model_name: str, method: str, hyperparameters: Dict, loss: float) -> None:
    """
    Log the best hyperparameters found during optimization.

    Args:
        model_name: Name of the model (e.g., 'arima', 'lstm_direct').
        method: Optimization method used (e.g., 'optuna', 'auto_arima').
        hyperparameters: Dictionary of best hyperparameters.
        loss: Best loss achieved with the hyperparameters.

    Raises:
        ValueError: If model_name or method is empty, hyperparameters is not a dictionary,
            or loss is invalid.
    """
    if not model_name:
        raise ValueError("model_name cannot be empty.")
    if not method:
        raise ValueError("method cannot be empty.")
    if not isinstance(hyperparameters, Dict):
        raise ValueError("hyperparameters must be a dictionary.")
    if not isinstance(loss, (int, float)) or loss < 0:
        raise ValueError("loss must be a non-negative number.")

    logger.info(f"[{model_name}] Best hyperparameters ({method}): {hyperparameters}, Best loss: {float(loss):.6f}")
