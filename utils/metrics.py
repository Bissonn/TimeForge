"""Module for calculating evaluation metrics for time series forecasting.

This module provides functions to compute standard metrics such as MAE, RMSE, SMAPE, and MASE
for evaluating the performance of forecasting models.
"""

import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> Dict[str, float]:
    """
    Calculate evaluation metrics: MAE, RMSE, SMAPE, and MASE for time series predictions.

    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.
        epsilon: Small constant to avoid division by zero in SMAPE and MASE calculations. Defaults to 1e-10.

    Returns:
        Dictionary containing the following metrics:
            - 'mae': Mean Absolute Error.
            - 'mse': Mean Squared Error.
            - 'rmse': Root Mean Squared Error.
            - 'smape': Symmetric Mean Absolute Percentage Error (in percentage).
            - 'mase': Mean Absolute Scaled Error, using a naive forecast as baseline.

    Raises:
        ValueError: If inputs have incompatible shapes, contain NaNs, or are insufficient for MASE.
    """
    # Convert inputs to NumPy arrays if they aren't already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty.")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays cannot contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Input arrays cannot contain infinite values.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    # This validation is now correctly placed outside the try block
    if y_true.size <= 1:
        raise ValueError("MASE requires at least two actual values for naive forecast calculation.")

    try:
        # The try block now only covers the calculations
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100

        naive_error = np.mean(np.abs(y_true[1:] - y_true[:-1]))
        if naive_error < epsilon:
            logger.warning(
                f"Naive error is very small ({naive_error}). "
                f"Using epsilon ({epsilon}) to avoid division by zero."
            )
            naive_error = epsilon
        mase = mae / naive_error

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "smape": float(smape),
            "mase": float(mase),
        }
    except Exception as e:
        logger.error(f"An unexpected error occurred during metric calculation: {str(e)}", exc_info=True)
        raise RuntimeError(f"Metric calculation failed unexpectedly: {str(e)}")

def _as_2d_channels_last(y: np.ndarray) -> np.ndarray:
    """Helper: Normalizes array to (Samples, Channels) shape."""
    y = np.asarray(y)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    if y.ndim == 2:
        return y
    if y.ndim == 3:
        n, h, c = y.shape
        return y.reshape(n * h, c)
    raise ValueError(f"Unsupported shape: {y.shape}")

def calculate_metrics_per_channel(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: List[str],
        channel_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculates metrics separately for each channel.
    Wraps the fixed calculate_metrics function.
    """
    yt = _as_2d_channels_last(y_true)
    yp = _as_2d_channels_last(y_pred)

    if yt.shape != yp.shape:
        logger.warning(f"Shape mismatch in per-channel calculation: {yt.shape} vs {yp.shape}")
        return {}

    n, c = yt.shape
    if len(channel_names) != c:
        # Fallback if names don't match dimensions
        channel_names = [f"ch_{i}" for i in range(c)]

    # Prepare output structure: {'mae': {}, 'rmse': {} ...}
    out = {m: {} for m in metrics}

    for j, name in enumerate(channel_names):
        ytj = yt[:, j]
        ypj = yp[:, j]

        try:
            # This returns all available metrics: {'mae': X, 'rmse': Y, 'smape': Z, ...}
            full_results = calculate_metrics(ytj, ypj)

            # Filter: only store the metrics requested in the 'metrics' list
            for m in metrics:
                if m in full_results:
                    out[m][name] = full_results[m]

        except Exception:
            # If calculation fails for a specific channel (e.g. constant value), skip it
            continue

    return out