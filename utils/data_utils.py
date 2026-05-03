"""
Module for preparing time series data for forecasting models.

This module provides utilities to create training data using sliding window techniques,
suitable for models like LSTM, Transformer, and Generic Transformer.
"""

import logging
from typing import Tuple, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def create_sliding_window(
        data: np.ndarray,
        window_size: int,
        forecast_steps: int,
        target_indices: Optional[List[int]] = None,
        decoder_exog_indices: Optional[List[int]] = None,
        step: int = 1,
        exog_forecast_steps: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Create training data (X, y_targets, y_decoder_exog) using a sliding window.

    Supports different horizons for targets and exogenous variables. This is crucial
    for iterative training modes where targets are 1-step (to train the autoregressive loop)
    but exogenous variables (like calendar features) are needed for the full forecast horizon
    to provide context (e.g. for Future Injection).

    Args:
        data: Time series data as a NumPy array (time, features).
        window_size: Number of input time steps (look-back window).
        forecast_steps: Number of future time steps for target predictions.
        target_indices: List of column indices for the target variables.
        decoder_exog_indices: List of column indices for future-known exogenous
            variables needed by the decoder.
        step: Step size for sliding the window.
        exog_forecast_steps: Number of time steps to retrieve for exogenous variables.
            If None, defaults to `forecast_steps`.
            Example: forecast_steps=1 (iterative training), exog_forecast_steps=30 (full context).

    Returns:
        A tuple of NumPy arrays:
            - X: Input sequences (n_samples, window_size, n_features).
            - y_targets: Target sequences (n_samples, forecast_steps, n_targets).
            - y_decoder_exog: Future exogenous sequences for the decoder
              (n_samples, exog_horizon, n_decoder_exog), or None.
              Note: The time dimension here matches `exog_forecast_steps`.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a NumPy array.")
    if data.size == 0:
        raise ValueError("data cannot be empty.")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if not isinstance(forecast_steps, int) or forecast_steps <= 0:
        raise ValueError("forecast_steps must be a positive integer.")
    if not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("data cannot contain NaN or infinite values.")

    # Validate exog_forecast_steps if provided
    if exog_forecast_steps is not None:
        if not isinstance(exog_forecast_steps, int) or exog_forecast_steps < 1:
            raise ValueError(f"exog_forecast_steps must be a positive integer, got {exog_forecast_steps}")

    # Determine horizons for targets and exog separately
    y_horizon = forecast_steps
    exog_horizon = exog_forecast_steps if exog_forecast_steps is not None else forecast_steps

    # We need enough data for the longest horizon required
    max_horizon = max(y_horizon, exog_horizon)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if len(data) < window_size + max_horizon:
        raise ValueError(
            f"Data length ({len(data)}) is insufficient for window_size "
            f"({window_size}) and required horizon ({max_horizon})."
        )

    n_samples = (len(data) - window_size - max_horizon + 1) // step

    X = np.zeros((n_samples, window_size, data.shape[1]))
    y_targets = np.zeros((n_samples, y_horizon, len(target_indices or [])))

    y_decoder_exog = np.zeros(
        (n_samples, exog_horizon, len(decoder_exog_indices or []))) if decoder_exog_indices else None

    for i in range(n_samples):
        start_idx = i * step
        X[i] = data[start_idx: start_idx + window_size]

        # Slice targets (using y_horizon)
        if target_indices:
            y_window_target = data[start_idx + window_size: start_idx + window_size + y_horizon]
            y_targets[i] = y_window_target[:, target_indices]

        # Slice exog (using exog_horizon - potentially longer!)
        if decoder_exog_indices and y_decoder_exog is not None:
            y_window_exog = data[start_idx + window_size: start_idx + window_size + exog_horizon]
            y_decoder_exog[i] = y_window_exog[:, decoder_exog_indices]

    return X, y_targets, y_decoder_exog


def infer_seasonal_period(freq: Optional[str]) -> int:
    """
    Infers a typical seasonal period from a pandas frequency string.

    Heuristics:
    - 'T' / 'min' (Minute) -> 60 (Hour cycle)
    - 'H' (Hour) -> 24 (Daily cycle)
    - 'D' (Day) -> 7 (Weekly cycle)
    - 'W' (Week) -> 52 (Yearly cycle)
    - 'M' (Month) -> 12 (Yearly cycle)
    - 'Q' (Quarter) -> 4 (Yearly cycle)

    Args:
        freq: Frequency string (e.g., 'H', 'D', 'W-TUE').

    Returns:
        int: Inferred period, or 1 if inference is not possible.
    """
    if not freq:
        return 1

    freq_upper = freq.upper()

    # The order of conditions matters (starting from the most specific).

    FREQ_PRIORITY = [
        ('T', 60),  # Minute (e.g. '15T')
        ('MIN', 60),  # Minute (e.g. '15MIN')
        ('H', 24),  # Hour
        ('D', 7),  # Day -> Weekly seasonality
        ('W', 52),  # Week -> Annual seasonality
        ('M', 12),  # Month -> Annual seasonality
        ('Q', 4)  # Quarter -> Annual seasonality
    ]
    for code, period in FREQ_PRIORITY:
        if code in freq_upper:
            return period
    logger.warning(f"Could not infer seasonal period from frequency '{freq}'. Defaulting to 1.")
    return 1