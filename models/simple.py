"""Module for Simple Seasonal Baseline forecaster.

This model implements a naive seasonal baseline that repeats the last observed
cycle of data into the future. It serves as a benchmark for evaluating more
complex models.
"""

import logging
from typing import Dict, Any, Optional, Set, List

import numpy as np
import pandas as pd

from core.context import RunContext
from models.base import StatTSForecaster
from models.model_registry import register_model
from utils.dataset import TimeSeriesDataset
from utils.data_utils import infer_seasonal_period

logger = logging.getLogger(__name__)

@register_model("simple_seasonal", is_univariate=False)
class SimpleSeasonalForecaster(StatTSForecaster):
    """
    A simple baseline model that forecasts by repeating the last observed season.

    Logic:
        Forecast[t+h] = Actual[t+h - k*seasonal_period]

    Where k is an integer such that the index falls back into the last observed window.
    Basically, it tiles the last `seasonal_period` values to fill the forecast horizon.

    Attributes:
        seasonal_period (int): The length of the seasonal cycle to repeat.
    """

    def __init__(
            self,
            model_params: Dict[str, Any],
            num_features: int,
            forecast_steps: int,
            window_size: int,
            dataset: TimeSeriesDataset,
            run_context: RunContext,
            **kwargs
    ) -> None:
        """
        Initialize the Simple Seasonal Forecaster.

        Args:
            model_params: Dictionary must contain 'seasonal_period'.
            num_features: Number of features (supports multivariate).
            forecast_steps: Horizon length.
            window_size: Lookback window (used only for validation constraints).
            dataset: The dataset object.
            run_context: RunContext object.
            kwargs: Additional keyword arguments.
        """
        if "seasonal_period" not in model_params:
            model_params["seasonal_period"] = infer_seasonal_period(dataset.freq)
        super().__init__(
            model_params=model_params,
            num_features=num_features,
            forecast_steps=forecast_steps,
            window_size=window_size,
            dataset=dataset,
            run_context=run_context,
            **kwargs)

        self.seasonal_period = model_params.get("seasonal_period", 1)
        if not isinstance(self.seasonal_period, int) or self.seasonal_period < 1:
            raise ValueError("Parameter 'seasonal_period' must be a positive integer.")

        self.last_season_buffer: Optional[pd.DataFrame] = None
        self.target_columns: Optional[List[str]] = None

        logger.info(f"Initialized SimpleSeasonalForecaster with seasonal_period={self.seasonal_period}")

    def fit(
        self,
        train_series: pd.DataFrame,
        exog_series: Optional[pd.DataFrame] = None,
        dataset: Optional[TimeSeriesDataset] = None,
        is_final_fit: bool = False,
        **kwargs
    ) -> tuple:
        """
        Fits the model by storing the last 'seasonal_period' observations of the target series.

        Args:
            train_series: Training data (targets).
            exog_series: Exogenous variables (ignored by this model).
            dataset: Dataset object (ignored by this model, for signature compatibility).
            is_final_fit: Whether this is final fit (ignored by this model).
            **kwargs: Additional arguments (ignored, for signature compatibility).

        Returns:
            Tuple of (validation_loss, training_history). For statistical models, returns (0.0, {}).
        """
        if len(train_series) < self.seasonal_period:
            raise ValueError(
                f"Training series length ({len(train_series)}) is shorter than "
                f"seasonal_period ({self.seasonal_period}). Cannot establish baseline."
            )

        self.target_columns = train_series.columns.tolist()

        # Store the last cycle (the "season" to repeat)
        self.last_season_buffer = train_series.iloc[-self.seasonal_period:].copy()

        self.fitted = True
        self.last_fit_timestamp = train_series.index[-1]

        logger.info("SimpleSeasonalForecaster fitted (stored last season buffer).")

        return 0.0, {}  # Statistical models don't have validation loss or training history

    def predict(self, future_exog: Optional[pd.DataFrame] = None, forecast_steps: Optional[int] = None) -> pd.DataFrame:
        """
        Generates forecasts by repeating the stored seasonal buffer.

        Args:
            future_exog: Ignored.
            forecast_steps: Number of steps to forecast.

        Returns:
            pd.DataFrame: The forecast values with proper index.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting.")

        steps = forecast_steps if forecast_steps is not None else self.forecast_steps

        # 1. Generate values by tiling the buffer
        # Buffer shape: (seasonal_period, features)
        buffer_values = self.last_season_buffer.values

        # Calculate how many times we need to repeat the buffer to cover the horizon
        num_repeats = (steps // self.seasonal_period) + 1

        # Tile vertically (along time axis)
        tiled_values = np.tile(buffer_values, (num_repeats, 1))

        # Slice to the exact requested horizon length
        pred_values = tiled_values[:steps]

        # 2. Generate Prediction Index
        last_ts = self.last_fit_timestamp

        # Try to infer frequency from the buffer index
        freq = getattr(self.last_season_buffer.index, 'freq', None)
        if freq is None:
            try:
                freq = pd.infer_freq(self.last_season_buffer.index)
            except (TypeError, ValueError):
                # TypeError: index is not datetime-like (e.g. RangeIndex)
                # ValueError: too few values to infer freq
                freq = None

        if freq is not None:
            # If we have a frequency, generate a DatetimeIndex
            pred_index = pd.date_range(start=last_ts, periods=steps + 1, freq=freq)[1:]
        else:
            # Fallback strategies for non-standard indices
            try:
                # Try to infer step size for numeric/other indices
                step_delta = self.last_season_buffer.index[-1] - self.last_season_buffer.index[-2]
                start_val = self.last_season_buffer.index[-1] + step_delta
                # Create generic index by adding steps
                # Note: This simple addition works for RangeIndex and some numeric types
                pred_index = np.array([start_val + i * step_delta for i in range(steps)])
            except Exception:
                logger.warning("Could not infer frequency/step for prediction index. Returning RangeIndex.")
                pred_index = pd.RangeIndex(start=0, stop=steps)

        return pd.DataFrame(pred_values, index=pred_index, columns=self.target_columns)

    def get_valid_params(self) -> Set[str]:
        """Returns allowed configuration parameters."""
        return {"seasonal_period"}