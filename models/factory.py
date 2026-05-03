"""Module for creating forecasting model instances using a factory pattern.

This module defines the ModelFactory class, which provides a high-level interface for
instantiating registered forecasting models (e.g., ARIMA, VAR, LSTM, Transformer) by
delegating to the model registry.
"""

import logging
from typing import Any, Dict, List, TYPE_CHECKING

from models.model_registry import model_registry, list_registered_models
from utils.dataset import TimeSeriesDataset
from utils.dependencies import check_dependencies
if TYPE_CHECKING:
    from models.base import TSForecaster

if TYPE_CHECKING:
    # Prevent circular imports
    from core.context import RunContext

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating instances of registered forecasting models."""

    @staticmethod
    def get_available_models() -> List[str]:
        """Zwraca listę nazw zarejestrowanych modeli."""
        return list(model_registry.keys())

    @staticmethod
    def create(
        model_type: str,
        model_name: str,
        model_params: Dict[str, Any],
        num_features: int,
        forecast_steps: int,
        window_size: int,
        dataset: TimeSeriesDataset,
        run_context: 'RunContext',
    ) -> 'TSForecaster':
        """
        Create an instance of a registered forecasting model.

        Args:
            model_type: Type of the model to instantiate (e.g., 'arima', 'transformer').
                        This MUST correspond to a key in the model registry.
            model_name: Unique name for this model instance (e.g., 'arima_baseline', 'transformer_large').
                        Used for logging and identifying the specific configuration.
            model_params: Parameters for this model instance (e.g., ARIMA, VAR, LSTM, etc.).
            dataset: TimeSeriesDataset instance for this model instance (e.g., ARIMA).
            run_context: Execution context containing paths and metadata (Single Source of Truth).

        Returns:
            Instance of the registered model class.

        Raises:
            ValueError: If model_type is empty or not registered.
            RuntimeError: If model instantiation fails due to invalid arguments or other errors.
        """
        if not model_type:
            raise ValueError("model_type cannot be empty.")
        if not model_name:
            raise ValueError("model_name cannot be empty.")
        if run_context is None:
            raise ValueError("run_context is required to create a model.")

        if model_type not in model_registry:
            raise ValueError(
                f"Model type '{model_type}' is not registered. Available models: {list_registered_models()}"
            )
        check_dependencies([model_type])
        model_cls = model_registry[model_type]

        # --- sanity soft-checks ---

        if not hasattr(dataset, 'target_columns'):
            logger.warning("[ModelFactory] Dataset missing 'target_columns' attribute")

        dataset_num_features = len(getattr(dataset, "target_columns", []))
        if num_features != dataset_num_features:
            logger.warning(
                f"[ModelFactory] num_features mismatch: "
                f"passed={num_features}, dataset={dataset_num_features}"
            )

        context_forecast_steps = getattr(run_context, 'forecast_steps', None)
        if context_forecast_steps is not None and forecast_steps != context_forecast_steps:
            logger.warning(
                f"[ModelFactory] forecast_steps mismatch: "
                f"passed={forecast_steps}, context={context_forecast_steps}"
            )

        context_window_size = getattr(run_context, 'window_size', None)
        if context_window_size is not None and window_size != context_window_size:
            logger.warning(
                f"[ModelFactory] window_size mismatch: "
                f"passed={window_size}, context={context_window_size}"
            )

        # --- soften invalid values ---

        if forecast_steps is None or forecast_steps <= 0:
            fallback = getattr(run_context, "forecast_steps", None)
            if fallback and fallback > 0:
                logger.warning(
                    f"[ModelFactory] Invalid forecast_steps={forecast_steps}, "
                    f"fallback to {fallback}"
                )
                forecast_steps = fallback

        if window_size is None or window_size <= 0:
            fallback = getattr(run_context, "window_size", None)
            if fallback and fallback > 0:
                logger.warning(
                    f"[ModelFactory] Invalid window_size={window_size}, "
                    f"fallback to {fallback}"
                )
                window_size = fallback

        # --- creation ---

        try:
            model = model_cls(
                model_params=model_params,
                num_features=num_features,
                forecast_steps=forecast_steps,
                window_size=window_size,
                dataset=dataset,
                run_context=run_context
            )
            model.model_name = model_name
            model.model_type = model_type
            return model

        except Exception as e:
            logger.error(
                f"[ModelFactory] Failed to create model '{model_name}' "
                f"(type: {model_type}): {e}",
                exc_info=True
            )
            raise

    @staticmethod
    def list_models() -> list[str]:
        """
        Get a list of all registered model types.

        Returns:
            List of registered model types.
        """
        return list_registered_models()