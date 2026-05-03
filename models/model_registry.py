"""Module for registering and instantiating forecasting models.

This module provides a model registry system using a decorator to register forecasting models
and a factory function to create model instances dynamically. It is used to manage models
such as ARIMA, VAR, LSTM, and Transformer forecasters in the time series forecasting framework.
"""

from typing import Callable, Dict, Type, Any, List
from models.base import NeuralTSForecaster, StatTSForecaster

model_registry: Dict[str, Type] = {}
"""Global dictionary mapping model names to their respective classes."""


def register_model(name: str, is_univariate: bool = False) -> Callable:
    """
    Decorator to register a forecasting model in the model registry.

    Args:
        name: Unique name for the model (e.g., 'arima', 'lstm_direct').
        is_univariate: Whether the model supports only univariate time series. Defaults to False.

    Returns:
        Decorator function that registers the model class.

    Raises:
        ValueError: If name is empty or already registered, or if is_univariate is not a boolean.
        TypeError: If the decorated class is not a subclass of NeuralTSForecaster or StatTSForecaster.
    """
    if not name:
        raise ValueError("Model name cannot be empty.")
    if name in model_registry:
        raise ValueError(f"Model '{name}' is already registered. Registered models: {list(model_registry.keys())}")
    if not isinstance(is_univariate, bool):
        raise ValueError("is_univariate must be a boolean.")

    def decorator(model_class: Type) -> Type:
        if not issubclass(model_class, (NeuralTSForecaster, StatTSForecaster)):
            raise TypeError("Model class must be a subclass of NeuralTSForecaster or StatTSForecaster.")
        model_class.model_name = name
        model_class.is_univariate = is_univariate
        model_registry[name] = model_class
        return model_class

    return decorator


def create_model(name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Create an instance of a registered forecasting model.

    Args:
        name: Name of the model to instantiate (e.g., 'arima', 'lstm_direct').
        *args: Positional arguments to pass to the model class constructor.
        **kwargs: Keyword arguments to pass to the model class constructor.

    Returns:
        Instance of the registered model class.

    Raises:
        ValueError: If the model name is not registered.
    """
    if not name:
        raise ValueError("Model name cannot be empty.")
    if name not in model_registry:
        raise ValueError(f"Model '{name}' is not registered. Available models: {list(model_registry.keys())}")
    return model_registry[name](*args, **kwargs)


def list_registered_models() -> List[str]:
    """
    Get a list of all registered model names.

    Returns:
        List of registered model names.
    """
    return list(model_registry.keys())
