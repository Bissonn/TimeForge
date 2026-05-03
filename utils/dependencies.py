"""Module for checking dependencies required by the forecasting framework.

This module provides a function to verify the availability of external libraries needed for
specific forecasting models or features, raising informative errors if dependencies are missing.
"""

import importlib.util
import logging
from typing import List, Optional, Dict, Tuple

from models.model_registry import list_registered_models

logger = logging.getLogger(__name__)

# Mapping of models to their required libraries
MODEL_DEPENDENCIES: Dict[str, List[Tuple[str, str, str, str]]] = {
    "arima": [
        ("statsmodels", "statsmodels", "pip install statsmodels", "ARIMA implementation from statsmodels"),
        ("numpy", "NumPy", "pip install numpy", "numerical computations"),
        ("pandas", "pandas", "pip install pandas", "data handling"),
    ],
    "sarima": [
        ("statsmodels", "statsmodels", "pip install statsmodels", "SARIMA implementation from statsmodels"),
        ("numpy", "NumPy", "pip install numpy", "numerical computations"),
        ("pandas", "pandas", "pip install pandas", "data handling"),
    ],
    "var": [
        ("statsmodels", "statsmodels", "pip install statsmodels", "VAR implementation from statsmodels"),
        ("numpy", "NumPy", "pip install numpy", "numerical computations"),
        ("pandas", "pandas", "pip install pandas", "data handling"),
    ],
    "lstm": [
        ("torch", "PyTorch", "pip install torch", "neural network computations in LSTMDirectForecaster"),
        ("numpy", "NumPy", "pip install numpy", "numerical computations"),
        ("pandas", "pandas", "pip install pandas", "data handling"),
    ],
    "transformer": [
        ("torch", "PyTorch", "pip install torch", "neural network computations in TransformerForecaster"),
        ("numpy", "NumPy", "pip install numpy", "numerical computations"),
        ("pandas", "pandas", "pip install pandas", "data handling"),
    ],
    "generic_transformer": [
        ("torch", "PyTorch", "pip install torch", "neural network computations in GenericTransformerForecaster"),
        ("numpy", "NumPy", "pip install numpy", "numerical computations"),
        ("pandas", "pandas", "pip install pandas", "data handling"),
    ],
}

# Optional dependencies for specific features
OPTIONAL_DEPENDENCIES: List[Tuple[str, str, str, str]] = [
    ("optuna", "Optuna", "pip install optuna", "hyperparameter optimization"),
    ("matplotlib", "Matplotlib", "pip install matplotlib", "plotting in Visualizer"),
]


def check_dependencies(model_names: Optional[List[str]] = None, package_manager: str = "pip") -> None:
    """
    Check if required libraries are installed for specified models or features.

    Args:
        model_names: List of model names to check dependencies for. If None, checks all registered models
            and optional dependencies. Defaults to None.
        package_manager: Package manager for installation instructions ('pip' or 'conda'). Defaults to 'pip'.

    Raises:
        ValueError: If model_names contains unregistered models or package_manager is invalid.
        ImportError: If required libraries are missing, with instructions for installation.
    """
    if model_names is None:
        model_names = list_registered_models()
    else:
        registered_models = list_registered_models()
        invalid_models = [name for name in model_names if name not in registered_models]
        if invalid_models:
            raise ValueError(
                f"Invalid model names: {invalid_models}. Available models: {registered_models}"
            )

    if package_manager not in {"pip", "conda"}:
        raise ValueError("package_manager must be 'pip' or 'conda'.")

    missing_libraries = []
    checked_libraries = set()

    # Check model-specific dependencies
    for model_name in model_names:
        for module_name, lib_name, install_cmd, usage in MODEL_DEPENDENCIES.get(model_name, []):
            if module_name not in checked_libraries and importlib.util.find_spec(module_name) is None:
                install_cmd = install_cmd if package_manager == "pip" else f"conda install {module_name}"
                missing_libraries.append((lib_name, install_cmd, f"{usage} in {model_name}"))
                checked_libraries.add(module_name)

    # Check optional dependencies if no specific models are provided
    if model_names == list_registered_models():
        for module_name, lib_name, install_cmd, usage in OPTIONAL_DEPENDENCIES:
            if module_name not in checked_libraries and importlib.util.find_spec(module_name) is None:
                install_cmd = install_cmd if package_manager == "pip" else f"conda install {module_name}"
                missing_libraries.append((lib_name, install_cmd, usage))
                checked_libraries.add(module_name)

    if missing_libraries:
        error_message = "Missing required libraries:\n"
        for lib_name, install_cmd, usage in missing_libraries:
            error_message += f"- {lib_name}: Used for {usage}. Install with: {install_cmd}\n"
        logger.error(error_message)
        raise ImportError(error_message)

    logger.info(f"All required libraries for models {model_names} are installed.")
