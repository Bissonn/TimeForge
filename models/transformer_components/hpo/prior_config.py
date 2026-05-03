"""Configuration dataclass for smart prior generation."""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PriorGenerationConfig:
    """
    Configuration for prior generation.

    This dataclass encapsulates all the context needed to generate
    smart priors for Transformer hyperparameter optimization.

    Attributes:
        num_features: Number of features being modeled
        forecast_steps: Forecast horizon length
        window_size: Historical window size
        param_space: Parameters being optimized
        fixed_params: Fixed parameters from config
        dataset_info: Dataset metadata (freq, size_category, seasonal_period, etc.)
    """
    num_features: int
    forecast_steps: int
    window_size: int
    param_space: Dict[str, Any]
    fixed_params: Dict[str, Any]
    dataset_info: Dict[str, Any]
