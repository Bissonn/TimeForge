"""
Learning rate calculator for Transformer HPO.

Provides model-aware and dataset-aware learning rate heuristics with batch scaling.
"""
from typing import Dict, Any

from models.hpo_heuristics import (
    get_lr_scaling_config,
    sqrt_lr_scale,
    clamp_lr_by_dataset,
)


class LearningRateCalculator:
    """
    Calculates optimal learning rates for Transformer models.

    Implements a multi-stage LR calculation pipeline:
    1. Transformer-specific base LR (conservative for time series)
    2. Model-aware scaling (width/depth/strategy adjustments)
    3. Batch-size scaling (sqrt or linear)
    4. Dataset-size ceiling clamp (hard safety constraint)
    """

    def __init__(self, model_params: Dict[str, Any]):
        """
        Initialize calculator with model configuration.

        Args:
            model_params: Model configuration dictionary
        """
        self.model_params = model_params

    def calculate_lr(
        self,
        hidden_size: int,
        num_layers: int,
        strategy: str,
        dataset_size: str,
        batch_size: int = 32
    ) -> float:
        """
        Calculate optimal learning rate with model-aware heuristics.

        Order of operations:
        1. Transformer override (3e-4 default for time series)
        2. Model/data aware scaling (width/depth/strategy)
        3. Batch-size scaling (soft heuristic)
        4. Dataset-size LR ceiling clamp (hard safety constraint)

        Args:
            hidden_size: Model hidden dimension
            num_layers: Total number of transformer layers
            strategy: Prediction strategy ("direct" or "iterative")
            dataset_size: Size category ("small", "medium", "large", "very_large")
            batch_size: Training batch size (default: 32)

        Returns:
            Calculated learning rate
        """
        # Get base LR and scaling configuration
        mode, ref_batch, lr0 = get_lr_scaling_config(self.model_params)

        # Transformer-specific conservative base
        # If user didn't explicitly config HPO scaling, override generic 1e-3 with 3e-4.
        # This fixes the regression on medium datasets while keeping architecture clean.
        if "hpo_lr_scaling" not in self.model_params:
            lr0 = 3e-4  # Conservative start for Time Series

        # Calculate model-aware scale factor
        scale = self._calculate_scale_factor(
            hidden_size, num_layers, strategy, dataset_size
        )

        # Apply scale to base LR
        lr = lr0 * scale

        # Soft batch scaling (sqrt/linear controlled by config)
        lr = sqrt_lr_scale(lr, batch_size, ref_batch, mode=mode)

        # [HARD CLAMP] Batch scaling must NOT exceed the dataset-size ceiling
        # This ensures monotonicity w.r.t. overfitting risk regardless of batch size
        lr = clamp_lr_by_dataset(lr, dataset_size)

        return lr

    def _calculate_scale_factor(
        self,
        hidden_size: int,
        num_layers: int,
        strategy: str,
        dataset_size: str
    ) -> float:
        """
        Calculate model-aware scale factor for learning rate.

        Applies penalties for:
        - Wider models (higher hidden_size)
        - Deeper models (more layers)
        - Iterative strategy
        - Small datasets

        Applies bonuses for:
        - Large datasets

        Args:
            hidden_size: Model hidden dimension
            num_layers: Total number of layers
            strategy: Prediction strategy
            dataset_size: Size category

        Returns:
            Scale factor to multiply with base LR
        """
        scale = 1.0

        # Width penalties (Balanced)
        if hidden_size >= 256:
            scale *= 0.6
        if hidden_size >= 512:
            scale *= 0.4

        # Depth penalties (prevent early overfitting)
        # Gentle start (0.9) preventing early overfitting, stronger later
        if num_layers >= 2:
            scale *= 0.9
        if num_layers >= 4:
            scale *= 0.7
        if num_layers >= 6:
            scale *= 0.7

        # Strategy sensitivity
        if strategy == "iterative":
            scale *= 0.8

        # Dataset size effect (Soft adjustment)
        if dataset_size == "small":
            scale *= 0.7
        elif dataset_size in ["large", "very_large"]:
            scale *= 1.2

        return scale
