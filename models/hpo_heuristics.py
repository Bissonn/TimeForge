# models/hpo_heuristics.py

"""
Shared hyperparameter optimization heuristics for time-series forecasting models.

This module provides common utilities for:
- Dataset size categorization
- Learning rate scaling with batch size (Goyal et al.)
- Parameter clamping
- Common validation logic (complexity thresholds)

Used by: TransformerForecaster, LSTMForecaster.
"""

import math
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def categorize_dataset_size(training_length: int) -> str:
    """
    Categorize dataset size for capacity-aware HPO.

    Args:
        training_length: Number of training samples

    Returns:
        Size category: "small", "medium", "large", or "very_large"

    Thresholds:
        small: < 1000 samples
        medium: 1000-5000 samples
        large: 5000-20000 samples
        very_large: >= 20000 samples
    """
    if training_length < 1000:
        return "small"
    elif training_length < 5000:
        return "medium"
    elif training_length < 20000:
        return "large"
    else:
        return "very_large"


def sqrt_lr_scale(
        lr0: float,
        batch_size: int,
        ref_batch: int = 64,
        mode: str = "sqrt"
) -> float:
    """
    Scale learning rate proportionally to batch size.

    Sqrt scaling is safer and generalizes better than linear scaling,
    especially for time-series models with dropout and early stopping.

    Args:
        lr0: Base learning rate at reference batch size
        batch_size: Target batch size
        ref_batch: Reference batch size (default: 64)
        mode: "sqrt" (safe, default) or "linear" (aggressive)

    Returns:
        Scaled learning rate

    References:
        - Goyal et al. (2017): "Accurate, Large Minibatch SGD"
        - Sqrt scaling recommended for time-series forecasting
    """
    if mode == "linear":
        # Linear scaling: lr ∝ batch_size
        # More aggressive, can be unstable without warmup/scheduler
        return lr0 * (batch_size / ref_batch)
    else:  # sqrt (default)
        # Sqrt scaling: lr ∝ sqrt(batch_size)
        # Safer default, better generalization
        return lr0 * ((batch_size / ref_batch) ** 0.5)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi] range."""
    return max(lo, min(value, hi))


def get_lr_scaling_config(model_params: Dict[str, Any]) -> Tuple[str, int, float]:
    """
    Extract LR scaling configuration.
    Keeps 1e-3 as global default (aggressive), relying on model-specific overrides
    and clamping for safety.
    """
    lr_config = model_params.get("hpo_lr_scaling", {})
    mode = lr_config.get("mode", "sqrt")
    ref_batch = lr_config.get("ref_batch", 64)
    lr0 = lr_config.get("lr0", 1e-3)
    return mode, ref_batch, lr0


def get_complexity_threshold(
        model_params: Dict[str, Any],
        size_category: str,
        model_type: str = "transformer",
        num_features: int = 1
) -> int:
    constraints = model_params.get("hpo_constraints", {})

    if model_type == "lstm":
        default_map = {"small": 512, "medium": 1024, "large": 2048, "very_large": 4096}
    else:  # transformer
        default_map = {"small": 6144, "medium": 12288, "large": 24576, "very_large": 49152}

    key = f"max_complexity_{size_category}"
    base_threshold = int(constraints.get(key, default_map.get(size_category, default_map["medium"])))

    feature_scale = 1.0 + 0.05 * min(num_features, 20)
    return int(base_threshold * feature_scale)

# ═══════════════════════════════════════════════════════════════════════════
# HARD CONSTRAINTS (Safety Ceiling)
# ═══════════════════════════════════════════════════════════════════════════

def get_max_lr_by_dataset(size_category: str) -> float:
    """Return a hard upper bound for LR by dataset size category."""
    max_lr_map = {
        "small": 3e-4,      # Very conservative (<1000 samples)
        "medium": 1e-3,     # Moderate (1000-5000)
        "large": 3e-3,      # Aggressive
        "very_large": 5e-3, # Very aggressive
    }
    return max_lr_map.get(size_category, 1e-3)

def clamp_lr_by_dataset(lr: float, size_category: str) -> float:
    """
    Clamp LR so batch-size scaling cannot exceed dataset-size safety ceiling.
    Enforces: hard constraints (data risk) > soft heuristics (batch scaling).
    """
    return min(lr, get_max_lr_by_dataset(size_category))