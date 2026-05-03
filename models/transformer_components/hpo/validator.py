"""
Parameter validator for Transformer HPO.

Validates parameter combinations before training to reject invalid configs early.
"""
from typing import Dict, Any, Optional
import logging

from models.hpo_heuristics import (
    categorize_dataset_size,
    get_complexity_threshold,
)

logger = logging.getLogger(__name__)


class ParameterValidator:
    """
    Validates Transformer HPO parameter combinations.

    Performs multi-stage validation:
    1. MHA constraints (head_dim range, divisibility)
    2. Dataset-aware complexity threshold
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        num_features: int,
        dataset_length: Optional[int] = None
    ):
        """
        Initialize validator with model context.

        Args:
            model_params: Model configuration dictionary
            num_features: Number of target features
            dataset_length: Length of training dataset (for complexity thresholds)
        """
        self.model_params = model_params
        self.num_features = num_features
        self.dataset_length = dataset_length or 5000  # Default fallback

    def validate(self, params: Dict[str, Any]) -> bool:
        """
        Validate a parameter combination (fast pre-train rejection).

        Args:
            params: Dictionary of parameters to validate

        Returns:
            True if valid, False if should be pruned
        """
        return (
            self._validate_mha_constraints(params) and
            self._validate_complexity_threshold(params)
        )

    def _validate_mha_constraints(self, params: Dict[str, Any]) -> bool:
        """
        Validate multi-head attention constraints.

        Ensures:
        - hidden_size is divisible by num_heads
        - head_dim (hidden_size // num_heads) is in range [8, 128]

        Args:
            params: Parameter dictionary

        Returns:
            True if constraints satisfied, False otherwise
        """
        hidden_size = int(params.get("hidden_size", 128))
        num_heads = int(params.get("num_heads", 4))

        # Divisibility constraint
        if hidden_size % num_heads != 0:
            return False

        # Head dimension range constraint
        head_dim = hidden_size // num_heads
        if head_dim < 8 or head_dim > 128:
            return False

        return True

    def _validate_complexity_threshold(self, params: Dict[str, Any]) -> bool:
        """
        Validate against dataset-aware complexity threshold.

        Rejects parameter combinations that would create models too large
        for the dataset size (overfitting prevention).

        Args:
            params: Parameter dictionary

        Returns:
            True if within threshold, False otherwise
        """
        # Determine dataset size category
        size_category = categorize_dataset_size(self.dataset_length)

        # Extract layer counts (handle legacy/test fallback)
        enc_layers = params.get("num_encoder_layers")
        if enc_layers is None:
            enc_layers = params.get("num_layers", 2)

        dec_layers = params.get("num_decoder_layers", 0)

        try:
            total_layers = int(enc_layers) + int(dec_layers)
        except (TypeError, ValueError):
            total_layers = 2 + int(dec_layers) if dec_layers else 2

        # Calculate complexity threshold
        threshold = get_complexity_threshold(
            self.model_params,
            size_category,
            model_type="transformer",
            num_features=self.num_features
        )

        # Calculate actual complexity
        hidden_size = int(params.get("hidden_size", 128))
        complexity = hidden_size * total_layers

        # Strict inequality as required by tests
        if complexity > threshold:
            logger.debug(
                f"[HPO] Rejected: complexity {complexity} > threshold {threshold} "
                f"({size_category} dataset)"
            )
            return False

        return True
