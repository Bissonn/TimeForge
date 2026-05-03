"""Strategy Pattern implementation for smart prior generation."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from .prior_config import PriorGenerationConfig

logger = logging.getLogger(__name__)


class BasePriorStrategy(ABC):
    """
    Base class for prior generation strategies.

    Provides common functionality for all architecture/strategy combinations:
    - Adaptive sizing based on num_features
    - Dataset adjustments (small dataset → higher dropout)
    - Helper methods for num_heads computation
    """

    @abstractmethod
    def generate_priors(self, config: PriorGenerationConfig) -> List[Dict[str, Any]]:
        """
        Generate 2-3 prior configurations for this architecture/strategy combo.

        Args:
            config: PriorGenerationConfig with all necessary context

        Returns:
            List of 2-3 prior configuration dictionaries
        """
        pass

    def _get_base_sizes_and_dropout(
        self,
        num_features: int,
        strategy: str,
        dataset_info: Dict[str, Any]
    ) -> tuple[List[int], List[float]]:
        """
        Compute adaptive base sizes and dropout ranges based on data dimensions.

        Rule 1: Base hidden_size on num_features
        Guideline: hidden_size ≈ num_features × 8 to 16

        Rule 2: Adjust for iterative mode (need smaller/faster models)

        Rule 3: Adjust dropout for iterative (higher for error accumulation)

        Rule 4: Adjust for small datasets (higher dropout, capped sizes)

        Args:
            num_features: Number of features being modeled
            strategy: "direct" or "iterative"
            dataset_info: Dataset metadata

        Returns:
            Tuple of (base_sizes, dropout_range)
        """
        # Rule 1: Base hidden_size on num_features
        if num_features <= 5:
            # Low dimensional (1-5 features): e.g., univariate or few targets
            base_sizes = [32, 64, 96]
            dropout_range = [0.1, 0.15, 0.2]
            logger.debug(
                f"[SmartHPO] Low dimensional data (num_features={num_features}) "
                f"→ base_sizes={base_sizes}, dropout={dropout_range}"
            )
        elif num_features <= 20:
            # Medium dimensional (6-20 features): typical multivariate
            base_sizes = [64, 128, 192]
            dropout_range = [0.15, 0.2, 0.25]
            logger.debug(
                f"[SmartHPO] Medium dimensional data (num_features={num_features}) "
                f"→ base_sizes={base_sizes}, dropout={dropout_range}"
            )
        elif num_features <= 100:
            # High dimensional (21-100 features): many targets
            base_sizes = [128, 256, 384]
            dropout_range = [0.2, 0.25, 0.3]
            logger.debug(
                f"[SmartHPO] High dimensional data (num_features={num_features}) "
                f"→ base_sizes={base_sizes}, dropout={dropout_range}"
            )
        else:
            # Very high dimensional (>100 features): rare but possible
            base_sizes = [256, 512, 768]
            dropout_range = [0.2, 0.25, 0.3]
            logger.debug(
                f"[SmartHPO] Very high dimensional data (num_features={num_features}) "
                f"→ base_sizes={base_sizes}, dropout={dropout_range}"
            )

        # Rule 2: Adjust for iterative mode (need smaller/faster models)
        if strategy == "iterative":
            # Reduce by ~40% for iterative (speed critical - N forward passes)
            base_sizes = [max(32, int(s * 0.6)) for s in base_sizes]
            logger.debug(
                f"[SmartHPO] Iterative mode → reduced base_sizes to {base_sizes}"
            )

        # Rule 3: Adjust dropout for iterative (higher for error accumulation)
        if strategy == "iterative":
            dropout_range = [min(0.5, d + 0.05) for d in dropout_range]
            logger.debug(
                f"[SmartHPO] Iterative mode → increased dropout to {dropout_range}"
            )

        # Rule 4: Small dataset → higher dropout and capped sizes
        size_category = dataset_info.get("size_category")
        if size_category == "small":
            # Small dataset (<1000 timesteps) → increase dropout by 0.1
            dropout_range = [min(0.5, d + 0.1) for d in dropout_range]
            logger.debug(
                f"[SmartHPO] Small dataset detected → increased dropout to {dropout_range}"
            )

            # Hard cap hidden_size for small datasets
            # Small datasets can't support large models - they'll overfit badly
            original_base_sizes = base_sizes.copy()
            base_sizes = [min(s, 128) for s in base_sizes]

            if original_base_sizes != base_sizes:
                logger.info(
                    f"[SmartHPO] Small dataset → capping hidden_size "
                    f"from {original_base_sizes} to {base_sizes}"
                )

        return base_sizes, dropout_range

    @staticmethod
    def compute_num_heads(hidden_size: int, target_head_dim: int = 16) -> int:
        """
        Compute num_heads to achieve target head_dim while ensuring divisibility.

        Args:
            hidden_size: Model hidden dimension
            target_head_dim: Target dimension per head (default: 16)

        Returns:
            Number of attention heads (ensures hidden_size % num_heads == 0)
        """
        num_heads = max(2, hidden_size // target_head_dim)
        # Adjust to ensure divisibility
        while hidden_size % num_heads != 0 and num_heads > 2:
            num_heads -= 1
        return num_heads


class EncoderOnlyDirectPriors(BasePriorStrategy):
    """
    Encoder-Only + Direct strategy.

    Characteristics: Fast, one-shot prediction, can be larger.
    """

    def generate_priors(self, config: PriorGenerationConfig) -> List[Dict[str, Any]]:
        """Generate 3 priors for Encoder-Only + Direct mode."""
        logger.debug("[SmartHPO] Generating priors for Encoder-Only + Direct mode")

        base_sizes, dropout_range = self._get_base_sizes_and_dropout(
            config.num_features, "direct", config.dataset_info
        )

        priors = []
        size_category = config.dataset_info.get("size_category")

        # Prior 1: Small (fast baseline)
        prior1 = {
            "hidden_size": base_sizes[0],
            "num_heads": self.compute_num_heads(base_sizes[0]),
            "num_encoder_layers": 2,
            "dropout": dropout_range[0],
        }

        # Small dataset: limit complexity
        if size_category == "small":
            prior1["num_encoder_layers"] = min(prior1["num_encoder_layers"], 2)
            logger.debug(
                f"[SmartHPO] Small dataset → prior1 capped at "
                f"{prior1['num_encoder_layers']} encoder layers"
            )

        priors.append(prior1)

        # Prior 2: Medium (balanced, often best)
        prior2 = {
            "hidden_size": base_sizes[1],
            "num_heads": self.compute_num_heads(base_sizes[1]),
            "num_encoder_layers": 3,
            "dropout": dropout_range[1],
        }

        # Small dataset: limit complexity
        if size_category == "small":
            prior2["num_encoder_layers"] = min(prior2["num_encoder_layers"], 3)
            logger.debug(
                f"[SmartHPO] Small dataset → prior2 capped at "
                f"{prior2['num_encoder_layers']} encoder layers"
            )

        priors.append(prior2)

        # Prior 3: Large (high capacity, if dataset is large)
        prior3 = {
            "hidden_size": base_sizes[2],
            "num_heads": self.compute_num_heads(base_sizes[2]),
            "num_encoder_layers": 4,
            "dropout": dropout_range[2],
        }

        # Small dataset: CRITICAL cap to prevent overfitting
        if size_category == "small":
            prior3["num_encoder_layers"] = min(prior3["num_encoder_layers"], 3)
            logger.debug(
                f"[SmartHPO] Small dataset → prior3 capped at "
                f"{prior3['num_encoder_layers']} encoder layers (was 4)"
            )

        priors.append(prior3)

        return priors


class EncoderOnlyIterativePriors(BasePriorStrategy):
    """
    Encoder-Only + Iterative strategy.

    Characteristics: Encoder reused N times, needs to be fast.
    """

    def generate_priors(self, config: PriorGenerationConfig) -> List[Dict[str, Any]]:
        """Generate 3 priors for Encoder-Only + Iterative mode."""
        logger.debug("[SmartHPO] Generating priors for Encoder-Only + Iterative mode")

        base_sizes, dropout_range = self._get_base_sizes_and_dropout(
            config.num_features, "iterative", config.dataset_info
        )

        priors = []
        size_category = config.dataset_info.get("size_category")

        # Prior 1: Micro (speed priority)
        prior1 = {
            "hidden_size": base_sizes[0],
            "num_heads": self.compute_num_heads(base_sizes[0]),
            "num_encoder_layers": 2,
            "dropout": dropout_range[0],
        }

        # Small dataset: already at 2 layers, but log it
        if size_category == "small":
            logger.debug(
                f"[SmartHPO] Small dataset → prior1 already minimal "
                f"({prior1['num_encoder_layers']} layers)"
            )

        priors.append(prior1)

        # Prior 2: Small (balanced, often best for iterative)
        prior2 = {
            "hidden_size": base_sizes[1],
            "num_heads": self.compute_num_heads(base_sizes[1]),
            "num_encoder_layers": 2,
            "dropout": dropout_range[1],
        }

        # Small dataset: already at 2 layers
        if size_category == "small":
            logger.debug(
                f"[SmartHPO] Small dataset → prior2 already minimal "
                f"({prior2['num_encoder_layers']} layers)"
            )

        priors.append(prior2)

        # Prior 3: Medium (quality priority, if speed is not critical)
        prior3 = {
            "hidden_size": base_sizes[2],
            "num_heads": self.compute_num_heads(base_sizes[2]),
            "num_encoder_layers": 3,
            "dropout": dropout_range[2],
        }

        # Small dataset: cap to 3 layers
        if size_category == "small":
            prior3["num_encoder_layers"] = min(prior3["num_encoder_layers"], 3)
            logger.debug(
                f"[SmartHPO] Small dataset → prior3 capped at "
                f"{prior3['num_encoder_layers']} encoder layers"
            )

        priors.append(prior3)

        return priors


class EncoderDecoderDirectPriors(BasePriorStrategy):
    """
    Encoder-Decoder + Direct strategy.

    Characteristics: Most expressive, direct horizon prediction.
    """

    def generate_priors(self, config: PriorGenerationConfig) -> List[Dict[str, Any]]:
        """Generate 3 priors for Encoder-Decoder + Direct mode."""
        logger.debug("[SmartHPO] Generating priors for Encoder-Decoder + Direct mode")

        base_sizes, dropout_range = self._get_base_sizes_and_dropout(
            config.num_features, "direct", config.dataset_info
        )

        priors = []
        size_category = config.dataset_info.get("size_category")
        seasonal_period = config.dataset_info.get("seasonal_period")

        # Prior 1: Small (fast baseline)
        prior1 = {
            "hidden_size": base_sizes[0],
            "num_heads": self.compute_num_heads(base_sizes[0]),
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dropout": dropout_range[0],
        }

        # Add tgt_init if being optimized
        if "tgt_init" in config.param_space:
            prior1["tgt_init"] = "last_value"  # Safe default

        # Small dataset: limit complexity
        if size_category == "small":
            prior1["num_encoder_layers"] = min(prior1["num_encoder_layers"], 2)
            prior1["num_decoder_layers"] = min(prior1["num_decoder_layers"], 2)
            logger.debug(
                f"[SmartHPO] Small dataset → prior1 enc/dec capped at "
                f"{prior1['num_encoder_layers']}/{prior1['num_decoder_layers']}"
            )

        priors.append(prior1)

        # Prior 2: Medium (balanced, often best)
        prior2 = {
            "hidden_size": base_sizes[1],
            "num_heads": self.compute_num_heads(base_sizes[1]),
            "num_encoder_layers": 3,
            "num_decoder_layers": 2,
            "dropout": dropout_range[1],
        }

        if "tgt_init" in config.param_space:
            # Use seasonal if available (better for periodic data), else trend
            prior2["tgt_init"] = "seasonal" if seasonal_period else "trend"

        # Small dataset: cap encoder at 3
        if size_category == "small":
            prior2["num_encoder_layers"] = min(prior2["num_encoder_layers"], 3)
            prior2["num_decoder_layers"] = min(prior2["num_decoder_layers"], 2)
            logger.debug(
                f"[SmartHPO] Small dataset → prior2 enc/dec capped at "
                f"{prior2['num_encoder_layers']}/{prior2['num_decoder_layers']}"
            )

        priors.append(prior2)

        # Prior 3: Large (high capacity)
        prior3 = {
            "hidden_size": base_sizes[2],
            "num_heads": self.compute_num_heads(base_sizes[2]),
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "dropout": dropout_range[2],
        }

        if "tgt_init" in config.param_space:
            prior3["tgt_init"] = "mean"  # Robust choice

        # Small dataset: CRITICAL limit
        if size_category == "small":
            prior3["num_encoder_layers"] = min(prior3["num_encoder_layers"], 3)
            prior3["num_decoder_layers"] = min(prior3["num_decoder_layers"], 2)
            logger.debug(
                f"[SmartHPO] Small dataset → prior3 enc/dec capped at "
                f"{prior3['num_encoder_layers']}/{prior3['num_decoder_layers']} (was 3/3)"
            )

        priors.append(prior3)

        return priors


class EncoderDecoderIterativePriors(BasePriorStrategy):
    """
    Encoder-Decoder + Iterative strategy.

    Characteristics: Most complex, decoder called N times, speed critical.
    """

    def generate_priors(self, config: PriorGenerationConfig) -> List[Dict[str, Any]]:
        """Generate 3 priors for Encoder-Decoder + Iterative mode."""
        logger.debug("[SmartHPO] Generating priors for Encoder-Decoder + Iterative mode")

        base_sizes, dropout_range = self._get_base_sizes_and_dropout(
            config.num_features, "iterative", config.dataset_info
        )

        priors = []
        size_category = config.dataset_info.get("size_category")

        # Prior 1: Micro (speed critical!)
        prior1 = {
            "hidden_size": base_sizes[0],
            "num_heads": self.compute_num_heads(base_sizes[0]),
            "num_encoder_layers": 2,
            "num_decoder_layers": 1,  # Very light decoder
            "dropout": dropout_range[0],
        }

        if "tgt_init" in config.param_space:
            prior1["tgt_init"] = "last_value"  # Simple and fast

        # Small dataset: already minimal (2/1)
        if size_category == "small":
            logger.debug(
                f"[SmartHPO] Small dataset → prior1 already minimal "
                f"({prior1['num_encoder_layers']}/{prior1['num_decoder_layers']})"
            )

        priors.append(prior1)

        # Prior 2: Small (balanced)
        prior2 = {
            "hidden_size": base_sizes[1],
            "num_heads": self.compute_num_heads(base_sizes[1]),
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dropout": dropout_range[1],
        }

        if "tgt_init" in config.param_space:
            prior2["tgt_init"] = "last_value"

        # Small dataset: already at 2/2
        if size_category == "small":
            logger.debug(
                f"[SmartHPO] Small dataset → prior2 already conservative "
                f"({prior2['num_encoder_layers']}/{prior2['num_decoder_layers']})"
            )

        priors.append(prior2)

        # Prior 3: Medium (quality focus, if speed acceptable)
        prior3 = {
            "hidden_size": base_sizes[2],
            "num_heads": self.compute_num_heads(base_sizes[2]),
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dropout": dropout_range[2],
        }

        if "tgt_init" in config.param_space:
            prior3["tgt_init"] = "mean"  # More stable for long chains

        # Small dataset: already at 2/2
        if size_category == "small":
            logger.debug(
                f"[SmartHPO] Small dataset → prior3 already conservative "
                f"({prior3['num_encoder_layers']}/{prior3['num_decoder_layers']})"
            )

        priors.append(prior3)

        return priors
