"""Smart prior generator for Transformer HPO using Strategy Pattern."""

import logging
from typing import Dict, List, Any, Optional

from models.hpo_heuristics import categorize_dataset_size
from .prior_config import PriorGenerationConfig
from .prior_strategies import (
    BasePriorStrategy,
    EncoderOnlyDirectPriors,
    EncoderOnlyIterativePriors,
    EncoderDecoderDirectPriors,
    EncoderDecoderIterativePriors,
)
from .lr_calculator import LearningRateCalculator

logger = logging.getLogger(__name__)


class SmartPriorGenerator:
    """
    Generates smart priors for Transformer HPO using Strategy Pattern.

    This class coordinates the generation of optimized starting configurations
    for different Transformer modes:
    - Encoder-only + Direct: Large capacity, single forward pass
    - Encoder-only + Iterative: Smaller, faster (reused N times)
    - Encoder-decoder + Direct: Balanced, with tgt_init
    - Encoder-decoder + Iterative: Lightweight, speed-critical

    Priors are automatically adapted to:
    - Actual number of features being modeled (num_features)
    - Dataset characteristics (frequency, size, exog variables)
    - Outlier robustness (Huber loss prioritization)
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        num_features: int,
        forecast_steps: int,
        filter_search_space_fn: Any,
        infer_seasonal_period_fn: Any,
        suggest_batch_sizes_fn: Any,
        is_valid_prior_value_fn: Any
    ):
        """
        Initialize the SmartPriorGenerator.

        Args:
            model_params: Model parameters dictionary
            num_features: Number of features being modeled
            forecast_steps: Forecast horizon length
            filter_search_space_fn: Function to filter search space
            infer_seasonal_period_fn: Function to infer seasonal period from frequency
            suggest_batch_sizes_fn: Function to suggest batch sizes
            is_valid_prior_value_fn: Function to validate prior values
        """
        self.model_params = model_params
        self.num_features = num_features
        self.forecast_steps = forecast_steps
        self.filter_search_space = filter_search_space_fn
        self._infer_seasonal_period = infer_seasonal_period_fn
        self._suggest_batch_sizes = suggest_batch_sizes_fn
        self._is_valid_prior_value = is_valid_prior_value_fn

        # Initialize strategies
        self._strategies: Dict[tuple, BasePriorStrategy] = {
            ("encoder-only", "direct"): EncoderOnlyDirectPriors(),
            ("encoder-only", "iterative"): EncoderOnlyIterativePriors(),
            ("encoder-decoder", "direct"): EncoderDecoderDirectPriors(),
            ("encoder-decoder", "iterative"): EncoderDecoderIterativePriors(),
        }

    def suggest_priors(
        self,
        param_space: Dict[str, Any],
        fixed_params: Dict[str, Any],
        dataset: Optional['TimeSeriesDataset'] = None,
        window_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Main entry point - matches the old suggest_smart_priors signature.

        Enhanced smart priors based on architecture, strategy, data dimensions,
        AND dataset metadata.

        Args:
            param_space: Parameters being optimized
            fixed_params: Fixed parameters from config
            dataset: Optional TimeSeriesDataset for metadata extraction
            window_size: Historical window size (optional, defaults to forecast_steps)

        Returns:
            List of 2-3 prior configurations per mode (max 10 total)
        """
        # Filter param_space first (remove invalid parameters)
        param_space = self.filter_search_space(param_space, fixed_params)

        # Get actual data dimensions
        window_size = window_size or self.forecast_steps

        logger.info(
            f"[SmartHPO] Data dimensions: num_features={self.num_features}, "
            f"forecast_steps={self.forecast_steps}, window_size={window_size}"
        )

        # Extract dataset metadata for enhanced priors
        dataset_info = self._extract_dataset_metadata(dataset)

        # Dataset-aware batch size suggestions
        batch_sizes = None
        if "batch_size" in param_space and dataset is not None:
            series_length = dataset.training_length
            num_features = (
                len(dataset.target_columns) + len(dataset.past_covariates or [])
                + len(dataset.future_covariates or [])
            )
            batch_sizes = self._suggest_batch_sizes(series_length, num_features)
            logger.info(f"[Transformer SmartHPO] Suggested batch sizes: {batch_sizes}")

        # Determine configuration
        arch = fixed_params.get("architecture")
        strategy = fixed_params.get("strategy", "direct")

        # Handle case where architecture/strategy are being optimized
        if not arch and "architecture" in param_space:
            archs_to_consider = param_space["architecture"]
            if isinstance(archs_to_consider, list):
                # Multiple architectures being optimized
                # Generate priors for each, but limit to 2 per combo
                all_priors = []

                strategies_to_consider = (
                    param_space.get("strategy", ["direct"])
                    if "strategy" in param_space
                    else [strategy]
                )

                if not isinstance(strategies_to_consider, list):
                    strategies_to_consider = [strategies_to_consider]

                for arch_option in archs_to_consider:
                    for strat_option in strategies_to_consider:
                        combo_priors = self._generate_priors_for_config(
                            arch_option,
                            strat_option,
                            param_space,
                            fixed_params,
                            window_size,
                            dataset_info
                        )
                        # Take only 2 priors per combination
                        all_priors.extend(combo_priors[:2])

                # Limit to 10 total priors (Optuna best practice)
                return all_priors[:10]
            else:
                arch = archs_to_consider

        # Strategy might also be in search space
        if not strategy and "strategy" in param_space:
            strategies = param_space["strategy"]
            if isinstance(strategies, list):
                # Multiple strategies - generate for each
                all_priors = []
                for strat in strategies:
                    priors = self._generate_priors_for_config(
                        arch, strat, param_space, fixed_params,
                        window_size, dataset_info
                    )
                    all_priors.extend(priors[:2])
                return all_priors[:10]
            else:
                strategy = strategies

        # Single fixed configuration
        priors = self._generate_priors_for_config(
            arch,
            strategy,
            param_space,
            fixed_params,
            window_size,
            dataset_info
        )

        # Distribute batch sizes across priors
        if batch_sizes is not None and len(priors) > 0:
            # Assign batch sizes to first few priors
            for i, prior in enumerate(priors[:len(batch_sizes)]):
                prior["batch_size"] = batch_sizes[i]

            # For remaining priors, use the smallest recommended batch
            for prior in priors[len(batch_sizes):]:
                prior["batch_size"] = batch_sizes[0]

            logger.info(f"[Transformer SmartHPO] Added batch_size to {len(priors)} priors")

        logger.info(
            f"[SmartHPO] Generated {len(priors)} config-aware priors for "
            f"architecture={arch}, strategy={strategy}"
        )

        return priors

    def _extract_dataset_metadata(
        self,
        dataset: Optional['TimeSeriesDataset']
    ) -> Dict[str, Any]:
        """
        Extract dataset metadata for enhanced priors.

        Args:
            dataset: Optional TimeSeriesDataset

        Returns:
            Dictionary with dataset metadata (freq, size_category, seasonal_period, etc.)
        """
        dataset_info = {}

        if dataset:
            # 1. Frequency → Implied seasonal period
            freq = getattr(dataset, "freq", None)
            if freq:
                dataset_info["freq"] = freq
                dataset_info["seasonal_period"] = self._infer_seasonal_period(freq)
                logger.info(
                    f"[SmartHPO] Dataset frequency: {freq}, "
                    f"implied seasonal_period: {dataset_info.get('seasonal_period')}"
                )

            # 2. Dataset size → Overfitting risk assessment
            if hasattr(dataset, "series") and dataset.series is not None:
                series_length = dataset.training_length
                dataset_info["series_length"] = series_length
                dataset_info["size_category"] = categorize_dataset_size(series_length)
                logger.info(
                    f"[SmartHPO] Dataset size: {series_length} ({dataset_info['size_category']})"
                )

            # 3. Exogenous variables → Architecture hints
            has_past_exog = len(getattr(dataset, "past_covariates", []) or []) > 0
            has_future_exog = len(getattr(dataset, "future_covariates", []) or []) > 0
            if has_past_exog or has_future_exog:
                dataset_info["has_past_exog"] = has_past_exog
                dataset_info["has_future_exog"] = has_future_exog
                logger.info(
                    f"[SmartHPO] Exogenous variables: "
                    f"past={has_past_exog}, future={has_future_exog}"
                )

        return dataset_info

    def _generate_priors_for_config(
        self,
        architecture: str,
        strategy: str,
        param_space: Dict[str, Any],
        fixed_params: Dict[str, Any],
        window_size: int,
        dataset_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate configuration-specific smart priors.

        Args:
            architecture: "encoder-only" or "encoder-decoder"
            strategy: "direct" or "iterative"
            param_space: Parameters being optimized
            fixed_params: Fixed parameters
            window_size: Historical window size
            dataset_info: Dataset metadata

        Returns:
            List of 2-3 prior configurations optimized for this mode and data
        """
        # Default to "direct" if strategy not specified
        if strategy not in ["direct", "iterative"]:
            strategy = "direct"
            logger.debug(f"[SmartHPO] Strategy '{strategy}' unknown, defaulting to 'direct'")

        # Create configuration
        config = PriorGenerationConfig(
            num_features=self.num_features,
            forecast_steps=self.forecast_steps,
            window_size=window_size,
            param_space=param_space,
            fixed_params=fixed_params,
            dataset_info=dataset_info,
        )

        # Select strategy and generate priors
        strategy_key = (architecture, strategy)
        if strategy_key in self._strategies:
            priors = self._strategies[strategy_key].generate_priors(config)
        else:
            # Fallback for unexpected configurations
            logger.warning(
                f"[SmartHPO] Unknown configuration: architecture={architecture}, "
                f"strategy={strategy}. Using generic prior."
            )
            priors = [{
                "hidden_size": 64,
                "num_heads": 4,
                "num_encoder_layers": 2,
            }]

        # Post-generation enhancements
        priors = self._add_learning_rates(priors, param_space, strategy, dataset_info)
        priors = self._add_seasonal_periods(priors, param_space, dataset_info)
        priors = self._prioritize_huber_loss(priors, param_space)
        priors = self._filter_priors(priors, param_space)

        return priors

    def _add_learning_rates(
        self,
        priors: List[Dict[str, Any]],
        param_space: Dict[str, Any],
        strategy: str,
        dataset_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Add learning rates to priors based on model characteristics."""
        if "learning_rate" not in param_space:
            return priors

        logger.info("[SmartHPO] Adding learning rates to priors based on model characteristics")

        calculator = LearningRateCalculator(self.model_params)

        for idx, prior in enumerate(priors, 1):
            # Extract model characteristics
            hs = prior.get("hidden_size", 128)
            nl_enc = prior.get("num_encoder_layers", 2)
            nl_dec = prior.get("num_decoder_layers", 0)
            total_layers = nl_enc + nl_dec
            bs = prior.get("batch_size", 32)

            # Compute smart learning rate
            lr = calculator.calculate_lr(
                hidden_size=hs,
                num_layers=total_layers,
                strategy=strategy,
                dataset_size=dataset_info.get("size_category", "medium"),
                batch_size=bs
            )

            # Clamp to search space bounds (safety check)
            lr_space = param_space["learning_rate"]
            if isinstance(lr_space, dict):
                lr_min = lr_space.get("min", 1e-5)
                lr_max = lr_space.get("max", 1e-2)
                lr = max(lr_min, min(lr, lr_max))

            prior["learning_rate"] = lr

            logger.debug(
                f"[SmartHPO] Prior {idx}: added learning_rate={lr:.2e} "
                f"(hs={hs}, layers={total_layers}, strategy={strategy}, "
                f"size={dataset_info.get('size_category')})"
            )

        return priors

    def _add_seasonal_periods(
        self,
        priors: List[Dict[str, Any]],
        param_space: Dict[str, Any],
        dataset_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Add seasonal_period to seasonal tgt_init priors."""
        seasonal_period = dataset_info.get("seasonal_period")

        if seasonal_period and "tgt_init" in param_space:
            for prior in priors:
                if prior.get("tgt_init") == "seasonal":
                    # Add seasonal_period if it's also in param_space
                    if "seasonal_period" in param_space:
                        prior["seasonal_period"] = seasonal_period
                        logger.debug(
                            f"[SmartHPO] Added seasonal_period={seasonal_period} "
                            f"to prior with tgt_init='seasonal'"
                        )

        return priors

    def _prioritize_huber_loss(
        self,
        priors: List[Dict[str, Any]],
        param_space: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prioritize Huber loss for outlier robustness."""
        if "loss" not in param_space:
            return priors

        loss_options = param_space["loss"]
        # Check if huber is available in search space
        if isinstance(loss_options, list) and "huber" in loss_options:
            # Ensure at least ONE prior uses Huber loss
            huber_prior_exists = any(p.get("loss") == "huber" for p in priors)

            if not huber_prior_exists and len(priors) >= 2:
                # Assign Huber to Prior 2 (middle prior, often best baseline)
                priors[1]["loss"] = "huber"

                # Add default delta parameter if loss_params is in search space
                if "loss_params" in param_space or "delta" in param_space:
                    priors[1]["loss_params"] = {"delta": 1.0}

                logger.info(
                    "[SmartHPO] Prioritized Huber loss in Prior 2 for outlier robustness. "
                    "This ensures robust loss is tested early (trial 2-3) rather than late (trial 5+)."
                )

        return priors

    def _filter_priors(
        self,
        priors: List[Dict[str, Any]],
        param_space: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter priors to only include params in search space."""
        filtered_priors = []

        for idx, prior in enumerate(priors, 1):
            # Only include parameters that are in param_space (being optimized)
            valid_prior = {
                k: v for k, v in prior.items()
                if self._is_valid_prior_value(k, v, param_space)
            }

            # Better logging
            invalid = {k: v for k, v in prior.items()
                       if k in param_space and not self._is_valid_prior_value(k, v, param_space)}
            if invalid:
                logger.debug(f"[Transformer Prior {idx}] Filtered invalid: {invalid}")

            # Only add if non-empty
            if valid_prior:
                filtered_priors.append(valid_prior)
                logger.debug(f"[SmartHPO] Prior {idx}: {valid_prior}")
            else:
                logger.debug(
                    f"[SmartHPO] Prior {idx} skipped (no params in search space)"
                )

        return filtered_priors
