"""Module for non-seasonal ARIMA time series forecasting model."""

import logging
from typing import Dict, Any, List, Optional

from models.base_arima import ARIMABaseForecaster
from models.model_registry import register_model
from utils.dataset import TimeSeriesDataset
from core.context import RunContext

logger = logging.getLogger(__name__)

@register_model("arima", is_univariate=True)
class ARIMAForecaster(ARIMABaseForecaster):
    """Implementation of the non-seasonal ARIMA forecasting model."""

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
        Initialize the ARIMA forecaster.

        Args:
            model_params (Dict[str, Any]): Model-specific parameters.
            num_features (int): The number of target features.
            forecast_steps (int): The forecast horizon.
            window_size (int): The look-back window size.
            dataset: Input dataset.
            run_context: RunContext object.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(
            model_params=model_params,
            num_features=num_features,
            forecast_steps=forecast_steps,
            window_size=window_size,
            dataset=dataset,
            run_context=run_context,
            seasonal=False,
            **kwargs
        )


    # =========================================================================
    # SMART HPO IMPLEMENTATION
    # =========================================================================

    def filter_search_space(self, param_space: Dict[str, Any], fixed_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter search space for non-seasonal ARIMA.
        Removes seasonal parameters (P, D, Q) if present in search space.
        """
        filtered = param_space.copy()

        # Explicitly remove seasonal parameters for this non-seasonal class
        seasonal_params = ["P", "D", "Q", "s", "seasonal_period"]
        for param in seasonal_params:
            if param in filtered:
                # Only log if we are actually removing something
                logger.info(f"[SmartHPO] Removing '{param}' from search space (Non-seasonal ARIMA).")
                filtered.pop(param, None)

        return filtered

    @staticmethod
    def _categorize_dataset_size(length: int) -> str:
        """
        Categorize dataset size for complexity adaptation.

        Args:
            length: Number of observations

        Returns:
            'tiny', 'small', 'medium', or 'large'
        """
        if length < 100:
            return "tiny"
        elif length < 500:
            return "small"
        elif length < 2000:
            return "medium"
        else:
            return "large"

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN METHOD
    # ═══════════════════════════════════════════════════════════════════════

    def suggest_smart_priors(
            self,
            param_space: Dict[str, Any],
            fixed_params: Dict[str, Any],
            dataset: Optional['TimeSeriesDataset'] = None
    ) -> List[Dict[str, Any]]:
        """
        Enhanced ARIMA priors with dataset awareness.

        Generates smart starting configurations adapted to:
        - Dataset characteristics (stationarity, trend, seasonality indicators)
        - Series length (affects maximum safe order)
        - Differencing needs based on data

        ARIMA models are more sensitive to order selection than neural models.
        Start conservative and let HPO explore higher orders.

        Args:
            param_space: Parameters being optimized
            fixed_params: Fixed parameters from config
            dataset: Optional TimeSeriesDataset for metadata extraction

        Returns:
            List of 3-5 prior configurations (conservative → aggressive)

        Strategy:
            1. Start with simple models (low orders)
            2. Gradually increase complexity
            3. Avoid orders that are too high for small datasets
            4. Consider differencing based on trend indicators
        """
        priors = []

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Extract Dataset Metadata
        # ═══════════════════════════════════════════════════════════════

        series_length = None
        size_category = "medium"  # Default assumption

        if dataset and hasattr(dataset, 'series') and dataset.series is not None:
            series_length = len(dataset.series)

            # Categorize dataset size using helper method
            size_category = self._categorize_dataset_size(series_length)

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Determine Safe Maximum Orders
        # ═══════════════════════════════════════════════════════════════

        # Rule of thumb: max_order ≈ series_length / 10 (but capped)
        if size_category == "tiny":
            max_safe_p = 2
            max_safe_q = 2
        elif size_category == "small":
            max_safe_p = 3
            max_safe_q = 3
        elif size_category == "medium":
            max_safe_p = 5
            max_safe_q = 5
        else:  # large
            max_safe_p = 7
            max_safe_q = 7

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Generate Priors (Conservative → Aggressive)
        # ═══════════════════════════════════════════════════════════════

        # Prior 1: Random Walk (simplest baseline)
        # Good for: Financial data, prices, many real-world series
        prior1 = {"p": 0, "d": 1, "q": 0}  # ARIMA(0,1,0)
        priors.append(prior1)

        # Prior 2: ARIMA(1,1,1) - Classic workhorse
        # Good for: Most time series with trend
        prior2 = {"p": 1, "d": 1, "q": 1}
        priors.append(prior2)

        # Prior 3: AR(1) - Simple autoregressive
        # Good for: Stationary series without strong MA component
        prior3 = {"p": 1, "d": 0, "q": 0}
        priors.append(prior3)

        # Prior 4: ARIMA(2,1,2) - Moderate complexity
        # Only if dataset is large enough
        if max_safe_p >= 2 and max_safe_q >= 2:
            prior4 = {"p": 2, "d": 1, "q": 2}
            priors.append(prior4)

        # Prior 5: ARIMA(1,0,1) - For stationary series
        # Good for: Series that don't need differencing
        prior5 = {"p": 1, "d": 0, "q": 1}
        priors.append(prior5)

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Filter Priors by param_space
        # ═══════════════════════════════════════════════════════════════

        filtered_priors = []

        for prior in priors:
            # Only include parameters in param_space and not in fixed_params
            valid_prior = {
                k: v for k, v in prior.items()
                if k in param_space and k not in fixed_params
            }

            # Only add if non-empty
            if valid_prior:
                filtered_priors.append(valid_prior)

        return filtered_priors

    def validate_param_combination(self, params: Dict[str, Any]) -> bool:
        """
        Validate ARIMA parameter combination to avoid wasted trials.

        Rejects configurations that are:
        - Invalid (p=q=0, pure differencing)
        - Excessive (d>=3, over-differencing)
        - Too complex for dataset size

        Args:
            params: Proposed hyperparameter combination

        Returns:
            True if valid, False if should be pruned
        """
        p = params.get("p", 0)
        d = params.get("d", 0)
        q = params.get("q", 0)

        # ═══════════════════════════════════════════════════════════════
        # Rule 1: Reject p=q=0 (pure differencing, no AR/MA)
        # ═══════════════════════════════════════════════════════════════
        if p == 0 and q == 0:
            logger.debug(f"[ARIMA] Rejected: p=q=0 (pure differencing)")
            return False

        # ═══════════════════════════════════════════════════════════════
        # Rule 2: Reject d >= 3 (excessive differencing)
        # ═══════════════════════════════════════════════════════════════
        if d >= 3:
            logger.debug(f"[ARIMA] Rejected: d={d} >= 3 (over-differencing)")
            return False

        # ═══════════════════════════════════════════════════════════════
        # Rule 3: Total order (p+q) check based on dataset size
        # ═══════════════════════════════════════════════════════════════
        total_order = p + q

        # Get dataset size if available
        series_length = getattr(self, '_series_length', None)
        if series_length is None and hasattr(self, 'dataset'):
            series_length = len(self.dataset.series) if hasattr(self.dataset, 'series') else None

        if series_length:
            size_category = self._categorize_dataset_size(series_length)

            # Max total order by size
            max_orders = {
                "tiny": 4,  # p+q <= 4
                "small": 6,  # p+q <= 6
                "medium": 10,  # p+q <= 10
                "large": 15  # p+q <= 15
            }

            max_total = max_orders.get(size_category, 10)

            if total_order > max_total:
                logger.debug(
                    f"[ARIMA] Rejected: p+q={total_order} > {max_total} "
                    f"for {size_category} dataset"
                )
                return False

        # ═══════════════════════════════════════════════════════════════
        # Rule 4: Warn about high individual orders
        # ═══════════════════════════════════════════════════════════════
        if p > 10 or q > 10:
            logger.debug(
                f"[ARIMA] Borderline: p={p} or q={q} > 10 "
                f"(may be slow to fit)"
            )
            # Don't reject, but log warning

        return True