"""Module for seasonal SARIMA time series forecasting model."""


import logging
from typing import Dict, Any, List, Optional

from core.context import RunContext
from models.base_arima import ARIMABaseForecaster
from models.model_registry import register_model
from utils.dataset import TimeSeriesDataset
from utils.data_utils import infer_seasonal_period

logger = logging.getLogger(__name__)

@register_model("sarima", is_univariate=True)
class SARIMAForecaster(ARIMABaseForecaster):
    """Implementation of the seasonal SARIMA forecasting model."""

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
        Initialize the SARIMA forecaster.

        Args:
            model_params: Model-specific parameters (e.g., p, d, q, P, D, Q, seasonal_period).
            num_features: Number of features in the time series data (must be 1).
            forecast_steps: Number of steps to forecast.
            window_size: The look-back window size.
            dataset: TimeSeriesDataset object.
            run_context: RunContext object.
        """
        if "seasonal_period" not in model_params:
            inferred_period = infer_seasonal_period(dataset.freq)
            # Modify param dict (local copy inside of init, safe)
            model_params["seasonal_period"] = inferred_period
            from logging import getLogger
            getLogger(__name__).info(f"Auto-inferred seasonal_period={inferred_period} from freq='{dataset.freq}'")
        super().__init__(
            model_params=model_params,
            num_features=num_features,
            forecast_steps=forecast_steps,
            window_size=window_size,
            dataset=dataset,
            run_context=run_context,
            seasonal=True,
            **kwargs
        )

    @staticmethod
    def _infer_seasonal_period(freq: str) -> Optional[int]:
        """
        Infer typical seasonal period from pandas frequency string.

        This is a helper method that should ideally be in the base class,
        but is included here as a fallback.

        Args:
            freq: Pandas frequency string (e.g., 'H', 'h', 'D', 'W', 'M', 'ME')

        Returns:
            Seasonal period or None
        """
        freq_map = {
            "H": 24, "h": 24,  # Hourly
            "T": 1440, "min": 1440,  # Minutely
            "D": 7,  # Daily
            "W": 52,  # Weekly
            "M": 12, "ME": 12, "MS": 12,  # Monthly
            "Q": 4, "QE": 4, "QS": 4,  # Quarterly
            "Y": 1, "YE": 1, "YS": 1,  # Yearly
        }

        if not freq:
            return None

        # Handle 'min' suffix
        if freq.endswith('min'):
            if len(freq) > 3:
                try:
                    multiplier = int(freq[:-3])
                    return freq_map.get('min', 1440) // multiplier
                except (ValueError, ZeroDivisionError):
                    return freq_map.get('min')
            return freq_map.get('min')

        # Multi-character codes (ME, MS, QE, etc.)
        if len(freq) >= 2:
            two_char = freq[-2:]
            if two_char in freq_map:
                return freq_map[two_char]

        # Single character
        base_freq = freq[-1]
        period = freq_map.get(base_freq)

        # Handle composite (15T, 2H)
        if period and base_freq in ["T", "H", "h"] and len(freq) > 1:
            try:
                multiplier = int(freq[:-1])
                if multiplier > 0:
                    period = period // multiplier
            except (ValueError, ZeroDivisionError):
                pass

        return period

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
        Enhanced SARIMA priors with seasonal and dataset awareness.

        Generates smart starting configurations adapted to:
        - Dataset frequency → inferred seasonal period (s)
        - Series length (affects maximum safe order)
        - Seasonal vs non-seasonal component prioritization

        SARIMA(p,d,q)(P,D,Q,s) models require careful order selection
        especially for seasonal components which multiply the parameters.

        Args:
            param_space: Parameters being optimized
            fixed_params: Fixed parameters from config
            dataset: Optional TimeSeriesDataset for metadata extraction

        Returns:
            List of 3-5 prior configurations adapted to seasonality

        Strategy:
            1. Infer seasonal period from data frequency
            2. Start with simple seasonal models
            3. Consider non-seasonal ARIMA if s is unknown
            4. Balance non-seasonal and seasonal components
        """
        priors = []

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Extract Dataset Metadata
        # ═══════════════════════════════════════════════════════════════

        series_length = None
        size_category = "medium"
        seasonal_period = None

        if dataset:
            # 1. Get series length
            if hasattr(dataset, 'series') and dataset.series is not None:
                series_length = len(dataset.series)

                if series_length < 100:
                    size_category = "tiny"
                elif series_length < 500:
                    size_category = "small"
                elif series_length < 2000:
                    size_category = "medium"
                else:
                    size_category = "large"

            # 2. Infer seasonal period from frequency
            freq = getattr(dataset, "freq", None)
            if freq:
                # Use inherited helper method from base class
                seasonal_period = self._infer_seasonal_period(freq)

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Determine Safe Maximum Orders
        # ═══════════════════════════════════════════════════════════════

        if size_category == "tiny":
            max_safe_p, max_safe_q = 1, 1
            max_safe_P, max_safe_Q = 1, 1
        elif size_category == "small":
            max_safe_p, max_safe_q = 2, 2
            max_safe_P, max_safe_Q = 1, 1
        elif size_category == "medium":
            max_safe_p, max_safe_q = 3, 3
            max_safe_P, max_safe_Q = 2, 2
        else:  # large
            max_safe_p, max_safe_q = 5, 5
            max_safe_P, max_safe_Q = 2, 2

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Set Seasonal Period (s)
        # ═══════════════════════════════════════════════════════════════

        # Use inferred period if available, otherwise check param_space
        s = seasonal_period

        # If not inferred and 's' is in param_space, use a sensible default
        if s is None and 's' in param_space:
            s_options = param_space['s']
            if isinstance(s_options, list) and len(s_options) > 0:
                # Pick most common: 12 (monthly), 7 (daily), 4 (quarterly)
                if 12 in s_options:
                    s = 12
                elif 7 in s_options:
                    s = 7
                elif 4 in s_options:
                    s = 4
                else:
                    s = s_options[0]  # First available

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Generate Priors
        # ═══════════════════════════════════════════════════════════════

        # Prior 1: Simple seasonal model SARIMA(1,1,1)(1,1,1,s)
        # Good baseline for most seasonal data
        prior1 = {
            "p": 1, "d": 1, "q": 1,
            "P": 1, "D": 1, "Q": 1,
        }
        if s is not None:
            prior1["s"] = s
        priors.append(prior1)

        # Prior 2: Seasonal random walk SARIMA(0,1,0)(0,1,1,s)
        # Good for: Series with seasonal pattern but no complex dynamics
        prior2 = {
            "p": 0, "d": 1, "q": 0,
            "P": 0, "D": 1, "Q": 1,
        }
        if s is not None:
            prior2["s"] = s
        priors.append(prior2)

        # Prior 3: Only non-seasonal ARIMA(1,1,1)(0,0,0,s)
        # Good for: When seasonal component is weak or unknown
        prior3 = {
            "p": 1, "d": 1, "q": 1,
            "P": 0, "D": 0, "Q": 0,
        }
        if s is not None:
            prior3["s"] = s
        priors.append(prior3)

        # Prior 4: Moderate SARIMA(2,1,2)(1,1,1,s)
        # Only for larger datasets
        if max_safe_p >= 2 and max_safe_q >= 2:
            prior4 = {
                "p": 2, "d": 1, "q": 2,
                "P": 1, "D": 1, "Q": 1,
            }
            if s is not None:
                prior4["s"] = s
            priors.append(prior4)

        # Prior 5: Strong seasonal SARIMA(0,1,1)(1,1,1,s)
        # Good for: Data dominated by seasonal pattern
        if max_safe_P >= 1 and max_safe_Q >= 1:
            prior5 = {
                "p": 0, "d": 1, "q": 1,
                "P": 1, "D": 1, "Q": 1,
            }
            if s is not None:
                prior5["s"] = s
            priors.append(prior5)

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Filter Priors by param_space
        # ═══════════════════════════════════════════════════════════════

        filtered_priors = []

        for prior in priors:
            valid_prior = {
                k: v for k, v in prior.items()
                if k in param_space and k not in fixed_params
            }

            if valid_prior:
                filtered_priors.append(valid_prior)

        return filtered_priors

    def validate_param_combination(self, params: Dict[str, Any]) -> bool:
        """
        Validate SARIMA parameter combination to avoid wasted trials.

        Rejects configurations that are:
        - Invalid (all AR/MA components zero)
        - Excessive seasonal differencing
        - Too complex for dataset size
        - Seasonal period >= series length / 2

        Args:
            params: Proposed hyperparameter combination

        Returns:
            True if valid, False if should be pruned
        """
        # Non-seasonal orders
        p = params.get("p", 0)
        d = params.get("d", 0)
        q = params.get("q", 0)

        # Seasonal orders
        P = params.get("P", 0)
        D = params.get("D", 0)
        Q = params.get("Q", 0)
        s = params.get("seasonal_period", 0)

        # ═══════════════════════════════════════════════════════════════
        # Rule 1: At least one AR/MA component must be non-zero
        # ═══════════════════════════════════════════════════════════════
        if p == 0 and q == 0 and P == 0 and Q == 0:
            logger.debug(f"[SARIMA] Rejected: p=q=P=Q=0 (no AR/MA components)")
            return False

        # ═══════════════════════════════════════════════════════════════
        # Rule 2: Excessive differencing
        # ═══════════════════════════════════════════════════════════════
        if d >= 3:
            logger.debug(f"[SARIMA] Rejected: d={d} >= 3 (over-differencing)")
            return False

        if D >= 2:
            logger.debug(f"[SARIMA] Rejected: D={D} >= 2 (excessive seasonal differencing)")
            return False

        # ═══════════════════════════════════════════════════════════════
        # Rule 3: Seasonal period validation
        # ═══════════════════════════════════════════════════════════════
        series_length = getattr(self, '_series_length', None)
        if series_length is None and hasattr(self, 'dataset'):
            series_length = len(self.dataset.series) if hasattr(self.dataset, 'series') else None

        if series_length and s > 0:
            # Need at least 2 full seasonal cycles
            if s >= series_length / 2:
                logger.debug(
                    f"[SARIMA] Rejected: seasonal_period={s} >= series_length/2 "
                    f"({series_length}/2 = {series_length // 2})"
                )
                return False

            # Very short series: limit seasonal complexity
            if series_length < 200 and (P > 1 or Q > 1):
                logger.debug(
                    f"[SARIMA] Rejected: P={P} or Q={Q} > 1 on short series "
                    f"(length={series_length})"
                )
                return False

        # ═══════════════════════════════════════════════════════════════
        # Rule 4: Over-differencing combination (d=2 AND D=1)
        # ═══════════════════════════════════════════════════════════════
        if d >= 2 and D >= 1:
            if series_length and series_length < 500:
                logger.debug(
                    f"[SARIMA] Rejected: d={d} AND D={D} on short series "
                    f"(over-differencing risk)"
                )
                return False

        # ═══════════════════════════════════════════════════════════════
        # Rule 5: Total complexity check
        # ═══════════════════════════════════════════════════════════════
        total_order = p + q + P + Q

        if series_length:
            size_category = self._categorize_dataset_size(series_length)

            max_orders = {
                "tiny": 5,
                "small": 7,
                "medium": 12,
                "large": 20
            }

            max_total = max_orders.get(size_category, 12)

            if total_order > max_total:
                logger.debug(
                    f"[SARIMA] Rejected: p+q+P+Q={total_order} > {max_total} "
                    f"for {size_category} dataset"
                )
                return False

        return True

    def filter_search_space(
            self,
            param_space: Dict[str, Any],
            fixed_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter search space based on fixed parameters.

        Removes seasonal parameters (P, D, Q) when seasonal_period is 0.

        Args:
            param_space: Original search space
            fixed_params: Fixed parameters

        Returns:
            Filtered search space
        """
        space = dict(param_space)

        # If seasonal_period is fixed to 0, remove seasonal parameters
        if fixed_params.get("seasonal_period") == 0:
            space.pop("P", None)
            space.pop("D", None)
            space.pop("Q", None)
            logger.debug(
                "[SARIMA] Removed P,D,Q from search space "
                "(seasonal_period=0)"
            )

        return space