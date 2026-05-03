"""Module for Vector Autoregression (VAR) time series forecasting model.

This module defines the VARForecaster class, which implements a multivariate VAR model
using the statsmodels library, extending the StatTSForecaster base class.
"""

import logging
from typing import Dict, Optional, Set, Any, List

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR

from core.context import RunContext
from models.base import StatTSForecaster
from models.model_registry import register_model
from utils.preprocessor import Preprocessor
from utils.dataset import TimeSeriesDataset

logger = logging.getLogger(__name__)

@register_model("var", is_univariate=False)
class VARForecaster(StatTSForecaster):
    """Implementation of the Vector Autoregression (VAR) forecasting model."""

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
        Initialize the VAR forecaster.

        Args:
            model_params: Model-specific parameters (e.g., max_lags).
            num_features: Number of features in the time series data (must be > 1 for VAR).
            forecast_steps: Number of steps to forecast.
            window_size: The look-back window size (used for validation).
            **kwargs: Additional keyword arguments passed to the base class, including `input_size`.

        Raises:
            ValueError: If num_features is less than 2 or model parameters are invalid.
        """
        if num_features < 2:
            raise ValueError(f"VAR/MAX requires at least 2 endogenous variables (columns). Got {num_features}.")

        super().__init__(
            model_params=model_params,
            num_features=num_features,
            forecast_steps=forecast_steps,
            window_size=window_size,
            dataset=dataset,
            run_context=run_context,
            **kwargs
        )
        self.max_lags = model_params.get("max_lags", 1)
        self.maxiter = model_params.get("maxiter", 100)
        # Information criterion for automatic lag order selection.
        # When set, fit() iterates over 1..max_lags and picks the lag that
        # minimises the chosen IC.  Supported: "aic", "bic", "hqic".
        # When None (default), max_lags is used directly as the fixed order.
        self.ic: Optional[str] = model_params.get("ic", None)
        self.fitted_exog_names: Optional[List[str]] = None
        self.target_columns: Optional[List[str]] = None
        # Set in fit() to the actual lag order used (may differ from max_lags when ic is set).
        self._fitted_lag: int = self.max_lags
        self.use_varmax = False

        # Cached preprocessed training data used by _apply_window_context().
        # Populated in fit(), consumed in predict(). Stored in transformed space
        # (same space the Kalman filter was trained in).
        self._train_endog_processed: Optional[pd.DataFrame] = None
        self._train_exog_processed: Optional[pd.DataFrame] = None

        # ═══════════════════════════════════════════════════════════════════
        # Configurable stationarity/invertibility enforcement
        # ═══════════════════════════════════════════════════════════════════
        # Allow users to control VARMAX constraints from config
        # enforce_stationarity=True can cause fitting failures for naturally non-stationary series
        self.enforce_stationarity = model_params.get("enforce_stationarity", True)
        self.enforce_invertibility = model_params.get("enforce_invertibility", True)

        # Deterministic trend component passed to VARMAX.
        # "n"  — no deterministic component
        # "c"  — constant (intercept)         ← statsmodels VARMAX default
        # "t"  — linear time trend
        # "ct" — constant + linear time trend
        self.trend: str = model_params.get("trend", "c")

        # Structure of the error covariance matrix.
        # "unstructured" — full k×k covariance estimated (default; most flexible)
        # "diagonal"     — only k diagonal variances estimated (fewer params, more stable)
        # "scalar"       — single variance parameter (most constrained)
        # With k=7 variables: "unstructured"=28 params, "diagonal"=7, "scalar"=1.
        # Use "diagonal" when T is small or the model struggles to converge.
        self.error_cov_type: str = model_params.get("error_cov_type", "unstructured")

        # ═══════════════════════════════════════════════════════════════════
        # OPTIMIZATION: Configurable solver and iteration limits
        # ═══════════════════════════════════════════════════════════════════
        # Default solver 'lbfgs' is accurate but slow
        # Alternative solvers: 'bfgs', 'newton', 'nm', 'powell'
        # maxiter controls optimization iterations (default 100)
        self.method = model_params.get("method", "lbfgs")

        # Memory management: remove_data after fit to reduce memory footprint in HPO
        self.remove_data_after_fit = model_params.get("remove_data_after_fit", False)

        self._validate_model_params()
        logger.info(
            f"Initialized {self.__class__.__name__} with max_lags={self.max_lags}, "
            f"enforce_stationarity={self.enforce_stationarity}, "
            f"enforce_invertibility={self.enforce_invertibility}, "
            f"method={self.method}, maxiter={self.maxiter}"
        )

    def _compute_var_warmup(self) -> int:
        """
        Compute additional points needed on top of window_size to properly
        initialize the Kalman filter before the effective forecast context begins.

        VAR(p) is fitted via VARMAX with order=(p, 0), meaning no differencing
        (d=0).  There is therefore no diffuse initialization phase.  The only
        warm-up required is for the multivariate Kalman filter state-error
        covariance to converge, which is slower than in the univariate case due
        to the k×k covariance matrix estimation (k = number of endogenous
        variables).  We use max_lags * 3 as a conservative heuristic.

        Returns:
            Number of warm-up points to prepend to window_size.
        """
        # No integration order in VAR (d=0 by construction), so no diffuse phase.
        # Convergence buffer is larger than for ARIMA to account for multivariate
        # covariance matrix stabilization.
        # Use _fitted_lag (actual order used) rather than max_lags so that IC
        # selection of a smaller lag produces a proportionally smaller warmup.
        return self._fitted_lag * 3

    def _apply_window_context(self) -> Any:
        """
        Apply the globally fitted model to the last (window_size + warmup) points
        of the training data to localize the Kalman filter state.

        This mirrors the behaviour of neural models, where window_size defines
        exactly how many historical observations inform each forecast: here the
        model parameters come from the global fit on the full training fold while
        the Kalman filter state is re-estimated from the recent context window
        only.

        The effective slice passed to apply() is:
            effective = window_size + _compute_var_warmup()

        The warm-up prefix ensures that by the time the filter reaches the last
        window_size points its state covariance has stabilized, so the effective
        context seen by get_forecast() corresponds to exactly window_size
        meaningful observations.

        Falls back to self.model (full-history state) with a warning when:
        - window_size is None (feature disabled)
        - _train_endog_processed is not cached (fit() not yet called)
        - available training data <= effective window (nothing to trim)

        Returns:
            VARMAXResults — either the result of apply() or the original
            self.model, depending on applicability.
        """
        if self.window_size is None or self._train_endog_processed is None:
            return self.model

        warmup = self._compute_var_warmup()
        effective = self.window_size + warmup
        n_train = len(self._train_endog_processed)

        if n_train <= effective:
            logger.warning(
                f"[{self.__class__.__name__}] Window context skipped: "
                f"n_train={n_train} <= window_size + warmup "
                f"({self.window_size} + {warmup} = {effective}). "
                f"Using full training history."
            )
            return self.model

        endog_window = self._train_endog_processed.iloc[-effective:]
        exog_window = (
            self._train_exog_processed.iloc[-effective:]
            if self._train_exog_processed is not None
            else None
        )

        logger.info(
            f"[{self.__class__.__name__}] Applying window context: "
            f"window_size={self.window_size}, warmup={warmup}, "
            f"effective_slice={effective} / {n_train} training points."
        )

        return self.model.apply(endog_window, exog=exog_window, refit=False)

    def _validate_model_params(self) -> None:
        """
        Validate VAR model parameters.

        Raises:
            ValueError: If max_lags is invalid .
        """
        if not isinstance(self.max_lags, int) or self.max_lags < 1:
            raise ValueError("max_lags must be a positive integer.")
        if not isinstance(self.maxiter, int) or self.maxiter < 1:
            raise ValueError("maxiter must be a positive integer.")
        _valid_ic = {"aic", "bic", "hqic"}
        if self.ic is not None and self.ic not in _valid_ic:
            raise ValueError(f"ic must be one of {sorted(_valid_ic)} or None. Got: '{self.ic}'.")
        _valid_trend = {"n", "c", "t", "ct"}
        if self.trend not in _valid_trend:
            raise ValueError(f"trend must be one of {sorted(_valid_trend)}. Got: '{self.trend}'.")
        _valid_error_cov = {"unstructured", "diagonal", "scalar"}
        if self.error_cov_type not in _valid_error_cov:
            raise ValueError(f"error_cov_type must be one of {sorted(_valid_error_cov)}. Got: '{self.error_cov_type}'.")

    def _validate_model_specific_inputs(
        self, train_series: pd.DataFrame, val_series: Optional[pd.DataFrame] = None,
        forecast_steps: Optional[int] = None
    ) -> None:
        """
        Validate inputs specific to VAR models.

        Args:
            train_series: Training data.
            val_series: Validation data (optional, ignored). Defaults to None.
            forecast_steps: Forecast steps (optional). Defaults to None.

        Raises:
            ValueError: If train_series has insufficient data or columns.
        """
        if not isinstance(train_series, pd.DataFrame):
            raise ValueError("train_series must be a pandas DataFrame.")
        # VAR/MAX requires at least 2 endog columns
        if train_series.shape[1] < 2:
            raise ValueError("VAR/MAX requires at least 2 endogenous variables (columns).")
        # no NaNs after preprocessing
        if train_series.isna().any().any():
            raise ValueError("train_series contains NaN values after preprocessing.")
        # p >= 1
        if not isinstance(self.max_lags, int) or self.max_lags < 1:
            raise ValueError("max_lags must be a positive integer.")
        # forecast_steps > 0
        if forecast_steps is not None and (not isinstance(forecast_steps, int) or forecast_steps < 1):
            raise ValueError("forecast_steps must be a positive integer if provided.")

    def fit(
        self,
        train_series: pd.DataFrame,
        exog_series: Optional[pd.DataFrame] = None,
        dataset: Optional[TimeSeriesDataset] = None,
        is_final_fit: bool = False,
        **kwargs
    ) -> tuple:
        """
        Fit the VAR model to the training data.

        Args:
            train_series: Training data (must have at least two columns).
            exog_series: Optional DataFrame of exogenous variables for training.
            dataset: Dataset object (ignored by this model, for signature compatibility).
            is_final_fit: Whether this is final fit (ignored by this model).
            **kwargs: Additional arguments (ignored, for signature compatibility).

        Returns:
            Tuple of (validation_loss, training_history). For statistical models, returns (0.0, {}).

        Raises:
            ValueError: If train_series is invalid or model parameters are invalid.
            RuntimeError: If VAR model fitting fails.
        """
        self.target_columns = train_series.columns.tolist()
        # NOTE: exogenous columns are handled via the Preprocessor and stored after preprocessing

        combined_train_data = (
            pd.concat([train_series, exog_series], axis=1) if exog_series is not None
            else train_series
        )

        # Re-initialize preprocessor with the correct column context for this fit
        self.preprocessor = Preprocessor(
            self.model_params.get("preprocessing", {}),
            target_columns=self.target_columns,
            exog_columns=exog_series.columns.tolist() if exog_series is not None else []
        )

        train_processed = self.preprocessor.fit_transform(combined_train_data)

        # ═══════════════════════════════════════════════════════════════════
        # Validate preprocessing output
        # ═══════════════════════════════════════════════════════════════════
        # Check for NaN/Inf introduced by preprocessing (e.g., StandardScaler with zero variance)
        # VAR is HIGHLY sensitive: k² × p parameters + covariance matrices make it more fragile
        if train_processed.isna().any().any():
            raise ValueError(
                "Preprocessing resulted in NaN values. "
                "This may be caused by zero-variance features in StandardScaler or similar transformations. "
                "VAR models are particularly sensitive to preprocessing issues "
                "due to multivariate covariance estimation."
            )
        if not np.isfinite(train_processed.values).all():
            raise ValueError(
                "Preprocessing resulted in non-finite values (Inf/-Inf). "
                "Check for numerical instabilities in transformations. "
                "VAR models require all variables to be well-scaled due to matrix inversions."
            )

        endog_data = train_processed[self.target_columns]
        exog_data = train_processed[exog_series.columns] if exog_series is not None else None

        # Remember processed exogenous column names for strict validation at prediction time
        if exog_data is not None:
            self.fitted_exog_names = exog_data.columns.tolist()
        else:
            self.fitted_exog_names = None

        # Cache preprocessed training data for _apply_window_context().
        # Stored in transformed space so that apply() receives data in the
        # same domain the Kalman filter was originally fitted on.
        # Note: remove_data_after_fit only strips data from the statsmodels
        # result object; this local cache is unaffected.
        self._train_endog_processed = endog_data
        self._train_exog_processed = exog_data

        self.use_varmax = exog_series is not None
        
        try:
            # ═══════════════════════════════════════════════════════════════════
            # OPTIONAL: IC-based automatic lag order selection
            # ═══════════════════════════════════════════════════════════════════
            # When ic is set, treat max_lags as the upper bound and find the lag
            # order 1..max_lags that minimises the chosen information criterion.
            # Each candidate is fitted with disp=False; failures are skipped.
            if not self.use_varmax:
                # FAST OLS PATH
                model = VAR(endog_data)
                selected_lag = self.max_lags
                if self.ic is not None:
                    selection = model.select_order(maxlags=self.max_lags)
                    selected_lag = selection.selected_orders[self.ic]
                self.model = model.fit(maxlags=selected_lag, trend=self.trend)
                self._fitted_lag = selected_lag
            else:
                # EXISTING VARMAX PATH
                selected_lag = self.max_lags
                if self.ic is not None:
                    best_ic_v, best_lag = float("inf"), 1
                    for lag in range(1, self.max_lags + 1):
                        try:
                            cand = VARMAX(endog_data, exog=exog_data, order=(lag, 0), trend=self.trend).fit(disp=False, maxiter=self.maxiter)
                            v = getattr(cand, self.ic)
                            if v < best_ic_v: best_ic_v, best_lag = v, lag
                        except: continue
                    selected_lag = best_lag
                
                model = VARMAX(endog_data, exog=exog_data, order=(selected_lag, 0), trend=self.trend,
                               error_cov_type=self.error_cov_type, enforce_stationarity=self.enforce_stationarity,
                               enforce_invertibility=self.enforce_invertibility)
                self.model = model.fit(disp=False, method=self.method, maxiter=self.maxiter)
                self._fitted_lag = selected_lag

            self.last_fit_timestamp = combined_train_data.index[-1]
            self.fitted = True

            # ═══════════════════════════════════════════════════════════════════
            # MEMORY OPTIMIZATION: Optional data removal after fit
            # ═══════════════════════════════════════════════════════════════════
            # In HPO loops with thousands of trials, VARMAXResultsWrapper objects
            # can accumulate significant memory (training data copies, residuals, covariance matrices)
            # VAR is MORE memory-intensive than ARIMA due to k² covariance matrices
            # Calling remove_data() frees this memory at the cost of losing some diagnostic capabilities
            if self.remove_data_after_fit and self.use_varmax:
                self.model.remove_data()
                logger.debug("Removed training data from fitted model to save memory")

            return 0.0, {}  # Statistical models don't have validation loss or training history

        except ValueError as e:
            # Specific handling for parameter/data validation errors
            logger.warning(f"VARMAX fitting failed due to invalid parameters or data: {str(e)}")
            self.fitted = False
            return float('inf'), {}  # Return infinite loss on failure
        except np.linalg.LinAlgError as e:
            # Specific handling for matrix singularity/conditioning issues
            logger.warning(f"VARMAX fitting failed due to linear algebra error (singular matrix?): {str(e)}")
            self.fitted = False
            return float('inf'), {}  # Return infinite loss on failure
        except RuntimeError as e:
            # Specific handling for convergence failures
            logger.warning(f"VARMAX fitting failed due to convergence issues: {str(e)}")
            self.fitted = False
            return float('inf'), {}  # Return infinite loss on failure

    def _validate_prediction_exog(self, exog_pred: Optional[pd.DataFrame], steps: int) -> None:
        """
        Validate that exogenous variables provided at prediction time
        are consistent with those used during training.
        """
        # Case 1: model trained WITH exog, but none (or empty) provided
        if self.fitted_exog_names is not None and (exog_pred is None or exog_pred.empty):
            raise ValueError(
                "Model was trained with exogenous variables but none were provided for prediction."
            )

        # Case 2: model trained WITHOUT exog, but user provided some
        if self.fitted_exog_names is None and exog_pred is not None and not exog_pred.empty:
            logger.warning(
                "Exogenous variables were provided for prediction, but the model was trained without them. "
                "They will be ignored."
            )
            return

        # Case 3: both present -> full strict validation
        if self.fitted_exog_names is not None and exog_pred is not None:
            pred_cols = exog_pred.columns.tolist()

            # 3a. Check number of columns
            if len(pred_cols) != len(self.fitted_exog_names):
                raise ValueError(
                    f"Unexpected number of exogenous columns after preprocessing. "
                    f"Expected {len(self.fitted_exog_names)}, got {len(pred_cols)}."
                )

            # 3b. Check column names and exact ORDER
            if pred_cols != self.fitted_exog_names:
                missing = set(self.fitted_exog_names) - set(pred_cols)
                extra = set(pred_cols) - set(self.fitted_exog_names)
                msg = (
                    f"Exogenous feature mismatch.\n"
                    f"Expected: {self.fitted_exog_names}\nGot: {pred_cols}"
                )

                if missing:
                    msg += f"\nMissing: {missing}"
                if extra:
                    msg += f"\nExtra: {extra}"
                raise ValueError(msg)

            # 3c. Check horizon length
            if len(exog_pred) != steps:
                raise ValueError(
                    f"Prediction exogenous data length mismatch: expected {steps}, got {len(exog_pred)}."
                )

    def predict(self, future_exog: Optional[pd.DataFrame] = None, forecast_steps: Optional[int] = None) -> pd.DataFrame:
        """
        Generate predictions for the specified horizon.

        For VAR models, predictions are based on the fitted model's history, and input_data is ignored.

        Args:
            input_data: Input data (ignored for VAR). Defaults to None.
            forecast_steps: Number of steps to forecast. Defaults to self.forecast_steps.

        Returns:
            Predicted values in the original scale.

        Raises:
            ValueError: If model is not fitted or forecast_steps is invalid.
            RuntimeError: If prediction fails.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting.")

        steps = forecast_steps if forecast_steps is not None else self.forecast_steps

        exog_pred_proc = self.preprocessor.transform(future_exog,
                                                     allow_subset=True) if future_exog is not None else None

        # Validate alignment of exogenous variables (if any)
        self._validate_prediction_exog(exog_pred_proc, steps)

        try:
            # If model was trained with exog, pass processed exog to VARMAX.
            # If not, ensure exog=None even if user supplied something (already warned).
            exog_arg = exog_pred_proc if self.fitted_exog_names is not None else None

            # Re-anchor the Kalman filter state to the recent context window
            # defined by window_size.  Falls back to full-history state when
            # window_size is unset or training data is too short.

            if self.use_varmax:
                results = self._apply_window_context()
                pred_obj = results.get_forecast(steps=steps, exog=exog_arg)
                predictions_proc = pred_obj.predicted_mean
            else:
                # OLS forecast using history context from the dataset
                context = self._train_endog_processed
                forecast = self.model.forecast(y=context.values[-self._fitted_lag:], steps=steps)
                predictions_proc = pd.DataFrame(forecast, columns=self.target_columns)

            
            # ═══════════════════════════════════════════════════════════════════
            # Validate numerical stability of predictions
            # ═══════════════════════════════════════════════════════════════════
            # VARMAX can produce NaN/Inf when:
            # - Model is near non-stationarity/non-invertibility
            # - Covariance matrices are near-singular
            # - Kalman filter encounters numerical precision issues
            # VAR is MORE sensitive than ARIMA due to multivariate Kalman filter
            predictions_proc_values = predictions_proc.values
            if not np.isfinite(predictions_proc_values).all():
                nan_count = np.isnan(predictions_proc_values).sum()
                inf_count = np.isinf(predictions_proc_values).sum()
                logger.error(
                    f"VARMAX prediction produced {nan_count} NaN and {inf_count} Inf values. "
                    f"Model may be misspecified or numerically unstable."
                )
                raise RuntimeError(
                    f"VARMAX prediction diverged: {nan_count} NaN, {inf_count} Inf values detected. "
                    f"This typically indicates model misspecification, parameters near unit roots, "
                    f"singular covariance matrices, or numerical instability in the Kalman filter."
                )

            # Ensure we work with a DataFrame containing ONLY target columns
            # (VARMAX returns a DataFrame with endogenous variables as columns).
            # OPTIMIZATION P3: Removed unnecessary .copy() - slicing already creates a copy
            pred_df = predictions_proc[self.target_columns]

            # Clean inverse transform: we only inverse-transform the targets.
            # Preprocessor is expected to handle subset inverse for target groups.
            predictions_original_df = self.preprocessor.inverse_transforms(
                pred_df, start_after=self.last_fit_timestamp
            )
            return predictions_original_df

        except ValueError as e:
            # Specific handling for parameter/data validation errors
            logger.error(f"VARMAX prediction failed due to invalid parameters or data: {str(e)}", exc_info=True)
            raise ValueError(f"VARMAX prediction failed: {str(e)}")
        except RuntimeError as e:
            # Specific handling for numerical stability issues (already logged above)
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"VARMAX prediction failed with unexpected error: {str(e)}", exc_info=True)
            raise RuntimeError(f"VARMAX prediction failed: {str(e)}")

    def get_valid_params(self) -> Set[str]:
        """
        Get the set of valid parameter names for the VAR model.

        Returns:
            Set of valid parameter names.
        """
        return {
            "max_lags", "ic", "trend", "error_cov_type", "maxiter", "window_size",
            "preprocessing", "n_trials", "enforce_stationarity", "enforce_invertibility",
            "method", "remove_data_after_fit"
        }

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
        elif length < 300:
            return "small"
        elif length < 1000:
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
        Enhanced VAR priors with multivariate and dataset awareness.

        Generates smart starting configurations adapted to:
        - Number of variables (affects parameter explosion)
        - Series length (critical for VAR due to many parameters)
        - IC criterion selection (AIC vs BIC tradeoff)

        VAR models have parameter count = k² × p where:
        - k = number of variables
        - p = lag order

        For k=5, p=3: 75 parameters! Very sensitive to overfitting.

        Args:
            param_space: Parameters being optimized
            fixed_params: Fixed parameters from config
            dataset: Optional TimeSeriesDataset for metadata extraction

        Returns:
            List of 3-5 prior configurations (conservative → moderate)

        Strategy:
            1. Start with low lag orders (p=1,2)
            2. Consider number of variables when setting max lag
            3. Prefer BIC for small datasets (penalizes complexity more)
            4. Only explore higher orders with sufficient data
        """
        priors = []

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Extract Dataset Metadata
        # ═══════════════════════════════════════════════════════════════

        series_length = None
        size_category = "medium"
        num_variables = None

        if dataset:
            # 1. Get series length
            if hasattr(dataset, 'series') and dataset.series is not None:
                series_length = len(dataset.series)

                # Categorize using helper method
                size_category = self._categorize_dataset_size(series_length)

            # 2. Get number of variables (VAR-specific!)
            if hasattr(dataset, 'data') and dataset.data is not None:
                if hasattr(dataset.data, 'shape'):
                    num_variables = dataset.data.shape[1] if len(dataset.data.shape) > 1 else 1

        # If not extracted from dataset, use model's num_features
        if num_variables is None:
            num_variables = getattr(self, 'num_features', 1)

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Determine Safe Maximum Lag Order
        # ═══════════════════════════════════════════════════════════════

        # Rule: Need at least k² × p × 5 observations (very rough heuristic)
        # For safety, we're more conservative

        if size_category == "tiny":
            max_safe_lag = 1
        elif size_category == "small":
            max_safe_lag = 2
        elif size_category == "medium":
            max_safe_lag = 4 if num_variables <= 3 else 3
        else:  # large
            max_safe_lag = 8 if num_variables <= 3 else 5

        # Additional constraint: more variables → lower max lag
        if num_variables >= 10:
            max_safe_lag = min(max_safe_lag, 3)
        elif num_variables >= 5:
            max_safe_lag = min(max_safe_lag, 4)

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Select Information Criterion
        # ═══════════════════════════════════════════════════════════════

        # BIC penalizes complexity more → better for small datasets
        # AIC allows more complexity → better for large datasets

        if size_category in ["tiny", "small"]:
            preferred_ic = "bic"
            alternative_ic = "aic"
        else:
            preferred_ic = "aic"
            alternative_ic = "bic"

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Generate Priors
        # ═══════════════════════════════════════════════════════════════

        # Prior 1: VAR(1) with preferred IC
        # Simplest stable model, good baseline
        prior1 = {"max_lags": 1}
        if "ic" in param_space:
            prior1["ic"] = preferred_ic
        priors.append(prior1)

        # Prior 2: VAR(2) with preferred IC (if safe)
        # Captures short-term dynamics
        if max_safe_lag >= 2:
            prior2 = {"max_lags": 2}
            if "ic" in param_space:
                prior2["ic"] = preferred_ic
            priors.append(prior2)

        # Prior 3: VAR(1) with alternative IC
        # Same lag but different criterion
        if "ic" in param_space:
            prior3 = {"max_lags": 1, "ic": alternative_ic}
            priors.append(prior3)

        # Prior 4: Moderate lag (3 or 4) if safe
        if max_safe_lag >= 3:
            moderate_lag = min(3, max_safe_lag)
            prior4 = {"max_lags": moderate_lag}
            if "ic" in param_space:
                prior4["ic"] = preferred_ic
            priors.append(prior4)

        # Prior 5: Higher lag (5+) only for large datasets
        if max_safe_lag >= 5 and size_category == "large":
            higher_lag = min(5, max_safe_lag)
            prior5 = {"max_lags": higher_lag}
            if "ic" in param_space:
                prior5["ic"] = "aic"  # AIC for more complex model
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
        Validate VAR parameter combination to avoid parameter explosion.

        VAR models have k² * p parameters, where:
        - k = number of variables (num_features)
        - p = number of lags (maxlags)

        This can easily lead to:
        - Over-parameterization (more params than samples)
        - Singular matrices
        - Numerical instability

        Args:
            params: Proposed hyperparameter combination

        Returns:
            True if valid, False if should be pruned
        """
        max_lags = params.get("max_lags", 1)
        trend = params.get("trend", "c")
        ic = params.get("ic", None)

        k = self.num_features

        # ═══════════════════════════════════════════════════════════════
        # Get series length
        # ═══════════════════════════════════════════════════════════════
        series_length = getattr(self, '_series_length', None)
        if series_length is None and hasattr(self, 'dataset'):
            series_length = len(self.dataset.series) if hasattr(self.dataset, 'series') else None

        if series_length is None:
            # Can't validate without knowing T
            return True

        T = series_length

        # ═══════════════════════════════════════════════════════════════
        # Rule 1: Parameter explosion check
        # VAR(p) has k² * p parameters
        # Rule of thumb: need T >= 5 * (k² * p) for stable estimation
        # ═══════════════════════════════════════════════════════════════
        num_params = (k ** 2) * max_lags
        min_required_T = 5 * num_params

        if T < min_required_T:
            logger.debug(
                f"[VAR] Rejected: T={T} < 5*k²*p={min_required_T} "
                f"(k={k}, p={max_lags})"
            )
            return False

        # ═══════════════════════════════════════════════════════════════
        # Rule 2: High lag + trend="ct" on small datasets
        # Linear trend adds k params, can cause singularity
        # ═══════════════════════════════════════════════════════════════
        if trend == "ct" and max_lags >= 4 and T < 200:
            logger.debug(
                f"[VAR] Rejected: trend='ct' + max_lags={max_lags} "
                f"on short series (T={T})"
            )
            return False

        # ═══════════════════════════════════════════════════════════════
        # Rule 3: Very high lag on small T
        # Even if T >= 5*k²*p, max_lags > T/10 is risky
        # ═══════════════════════════════════════════════════════════════
        if max_lags > T / 10:
            logger.debug(
                f"[VAR] Rejected: max_lags={max_lags} > T/10={T / 10:.1f}"
            )
            return False

        # ═══════════════════════════════════════════════════════════════
        # Rule 4: ic=None requires max_lags to be reasonable
        # Without IC selection, we use exact lag p
        # ═══════════════════════════════════════════════════════════════
        if ic is None and max_lags > 8:
            logger.debug(
                f"[VAR] Warning: ic=None with max_lags={max_lags} > 8 "
                f"(no automatic lag selection)"
            )
            # Don't reject, but risky

        return True

    def filter_search_space(
            self,
            param_space: Dict[str, Any],
            fixed_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adjusts the hyperparameter search space to prevent model instability
        based on window_size and feature count.
        """
        space = dict(param_space)
        T = self.window_size
        k = self.num_features

        # Determine model mode (VAR vs VARMAX)
        is_varmax = "max_ar_lags" in space or "max_ma_lags" in space
        mode_label = "VARMAX" if is_varmax else "VAR"

        # Calculate safety threshold (75% of window capacity)
        # Rule: T > 1.25 * (k * p) ensures enough degrees of freedom
        max_safe_p = int((T * 0.75) // k)
        limit = max(1, max_safe_p)

        # Select keys to monitor based on the detected mode
        keys_to_check = ["max_ar_lags", "max_ma_lags"] if is_varmax else ["max_lags"]

        for key in keys_to_check:
            if key in space:
                val = space[key]
                # Extract the maximum value from different possible config formats
                requested_max = val["max"] if isinstance(val, dict) else (max(val) if isinstance(val, list) else val)

                if requested_max > limit:
                    # Apply the limit to the parameter space
                    if isinstance(space[key], dict):
                        space[key]["max"] = limit
                    elif isinstance(space[key], list):
                        space[key] = [l for l in space[key] if l <= limit]
                    else:
                        space[key] = limit

                    # Notify the user about the dynamic adjustment
                    logger.warning(
                        f"[{mode_label}] Adjusted {key}: {requested_max} -> {limit}. "
                        f"Reason: window_size={T} is insufficient for k={k} features with p={requested_max}."
                    )

        return space
