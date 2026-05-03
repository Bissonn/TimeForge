"""Base module for ARIMA/SARIMA time series forecasting models.

This module defines the ARIMABaseForecaster class, which implements ARIMA/SARIMA models
using the statsmodels library, extending the StatTSForecaster base class.
"""

import logging
from typing import Any, Dict, List, Optional, Set
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from core.context import RunContext
from models.base import StatTSForecaster
from utils.preprocessor import Preprocessor
from utils.dataset import TimeSeriesDataset

logger = logging.getLogger(__name__)

class ARIMABaseForecaster(StatTSForecaster):
    """Base class for ARIMA/SARIMA time series forecasting models."""

    is_univariate: bool = True

    def __init__(
            self,
            model_params: Dict[str, Any],
            num_features: int,
            forecast_steps: int,
            window_size: int,
            dataset: TimeSeriesDataset,
            run_context: RunContext,
            seasonal: bool = False,
            **kwargs
    ) -> None:
        """
        Initialize the ARIMA/SARIMA forecaster.

        Args:
            model_params: Model-specific parameters (e.g., p, d, q for ARIMA; P, D, Q, seasonal_period for SARIMA).
            num_features: Number of features in the time series data (must be 1 for univariate models).
            forecast_steps: Number of steps to forecast.
            window_size: The look-back window size (used for validation).
            dataset: Instance of TimeSeriesDataset.
            run_context: RunContext
            seasonal: Whether to use SARIMA (True) or ARIMA (False). Defaults to False.
            **kwargs: Additional keyword arguments passed to the base class, including `input_size`.

        Raises:
            ValueError: If num_features is not 1 for univariate models or if model parameters are invalid.
        """
        if self.is_univariate and num_features != 1:
            raise ValueError("Univariate ARIMA/SARIMA models require num_features=1.")
        super().__init__(
            model_params=model_params,
            num_features=num_features,
            forecast_steps=forecast_steps,
            window_size=window_size,
            dataset=dataset,
            run_context=run_context,
            **kwargs
        )
        self.seasonal = seasonal
        self.target_columns: Optional[List[str]] = None
        self.fitted_exog_names: Optional[List[str]] = None

        # Cached preprocessed training data used by _apply_window_context().
        # Populated in fit(), consumed in predict(). Stored in transformed space
        # (same space the Kalman filter was trained in).
        self._train_endog_processed: Optional[pd.DataFrame] = None
        self._train_exog_processed: Optional[pd.DataFrame] = None

        # ═══════════════════════════════════════════════════════════════════
        # Configurable stationarity/invertibility enforcement
        # ═══════════════════════════════════════════════════════════════════
        # Allow users to control SARIMAX constraints from config
        # enforce_stationarity=True can cause fitting failures for naturally non-stationary series
        self.enforce_stationarity = model_params.get("enforce_stationarity", True)
        self.enforce_invertibility = model_params.get("enforce_invertibility", True)

        # Deterministic trend component passed to SARIMAX.
        # None  — let SARIMAX decide (no explicit term; default statsmodels behaviour)
        # "n"   — no deterministic component
        # "c"   — constant (intercept)
        # "t"   — linear time trend
        # "ct"  — constant + linear time trend
        self.trend: Optional[str] = model_params.get("trend", None)

        # ═══════════════════════════════════════════════════════════════════
        # OPTIMIZATION: Configurable solver and iteration limits
        # ═══════════════════════════════════════════════════════════════════
        # Default solver 'lbfgs' is accurate but slow
        # Alternative 'css-mle' (Conditional Sum of Squares -> MLE) is faster for simple models
        # maxiter controls optimization iterations (default 50, can be reduced for speed)
        self.method = model_params.get("method", "lbfgs")
        self.maxiter = model_params.get("maxiter", 50)

        # Memory management: remove_data after fit to reduce memory footprint in HPO
        self.remove_data_after_fit = model_params.get("remove_data_after_fit", False)

        self._validate_model_params()
        logger.info(
            f"Initialized {self.__class__.__name__} with seasonal={seasonal}, "
            f"enforce_stationarity={self.enforce_stationarity}, "
            f"enforce_invertibility={self.enforce_invertibility}, "
            f"method={self.method}, maxiter={self.maxiter}"
        )

    def _validate_model_params(self) -> None:
        """
        Validate ARIMA/SARIMA model parameters.

        Raises:
            ValueError: If model parameters are invalid (e.g., negative orders).
        """
        required_params = ["p", "d", "q"]
        seasonal_params = ["P", "D", "Q", "seasonal_period"] if self.seasonal else []

        for param in required_params + seasonal_params:
            if param not in self.model_params:
                raise ValueError(f"Missing required parameter: {param}")
            if not isinstance(self.model_params[param], int) or self.model_params[param] < 0:
                raise ValueError(f"Parameter {param} must be a non-negative integer.")

        if self.seasonal and self.model_params["seasonal_period"] <= 0:
            raise ValueError("seasonal_period must be positive for SARIMA models.")

        _valid_trend = {"n", "c", "t", "ct"}
        if self.trend is not None and self.trend not in _valid_trend:
            raise ValueError(f"trend must be one of {sorted(_valid_trend)} or None. Got: '{self.trend}'.")

    def _compute_arima_warmup(self) -> int:
        """
        Compute additional points needed on top of window_size to properly
        initialize the Kalman filter before the effective forecast context begins.

        Two phases are accounted for:

        Phase 1 — diffuse initialization (exact):
            The Kalman filter cannot determine the non-stationary baseline state
            from a cold start.  statsmodels uses exact diffuse initialization for
            the first (d + D*s) observations, during which log-likelihood
            contributions are excluded and state uncertainty is effectively
            infinite.  These points are consumed from the window but do not
            contribute meaningful context.

        Phase 2 — ARMA state covariance convergence (heuristic):
            After the diffuse phase the filter's state-error covariance P_t still
            needs several steps to converge.  The rate depends on the roots of the
            AR/MA polynomials (roots near the unit circle → slow convergence), so
            no closed-form formula exists.  We use max_ar_ma_lag * 2 as a
            conservative but practical buffer.

        Returns:
            Total number of warm-up points to prepend to the user-specified
            window_size.
        """
        d = self.model_params.get("d", 0)
        D = self.model_params.get("D", 0) if self.seasonal else 0
        s = self.model_params.get("seasonal_period", 0) if self.seasonal else 0

        p = self.model_params.get("p", 0)
        q = self.model_params.get("q", 0)
        P = self.model_params.get("P", 0) if self.seasonal else 0
        Q = self.model_params.get("Q", 0) if self.seasonal else 0

        # Phase 1: exact diffuse initialization length
        diffuse = d + D * s

        # Phase 2: heuristic convergence buffer based on the longest AR/MA lag
        seasonal_ar_lag = P * s if (P > 0 and s > 0) else 0
        seasonal_ma_lag = Q * s if (Q > 0 and s > 0) else 0
        max_lag = max(p, q, seasonal_ar_lag, seasonal_ma_lag)
        convergence_buffer = max_lag * 2

        return diffuse + convergence_buffer

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
            effective = window_size + _compute_arima_warmup()

        The warm-up prefix ensures that by the time the filter reaches the last
        window_size points its state covariance has stabilized, so the effective
        context seen by get_forecast() corresponds to exactly window_size
        meaningful observations.

        Falls back to self.model (full-history state) with a warning when:
        - window_size is None (feature disabled)
        - _train_endog_processed is not cached (fit() not yet called)
        - available training data <= effective window (nothing to trim)

        Returns:
            SARIMAXResults — either the result of apply() or the original
            self.model, depending on applicability.
        """
        if self.window_size is None or self._train_endog_processed is None:
            return self.model

        warmup = self._compute_arima_warmup()
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

    def _validate_prediction_exog(self, exog_pred: Optional[pd.DataFrame], steps: int) -> None:
        """
        Validates that the prediction exogenous variables match the training ones strictly.
        """
        # === Case 1: was trained with exog, but None provided ===
        if self.fitted_exog_names is not None and (exog_pred is None or exog_pred.empty):
            raise ValueError(
                "Model was trained with exogenous variables but none were provided for prediction."
            )

        # === Case 2: trained without exog, user provided some ===
        if self.fitted_exog_names is None and exog_pred is not None and not exog_pred.empty:
            logger.warning(
                "Exogenous variables provided for prediction, but model was trained without them. "
                "They will be ignored."
            )
            return

        # === Case 3: full validation ===
        if self.fitted_exog_names is not None and exog_pred is not None:
            pred_cols = exog_pred.columns.tolist()

            # Check number of columns after preprocessing
            if len(pred_cols) != len(self.fitted_exog_names):
                raise ValueError(
                    f"Unexpected number of exogenous columns after preprocessing. "
                    f"Expected {len(self.fitted_exog_names)}, got {len(pred_cols)}."
                )

            # Strict name & order validation
            if pred_cols != self.fitted_exog_names:
                missing = set(self.fitted_exog_names) - set(pred_cols)
                extra = set(pred_cols) - set(self.fitted_exog_names)

                raise ValueError(
                    f"Exogenous feature mismatch.\n"
                    f"Expected columns (ordered): {self.fitted_exog_names}\n"
                    f"Got: {pred_cols}\n"
                    f"{'Missing: ' + str(missing) if missing else ''}\n"
                    f"{'Extra: ' + str(extra) if extra else ''}"
                )

            # Check horizon length
            if len(exog_pred) != steps:
                raise ValueError(
                    f"Prediction exogenous length mismatch. "
                    f"Expected {steps}, got {len(exog_pred)}."
                )

    def fit(
        self,
        train_series: pd.DataFrame,
        exog_series: Optional[pd.DataFrame] = None,
        dataset: Optional[TimeSeriesDataset] = None,
        is_final_fit: bool = False,
        **kwargs
    ) -> tuple:
        """
        Fit the SARIMAX model to the training data.

        Args:
            train_series: Training data for the endogenous variable (target).
            exog_series: Optional DataFrame of exogenous variables for training.
            dataset: Dataset object (ignored by this model, for signature compatibility).
            is_final_fit: Whether this is final fit (ignored by this model).
            **kwargs: Additional arguments (ignored, for signature compatibility).

        Returns:
            Tuple of (validation_loss, training_history). For statistical models, returns (0.0, {}).

        Raises:
            ValueError: If train_series is invalid or model parameters are invalid.
            RuntimeError: If SARIMAX model fitting fails.
        """
        if self.is_univariate and train_series.shape[1] != 1:
            raise ValueError("Univariate ARIMA/SARIMA models require a single-column train_series.")

        # === SEASONALITY CHECK (SARIMA Safety) ===
        # Ensure we have enough data to estimate seasonal components.
        # Heuristic: We need at least 2 full cycles (2 * m).
        if self.seasonal:
            m = self.model_params.get("seasonal_period", 0)
            if m > 1 and len(train_series) < 1.5 * m:
                raise ValueError(
                    f"Dataset too short for seasonal period m={m}. "
                    f"Required n >= 2*m ({2 * m}), got n={len(train_series)}."
                )

        self.target_columns = train_series.columns.tolist()

        try:
            combined_train_data = (
                pd.concat([train_series, exog_series], axis=1) if exog_series is not None
                else train_series
            )
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
            if train_processed.isna().any().any():
                raise ValueError(
                    "Preprocessing resulted in NaN values. "
                    "This may be caused by zero-variance features in StandardScaler or similar transformations."
                )
            if not np.isfinite(train_processed.values).all():
                raise ValueError(
                    "Preprocessing resulted in non-finite values (Inf/-Inf). "
                    "Check for numerical instabilities in transformations."
                )

            endog_data = train_processed[train_series.columns]
            exog_data = train_processed[exog_series.columns] if exog_series is not None else None

            # Save fitted exog names for validation
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

            # ═══════════════════════════════════════════════════════════════════
            # OPTIMIZATION P5: Cache column ordering for inverse transform
            # ═══════════════════════════════════════════════════════════════════
            # Pre-compute column order for inverse transform to avoid O(n*m) list comprehension
            # in every predict() call
            all_columns = self.target_columns + (self.fitted_exog_names if self.fitted_exog_names else [])
            self._inverse_transform_column_order = [
                c for c in self.preprocessor._full_raw_data_context.columns
                if c in all_columns
            ]

            order = (
                self.model_params.get("p", 1),
                self.model_params.get("d", 1),
                self.model_params.get("q", 1),
            )
            seasonal_order = (
                self.model_params.get("P", 0),
                self.model_params.get("D", 0),
                self.model_params.get("Q", 0),
                self.model_params.get("seasonal_period", 0),
            ) if self.seasonal else (0, 0, 0, 0)

            logger.info(f"Fitting SARIMAX with order={order}, seasonal_order={seasonal_order}")

            model = SARIMAX(
                endog=endog_data,
                exog=exog_data,
                order=order,
                seasonal_order=seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            )

            # ═══════════════════════════════════════════════════════════════════
            # FIT WITH CONVERGENCE VALIDATION
            # ═══════════════════════════════════════════════════════════════════
            # Suppress ConvergenceWarning — we check convergence programmatically
            # via mle_retvals instead of relying on warning propagation.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning,
                                        message=".*Maximum Likelihood optimization failed to converge.*")
                self.model = model.fit(
                    disp=False,
                    method=self.method,
                    maxiter=self.maxiter
                )

            # ═══════════════════════════════════════════════════════════════════
            # POST-FIT CONVERGENCE & STABILITY CHECKS
            # ═══════════════════════════════════════════════════════════════════
            # Two tiers:
            #   hard_issues → reject the fit (return inf), model is unusable
            #   soft_issues → log warning but accept the fit (model may work)
            hard_issues = []
            soft_issues = []

            # ── Hard Check 1: MLE convergence flag ──
            mle_retvals = getattr(self.model, 'mle_retvals', None)
            if mle_retvals is not None:
                converged = mle_retvals.get('converged', True)
                warnflag = mle_retvals.get('warnflag', 0)
                if not converged or warnflag != 0:
                    hard_issues.append(
                        f"MLE did not converge (converged={converged}, warnflag={warnflag})"
                    )

            # ── Hard Check 2: NaN/Inf in estimated parameters ──
            params = self.model.params
            if not np.isfinite(params).all():
                nan_count = np.isnan(params).sum()
                inf_count = np.isinf(params).sum()
                hard_issues.append(
                    f"Non-finite parameters: {nan_count} NaN, {inf_count} Inf"
                )

            # ── Hard Check 3: Non-finite log-likelihood ──
            llf = self.model.llf
            if not np.isfinite(llf):
                hard_issues.append(f"Log-likelihood is non-finite: {llf}")

            # ── Soft Check 4: Covariance matrix health ──
            # cov_params() can fail on simple models with small data even when
            # the point estimates are perfectly usable (e.g. AR(1) on 100 points
            # of near-white noise).  Demoted to warning-only.
            try:
                cov = self.model.cov_params()
                if not np.isfinite(cov).all():
                    soft_issues.append("Covariance matrix contains non-finite values")
                elif np.linalg.cond(cov) > 1e12:
                    soft_issues.append(
                        f"Covariance matrix near-singular (cond={np.linalg.cond(cov):.2e})"
                    )
            except (np.linalg.LinAlgError, ValueError):
                soft_issues.append("Covariance matrix computation failed (p-values unavailable)")

            # ═══════════════════════════════════════════════════════════════
            # REJECT on hard issues, WARN on soft issues
            # ═══════════════════════════════════════════════════════════════
            if hard_issues:
                issue_str = "; ".join(hard_issues)
                logger.warning(
                    f"[{self.__class__.__name__}] Fit rejected (unstable): {issue_str}. "
                    f"order={order}, seasonal_order={seasonal_order}"
                )
                self.fitted = False
                return float('inf'), {}

            if soft_issues:
                issue_str = "; ".join(soft_issues)
                logger.info(
                    f"[{self.__class__.__name__}] Fit accepted with warnings: {issue_str}. "
                    f"order={order}, seasonal_order={seasonal_order}"
                )
            self.last_fit_timestamp = combined_train_data.index[-1]
            self.fitted = True

            # ═══════════════════════════════════════════════════════════════════
            # MEMORY OPTIMIZATION: Optional data removal after fit
            # ═══════════════════════════════════════════════════════════════════
            # In HPO loops with thousands of trials, ARIMAResultsWrapper objects
            # can accumulate significant memory (training data copies, residuals, covariance matrices)
            # Calling remove_data() frees this memory at the cost of losing some diagnostic capabilities
            if self.remove_data_after_fit:
                self.model.remove_data()
                logger.debug("Removed training data from fitted model to save memory")

            logger.info("SARIMAX model fitted successfully")
            return 0.0, {}  # Statistical models don't have validation loss or training history

        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            logger.warning(f"SARIMAX fitting failed (skipping fold): {str(e)}")
            self.fitted = False
            return float('inf'), {}  # Return infinite loss on failure

    def predict(self, future_exog: Optional[pd.DataFrame] = None, forecast_steps: Optional[int] = None) -> pd.DataFrame:
        """
        Generate predictions for the specified horizon.

        For ARIMA/SARIMA models, predictions are based on the fitted model's history, and input_data is ignored.

        Args:
            future_exog: DataFrame of future values for exogenous variables.
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

        # Transform future exogenous data using the already fitted preprocessor, allowing subsets (no targets)
        exog_pred_proc = self.preprocessor.transform(
            future_exog, allow_subset=True
        ) if future_exog is not None else None

        # VALIDATE ALIGNMENT
        self._validate_prediction_exog(exog_pred_proc, steps)

        # If the model was trained WITHOUT exogenous variables, we must ignore any
        # future_exog that was accidentally passed. The validator already emitted
        # a warning in this case; here we ensure the rest of the logic treats
        # exog as absent, so we don't try to align predictions with a wrong index.

        if self.fitted_exog_names is None:
            exog_pred_proc = None

        try:
            exog_arg = exog_pred_proc if self.fitted_exog_names is not None else None

            # Re-anchor the Kalman filter state to the recent context window
            # defined by window_size.  Falls back to full-history state when
            # window_size is unset or training data is too short.
            results = self._apply_window_context()
            pred_obj = results.get_forecast(steps=steps, exog=exog_arg)
            predictions_proc = pred_obj.predicted_mean
            predictions_proc_np = predictions_proc.values.reshape(-1, self.num_features)

            # ═══════════════════════════════════════════════════════════════════
            # Validate numerical stability of predictions
            # ═══════════════════════════════════════════════════════════════════
            # SARIMAX can produce NaN/Inf when model is near non-stationarity/non-invertibility
            # or when there are numerical precision issues in Kalman filter
            if not np.isfinite(predictions_proc_np).all():
                nan_count = np.isnan(predictions_proc_np).sum()
                inf_count = np.isinf(predictions_proc_np).sum()
                logger.error(
                    f"SARIMAX prediction produced {nan_count} NaN and {inf_count} Inf values. "
                    f"Model may be misspecified or numerically unstable."
                )
                raise RuntimeError(
                    f"SARIMAX prediction diverged: {nan_count} NaN, {inf_count} Inf values detected. "
                    f"This typically indicates model misspecification, parameters near unit roots, "
                    f"or numerical instability in the Kalman filter."
                )

            # ═══════════════════════════════════════════════════════════════════
            # Unified inverse transform path
            # ═══════════════════════════════════════════════════════════════════
            # Instead of two separate paths (with/without exog), use a unified approach
            # that constructs a DataFrame and handles both cases consistently

            # Step 1: Construct predictions with proper structure
            if exog_pred_proc is not None:
                # Case 1: With exogenous variables
                # Create predictions DataFrame aligned with exog index
                pred_df = pd.DataFrame(
                    predictions_proc_np,
                    columns=self.target_columns,
                    index=exog_pred_proc.index
                )
                # Combine predictions with exog
                combined_pred_df = pd.concat([pred_df, exog_pred_proc], axis=1)

                # Reorder columns using cached ordering (OPTIMIZATION P5)
                # This avoids O(n*m) list comprehension on every prediction
                combined_pred_df = combined_pred_df[self._inverse_transform_column_order]

                # Inverse transform
                predictions_original_df = self.preprocessor.inverse_transforms(
                    combined_pred_df,
                    start_after=self.last_fit_timestamp
                )
                return predictions_original_df[self.target_columns]
            else:
                # Case 2: Without exogenous variables
                # Preprocessor can handle numpy array directly (simpler path)
                return self.preprocessor.inverse_transforms(
                    predictions_proc_np,
                    start_after=self.last_fit_timestamp
                )

        except (ValueError, RuntimeError) as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"SARIMAX prediction failed: {str(e)}")

    def get_valid_params(self) -> Set[str]:
        """
        Get the set of valid parameter names for the ARIMA/SARIMA model.

        Returns:
            Set of valid parameter names.
        """
        valid_params = {"p", "d", "q", "trend", "preprocessing", "n_trials"}
        if self.seasonal:
            valid_params.update({"P", "D", "Q", "seasonal_period"})
        return valid_params

    def _validate_model_specific_inputs(self, *args) -> None:
        """Placeholder for any additional model-specific validation."""
        pass
