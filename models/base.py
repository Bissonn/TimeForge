"""Base module for time series forecasting models.

This module defines abstract base classes for statistical and neural network-based time series forecasting models.
It provides a framework for model initialization, hyperparameter optimization,
data preparation, fitting, and prediction.
"""

import os
import logging
import json
from abc import ABC, abstractmethod
import contextlib
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import torch

from utils.data_utils import create_sliding_window
from utils.dataset import TimeSeriesDataset
from utils.hyperopt.grid_params import generate_grid_params
from utils.hyperopt.random_params import generate_random_params
from utils.metrics import calculate_metrics
from utils.preprocessor import Preprocessor
from monitoring.gradient_monitor import GradientMonitor
from core.context import RunContext
if TYPE_CHECKING:
    from core.context import RunContext

try:
    import optuna
except ImportError:
    optuna = None

logger = logging.getLogger(__name__)


class PastCovariatePolicy(Enum):
    """
    Policy for handling past_covariates during iterative autoregressive prediction.

    Past covariates are features known only in history (not in the future).
    During autoregressive prediction, we need a policy for handling these features
    when predicting beyond the historical window.

    Policies:
        FROZEN: Use the last known window of past_covariates repeatedly (default).
                Safe and stable - prevents distribution shift.
                Example: Last 24h of sensor readings used for all future steps.

        LAST_WINDOW: Similar to FROZEN but updates the window with new predictions.
                     More dynamic but can accumulate errors.

        ZERO: Fill with zeros after history ends.
              Simple but may hurt performance if model depends on these features.

        CUSTOM: User-provided function to generate future values.
                Advanced use case - allows for custom forecasting or simulation.

    Note: This is a policy, not a hard-coded behavior. Models should implement
    the appropriate policy based on their architecture and use case.
    """
    FROZEN = "frozen"           # Repeat last known window (recommended default)
    LAST_WINDOW = "last_window" # Update window with predictions
    ZERO = "zero"               # Fill with zeros
    CUSTOM = "custom"           # User-provided function


@dataclass(frozen=True)
class FeatureLayout:
    """
    Architecture-agnostic feature layout declaration.

    Declares what features exist and their indices, but does NOT dictate
    how models use them. Models validate and interpret the layout based on
    their capabilities.

    Attributes:
        total_features: Total number of columns in the dataset
        target_columns: Names of target features to forecast
        target_size: Number of target features
        past_covariates: Features known only in history (encoder-only)
        future_covariates: Features known in both history and future
        past_covariates_size: Number of past covariates
        future_covariates_size: Number of future covariates
        encoder_feature_idx: Indices of features for encoder input window
        encoder_input_size: Total encoder input size (targets + past + future)
        decoder_input_size: Total decoder input size (targets + future)
    """
    total_features: int

    target_columns: List[str]
    target_size: int

    past_covariates: List[str]
    future_covariates: List[str]

    past_covariates_size: int
    future_covariates_size: int

    encoder_feature_idx: List[int]
    encoder_input_size: int
    decoder_input_size: int

    @property
    def continuous_size(self) -> int:
        """Legacy property for backward compatibility."""
        return self.encoder_input_size

    # ═══════════════════════════════════════════════════════════════════
    # BACKWARD COMPATIBILITY PROPERTIES (deprecated)
    # ═══════════════════════════════════════════════════════════════════

    @property
    def encoder_exog_columns(self) -> List[str]:
        """DEPRECATED: Use past_covariates + future_covariates instead."""
        return self.past_covariates + self.future_covariates

    @property
    def decoder_exog_columns(self) -> List[str]:
        """DEPRECATED: Use future_covariates instead."""
        return self.future_covariates

    @property
    def encoder_exog_size(self) -> int:
        """DEPRECATED: Use past_covariates_size + future_covariates_size instead."""
        return self.past_covariates_size + self.future_covariates_size

    @property
    def decoder_exog_size(self) -> int:
        """DEPRECATED: Use future_covariates_size instead."""
        return self.future_covariates_size

def compute_feature_layout_from_dataset(
    dataset: TimeSeriesDataset,
) -> FeatureLayout:
    """
    Computes feature layout from dataset declarations.

    This is a pure declaration function - it computes indices and sizes
    but does NOT make assumptions about how models will use the features.

    Args:
        dataset: TimeSeriesDataset with declared features

    Returns:
        FeatureLayout with computed indices and sizes

    Raises:
        ValueError: If declared columns are not found in dataset
    """
    target_cols = list(dataset.target_columns)
    past_cov = list(dataset.past_covariates)
    future_cov = list(dataset.future_covariates)

    all_columns = list(dataset.columns)

    # Sanity check: all declared columns must exist in dataset
    for c in target_cols + past_cov + future_cov:
        if c not in all_columns:
            raise ValueError(f"Column '{c}' not found in dataset.columns")

    # Encoder window: targets + past_covariates + future_covariates
    encoder_columns = target_cols + past_cov + future_cov
    encoder_feature_idx = [all_columns.index(c) for c in encoder_columns]

    encoder_input_size = len(encoder_feature_idx)
    # Decoder: targets + future_covariates
    decoder_input_size = len(target_cols) + len(future_cov)

    return FeatureLayout(
        total_features=len(all_columns),

        target_columns=target_cols,
        target_size=len(target_cols),

        past_covariates=past_cov,
        future_covariates=future_cov,

        past_covariates_size=len(past_cov),
        future_covariates_size=len(future_cov),

        encoder_feature_idx=encoder_feature_idx,
        encoder_input_size=encoder_input_size,
        decoder_input_size=decoder_input_size,
    )

class TSForecaster(ABC):
    """Abstract base class for time series forecasting models."""

    # Class attribute indicating if the model is univariate
    is_univariate: bool = False

    def __init__(
            self,
            model_params: Dict[str, Any],
            num_features: int,
            forecast_steps: int,
            window_size: int,
            dataset: TimeSeriesDataset,
            run_context: 'RunContext',
            **kwargs
    ) -> None:
        """
        Initialize the forecaster with model-specific parameters.

        Args:
            model_params: Model-specific parameters from the configuration.
            num_features: Number of target features to be forecasted.
            forecast_steps: Number of steps to forecast (the horizon).
            window_size: The look-back window size for the model.


        Raises:
            ValueError: If model_params is not a dictionary or forecast_steps is not positive.
        """
        # Security: Maximum bounds to prevent resource exhaustion
        MAX_FORECAST_STEPS = 10000  # Maximum forecast horizon
        MAX_WINDOW_SIZE = 50000     # Maximum lookback window
        MAX_NUM_FEATURES = 1000     # Maximum number of features

        if not isinstance(model_params, dict):
            raise ValueError("model_params must be a dictionary.")

        # Validate forecast_steps with bounds
        if not isinstance(forecast_steps, int):
            raise ValueError(f"forecast_steps must be an integer, got {type(forecast_steps).__name__}")
        if not (1 <= forecast_steps <= MAX_FORECAST_STEPS):
            raise ValueError(
                f"forecast_steps must be between 1 and {MAX_FORECAST_STEPS}, got {forecast_steps}"
            )

        # Validate window_size with bounds
        if not isinstance(window_size, int):
            raise ValueError(f"window_size must be an integer, got {type(window_size).__name__}")
        if not (1 <= window_size <= MAX_WINDOW_SIZE):
            raise ValueError(
                f"window_size must be between 1 and {MAX_WINDOW_SIZE}, got {window_size}"
            )

        # Validate num_features with bounds
        if not isinstance(num_features, int):
            raise ValueError(f"num_features must be an integer, got {type(num_features).__name__}")
        if not (1 <= num_features <= MAX_NUM_FEATURES):
            raise ValueError(
                f"num_features must be between 1 and {MAX_NUM_FEATURES}, got {num_features}"
            )

        # Store core parameters
        self.model_params = model_params
        self.model_type = model_params.get("type", "unknown")
        self.num_features = num_features
        self.forecast_steps = forecast_steps
        self.window_size = window_size
        self.run_context = run_context

        self.model = None
        self.fitted = False
        self.last_fit_timestamp = None
        self.preprocessor: Optional[Preprocessor] = None

        # Optimization config
        self.optimize = model_params.get("optimize", False)
        self.optimization_config = model_params.get(
            "optimization",
            {"method": "random", "params": {}})

        self.feature_layout = compute_feature_layout_from_dataset(
            dataset,
        )

        # Format params for readable logging
        params_formatted = json.dumps(model_params, indent=2, default=str)
        logger.info(
            f"Initialized {self.__class__.__name__} (type: {self.model_type})\n"
            f"Parameters:\n{params_formatted}"
        )

    def _get_artifact_path(self, category: str, suffix: str, extension: str) -> Path:
        """
        Generates a consistent file path for artifacts using RunContext.
        """
        # Direct delegation - context is guaranteed to exist
        return self.run_context.get_artifact_path(
            category=category,
            suffix=suffix,
            extension=extension,
            include_window=True,
            include_fold=True
        )

    def evaluate(
            self,
            y_true: pd.DataFrame,
            y_pred: pd.DataFrame,
            dataset: Optional['TimeSeriesDataset'] = None,
            metric_name: str = "mse"
    ) -> float:
        """
        Calculate the configured evaluation metric between true and predicted values.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            dataset: Optional dataset for automatic inverse differencing
            metric_name: The metric to calculate (e.g., 'mse', 'mae').
        Returns:
            The value of the configured metric.

            Returns +inf in the following cases:
              - empty inputs,
              - alignment results in empty frames,
              - non-finite values (NaN/inf) in aligned arrays,
              - metric computation fails (e.g. metrics.raise on NaN).
        """
        # === PREPARE DATA FOR EVALUATION (if dataset provided) ===
        if dataset is not None:
            model_used_raw = self.model_params.get('use_raw_data_source', False)
            try:
                y_true, y_pred = dataset.prepare_for_evaluation(
                    y_true, y_pred, model_used_raw
                )
            except Exception as e:
                logger.error(f"[evaluate] Failed to prepare data: {e}", exc_info=True)
                return float("inf")

        if y_true.empty or y_pred.empty:
            logger.warning("Empty DataFrames provided for evaluation.")
            return float("inf")

        # Align by columns to make sure we compare the same targets
        y_true_aligned, y_pred_aligned = y_true.align(
            y_pred, join="inner", axis=1
        )

        if y_true_aligned.empty or y_pred_aligned.empty:
            logger.warning(
                "Could not align y_true and y_pred on common columns; "
                "returning inf for  evaluation metric."
            )
            return float("inf")

        # Convert to numpy for metric computation
        y_true_values = y_true_aligned.values
        y_pred_values = y_pred_aligned.values

        # Hard guard: non-finite values in evaluation arrays
        if not np.isfinite(y_true_values).all() or not np.isfinite(y_pred_values).all():
            logger.warning(
                "Non-finite values detected in evaluation arrays "
                "(y_true or y_pred). Returning +inf for evaluation metric."
            )
            return float("inf")

        try:
            metrics = calculate_metrics(y_true_values, y_pred_values)
        except ValueError as e:
             # Most common: NaN / inf detected by calculate_metrics
            logger.warning(
                "Metric computation failed with ValueError: %s. "
                "Returning +inf so caller can treat this fold as diverged.",
                e,
            )
            return float("inf")
        except Exception as e:
            # Any unexpected issue in metrics is treated as divergence
            logger.error(
                "Unexpected error during metric computation: %s. "
                "Returning +inf for evaluation metric.",
                e,
                exc_info=True,
            )
            return float("inf")

        target_metric = metric_name.lower()
        if target_metric == 'mse':
             # calculate_metrics returns rmse, so we square it to get mse
             return float(metrics['rmse'] ** 2)

        # Ensure the requested metric exists to avoid KeyError
        if target_metric not in metrics:
            logger.warning(
                "Requested evaluation metric '%s' not found in calculated metrics: %s. "
                "Returning +inf.",
                target_metric,
                list(metrics.keys()),
            )
            return float("inf")

        return float(metrics[target_metric])

    @abstractmethod
    def _fit_and_evaluate_fold(
            self,
            train_fold: pd.DataFrame,
            eval_fold: pd.DataFrame,
            validation_params: Dict[str, Any],
            dataset: TimeSeriesDataset,
            is_final_fit: bool = False
    ) -> Tuple[float, pd.DataFrame, Dict[str, List[float]]]:
        """
        Fit on the provided training fold and evaluate STRICTLY on the provided external evaluation fold.
        This unified interface is used in both:
          - HPO (evaluation-fold acts as validation),
          - Backtesting (evaluation-fold acts as test).

        Args:
            train_fold: Training data fold.
            eval_fold: External evaluation fold (validation or test) aligned in time.
            validation_params: Validation/HPO configuration (e.g., ES %, n_folds, metrics).
            dataset: The dataset object providing column context and accessors.
            is_final_fit: True for backtesting, False for HPO

        Returns:
            float: Scalar loss computed on eval_fold.

        Raises:
            ValueError: If data or parameters are invalid.
        """
        raise NotImplementedError

    # =========================================================================
    # SMART HPO ARCHITECTURE (Virtual Methods)
    # =========================================================================

    def filter_search_space(self, param_space: Dict[str, Any], fixed_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter search space BEFORE optimization begins.
        Override to remove irrelevant parameters based on fixed configuration.

        Args:
            param_space: Dictionary defining the search space.
            fixed_params: Dictionary of fixed model parameters.

        Returns:
            Filtered search space dictionary.
        """
        return param_space.copy()

    def suggest_smart_priors(
        self, param_space: Dict[str, Any], fixed_params: Dict[str, Any],
        dataset: Optional[TimeSeriesDataset] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest good starting configurations for Optuna (enqueued as first trials).

        Args:
            param_space: Dictionary defining the search space.
            fixed_params: Dictionary of fixed model parameters.
            dataset: Dataset for metadata-driven suggestions.

        Returns:
            List of parameter dictionaries to try first.
        """
        return []

    def analyze_search_space(
        self, param_space: Dict[str, Any], fixed_params: Dict[str, Any], n_trials: int
    ) -> List[str]:
        """
        Analyze search space and return warnings/suggestions for the user.

        Args:
            param_space: Dictionary defining the search space.
            fixed_params: Dictionary of fixed model parameters.
            n_trials: Number of trials planned.

        Returns:
            List of warning/info messages.
        """
        return []

    def validate_param_combination(self, params: Dict[str, Any]) -> bool:
        """
        Validate a single parameter combination before training.

        Args:
            params: Dictionary of parameters to validate.

        Returns:
            True if valid, False if should be pruned immediately.
        """
        return True

    def _create_pruner(self, pruner_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create Optuna pruner from configuration.

        Args:
            pruner_config: Pruner configuration dict with 'type' and type-specific params.
                          If None, uses default PercentilePruner.

        Returns:
            Optuna pruner instance

        Supported types:
            - "median": MedianPruner
            - "percentile": PercentilePruner (default)
            - "hyperband": HyperbandPruner
            - "threshold": ThresholdPruner
            - "patient": PatientPruner (wraps another pruner)
            - "none": NopPruner (no pruning)
        """
        if pruner_config is None:
            # Default: PercentilePruner with conservative settings
            pruner_config = {
                "type": "percentile",
                "percentile": 25.0,
                "n_startup_trials": 10,
                "n_warmup_steps": 10,
                "n_min_trials": 5
            }

        pruner_type = pruner_config.get("type", "percentile")

        if pruner_type == "median":
            return optuna.pruners.MedianPruner(
                n_startup_trials=pruner_config.get("n_startup_trials", 5),
                n_warmup_steps=pruner_config.get("n_warmup_steps", 5),
                interval_steps=pruner_config.get("interval_steps", 1),
                n_min_trials=pruner_config.get("n_min_trials", 1)
            )

        elif pruner_type == "percentile":
            return optuna.pruners.PercentilePruner(
                percentile=pruner_config.get("percentile", 25.0),
                n_startup_trials=pruner_config.get("n_startup_trials", 10),
                n_warmup_steps=pruner_config.get("n_warmup_steps", 10),
                interval_steps=pruner_config.get("interval_steps", 1),
                n_min_trials=pruner_config.get("n_min_trials", 5)
            )

        elif pruner_type == "hyperband":
            return optuna.pruners.HyperbandPruner(
                min_resource=pruner_config.get("min_resource", 1),
                max_resource=pruner_config.get("max_resource", 100),
                reduction_factor=pruner_config.get("reduction_factor", 3)
            )

        elif pruner_type == "threshold":
            lower = pruner_config.get("lower", None)
            upper = pruner_config.get("upper", None)
            return optuna.pruners.ThresholdPruner(lower=lower, upper=upper)

        elif pruner_type == "patient":
            # PatientPruner wraps another pruner
            wrapped_config = pruner_config.get("wrapped_pruner", {"type": "median"})
            wrapped_pruner = self._create_pruner(wrapped_config)
            patience = pruner_config.get("patience", 2)
            return optuna.pruners.PatientPruner(wrapped_pruner, patience=patience)

        elif pruner_type == "none":
            return optuna.pruners.NopPruner()

        else:
            logger.warning(f"Unknown pruner type: '{pruner_type}'. Using default PercentilePruner.")
            return optuna.pruners.PercentilePruner(
                percentile=25.0,
                n_startup_trials=10,
                n_warmup_steps=10,
                n_min_trials=5
            )

    def _create_sampler(self, param_space: Dict[str, Any], fixed_params: Dict[str, Any]) -> Any:
        """Create optimal Optuna sampler based on search space dimensions (supports nested dicts)."""
        def count_dimensions(space: Dict[str, Any]) -> tuple:
            """Recursively count continuous and discrete dimensions."""
            n_cont, n_disc = 0, 0
            for v in space.values():
                if isinstance(v, list):
                    n_disc += 1
                elif isinstance(v, dict):
                    if 'min' in v and 'max' in v:
                        n_cont += 1
                    else:
                        # Nested dict (e.g., scheduler_config) - recurse
                        sub_cont, sub_disc = count_dimensions(v)
                        n_cont += sub_cont
                        n_disc += sub_disc
            return n_cont, n_disc

        n_continuous, n_discrete = count_dimensions(param_space)
        total_dim = n_continuous + n_discrete

        logger.info(f"HPO Search Space: {total_dim} dimensions ({n_continuous} continuous, {n_discrete} discrete)")

        return optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=max(5, total_dim * 2),
            multivariate=True if total_dim >= 2 else False,
            n_ei_candidates=24,
            consider_prior=True,
            warn_independent_sampling=False  # We use dependent sampling (num_heads -> hpo_head_dim)
        )

    def _suggest_optuna_params(self, trial: Any, param_space: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """
        Helper to map config dictionary to Optuna trial suggestions.
        Supports nested dictionaries (e.g., scheduler_config).

        Args:
            trial: Optuna trial object
            param_space: Parameter space definition (supports nested dicts)
            prefix: Internal prefix for nested parameter names (e.g., "scheduler_config.")

        Returns:
            Dictionary of sampled parameters (preserves nesting)
        """
        params = {}
        for key, value in param_space.items():
            # Full parameter name for Optuna (flattened with dots)
            optuna_name = f"{prefix}{key}" if prefix else key

            if isinstance(value, list):
                # List implies Categorical choice
                params[key] = trial.suggest_categorical(optuna_name, value)
            elif isinstance(value, dict):
                # Check if it's a range definition (has 'min' and 'max')
                if "min" in value and "max" in value:
                    # Dict implies Range (Int or Float)
                    low = value.get("min")
                    high = value.get("max")
                    log = value.get("log", False)
                    step = value.get("step", None)

                    # Heuristic: if min or max is float, treat as float range
                    if isinstance(low, float) or isinstance(high, float):
                        params[key] = trial.suggest_float(optuna_name, low, high, log=log, step=step)
                    else:
                        # Otherwise treat as int range
                        params[key] = trial.suggest_int(optuna_name, low, high, step=step or 1, log=log)
                else:
                    # Nested dictionary (e.g., scheduler_config) - recurse
                    params[key] = self._suggest_optuna_params(trial, value, prefix=f"{optuna_name}.")
        return params

    @staticmethod
    def _is_valid_prior_value(key: str, value: Any, param_space: Dict[str, Any]) -> bool:
        """Check if prior value is valid for param_space (shared utility)."""
        if key not in param_space:
            return False

        space_def = param_space[key]

        # Categorical (list)
        if isinstance(space_def, list):
            return value in space_def

        # Continuous (range)
        if isinstance(space_def, dict) and 'min' in space_def and 'max' in space_def:
            return space_def['min'] <= value <= space_def['max']

        return True

    def optimize_hyperparameters(
            self,
            dataset: TimeSeriesDataset,
            model_config: Dict[str, Any],
            validation_params: Dict[str, Any],
            folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    ) -> Tuple[Dict[str, Any], float]:
        """
        Perform HPO with support for interactive Optuna optimization.
        """
        if not folds:
            raise ValueError("`folds` must be provided by the ExperimentRunner.")

        hpo_metric = validation_params.get("evaluation_metric", "mse")
        opt_params = model_config.get("optimization", {"method": "grid", "params": {}})
        method = opt_params.get("method", "grid")
        n_trials_value = opt_params.get("n_trials", 10)

        # Security: Cap maximum trials to prevent DoS
        MAX_TRIALS = 1000
        if n_trials_value > MAX_TRIALS:
            logger.warning(
                f"n_trials ({n_trials_value}) exceeds maximum ({MAX_TRIALS}). "
                f"Capping to {MAX_TRIALS} trials."
            )
            n_trials_value = MAX_TRIALS

        n_folds = validation_params["n_folds"]

        best_model_params = {k: v for k, v in model_config.items() if k not in ["optimization", "optimize"]}

        logging.info(f"Using {len(folds)} folds for HPO (n_folds={n_folds}). Method: {method}")

        raw_param_space = opt_params.get("params", {})

        # 1. Analyze and Filter Search Space
        warnings = self.analyze_search_space(raw_param_space, best_model_params, n_trials_value)
        if warnings:
            logger.warning("--- HPO Search Space Analysis ---")
            for w in warnings: logger.warning(w)

        param_space = self.filter_search_space(raw_param_space, best_model_params)
        if len(param_space) < len(raw_param_space):
            logger.info(f"Filtered search space: reduced from {len(raw_param_space)} to {len(param_space)} parameters.")

        # --- SETUP OPTIMIZATION STRATEGY ---
        candidates_list = []
        study = None

        if method == "optuna":
            if optuna is None:
                raise ImportError("Optuna library is not installed.")

            # Read pruner configuration from optimization section
            pruner_config = opt_params.get("pruner_config", None)
            pruner = self._create_pruner(pruner_config)

            logger.info(f"HPO Pruner: {pruner.__class__.__name__}")
            if pruner_config:
                logger.debug(f"Pruner config: {pruner_config}")

            study = optuna.create_study(
                direction="minimize",
                sampler=self._create_sampler(param_space, best_model_params),
                pruner=pruner
            )

            # Enqueue Smart Priors
            priors = self.suggest_smart_priors(param_space, best_model_params, dataset=dataset)
            for prior in priors:
                # Safety check - filter invalid values
                valid_prior = {
                    k: v for k, v in prior.items()
                    if self._is_valid_prior_value(k, v, param_space)
                }

                # Log what was filtered
                invalid = {k: v for k, v in prior.items()
                           if k in param_space and not self._is_valid_prior_value(k, v, param_space)}
                if invalid:
                    logger.warning(f"[HPO Safety] Filtered invalid prior values: {invalid}")

                if valid_prior:
                    try:
                        study.enqueue_trial(valid_prior)
                        logger.info(f"Enqueued smart prior: {valid_prior}")
                    except Exception as e:
                        logger.warning(f"Failed to enqueue prior {valid_prior}: {e}")

        elif method == "grid":
            candidates_list = generate_grid_params(param_space=param_space, n_trials=n_trials_value)
        elif method == "random":
            candidates_list = generate_random_params(param_space=param_space, n_trials=n_trials_value)
        else:
            raise ValueError(f"Invalid optimization method: {method}")

        # If using static list (Grid/Random), filter/validate upfront
        if method != "optuna":
            candidates_list = self.filter_candidates(candidates_list, best_model_params)
            if not candidates_list:
                raise ValueError("No valid parameter combinations after filtering.")
            logger.info(f"Scheduled {len(candidates_list)} candidates for evaluation.")

        best_hpo_params: Dict[str, Any] = best_model_params.copy()
        best_loss: float = float("inf")
        best_epochs_list: List[int] = []  # Track epochs for best trial

        # Determine total iterations
        total_iterations = n_trials_value if method == "optuna" else len(candidates_list)

        # --- MAIN OPTIMIZATION LOOP ---
        for i in range(total_iterations):
            trial_idx = i + 1
            trial = None

            # 1. ACQUIRE PARAMETERS
            if method == "optuna":
                trial = study.ask()
                try:
                    # Use virtual method for suggestion (allows dependent sampling override)
                    sampled_params = self._suggest_optuna_params(trial, param_space)

                    # Backfilling: Fixed params override sampled ones (logic usually filters them out first)
                    # We merge to form a complete candidate for validation
                    # IMPORTANT: Use copy for shallow dict, nested dicts are referenced (OK for validation)
                    # If validation needs to modify nested dicts, change to deepcopy
                    full_candidate = best_model_params.copy()
                    # Use shallow update here - validation doesn't need deep merge
                    # (validation only reads params, doesn't modify nested dicts)
                    full_candidate.update(sampled_params)

                    # Validate Combination
                    if not self.validate_param_combination(full_candidate):
                        logger.info(f"Trial {trial_idx}: Invalid parameter combination. Pruning.")
                        study.tell(trial, float('inf'), state=optuna.trial.TrialState.PRUNED)
                        continue

                    # Compatibility with legacy filter_candidates
                    dummy_list = self.filter_candidates([sampled_params], best_model_params)
                    if not dummy_list:
                        logger.info(f"Trial {trial_idx}: Pruned by filter_candidates.")
                        study.tell(trial, float('inf'), state=optuna.trial.TrialState.PRUNED)
                        continue

                    params = sampled_params

                except Exception as e:
                    # Import here to avoid circular dependency
                    from models.transformer import ParameterConstraintViolation

                    # Constraint violations should be PRUNED, not FAILED
                    # This helps Optuna's sampler learn the valid parameter space
                    if isinstance(e, ParameterConstraintViolation):
                        logger.debug(f"Trial {trial_idx}: Constraint violation (pruned): {e}")
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    else:
                        # Other exceptions are actual failures
                        logger.warning(f"Trial {trial_idx}: Parameter suggestion failed: {e}")
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    continue
            else:
                # Take next from static list
                params = candidates_list[i]

            # Prepare full config (DEEP MERGE for nested dicts like scheduler_config)
            import copy
            current_params = copy.deepcopy(best_model_params)

            # Deep merge params into current_params
            def deep_merge(base, update):
                """Recursively merge update dict into base dict"""
                for key, value in update.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        deep_merge(base[key], value)  # Recurse for nested dicts
                    else:
                        base[key] = value  # Overwrite for non-dicts

            deep_merge(current_params, params)
            current_params["type"] = self.model_type

            fold_losses = []
            fold_epochs = []  # Track epochs per fold for this trial

            logger.info(f"--- Starting Trial {trial_idx}/{total_iterations} ---")
            logger.debug(f"Evaluating params: {params}")

            # 2. EVALUATE (CV Loop)
            for fold_idx, (train_fold, eval_fold) in enumerate(folds, start=1):
                try:
                    from models.factory import ModelFactory
                    candidate_name = f"{self.model_name}_trial_{i}"

                    # Use self.run_context as base if available (it must be per strict init)
                    candidate_context = None
                    if self.run_context:
                        candidate_context = self.run_context.with_metadata(
                            model_name=candidate_name,
                            fold_idx=fold_idx,
                            metadata={'is_hpo_trial': True}  # Signal: lightweight mode
                        )

                    candidate = ModelFactory.create(
                        model_type=self.model_type,
                        model_name=candidate_name,
                        num_features=self.num_features,
                        forecast_steps=self.forecast_steps,
                        window_size=self.window_size,
                        model_params=current_params,
                        dataset=dataset,
                        run_context=candidate_context
                    )

                    #for Optuna - to handle cross validation prunning
                    max_epochs = current_params.get("epochs", 100)
                    step_offset = (fold_idx - 1) * max_epochs

                    # Prepare Validation Params for this fold (Copy!)
                    fold_val_params = dict(validation_params)
                    fold_val_params["metric"] = hpo_metric  # Inject metric

                    # Unpack 3 values: loss, predictions, history (ignore history for HPO)
                    loss, _, history = candidate._fit_and_evaluate_fold(
                        train_fold=train_fold,
                        eval_fold=eval_fold,
                        validation_params=fold_val_params,
                        dataset=dataset,
                        is_final_fit=False,
                        optuna_trial=trial,
                        trial_step_offset=step_offset
                    )
                    fold_losses.append(loss)

                    # Extract trained epochs if available (Neural models)
                    if history and 'train_loss' in history:
                        actual_epochs = len(history['train_loss'])
                        fold_epochs.append(actual_epochs)
                        logger.debug(f"  Fold {fold_idx} trained for {actual_epochs} epochs")
                    elif method == "optuna":
                        # For statistical models (no epochs), report fold loss to enable pruning
                        # We use fold_idx as the step. Neural models report internal steps (epochs),
                        # so we strictly avoid reporting here if history exists to prevent step conflicts.
                        trial.report(loss, step=fold_idx)
                        if trial.should_prune():
                            logger.info(f"Trial {trial_idx} PRUNED by Optuna after fold {fold_idx} (Stat model).")
                            raise optuna.TrialPruned()

                    logger.info(f"  Fold {fold_idx}/{len(folds)} loss: {loss:.6f}")

                    # GPU + CPU Memory Cleanup after each fold (HPO optimization).
                    # gc.collect() is essential: forces Python to release DataLoader
                    # worker pipes immediately, preventing fd exhaustion across trials.
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                except optuna.TrialPruned:
                    logger.info(f"Trial {trial_idx} PRUNED by Optuna on fold {fold_idx}.")
                    if method == "optuna" and trial:
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    fold_losses = []  # Clear losses to skip averaging
                    fold_epochs = []

                    # GPU Memory Cleanup after pruned fold
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    break  # Stop processing folds for this trial

                except Exception as e:
                    logger.warning(f"Evaluation failed for Trial {trial_idx} on Fold {fold_idx}: {e}", exc_info=True)
                    fold_losses.append(float("inf"))

                    # GPU Memory Cleanup after failed fold
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

            # Skip reporting if pruned (fold_losses cleared)
            if not fold_losses and method == "optuna":
                continue

            # 3. AGGREGATE RESULTS
            avg_loss = np.nanmean([l if np.isfinite(l) else np.nan for l in fold_losses])

            # Calculate average epochs for this trial if available
            if fold_epochs:
                avg_epochs = float(np.mean(fold_epochs))
                logger.info(f"Trial {trial_idx} Avg Loss: {avg_loss:.6f}, Avg Epochs: {avg_epochs:.1f}")

            # Handle case where all folds failed
            if np.isnan(avg_loss):
                avg_loss = float("inf")

            # 4. FEEDBACK (TELL OPTUNA)
            if method == "optuna":
                if np.isfinite(avg_loss):
                    study.tell(trial, avg_loss)
                else:
                    # Tell Optuna this was a failure
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)

            # 5. TRACK BEST
            if np.isfinite(avg_loss) and avg_loss < best_loss:
                best_loss = avg_loss
                # Use deep_merge (defined above) to preserve nested dict fields
                deep_merge(best_hpo_params, params)

                # Update best epochs list when we find new best
                if fold_epochs:
                    best_epochs_list = fold_epochs.copy()

                logger.info(f"*** New best score: {best_loss:.6f} (found at Trial {trial_idx}) ***")

            # GPU Memory Cleanup after each trial (prevent memory fragmentation)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        if not np.isfinite(best_loss):
            logger.error("No valid parameter combinations found")
            raise ValueError("No valid parameter combinations found during optimization.")

        # Add epoch stats to best params
        if best_epochs_list:
            mean_e = float(np.mean(best_epochs_list))
            std_e = float(np.std(best_epochs_list))
            best_hpo_params['mean_trained_epochs'] = mean_e
            best_hpo_params['std_trained_epochs'] = std_e
            logger.info(f"HPO Finished. Best Loss: {best_loss:.6f}. Optimal Epochs: {mean_e:.1f} +/- {std_e:.1f}")
        else:
            logger.info(f"HPO Finished. Best Loss: {best_loss:.6f}")

        return best_hpo_params, best_loss

    @abstractmethod
    def fit(self, *args, **kwargs) -> Tuple[float, Dict]:
        """
        Fit the model to the training data.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass


    @abstractmethod
    def predict(self, *args, **kwargs) -> pd.DataFrame:
        """
        Generate predictions for the specified horizon.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Predictions in a 2D DataFrame.
        """
        pass

    def get_valid_params(self) -> set:
        """
        Get the set of valid parameter names for the model.

        Returns:
            Set of valid parameter names.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement get_valid_params.")

    def filter_candidates(self, candidates: List[Dict[str, Any]], best_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter candidate parameter combinations based on model-specific constraints.

        Args:
            candidates: List of parameter combinations.
            best_params: Fixed parameters.

        Returns:
            Filtered parameter combinations.
        """
        return candidates  # Default: no filtering

    def prepare_data(self, dataset: TimeSeriesDataset) -> List[TimeSeriesDataset]:
        """
        Prepare data for the model.

        Args:
            dataset: Input dataset.

        Returns:
            List of processed datasets.

        Raises:
            ValueError: If dataset is invalid.
        """
        return [dataset]

    def _validate_model_specific_inputs(self, *args) -> None:
        """
        Validate model-specific inputs.

        Args:
            *args: Variable length argument list.

        Note:
            Placeholder method to be implemented by subclasses if needed.
        """
        pass

class StatTSForecaster(TSForecaster, ABC):
    """
    Abstract base class for statistical time series forecasting models.

    This class provides a common structure for models based on statistical
    methods (e.g., ARIMA, VAR) from libraries like `statsmodels`. It handles
    the logic for walk-forward validation, data preparation for univariate models,
    and saving/loading the model state.
    """


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
        Initializes the statistical forecaster.

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
            **kwargs
        )

    def _fit_and_evaluate_fold(
            self,
            train_fold: pd.DataFrame,
            eval_fold: pd.DataFrame,
            validation_params: Dict[str, Any],
            dataset: TimeSeriesDataset,
            is_final_fit: bool = False,
            optuna_trial: Optional[Any] = None,
            trial_step_offset: int = 0
    ) -> Tuple[float, pd.DataFrame, Dict[str, List[float]]]:
        """
        Fit on 'train_fold' and compute loss on the 'eval_fold'.
        Works for ARIMA/VAR variants with/without exogenous variables.

        This method is used for both hyperparameter optimization (with a hold-out
        set) and for the final model training (on the entire development set).

        Args:
            train_fold: Training data fold.
            eval_fold: External evaluation fold (validation or test) aligned in time.
            validation_params: Validation/HPO configuration (e.g., ES %, n_folds, metrics).
            dataset: The dataset object providing column context and accessors.
            is_final_fit: True for backtesting, False for HPO

        Returns:
            float: The validation loss (e.g., MSE) on the hold-out set.
        """
        target_cols = dataset.target_columns

        # --- Statistical models use ONLY FUTURE COVARIATES ---
        # Past-only covariates cannot be used as regressors in ARIMA/VAR
        # because these models require future values for all regressors
        # to make a prediction (contemporaneous relationship).

        all_exog = getattr(dataset, "future_covariates", []) or []

        # Log warning if past-only covariates are ignored (good for debugging config)
        ignored_past = getattr(dataset, "past_covariates", [])
        if ignored_past:
            logger.debug(f"Stat model ignoring past-only covariates: {ignored_past}")

        # 1) Fit on the entire train fold.
        endog_fit = train_fold[target_cols]
        exog_fit = train_fold[all_exog] if all_exog else None
        fit_loss, fit_history = self.fit(endog_fit, exog_series=exog_fit, dataset=dataset, is_final_fit=is_final_fit)

        if not self.fitted or fit_loss == float('inf'):
            logger.warning(f"Model {self.__class__.__name__} failed to fit. Returning inf loss.")
            return float("inf"), pd.DataFrame(), {}

        # 2) Predict on the external eval fold (acts as validation in HPO or test in backtesting).
        endog_eval = eval_fold[target_cols]
        exog_eval = eval_fold[all_exog] if all_exog else None
        preds = self.predict(future_exog=exog_eval, forecast_steps=len(endog_eval))
        metric_to_optimize = validation_params.get("metric", "mse")
        loss = self.evaluate(endog_eval, preds, dataset=dataset,metric_name=metric_to_optimize)
        return loss, preds, {}

    def prepare_data(self, dataset: TimeSeriesDataset) -> List[TimeSeriesDataset]:
        """
        Prepares data for statistical models. For univariate models, this splits
        the data into one dataset per target variable, including all relevant
        exogenous variables for each.

        Args:
            dataset (TimeSeriesDataset): The input dataset to prepare.

        Returns:
            List[TimeSeriesDataset]: A list of dataset objects. For univariate models,
                this list will contain one dataset for each target column. For
                multivariate models, it will contain the original dataset.
        """
        if dataset.development_data is None or dataset.test_data is None:
            raise ValueError("Dataset must have development_data and test_data. Call dataset.split_data().")

        if self.is_univariate:
            datasets = []
            # Ensure all exogenous columns are passed to the new univariate datasets.
            all_exog_columns = dataset.past_covariates + dataset.future_covariates

            for target_col in dataset.target_columns:
                logger.info(f"Preparing univariate dataset for target: {target_col}")

                cols_for_this_run = [target_col] + all_exog_columns
                data_for_this_run = dataset.series[cols_for_this_run]

                local_dataset = TimeSeriesDataset(
                    dataset_name=f"{dataset.name}_{target_col}",
                    config=dataset.config,
                    num_features=1,
                    data=data_for_this_run.copy(),
                    columns=[target_col],
                    # Pass the correct set of covariate columns
                    past_covariates=dataset.past_covariates,
                    future_covariates=dataset.future_covariates,
                    freq=dataset.freq,
                )
                local_dataset.development_data = dataset.development_data[cols_for_this_run]
                local_dataset.test_data = dataset.test_data[cols_for_this_run]
                datasets.append(local_dataset)
            return datasets

        return [dataset]

class NeuralTSForecaster(TSForecaster, ABC):
    """Abstract base class for neural network-based time series forecasting models."""
    # ═══════════════════════════════════════════════════════════════════
    # UNIVERSAL TRAINING DEFAULTS (all neural models)
    # ═══════════════════════════════════════════════════════════════════
    DEFAULT_TRAINING_PARAMS = {
        "epochs": 100,
        "early_stopping_patience": 10,
        "batch_size": 32,
        "early_stopping_validation_percentage": 10,
        "num_workers": min(4, max(1, os.cpu_count() // 4)),  # Default: 1/4 of CPU cores, min 1, max 4
    }

    @staticmethod
    def _suggest_batch_sizes(series_length: int, num_features: int) -> List[int]:
        """
        Suggest appropriate batch sizes based on dataset characteristics.

        Args:
            series_length: Number of time steps in the series
            num_features: Number of features (targets + exogenous)

        Returns:
            List of 2-3 recommended batch sizes

        Rationale:
            - Small datasets: smaller batches for more gradient updates
            - Large datasets: larger batches for training efficiency
            - Multivariate: slightly smaller batches (more parameters)
        """
        # Base recommendations by dataset size
        if series_length < 500:
            base_sizes = [8, 16, 32]
        elif series_length < 1000:
            base_sizes = [16, 32, 64]
        elif series_length < 5000:
            base_sizes = [32, 64, 128]
        else:
            base_sizes = [64, 128, 256]

        # Adjust for multivariate (more features = smaller batches beneficial)
        if num_features > 5:
            base_sizes = [max(8, size // 2) for size in base_sizes]

        # Return top 2 suggestions
        return base_sizes[:2]

    def _get_training_param(self, param_name: str, default: Any = None) -> Any:
        """
        Get training parameter with proper fallback chain.

        Fallback order:
        1. model_params (for backward compatibility)
        2. DEFAULT_TRAINING_PARAMS (class-level defaults)
        3. Provided default argument

        Args:
            param_name: Name of training parameter
            default: Override default if needed

        Returns:
            Parameter value
        """
        # Check model_params first (backward compatibility)
        if param_name in self.model_params:
            return self.model_params[param_name]

        # Check class defaults
        if default is None and param_name in self.DEFAULT_TRAINING_PARAMS:
            return self.DEFAULT_TRAINING_PARAMS[param_name]

        # Use provided default
        return default if default is not None else None

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
        Initialize the neural forecaster with model-specific parameters.

        Args:
            model_params: Model-specific parameters.
            num_features: Number of features in the time series data.
            forecast_steps: Number of steps to forecast.
            window_size: The look-back window size.
            dataset: Input time series dataset.
            run_context: Run context for training.
            kwargs: Additional parameters for model.

        Raises:
            ValueError: If num_features or forecast_steps are invalid.
        """
        provided_device = kwargs.pop('device', None)
        if provided_device is not None:
            raise ValueError("Provided device is not supported")
        super().__init__(
            model_params=model_params,
            num_features=num_features,
            forecast_steps=forecast_steps,
            window_size=window_size,
            dataset=dataset,
            run_context=run_context,
            **kwargs
        )
        self.device = (
            provided_device if provided_device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Using device: {self.device}")
        # Attributes to store the column order contract learned during fit
        self._encoder_column_order: Optional[List[str]] = None
        self._decoder_column_order: Optional[List[str]] = None

        # Cache covariate column names from the dataset for later use (e.g. in predict()).
        # These lists are aligned with the feature layout indices computed above and allow
        # us to select appropriate features from input DataFrames.
        self.past_covariates: List[str] = list(getattr(dataset, "past_covariates", []) or [])
        self.future_covariates: List[str] = list(getattr(dataset, "future_covariates", []) or [])

        # Training history storage
        self.training_history = {}

        # Gradient monitoring (will be initialized in fit if run_context provided)
        self.gradient_monitor = None

    # ═══════════════════════════════════════════════════════════════════════
    # SHARED HPO HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════
    # These methods are inherited by all neural models (Transformer, LSTM, etc.)
    # to enable consistent dataset-aware hyperparameter optimization.
    # ═══════════════════════════════════════════════════════════════════════

    def _get_safe_optimizer_eps(self) -> float:
        """
        Returns a safe epsilon for Adam.
        For Mixed Precision (AMP), 1e-8 is not enough -> we use 1e-6.
        """
        # 1. Check if the user has set it manually in the config
        opt_config = self.model_params.get("optimizer_config", {})
        if "eps" in opt_config:
            return float(opt_config["eps"])

        #2. Check if we are using AMP (float16)
        use_amp = self.model_params.get("use_amp", False)

        #3. Return the appropriate value
        return 1e-6 if use_amp else 1e-8

    def _create_optimizer(self):
        """
        Creates an epsilon-safe optimizer with optional fused optimization.

        Automatically uses fused AdamW on CUDA with PyTorch 2.0+ for better performance.
        Falls back gracefully to standard optimizer on CPU or older PyTorch versions.

        Returns:
            torch.optim.Optimizer: Configured optimizer instance
        """
        # Get parameters
        lr = self.model_params.get("learning_rate", 0.001)
        wd = self.model_params.get("weight_decay", 0.0)
        opt_name = self.model_params.get("optimizer", "adamw").lower()  # ← Default changed to adamw
        safe_eps = self._get_safe_optimizer_eps()

        # Check if fused optimizer is available (PyTorch 2.0+ on CUDA)
        use_fused = False
        if torch.cuda.is_available():
            try:
                version_parts = torch.__version__.split('.')
                major = int(version_parts[0])
                minor = int(version_parts[1].split('+')[0] if '+' in version_parts[1] else version_parts[1])
                use_fused = (major >= 2)
            except (ValueError, IndexError):
                use_fused = False

        # Create optimizer based on type
        if opt_name == "adamw":
            if use_fused:
                logger.debug("Using Fused AdamW optimizer (CUDA optimized)")
                return torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=wd,
                    eps=safe_eps,
                    fused=True
                )
            else:
                logger.debug("Using AdamW optimizer")
                return torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=wd,
                    eps=safe_eps
                )

        elif opt_name == "adam":
            # Warn if using Adam with weight_decay > 0
            if wd > 0:
                logger.warning(
                    f"Using Adam with weight_decay={wd}. "
                    "Consider switching to AdamW (set optimizer='adamw' in config) "
                    "for better regularization. Adam applies weight decay incorrectly."
                )
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd,
                eps=safe_eps
            )

        elif opt_name == "sgd":
            # SGD with momentum
            momentum = self.model_params.get("momentum", 0.9)
            logger.debug(f"Using SGD optimizer with momentum={momentum}")
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=wd
            )

        else:
            # Unknown optimizer - fallback to AdamW with warning
            logger.warning(
                f"Unknown optimizer '{opt_name}'. Falling back to AdamW."
            )
            if use_fused:
                return torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=wd,
                    eps=safe_eps,
                    fused=True
                )
            else:
                return torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=wd,
                    eps=safe_eps
                )


    @staticmethod
    def _infer_seasonal_period(freq: str) -> Optional[int]:
        """
        Infer typical seasonal period from pandas frequency string.

        Shared helper for smart HPO - used by Transformer, LSTM, and other
        neural models to adapt priors based on data frequency.

        Supports both uppercase (pandas <2.2, deprecated) and lowercase (pandas 2.2+)
        frequency strings for backward and forward compatibility.

        Args:
            freq: Pandas frequency string (e.g., 'H'/'h', 'D', 'W', 'M', 'Q')
                Can also handle composite frequencies like '15T', '2h', '15min', etc.

        Returns:
            Typical seasonal period (number of timesteps in cycle), or None
            if frequency is not recognized or not provided.

        Frequency Mappings:
            - 'H'/'h' (Hourly) → 24 (daily cycle)
            - 'T'/'min' (Minutely) → 1440 (daily cycle)
            - 'D' (Daily) → 7 (weekly cycle)
            - 'W' (Weekly) → 52 (annual cycle)
            - 'M'/'ME' (Monthly) → 12 (annual cycle)
            - 'Q'/'QE' (Quarterly) → 4 (annual cycle)
            - 's'/'S' (Seconds) → 86400 (daily cycle)

        Examples:
            >>> NeuralTSForecaster._infer_seasonal_period('H')  # Old style
            24

            >>> NeuralTSForecaster._infer_seasonal_period('h')  # New style (pandas 2.2+)
            24

            >>> NeuralTSForecaster._infer_seasonal_period('15T')  # Old style minutes
            96  # 96 15-minute intervals per day

            >>> NeuralTSForecaster._infer_seasonal_period('15min')  # New style
            96

            >>> NeuralTSForecaster._infer_seasonal_period('2h')  # New style 2-hour
            12  # 12 2-hour intervals per day

        Note:
            This is a heuristic based on common patterns. Actual seasonal
            periods may vary depending on the specific domain and data.
        """
        # Map of frequency codes to base periods
        # Includes both old (uppercase) and new (lowercase) conventions
        freq_map = {
            # Hourly
            "H": 24,  # Old: uppercase (deprecated in pandas 2.2+)
            "h": 24,  # New: lowercase (recommended)

            # Minutely
            "T": 1440,  # Old: 'T' for time/minute
            "min": 1440,  # New: 'min' (recommended)

            # Daily
            "D": 7,  # Daily → weekly cycle

            # Weekly
            "W": 52,  # Weekly → annual cycle

            # Monthly
            "M": 12,  # Old: Month (deprecated, ambiguous)
            "ME": 12,  # New: Month End (recommended)
            "MS": 12,  # New: Month Start

            # Quarterly
            "Q": 4,  # Old: Quarter (deprecated)
            "QE": 4,  # New: Quarter End (recommended)
            "QS": 4,  # New: Quarter Start

            # Yearly
            "Y": 1,  # Old: Year (deprecated)
            "YE": 1,  # New: Year End (recommended)
            "YS": 1,  # New: Year Start

            # Seconds (less common but supported)
            "S": 86400,  # Uppercase (old)
            "s": 86400,  # Lowercase (new)
        }

        if not freq:
            return None

        # Special handling for 'min' suffix (e.g., '15min')
        if freq.endswith('min'):
            # Extract multiplier if present (e.g., '15' from '15min')
            if len(freq) > 3:
                try:
                    multiplier = int(freq[:-3])
                    base_period = freq_map.get('min', 1440)
                    return base_period // multiplier
                except (ValueError, ZeroDivisionError):
                    return freq_map.get('min')
            return freq_map.get('min')

        # Check for multi-character frequency codes (ME, MS, QE, QS, YE, YS)
        if len(freq) >= 2:
            # Try last 2 characters first (for ME, MS, QE, QS, YE, YS)
            two_char_code = freq[-2:]
            if two_char_code in freq_map:
                # For simple two-char codes like 'ME', 'QE'
                if len(freq) == 2:
                    return freq_map[two_char_code]

                # For composite like '2ME', '3QE' (less common but possible)
                try:
                    multiplier = int(freq[:-2])
                    base_period = freq_map[two_char_code]
                    # For time-based frequencies, divide the period
                    if two_char_code in ['ME', 'MS', 'QE', 'QS', 'YE', 'YS']:
                        return base_period  # Don't divide for calendar frequencies
                    return base_period // multiplier
                except (ValueError, ZeroDivisionError):
                    return freq_map[two_char_code]

        # Standard single-character frequency code
        base_freq = freq[-1]
        period = freq_map.get(base_freq)

        if period is None:
            return None

        # Handle composite frequencies with multipliers (e.g., '15T', '2H', '2h')
        # Only applies to time-based frequencies (H, h, T, S, s)
        if base_freq in ["T", "H", "h", "S", "s"] and len(freq) > 1:
            try:
                multiplier = int(freq[:-1])
                if multiplier > 0:
                    period = period // multiplier
            except (ValueError, ZeroDivisionError):
                pass  # Keep original period if parsing fails

        return period

    @staticmethod
    def _categorize_dataset_size(length: int) -> str:
        """
        Categorize dataset by size for overfitting risk assessment.

        Shared helper for smart HPO - used to adapt dropout, model capacity,
        and other hyperparameters based on dataset size.

        This method categorizes datasets into three risk levels based on the
        number of timesteps. Smaller datasets have higher overfitting risk and
        should use higher dropout, smaller models, and more regularization.

        Args:
            length: Number of timesteps in dataset

        Returns:
            Size category as string: 'small', 'medium', or 'large'

        Categories:
            - small (< 1,000 timesteps):
                * High overfitting risk
                * Recommendations: higher dropout (+0.1), smaller models,
                  more regularization, simpler architectures

            - medium (1,000 - 10,000 timesteps):
                * Moderate overfitting risk
                * Recommendations: standard settings, balanced capacity

            - large (> 10,000 timesteps):
                * Low overfitting risk
                * Recommendations: can use lower dropout, larger models,
                  more complex architectures

        Examples:
            >>> NeuralTSForecaster._categorize_dataset_size(500)
            'small'  # High risk, use higher dropout

            >>> NeuralTSForecaster._categorize_dataset_size(5000)
            'medium'  # Moderate risk, standard settings

            >>> NeuralTSForecaster._categorize_dataset_size(50000)
            'large'  # Low risk, can use larger models

        Note:
            Thresholds (1,000 and 10,000) are heuristics for neural time series
            models. Actual optimal thresholds may vary by domain and model type.
        """
        if length < 1000:
            return "small"  # High overfitting risk
        elif length < 10000:
            return "medium"  # Moderate overfitting risk
        else:
            return "large"  # Low overfitting risk

    @staticmethod
    def _extract_dataset_metadata(dataset: Optional['TimeSeriesDataset']) -> Dict[str, Any]:
        """
        Extract metadata from dataset for HPO adaptation.

        Consolidates common dataset metadata extraction logic used across
        Transformer, LSTM, and other neural models. This method extracts all
        relevant information in a single pass, avoiding repeated attribute
        access and providing a consistent interface.

        Args:
            dataset: Optional TimeSeriesDataset object. If None, returns empty dict.

        Returns:
            Dictionary with extracted metadata (empty dict if dataset is None):

            Keys (all optional, included only if information is available):
                - freq (str): Frequency string (e.g., 'H', 'D', 'W', 'M')
                - seasonal_period (int): Inferred seasonal period
                - series_length (int): Number of timesteps in dataset
                - size_category (str): 'small', 'medium', or 'large'
                - has_past_exog (bool): True if encoder exogenous variables present
                - has_future_exog (bool): True if decoder exogenous variables present

        Examples:
            >>> # Hourly dataset with 17,420 timesteps and past exog
            >>> metadata = NeuralTSForecaster._extract_dataset_metadata(dataset)
            >>> print(metadata)
            {
                'freq': 'H',
                'seasonal_period': 24,
                'series_length': 17420,
                'size_category': 'large',
                'has_past_exog': True,
                'has_future_exog': False
            }

            >>> # Small daily dataset without exog
            >>> metadata = NeuralTSForecaster._extract_dataset_metadata(small_dataset)
            >>> print(metadata)
            {
                'freq': 'D',
                'seasonal_period': 7,
                'series_length': 800,
                'size_category': 'small',
                'has_past_exog': False,
                'has_future_exog': False
            }

            >>> # No dataset provided
            >>> metadata = NeuralTSForecaster._extract_dataset_metadata(None)
            >>> print(metadata)
            {}

        Usage in suggest_smart_priors:
            >>> def suggest_smart_priors(self, param_space, fixed_params, dataset=None):
            ...     # Extract all metadata at once
            ...     dataset_info = self._extract_dataset_metadata(dataset)
            ...
            ...     # Access specific values
            ...     seasonal_period = dataset_info.get("seasonal_period")
            ...     size_category = dataset_info.get("size_category")
            ...
            ...     # Use for HPO adaptation
            ...     if size_category == "small":
            ...         # Increase dropout for small datasets
            ...         priors = [{"dropout": 0.3}, {"dropout": 0.4}]

        Note:
            This method uses safe attribute access with getattr() and hasattr()
            to handle datasets that may not have all attributes. Missing
            attributes are simply not included in the returned dictionary.
        """
        dataset_info = {}

        if not dataset:
            return dataset_info

        # 1. Frequency → Implied seasonal period
        freq = getattr(dataset, "freq", None)
        if freq:
            dataset_info["freq"] = freq
            dataset_info["seasonal_period"] = NeuralTSForecaster._infer_seasonal_period(freq)

        # 2. Dataset size → Overfitting risk assessment
        if hasattr(dataset, "series") and dataset.series is not None:
            series_length = len(dataset.series)
            dataset_info["series_length"] = series_length
            dataset_info["size_category"] = NeuralTSForecaster._categorize_dataset_size(series_length)

        # 3. Covariates → Architecture hints
        has_past_cov = len(getattr(dataset, "past_covariates", []) or []) > 0
        has_future_cov = len(getattr(dataset, "future_covariates", []) or []) > 0
        if has_past_cov or has_future_cov:
            dataset_info["has_past_exog"] = has_past_cov or has_future_cov  # Any encoder input
            dataset_info["has_future_exog"] = has_future_cov

        return dataset_info

    def _get_loss_function(self) -> torch.nn.Module:
        """
        Factory method to initialize the loss function based on model configuration.
        Supported losses: 'mse', 'l1' (mae), 'huber'.

        Returns:
            nn.Module: The instantiated PyTorch loss function.
        """
        loss_name = self.model_params.get("loss", "mse").lower()
        if loss_name == "mse":
            return torch.nn.MSELoss(reduction="mean")
        elif loss_name in ["l1", "mae"]:
            return torch.nn.L1Loss(reduction="mean")
        elif loss_name == "huber":
            # Default delta is 1.0, but can be customized via 'loss_params'
            loss_params = self.model_params.get("loss_params", {})
            delta = float(loss_params.get("delta", 1.0))
            logger.info(f"Using HuberLoss with delta={delta}")
            return torch.nn.HuberLoss(reduction="mean", delta=delta)
        else:
            raise ValueError(f"Unsupported loss function: '{loss_name}'. Supported: 'mse', 'l1', 'huber'.")

    def _get_y_window_steps(self) -> int:
        """
        Get the number of target steps for creating sliding windows.

        Returns:
            Number of steps (defaults to forecast_steps, can be overridden by subclasses).
        """
        return self.forecast_steps

    def _fit_and_evaluate_fold(
            self,
            train_fold: pd.DataFrame,
            eval_fold: pd.DataFrame,
            validation_params: Dict[str, Any],
            dataset: TimeSeriesDataset,
            is_final_fit: bool = False,
            optuna_trial: Optional[Any] = None,
            trial_step_offset: int = 0
    ) -> Tuple[float, pd.DataFrame, Dict[str, List[float]]]:
        """
        Train with internal Early Stopping (ES) on the TRAIN fold only,
        then score on the EXTERNAL eval fold.

        Args:
            train_fold: Training data fold.
            eval_fold: External evaluation fold (validation or test) aligned in time.
            validation_params: Validation/HPO configuration (e.g., ES %, n_folds, metrics).
            dataset: The dataset object providing column context and accessors.
            is_final_fit: True for backtesting, False for HPO

        Returns:
            float: The validation loss (e.g., MSE) on the hold-out set.
        """

        es_pct = validation_params.get("early_stopping_validation_percentage", 10)

        # 1) Train with ES on train_fold
        val_loss, history = self.fit(
            train_series = train_fold,
            is_final_fit = is_final_fit,
            early_stopping_validation_percentage = es_pct,
            dataset = dataset,
            optuna_trial=optuna_trial,
            trial_step_offset=trial_step_offset
        )

        # 2) Build prediction inputs for the external eval fold
        window_size = getattr(self, "window_size", self.forecast_steps)
        context = train_fold.iloc[-window_size:]

        # Get all covariate columns (both past and future, deduplicated)
        past_cov = list(getattr(dataset, "past_covariates", []) or [])
        future_cov = list(getattr(dataset, "future_covariates", []) or [])
        all_exog = sorted(list(set(past_cov + future_cov)))

        # IMPORTANT: future_exog should only contain future_covariates, not all exog
        future_exog = eval_fold[future_cov] if future_cov else None
        y_true = eval_fold[dataset.target_columns]

        preds = self.predict(input_data=context, future_exog=future_exog)
        metric_to_optimize = validation_params.get("metric", "mse")
        loss = self.evaluate(y_true, preds,dataset=dataset,metric_name=metric_to_optimize)
        return loss, preds, history

    def fit(
            self,
            train_series: pd.DataFrame,
            is_final_fit: bool = False,
            early_stopping_validation_percentage: Optional[float] = None,
            dataset: Optional[TimeSeriesDataset] = None,
            optuna_trial: Optional[Any] = None,
            trial_step_offset: int = 0
    ) -> Tuple[float, Dict[str, List[float]]]:
        """
        Fits the neural network model to the training data.

        This method orchestrates the entire fitting pipeline for neural forecasters.
        Its responsibilities include:
        1.  Initializing the preprocessor with the correct column context from the `dataset` object.
        2.  Applying the preprocessing pipeline to the training data.
        3.  Identifying the column indices for targets and future-known (decoder) exogenous variables.
        4.  Calling `create_sliding_window` to transform the flat time series into input windows (X)
            and corresponding future targets (y_targets and y_decoder_exog).
        5.  Splitting the generated windows into training and validation sets for early stopping.
        6.  Converting the NumPy arrays into PyTorch tensors.
        7.  Calling the model-specific `_train_model` method, passing all necessary tensors.
        8.  Returning the best validation loss achieved during training.

        Args:
            train_series (pd.DataFrame): The DataFrame containing the training data for this run.
            is_final_fit (bool): If True, the model is trained on the entire `train_series`
                without creating a validation split. Defaults to False.
            early_stopping_validation_percentage (Optional[float]): The percentage of the
                training data to be used as a validation set for early stopping.
                Used only if `is_final_fit` is False. Defaults to None.
            dataset (Optional[TimeSeriesDataset]): The `TimeSeriesDataset` object for the
                current run. It is required to provide context about column roles
                (targets, encoder/decoder exogenous).

        Returns:
            float: The best validation loss achieved during training. Returns 0.0 if
                `is_final_fit` is True, as no evaluation is performed.

        Raises:
            ValueError: If the `dataset` object is not provided, or if essential
                parameters like `window_size` are missing.
        """
        logger.info(f"[{self.__class__.__name__}] Starting fit process...")

        # The dataset object is crucial for providing context about column roles.
        if not dataset:
            raise ValueError("A 'dataset' object must be provided to the fit method for NeuralTSForecaster.")

        preprocessing_config = self.model_params.get("preprocessing", {})
        self.preprocessor = Preprocessor(
            preprocessing_config,
            target_columns=dataset.target_columns,
            # Combine all covariate columns for the preprocessor
            exog_columns=(dataset.past_covariates or []) + (dataset.future_covariates or [])
        )

        # Calculate indices based on the actual column order returned by the preprocessor.
        train_proc = self.preprocessor.fit_transform(train_series)

        all_cols = list(train_proc.columns)
        target_indices = [all_cols.index(c) for c in dataset.target_columns if c in all_cols]
        future_cov_cols = getattr(dataset, "future_covariates", []) or []
        decoder_exog_indices = [all_cols.index(c) for c in future_cov_cols if c in all_cols]

        # Create the sliding windows, now separating targets and future exogenous variables
        steps_for_window = self._get_y_window_steps()

        X_all, y_targets_all, y_decoder_exog_all = create_sliding_window(
            data=train_proc.values,
            window_size=self.window_size,
            forecast_steps=steps_for_window,
            target_indices=target_indices,
            decoder_exog_indices=decoder_exog_indices,
            exog_forecast_steps=self.forecast_steps
        )
        # Ensure encoder inputs contain only encoder features (no decoder-only exogenous)
        X_all = X_all[:, :, self.feature_layout.encoder_feature_idx]

        # --- LSTM Train/Inference Mismatch Mitigation: Input Noise Injection ---
        # Add noise to target features in last timestep to simulate prediction errors
        if self.model_params.get("input_noise_injection", {}).get("enabled", False):
            noise_config = self.model_params["input_noise_injection"]
            noise_std = noise_config.get("std", 0.01)
            prob = noise_config.get("probability", 1.0)

            # Runtime validation (defense in depth - also validated in config parser)
            if noise_std <= 0:
                raise ValueError(
                    f"input_noise_injection.std must be positive, got {noise_std}"
                )
            if not 0.0 <= prob <= 1.0:
                raise ValueError(
                    f"input_noise_injection.probability must be in [0, 1], got {prob}"
                )

            n_samples = X_all.shape[0]
            target_dim = self.feature_layout.target_size  # Number of target features

            # Apply noise only to target features (not exogenous/future covariates)
            mask = np.random.random(n_samples) < prob
            noise = np.random.normal(0, noise_std, X_all[mask, -1:, :target_dim].shape)
            X_all[mask, -1:, :target_dim] += noise

            logger.info(f"[Input Noise] Augmented {mask.sum()}/{n_samples} samples "
                        f"(std={noise_std}, prob={prob}, target_dim={target_dim})")

        # --- Split windows into training and validation sets ---
        X_train, y_train = np.array([]), np.array([])
        X_val, y_val = np.array([]), np.array([])
        total_windows = X_all.shape[0]

        if is_final_fit:
            X_train, y_train = X_all, y_targets_all
            logger.info(f"[{self.__class__.__name__}] Final training on {X_train.shape[0]} windows.")
        else:
            # Determine the number of windows for the validation set
            min_absolute_val_windows = self.forecast_steps

            # Smart handling of percentage vs fraction
            es_val = early_stopping_validation_percentage if early_stopping_validation_percentage is not None else 0.0
            ratio = es_val if es_val <= 1.0 else es_val / 100.0

            num_val_windows = (
                max(
                    min_absolute_val_windows,
                    int(total_windows * ratio)
                )
                if es_val > 0 else min_absolute_val_windows
            )

            logger.info(
                f"[{self.__class__.__name__}] Total windows: {total_windows}, Validation windows: {num_val_windows}"
            )

            if total_windows > num_val_windows:
                X_train = X_all[:-num_val_windows]
                y_train = y_targets_all[:-num_val_windows]
                X_val = X_all[-num_val_windows:]
                y_val = y_targets_all[-num_val_windows:]
            else:
                logger.warning(
                    f"[{self.__class__.__name__}] Not enough windows ({total_windows}) for validation "
                    f"(needed: {num_val_windows}). Training on full fold without validation."
                )
                X_train, y_train = X_all, y_targets_all

        if X_train.shape[0] == 0:
            logger.error(f"[{self.__class__.__name__}] No training windows were generated from the data.")
            self.fitted = False
            return float("inf"), {}

        # Also split the decoder-specific exogenous tensor
        y_decoder_exog_train, y_decoder_exog_val = None, None
        if y_decoder_exog_all is not None:
            if is_final_fit:
                y_decoder_exog_train = y_decoder_exog_all
            elif total_windows > num_val_windows:
                y_decoder_exog_train = y_decoder_exog_all[:-num_val_windows]
                y_decoder_exog_val = y_decoder_exog_all[-num_val_windows:]
            else:
                y_decoder_exog_train = y_decoder_exog_all

        # --- Convert all data arrays to PyTorch Tensors ---
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = (torch.FloatTensor(X_val) if X_val.shape[0] > 0 else torch.empty(0)).to(self.device)
        y_val_tensor = (torch.FloatTensor(y_val) if y_val.shape[0] > 0 else torch.empty(0)).to(self.device)
        y_dec_exog_train_tensor = torch.FloatTensor(y_decoder_exog_train).to(
            self.device) if y_decoder_exog_train is not None else None
        y_dec_exog_val_tensor = torch.FloatTensor(y_decoder_exog_val).to(
            self.device) if y_decoder_exog_val is not None else None

        if self.model is None:
            raise NotImplementedError("Model (self.model) must be initialized in the subclass.")

        self.model.to(self.device)

        # --- Configure Training Duration & Early Stopping ---
        # Default for HPO/Normal training
        current_patience = self.model_params.get("early_stopping_patience", 10)

        if is_final_fit:
            # 1. Disable Early Stopping for backtesting (train on full data)
            current_patience = None

            # 2. Adjust Epochs based on HPO stats if available
            if 'mean_trained_epochs' in self.model_params:
                mean_epochs = self.model_params['mean_trained_epochs']
                std_epochs = self.model_params.get('std_trained_epochs', 0.0)

                # Formula: mean + 1.0 * std (Safe margin)
                target_epochs = int(np.ceil(mean_epochs + 1.0 * std_epochs))
                # Ensure reasonable minimum
                target_epochs = max(target_epochs, 5)

                logger.info(
                    f"[{self.__class__.__name__}] Final Fit: Adjusting epochs from HPO stats. "
                    f"Mean={mean_epochs:.1f}, Std={std_epochs:.1f} -> Target={target_epochs}"
                )

                # Apply to model params so _train_model picks it up
                self.model_params['epochs'] = target_epochs
            else:
                logger.info(
                    f"[{self.__class__.__name__}] Final Fit: "
                    f"Using config epochs ({self.model_params.get('epochs')})"
                )

        # Determine if we are in HPO mode (lightweight)
        is_hpo = self.run_context.metadata.get('is_hpo_trial', False) if self.run_context else False

        # Extract config for gradient monitor
        grad_mon_config = self.model_params.get("gradient_monitor", {})
        log_interval = grad_mon_config.get("log_interval", 50)  # Default to 50
        should_enable = grad_mon_config.get("enabled", False)

        # Enable only if:
        # 1. Config explicitly says 'enabled: True'
        # 2. Context exists (to create file path)
        # 3. It's NOT an HPO trial (to save resources)
        if should_enable and self.run_context and not is_hpo:
            self.gradient_monitor = GradientMonitor(
                model=self.model,
                save_dir=self.run_context.gradients_dir,
                model_name=self.run_context.model_name,
                fold_idx=self.run_context.fold_idx,
                window_size=self.run_context.window_size,
                enabled=True,
                log_interval=log_interval
            )
            logger.info(f"[{self.__class__.__name__}] Gradient monitoring enabled")
        else:
            self.gradient_monitor = None

        fail_on_instability = self.model_params.get('fail_on_numerical_instability', False)

        # Call the model-specific training method, passing all tensors
        trained_model_instance = self._train_model(
            X_train=X_train_tensor, y_train=y_train_tensor,
            X_val=X_val_tensor, y_val=y_val_tensor,
            y_decoder_exog_train=y_dec_exog_train_tensor,
            y_decoder_exog_val=y_dec_exog_val_tensor,
            dataset=dataset,  # Pass the dataset object for model-specific context
            early_stopping_patience=current_patience,
            optuna_trial=optuna_trial,
            trial_step_offset=trial_step_offset,
            gradient_monitor=self.gradient_monitor,
            fail_on_instability=fail_on_instability
        )
        self.model = trained_model_instance
        val_loss = getattr(trained_model_instance, "best_val_loss", float("inf"))

        # Close gradient monitoring stream
        if self.gradient_monitor:
            self.gradient_monitor.close()
            logger.info(f"[{self.__class__.__name__}] Gradient monitoring complete")

        # Capture history
        history = getattr(self, "training_history", {})

        self.fitted = True
        logger.info(f"[{self.__class__.__name__}] Fit process completed with validation loss: {val_loss:.6f}")
        return val_loss, history

    # ═══════════════════════════════════════════════════════════════
    # INFERENCE HOOKS
    # ═══════════════════════════════════════════════════════════════
    def _inference_context(self):
        """Hook for inference instrumentation (attention capture, profiling, etc)."""
        return contextlib.nullcontext()

    def _internal_predict(self, input_tensor: torch.Tensor, **kwargs) -> np.ndarray:
        """
        LSP-safe internal prediction engine.

        Contract:
          - Input: torch.Tensor (B, W, F_in)
          - Output: np.ndarray (B, H, F_out) or (H, F_out)
          - Subclasses may accept kwargs (e.g., future_exog_tensor). Base ignores unused.
        """
        if self.model is None or not self.fitted:
            raise ValueError("Model must be initialized and fitted before prediction.")

        if input_tensor.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch_size, window_size, n_features), "
                f"but got tensor with shape {tuple(input_tensor.shape)}"
            )

        self.model.eval()
        with torch.inference_mode():
            # Default: single forward pass, ignore kwargs
            output = self.model(input_tensor)
            return output.detach().cpu().numpy()

    def _prepare_input_tensor(self, input_data: pd.DataFrame) -> torch.Tensor:
        """
        Shared: DataFrame -> preprocessor.transform -> FloatTensor -> (1, W, F).
        Subclasses may override (e.g., Transformer slicing).
        """
        input_proc = self.preprocessor.transform(input_data)
        return torch.tensor(input_proc.values, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _prepare_future_exog_tensor(self, future_exog: Optional[pd.DataFrame]) -> Optional[torch.Tensor]:
        """Default: no future exog support."""
        return None

    def _sanitize_predictions_np(self, pred: Any) -> np.ndarray:
        """
        Normalize raw outputs into safe 2D float32 array (H, F).
        - Squeeze (1, H, F) -> (H, F)
        - Reshape (H,) -> (H, 1)
        - Validate shape (H == forecast_steps, F == num_features)
        - Replace any invalid / non-finite with NaN array
        """
        pred_np = np.asarray(pred)
        if pred_np.ndim == 3 and pred_np.shape[0] == 1:
            pred_np = pred_np.squeeze(0)

        if pred_np.ndim == 1:
            pred_np = pred_np.reshape(-1, 1)

        if pred_np.ndim != 2:
            logger.warning(
                f"[{self.__class__.__name__}] Invalid prediction ndim={pred_np.ndim}. "
                f"Expected 2D. Returning NaN array."
            )
            return np.full((self.forecast_steps, self.num_features), np.nan, dtype=np.float32)

        pred_np = pred_np.astype(np.float32, copy=False)

        h, f = pred_np.shape
        if h != self.forecast_steps or f != self.num_features:
            logger.warning(
                f"[{self.__class__.__name__}] Shape mismatch: got {pred_np.shape}, "
                f"expected ({self.forecast_steps}, {self.num_features}). Returning NaN array."
            )
            return np.full((self.forecast_steps, self.num_features), np.nan, dtype=np.float32)

        if not np.isfinite(pred_np).all():
            logger.warning(
                f"[{self.__class__.__name__}] Non-finite values detected. Returning NaN array."
            )
            return np.full((self.forecast_steps, self.num_features), np.nan, dtype=np.float32)

        return pred_np

    def _fallback_nan_dataframe(self, start_after: Optional[Union[pd.Timestamp, int]]) -> pd.DataFrame:
        """
        Last-resort fallback predictions with proper temporal index for downstream alignment.
        Mirrors Preprocessor.inverse_transforms index reconstruction logic.
        """
        cols = self.preprocessor.target_columns
        nan_arr = np.full((self.forecast_steps, self.num_features), np.nan, dtype=np.float32)

        try:
            context_index = self.preprocessor._full_raw_data_context.index

            if isinstance(context_index, pd.DatetimeIndex) and start_after is not None:
                freq = context_index.freq or pd.infer_freq(context_index)
                if freq is not None:
                    pred_index = pd.date_range(
                        start = start_after + pd.tseries.frequencies.to_offset(freq),
                        periods = self.forecast_steps,
                        freq = freq
                    )
                    return pd.DataFrame(nan_arr, index=pred_index, columns=cols)

            if isinstance(context_index, pd.RangeIndex) and isinstance(start_after, int):
                pred_index = pd.RangeIndex(
                    start = start_after + 1,
                    stop = start_after + 1 + self.forecast_steps
                )
                return pd.DataFrame(nan_arr, index=pred_index, columns=cols)

        except Exception as e:
            logger.warning(f"[{self.__class__.__name__}] Could not build fallback index: {e}")

        return pd.DataFrame(nan_arr, columns=cols)

    def predict(
        self,
        input_data: pd.DataFrame,
        future_exog: Optional[pd.DataFrame] = None,
        ** kwargs
    ) -> pd.DataFrame:
        """
        Unified prediction pipeline for neural models:
          1) preprocess -> tensor
          2) optional future exog tensor hook
          3) _internal_predict (LSP-safe kwargs)
          4) sanitize (shape + finite) -> always (H, F)
          5) inverse_transforms with robust fallback (preserve index)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting.")
        if input_data.empty:
            raise ValueError("Input data cannot be empty.")

        start_after_ts = input_data.index[-1]

        try:
            input_tensor = self._prepare_input_tensor(input_data)

            future_exog_tensor = kwargs.pop("future_exog_tensor", None)
            if future_exog_tensor is None:
                batch_size = input_tensor.shape[0]
                future_exog_tensor = self._prepare_future_exog_tensor(future_exog, batch_size=batch_size)

            with self._inference_context():
                raw_pred = self._internal_predict(
                    input_tensor,
                    future_exog_tensor = future_exog_tensor,
                    ** kwargs
                )

            pred_np_2d = self._sanitize_predictions_np(raw_pred)

            try:
                pred_df = self.preprocessor.inverse_transforms(
                    pred_np_2d,
                    start_after = start_after_ts
                )
            except Exception as e_inv:
                logger.warning(
                    f"[{self.__class__.__name__}] inverse_transforms failed: {e_inv}. "
                    f"Returning NaN fallback."
                )
                pred_df = self._fallback_nan_dataframe(start_after=start_after_ts)

            if pred_df.isna().any().any():
                logger.warning(
                    f"[{self.__class__.__name__}] Final predictions contain NaNs."
                )

            return pred_df

        except Exception as e:
            logger.exception(
                f"[{self.__class__.__name__}] Prediction crashed: {e}. Returning NaN fallback."
            )
            return self._fallback_nan_dataframe(start_after=start_after_ts)

    def _predict_iterative_ar(
            self,
            input_tensor: torch.Tensor,
            *,
            window_size: int,
            forecast_steps: int,
            num_features: int,
            encode_fn: Callable[[torch.Tensor], torch.Tensor],
            readout_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Pure autoregressive iterative prediction.
        NO exogenous variables.
        NO PC-mode.
        Single source of truth for AR iterative logic.
        """
        device = input_tensor.device
        B, W, _ = input_tensor.shape
        H = forecast_steps
        F = num_features

        # rolling buffer: [B, W+H, F]
        buffer = torch.zeros(B, W + H, F, device=device, dtype=input_tensor.dtype)
        buffer[:, :W, :] = input_tensor[:, :, :F]

        output = torch.zeros(B, H, F, device=device, dtype=input_tensor.dtype)

        for step in range(H):
            # Attention capture support
            if hasattr(self, 'model') and hasattr(self.model, 'attn_capture'):
                if self.model.attn_capture.enabled:
                    self.model.attn_capture.set_step(step)

            x_step = buffer[:, step:step + W, :]  # [B, W, F]
            enc = encode_fn(x_step)
            one_step = readout_fn(enc)

            # Normalize output to (B, 1, F)
            if one_step.dim() == 4:
                one_step = one_step.squeeze(2)
            if one_step.dim() == 3 and one_step.size(1) > 1:
                one_step = one_step[:, -1:, :]
            elif one_step.dim() == 2:
                one_step = one_step.unsqueeze(1)

            buffer[:, W + step:W + step + 1, :] = one_step
            output[:, step:step + 1, :] = one_step

        return output

    def _validate_model_specific_inputs(
        self, train_series: pd.DataFrame, val_series: Optional[pd.DataFrame] = None,
        forecast_steps: Optional[int] = None
    ) -> None:
        """
        Validate inputs specific to neural models.

        Args:
            train_series: Training data.
            val_series: Validation data (optional). Defaults to None.
            forecast_steps: Forecast steps (optional). Defaults to None.

        Raises:
            ValueError: If inputs are invalid.
        """
        if len(train_series) < self.window_size + self.forecast_steps:
            raise ValueError("Training series too short for specified window_size and forecast_steps.")

    def prepare_data(self, dataset: TimeSeriesDataset) -> List[TimeSeriesDataset]:
        """
        Prepare data for neural models.

        Args:
            dataset: Input dataset.

        Returns:
            List containing the input dataset.
        """
        return [dataset]

    @abstractmethod
    def _train_model(self, *args, **kwargs) -> Any:
        """
        Run the training loop for the neural model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Trained model instance.
        """
        pass

