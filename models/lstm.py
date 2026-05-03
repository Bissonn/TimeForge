r"""
Module for LSTM-based time series forecasting models.

Implements encoder-only LSTM models supporting:
- direct forecasting (one-shot)
- iterative forecasting (autoregressive)

COVARIATE HANDLING (v2 API):
────────────────────────────────────────────────────────────────────
  past_covariates:   Features known only in history (encoder-only)
                     Handled via PastCovariatePolicy (default: FROZEN)

  future_covariates: Features known in both history and future
                     Currently stored but NOT consumed by encoder-only
                     architecture (forward-compatible for future use)

The model operates strictly on historical data during training.

KNOWN LIMITATIONS:
────────────────────────────────────────────────────────────────────
  ⚠️ PastCovariatePolicy.FROZEN: Default policy, but not fully implemented.
     Current behavior: past_covariates frozen at last window value during
     iterative inference. Known tech debt - use with caution in production.

  ⚠️ _prepare_future_exog_tensor: Called O(H) times during iterative inference.
     Performance bottleneck for long horizons. Consider batched refactor if
     profiling shows this as critical path.
"""

import logging
from typing import Dict, Any, Set, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.context import RunContext
from models.base import NeuralTSForecaster, PastCovariatePolicy
from models.model_registry import register_model
from utils.train_loop import run_train_loop
from utils.dataset import TimeSeriesDataset
from utils.scheduler import create_scheduler

from models.hpo_heuristics import (
    categorize_dataset_size,
    sqrt_lr_scale,
    clamp,
    get_lr_scaling_config,
    get_complexity_threshold,
    clamp_lr_by_dataset
)

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """ Standard LSTM neural network architecture for time series forecasting."""

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_steps: int,
            output_features: int,
            dropout: float,
            fc_dropout: float = 0.0
    ) -> None:
        """
        Initializes the LSTM model layers.

        Args:
            input_size (int): The number of input features per time step
                (targets + all exogenous variables).
            hidden_size (int): The number of features in the hidden state of the LSTM.
            num_layers (int): The number of recurrent LSTM layers.
            output_steps (int): The number of time steps in the output sequence. This is
                equal to the forecast horizon for direct strategy, and 1 for iterative.
            output_features (int): The number of target features to be forecasted.
            dropout (float): The dropout rate for all but the last LSTM layer.
            fc_dropout (float): The dropout rate before the fully connected layer.
                Applied to the final hidden state. Always active (works even for num_layers=1).
        """
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1 or output_steps < 1 or output_features < 1:
            raise ValueError("input_size, hidden_size, num_layers, output_steps, and output_features must be positive.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0.")
        if not 0.0 <= fc_dropout <= 1.0:
            raise ValueError("fc_dropout must be between 0.0 and 1.0.")
        if num_layers == 1 and dropout != 0.0:
            logger.warning(
                "Dropout is not applied for a single-layer LSTM. Setting it to 0.0.")
            dropout = 0.0

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(hidden_size, output_steps * output_features)
        self.output_steps = output_steps
        self.output_features = output_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the LSTM model.

        Args:
            x (torch.Tensor): The input tensor of shape
                (batch_size, window_size, input_size).

        Returns:
            torch.Tensor: The output tensor of shape
                (batch_size, output_steps, output_features).
        """
        if x.dim() != 3 or x.size(2) != self.lstm.input_size:
            raise ValueError(f"Expected input shape (batch_size, window_size, {self.lstm.input_size}), got {x.shape}")
        output, _ = self.lstm(x)
        # Extract last timestep hidden state and apply dropout (mismatch mitigation)
        h_last = output[:, -1, :]
        h_last = self.fc_dropout(h_last)
        # Pass to the fully connected layer
        prediction = self.fc(h_last)
        # Reshape the output to (batch_size, output_steps, output_features)
        return prediction.reshape(-1, self.output_steps, self.output_features)


class FutureContextEncoder(nn.Module):
    """
    Encodes future exogenous variables into a fixed-size context vector.

    Used for global conditioning in Direct mode. Supports multiple pooling
    strategies, optional compression, and regularization.

    Args:
        future_cov_size: Number of future covariate features
        forecast_steps: Number of forecast steps (temporal dimension)
        pooling: Pooling strategy - "mean", "last", or "learnable"
        compression_dim: Optional compression dimension (if < future_cov_size)
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        future_cov_size: int,
        forecast_steps: int,
        pooling: str = "mean",
        compression_dim: Optional[int] = None,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        if future_cov_size < 1:
            raise ValueError("future_cov_size must be positive")
        if forecast_steps < 1:
            raise ValueError("forecast_steps must be positive")
        if pooling not in ["mean", "last", "learnable"]:
            raise ValueError(f"pooling must be 'mean', 'last', or 'learnable', got '{pooling}'")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")

        self.pooling = pooling
        self.forecast_steps = forecast_steps

        # Optional compression layer
        if compression_dim is not None and compression_dim < future_cov_size:
            self.compression = nn.Linear(future_cov_size, compression_dim)
            effective_dim = compression_dim
        else:
            self.compression = None
            effective_dim = future_cov_size

        # Learnable attention pooling
        if pooling == "learnable":
            self.attn_pooling = nn.Linear(forecast_steps, 1)
        else:
            self.attn_pooling = None

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.output_size = effective_dim

    def forward(self, future_exog: torch.Tensor) -> torch.Tensor:
        """
        Encode future exogenous variables into context vector.

        Args:
            future_exog: (batch_size, forecast_steps, future_cov_size)

        Returns:
            context: (batch_size, output_size)
        """
        if future_exog.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor (batch, steps, features), got shape {future_exog.shape}"
            )

        B, H, F = future_exog.shape
        if H != self.forecast_steps:
            raise ValueError(
                f"Expected {self.forecast_steps} forecast steps, got {H}"
            )

        # Optional compression: (B, H, F) -> (B, H, F_compressed)
        x = self.compression(future_exog) if self.compression is not None else future_exog

        # Temporal pooling: (B, H, F) -> (B, F)
        if self.pooling == "mean":
            ctx = x.mean(dim=1)
        elif self.pooling == "last":
            ctx = x[:, -1, :]
        elif self.pooling == "learnable":
            # Learnable attention weights: (B, F, H) -> (B, F, 1) -> (B, F, H)
            weights = torch.softmax(self.attn_pooling(x.transpose(1, 2)), dim=2)
            ctx = (x * weights.transpose(1, 2)).sum(dim=1)
        else:
            raise RuntimeError(f"Unexpected pooling mode: {self.pooling}")

        # Regularization
        return self.dropout(ctx)


class LSTMModelWithFutureContext(nn.Module):
    """
    LSTM model with optional future context injection for Direct mode.

    Backward-compatible wrapper that composes:
    - Base LSTMModel (unchanged, handles historical sequence)
    - Optional FutureContextEncoder (for future exogenous variables)
    - Joint projection head (combines LSTM hidden state + future context)

    When future_ctx_encoder is None, behaves identically to base LSTMModel.
    When enabled, performs late concatenation: hidden + context -> output.

    REGULARIZATION:
    The forward() method applies base_model.fc_dropout to h_last before using
    it in either path (standard or enhanced). This ensures consistent dropout
    behavior for train/inference mismatch mitigation.

    Args:
        base_model: Standard LSTMModel instance
        future_ctx_encoder: Optional FutureContextEncoder instance
    """

    def __init__(
        self,
        base_model: LSTMModel,
        future_ctx_encoder: Optional[FutureContextEncoder] = None
    ) -> None:
        super().__init__()

        self.base_model = base_model
        self.future_ctx_encoder = future_ctx_encoder

        # If future context enabled, create joint projection head
        if future_ctx_encoder is not None:
            hidden_size = base_model.lstm.hidden_size
            ctx_size = future_ctx_encoder.output_size
            joint_size = hidden_size + ctx_size

            output_dim = base_model.output_steps * base_model.output_features
            self.joint_fc = nn.Linear(joint_size, output_dim)

            # Store output shape for reshaping
            self.output_steps = base_model.output_steps
            self.output_features = base_model.output_features

            logger.info(
                f"[LSTMModelWithFutureContext] Late concatenation enabled: "
                f"hidden({hidden_size}) + context({ctx_size}) -> fc({output_dim})"
            )
        else:
            self.joint_fc = None
            self.output_steps = base_model.output_steps
            self.output_features = base_model.output_features

    def forward(
        self,
        x: torch.Tensor,
        tgt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional future context conditioning.

        Args:
            x: (batch_size, window_size, input_size) - historical input
            tgt: (batch_size, forecast_steps, future_cov_size) - future exogenous (optional)
                 Named 'tgt' for API consistency with Transformer (training loop compatibility)

        Returns:
            predictions: (batch_size, output_steps, output_features)
        """
        # Alias for internal clarity
        future_ctx = tgt

        # Extract LSTM hidden state from last timestep
        lstm_output, _ = self.base_model.lstm(x)
        h_last = lstm_output[:, -1, :]  # (B, hidden_size)

        # Apply fc_dropout for train/inference mismatch mitigation
        # (consistent with base model's forward pass)
        h_last = self.base_model.fc_dropout(h_last)

        # Standard path: no future context or encoder disabled
        if future_ctx is None or self.future_ctx_encoder is None:
            prediction = self.base_model.fc(h_last)
            return prediction.reshape(-1, self.output_steps, self.output_features)

        # Enhanced path: late concatenation with future context
        ctx = self.future_ctx_encoder(future_ctx)  # (B, ctx_size)
        joint = torch.cat([h_last, ctx], dim=1)    # (B, hidden + ctx)
        prediction = self.joint_fc(joint)

        return prediction.reshape(-1, self.output_steps, self.output_features)

    @property
    def lstm(self):
        """Backward compatibility: expose base LSTM layer."""
        return self.base_model.lstm

    @property
    def fc(self):
        """Backward compatibility: expose fc layer (joint or base depending on config)."""
        if self.joint_fc is not None:
            return self.joint_fc
        return self.base_model.fc


@register_model("lstm", is_univariate=False)
class LSTMForecaster(NeuralTSForecaster):
    """
    Unified LSTM forecaster supporting both 'direct' and 'iterative' strategies.
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
        Initializes the base LSTM forecaster.

        It dynamically sets the `input_size` based on the columns present in the
        provided `run_dataset` object, ensuring the model adapts to the specific
        set of features used for a particular run.
        """
        # ============================================================
        # Initialize prediction strategy
        # ============================================================
        self.strategy = model_params.get("strategy", "direct")
        if self.strategy not in ["direct", "iterative"]:
            raise ValueError(f"Invalid strategy '{self.strategy}'. Must be 'direct' or 'iterative'.")
        # ============================================================
        # Base initialization
        # ============================================================
        super().__init__(
            model_params=model_params,
            num_features=num_features,
            forecast_steps=forecast_steps,
            window_size=window_size,
            dataset=dataset,
            run_context=run_context,
            **kwargs)

        self.architecture = model_params.get("architecture", "encoder-only")
        self._validate_model_params()

        # ============================================================
        # STEP 3.5: Configure covariate handling
        # ============================================================
        # Policy for handling past_covariates during iterative prediction
        policy_str = model_params.get("past_covariate_policy", "frozen")
        try:
            self.past_covariate_policy = PastCovariatePolicy(policy_str)
        except ValueError:
            logger.warning(
                f"Invalid past_covariate_policy '{policy_str}'. "
                f"Using default FROZEN policy."
            )
            self.past_covariate_policy = PastCovariatePolicy.FROZEN

        # ============================================================
        # STEP 3.6: Configure future covariate mode
        # ============================================================
        self.future_covariate_mode = model_params.get("future_covariate_mode", "none")

        # Validate mode
        valid_modes = ["none", "global", "stepwise"]  # stepwise → NotImplementedError below
        if self.future_covariate_mode not in valid_modes:
            raise ValueError(
                f"Invalid future_covariate_mode='{self.future_covariate_mode}'. "
                f"Valid modes: {valid_modes}"
            )

        # Check for not-yet-implemented modes (Phase 2b)
        # stepwise is now implemented in Phase 2a

        # Validate mode vs strategy compatibility
        if self.future_covariate_mode == "global" and self.strategy != "direct":
            raise ValueError(
                f"future_covariate_mode='global' requires strategy='direct'. "
                f"Got strategy='{self.strategy}'."
            )

        if self.future_covariate_mode == "stepwise" and self.strategy != "iterative":
            raise ValueError(
                f"future_covariate_mode='stepwise' requires strategy='iterative'. "
                f"Got strategy='{self.strategy}'."
            )

        # Log warning if future_covariates declared but mode=none
        if (self.future_covariate_mode == "none" and
            self.feature_layout.future_covariates_size > 0):
            logger.warning(
                f"Dataset declares {self.feature_layout.future_covariates_size} "
                f"future_covariates, but future_covariate_mode='none'. "
                f"Future exog will be IGNORED during prediction. "
                f"Set future_covariate_mode='global' (direct) or 'stepwise' (iterative) "
                f"to enable future context."
            )

        # Log training-inference mismatch warning for stepwise mode
        if self.future_covariate_mode == "stepwise":
            logger.warning(
                f"[LSTM Iterative + Stepwise] Training-inference mismatch: "
                f"Model will use teacher forcing during training but autoregressive "
                f"predictions during inference. This may lead to error accumulation "
                f"for long forecast horizons. Recommended max horizon: {min(self.forecast_steps, 48)} steps. "
                f"See documentation for details."
            )

        # ============================================================
        # STEP 3.7: Configure stateful iterative prediction
        # ============================================================
        # Whether to propagate LSTM hidden state (h, c) across autoregressive steps
        # Only applies to iterative strategy
        # Default: True (17% better than stateless)
        self.iterative_stateful = model_params.get("iterative_stateful", True)

        if self.strategy == "iterative" and self.iterative_stateful:
            logger.info(
                "[LSTM Iterative] Using stateful prediction: LSTM state (h, c) "
                "will be propagated across forecast steps for improved long-horizon accuracy."
            )

        # ============================================================
        # STEP 4: Build Model
        # ============================================================
        model_output_steps = self.forecast_steps if self.strategy == "direct" else 1

        # TRAINING input size (always encoder layout)
        train_input_size = self.feature_layout.encoder_input_size

        # Build base LSTM model
        base_lstm = LSTMModel(
            input_size=train_input_size,
            hidden_size=self.model_params.get("hidden_size", 50),
            num_layers=self.model_params.get("num_layers", 1),
            output_steps=model_output_steps,
            output_features=self.num_features,
            dropout=self.model_params.get("dropout", 0.0),
            fc_dropout=self.model_params.get("fc_dropout", 0.0),
        )

        # Validate stepwise mode requirements
        if self.future_covariate_mode == "stepwise":
            if self.feature_layout.future_covariates_size == 0:
                raise ValueError(
                    "future_covariate_mode='stepwise' requires future_covariates in dataset, "
                    "but none were declared. Please add future_covariates to dataset config."
                )

        # Build future context encoder (if enabled for global mode)
        future_ctx_encoder = None
        if self.future_covariate_mode == "global":
            future_cov_size = self.feature_layout.future_covariates_size

            if future_cov_size == 0:
                raise ValueError(
                    "future_covariate_mode='global' requires future_covariates in dataset, "
                    "but none were declared. Please add future_covariates to dataset config."
                )

            # Extract configuration for future context encoder
            future_ctx_config = model_params.get("future_context_config", {})
            pooling = future_ctx_config.get("pooling", "mean")
            compression_dim = future_ctx_config.get("compression_dim", None)
            dropout = future_ctx_config.get("dropout", 0.0)

            future_ctx_encoder = FutureContextEncoder(
                future_cov_size=future_cov_size,
                forecast_steps=self.forecast_steps,
                pooling=pooling,
                compression_dim=compression_dim,
                dropout=dropout,
            )

            logger.info(
                f"Enabled global future context: pooling={pooling}, "
                f"compression_dim={compression_dim}, dropout={dropout}"
            )

        # Wrap with context support
        self.model = LSTMModelWithFutureContext(
            base_model=base_lstm,
            future_ctx_encoder=future_ctx_encoder
        ).to(self.device)

        # Make sure the attribute exists, even if not set by dataset config
        self.target_columns = None

        logger.info(
            f"Initialized LSTMForecaster (strategy={self.strategy}, "
            f"past_covariate_policy={self.past_covariate_policy.value})."
        )

    # =========================================================================
    # SMART HPO IMPLEMENTATION
    # =========================================================================

    def _default_lr_for_lstm(
            self,
            hidden_size: int,
            num_layers: int,
            strategy: str,
            dataset_size: str,
            batch_size: int = 32
    ) -> float:
        """
        LSTM-specific LR heuristic with batch scaling and hard clamping.

        LSTMs are strictly less stable than Transformers at high LRs,
        so we apply conservative overrides and stricter penalties.
        """
        mode, ref_batch, lr0 = get_lr_scaling_config(self.model_params)

        # Override generic 1e-3 with 3e-4 for safety
        if "hpo_lr_scaling" not in self.model_params:
            lr0 = 3e-4

        scale = 1.0

        # Hidden Size Adjustment (LSTM prone to exploding gradients with width)
        if hidden_size >= 256: scale *= 0.6
        if hidden_size >= 512: scale *= 0.5

        # Layer Count Adjustment (LSTM notoriously hard to train deep)
        if num_layers >= 3: scale *= 0.7
        if num_layers >= 4: scale *= 0.6  # Deeper = much lower LR

        # Strategy Adjustment (Iterative accumulation of error)
        if strategy == "iterative":
            scale *= 0.6  # Aggressive reduction for iterative LSTM

        # Dataset Size Adjustment
        if dataset_size == "small":
            scale *= 0.7
        elif dataset_size in ["large", "very_large"]:
            scale *= 1.2

        lr = lr0 * scale

        # Soft batch scaling
        lr = sqrt_lr_scale(lr, batch_size, ref_batch, mode)

        # [HARD CLAMP] Dataset ceiling
        lr = clamp_lr_by_dataset(lr, dataset_size)

        logger.debug(
            f"[HPO] LSTM LR: base={lr0:.2e}, scale={scale:.3f}, "
            f"batch={batch_size} → lr={lr:.2e}"
        )

        return lr

    def validate_param_combination(self, params: Dict[str, Any]) -> bool:
        """
        Validate LSTM HPO candidate params.
        Enforces complexity thresholds and LSTM-specific constraints.
        """
        # 1. Existing checks (Hidden size vs Window, Dropout)
        if "hidden_size" in params and params["hidden_size"] > 4 * self.window_size:
            return False

        # Dropout is invalid for single-layer LSTM in PyTorch
        if params.get("num_layers", 2) == 1 and params.get("dropout", 0.0) > 0.0:
            return False

        # 2. Complexity Check

        # Robust dataset length resolution
        if hasattr(self, 'run_context') and self.run_context and hasattr(self.run_context, 'dataset'):
            training_len = len(self.run_context.dataset.series)
        elif hasattr(self, 'dataset') and self.dataset:
            training_len = len(self.dataset.series)
        else:
            training_len = 5000  # Fallback

        size_category = categorize_dataset_size(training_len)

        hidden_size = params.get("hidden_size", 128)
        num_layers = params.get("num_layers", 2)

        # Use centralized threshold (pass model_type="lstm"!)
        threshold = get_complexity_threshold(
            self.model_params,
            size_category,
            model_type="lstm",
            num_features=self.num_features
        )

        complexity = hidden_size * num_layers

        # Strict inequality
        if complexity > threshold:
            logger.debug(
                f"[HPO] Rejected: LSTM complexity={complexity} > threshold {threshold} "
                f"({size_category} dataset)"
            )
            return False

        return True

    def filter_search_space(self, param_space: Dict[str, Any], fixed_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter LSTM search space.
        Removes 'dropout' if num_layers is 1 (as it has no effect in PyTorch LSTM).
        """
        filtered = param_space.copy()

        # If num_layers is fixed to 1, remove dropout from optimization
        if fixed_params.get("num_layers") == 1:
            if "dropout" in filtered:
                logger.info("[SmartHPO] Removing 'dropout' from search space (num_layers=1).")
                filtered.pop("dropout", None)

        return filtered

    def suggest_smart_priors(
            self,
            param_space: Dict[str, Any],
            fixed_params: Dict[str, Any],
            dataset: Optional[Any] = None  # ← ADD dataset parameter
    ) -> List[Dict[str, Any]]:
        """
        Suggest smart starting points for LSTM HPO.

        Adapts to dataset size: small → high dropout, large → low dropout
        """
        # Determine dataset size
        if dataset is not None:
            training_length = getattr(dataset, 'training_length', None)
            if training_length is None:
                series = getattr(dataset, 'series', None)
                training_length = len(series) if series is not None else 0

            if training_length < 1000:
                size_category = "small"
            elif training_length < 10000:
                size_category = "medium"
            else:
                size_category = "large"
        else:
            size_category = "medium"

        # Dataset-aware dropout ranges
        if size_category == "small":
            dropout_moderate = 0.3
            dropout_high = 0.4
        elif size_category == "medium":
            dropout_moderate = 0.2
            dropout_high = 0.3
        else:  # large
            dropout_moderate = 0.1
            dropout_high = 0.2

        # Feature-aware hidden size
        num_features = getattr(self, 'num_features', 1)
        base_hidden = 64 if num_features <= 16 else 128

        # Generate candidates
        candidates = [
            {"hidden_size": base_hidden, "num_layers": 1, "dropout": 0.0, "batch_size": 64},
            {"hidden_size": base_hidden, "num_layers": 2, "dropout": dropout_moderate, "batch_size": 64},
            {"hidden_size": base_hidden * 2, "num_layers": 2, "dropout": dropout_moderate, "batch_size": 64},
            {"hidden_size": base_hidden, "num_layers": 2, "dropout": dropout_high, "batch_size": 32},
        ]

        # Filter to param_space
        priors = []
        for candidate in candidates:
            filtered = {k: v for k, v in candidate.items() if k in param_space}
            if filtered:
                priors.append(filtered)

        return priors

    def _validate_model_params(self) -> None:
        """
        Validates essential LSTM model parameters from the configuration.
        """
        required = { "hidden_size", "num_layers"}
        missing = [p for p in required if p not in self.model_params]
        if missing:
            raise ValueError(f"Missing required LSTM parameter(s): {missing}")
        if not isinstance(self.model_params.get("hidden_size"), int) or self.model_params["hidden_size"] < 1:
            raise ValueError("hidden_size must be a positive integer.")
        if not isinstance(self.model_params.get("num_layers"), int) or self.model_params["num_layers"] < 1:
            raise ValueError("num_layers must be a positive integer.")

        dropout = self.model_params.get("dropout", 0.0)
        if not isinstance(dropout, (float, int)) or not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be a float between 0.0 and 1.0.")

    def get_valid_params(self) -> Set[str]:
        """
        Returns the set of valid hyperparameter names for the LSTM model.
        """
        return {
            "hidden_size", "num_layers", "dropout", "batch_size", "learning_rate",
            "epochs", "early_stopping_patience", "weight_decay", "n_trials", "strategy",
        }

    def filter_candidates(self, candidates: List[Dict[str, Any]], model_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filters hyperparameter combinations based on model-specific constraints.

        For LSTMs, this prevents using a `hidden_size` that is excessively large
        compared to the `window_size`, which can be a heuristic to avoid overfitting.

        Args:
            candidates (List[Dict[str, Any]]): A list of candidate parameter dictionaries
                generated by the hyperparameter optimization strategy.
            model_params (Dict[str, Any]): The base model parameters.

        Returns:
            List[Dict[str, Any]]: A filtered list of valid candidate parameter dictionaries.
        """
        return [c for c in candidates
                if c.get("hidden_size", model_params.get("hidden_size", 10)) <= 4 * self.window_size]

    def _get_y_window_steps(self) -> int:
        return self.forecast_steps if self.strategy == "direct" else 1

    def _train_model(self,
                     X_train: torch.Tensor,
                     y_train: torch.Tensor,
                     X_val: torch.Tensor,
                     y_val: torch.Tensor,
                     dataset: TimeSeriesDataset,
                     **kwargs
                     ) -> nn.Module:
        """
        Train LSTM model on historical windows.

        Training data strictly reflects past observations.
        No inference-time data alignment is performed here.
        """
        # Extract future covariates from kwargs (analogous to Transformer)
        future_cov_train = kwargs.get('y_decoder_exog_train')
        future_cov_val = kwargs.get('y_decoder_exog_val')

        try:
            # Use centralized optimizer creation (handles Adam/AdamW, fused, epsilon-safety)
            optimizer = self._create_optimizer()

            # ─────────────────────────────────────────────────────────────
            # CREATE SCHEDULER (if configured)
            # ─────────────────────────────────────────────────────────────
            scheduler = None
            scheduler_config = self.model_params.get("scheduler_config", {})
            if scheduler_config and scheduler_config.get("type"):
                scheduler = create_scheduler(
                    optimizer=optimizer,
                    scheduler_config=scheduler_config,
                    train_size=X_train.shape[0],
                    batch_size=self.model_params.get("batch_size", 32),
                    max_epochs=self.model_params.get("epochs", 100),
                    default_lr=self.model_params.get("learning_rate", 0.001)
                )
            # ─────────────────────────────────────────────────────────────
            criterion = self._get_loss_function()

            # --- Use patience passed from base.fit() ---
            # CRITICAL: Don't use 'or' - None is a valid value (disables early stopping)
            if "early_stopping_patience" in kwargs:
                patience = kwargs["early_stopping_patience"]  # Can be None (for is_final_fit)
            else:
                patience = self._get_training_param("early_stopping_patience")

            # Advanced training params
            use_amp = self.model_params.get("use_amp", True)
            max_grad_norm = self.model_params.get("max_grad_norm", 1.0)

            # per horizon diagnostics
            save_horizon_csv = self.model_params.get("save_horizon_csv", False)
            auto_tune_horizon = self.model_params.get("auto_tune_horizon", False)
            degradation_threshold = self.model_params.get("degradation_threshold", 3.0)

            # Construct horizon CSV path
            horizon_csv_path = None
            if save_horizon_csv:
                # Generate path in results/diagnostics
                import os
                os.makedirs("results/diagnostics", exist_ok=True)
                # Use class name and time or model name to avoid name conflict
                horizon_csv_path = f"results/diagnostics/{self.__class__.__name__}_horizon_stats.csv"

            fail_on_instability = kwargs.get("fail_on_instability")

            # Extract scaler params for original-scale validation metrics (best practice)
            scaler_params = None
            if hasattr(self, 'preprocessor') and self.preprocessor is not None:
                scaler_params = self.preprocessor.get_fast_inverse_scaling_params()
                if scaler_params is not None:
                    logger.info("[LSTM] Validation metrics will be computed in ORIGINAL scale")

            trained_model_instance, history = run_train_loop(
                model=self.model,
                encoder_inputs_train=X_train,
                decoder_inputs_train=future_cov_train,  # Pass future_cov as decoder_inputs
                true_outputs_train=y_train,
                encoder_inputs_val=X_val,
                decoder_inputs_val=future_cov_val,      # Pass future_cov as decoder_inputs
                true_outputs_val=y_val,
                loss_fn=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=self._get_training_param("epochs"),
                early_stopping_patience=patience,
                min_epochs=self.model_params.get("min_epochs", 5),  # Minimum epochs before early stopping
                device=self.device,
                batch_size=self._get_training_param("batch_size"),
                model_name=self.__class__.__name__,
                # advanced training parameters
                use_amp=use_amp,
                max_grad_norm=max_grad_norm,
                save_horizon_csv=save_horizon_csv,
                horizon_csv_path=horizon_csv_path,
                auto_tune_horizon=auto_tune_horizon,
                degradation_threshold=degradation_threshold,
                optuna_trial=kwargs.get("optuna_trial"),
                trial_step_offset=kwargs.get("trial_step_offset"),
                gradient_monitor=kwargs.get("gradient_monitor"),
                save_scheduler_plot=self.model_params.get("save_scheduler_plot", False),
                save_scheduler_csv=self.model_params.get("save_scheduler_csv", False),
                run_context=self.run_context,
                fail_on_numerical_instability=fail_on_instability,
                num_workers=self._get_training_param("num_workers"),  # DataLoader workers for multiprocessing
                scaler_params=scaler_params  # Fast inverse scaling for original-scale validation
            )
            self.training_history = history
            return trained_model_instance
        except RuntimeError as e:
            # Don't log ERROR for expected divergence
            if "Numerical instability" in str(e):
                raise  # Re-raise silently
            # Log other errors
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
        except ValueError as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise RuntimeError(f"Training failed: {e}")

    def _prepare_input_tensor(self, input_data: pd.DataFrame) -> torch.Tensor:
        """
        Override base class to handle time_features correctly with LSTM.

        Problem:
            preprocessor.transform() adds time_features to DataFrame (e.g., hour_sin, hour_cos),
            but LSTM was initialized with input_size=encoder_input_size (only basic features).
            This causes dimension mismatch during prediction.

        Solution:
            Slice tensor to only encoder features (same as Transformer implementation).
            Uses feature_layout.encoder_feature_idx to select correct columns.

        Returns:
            Tensor of shape (1, window_size, encoder_input_size)
        """
        # Use full preprocessing (includes time_features if configured)
        df_proc = self.preprocessor.transform(input_data, allow_subset=False)
        values = df_proc.values.astype("float32")
        tensor = torch.from_numpy(values).unsqueeze(0).to(self.device)

        # Safety slice to encoder input size (remove time_features if present)
        if tensor.size(-1) != self.feature_layout.encoder_input_size:
            tensor = tensor[..., self.feature_layout.encoder_feature_idx]

        return tensor

    def _prepare_future_exog_tensor(self, future_exog, batch_size=1):
        """
        Prepare future exogenous tensor for LSTM iterative predictions.

        Args:
            future_exog: Future exogenous data (DataFrame or None)
            batch_size: Batch size for zero tensor generation (default: 1)

        Contract:
        - If model has no future_covariates -> return None
        - If future_exog is None -> return zero tensor with correct feature dim and batch_size
        - If future_exog provided -> validate shape and return tensor

        ⚠️ PERFORMANCE WARNING:
        This method is called O(H) times during iterative inference (once per
        prediction step). For long horizons, consider batched refactoring to
        prepare all future exog steps upfront.
        """
        # Check if model has future covariates
        future_cov_size = self.feature_layout.future_covariates_size

        if future_cov_size == 0:
            return None

        # No future_exog provided → ZERO SLOT with correct batch size
        if future_exog is None:
            return torch.zeros(
                (batch_size, self.forecast_steps, future_cov_size),
                dtype=torch.float32,
                device=self.device,
            )

        # Convert DataFrame to Tensor if needed
        if isinstance(future_exog, pd.DataFrame):
            # Extract future covariate columns
            future_cov_cols = self.feature_layout.future_covariates
            if not future_cov_cols:
                return None

            # Validate that all required columns are present
            missing = set(future_cov_cols) - set(future_exog.columns)
            if missing:
                raise ValueError(f"future_exog DataFrame missing required columns: {missing}")

            # CRITICAL: Apply preprocessing to future_exog (same as input_data)
            # Future covariates must be scaled/normalized the same way as during training
            # allow_subset=True because we're only transforming future covariates, not targets
            future_exog_proc = self.preprocessor.transform(future_exog[future_cov_cols], allow_subset=True)
            # .copy() ensures a C-contiguous array; column reordering can
            # produce a non-contiguous view with negative strides that PyTorch rejects.
            future_exog_np = future_exog_proc.values.copy()
            future_exog = torch.FloatTensor(future_exog_np).unsqueeze(0).to(self.device)

        # Validate tensor shape
        if not isinstance(future_exog, torch.Tensor):
            raise TypeError("future_exog must be a torch.Tensor or pandas.DataFrame")

        if future_exog.ndim != 3:
            raise ValueError(
                f"future_exog must have shape (batch, steps, features), got {future_exog.shape}"
            )

        _, steps, feats = future_exog.shape
        if steps != self.forecast_steps:
            raise ValueError(
                f"future_exog steps mismatch: expected {self.forecast_steps}, got {steps}"
            )

        if feats != future_cov_size:
            raise ValueError(
                f"future_exog features mismatch: expected {future_cov_size}, got {feats}"
            )

        return future_exog

    def _internal_predict(
            self,
            input_tensor: torch.Tensor,
            future_exog_tensor: Optional[torch.Tensor] = None,
            **kwargs
    ) -> np.ndarray:
        """
        Internal prediction dispatch. Automatically selects the correct strategy.

        Args:
            input_tensor: Historical window (B, W, encoder_input_size)
                         where encoder_input_size = targets + past_cov + future_cov
            future_exog_tensor: Future covariates for forecast horizon (B, H, future_cov_size)
                               FORWARD-COMPATIBLE: accepted but not yet consumed
                               by encoder-only architecture

        Returns:
            Raw model outputs as NumPy array with shape (B, H, F)
        """
        # Move tensors to device with non_blocking for async transfer
        input_tensor = input_tensor.to(self.device, non_blocking=True)

        # future_exog_tensor is accepted for forward-compatibility
        # but currently NOT consumed by encoder-only LSTM architecture
        if future_exog_tensor is not None:
            kwargs["future_exog_tensor"] = future_exog_tensor.to(
                self.device, non_blocking=True
            )

        if self.strategy == "direct":
            output =  self._predict_direct(input_tensor, **kwargs)
        else:
            # Iterative strategy: dispatch to stateful or stateless version
            if self.iterative_stateful:
                output = self._predict_iterative_stateful(input_tensor, **kwargs)
            else:
                output = self._predict_iterative(input_tensor, **kwargs)

        # Ensure we always return a NumPy array
        return output.detach().float().cpu().numpy()

    def _predict_direct(
            self,
            input_tensor: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        """
        Direct (one-shot) forecasting with optional future context.

        Architecture: encoder-only direct
        ────────────────────────────────────────────────────────────────────
        PAST_COVARIATES handling:
          - Used from historical window (no policy needed for direct)

        FUTURE_COVARIATES handling:
          - mode='none': IGNORED (backward compatible)
          - mode='global': Used for global conditioning via late concatenation
            Requires future_exog_tensor in kwargs

        The model predicts all H forecast steps in a single forward pass.
        """
        self.model.eval()
        with torch.no_grad():
            # Extract future context if enabled
            future_ctx_tensor = None

            if self.future_covariate_mode == "global":
                future_ctx_tensor = kwargs.get("future_exog_tensor", None)

                if future_ctx_tensor is None:
                    raise ValueError(
                        "future_exog_tensor is required when future_covariate_mode='global', "
                        "but got None. Please provide future_exog when calling predict()."
                    )

            # Forward pass (model handles None gracefully for backward compatibility)
            output = self.model(input_tensor, tgt=future_ctx_tensor)
            return output

    def _predict_iterative(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Iterative autoregressive prediction for encoder-only LSTM with covariate handling.

        Architecture: encoder-only iterative
        ────────────────────────────────────────────────────────────────────
        PAST_COVARIATES handling (policy-based):
          - FROZEN (default): Last known window repeated for all steps
          - Prevents distribution shift by using stable historical context

        FUTURE_COVARIATES handling (mode-dependent):
          - mode='none': IGNORED (backward compatible)
          - mode='stepwise': Per-step injection (future_exog[k] → prediction[k])

        ⚠️ IMPORTANT: Stepwise mode uses TEACHER FORCING during training
        and AUTOREGRESSIVE predictions during inference. This may lead to
        error accumulation for long forecast horizons due to training-inference
        mismatch. See documentation for details.

        At each step, the model consumes:
          - Its own previous prediction (autoregressive)
          - Frozen past_covariates (PastCovariatePolicy.FROZEN)
          - Future_covariates[step] (if mode='stepwise')
        """
        B, W, _ = input_tensor.shape
        H = self.forecast_steps
        F = self.num_features

        # Extract feature layout information
        target_size = self.feature_layout.target_size
        past_cov_size = self.feature_layout.past_covariates_size
        future_cov_size = self.feature_layout.future_covariates_size

        # Validate input dimensions
        expected_input_size = target_size + past_cov_size + future_cov_size
        if input_tensor.shape[2] != expected_input_size:
            raise ValueError(
                f"Input tensor size mismatch: expected {expected_input_size} "
                f"(targets={target_size}, past_cov={past_cov_size}, future_cov={future_cov_size}), "
                f"got {input_tensor.shape[2]}"
            )

        # Validate future_exog requirement (for stepwise mode)
        future_exog_tensor = None
        if future_cov_size > 0 and self.future_covariate_mode == "stepwise":
            future_exog_tensor = kwargs.get('future_exog_tensor', None)
            if future_exog_tensor is None:
                raise ValueError(
                    f"future_exog_tensor is required when future_covariate_mode='stepwise' "
                    f"and model has {self.feature_layout.future_covariates_size} future_covariates, "
                    f"but got None. Please provide future_exog when calling predict()."
                )

        output = torch.zeros((B, H, F), device=input_tensor.device, dtype=input_tensor.dtype)

        self.model.eval()
        with torch.no_grad():
            # ═══════════════════════════════════════════════════════════════════
            # COVARIATE EXTRACTION (PastCovariatePolicy.FROZEN)
            # ═══════════════════════════════════════════════════════════════════
            # Extract last known window of past_covariates and freeze it
            # Feature order: [targets, past_covariates, future_covariates]
            if past_cov_size > 0:
                past_cov_start = target_size
                past_cov_end = target_size + past_cov_size
                frozen_past_covariates = input_tensor[:, :, past_cov_start:past_cov_end]
            else:
                frozen_past_covariates = None

            # Extract initial future_covariates (stored but not consumed yet)
            if future_cov_size > 0:
                future_cov_start = target_size + past_cov_size
                # For forward-compatibility: accept future_exog_tensor from kwargs
                # but don't use it in encoder-only iterative (yet)
                pass

            # Initialize context window
            context = input_tensor

            # ═══════════════════════════════════════════════════════════════════
            # AUTOREGRESSIVE LOOP
            # ═══════════════════════════════════════════════════════════════════
            for step in range(H):
                # Predict next step
                one_step = self.model(context)
                output[:, step: step + 1, :] = one_step

                # ───────────────────────────────────────────────────────────────
                # CONTEXT UPDATE with FROZEN past_covariates
                # ───────────────────────────────────────────────────────────────
                # Shift window: drop oldest timestep, append new prediction
                new_context_targets = torch.cat([context[:, 1:, :target_size], one_step], dim=1)

                if past_cov_size > 0:
                    # Apply FROZEN policy: reuse last known past_covariates
                    new_context_past = frozen_past_covariates
                    context = torch.cat([new_context_targets, new_context_past], dim=2)
                else:
                    context = new_context_targets

                if future_cov_size > 0:
                    # Future covariate injection (mode-dependent)
                    if self.future_covariate_mode == "stepwise":
                        # ═══════════════════════════════════════════════════════════
                        # STEPWISE MODE: Per-step injection
                        # ═══════════════════════════════════════════════════════════
                        # Extract current step's future_exog: (B, 1, F_future)
                        future_step = future_exog_tensor[:, step:step+1, :]
                        # Expand to window size: (B, W, F_future)
                        # Model sees same future_exog value across entire window
                        future_cov_context = future_step.expand(-1, W, -1)
                    else:
                        # ═══════════════════════════════════════════════════════════
                        # NONE MODE: Historical window (backward compatible)
                        # ═══════════════════════════════════════════════════════════
                        # Use last W steps from input (historical values)
                        future_cov_context = input_tensor[:, :, future_cov_start:]

                    context = torch.cat([context, future_cov_context], dim=2)

            return output.float().cpu()

    def _predict_iterative_stateful(
        self,
        input_tensor: torch.Tensor,
        future_exog_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Stateful iterative prediction with LSTM state propagation.

        Key difference from stateless (_predict_iterative):
        - Warm-up on history to get initial (h, c) state
        - Each step uses lstm(..., (h, c)) to preserve and propagate state
        - State flows: history → pred[0] → pred[1] → ... → pred[H-1]
        - True autoregressive recurrence with cumulative memory

        This approach significantly improves long-horizon forecasting (H≥48)
        by allowing the LSTM to accumulate information across the forecast.

        Args:
            input_tensor: Input history window (B, W, C) where C = [targets, past_cov, future_cov]
            future_exog_tensor: Future exogenous covariates (B, H, F_future) for stepwise mode

        Returns:
            Predictions tensor (B, H, F) on CPU
        """
        B, W, C = input_tensor.shape
        H = self.forecast_steps
        F = self.num_features

        # Feature layout
        target_size = self.feature_layout.target_size
        past_cov_size = self.feature_layout.past_covariates_size
        future_cov_size = self.feature_layout.future_covariates_size

        # Extract frozen past covariates (last known window)
        if past_cov_size > 0:
            frozen_past_cov = input_tensor[:, :, target_size:target_size+past_cov_size]
        else:
            frozen_past_cov = None

        # Validate future_exog requirement (for stepwise mode)
        if future_cov_size > 0 and self.future_covariate_mode == "stepwise":
            if future_exog_tensor is None:
                raise ValueError(
                    f"future_exog_tensor is required when future_covariate_mode='stepwise' "
                    f"and model has {future_cov_size} future_covariates, but got None. "
                    f"Please provide future_exog when calling predict()."
                )

        output = torch.zeros((B, H, F), device=input_tensor.device, dtype=input_tensor.dtype)

        self.model.eval()
        with torch.no_grad():
            # ═══════════════════════════════════════════════════════════
            # WARM-UP: Process history to initialize LSTM state
            # ═══════════════════════════════════════════════════════════
            # Run LSTM on full history window to get initial hidden state
            _, (h, c) = self.model.base_model.lstm(input_tensor)
            # h, c shape: (num_layers, B, hidden_size)

            # Last timestep target for first prediction
            last_target = input_tensor[:, -1:, :target_size]  # (B, 1, target_size)

            # ═══════════════════════════════════════════════════════════
            # AUTOREGRESSIVE LOOP with STATE PROPAGATION
            # ═══════════════════════════════════════════════════════════
            for step in range(H):
                # ───────────────────────────────────────────────────────
                # BUILD INPUT for current step
                # ───────────────────────────────────────────────────────
                # Input components: [last_prediction, frozen_past, future_exog[step]]
                step_input_parts = [last_target]

                # Add frozen past covariates (most recent timestep)
                if past_cov_size > 0:
                    # Use LAST timestep of frozen window (most recent known context)
                    step_input_parts.append(frozen_past_cov[:, -1:, :])

                # Add current step's future covariates
                if future_cov_size > 0 and self.future_covariate_mode == "stepwise":
                    future_step = future_exog_tensor[:, step:step+1, :]
                    step_input_parts.append(future_step)

                # Concatenate all input components
                step_input = torch.cat(step_input_parts, dim=2)  # (B, 1, input_size)

                # ───────────────────────────────────────────────────────
                # LSTM STEP with state propagation
                # ───────────────────────────────────────────────────────
                # Process single timestep while preserving hidden state
                lstm_out, (h, c) = self.model.base_model.lstm(step_input, (h, c))
                # lstm_out: (B, 1, hidden_size)
                # h, c: (num_layers, B, hidden_size) - UPDATED state for next step

                # ───────────────────────────────────────────────────────
                # OUTPUT HEAD
                # ───────────────────────────────────────────────────────
                # Apply FC layer to LSTM output
                step_output = self.model.base_model.fc(lstm_out[:, -1, :])
                step_output = step_output.reshape(-1, 1, F)  # (B, 1, F)

                # Store prediction
                output[:, step:step+1, :] = step_output

                # ───────────────────────────────────────────────────────
                # UPDATE for next iteration
                # ───────────────────────────────────────────────────────
                # Use current prediction as input for next step
                last_target = step_output  # (B, 1, target_size)

        return output.float().cpu()
