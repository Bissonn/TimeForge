"""
Target initialization strategies for encoder-decoder Transformer models.

This module provides various strategies for initializing decoder input tensors (tgt)
during training (teacher forcing) and inference (direct/iterative prediction).
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn


class TgtInitializer(ABC):
    """
    Abstract base class for target (tgt) initialization strategies in encoder-decoder Transformers.
    """

    def __init__(self, decoder_uses_exog: bool, num_exog_decoder: int):
        self.decoder_uses_exog = decoder_uses_exog
        self.num_exog_decoder = num_exog_decoder

    @abstractmethod
    def _create_base_tgt(
            self,
            src: torch.Tensor,
            steps: int,
            num_features: int,
            device: torch.device,
            dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Hook method to create the base target tensor (without exogenous variables).
        """
        pass

    def get_sos_token(self, src: torch.Tensor, num_features: int, device: torch.device,
                      dtype: torch.dtype) -> torch.Tensor:
        """
        Public method to get just the Start-Of-Sequence token (step=1).
        """
        return self._create_base_tgt(src, 1, num_features, device, dtype)

    def initialize_direct(
            self,
            src: torch.Tensor,
            forecast_steps: int,
            num_features: int,
            device: torch.device,
            future_exog_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Initialize the tgt tensor for direct prediction mode.
        """
        B = src.size(0)
        dtype = src.dtype

        # 0. SOS – the same as in training
        sos_token = self.get_sos_token(
            src=src,
            num_features=num_features,
            device=device,
            dtype=dtype,
        )

        if sos_token.dim() == 2:
            sos_token = sos_token.unsqueeze(1)  # (B, 1, F)

        if forecast_steps > 1:
            base_tail = self._create_base_tgt(
                src,
                forecast_steps - 1,
                num_features,
                device,
                dtype,
            )
            if base_tail.dim() == 2:
                base_tail = base_tail.unsqueeze(1)
            y_base = torch.cat([sos_token, base_tail], dim=1)
        else:
            y_base = sos_token

        # 2. Handle exogenous variables (common logic)
        if self.decoder_uses_exog:
            if future_exog_tensor is None:
                raise ValueError("Model requires 'future_exog_tensor' for decoder in direct prediction.")
            if future_exog_tensor.shape[1] < forecast_steps:
                raise ValueError(
                    f"future_exog_tensor too short: expected >= {forecast_steps}, got {future_exog_tensor.shape[1]}")

            # Ensure time dimension matches forecast_steps exactly (slice if longer)
            future_exog_tensor = future_exog_tensor[:, :forecast_steps, :self.num_exog_decoder]

            if future_exog_tensor.dim() == 2:
                future_exog_tensor = future_exog_tensor.unsqueeze(0)
            if future_exog_tensor.size(0) == 1 and B > 1:
                future_exog_tensor = future_exog_tensor.expand(B, -1, -1)

            future_exog_tensor = future_exog_tensor.to(device=device, dtype=dtype)
            tgt = torch.cat([y_base, future_exog_tensor], dim=-1)
        else:
            tgt = y_base

        return tgt

    def initialize_iterative(
            self,
            src: torch.Tensor,
            num_features: int,
            device: torch.device,
            future_exog_tensor: Optional[torch.Tensor] = None,
            step: int = 0
    ) -> torch.Tensor:
        """
        Initialize the tgt tensor for iterative prediction mode at a given step.
        """
        y_base_step = self._create_base_tgt(src, 1, num_features, device, src.dtype)

        if self.decoder_uses_exog:
            if future_exog_tensor is None:
                raise ValueError("Model requires 'future_exog_tensor' for decoder in iterative prediction.")
            if future_exog_tensor.shape[1] <= step:
                raise ValueError(f"future_exog_tensor too short for step {step}.")

            exog_step = future_exog_tensor[:, step:step + 1, :self.num_exog_decoder]
            return torch.cat([y_base_step, exog_step], dim=2)

        return y_base_step


class ZerosTgtInitializer(TgtInitializer):
    def _create_base_tgt(self, src, steps, num_features, device, dtype):
        B = src.size(0)
        return torch.zeros(B, steps, num_features, device=device, dtype=dtype)


class LastValueTgtInitializer(TgtInitializer):
    def _create_base_tgt(self, src, steps, num_features, device, dtype):
        last_val = src[:, -1:, :num_features]
        last_val = last_val.to(device=device, dtype=dtype)
        return last_val.expand(-1, steps, -1).clone()


class MeanTgtInitializer(TgtInitializer):
    def _create_base_tgt(self, src, steps, num_features, device, dtype):
        mean_val = src[:, :, :num_features].mean(dim=1, keepdim=True)
        mean_val = mean_val.to(device=device, dtype=dtype)
        return mean_val.expand(-1, steps, -1).clone()


class MedianTgtInitializer(TgtInitializer):
    def _create_base_tgt(self, src, steps, num_features, device, dtype):
        median_val = src[:, :, :num_features].median(dim=1, keepdim=True).values
        median_val = median_val.to(device=device, dtype=dtype)
        return median_val.expand(-1, steps, -1).clone()


class TrendTgtInitializer(TgtInitializer):
    @staticmethod
    def _linalg_compute_dtype(dtype: torch.dtype) -> torch.dtype:
        """
        Torch linalg ops (inverse/solve/pinv) often do not support low-precision dtypes
        on CUDA (e.g., float16/bfloat16). We compute in float32 and cast the final tensor
        back to requested dtype.
        """
        if dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return dtype

    def _create_base_tgt(self, src, steps, num_features, device, dtype):
        B, L, _ = src.shape
        if L < 2:
            last_val = src[:, -1:, :num_features].to(device=device, dtype=dtype)
            return last_val.expand(-1, steps, -1).clone()

        # IMPORTANT:
        # Under AMP/autocast inference, src.dtype can be float16/bfloat16.
        # We must avoid low-precision linalg on CUDA and avoid mixed dtypes in einsum.
        # Compute the entire regression path in float32 (or higher), then cast final output to dtype.
        compute_dtype = self._linalg_compute_dtype(dtype)

        # Locally disable autocast for this numeric block.
        # This ensures torch.inverse / einsum runs in compute_dtype deterministically.
        autocast_off = (
            torch.autocast(device_type="cuda", enabled=False)
            if (torch.device(device).type == "cuda" and torch.cuda.is_available())
            else nullcontext()
        )

        with autocast_off:
            # Build design matrix directly in compute_dtype
            time_indices = torch.arange(L, device=device, dtype=compute_dtype)
            ones = torch.ones(L, device=device, dtype=compute_dtype)
            X = torch.stack([ones, time_indices], dim=1)  # (L, 2)

            # (X^T X)^{-1} X^T  in compute_dtype
            XtX = X.T @ X
            XtX_inv = torch.inverse(XtX)
            X_pinv = XtX_inv @ X.T  # (2, L)

            # Targets in compute_dtype (avoid mixed dtype in einsum)
            Y = src[..., :num_features].to(device=device, dtype=compute_dtype)  # (B, L, F)
            beta = torch.einsum("pl,blf->bpf", X_pinv, Y)  # (B, 2, F)

            # Future design matrix in compute_dtype
            future_time_indices = torch.arange(L, L + steps, device=device, dtype=compute_dtype)
            future_ones = torch.ones(steps, device=device, dtype=compute_dtype)
            X_future = torch.stack([future_ones, future_time_indices], dim=1)  # (steps, 2)
            tgt_trend = torch.einsum("st,btf->bsf", X_future, beta)  # (B, steps, F)

        # Final cast back to requested dtype for compatibility with the rest of the model.
        return tgt_trend.to(device=device, dtype=dtype)

class SeasonalTgtInitializer(TgtInitializer):
    """
    Initializes the target by copying values from 'season_length' steps ago.
    Ideal for data with clear seasonality (e.g., daily pattern, yearly pattern).
    """
    def __init__(self, decoder_uses_exog: bool, num_exog_decoder: int, season_length: int):
        super().__init__(decoder_uses_exog, num_exog_decoder)
        self.season_length = season_length
        if self.season_length < 1:
            # Fallback or error - here we treat 1 as last value
            self.season_length = 1

    def _create_base_tgt(self, src, steps, num_features, device, dtype):
        B, W, F = src.shape

        # Safety check: if history window is shorter than season,
        # fallback to last value (simple persistence)
        if W < self.season_length:
            last_val = src[:, -1:, :num_features]
            return last_val.expand(-1, steps, -1).clone()

        # Extract the last full cycle from history
        # Shape: (B, season_length, F)
        last_cycle = src[:, -self.season_length:, :num_features]

        # Tile the cycle to cover the forecast horizon 'steps'
        # e.g., if steps=20 and season=12, we need 2 repeats to get 24 steps
        num_repeats = (steps // self.season_length) + 1
        tiled = last_cycle.repeat(1, num_repeats, 1)

        # Slice to exact horizon length
        tgt = tiled[:, :steps, :].to(device=device, dtype=dtype)

        return tgt


class CopyHistoryTgtInitializer(TgtInitializer):
    """
    Initializes the target by copying the immediate history (last 'H' steps).
    Useful for signals with local continuity or momentum, acting as a "soft start"
    that continues the recent dynamics.
    """
    def _create_base_tgt(self, src, steps, num_features, device, dtype):
        B, W, F = src.shape

        # If we need more steps than we have history, repeat history
        if W < steps:
             num_repeats = (steps // W) + 1
             tiled = src[:, :, :num_features].repeat(1, num_repeats, 1)
             # Take the LAST 'steps' from the repeated sequence
             return tiled[:, -steps:, :].to(device=device, dtype=dtype)

        # Otherwise, just take the last 'steps' from history
        # This assumes the future will look like the immediate past
        tgt = src[:, -steps:, :num_features].clone()
        return tgt.to(device=device, dtype=dtype)


def build_tgt_train(
    target_true: torch.Tensor,
    src: torch.Tensor,
    initializer: TgtInitializer,
    decoder_exog: Optional[torch.Tensor],
    noise_config: Optional[dict] = None,
) -> torch.Tensor:
    """
    Constructs the 'tgt' tensor for Transformer decoder training (teacher forcing).

    Optionally injects Gaussian noise into the target sequence to reduce
    training-inference mismatch in autoregressive models. Noise simulates
    prediction errors that occur during inference, making the model more
    robust to imperfect history.

    Args:
        target_true: Ground truth targets (B, H, F)
        src: Source encoder input for SOS token initialization (B, W, C)
        initializer: TgtInitializer for creating SOS token
        decoder_exog: Optional decoder exogenous variables (B, >=H, E_dec)
        noise_config: Optional dict with keys:
            - 'enabled' (bool): Whether to inject noise
            - 'std' (float): Noise standard deviation (relative to target scale)
            - 'schedule' (str): 'constant' or 'curriculum' (increasing over training)
            - 'training_progress' (float): Current progress [0, 1] for curriculum
            - 'apply_to_exog' (bool): Whether to also noise exogenous features

    Returns:
        Decoder input tensor (B, H, decoder_input_size) where:
            decoder_input_size = F (targets only) or F + E_dec (with exog)

    Noise Injection Details:
        - Only applied to TARGET features (y_shifted), not SOS token
        - Noise is additive Gaussian: y_noisy = y + N(0, std²)
        - By default, exogenous features are NOT noised (assumed known)
        - Curriculum schedule: std increases linearly with training_progress

    Example:
        >>> noise_cfg = {'enabled': True, 'std': 0.05, 'schedule': 'constant'}
        >>> tgt = build_tgt_train(y_true, src, initializer, exog, noise_cfg)

    References:
        - Bengio et al. (2015). "Scheduled Sampling for Sequence Prediction"
        - Similar to input dropout but continuous and Gaussian
    """
    B, H, F = target_true.shape
    device = target_true.device
    dtype = target_true.dtype

    sos_token = initializer.get_sos_token(src, F, device, dtype)

    if initializer.decoder_uses_exog and decoder_exog is None:
        raise ValueError(
             "Model is configured to use decoder exogenous features, but 'decoder_exog' is None during training."
        )

    if H > 1:
        y_shifted = torch.cat([sos_token, target_true[:, :-1, :]], dim=1)
    else:
        y_shifted = sos_token

    # ──────────────────────────────────────────────────────────────────
    # PREDICTION NOISE INJECTION
    # ──────────────────────────────────────────────────────────────────
    if noise_config is not None and noise_config.get('enabled', False):
        noise_std = noise_config.get('std', 0.05)
        schedule = noise_config.get('schedule', 'constant')
        apply_to_exog = noise_config.get('apply_to_exog', False)

        # Curriculum: increase noise over training
        if schedule == 'curriculum':
            training_progress = noise_config.get('training_progress', 0.0)
            # Linear ramp: 0 -> noise_std
            noise_std = noise_std * training_progress

        # Only noise the targets (not SOS token)
        if H > 1:
            # Generate noise: (B, H-1, F) since y_shifted[:, 0, :] is SOS
            target_noise = torch.randn(
                B, H - 1, F,
                device=device,
                dtype=dtype
            ) * noise_std

            # Apply noise to shifted targets (skip SOS token at position 0)
            y_shifted[:, 1:, :] = y_shifted[:, 1:, :] + target_noise

    # ──────────────────────────────────────────────────────────────────
    # CONCATENATE DECODER EXOGENOUS
    # ──────────────────────────────────────────────────────────────────
    if decoder_exog is not None:
        if decoder_exog.shape[1] < H:
            raise ValueError(
                f"Decoder exogenous features are too short. Expected at least {H} time steps."
            )
        exog_sliced = decoder_exog[:, :H, :]

        # Optional: noise exogenous features (rarely used, disabled by default)
        if (noise_config is not None and
            noise_config.get('enabled', False) and
            noise_config.get('apply_to_exog', False)):

            noise_std = noise_config.get('std', 0.05)
            if noise_config.get('schedule') == 'curriculum':
                noise_std = noise_std * noise_config.get('training_progress', 0.0)

            exog_noise = torch.randn_like(exog_sliced) * noise_std
            exog_sliced = exog_sliced + exog_noise

        tgt = torch.cat([y_shifted, exog_sliced], dim=-1)
        return tgt

    return y_shifted


def create_tgt_initializer(
    tgt_init: str,
    decoder_uses_exog: bool,
    num_exog_decoder: int,
    seasonal_period: Optional[int] = None,
) -> TgtInitializer:
    """
    Factory function for creating TgtInitializer instances.

    Encapsulates the logic of selecting and instantiating the appropriate
    TgtInitializer subclass based on the strategy name.

    Args:
        tgt_init: Strategy name ("zeros", "last_value", "mean", "median",
                  "trend", "seasonal", "copy_history")
        decoder_uses_exog: Whether decoder uses exogenous variables
        num_exog_decoder: Number of decoder exogenous features
        seasonal_period: Period for seasonal initializer (required if tgt_init="seasonal")

    Returns:
        Configured TgtInitializer instance

    Raises:
        ValueError: If tgt_init is not a valid strategy name
        ValueError: If tgt_init="seasonal" but seasonal_period is not provided
    """
    initializers = {
        "zeros": ZerosTgtInitializer,
        "last_value": LastValueTgtInitializer,
        "mean": MeanTgtInitializer,
        "median": MedianTgtInitializer,
        "trend": TrendTgtInitializer,
        "seasonal": SeasonalTgtInitializer,
        "copy_history": CopyHistoryTgtInitializer,
    }

    if tgt_init not in initializers:
        valid_options = ", ".join(initializers.keys())
        raise ValueError(f"Invalid tgt_init: '{tgt_init}'. Valid options: {valid_options}")

    # Seasonal initializer requires seasonal_period parameter
    if tgt_init == "seasonal":
        if seasonal_period is None:
            raise ValueError("tgt_init='seasonal' requires seasonal_period parameter")
        return SeasonalTgtInitializer(
            decoder_uses_exog=decoder_uses_exog,
            num_exog_decoder=num_exog_decoder,
            season_length=seasonal_period
        )

    # All other initializers use the same constructor signature
    return initializers[tgt_init](
        decoder_uses_exog=decoder_uses_exog,
        num_exog_decoder=num_exog_decoder
    )
