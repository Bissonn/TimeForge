r"""Module for Transformer-based time series forecasting models.

This module defines the TransformerForecaster class, which implements both direct and
iterative forecasting strategies using a Transformer neural network.
It supports both encoder-only and encoder-decoder architectures with various
configuration options including SOTA techniques like RevIN, GELU activations,
Pre-LN architecture.

COVARIATE HANDLING (v2 API):
────────────────────────────────────────────────────────────────────
  past_covariates:   Features known only in history (encoder-only)
                     - Encoder-decoder iterative: NOT used (encoder uses full history)
                     - Encoder-only iterative: Currently pure AR (no covariates)
                     - Ready for PastCovariatePolicy implementation

  future_covariates: Features known in both history and future
                     - Encoder-decoder iterative: Fully supported (decoder path)
                     - Encoder-only: Currently stored but NOT consumed
                     - Forward-compatible for future use

MEMORY OPTIMIZATION (Iterative Decoder):
────────────────────────────────────────────────────────────────────
  Buffer mode (preallocated): Eliminates O(H) allocations and O(H²×B×F) memory copies
                              by preallocating full decoder sequence buffer upfront.
                              Auto-enabled for H > 512. See: docs/analysis_oh2_concat_problem.md

  Note: This optimization removes memory-management overhead but does NOT reduce
        asymptotic attention cost (O(H³) remains unchanged). This is fundamental
        to autoregressive decoding and cannot be eliminated without architectural
        changes (e.g., sparse attention, sliding window decoder).

NON-GOALS (Intentional Design Boundaries):
────────────────────────────────────────────────────────────────────
  ❌ Making encoder-only iterative support covariates
     Reason: Zero-padding would create silent train/inference mismatch.
             Use encoder-decoder iterative or direct strategy instead.

  ❌ Eliminating O(H³) decoder attention cost
     Reason: This is fundamental to autoregressive decoding with full context.
             Sparse/linear attention is a separate research direction.

  ❌ Unifying encoder-only and encoder-decoder training semantics
     Reason: These are fundamentally different architectures with different
             use cases and trade-offs. Forced unification would compromise both.

COMPLEXITY NOTICE:
────────────────────────────────────────────────────────────────────
This module has high cognitive complexity (1600+ LOC) due to supporting:
  - 2 architectures (encoder-only, encoder-decoder)
  - 2 strategies (direct, iterative)
  - 2 decoder modes (concat, buffer)
  - 2 precision modes (AMP, FP32)
  - Multiple research features (RevIN, aux loss, prediction noise, attention capture)

This complexity is JUSTIFIED and INTENTIONAL. Do not refactor "for cleanliness"
without clear functional benefit - this code is in stabilization phase.
"""
import logging
import json
from typing import Dict, List, Tuple, Any, Optional, Literal

import torch

from core.context import RunContext


# Custom exception for HPO parameter constraint violations
class ParameterConstraintViolation(ValueError):
    """Raised when parameter combination violates constraints during HPO."""
    pass
import torch.nn as nn
import torch.backends.cuda
from utils.amp_utils import get_autocast_context
from utils.scheduler import create_scheduler
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except ImportError:
    # Fallback for older PyTorch versions (<2.0)
    sdpa_kernel = None
    SDPBackend = None

import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime, timezone

import contextlib
from contextlib import nullcontext
from typing import Generator

from models.base import NeuralTSForecaster, PastCovariatePolicy
from models.model_registry import register_model
from utils.train_loop import run_train_loop
from utils.dataset import TimeSeriesDataset
from utils.losses import AuxiliaryMultiStepLoss

# Import Transformer components
from models.transformer_components import (
    AttentionCaptureBuffer,
    CapturingMHA,
    RevIN,
    TgtInitializer,
    build_tgt_train,
    create_tgt_initializer,
    LocalAttention,
    GlobalSelfAttention,
    CustomTransformerEncoderLayer,
    create_positional_encoding,
    create_output_head,
    SearchSpaceAnalyzer,
    SearchSpaceFilter,
    ParameterValidator,
    SmartPriorGenerator,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT PARAMETERS FOR TRANSFORMERFORECASTER
# ═══════════════════════════════════════════════════════════════════════════
# Single source of truth for all default parameter values.
# DO NOT MODIFY THIS DICT AT RUNTIME.
TRANSFORMER_DEFAULTS = {
    # ═══ Architecture & Strategy ═══
    "architecture": "encoder-only",
    "strategy": "direct",

    # ═══ Universal Model Dimensions ═══
    "hidden_size": 128,
    "num_encoder_layers": 4,
    "num_heads": 4,
    "dim_ff_multiplier": 4.0,
    "dropout": 0.1,

    # ═══ Attention ═══
    "attention_type": "full",

    # ═══ Optimizer ═══
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "max_grad_norm": 1.0,

    # ═══ RevIN ═══
    "use_revin": False,
    "revin_affine": True,
    "revin_eps": 1e-5,
    "revin_robust": False,

    # ═══ AMP Inference ═══
    "use_amp_inference": True,
    "amp_inference_dtype": None,  # Auto-detect (BF16 on Ampere)
    "debug_amp_inference": False,

    # ═══ Output Head ═══
    "output_head_strategy": "shared",
    "head_type": "linear",

    # ═══ Performance & Debugging ═══
    "nan_guard_enabled": True,  # Enable NaN checks in predictions (small overhead, safe default)
    "device_safety_checks": False,  # Enable device transfer checks in forward() (compatibility mode)
}

class TransformerModel(nn.Module):
    """
    Transformer neural network for time series.

    REGULARIZATION TOPOLOGY:
    ────────────────────────────────────────────────────────────────────
    ⚠️  Encoder-only and encoder-decoder use DIFFERENT dropout injection points:

        - Encoder-only:  encoder -> readout -> output_head (with head_dropout)
        - Encoder-decoder: encoder -> decoder -> dropout_layer -> output_head (with head_dropout)

    This is an intentional design difference, NOT a bug. Different strategies
    have different regularization requirements. DO NOT unify without careful
    benchmarking across both architectures.
    """

    def __init__(
            self,
            encoder_input_size: int,
            decoder_input_size: int,
            num_features: int,
            forecast_steps: int,
            window_size: int,
            output_head_strategy: Literal["shared", "multiple"] = "shared",
            head_type: Literal["linear", "mlp"] = "linear",
            head_dropout: float = 0.0,
            hidden_size: int = 128,
            num_heads: int = 4,
            num_encoder_layers: int = 4,
            num_decoder_layers: int = 4,
            dim_ff_multiplier: float = 4.0,
            dropout: float = 0.1,
            architecture: Literal["encoder-only", "encoder-decoder"] = "encoder-only",
            positional_encoding_config: Optional[Dict[str, Any]] = None,
            readout: Literal["last", "mean", "max", "cls"] = "last",
            attention_type: Literal["full", "local"] = "full",
            attention_window_size: int = 32,
            # RevIN parameters
            use_revin: bool = True,
            revin_affine: bool = True,
            revin_eps: float = 1e-5,
            revin_robust: bool = False,
            # Architecture tweaking
            activation: str = "gelu",
            norm_first: bool = True,
            # Performance flags
            device_safety_checks: bool = False
    ):
        super().__init__()

        # Validate params based on architecture
        if architecture == "encoder-only":
            if readout not in ["last", "mean", "max", "cls"]:
                raise ValueError(f"Invalid readout: {readout}")

        elif architecture == "encoder-decoder":
            if num_decoder_layers < 1:
                raise ValueError(f"Invalid num_decoder_layers: {num_decoder_layers}")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        if output_head_strategy not in {"shared", "multiple"}:
            raise ValueError("output_head_strategy must be 'shared' or 'multiple'")

        if output_head_strategy == "multiple":
            logger.warning("Using 'multiple' heads. Consider 'shared' for better quality/speed.")

        if architecture not in ['encoder-only', 'encoder-decoder']:
            raise ValueError("architecture must be either 'encoder-only' or 'encoder-decoder'.")

        self.architecture = architecture
        self.window_size = window_size
        self.forecast_steps = forecast_steps
        self.num_features = num_features
        self.readout = readout
        pe_config = positional_encoding_config or {}
        self.positional_encoding = pe_config.get("type", "sinusoidal")
        self.attention_type = attention_type
        self.last_attention_weights = None
        self.output_head_strategy = output_head_strategy
        dim_feedforward = int(round(dim_ff_multiplier * hidden_size))
        # capture buffer (no monkey patching)
        self.attn_capture = AttentionCaptureBuffer()
        # Performance flags
        self.device_safety_checks = device_safety_checks

        # --- Initialize RevIN ---
        self.use_revin = use_revin
        if self.use_revin:
            # Normalize targets only
            self.revin = RevIN(num_features, eps=revin_eps, affine=revin_affine, robust=revin_robust)
        else:
            self.revin = None

        # --- Shared Layers ---
        self.input_projection = nn.Linear(encoder_input_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(hidden_size)

        # --- Encoder Positional Encoding ---
        encoder_max_len = window_size
        if self.architecture == "encoder-only" and self.readout == "cls":
            encoder_max_len += 1

        self.pos_encoder = create_positional_encoding(
            hidden_size=hidden_size,
            max_len=encoder_max_len,
            **pe_config
        )

        # --- Encoder ---
        # UNIFIED ARCHITECTURE: Use CustomTransformerEncoderLayer for BOTH
        # 'local' and 'full' attention types.
        #
        # Solution: Use specific wrappers (LocalAttention / GlobalSelfAttention)
        # that expose a unified interface (x, mask, key_padding_mask) required
        # by CustomTransformerEncoderLayer.

        if attention_type == 'local':
            attention_module = LocalAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                window_size=attention_window_size,
                dropout=dropout
            )
        else:  # 'full'
            # Use GlobalSelfAttention wrapper to adapt nn.MultiheadAttention signature
            # to (x, mask, ...) format required by our custom layer.
            attention_module = GlobalSelfAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            )

        # Create custom encoder layer (works with any attention_module)
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=hidden_size,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attention_module=attention_module,
            activation=activation,
            norm_first=norm_first
        )

        # Add final LayerNorm for Pre-LN architecture stability
        encoder_norm = nn.LayerNorm(hidden_size) if norm_first else None

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm,
            enable_nested_tensor=False
        )

        # Assign deterministic per-layer capture keys (works because layers exist as a list)
        for i, lyr in enumerate(self.transformer_encoder.layers):
            if isinstance(lyr, CustomTransformerEncoderLayer):
                attn_mod = lyr.self_attn
                # GlobalSelfAttention or LocalAttention
                if hasattr(attn_mod, "capture"):
                    attn_mod.capture = self.attn_capture
                if hasattr(attn_mod, "key_prefix"):
                    attn_mod.key_prefix = f"enc_self_layer_{i}"

        # --- Head & Decoder Logic ---
        head_input_dim = hidden_size

        # --- Decoder (Conditional) ---
        if self.architecture == 'encoder-decoder':
            self.pos_decoder = create_positional_encoding(
                hidden_size=hidden_size,
                max_len=forecast_steps,
                **pe_config
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation=activation,
                norm_first=norm_first
            )

            # Add Final LayerNorm for Pre-LN architecture stability
            decoder_norm = nn.LayerNorm(hidden_size) if norm_first else None
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_decoder_layers,
                norm=decoder_norm
            )

            # Wrap decoder MHAs (self + cross) with CapturingMHA, per-layer keys.
            # This does NOT change the decoder logic; it only overrides need_weights
            # when capture is enabled.
            for i, lyr in enumerate(self.transformer_decoder.layers):
                if hasattr(lyr, "self_attn") and isinstance(lyr.self_attn, nn.MultiheadAttention):
                    lyr.self_attn = CapturingMHA(lyr.self_attn, self.attn_capture, f"dec_self_layer_{i}")
                if hasattr(lyr, "multihead_attn") and isinstance(lyr.multihead_attn, nn.MultiheadAttention):
                    lyr.multihead_attn = CapturingMHA(lyr.multihead_attn, self.attn_capture,f"dec_cross_layer_{i}")

            # Pre-generate and cache causal mask for decoder (10-15% speedup)
            # Most predictions use forecast_steps length, so we cache that size
            # Falls back to dynamic generation for longer sequences (rare)
            self.register_buffer(
                "causal_mask_buffer",
                torch.nn.Transformer.generate_square_subsequent_mask(forecast_steps),
                persistent=False  # Don't save in state_dict (can be regenerated)
            )

            # Decoder's projection now uses its own specific input size
            self.tgt_projection = nn.Linear(decoder_input_size, hidden_size)
            head_input_dim = hidden_size
        else:  # encoder-only
            self.transformer_decoder = None
            self.tgt_projection = None
            if self.readout == "cls":
                self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
                nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
            else:
                self.register_parameter("cls_token", None)


            head_input_dim = hidden_size

        # --- Create Output Head via Strategy ---
        self.output_head = create_output_head(
            self.architecture,
            output_head_strategy,
            head_input_dim,
            forecast_steps,
            num_features,
            head_type,
            head_dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None: nn.init.zeros_(self.input_projection.bias)

        for m in self.output_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

        if self.tgt_projection is not None:
            nn.init.xavier_uniform_(self.tgt_projection.weight)
            if self.tgt_projection.bias is not None:
                nn.init.zeros_(self.tgt_projection.bias)
        if isinstance(self.transformer_encoder.layers[0], CustomTransformerEncoderLayer):
            enc0 = self.transformer_encoder.layers[0]
            nn.init.xavier_uniform_(enc0.linear1.weight)
            nn.init.zeros_(enc0.linear1.bias)
            nn.init.xavier_uniform_(enc0.linear2.weight)
            nn.init.zeros_(enc0.linear2.bias)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        # --- RevIN Normalization (Encoder Input) ---
        src = self.normalize_input(src, mode='norm')

        src = self.input_projection(src)

        if self.architecture == "encoder-decoder":
            src = self.pos_encoder(src)
            src = self.input_norm(src)
            memory = self.transformer_encoder(src)
            return memory
        else:
            if self.readout == "cls":
                cls_tokens = self.cls_token.expand(src.shape[0], -1, -1)
                src = torch.cat([cls_tokens, src], dim=1)

            src = self.pos_encoder(src)
            src = self.input_norm(src)
            output = self.transformer_encoder(src)
            return output

    def decode(
        self, tgt: torch.Tensor, memory: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.architecture != "encoder-decoder":
            raise ValueError("decode() can only be used when architecture='encoder-decoder'.")

        # --- RevIN Normalization (Decoder Input) ---
        # Apply stats from encoder to decoder targets
        tgt = self.normalize_input(tgt, mode='apply')

        tgt_proj = self.tgt_projection(tgt)
        tgt_proj = self.pos_decoder(tgt_proj)

        # Generate causal mask if not provided (use cached buffer when possible)
        if tgt_mask is None:
            sz = tgt_proj.size(1)
            # Use cached buffer for common case (sequence <= forecast_steps)
            if sz <= self.causal_mask_buffer.size(0):
                # Slice cached buffer (zero-cost view, no allocation)
                tgt_mask = self.causal_mask_buffer[:sz, :sz]
            else:
                # Fallback for longer sequences (rare in practice)
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(sz).to(tgt.device)

        # PyTorch 2.6 MultiHeadAttention requires BOTH mask and is_causal hint:
        # - tgt_mask: Explicit causal mask (required by MultiHeadAttention)
        # - tgt_is_causal=True: Performance hint enabling Flash Attention optimizations
        #
        # Note: Unlike scaled_dot_product_attention (which can use is_causal alone),
        # MultiHeadAttention throws "Need attn_mask if specifying the is_causal hint"
        # when mask=None and is_causal=True. This is a known PyTorch 2.6 behavior.
        #
        # Verified: Flash Attention DOES work with both parameters simultaneously.
        decoder_out = self.transformer_decoder(
            tgt_proj,
            memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=True
        )
        decoder_output = self.dropout_layer(decoder_out)

        return decoder_output

    def normalize_input(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Apply RevIN normalization to input tensor.

        Splits input into target features and exogenous features, applies RevIN
        normalization only to targets, then concatenates back.

        IMPORTANT:
        RevIN statistics (mean, std) are computed ONLY from encoder input (mode='norm').
        Decoder must NEVER recompute statistics - it always uses encoder's stats (mode='apply').
        This ensures consistent normalization across the entire forward pass.

        Args:
            x: Input tensor (B, T, F) where F includes both targets and exog
            mode: RevIN mode - 'norm' for encoder (compute stats), 'apply' for decoder (use encoder stats)

        Returns:
            Normalized tensor with same shape as input
        """
        if self.use_revin:
            # SAFETY CHECK: Input must have at least num_features targets
            if x.shape[-1] < self.num_features:
                raise ValueError(
                    f"Input tensor has {x.shape[-1]} features, but model expects "
                    f"at least {self.num_features} target features."
                )

            # FAST PATH: No exogenous features (most common case)
            # Avoids costly slice & cat operations (~1GB savings over long training)
            if x.shape[-1] == self.num_features:
                return self.revin(x, mode=mode)

            # SLOW PATH: Has exogenous features (targets + exog)
            x_target = x[:, :, :self.num_features]
            x_exog = x[:, :, self.num_features:]
            # Apply RevIN to targets only
            x_target = self.revin(x_target, mode=mode)
            return torch.cat([x_target, x_exog], dim=-1)
        return x

    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN denormalization to output tensor.

        Args:
            y: Output tensor (B, T, F) - targets only

        Returns:
            Denormalized tensor
        """
        if self.use_revin:
            return self.revin(y, mode='denorm')
        return y

    def _encoder_readout_only(self, encoder_output: torch.Tensor) -> torch.Tensor:
        if self.readout == "mean":
            feat = encoder_output.mean(dim=1)
        elif self.readout == "max":
            feat = encoder_output.max(dim=1).values
        elif self.readout == "cls":
            feat = encoder_output[:, 0, :]
        else:
            feat = encoder_output[:, -1, :]

        feat = self.dropout_layer(feat)
        return self.output_head(feat)

    def encode_and_readout(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encoder-only path:
        - encode
        - apply readout strategy
        - apply dropout
        - output head
        - denormalize

        Returns:
            Tensor of shape (B, F)
        """
        assert self.architecture == "encoder-only"
        encoder_output = self.encode(src)
        out = self._encoder_readout_only(encoder_output=encoder_output)
        return self.denormalize_output(out)

    def forward(
        self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # DEBUG AMP
        import os
        if os.getenv('DEBUG_AMP'):
            logger.debug(f"[FWD] src dtype: {src.dtype}, is_autocast: {torch.is_autocast_enabled()}")

        if self.architecture == 'encoder-decoder':
            # --- DEVICE SAFETY (raw forward compatibility) ---
            # Performance: Disabled by default (assumes tensors are on correct device)
            # Enable via config `device_safety_checks: true` for compatibility testing
            if self.device_safety_checks:
                model_device = self.input_projection.weight.device
                if src.device != model_device:
                    src = src.to(model_device)
                if tgt is not None and tgt.device != model_device:
                    tgt = tgt.to(model_device)
                if tgt_mask is not None and tgt_mask.device != model_device:
                    tgt_mask = tgt_mask.to(model_device)

            if tgt is None:
                raise ValueError("tgt (decoder input) must be provided for encoder-decoder architecture.")
            memory = self.encode(src)
            decoder_out = self.decode(tgt, memory, tgt_mask=tgt_mask)
            # Polymorphic call
            output = self.output_head(decoder_out)

            if os.getenv('DEBUG_AMP'):
                logger.debug(
                    f"[FWD] output dtype: {output.dtype}, "
                    f"values: {output[0,:3,0] if output.numel() > 0 else 'empty'}"
                )

            return self.denormalize_output(output)
        else:
            # Encoder-only
            return self.encode_and_readout(src)


@register_model("transformer", is_univariate=False)
class TransformerForecaster(NeuralTSForecaster):
    """Implementation of the Transformer model supporting multiple architectures and strategies."""

    def _get_inference_sdpa_ctx(self):
        """
        Selects the best attention backend based on model dimensions and attention type.
        Does NOT block small models (e.g., hidden_size=64); it simply disables
        Flash Attention for them to avoid overhead or kernel errors.

        IMPORTANT:
        Flash Attention is a PERFORMANCE OPTIMIZATION, not a correctness requirement.
        The MATH backend produces identical results - Flash Attention only provides
        memory and speed improvements for suitable model configurations.
        """
        dev = torch.device(self.device) if self.device is not None else torch.device("cpu")
        if dev.type != "cuda" or not torch.cuda.is_available():
            return nullcontext()

        if sdpa_kernel is None or SDPBackend is None:
            return nullcontext()

        # CHECK 0: Local attention requires custom masks, which only MATH backend supports
        attention_type = self.model_params.get("attention_type", "full")
        if attention_type == "local":
            # Local attention uses custom local causal masks that Flash/Memory-efficient don't support
            return sdpa_kernel(SDPBackend.MATH)

        # Retrieve parameters using the correct keys
        hidden_size = int(self.model_params.get("hidden_size", 128))
        num_heads = int(self.model_params.get("num_heads", 4))

        # Avoid division by zero
        if num_heads == 0:
            head_dim = 0
        else:
            head_dim = hidden_size / num_heads

        # CHECK 1: Is the dimension aligned for GPU Tensor Cores?
        # Flash Attention typically requires head_dim to be a multiple of 8.
        is_aligned = (head_dim % 8 == 0) and (head_dim.is_integer())

        # CHECK 2: Is the model large enough to benefit from Flash Attention?
        # For small models (e.g., hidden_size=64), the kernel launch overhead
        # of Flash Attention often makes it slower than standard Math attention.
        is_large_enough = hidden_size >= 128

        # Logic:
        if is_aligned and is_large_enough:
            # Enable Flash Attention for large, aligned models.
            # We disable 'math' here to ensure we are actually using the optimized kernel.
            # (Note: PyTorch might still fallback internally if hardware doesn't support it)
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)

        else:
            # Fallback for small models (hidden_size < 128) or unaligned dimensions.
            # Force MATH backend. This is crucial for HPO starting at hidden_size=64,
            # ensuring we don't crash or get warnings about unused kernels.
            return sdpa_kernel(SDPBackend.MATH)

    @contextlib.contextmanager
    def capture_attention_weights(self) -> Generator[None, None, None]:
        # If the underlying model does not support attention capture, do nothing.
        # This keeps predict() usable with lightweight dummies/mocks used in tests and debugging.
        if (self.model is None) or (not hasattr(self.model, "attn_capture")):
            yield
            return

        try:
            # Enable capture (steps configured per-mode elsewhere)
            self.model.attn_capture.configure(enabled=True, steps_to_capture=None)
            yield
        finally:
            self.model.attn_capture.configure(enabled=False, steps_to_capture=None)

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
        # Apply all default parameter values (static + conditional)
        self._apply_smart_defaults(model_params)

        pe_config = model_params.get("positional_encoding_config")

        super().__init__(
            model_params=model_params,
            num_features=num_features,
            forecast_steps=forecast_steps,
            window_size=window_size,
            dataset=dataset,
            run_context=run_context,
            **kwargs)
        self._validate_model_params()
        self.last_attention_weights = None

        # ============================================================
        # COVARIATE HANDLING POLICY (forward-compatible)
        # ============================================================
        # Policy for handling past_covariates during iterative prediction
        # Currently: encoder-only iterative uses pure AR (no covariates)
        # Future: will apply policy when covariate support is added
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
        # RUNTIME INPUT VIEW (single source of truth for inference)
        # ============================================================

        self.strategy = self.model_params["strategy"]
        self.architecture = self.model_params["architecture"]

        if self.architecture == "encoder-decoder":
            self.tgt_initializer = self._init_tgt_initializer(self.model_params["tgt_init"])

        # ============================================================
        # CRITICAL VALIDATION: Encoder-Only Iterative Covariates
        # ============================================================
        # 🏆 DESIGN WIN: This validation is intentionally restrictive to avoid
        #    silent train/inference mismatch. DO NOT remove or make "forward-compatible"
        #    without explicit architectural changes to support covariates correctly.
        #
        #    This is a conscious engineering decision, not a limitation to "fix".
        # ============================================================
        if self.architecture == "encoder-only" and self.strategy == "iterative":
            # NOTE:
            # This restriction is INTENTIONAL. DO NOT GENERALIZE.
            # Encoder-only iterative + covariates cannot be made correct
            # without changing the architecture (decoder or explicit policy).
            #
            # Encoder-only iterative uses pure autoregressive loop that:
            # 1. Slices input to targets-only: x[:, :, :target_f]
            # 2. Zero-pads covariates in ar_encode_fn
            # This creates TRAINING-INFERENCE MISMATCH if covariates are present!
            if (self.feature_layout.past_covariates_size > 0 or
                self.feature_layout.future_covariates_size > 0):
                raise ValueError(
                    "Transformer encoder-only iterative mode does NOT support covariates "
                    "(past_covariates or future_covariates).\n"
                    "\n"
                    "REASON: Covariates are replaced with zero-padding during inference, "
                    "creating training-inference mismatch that degrades performance.\n"
                    "\n"
                    "SOLUTIONS:\n"
                    "  1. Use encoder-decoder iterative (full covariate support)\n"
                    "  2. Use encoder-only direct (historical covariates supported)\n"
                    "  3. Remove covariates from dataset\n"
                    "\n"
                    f"Current dataset has:\n"
                    f"  - past_covariates: {self.feature_layout.past_covariates_size} features\n"
                    f"  - future_covariates: {self.feature_layout.future_covariates_size} features"
                )
            model_forecast_steps = 1
        else:
            model_forecast_steps = forecast_steps

        # Always use the FeatureLayout as the source of truth for input size
        encoder_input_size = self.feature_layout.encoder_input_size
        if self.architecture == "encoder-decoder":
            # Decoder receives: targets + future_covariates
            decoder_input_size = (
                    self.num_features + self.feature_layout.future_covariates_size
            )
        else:
            decoder_input_size = 0

        self.model = TransformerModel(
            encoder_input_size=encoder_input_size,
            decoder_input_size=decoder_input_size,
            num_features=self.num_features,
            forecast_steps=model_forecast_steps,
            window_size=self.window_size,
            hidden_size=self.model_params["hidden_size"],
            num_encoder_layers=self.model_params["num_encoder_layers"],
            num_decoder_layers=self.model_params.get("num_decoder_layers",4),
            num_heads=self.model_params["num_heads"],
            dim_ff_multiplier=self.model_params["dim_ff_multiplier"],
            dropout=self.model_params["dropout"],
            architecture=self.model_params["architecture"],
            positional_encoding_config=pe_config,
            readout=self.model_params.get("readout", "last"),
            attention_type=self.model_params["attention_type"],
            attention_window_size=self.model_params.get("attention_window_size",32),
            use_revin=self.model_params["use_revin"],
            revin_affine=self.model_params["revin_affine"],
            revin_eps=self.model_params["revin_eps"],
            revin_robust=self.model_params["revin_robust"],
            output_head_strategy=self.model_params["output_head_strategy"],
            head_type = self.model_params["head_type"],
            head_dropout = self.model_params.get("head_dropout", 0.0),
            device_safety_checks=self.model_params["device_safety_checks"],
        ).to(self.device)

        # Log AMP inference config early, because it explains Flash SDPA engagement.
        use_amp_inf = bool(self.model_params.get("use_amp_inference", True))
        raw_dtype = self.model_params.get("amp_inference_dtype")
        dtype_str = str(raw_dtype) if raw_dtype else "auto"
        dev = torch.device(self.device) if self.device is not None else torch.device("cpu")
        if dev.type == "cuda" and torch.cuda.is_available():
            if use_amp_inf:
                logger.info(
                    f"[{self.__class__.__name__}] AMP inference ENABLED (dtype={dtype_str}). "
                    "Flash/Efficient SDPA may engage when need_weights=False."
                )
            else:
                logger.warning(
                    f"[{self.__class__.__name__}] AMP inference DISABLED (FP32 inference). "
                    "Flash SDPA likely won't engage."
                )
        else:
            logger.info(f"[{self.__class__.__name__}] Non-CUDA device -> AMP inference inactive.")

        logger.info(
            f"Initialized {self.__class__.__name__} with dynamic sizes: "
            f"encoder_input={encoder_input_size},"
            f" decoder_input={decoder_input_size}, "
            f"total_features={self.feature_layout.total_features}, "
            f"model_internal_horizon={model_forecast_steps}."
        )

        # Initialize SmartPriorGenerator for HPO
        self._smart_prior_generator = SmartPriorGenerator(
            model_params=self.model_params,
            num_features=self.num_features,
            forecast_steps=self.forecast_steps,
            filter_search_space_fn=self.filter_search_space,
            infer_seasonal_period_fn=self._infer_seasonal_period,
            suggest_batch_sizes_fn=self._suggest_batch_sizes,
            is_valid_prior_value_fn=self._is_valid_prior_value
        )

    # =========================================================================
    # INITIALIZATION HELPERS
    # =========================================================================

    def _apply_smart_defaults(self, params: Dict[str, Any]) -> None:
        """
        Apply default parameter values with architecture-specific logic.

        This method provides a single source of truth for all default parameters,
        combining static defaults from TRANSFORMER_DEFAULTS with conditional
        defaults based on architecture and attention configuration.

        Resolution order:
        1. Static defaults from TRANSFORMER_DEFAULTS (unconditional)
        2. Architecture-specific conditionals (encoder-only vs encoder-decoder)
        3. Attention-specific conditionals (local attention window size)

        Args:
            params: Model parameters dict (modified in-place)
        """
        # ═══ STEP 1: Apply Static Defaults ═══
        for key, value in TRANSFORMER_DEFAULTS.items():
            params.setdefault(key, value)

        # ═══ STEP 2: Architecture-Specific Conditionals ═══
        arch = params["architecture"]

        if arch == "encoder-only":
            params.setdefault("readout", "last")

        elif arch == "encoder-decoder":
            params.setdefault("num_decoder_layers", 4)
            # Changed from "last_value" - more robust, avoids outliers
            params.setdefault("tgt_init", "mean")

        # ═══ STEP 3: Attention-Specific Conditionals ═══
        if params["attention_type"] == "local":
            params.setdefault("attention_window_size", 32)

    # =========================================================================
    # SMART HPO IMPLEMENTATION
    # =========================================================================

    def validate_param_combination(self, params: Dict[str, Any]) -> bool:
        """
        Validate Transformer HPO candidate params (fast pre-train rejection).

        Delegates to ParameterValidator which enforces:
        - MHA constraints (divisibility, head_dim range)
        - Dataset-aware complexity thresholds

        Args:
            params: Parameter dictionary to validate

        Returns:
            True if valid, False if should be pruned
        """
        # Robustly resolve dataset length (handles Context vs Direct/Test usage)
        if hasattr(self, 'run_context') and self.run_context and hasattr(self.run_context, 'dataset'):
            training_len = len(self.run_context.dataset.series)
        elif hasattr(self, 'dataset') and self.dataset:
            training_len = len(self.dataset.series)
        else:
            training_len = None  # Let validator use its default

        validator = ParameterValidator(
            model_params=self.model_params,
            num_features=self.num_features,
            dataset_length=training_len
        )
        return validator.validate(params)

    def analyze_search_space(
            self,
            param_space: Dict[str, Any],
            fixed_params: Dict[str, Any],
            n_trials: int
    ) -> List[str]:
        """
        Analyze search space complexity and warn if too large relative to n_trials.

        Delegates to SearchSpaceAnalyzer which provides detailed analysis and warnings.

        Args:
            param_space: HPO search space definition
            fixed_params: Fixed model parameters (unused, kept for interface compatibility)
            n_trials: Number of trials to run

        Returns:
            List of warning/info messages to display to the user
        """
        result = SearchSpaceAnalyzer.analyze(param_space, n_trials)
        return result["warnings"]

    def filter_search_space(
            self,
            param_space: Dict[str, Any],
            fixed_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter search space based on architecture, attention type, and constraints.

        Delegates to SearchSpaceFilter which removes parameters that are not
        applicable given the fixed configuration.

        Args:
            param_space: Parameters to optimize
            fixed_params: Fixed parameters from config

        Returns:
            Filtered param_space with invalid parameters removed
        """
        return SearchSpaceFilter.filter(param_space, fixed_params)

    def suggest_smart_priors(
        self,
        param_space: Dict[str, Any],
        fixed_params: Dict[str, Any],
        dataset: Optional['TimeSeriesDataset'] = None
    ) -> List[Dict[str, Any]]:
        """
        Enhanced smart priors based on architecture, strategy, data dimensions, AND dataset metadata.

        Delegates to SmartPriorGenerator which implements the Strategy Pattern for
        generating optimized starting configurations for different Transformer modes:
        - Encoder-only + Direct: Large capacity, single forward pass
        - Encoder-only + Iterative: Smaller, faster (reused N times)
        - Encoder-decoder + Direct: Balanced, with tgt_init
        - Encoder-decoder + Iterative: Lightweight, speed-critical

        Priors are automatically adapted to:
        - Actual number of features being modeled (num_features)
        - Dataset characteristics (frequency, size, exog variables)
        - Outlier robustness (Huber loss prioritization)

        Args:
            param_space: Parameters being optimized
            fixed_params: Fixed parameters from config
            dataset: Optional TimeSeriesDataset for metadata extraction

        Returns:
            List of 2-3 prior configurations per mode (max 10 total)
        """
        # Delegate to SmartPriorGenerator
        return self._smart_prior_generator.suggest_priors(
            param_space=param_space,
            fixed_params=fixed_params,
            dataset=dataset,
            window_size=getattr(self, "window_size", self.forecast_steps)
        )

    def _suggest_optuna_params(self, trial: Any, param_space: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """
        Dependent sampling logic for Transformer.
        Prioritizes mathematical consistency (hidden_size = n_heads * head_dim).

        Args:
            trial: Optuna trial object
            param_space: Parameter space dictionary
            prefix: Prefix for nested parameter names (used for scheduler_config, etc.)
        """
        params = {}
        # Work on a copy to remove handled keys
        remaining_space = param_space.copy()

        # --- 1. Architecture ---
        if "architecture" in remaining_space:
            arch = trial.suggest_categorical("architecture", remaining_space["architecture"])
            params["architecture"] = arch
            del remaining_space["architecture"]
        else:
            arch = self.model_params.get("architecture")

        # --- 2. Heads & Dimensions ---
        # Check if we optimize num_heads AND hidden_size
        if "num_heads" in remaining_space and "hidden_size" in remaining_space:
            # A. Sample num_heads first
            n_heads = trial.suggest_categorical("num_heads", remaining_space["num_heads"])
            params["num_heads"] = n_heads
            del remaining_space["num_heads"]

            # B. Instead of sampling hidden_size from arbitrary range, sample head_dim
            # This guarantees divisibility.

            # Get max hidden_size constraint
            hs_conf = remaining_space["hidden_size"]
            max_hs = None

            if isinstance(hs_conf, dict) and "max" in hs_conf:
                max_hs = hs_conf["max"]
            elif isinstance(hs_conf, list):
                max_hs = max(hs_conf)

            # Use FIXED choices for Optuna (no dynamic filtering)
            # Optuna requires consistent parameter space across all trials
            all_head_dims = [16, 32, 64, 128]
            head_dim = trial.suggest_categorical("hpo_head_dim", all_head_dims)

            # Calculate hidden_size
            hidden_size = n_heads * head_dim

            # Validate against max constraint AFTER sampling
            if max_hs is not None and hidden_size > max_hs:
                # Raise constraint violation - caller will mark trial as PRUNED
                raise ParameterConstraintViolation(
                    f"n_heads={n_heads} * head_dim={head_dim} = "
                    f"{hidden_size} exceeds max_hidden_size={max_hs}"
                )

            params["hidden_size"] = hidden_size

            # Remove hidden_size from remaining because we set it manually
            del remaining_space["hidden_size"]

        # --- 3. Delegate remaining independent parameters ---
        # Use parent implementation for lr, dropout, layers, etc.
        other_params = super()._suggest_optuna_params(trial, remaining_space, prefix=prefix)
        params.update(other_params)

        return params

    def save_attention_to_disk(self) -> None:
        """
        Save captured attention weights to NPZ + metadata JSON sidecar.
          - NPZ: only arrays
          - JSON: metadata (primary_map, mode, sampling, etc.)
        """
        if not self.run_context:
            return

        # Skip saving artifacts during HPO trials to prevent disk bloat
        if self.run_context.metadata.get('is_hpo_trial', False):
            return

        if not (self.model and hasattr(self.model, "attn_capture")):
            return
        if not self.model.attn_capture.data:
            logger.debug("[Transformer] No attention captured, skipping save.")
            return

        npz_path = self._get_artifact_path(
            category = "attention",
            suffix = "attention",
            extension = "npz"
        )
        meta_path = str(Path(npz_path).with_name(Path(npz_path).stem + "_metadata.json"))
        try:
            # CRITICAL: Synchronize GPU before numpy conversion
            # Attention tensors were transferred asynchronously (.cpu(non_blocking=True))
            # Must wait for all GPU→CPU transfers to complete before accessing data
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Convert all captured tensors to numpy (CPU) at save time
            arrays: Dict[str, np.ndarray] = {}
            for k, t in self.model.attn_capture.data.items():
                if isinstance(t, torch.Tensor):
                    # Tensor already on CPU (async transfer), just convert to numpy
                    arr = t.detach().float().cpu().numpy()
                else:
                    arr = np.asarray(t)

                # Defensive: skip empty arrays (edge cases only)
                if hasattr(arr, "size") and arr.size == 0:
                    logger.warning(f"[Transformer] Skipping empty attention array: {k}")
                    continue

                arrays[k] = arr

            np.savez_compressed(npz_path, **arrays)

            # Build metadata (publication + offline-friendly)
            arch = self.model_params.get("architecture", "encoder-only")
            strategy = self.model_params.get("strategy", "direct")
            n_enc = int(self.model_params.get("num_encoder_layers", 0) or 0)
            n_dec = int(self.model_params.get("num_decoder_layers", 0) or 0)
            attn_type = self.model_params.get("attention_type", "full")
            n_heads = int(self.model_params.get("num_heads", 0) or 0)

            # sampling meta
            sampling_mode = self.model_params.get("attention_capture_sampling", None)
            sampling_steps = self.model_params.get("attention_capture_steps", None)

            keys = sorted(list(arrays.keys()))
            primary = self._choose_primary_map(keys=keys)

            meta = {
                "model": "transformer",
                "architecture": arch,
                "strategy": strategy,
                "window_size": int(self.window_size),
                "forecast_steps": int(self.forecast_steps),
                "attention_type": attn_type,
                "num_heads": n_heads,
                "num_encoder_layers": n_enc,
                "num_decoder_layers": n_dec,
                "batch_averaged": True,
                "heads_preserved": True,
                "sampling_mode": sampling_mode,
                "steps_saved": sampling_steps,
                "primary_map": primary,
                "keys": keys,
                "note": "Attention maps are batch-averaged unless otherwise specified",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"[Transformer] Saved attention to {npz_path}")
            logger.info(f"[Transformer] Saved attention metadata to {meta_path}")

        except Exception as e:
            logger.error(f"[Transformer] Failed to save attention: {e}")

    def _choose_primary_map(self, keys: List[str]) -> str:
        """
        Auto-select a sensible default map for plotting.
        - encoder-only: last encoder layer self-attn
        - encoder-decoder: last decoder layer cross-attn
        Fallback: first key.
        """

        arch = self.model_params.get("architecture", "encoder-only")
        n_enc = int(self.model_params.get("num_encoder_layers", 0) or 0)
        n_dec = int(self.model_params.get("num_decoder_layers", 0) or 0)

        if arch == "encoder-decoder":
            target = f"dec_cross_layer_{max(0, n_dec - 1)}"
            # prefer non-step version if available, else any step version
            if target in keys:
                return target
            for k in keys:
                if k.startswith(target + "_step_"):
                    return k
        else:
            target = f"enc_self_layer_{max(0, n_enc - 1)}"
            if target in keys:
                return target
            for k in keys:
                if k.startswith(target + "_step_"):
                    return k

        return keys[0] if keys else ""

    def _resolve_capture_steps(self, horizon: int) -> Optional[List[int]]:
        """
        Sampling policy:
          - explicit list in attention_capture_steps wins
          - else sampling_mode in {"all","first_last","log"}
          - else None (capture everything caller enables)
        """

        steps = self.model_params.get("attention_capture_steps", None)
        if steps is not None:
            if not isinstance(steps, (list, tuple)):
                raise ValueError("attention_capture_steps must be a list/tuple of ints")

            try:
                coerced = [int(x) for x in steps]
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"attention_capture_steps must contain only integers, got: {steps}"
                ) from e

            s = sorted({x for x in coerced if 0 <= x < horizon})

            # If user provided only out-of-range values, fall back to defaults
            if not s:
                logger.warning(
                    f"[Transformer] attention_capture_steps={steps} filtered to empty for horizon={horizon}. "
                    f"Falling back to sampling_mode/defaults."
                )
                return None

            if 0 not in s:
                s = [0] + s
            if (horizon - 1) not in s:
                s = s + [horizon - 1]
            return s

        mode = self.model_params.get("attention_capture_sampling", None)
        if mode is None:
            return None
        if mode not in {"all", "first_last", "log"}:
            raise ValueError(f"Invalid attention_capture_sampling={mode}. Use all|first_last|log.")

        if mode == "all":
            return list(range(horizon))
        if mode == "first_last":
            return [0, horizon - 1] if horizon > 1 else [0]

        # log sampling: cover early steps densely + include last
        # e.g. horizon=100 -> 0,1,2,4,8,16,32,64,99
        s = set([0, max(0, horizon - 1)])
        k = 1
        while k < horizon - 1:
            s.add(k)
            k *= 2
        return sorted(s)

    def _get_y_window_steps(self) -> int:
        if (self.model_params.get("architecture", "encoder-only") == "encoder-only"
                and self.model_params.get("strategy", "direct") == "iterative"):
            return 1
        return self.forecast_steps

    def _init_tgt_initializer(self, tgt_init: str) -> TgtInitializer:
        """
        Initialize target initializer using factory function.

        Delegates to create_tgt_initializer() which encapsulates all the logic
        for selecting and instantiating the appropriate TgtInitializer subclass.
        """
        return create_tgt_initializer(
            tgt_init=tgt_init,
            decoder_uses_exog=self.feature_layout.future_covariates_size > 0,
            num_exog_decoder=self.feature_layout.future_covariates_size,
            seasonal_period=self.model_params.get("seasonal_period")
        )

    def predict(self, input_data: pd.DataFrame, future_exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Custom predict method for Transformer with attention capture.
        """
        # Capture attention if run_context provided
        should_capture = False
        if self.run_context:
            is_hpo = self.run_context.metadata.get('is_hpo_trial', False)
            should_capture = not is_hpo

        if should_capture:
            with self.capture_attention_weights():  # Uses existing context manager
                predictions = super().predict(input_data, future_exog=future_exog)
                self.save_attention_to_disk()       # Save attention after prediction
        else:
            predictions = super().predict(input_data, future_exog=future_exog)

        return predictions

    # ---------------------------------------------------------------------
    # Hooks used by NeuralTSForecaster.predict()
    # ---------------------------------------------------------------------
    def _prepare_input_tensor(self, input_data: pd.DataFrame) -> torch.Tensor:
        # use full preprocessing
        df_proc = self.preprocessor.transform(input_data, allow_subset=False)
        values = df_proc.values.astype("float32")
        tensor = torch.from_numpy(values).unsqueeze(0)

        # Final safety slice to encoder input size
        if tensor.size(-1) != self.feature_layout.encoder_input_size:
            tensor = tensor[..., self.feature_layout.encoder_feature_idx]

        return tensor

    def _prepare_future_exog_tensor(self, future_exog, batch_size=1):
        """
        Prepare future exogenous tensor for decoder.

        Args:
            future_exog: Future exogenous data (DataFrame or None)
            batch_size: Batch size for zero tensor generation (default: 1)

        Contract:
        - encoder-only: decoder exog is NEVER used -> always return None
        - encoder-decoder:
            - if decoder_exog_columns is empty -> return None
            - if future_exog is None -> return ZERO tensor with correct feature dim and batch_size
            - if future_exog provided -> validate shape and return tensor
        """

        # ─────────────────────────────────────────────
        # 1. Encoder-only architecture → no decoder exog
        # ─────────────────────────────────────────────
        if self.architecture == "encoder-only":
            return None

        # ─────────────────────────────────────────────
        # 2. How many decoder exog features SHOULD exist
        # ─────────────────────────────────────────────
        # NEW API: Use feature_layout instead of dataset
        dec_exog_dim = self.feature_layout.decoder_exog_size

        if dec_exog_dim == 0:
            return None

        # ─────────────────────────────────────────────
        # 3. No future_exog provided → ZERO SLOT (IMPORTANT)
        # ─────────────────────────────────────────────
        if future_exog is None:
            return torch.zeros(
                (batch_size, self.forecast_steps, dec_exog_dim),
                dtype=torch.float32,
                device=self.device,
            )

        # ─────────────────────────────────────────────
        # 4. Convert DataFrame to Tensor if needed
        # ─────────────────────────────────────────────
        if isinstance(future_exog, pd.DataFrame):
            # Extract decoder exog columns (future_covariates in new API)
            decoder_cols = self.feature_layout.decoder_exog_columns
            if not decoder_cols:
                return None

            # Validate that all required columns are present
            missing = set(decoder_cols) - set(future_exog.columns)
            if missing:
                raise ValueError(f"future_exog DataFrame missing required columns: {missing}")

            # Convert to numpy then tensor.
            # .copy() ensures a C-contiguous array: column reordering via fancy
            # indexing produces a non-contiguous view whose strides can be
            # negative, which PyTorch rejects.
            future_exog_np = future_exog[decoder_cols].values.copy()
            future_exog = torch.FloatTensor(future_exog_np).unsqueeze(0).to(self.device)

        # ─────────────────────────────────────────────
        # 5. Validate tensor shape
        # ─────────────────────────────────────────────
        if not isinstance(future_exog, torch.Tensor):
            raise TypeError("future_exog must be a torch.Tensor or pandas.DataFrame")

        if future_exog.ndim != 3:
            raise ValueError(
                f"future_exog must have shape (batch, steps, features), got {future_exog.shape}"
            )

        if future_exog.shape[1] != self.forecast_steps:
            raise ValueError(
                f"future_exog length mismatch: "
                f"expected {self.forecast_steps}, got {future_exog.shape[1]}"
            )

        if future_exog.shape[2] != dec_exog_dim:
            raise ValueError(
                f"future_exog feature dim mismatch: "
                f"expected {dec_exog_dim}, got {future_exog.shape[2]}"
            )

        return future_exog.to(self.device)

    def _internal_predict(self, input_tensor: torch.Tensor, **kwargs) -> np.ndarray:
        """
        Dispatcher-only internal predict:
          - routes to direct/iterative path based on configured strategy
          - keeps all model calling logic inside _predict_direct/_predict_iterative
        """
        if self.model is None or not self.fitted:
            raise ValueError("Model must be initialized and fitted.")

        if self.strategy == "direct":
            return self._predict_direct(input_tensor, **kwargs)
        else:
            return self._predict_iterative(input_tensor, **kwargs)

    def _prepare_inference_inputs(
        self,
        input_tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
        """
        Prepare inputs for inference: move to device, set eval mode.

        This method eliminates duplication between direct and iterative prediction
        by centralizing the common setup logic.

        Args:
            input_tensor: Input tensor to prepare
            **kwargs: Additional arguments (may contain future_exog_tensor)

        Returns:
            Tuple of (x, future_exog, device) where:
                - x: Input tensor moved to device
                - future_exog: Exogenous features moved to device (or None)
                - device: Device string for reference
        """
        self.model.eval()
        device = self.device

        # Move tensors to device with non_blocking for async transfer
        x = input_tensor.to(device, non_blocking=True)
        future_exog = kwargs.get("future_exog_tensor")
        if future_exog is not None:
            future_exog = future_exog.to(device, non_blocking=True)

        return x, future_exog, device

    @contextlib.contextmanager
    def _inference_execution_context(self):
        """
        Setup execution context for inference: no_grad + AMP + SDPA.

        This context manager encapsulates the nested context setup required
        for inference, including:
        - torch.no_grad() for inference mode
        - SDPA backend selection
        - AMP (Automatic Mixed Precision) configuration

        Yields within the active inference context.
        """
        use_amp_infer = bool(self.model_params.get("use_amp_inference", True))
        amp_dtype = self.model_params.get("amp_inference_dtype")

        if amp_dtype:
            dtype_map = {
                "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                "fp16": torch.float16, "float16": torch.float16
            }
            amp_dtype = dtype_map.get(str(amp_dtype).lower())

        with torch.no_grad(), \
             self._get_inference_sdpa_ctx(), \
             get_autocast_context(self.device, use_amp=use_amp_infer, dtype=amp_dtype):
            yield

    def _predict_direct(self, input_tensor: torch.Tensor, **kwargs) -> np.ndarray:
        """
        Direct prediction strategy: one-shot forecast.

        Refactored to use helper methods for inference setup (DRY principle).
        """
        # Prepare inputs (eliminates 19 lines of duplication)
        x, future_exog, _device = self._prepare_inference_inputs(input_tensor, **kwargs)
        uses_decoder = self.model.architecture == "encoder-decoder"

        # Execute within proper inference context
        with self._inference_execution_context():
            if hasattr(self.model, "attn_capture") and self.model.attn_capture.enabled:
                self.model.attn_capture.set_step(None)

            if uses_decoder:
                # ------------------------------------------------------------------
                # Explicitly slice input to encoder_input_size to exclude
                # decoder-only variables if they are present in the input tensor.
                # ------------------------------------------------------------------
                enc_size = self.feature_layout.encoder_input_size
                x_enc = x[..., :enc_size]
                memory = self.model.encode(x_enc)
                tgt = self.tgt_initializer.initialize_direct(
                    src=x,
                    forecast_steps=self.forecast_steps,
                    num_features=self.num_features,
                    device=self.device,
                    future_exog_tensor=future_exog,
                )
                # tgt is already on device from initialize_direct
                decoder_out = self.model.decode(tgt, memory)
                output = self.model.output_head(decoder_out)
                output = self.model.denormalize_output(output)
            else:
                output = self.model(x)

        return output.float().cpu().numpy()

    def _predict_iterative_buffer(self, input_tensor: torch.Tensor, **kwargs) -> np.ndarray:
        """
        Iterative encoder-decoder prediction with preallocated buffer (memory-optimized).

        OPTIMIZATION: Eliminates O(H) allocations and O(H²×B×F) memory copies by
        preallocating full decoder sequence buffer upfront.

        SEMANTIC GUARANTEE: Decoder sees identical prefix sequences as concat mode.
        Only the memory construction mechanism differs.

        COMPLEXITY:
        - Runtime allocations: O(1) (vs O(H) in concat mode)
        - Memory copies: O(H×B×F) (vs O(H²×B×F) in concat mode)
        - Decoder attention: O(H³) (unchanged)

        WHEN TO USE:
        - H > 512: Noticeable performance improvement (10-30% faster)
        - H > 1024: Critical (can be 2-3× faster)
        - Low-latency production: Predictable memory usage, no fragmentation

        Returns:
            np.ndarray: Predictions (H, F)
        """
        # ------------------------------------------------------------
        # Prepare inputs
        # ------------------------------------------------------------
        x, future_exog, device = self._prepare_inference_inputs(input_tensor, **kwargs)

        if self.model.architecture != "encoder-decoder":
            raise ValueError(
                "Buffer mode is only supported for encoder-decoder architecture. "
                "Encoder-only iterative uses AR loop without concat overhead."
            )

        # ------------------------------------------------------------
        # Attention capture
        # ------------------------------------------------------------
        if hasattr(self.model, "attn_capture") and self.model.attn_capture.enabled:
            steps = self._resolve_capture_steps(self.forecast_steps)
            self.model.attn_capture.configure(enabled=True, steps_to_capture=steps)

        # ------------------------------------------------------------
        # Inference context
        # ------------------------------------------------------------
        with self._inference_execution_context():
            # ------------------------------------------------------------------
            # Encode input
            # ------------------------------------------------------------------
            enc_size = self.feature_layout.encoder_input_size
            x_enc = x[..., :enc_size]
            memory = self.model.encode(x_enc)

            # ------------------------------------------------------------------
            # Initialize decoder sequence
            # ------------------------------------------------------------------
            tgt_init = self.tgt_initializer.initialize_iterative(
                src=x,
                num_features=self.num_features,
                device=device,
                future_exog_tensor=future_exog,
                step=0,
            )

            expected = self.feature_layout.decoder_input_size
            if tgt_init.shape[-1] != expected:
                raise RuntimeError(
                    f"TgtInitializer returned invalid feature dim: "
                    f"{tgt_init.shape[-1]} != {expected}"
                )

            B, init_len, F_dec = tgt_init.shape
            H = self.forecast_steps
            F = self.num_features

            # ------------------------------------------------------------------
            # PREALLOCATE full decoder buffer: (B, init_len + H, F_dec)
            # ------------------------------------------------------------------
            tgt_buffer = torch.zeros(
                (B, init_len + H, F_dec),
                dtype=tgt_init.dtype,
                device=device
            )

            # Initialize buffer with tgt_init
            tgt_buffer[:, :init_len, :] = tgt_init

            # Output buffer
            output = torch.zeros((B, H, F), dtype=tgt_init.dtype, device=device)

            # ------------------------------------------------------------------
            # Autoregressive loop (NO CONCAT - use views into buffer)
            # ------------------------------------------------------------------
            # Pre-check attention capture (avoids hasattr in hot loop)
            capture_enabled = hasattr(self.model, "attn_capture") and self.model.attn_capture.enabled

            for step in range(H):
                if capture_enabled:
                    self.model.attn_capture.set_step(step)

                # View into buffer (zero-cost, just pointer arithmetic)
                # Decoder sees: [tgt_0, tgt_1, ..., tgt_{step-1}]
                tgt_view = tgt_buffer[:, :init_len + step, :]

                # Decode
                dec_out = self.model.decode(tgt_view, memory)
                one_step = self.model.output_head(dec_out[:, -1:, :])
                one_step = self.model.denormalize_output(one_step)

                # Store output
                output[:, step:step + 1, :] = one_step

                if step == H - 1:
                    break

                # Prepare next decoder input (in-place write to buffer)
                next_tgt = one_step.squeeze(1)  # (B, F)

                # Append future_covariates if needed
                if future_exog is not None and self.feature_layout.future_covariates_size > 0:
                    ex = future_exog[:, step + 1, : self.feature_layout.future_covariates_size]
                    # Concatenate on feature dim (still needed, but cheap: (B, F) + (B, F_exog))
                    next_tgt = torch.cat([next_tgt, ex], dim=1)

                # Write directly into buffer (in-place, NO allocation)
                tgt_buffer[:, init_len + step, :] = next_tgt

            return self._nan_guard_to_numpy(output, context="iterative encoder-decoder (buffer mode)")

    def _predict_iterative(self, input_tensor: torch.Tensor, **kwargs) -> np.ndarray:
        """
        Iterative prediction strategy with covariate handling.

        MEMORY OPTIMIZATION (NEW):
        ────────────────────────────────────────────────────────────────────
        Encoder-decoder iterative supports two modes (controlled by config):
          - "concat" (default for H≤512): Current implementation with O(H²) memory copies
          - "buffer" (auto-enabled for H>512): Preallocated buffer with O(1) allocations
          - "auto": Automatically selects buffer for H>512, concat otherwise

        Config parameter: `iterative_decoder_mode` (default: "auto")
        See: docs/analysis_oh2_concat_problem.md for details

        COVARIATE CONTRACT (v2 API):
        ────────────────────────────────────────────────────────────────────
        Encoder-decoder iterative:
          - past_covariates: NOT used (encoder sees full historical window)
          - future_covariates: Fully supported via decoder path

        Encoder-only iterative:
          - past_covariates: Currently pure AR (no covariates consumed)
          - future_covariates: Stored but NOT consumed (forward-compatible)
          - Ready for PastCovariatePolicy implementation in future

        ⚠️  WARNING: Known limitations of iterative Transformer inference
        ────────────────────────────────────────────────────────────────────
        Encoder-decoder:
          - O(H) decoder growth (tgt concatenation at each step in concat mode)
          - Teacher-forced training vs autoregressive inference mismatch
            mitigated only partially via auxiliary_loss / prediction_noise
          - Buffer mode eliminates concat overhead but NOT decoder attention cost (O(H³))

        Encoder-only:
          - Pure autoregressive loop (sliding window approach)
          - NO covariate consumption (past or future) - intentional restriction
          - Zero-padding used internally for feature alignment
        """

        # ------------------------------------------------------------
        # Dispatch: Buffer mode vs Concat mode (encoder-decoder only)
        # ------------------------------------------------------------
        mode = self.model_params.get("iterative_decoder_mode", "auto")

        # Auto-select based on horizon
        if mode == "auto":
            # Buffer mode beneficial for H > 96 (5-30% speedup, scales with H)
            # Negligible overhead for small H, significant gains for large H
            # See: docs/analysis_oh2_concat_problem.md, docs/analysis_transformer_performance_bottlenecks.md
            mode = "buffer" if self.forecast_steps > 96 else "concat"

        # Validate mode
        if mode not in ("concat", "buffer"):
            raise ValueError(
                f"Invalid iterative_decoder_mode: {mode}. "
                f"Must be 'concat', 'buffer', or 'auto'."
            )

        # Dispatch to buffer mode for encoder-decoder (if enabled)
        if mode == "buffer" and self.model.architecture == "encoder-decoder":
            logger.debug(
                f"Using buffer mode for iterative decoder (H={self.forecast_steps})"
            )
            return self._predict_iterative_buffer(input_tensor, **kwargs)

        # Otherwise, use concat mode (current implementation)
        # Note: Encoder-only always uses AR loop (no concat overhead)

        # ------------------------------------------------------------
        # Prepare inputs
        # ------------------------------------------------------------
        x, future_exog, device = self._prepare_inference_inputs(input_tensor, **kwargs)

        uses_decoder = (self.model.architecture == "encoder-decoder")

        # ------------------------------------------------------------
        # Attention capture
        # ------------------------------------------------------------
        if hasattr(self.model, "attn_capture") and self.model.attn_capture.enabled:
            steps = self._resolve_capture_steps(self.forecast_steps)
            self.model.attn_capture.configure(enabled=True, steps_to_capture=steps)

        # ------------------------------------------------------------
        # Inference context
        # ------------------------------------------------------------
        with self._inference_execution_context():

            # ============================================================
            # 1) ITERATIVE ENCODER–DECODER (NO PC EVER)
            # ============================================================
            if uses_decoder:
                # ------------------------------------------------------------------
                # Explicitly slice input to encoder_input_size to exclude
                # decoder-only variables if they are present in the input tensor.
                # ------------------------------------------------------------------
                enc_size = self.feature_layout.encoder_input_size
                x_enc = x[..., :enc_size]
                memory = self.model.encode(x_enc)

                tgt_init = self.tgt_initializer.initialize_iterative(
                    src=x,
                    num_features=self.num_features,
                    device=device,
                    future_exog_tensor=future_exog,
                    step=0,
                )

                expected = self.feature_layout.decoder_input_size
                if tgt_init.shape[-1] != expected:
                    raise RuntimeError(
                        f"TgtInitializer returned invalid feature dim: "
                        f"{tgt_init.shape[-1]} != {expected}"
                    )

                B, _init_len, _ = tgt_init.shape  # init_len reserved for future logic
                H = self.forecast_steps
                F = self.num_features

                output = torch.zeros((B, H, F), dtype=tgt_init.dtype, device=device)
                tgt = tgt_init

                # Pre-check attention capture (avoids hasattr in hot loop)
                capture_enabled = hasattr(self.model, "attn_capture") and self.model.attn_capture.enabled

                for step in range(H):
                    if capture_enabled:
                        self.model.attn_capture.set_step(step)

                    dec_out = self.model.decode(tgt, memory)
                    one_step = self.model.output_head(dec_out[:, -1:, :])
                    one_step = self.model.denormalize_output(one_step)

                    output[:, step:step + 1, :] = one_step

                    if step == H - 1:
                        break

                    next_tgt = one_step
                    # Append future_covariates for next decoder step
                    if future_exog is not None and self.feature_layout.future_covariates_size > 0:
                        ex = future_exog[:, step + 1:step + 2, : self.feature_layout.future_covariates_size]
                        next_tgt = torch.cat([next_tgt, ex], dim=2)

                    tgt = torch.cat([tgt, next_tgt], dim=1)

                return self._nan_guard_to_numpy(output, context="iterative encoder-decoder")

            # ============================================================
            # 2) ENCODER-ONLY — PURE AR
            # ============================================================

            expected_f = self.feature_layout.encoder_input_size
            target_f = self.num_features

            def ar_encode_fn(x_step: torch.Tensor) -> torch.Tensor:
                if expected_f == target_f:
                    return self.model.encode(x_step)

                pad = torch.zeros(
                    x_step.size(0),
                    x_step.size(1),
                    expected_f - target_f,
                    device=x_step.device,
                    dtype=x_step.dtype,
                )
                return self.model.encode(torch.cat([x_step, pad], dim=2))

            return self._nan_guard_to_numpy(
                self._predict_iterative_ar(
                    input_tensor=x[..., :target_f],
                    window_size=self.window_size,
                    forecast_steps=self.forecast_steps,
                    num_features=target_f,
                    encode_fn=ar_encode_fn,
                    readout_fn=self.model._encoder_readout_only,
                ),
                context="encoder-only iterative AR",
            )

    def _nan_guard_to_numpy(self, tensor: torch.Tensor, context: str) -> np.ndarray:
        """
        Shared guard for inference paths.
        Ensures non-finite iterative predictions are handled consistently.

        Performance Note:
        - NaN checking forces GPU→CPU synchronization (5-10% overhead)
        - Disable via config `nan_guard_enabled: false` for production if predictions are stable
        - Enabled by default for safety
        """
        tensor = tensor.float()

        # Fast path: Skip NaN check if disabled (production performance optimization)
        if not self.model_params.get("nan_guard_enabled", True):
            return tensor.cpu().numpy()

        # Full check (safe default, small overhead)
        if not torch.isfinite(tensor).all():
            logger.warning(
                f"[TransformerForecaster] Non-finite values detected during {context}. "
                "Returning NaN array."
            )
            nan = torch.full_like(tensor, float("nan"))
            return nan.cpu().numpy()
        return tensor.cpu().numpy()

    @property
    def continuous_size(self) -> int:
        return self.feature_layout.encoder_input_size

    def _train_model(self, X_train, y_train, X_val, y_val, **kwargs) -> nn.Module:
        y_decoder_exog_train = kwargs.get('y_decoder_exog_train')
        y_decoder_exog_val = kwargs.get('y_decoder_exog_val')

        # ─────────────────────────────────────────────────────────────
        # EXTRACT TRAINING-INFERENCE MISMATCH REDUCTION CONFIGS
        # ─────────────────────────────────────────────────────────────
        aux_loss_config = self.model_params.get("auxiliary_loss", {})
        noise_config = self.model_params.get("prediction_noise", {})

        # Build noise config for build_tgt_train
        noise_config_for_build = None
        if noise_config.get("enabled", False):
            noise_config_for_build = {
                'enabled': True,
                'std': noise_config.get('std', 0.05),
                'schedule': noise_config.get('schedule', 'constant'),
                'training_progress': 0.0,  # Will be updated if curriculum
                'apply_to_exog': noise_config.get('apply_to_exog', False),
            }

        # ─────────────────────────────────────────────────────────────
        # BUILD DECODER INPUTS WITH OPTIONAL NOISE INJECTION
        # ─────────────────────────────────────────────────────────────
        tgt_train, tgt_val = None, None
        if self.model_params["architecture"] == 'encoder-decoder':
            if not hasattr(self, "tgt_initializer"):
                raise RuntimeError("tgt_initializer missing.")

            tgt_train = build_tgt_train(
                y_train, X_train, self.tgt_initializer, y_decoder_exog_train,
                noise_config=noise_config_for_build
            )

            if X_val.shape[0] > 0:
                # IMPORTANT: No noise injection for validation
                tgt_val = build_tgt_train(
                    y_val, X_val, self.tgt_initializer, y_decoder_exog_val,
                    noise_config=None  # Never inject noise in validation
                )

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
                    batch_size=self.model_params["batch_size"],
                    max_epochs=self.model_params["epochs"],
                    default_lr=self.model_params["learning_rate"]
                )
            # ─────────────────────────────────────────────────────────────

            # ─────────────────────────────────────────────────────────────
            # CREATE LOSS FUNCTION WITH OPTIONAL AUXILIARY MULTI-STEP LOSS
            # ─────────────────────────────────────────────────────────────
            base_loss = self._get_loss_function()

            if aux_loss_config.get("enabled", False):
                auxiliary_weight = aux_loss_config.get("weight", 0.1)
                position_weighting = aux_loss_config.get("position_weighting", True)

                criterion = AuxiliaryMultiStepLoss(
                    base_loss=base_loss,
                    auxiliary_weight=auxiliary_weight,
                    position_weighting=position_weighting,
                    reduction='mean'
                )

                logger.info(
                    f"[TransformerForecaster] Auxiliary Multi-Step Loss enabled: "
                    f"weight={auxiliary_weight:.3f}, position_weighting={position_weighting}"
                )
            else:
                criterion = base_loss

            # Log prediction noise configuration
            if noise_config.get("enabled", False):
                logger.info(
                    f"[TransformerForecaster] Prediction noise injection enabled: "
                    f"std={noise_config.get('std', 0.05):.4f}, "
                    f"schedule='{noise_config.get('schedule', 'constant')}'"
                )

            # ─────────────────────────────────────────────────────────────

            # CRITICAL: Don't use 'or' - None is a valid value (disables early stopping)
            if "early_stopping_patience" in kwargs:
                patience = kwargs["early_stopping_patience"]  # Can be None (for is_final_fit)
            else:
                patience = self._get_training_param("early_stopping_patience")

            fail_on_instability = kwargs.get("fail_on_instability")

            # Extract scaler params for original-scale validation metrics (best practice)
            scaler_params = None
            if hasattr(self, 'preprocessor') and self.preprocessor is not None:
                scaler_params = self.preprocessor.get_fast_inverse_scaling_params()
                if scaler_params is not None:
                    logger.info("[Transformer] Validation metrics will be computed in ORIGINAL scale")

            trained_model_instance, history = run_train_loop(
                model=self.model,
                encoder_inputs_train=X_train,
                decoder_inputs_train=tgt_train,
                true_outputs_train=y_train,
                encoder_inputs_val=X_val,
                decoder_inputs_val=tgt_val,
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
                use_amp=self.model_params.get("use_amp", True),
                max_grad_norm = self.model_params.get("max_grad_norm", 1.0),
                save_horizon_csv = self.model_params.get("save_horizon_csv", False),
                auto_tune_horizon = self.model_params.get("auto_tune_horizon", False),
                degradation_threshold = self.model_params.get("degradation_threshold", 3.0),
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

    def _validate_model_params(self) -> None:
        """Validate essential model parameters from the configuration."""
        required_params = {"hidden_size", "num_heads", "num_encoder_layers"}
        missing = [p for p in required_params if p not in self.model_params]
        if missing:
            raise ValueError(f"Missing required parameter(s): {missing}")
        attn_type = self.model_params["attention_type"]
        if attn_type not in {"full", "local"}:
            raise ValueError(f"Invalid attention_type '{attn_type}'.")
        if attn_type == "local":
            if self.model_params["attention_window_size"] is None:
                raise ValueError("`attention_window_size` required for 'local'.")
        architecture = self.model_params["architecture"]
        if architecture == "encoder-only":
            readout = self.model_params["readout"]
            if readout not in {"last", "mean", "max", "cls"}:
                raise ValueError("readout must be one of {'last', 'mean', 'max', 'cls'} for encoder-only.")

        # ═══════════════════════════════════════════════════════════════════
        # ARCHITECTURE-SPECIFIC WARNINGS (unused parameters)
        # ═══════════════════════════════════════════════════════════════════
        architecture = self.model_params.get("architecture", "encoder-only")

        if architecture == "encoder-only":
            unused_params = []
            if "num_decoder_layers" in self.model_params:
                unused_params.append("num_decoder_layers")
            if "tgt_init" in self.model_params:
                unused_params.append("tgt_init")

            if unused_params:
                logger.warning(
                    f"[Transformer] Encoder-only architecture has unused parameters: "
                    f"{unused_params}. These will be ignored."
                )

        elif architecture == "encoder-decoder":
            if "readout" in self.model_params:
                logger.warning(
                    "[Transformer] Encoder-decoder doesn't use 'readout' parameter. "
                    "It will be ignored."
                )

        # ═══════════════════════════════════════════════════════════════════
        # ATTENTION-SPECIFIC WARNINGS
        # ═══════════════════════════════════════════════════════════════════
        attn_type = self.model_params.get("attention_type", "full")
        if attn_type == "full" and "attention_window_size" in self.model_params:
            logger.warning(
                "[Transformer] 'attention_window_size' set but attention_type is 'full'. "
                "This parameter will be ignored."
            )

        # ═══════════════════════════════════════════════════════════════════
        # TRAINING-INFERENCE MISMATCH REDUCTION FEATURES VALIDATION
        # ═══════════════════════════════════════════════════════════════════

        # Validate auxiliary_loss configuration
        aux_loss_config = self.model_params.get("auxiliary_loss", {})
        if aux_loss_config.get("enabled", False):
            weight = aux_loss_config.get("weight", 0.1)
            if not 0.0 <= weight <= 1.0:
                raise ValueError(
                    f"auxiliary_loss.weight must be in [0, 1], got {weight}"
                )

            # Auxiliary loss only makes sense for iterative/autoregressive strategies
            strategy = self.model_params.get("strategy", "direct")
            if strategy != "iterative":
                logger.warning(
                    f"[Transformer] auxiliary_loss is enabled but strategy is '{strategy}'. "
                    "Auxiliary loss is most effective with iterative strategy where training-inference "
                    "mismatch is most pronounced. Consider using strategy='iterative'."
                )

        # Validate prediction_noise configuration
        noise_config = self.model_params.get("prediction_noise", {})
        if noise_config.get("enabled", False):
            std = noise_config.get("std", 0.05)
            if std <= 0:
                raise ValueError(
                    f"prediction_noise.std must be positive, got {std}"
                )
            if std > 1.0:
                logger.warning(
                    f"[Transformer] prediction_noise.std={std:.3f} is very high (>1.0). "
                    "This may destabilize training. Recommended range: [0.01, 0.2]"
                )

            schedule = noise_config.get("schedule", "constant")
            if schedule not in {"constant", "curriculum"}:
                raise ValueError(
                    f"prediction_noise.schedule must be 'constant' or 'curriculum', got '{schedule}'"
                )

            # Noise only makes sense for encoder-decoder
            if self.model_params.get("architecture") != "encoder-decoder":
                logger.warning(
                    "[Transformer] prediction_noise is enabled but architecture is not 'encoder-decoder'. "
                    "Noise injection is designed for encoder-decoder iterative training. "
                    "It will have no effect on encoder-only models."
                )

            # Most effective with iterative strategy
            strategy = self.model_params.get("strategy", "direct")
            if strategy != "iterative":
                logger.warning(
                    f"[Transformer] prediction_noise is enabled but strategy is '{strategy}'. "
                    "Noise injection is most effective with iterative strategy."
                )
