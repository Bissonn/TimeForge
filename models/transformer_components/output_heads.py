"""
Output head strategies for time series forecasting.

This module implements the Strategy Pattern for projecting latent
representations to time series forecasts. It supports multiple
architectures (encoder-only, encoder-decoder) and projection
strategies (shared, independent).

Classes:
    TimeSeriesOutputHead: Abstract base class
    EncoderOnlySharedHead: Shared projection for encoder-only
    EncoderOnlyIndependentHead: Independent heads for encoder-only
    DecoderSharedHead: Shared projection for decoder
    DecoderIndependentHead: Independent heads for decoder

Functions:
    create_output_head: Factory for creating head strategies
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TimeSeriesOutputHead(nn.Module, ABC):
    """
    Abstract base class for output projection strategies.

    All head strategies must implement forward() method that
    projects input tensors to forecast tensors.
    """

    def __init__(self, forecast_steps: int, num_features: int):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.num_features = num_features

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to forecast.

        Args:
            x: Input tensor (shape depends on architecture/strategy)

        Returns:
            [Batch, Forecast_Steps, Num_Features] forecast tensor

        Raises:
            RuntimeError: If input shape is invalid
        """
        pass

    def _validate_output_shape(
            self,
            output: torch.Tensor,
            batch_size: int
    ) -> None:
        """Validate output tensor has correct shape."""
        expected_shape = (batch_size, self.forecast_steps, self.num_features)
        if output.shape != expected_shape:
            raise RuntimeError(
                f"{self.__class__.__name__}: Output shape mismatch. "
                f"Expected {expected_shape}, got {output.shape}"
            )


class EncoderOnlySharedHead(TimeSeriesOutputHead):
    """
    Shared projection head for encoder-only architecture.

    Projects pooled encoder output to forecast via single linear/MLP layer.
    Captures cross-channel and cross-temporal correlations in final projection.

    Mathematical Formulation:
        Linear: y = W @ x + b, where W ∈ ℝ^(S×F × D)
        MLP:    y = W₂ @ GELU(W₁ @ x + b₁) + b₂

    Shape Transformation:
        [Batch, Input_Dim] → [Batch, Steps × Features] → [Batch, Steps, Features]

    Args:
        input_dim: Dimension of pooled encoder output
        forecast_steps: Number of timesteps to forecast
        num_features: Number of output features
        head_type: "linear" (fast) or "mlp" (expressive)
        dropout: Dropout rate for MLP (default: 0.0)

    Example:
        >>> head = EncoderOnlySharedHead(128, 24, 7, 'linear')
        >>> x = torch.randn(32, 128)
        >>> out = head(x)
        >>> out.shape
        torch.Size([32, 24, 7])
    """

    def __init__(
            self,
            input_dim: int,
            forecast_steps: int,
            num_features: int,
            head_type: str = "linear",
            dropout: float = 0.0,
    ):
        super().__init__(forecast_steps, num_features)

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if forecast_steps <= 0:
            raise ValueError(f"forecast_steps must be positive, got {forecast_steps}")
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")

        self.input_dim = input_dim
        self.head_type = head_type
        output_dim = forecast_steps * num_features

        # Build projection
        if head_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim, output_dim)
            )
        else:
            self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Input_Dim] pooled encoder output

        Returns:
            [Batch, Forecast_Steps, Num_Features] forecast

        Raises:
            RuntimeError: If input shape is invalid
        """
        # Validate input dimension
        if x.dim() != 2:
            raise RuntimeError(
                f"{self.__class__.__name__}: Expected 2D input [Batch, Input_Dim], "
                f"got {x.dim()}D tensor with shape {x.shape}"
            )

        batch_size = x.shape[0]

        if x.shape[1] != self.input_dim:
            raise RuntimeError(
                f"{self.__class__.__name__}: Expected input_dim={self.input_dim}, "
                f"got {x.shape[1]}"
            )

        # Project to flat output
        out_flat = self.projection(x)  # [Batch, Steps * Features]

        # Reshape to [Batch, Steps, Features]
        # Use reshape (not view) for safety with non-contiguous tensors
        output = out_flat.reshape(batch_size, self.forecast_steps, self.num_features)

        # Validate output
        self._validate_output_shape(output, batch_size)

        return output

    def __repr__(self) -> str:
        proj_type = 'mlp' if isinstance(self.projection, nn.Sequential) else 'linear'
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"forecast_steps={self.forecast_steps}, "
            f"num_features={self.num_features}, "
            f"head_type={proj_type})"
        )


class EncoderOnlyIndependentHead(TimeSeriesOutputHead):
    """
    Independent projection heads for encoder-only architecture (legacy).

    Uses separate linear layer for each output feature. Does NOT capture
    cross-channel correlations in the final projection layer.

    Note: Generally not recommended. Use EncoderOnlySharedHead instead
    for better performance and cross-channel modeling.

    Shape Transformation:
        [Batch, Input_Dim] → [Batch, Steps] (per feature) → [Batch, Steps, Features]

    Args:
        input_dim: Dimension of pooled encoder output
        forecast_steps: Number of timesteps to forecast
        num_features: Number of output features

    Example:
        >>> head = EncoderOnlyIndependentHead(128, 24, 7)
        >>> x = torch.randn(32, 128)
        >>> out = head(x)
        >>> out.shape
        torch.Size([32, 24, 7])
    """

    def __init__(
            self,
            input_dim: int,
            forecast_steps: int,
            num_features: int
    ):
        super().__init__(forecast_steps, num_features)

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if forecast_steps <= 0:
            raise ValueError(f"forecast_steps must be positive, got {forecast_steps}")
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")

        self.input_dim = input_dim

        # Each head projects: Input_Dim → Forecast_Steps
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, forecast_steps)
            for _ in range(num_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Input_Dim] pooled encoder output

        Returns:
            [Batch, Forecast_Steps, Num_Features] forecast

        Raises:
            RuntimeError: If input shape is invalid
        """
        # Validate input
        if x.dim() != 2:
            raise RuntimeError(
                f"{self.__class__.__name__}: Expected 2D input [Batch, Input_Dim], "
                f"got {x.dim()}D tensor with shape {x.shape}"
            )

        batch_size = x.shape[0]

        if x.shape[1] != self.input_dim:
            raise RuntimeError(
                f"{self.__class__.__name__}: Expected input_dim={self.input_dim}, "
                f"got {x.shape[1]}"
            )

        # Apply each head independently: [Batch, Input_Dim] → [Batch, Steps]
        outputs = [head(x) for head in self.heads]

        # Stack along last dimension: [Batch, Steps, Features]
        output = torch.stack(outputs, dim=-1)

        # Validate output
        self._validate_output_shape(output, batch_size)

        return output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"forecast_steps={self.forecast_steps}, "
            f"num_features={self.num_features})"
        )


class DecoderSharedHead(TimeSeriesOutputHead):
    """
    Shared projection head for encoder-decoder architecture.

    Projects each decoder timestep from hidden dimension to output features.
    Applied uniformly across all timesteps.

    Shape Transformation:
        [Batch, Steps, Hidden] → [Batch, Steps, Features]

    Args:
        input_dim: Hidden dimension of decoder output
        num_features: Number of output features
        head_type: "linear" or "mlp"
        dropout: Dropout rate for MLP

    Example:
        >>> head = DecoderSharedHead(128, 7, 'linear')
        >>> x = torch.randn(32, 24, 128)
        >>> out = head(x)
        >>> out.shape
        torch.Size([32, 24, 7])
    """

    def __init__(
            self,
            input_dim: int,
            num_features: int,
            head_type: str = "linear",
            dropout: float = 0.0
    ):
        # forecast_steps is dynamic (determined by input sequence length)
        super().__init__(forecast_steps=0, num_features=num_features)

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")

        self.input_dim = input_dim

        # Build projection
        if head_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim, num_features)
            )
        else:
            self.projection = nn.Linear(input_dim, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Steps, Hidden] decoder output

        Returns:
            [Batch, Steps, Features] forecast

        Raises:
            RuntimeError: If input shape is invalid
        """
        # Validate input
        if x.dim() != 3:
            raise RuntimeError(
                f"{self.__class__.__name__}: Expected 3D input [Batch, Steps, Hidden], "
                f"got {x.dim()}D tensor with shape {x.shape}"
            )

        if x.shape[2] != self.input_dim:
            raise RuntimeError(
                f"{self.__class__.__name__}: Expected hidden_dim={self.input_dim}, "
                f"got {x.shape[2]}"
            )

        # Project last dimension: [Batch, Steps, Hidden] → [Batch, Steps, Features]
        # Linear layer automatically applies to last dimension
        output = self.projection(x)

        # Validate output features
        expected_features = self.num_features
        if output.shape[2] != expected_features:
            raise RuntimeError(
                f"{self.__class__.__name__}: Output feature mismatch. "
                f"Expected {expected_features}, got {output.shape[2]}"
            )

        return output

    def __repr__(self) -> str:
        proj_type = 'mlp' if isinstance(self.projection, nn.Sequential) else 'linear'
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"num_features={self.num_features}, "
            f"head_type={proj_type})"
        )


class DecoderIndependentHead(TimeSeriesOutputHead):
    """
    Independent projection heads for decoder (legacy).

    Uses separate linear layer for each output feature.

    Shape Transformation:
        [Batch, Steps, Hidden] → [Batch, Steps, 1] (per feature) → [Batch, Steps, Features]

    Args:
        input_dim: Hidden dimension of decoder output
        num_features: Number of output features

    Example:
        >>> head = DecoderIndependentHead(128, 7)
        >>> x = torch.randn(32, 24, 128)
        >>> out = head(x)
        >>> out.shape
        torch.Size([32, 24, 7])
    """

    def __init__(self, input_dim: int, num_features: int):
        super().__init__(forecast_steps=0, num_features=num_features)

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")

        self.input_dim = input_dim

        # Each head projects: Hidden → 1
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, 1)
            for _ in range(num_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Steps, Hidden] decoder output

        Returns:
            [Batch, Steps, Features] forecast

        Raises:
            RuntimeError: If input shape is invalid
        """
        # Validate input
        if x.dim() != 3:
            raise RuntimeError(
                f"{self.__class__.__name__}: Expected 3D input [Batch, Steps, Hidden], "
                f"got {x.dim()}D tensor with shape {x.shape}"
            )

        if x.shape[2] != self.input_dim:
            raise RuntimeError(
                f"{self.__class__.__name__}: Expected hidden_dim={self.input_dim}, "
                f"got {x.shape[2]}"
            )

        # Apply each head: [Batch, Steps, Hidden] → [Batch, Steps, 1]
        outputs = [head(x) for head in self.heads]

        # Concatenate: [Batch, Steps, Features]
        output = torch.cat(outputs, dim=-1)

        # Validate output features
        expected_features = self.num_features
        if output.shape[2] != expected_features:
            raise RuntimeError(
                f"{self.__class__.__name__}: Output feature mismatch. "
                f"Expected {expected_features}, got {output.shape[2]}"
            )

        return output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"num_features={self.num_features})"
        )


# ==============================================================================
# FACTORY
# ==============================================================================

VALID_ARCHITECTURES = {'encoder-only', 'encoder-decoder'}
VALID_STRATEGIES = {'shared', 'multiple'}
VALID_HEAD_TYPES = {'linear', 'mlp'}


def create_output_head(
        architecture: Literal["encoder-only", "encoder-decoder"],
        strategy: Literal["shared", "multiple"],
        input_dim: int,
        forecast_steps: int,
        num_features: int,
        head_type: Literal["linear", "mlp"] = "linear",
        dropout: float = 0.0
) -> TimeSeriesOutputHead:
    """
    Factory for creating output head strategies.

    Args:
        architecture: Model architecture type
        strategy: Projection strategy (shared captures cross-channel correlations)
        input_dim: Input dimension to the head
        forecast_steps: Number of forecast timesteps
        num_features: Number of output features
        head_type: Projection type ("linear" recommended for speed)
        dropout: Dropout rate for MLP heads

    Returns:
        Configured output head strategy

    Raises:
        ValueError: If parameters are invalid or incompatible

    Examples:
        >>> # Encoder-only with shared linear head
        >>> head = create_output_head(
        ...     architecture='encoder-only',
        ...     strategy='shared',
        ...     input_dim=128,
        ...     forecast_steps=24,
        ...     num_features=7
        ... )

        >>> # Encoder-decoder with MLP head
        >>> head = create_output_head(
        ...     architecture='encoder-decoder',
        ...     strategy='shared',
        ...     input_dim=256,
        ...     forecast_steps=96,
        ...     num_features=1,
        ...     head_type='mlp',
        ...     dropout=0.1
        ... )
    """
    # Validate inputs
    if architecture not in VALID_ARCHITECTURES:
        raise ValueError(
            f"Invalid architecture '{architecture}'. "
            f"Valid options: {VALID_ARCHITECTURES}"
        )

    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy}'. "
            f"Valid options: {VALID_STRATEGIES}"
        )

    if head_type not in VALID_HEAD_TYPES:
        raise ValueError(
            f"Invalid head_type '{head_type}'. "
            f"Valid options: {VALID_HEAD_TYPES}"
        )

    # Warn if using independent heads (suboptimal)
    if strategy == "multiple":
        logger.warning(
            f"Using '{strategy}' (independent) heads. "
            f"Consider 'shared' for better cross-channel modeling and speed."
        )

    # Create head based on architecture and strategy
    if architecture == "encoder-only":
        if strategy == "shared":
            return EncoderOnlySharedHead(
                input_dim=input_dim,
                forecast_steps=forecast_steps,
                num_features=num_features,
                head_type=head_type,
                dropout=dropout,
            )
        else:  # multiple
            return EncoderOnlyIndependentHead(
                input_dim=input_dim,
                forecast_steps=forecast_steps,
                num_features=num_features
            )

    else:  # encoder-decoder
        if strategy == "shared":
            return DecoderSharedHead(
                input_dim=input_dim,
                num_features=num_features,
                head_type=head_type,
                dropout=dropout
            )
        else:  # multiple
            return DecoderIndependentHead(
                input_dim=input_dim,
                num_features=num_features
            )