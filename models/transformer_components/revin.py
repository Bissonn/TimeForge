"""
Reversible Instance Normalization (RevIN) for Transformer models.

This module provides RevIN, a technique for handling distribution shift
in time series forecasting by normalizing inputs and denormalizing outputs.

References: https://openreview.net/pdf?id=cGDAkQo1C0p
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN).
    A technique to handle distribution shift in time series forecasting.
    It normalizes the input to zero mean and unit variance, and denormalizes the output.

    References: https://openreview.net/pdf?id=cGDAkQo1C0p

    NOTE on State Management:
    This implementation uses buffers to store statistics. In 'encoder-decoder' architectures,
    ensure 'mode="norm"' is called (encode) before 'mode="apply"' (decode) to establish
    the correct context statistics.
    Args:
            num_features: number of features or channels
            eps: a value added for numerical stability
            affine: if True, RevIN has learnable affine parameters
            robust: if True, uses Median/MAD (R^2-IN) instead of Mean/Std.
                    Optimized to avoid checks during the forward pass.
                    References: https://arxiv.org/abs/2510.04667
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, robust: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

        # Use buffers for stats to handle device placement automatically
        # Buffers for statistics
        # NOTE: In robust mode, 'mean' stores the Median, and 'stdev' stores the MAD.
        # We keep the names consistent to avoid complicating normalize/denormalize logic.
        self.register_buffer("mean", torch.zeros(1, 1, num_features), persistent=False)
        self.register_buffer("stdev", torch.ones(1, 1, num_features), persistent=False)

        # OPTIMIZATION: Assign the appropriate method at initialization.
        # This removes the need for an 'if self.robust' check in every forward pass.
        if robust:
            self._get_statistics = self._get_statistics_robust
        else:
            self._get_statistics = self._get_statistics_standard

    def forward(self, x: torch.Tensor, mode: str):
        if mode == 'norm':
            self._get_statistics(x) # Calls the method assigned in __init__
            x = self._normalize(x)
        elif mode == 'apply':
            # Apply previously calculated statistics (from 'norm' pass)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return x

    def _get_statistics_standard(self, x):
        """Original RevIN logic: Mean and Standard Deviation"""
        mean = torch.mean(x, dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        self._update_buffers(mean, stdev)

    def _get_statistics_robust(self, x):
        """R^2-IN logic: Median and Median Absolute Deviation (MAD)"""
        # Median
        median = torch.median(x, dim=1, keepdim=True).values.detach()

        # MAD
        abs_deviation = torch.abs(x - median)
        mad = torch.median(abs_deviation, dim=1, keepdim=True).values.detach()
        mad = mad + self.eps  # Numerical stability

        self._update_buffers(median, mad)

    def _update_buffers(self, center, scale):
        """Helper to update buffers handling resize logic"""
        if self.mean.shape != center.shape:
            self.mean.resize_as_(center)
        if self.stdev.shape != scale.shape:
            self.stdev.resize_as_(scale)

        self.mean.copy_(center)
        self.stdev.copy_(scale)

    def _normalize(self, x):
        x = x - self.mean
        # Clamp stdev to prevent division by very small values
        # Even though eps is added in sqrt(var + eps), extreme cases could still cause issues
        safe_stdev = torch.clamp(self.stdev, min=self.eps)
        x = x / safe_stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            # Clamp learned weight directly
            # Previous logic (affine_weight + eps) was incorrect - adding eps to a learned
            # parameter could still result in zero if weight is negative and close to -eps.
            # RevIN affine weights are initialized to 1.0 and should remain positive.
            safe_weight = torch.clamp(self.affine_weight, min=self.eps)
            x = x / safe_weight
        x = x * self.stdev + self.mean
        return x
