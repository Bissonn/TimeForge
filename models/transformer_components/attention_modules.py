"""
Custom attention modules for Transformer models.

This module provides specialized attention mechanisms:
- LocalAttention: Memory-efficient local causal attention using attention masks
- GlobalSelfAttention: Wrapper around nn.MultiheadAttention with SDPA backend support
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except ImportError:
    # Fallback for older PyTorch versions (<2.0)
    sdpa_kernel = None
    SDPBackend = None

from .attention_capture import AttentionCaptureBuffer

logger = logging.getLogger(__name__)


class LocalAttention(nn.Module):
    """
    Memory-efficient Local Causal Attention using attention mask.

    Instead of unfold() which duplicates memory O(L*W), uses band mask O(L^2).
    For typical time series (L=96, W=32): saves ~94x memory!

    If window_size >= seq_len, degrades gracefully to standard full causal attention.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        window_size: Local attention window size
        dropout: Dropout probability
    """

    def __init__(self, embed_dim: int, num_heads: int, window_size: int, dropout: float = 0.0):
        super().__init__()
        if window_size < 1:
            raise ValueError("window_size must be >= 1.")

        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = True
        # capture wiring (optional)
        self.capture: Optional[AttentionCaptureBuffer] = None
        self.key_prefix: str = ""

        # Standard PyTorch MHA
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cache for attention mask
        self._cached_mask: Optional[torch.Tensor] = None
        self._cached_mask_size: Optional[int] = None

    def _create_local_causal_mask(
            self,
            seq_len: int,
            device: torch.device,
            dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Create local causal attention mask: [max(0, i-W+1), i].
        Returns additive mask: 0.0 = allow, -1e9 = block.
        """
        # Create position indices
        i = torch.arange(seq_len, device=device).unsqueeze(1)  # (L, 1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, L)

        # Causal constraint: j > i
        causal_mask = j > i

        # Local constraint: j < i - W + 1
        local_mask = j < (i - self.window_size + 1)

        # Combine masks
        combined_mask = causal_mask | local_mask

        # Convert to additive mask
        attn_mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        attn_mask.masked_fill_(combined_mask, -1e9)

        return attn_mask

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, seq_len, embed_dim = x.shape

        # --- Mask Caching Logic ---
        # Regenerate if: cache empty OR size mismatch
        if self._cached_mask is None or self._cached_mask_size != seq_len:
            self._cached_mask = self._create_local_causal_mask(seq_len, x.device, x.dtype)
            self._cached_mask_size = seq_len
        else:
            # Move/Cast if device/dtype mismatch (e.g. mixed precision training)
            if self._cached_mask.device != x.device or self._cached_mask.dtype != x.dtype:
                self._cached_mask = self._cached_mask.to(device=x.device, dtype=x.dtype)

        local_mask = self._cached_mask

        # Combine with external mask if provided
        if attn_mask is not None:
            local_mask = local_mask + attn_mask

        # Forward pass
        need = bool(self.capture and self.capture.enabled)
        attn_output, attn_weights = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=local_mask,
            need_weights=need,
            average_attn_weights = False if need else True,
        )

        if need and self.capture:
            self.capture.store(self.key_prefix, attn_weights)

        return attn_output, attn_weights


class GlobalSelfAttention(nn.Module):
    """
    Wrapper around nn.MultiheadAttention providing a unified interface:
    (x, attn_mask, key_padding_mask) -> (out, weights).

    This ensures compatibility with CustomTransformerEncoderLayer, which expects
    a single input tensor 'x' (self-attention), whereas nn.MultiheadAttention
    expects (query, key, value).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.batch_first = True

        # capture wiring (optional)
        self.capture: Optional[AttentionCaptureBuffer] = None
        self.key_prefix: str = ""

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq, embed)
            attn_mask: Additive mask (seq, seq)
            key_padding_mask: Bool mask (batch, seq)

        Note:
            PyTorch automatically selects the best SDPA backend (Flash/Efficient/Math)
            based on hardware, dtype, and mask configuration. No manual dispatch needed.
        """
        need = bool(self.capture and self.capture.enabled)

        # PyTorch MultiheadAttention automatically uses SDPA when need_weights=False
        # and selects the optimal backend (Flash > Efficient > Math) based on:
        # - GPU compute capability (SM80+ for Flash)
        # - Input dtype (fp16/bf16 for Flash, fp32 for Math)
        # - Mask compatibility (Flash works with both attn_mask and is_causal)
        out, weights = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need,
            average_attn_weights=True if not need else False,
        )

        if need and self.capture:
            self.capture.store(self.key_prefix, weights)

        return out, weights
