from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch import nn


class BasePositionalEncoding(nn.Module):
    """
    Base class for positional encodings.

    Expected input shape: (batch_size, seq_len, hidden_size)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface only
        raise NotImplementedError("Subclasses must implement forward().")


@dataclass
class PositionalEncodingConfig:
    """Configuration object for positional encodings."""
    type: str = "sinusoidal"  # "sinusoidal", "learnable", "none"
    max_len: int = 5000
    pe_dropout: float = 0.0
    scale_with_sqrt_hidden_size: bool = True


class SinusoidalPositionalEncoding(BasePositionalEncoding):
    """
    Classic sinusoidal positional encoding from 'Attention Is All You Need'.

    Input:  (batch, seq_len, hidden_size)
    Output: (batch, seq_len, hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        max_len: int = 5000,
        pe_dropout: float = 0.0,
        scale_with_sqrt_hidden_size: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.max_len = max_len
        self.scale = math.sqrt(hidden_size) if scale_with_sqrt_hidden_size else None
        self.pe_dropout = nn.Dropout(pe_dropout) if pe_dropout > 0.0 else nn.Identity()

        # We start with an initial buffer; we will extend it on the fly if needed.
        pe = self._build_pe(max_len, hidden_size)
        # `persistent=False` so the buffer is not stored in checkpoints as a huge tensor.
        self.register_buffer("pe", pe, persistent=False)

        # Cache for type-casted PE (AMP optimization - avoids float32->float16 conversion every forward)
        self._pe_cache = None

    @staticmethod
    def _build_pe(max_len: int, hidden_size: int) -> torch.Tensor:
        """
        Build sinusoidal positional encodings of shape (1, max_len, hidden_size).
        """
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float) * (-math.log(10000.0) / hidden_size)
        )  # (hidden_size//2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, hidden_size)
        return pe

    def _ensure_length(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        Ensure that the internal buffer `pe` is at least `seq_len` long.
        If not, rebuild it with a larger max_len.
        """
        if seq_len <= self.pe.size(1):
            return

        new_max_len = max(seq_len, int(self.pe.size(1) * 1.5))
        pe = self._build_pe(new_max_len, self.hidden_size).to(device=device, dtype=dtype)
        self.pe = pe  # type: ignore[assignment]
        self.max_len = new_max_len

        # Invalidate cache when PE buffer is rebuilt
        self._pe_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_size)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (batch, seq_len, hidden_size), got {x.shape}")

        batch_size, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Input hidden_size={hidden_size} does not match configured hidden_size={self.hidden_size}."
            )

        # Make sure buffer is on the correct device/dtype and has enough length
        self._ensure_length(seq_len, device=x.device, dtype=x.dtype)

        # OPTIMIZATION: Cache type-casted PE to avoid memory allocation every forward pass
        # This is critical for AMP (float16) where .to() would allocate new tensor each time
        cache_invalid = (
            self._pe_cache is None or
            self._pe_cache.dtype != x.dtype or
            self._pe_cache.device != x.device
        )

        if cache_invalid:
            # Only convert dtype/device when context changes (e.g., first forward or AMP toggle)
            self._pe_cache = self.pe.to(device=x.device, dtype=x.dtype)

        # Zero-copy slice from cached PE
        pe = self._pe_cache[:, :seq_len, :]  # (1, seq_len, hidden_size)

        if self.scale is not None:
            x = x * self.scale

        x = x + pe
        x = self.pe_dropout(x)
        return x


class LearnablePositionalEncoding(BasePositionalEncoding):
    """
    Learnable positional encoding using nn.Embedding.

    Positions: [0, 1, ..., seq_len-1]
    """

    def __init__(
        self,
        hidden_size: int,
        max_len: int = 5000,
        pe_dropout: float = 0.0,
        scale_with_sqrt_hidden_size: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.max_len = max_len
        self.scale = math.sqrt(hidden_size) if scale_with_sqrt_hidden_size else None
        self.pe_dropout = nn.Dropout(pe_dropout) if pe_dropout > 0.0 else nn.Identity()

        self.embedding = nn.Embedding(max_len, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_size)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (batch, seq_len, hidden_size), got {x.shape}")

        batch_size, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Input hidden_size={hidden_size} does not match configured hidden_size={self.hidden_size}."
            )
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len={self.max_len} for learnable encoding."
            )

        device = x.device
        positions = torch.arange(seq_len, device=device, dtype=torch.long)  # (seq_len,)
        pos_emb = self.embedding(positions)  # (seq_len, hidden_size)
        pos_emb = pos_emb.unsqueeze(0)  # (1, seq_len, hidden_size)

        if self.scale is not None:
            x = x * self.scale

        x = x + pos_emb
        x = self.pe_dropout(x)
        return x


class NoPositionalEncoding(BasePositionalEncoding):
    """
    Identity positional encoding: returns input unchanged.
    Useful for ablation studies or when positional info is injected elsewhere.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def create_positional_encoding(
    hidden_size: int,
    config: Optional[PositionalEncodingConfig] = None,
    **overrides: Any,
) -> BasePositionalEncoding:
    """
    Factory function to create positional encoding modules.

    Parameters
    ----------
    hidden_size : int
        Model dimensionality.
    config : PositionalEncodingConfig, optional
        Base configuration.
    overrides : Any
        Keyword arguments to override fields in `config`.
        Example: create_positional_encoding(hidden_size, type="learnable", max_len=1024)

    Returns
    -------
    BasePositionalEncoding
        A positional encoding instance.
    """
    if config is None:
        config = PositionalEncodingConfig()

    # Convert dataclass to a mutable dict and apply overrides
    cfg_dict: Dict[str, Any] = {
        "type": config.type,
        "max_len": config.max_len,
        "pe_dropout": config.pe_dropout,
        "scale_with_sqrt_hidden_size": config.scale_with_sqrt_hidden_size,
    }
    cfg_dict.update(overrides)

    pe_type = cfg_dict.pop("type", "sinusoidal").lower()

    if pe_type == "sinusoidal":
        return SinusoidalPositionalEncoding(hidden_size=hidden_size, **cfg_dict)
    elif pe_type == "learnable":
        return LearnablePositionalEncoding(hidden_size=hidden_size, **cfg_dict)
    elif pe_type in ("none", "identity"):
        return NoPositionalEncoding()
    else:
        raise ValueError(f"Unknown positional encoding type: {pe_type}")
