"""
Custom encoder layer implementation for Transformer models.

This module provides a custom TransformerEncoderLayer that supports:
- Pre-LN (NormFirst) and Post-LN architectures
- GELU and ReLU activations
- Integration with custom attention modules
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CustomTransformerEncoderLayer(nn.Module):
    """
    Custom Encoder Layer supporting Pre-LN (NormFirst) and GELU.
    """
    def __init__(
        self, d_model: int, dim_feedforward: int, dropout: float,
        attention_module: nn.Module, activation: str = "relu", norm_first: bool = False
    ):
        super().__init__()
        self.self_attn = attention_module
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            ** kwargs
    ) -> torch.Tensor:
        # Debug logging removed - was too verbose during training
        # (logged on every forward pass for every layer)
        if self.norm_first:
            # Pre-LN: Norm -> Attn -> Add -> Norm -> FFN -> Add
            src_norm = self.norm1(src)
            src2, _ = self.self_attn(src_norm, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)

            src_norm = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
            src = src + self.dropout2(src2)
        else:
            # Post-LN
            src2, _ = self.self_attn(src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src
