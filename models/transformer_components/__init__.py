"""
Transformer components package.

This package provides modular components for Transformer-based time series forecasting:
- Attention capture utilities
- Reversible Instance Normalization (RevIN)
- Target initialization strategies for encoder-decoder architectures
- Custom attention modules (LocalAttention, GlobalSelfAttention)
- Custom encoder layer with Pre-LN and GELU support
- Positional encoding implementations
- Output head strategies
- HPO (Hyperparameter Optimization) utilities

All components are re-exported at package level for backward compatibility
with existing code that imports from models.transformer.
"""

# Attention capture components
from .attention_capture import (
    AttentionCaptureBuffer,
    CapturingMHA,
)

# Normalization
from .revin import RevIN

# Target initializers
from .tgt_initializers import (
    TgtInitializer,
    ZerosTgtInitializer,
    LastValueTgtInitializer,
    MeanTgtInitializer,
    MedianTgtInitializer,
    TrendTgtInitializer,
    SeasonalTgtInitializer,
    CopyHistoryTgtInitializer,
    build_tgt_train,
    create_tgt_initializer,
)

# Attention modules
from .attention_modules import (
    LocalAttention,
    GlobalSelfAttention,
)

# Custom encoder layer
from .custom_encoder_layer import CustomTransformerEncoderLayer

# Positional encoding
from .positional_encoding import (
    BasePositionalEncoding,
    PositionalEncodingConfig,
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    NoPositionalEncoding,
    create_positional_encoding,
)

# Output heads
from .output_heads import (
    TimeSeriesOutputHead,
    EncoderOnlySharedHead,
    EncoderOnlyIndependentHead,
    DecoderSharedHead,
    DecoderIndependentHead,
    create_output_head,
)

# HPO utilities
from .hpo import (
    SearchSpaceAnalyzer,
    SearchSpaceFilter,
    ParameterValidator,
    LearningRateCalculator,
    SmartPriorGenerator,
)

__all__ = [
    # Attention capture
    "AttentionCaptureBuffer",
    "CapturingMHA",
    # Normalization
    "RevIN",
    # Target initializers
    "TgtInitializer",
    "ZerosTgtInitializer",
    "LastValueTgtInitializer",
    "MeanTgtInitializer",
    "MedianTgtInitializer",
    "TrendTgtInitializer",
    "SeasonalTgtInitializer",
    "CopyHistoryTgtInitializer",
    "build_tgt_train",
    "create_tgt_initializer",
    # Attention modules
    "LocalAttention",
    "GlobalSelfAttention",
    # Custom encoder layer
    "CustomTransformerEncoderLayer",
    # Positional encoding
    "BasePositionalEncoding",
    "PositionalEncodingConfig",
    "SinusoidalPositionalEncoding",
    "LearnablePositionalEncoding",
    "NoPositionalEncoding",
    "create_positional_encoding",
    # Output heads
    "TimeSeriesOutputHead",
    "EncoderOnlySharedHead",
    "EncoderOnlyIndependentHead",
    "DecoderSharedHead",
    "DecoderIndependentHead",
    "create_output_head",
    # HPO utilities
    "SearchSpaceAnalyzer",
    "SearchSpaceFilter",
    "ParameterValidator",
    "LearningRateCalculator",
    "SmartPriorGenerator",
]
