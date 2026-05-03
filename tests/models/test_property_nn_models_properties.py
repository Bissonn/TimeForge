# tests/models/test_property_nn_models_properties.py
import numpy as np
import torch
import torch.nn as nn

from hypothesis import given, settings, strategies as st

from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.transformer_components.tgt_initializers import (
    LastValueTgtInitializer,
    ZerosTgtInitializer,
    MeanTgtInitializer,
)


# ---------- Strategies ----------

def small_batch():
    return st.integers(min_value=1, max_value=4)


def small_window():
    return st.integers(min_value=2, max_value=8)


def small_forecast_steps():
    return st.integers(min_value=1, max_value=6)


def small_feature_dim():
    return st.integers(min_value=1, max_value=4)


# ---------- LSTM: backpropagation connectivity ----------

@settings(deadline=None, max_examples=40)
@given(
    B=small_batch(),
    W=small_window(),
    F=small_feature_dim(),
    H=small_forecast_steps(),
)
def test_lstm_model_backprop_connectivity(B, W, F, H):
    """
    Property:
      For random (B, W, F, H), the LSTMModel must support backprop:

        - forward pass produces shape (B, H, F_out),
        - loss.backward() populates all trainable parameter gradients (not None),
        - no runtime errors or NaNs are produced.

    This does not check learning quality, only that the computational
    graph is intact for a variety of shapes.
    """
    torch.manual_seed(0)

    model = LSTMModel(
        input_size=F,
        hidden_size=8,
        num_layers=1,
        output_steps=H,
        output_features=F,  # Fixed: num_features -> output_features
        dropout=0.0
    )

    x = torch.randn(B, W, F, requires_grad=True)
    y = torch.randn(B, H, F)

    out = model(x)
    assert out.shape == (B, H, F)

    loss = nn.MSELoss()(out, y)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), "Some LSTM parameters did not receive gradients."
    assert not any(torch.isnan(g).any() for g in grads), "NaN gradients detected in LSTM parameters."


# ---------- Transformer: backpropagation connectivity ----------

@settings(deadline=None, max_examples=40)
@given(
    B=small_batch(),
    W=small_window(),
    F=small_feature_dim(),
    H=small_forecast_steps(),
    readout=st.sampled_from(["last", "mean"]),
    attention_type=st.sampled_from(["full", "local"]), # Changed "causal" to "local" as per valid options
)
def test_transformer_model_backprop_connectivity(B, W, F, H, readout, attention_type):
    """
    Property:
      For random shapes and configuration:

        - TransformerModel forward works without error,
        - output shape is (B, H, F) in encoder-decoder mode,
        - loss.backward() populates gradients for all trainable parameters.

      We vary readout and attention_type to stress different code paths.
    """
    torch.manual_seed(0)

    # Minimal but valid configuration for encoder-decoder TransformerModel
    model = TransformerModel(
        encoder_input_size=F,
        decoder_input_size=F,
        num_features=F,
        forecast_steps=H,
        window_size=W,
        hidden_size=16,
        num_heads=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_ff_multiplier=2.0,
        dropout=0.0,
        attention_type=attention_type,
        attention_window_size=max(1, W), # Valid window size for local attention
        readout=readout,
        architecture="encoder-decoder",
    )

    src = torch.randn(B, W, F, requires_grad=True)
    tgt = torch.randn(B, H, F)

    out = model(src, tgt)
    assert out.shape == (B, H, F)

    loss = nn.MSELoss()(out, tgt)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), "Some Transformer parameters did not receive gradients."
    assert not any(torch.isnan(g).any() for g in grads), "NaN gradients detected in Transformer parameters."


# ---------- Transformer: readout semantics for single-step forecasts ----------

@settings(deadline=None, max_examples=30)
@given(
    B=small_batch(),
    W=small_window(),
    F=small_feature_dim(),
    readout=st.sampled_from(["last", "mean"]),
    attention_type=st.sampled_from(["full", "local"]), # Changed "causal" to "local"
)
def test_transformer_readout_semantics_single_step(B, W, F, readout, attention_type):
    """
    Property:
      When forecast_steps == 1, different readout strategies must be
      shape-compatible and produce finite outputs.

      For H=1 the semantics of "last" vs "mean" are degenerate but must
      not break the forward pass or change output shape.
    """
    H = 1
    torch.manual_seed(1)

    model = TransformerModel(
        encoder_input_size=F,
        decoder_input_size=F,
        num_features=F,
        forecast_steps=H,
        window_size=W,
        hidden_size=8,
        num_heads=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_ff_multiplier=2.0,
        dropout=0.0,
        attention_type=attention_type,
        attention_window_size=max(1, W),
        readout=readout,
        architecture="encoder-decoder",
    )

    src = torch.randn(B, W, F)
    tgt = torch.randn(B, H, F)

    out = model(src, tgt)
    assert out.shape == (B, H, F)
    assert torch.isfinite(out).all(), "Transformer readout produced non-finite values."


# ---------- Transformer tgt_init: initializer semantics with exog ----------

@settings(deadline=None, max_examples=40)
@given(
    B=small_batch(),
    W=small_window(),
    F=small_feature_dim(),
    H=small_forecast_steps(),
    E=st.integers(min_value=1, max_value=3),
    initializer_cls=st.sampled_from([LastValueTgtInitializer, ZerosTgtInitializer, MeanTgtInitializer]),
)
def test_tgt_initializer_direct_with_exog_semantics(B, W, F, H, E, initializer_cls):
    """
    Property:
      For any tgt initializer (last_value / zeros / mean):

        - initialize_direct returns a tensor of shape (B, H, F + E)
          when decoder_uses_exog == True,
        - the first feature block (targets) respects the initializer semantics
          (we do not assert exact values for mean here, just stability),
        - exogenous block is copied from future_exog_tensor one-to-one.

      This tests the coupling between TgtInitializer and future_exog_tensor.
    """
    torch.manual_seed(2)

    src = torch.randn(B, W, F + E)
    future_exog = torch.randn(B, H, E)

    # Fixed: Added num_exog_decoder argument
    initializer = initializer_cls(decoder_uses_exog=True, num_exog_decoder=E)

    tgt = initializer.initialize_direct(
        src=src,
        forecast_steps=H,
        num_features=F,
        device=torch.device("cpu"),
        future_exog_tensor=future_exog,
    )

    assert tgt.shape == (B, H, F + E)

    # Check that exogenous slice matches future_exog exactly
    exog_slice = tgt[:, :, F:]
    assert torch.allclose(exog_slice, future_exog), "Decoder exog has been mis-aligned in tgt initialization."

    # Basic sanity on target part: finite values, non-empty
    tgt_y = tgt[:, :, :F]
    assert tgt_y.shape == (B, H, F)
    assert torch.isfinite(tgt_y).all()