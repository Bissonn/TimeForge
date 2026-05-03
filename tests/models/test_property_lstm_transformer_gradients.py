# tests/models/test_property_lstm_transformer_gradients.py

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st

from models.lstm import LSTMModel
from models.transformer import TransformerModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def all_grads_finite(params: List[nn.Parameter]) -> bool:
    """
    Utility: check that every parameter has a gradient and that it is finite.
    """
    for p in params:
        if not p.requires_grad:
            # Ignore frozen parameters
            continue
        if p.grad is None:
            return False
        if not torch.isfinite(p.grad).all():
            return False
    return True


# ---------------------------------------------------------------------------
# 1. LSTMModel – gradient connectivity and stability
# ---------------------------------------------------------------------------

@settings(max_examples=50, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    window_size=st.integers(min_value=2, max_value=8),
    input_size=st.integers(min_value=1, max_value=4),
    output_steps=st.integers(min_value=1, max_value=4),
)
def test_lstm_model_backward_is_stable(batch_size, window_size, input_size, output_steps):
    """
    Property: For random (batch_size, window_size, input_size) and output_steps,
    the LSTMModel forward + backward pass should:

      - run without errors,
      - produce finite gradients for all trainable parameters.

    This checks that the architecture has no shape bugs or broken graph.
    """
    torch.manual_seed(42)

    # Small hidden size and 1 layer to keep the test fast
    model = LSTMModel(
        input_size=input_size,
        hidden_size=8,
        num_layers=1,
        output_steps=output_steps,
        output_features=input_size,
        dropout=0.0  # Added required dropout argument
    )

    # Random input
    x = torch.randn(batch_size, window_size, input_size, requires_grad=True)

    # Forward
    y = model(x)  # expected shape: (B, output_steps, output_features)
    assert y.shape == (batch_size, output_steps, input_size)

    # Scalar loss
    loss = (y ** 2).mean()
    loss.backward()

    # Check gradients
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    params = [p for p in model.parameters()]
    assert all_grads_finite(params), "Some LSTM parameters have missing or non-finite gradients."


# ---------------------------------------------------------------------------
# 2. TransformerModel – encoder-only architecture
# ---------------------------------------------------------------------------

@settings(max_examples=40, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=3),
    window_size=st.integers(min_value=2, max_value=6),
    num_features=st.integers(min_value=1, max_value=3),
    forecast_steps=st.integers(min_value=1, max_value=4),
)
def test_transformer_encoder_only_backward_is_stable(
        batch_size,
        window_size,
        num_features,
        forecast_steps,
):
    """
    Property: Encoder-only TransformerModel should support a stable backward pass
    for a variety of shapes (B, W, F, H):

      - forward(src) runs without errors,
      - produces output of shape (B, H, F),
      - all trainable parameters get finite gradients.
    """
    torch.manual_seed(123)

    encoder_input_size = num_features  # no encoder exog in this test

    model = TransformerModel(
        encoder_input_size=encoder_input_size,
        decoder_input_size=num_features,  # unused in encoder-only mode
        num_features=num_features,
        forecast_steps=forecast_steps,
        window_size=window_size,
        hidden_size=16,
        num_heads=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_ff_multiplier=2.0,
        dropout=0.0,
        architecture="encoder-only",
        readout="last",
        attention_type="full",
        attention_window_size=window_size,
        # Removed deprecated arguments: num_exog_encoder, num_exog_decoder, decoder_uses_exog
    )

    # src: (B, W, F)
    src = torch.randn(batch_size, window_size, num_features, requires_grad=True)

    # Forward (encoder-only ignores tgt/future_exog)
    y = model(src)

    assert y.shape == (batch_size, forecast_steps, num_features)

    loss = (y ** 2).mean()
    loss.backward()

    # Gradients for input
    assert src.grad is not None
    assert torch.isfinite(src.grad).all()

    params = [p for p in model.parameters()]
    assert all_grads_finite(params), "Some encoder-only Transformer parameters have bad gradients."


# ---------------------------------------------------------------------------
# 3. TransformerModel – encoder-decoder architecture (without exog)
# ---------------------------------------------------------------------------

@settings(max_examples=40, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=3),
    window_size=st.integers(min_value=2, max_value=6),
    num_features=st.integers(min_value=1, max_value=3),
    forecast_steps=st.integers(min_value=1, max_value=4),
)
def test_transformer_encoder_decoder_backward_is_stable_no_exog(
        batch_size,
        window_size,
        num_features,
        forecast_steps,
):
    """
    Property: Encoder-decoder TransformerModel (without future exog) should
    support a stable backward pass for random shapes.

    We test:
      - forward(src, tgt) is well-defined,
      - output shape = (B, H, F),
      - gradients on all parameters are finite.
    """
    torch.manual_seed(321)

    encoder_input_size = num_features
    decoder_input_size = num_features

    model = TransformerModel(
        encoder_input_size=encoder_input_size,
        decoder_input_size=decoder_input_size,
        num_features=num_features,
        forecast_steps=forecast_steps,
        window_size=window_size,
        hidden_size=16,
        num_heads=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_ff_multiplier=2.0,
        dropout=0.0,
        architecture="encoder-decoder",
        readout="last",
        attention_type="full",
        attention_window_size=window_size,
        # Removed deprecated arguments
    )

    # src: (B, W, F), tgt: (B, H, F)
    src = torch.randn(batch_size, window_size, num_features, requires_grad=True)
    tgt = torch.randn(batch_size, forecast_steps, num_features, requires_grad=True)

    y = model(src, tgt=tgt)

    assert y.shape == (batch_size, forecast_steps, num_features)

    # Use both src and tgt in the loss so their gradients matter
    loss = ((y - tgt) ** 2).mean()
    loss.backward()

    # Gradients for src and tgt
    assert src.grad is not None
    assert torch.isfinite(src.grad).all()
    assert tgt.grad is not None
    assert torch.isfinite(tgt.grad).all()

    params = [p for p in model.parameters()]
    assert all_grads_finite(params), "Some encoder-decoder Transformer parameters have bad gradients."


# ---------------------------------------------------------------------------
# 4. TransformerModel – encoder-decoder with decoder exog
# ---------------------------------------------------------------------------

@settings(max_examples=25, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    window_size=st.integers(min_value=2, max_value=5),
    num_features=st.integers(min_value=1, max_value=2),
    num_exog_decoder=st.integers(min_value=1, max_value=2),
    forecast_steps=st.integers(min_value=1, max_value=4),
)
def test_transformer_encoder_decoder_backward_with_decoder_exog(
        batch_size,
        window_size,
        num_features,
        num_exog_decoder,
        forecast_steps,
):
    """
    Property: encoder-decoder Transformer with decoder exogenous inputs should
    still have a stable backward pass.

    We construct:
      - src: (B, W, F)
      - tgt: (B, H, F + E_dec)  -> we concatenate random exog to tgt

    and run forward + backward, checking that gradients flow to all parameters.
    """
    torch.manual_seed(999)

    encoder_input_size = num_features
    decoder_input_size = num_features + num_exog_decoder

    model = TransformerModel(
        encoder_input_size=encoder_input_size,
        decoder_input_size=decoder_input_size,
        num_features=num_features,
        forecast_steps=forecast_steps,
        window_size=window_size,
        hidden_size=16,
        num_heads=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_ff_multiplier=2.0,
        dropout=0.0,
        architecture="encoder-decoder",
        readout="last",
        attention_type="full",
        attention_window_size=window_size,
        # Removed deprecated arguments
    )

    src = torch.randn(batch_size, window_size, num_features, requires_grad=True)

    # Decoder exog and tgt
    # In this test we simulate the 'tgt' passed to model.forward() which already
    # contains concatenated [targets, exog].
    future_exog = torch.randn(batch_size, forecast_steps, num_exog_decoder, requires_grad=True)
    base_tgt = torch.randn(batch_size, forecast_steps, num_features, requires_grad=True)

    # The 'tgt' passed to the model forward()
    tgt_input = torch.cat([base_tgt, future_exog], dim=-1)  # (B, H, F + E_dec)

    y = model(src, tgt=tgt_input)

    assert y.shape == (batch_size, forecast_steps, num_features)

    # Loss uses both y and base_tgt
    loss = ((y - base_tgt) ** 2).mean()
    loss.backward()

    # Check grads on inputs
    assert src.grad is not None
    assert torch.isfinite(src.grad).all()
    assert base_tgt.grad is not None
    assert torch.isfinite(base_tgt.grad).all()
    assert future_exog.grad is not None
    assert torch.isfinite(future_exog.grad).all()

    params = [p for p in model.parameters()]
    assert all_grads_finite(params), "Some Transformer parameters have bad gradients in exog decoder mode."
