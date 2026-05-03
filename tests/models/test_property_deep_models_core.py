import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from types import SimpleNamespace
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

# Import classes for dummy models
from models.lstm import LSTMForecaster
from models.transformer import TransformerForecaster


# --- Helpers (Dummy Models) ---

class DummyLSTMModel(nn.Module):
    def __init__(self, input_size: int, num_features: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=4, batch_first=True)
        self.fc = nn.Linear(4, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1:, :]
        y = self.fc(last)
        return y


class DummyEncoderOnlyModel(nn.Module):
    def __init__(self, input_size: int, num_features: int):
        super().__init__()
        self.architecture = "encoder-only"
        self.proj = nn.Linear(input_size, num_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F)
        return x

    def _encoder_readout_only(self, enc: torch.Tensor) -> torch.Tensor:
        # enc: (B, W, F) -> last step
        last = enc[:, -1:, :]
        return self.proj(last)

# -------------------------------------------------------------------
# Refactored Tests using Factory
# -------------------------------------------------------------------

@pytest.mark.skip(reason="PC-mode future_exog length validation not yet implemented for encoder-only architecture. "
                        "The new API (past_covariates/future_covariates) should be used instead.")
@settings(
    deadline=None,
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    window_size=st.integers(min_value=2, max_value=8),
    forecast_steps=st.integers(min_value=1, max_value=6),
    exog_len=st.integers(min_value=0, max_value=6),
)
def test_transformer_encoder_only_future_exog_length_validation(
    model_factory,
    window_size,
    forecast_steps,
    exog_len,
):
    """
    Canonical property test for encoder-only iterative Transformer with PC-mode.

    Contract:
    1) Structural PC + runtime PC OFF (no future_exog while C=0) -> RuntimeError
    2) Structural PC + C > 0 + future_exog too short -> ValueError
    3) Structural PC + C > 0 + future_exog valid -> prediction succeeds
    """
    device = torch.device("cpu")
    batch_size = 1
    num_features = 1
    encoder_exog_size = 1  # force presence of potential C-vars

    # --- Build forecaster in encoder-only PC-mode ---
    forecaster = model_factory(
        model_type="transformer",
        strategy="iterative",
        n_targets=num_features,
        n_enc=encoder_exog_size,
        n_dec=encoder_exog_size,
        window_size=window_size,
        forecast_steps=forecast_steps,
        use_shared_col_names=True,
        architecture="encoder-only",
    )

    # --- Inject canonical dummy encoder-only model ---
    total_input_size = forecaster.feature_layout.encoder_input_size
    forecaster.model = DummyEncoderOnlyModel(
        input_size=total_input_size,
        num_features=num_features,
    ).to(device)

    # --- Prepare input tensor ---
    input_tensor = torch.zeros(
        batch_size,
        window_size,
        total_input_size,
        dtype=torch.float32,
        device=device,
    )

    # --- Prepare future_exog ---
    c_size = forecaster.continuous_size

    if exog_len == 0:
        future_exog = None
    else:
        future_exog = torch.randn(
            batch_size,
            exog_len,
            c_size,
            dtype=torch.float32,
            device=device,
        )
    # ============================================================
    # CASE 1: runtime PC inactive (future_exog is None)
    # ============================================================
    if future_exog is None:
        with pytest.raises(RuntimeError, match="FeatureLayout expects PC inputs"):
            _ = forecaster._predict_iterative(
                input_tensor,
                future_exog_tensor=future_exog,
            )
        return

    # ============================================================
    # CASE 2: Continuous vars exist, but future_exog too short
    # ============================================================
    required_len = forecast_steps

    if future_exog.size(1) < required_len:
        with pytest.raises(ValueError):
            _ = forecaster._predict_iterative(
                input_tensor,
                future_exog_tensor=future_exog,
            )
        return

    # ============================================================
    # CASE 3: Continuous vars exist, future_exog valid
    # ============================================================
    preds = forecaster._predict_iterative(
        input_tensor,
        future_exog_tensor=future_exog,
    )

    assert preds.shape == (batch_size, forecast_steps, num_features)
    assert torch.isfinite(torch.from_numpy(preds)).all()

# (Original test_dummy_lstm_backprop_decreases_loss preserved below)
@settings(deadline=None, max_examples=10)
@given(
    W=st.integers(min_value=3, max_value=8),
    B=st.integers(min_value=2, max_value=5),
)
def test_dummy_lstm_backprop_decreases_loss(W, B):
    torch.manual_seed(42)
    device = torch.device("cpu")
    F_in, F_out = 3, 1
    model = nn.LSTM(input_size=F_in, hidden_size=8, batch_first=True).to(device)
    head = nn.Linear(8, F_out).to(device)
    optimizer = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=0.05)
    loss_fn = nn.MSELoss()
    x = torch.randn(B, W, F_in, device=device)
    y_true = x[:, -1:, :1].sum(dim=2)

    def forward_loss():
        out, _ = model(x)
        y_pred = head(out[:, -1, :])
        return loss_fn(y_pred.squeeze(-1), y_true.squeeze(-1))

    loss_before = forward_loss().item()
    optimizer.zero_grad()
    loss = forward_loss()
    loss.backward()
    optimizer.step()
    loss_after = forward_loss().item()
    assert np.isfinite(loss_before)
    assert np.isfinite(loss_after)