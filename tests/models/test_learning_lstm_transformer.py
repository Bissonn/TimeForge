# tests/models/test_learning_lstm_transformer.py
import numpy as np
import torch
import torch.nn as nn

from models.lstm import LSTMModel
from models.transformer import TransformerModel


def _generate_ar1_series(T: int, phi: float = 0.8, noise_std: float = 0.05) -> np.ndarray:
    """
    Generate a simple AR(1) series:
        y_t = phi * y_{t-1} + eps_t

    Returns:
        y: np.ndarray of shape (T,)
    """
    rng = np.random.default_rng(0)
    y = np.zeros(T, dtype=np.float32)
    noise = rng.normal(0.0, noise_std, size=T).astype(np.float32)

    for t in range(1, T):
        y[t] = phi * y[t - 1] + noise[t]

    return y


def test_lstm_learns_ar1_one_step():
    """
    Learning test: LSTMModel should be able to learn a simple AR(1) one-step-ahead mapping.

    Setup:
      - Generate AR(1) series of length T.
      - Build dataset of sliding windows (X) and next-step targets (Y).
      - Train a small LSTMModel for a few epochs with MSE loss.
      - Assert that the final loss is significantly lower than the initial loss.

    This checks:
      - backpropagation works end-to-end,
      - gradients flow through all parameters,
      - the model actually *learns* something non-trivial on a simple synthetic task.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    T = 80
    window_size = 8
    input_dim = 1     # univariate series
    output_steps = 1  # one-step ahead

    # 1. Generate AR(1) series
    y = _generate_ar1_series(T=T, phi=0.8, noise_std=0.05)  # shape (T,)

    # 2. Build supervised dataset: windows -> next value
    #    X: (N, W, 1), Y: (N, 1, 1)
    N = T - window_size
    X_np = np.zeros((N, window_size, input_dim), dtype=np.float32)
    Y_np = np.zeros((N, output_steps, input_dim), dtype=np.float32)

    for i in range(N):
        X_np[i, :, 0] = y[i : i + window_size]
        Y_np[i, 0, 0] = y[i + window_size]

    X = torch.from_numpy(X_np)
    Y = torch.from_numpy(Y_np)

    # 3. Define a small LSTMModel
    model = LSTMModel(
        input_size=input_dim,
        hidden_size=16,
        num_layers=1,
        output_steps=output_steps,
        output_features=input_dim,
        dropout=0.0
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()

    # 4. Measure initial loss
    model.train()
    with torch.no_grad():
        initial_out = model(X)
        initial_loss = loss_fn(initial_out, Y).item()

    # 5. Train for a few epochs (full-batch, to keep it simple and deterministic)
    n_epochs = 40
    for _ in range(n_epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()

    # 6. Final loss
    model.eval()
    with torch.no_grad():
        final_out = model(X)
        final_loss = loss_fn(final_out, Y).item()

    # 7. Assertions:
    #    - final_loss is finite
    #    - final_loss is significantly smaller than initial_loss
    assert np.isfinite(final_loss), "LSTM final loss is not finite."
    assert final_loss < initial_loss, (
        f"LSTM did not improve the loss: initial={initial_loss:.6f}, final={final_loss:.6f}"
    )
    # A bit stronger condition (but still robust for this simple AR(1)):
    assert final_loss < 0.5 * initial_loss, (
        f"LSTM did not learn enough: initial={initial_loss:.6f}, final={final_loss:.6f}"
    )


def test_transformer_learns_identity_mapping_encoder_decoder():
    """
    Learning test: encoder–decoder TransformerModel should learn a simple
    "copy-the-input" mapping on the last timestep.

    Task:
      - src: random sequence of shape (B, W, F)
      - tgt: last encoder timestep (broadcasted over horizon H):
             y_t = src[:, -1, :]
      - H can be >= 1; model predicts H steps, all should match last src step.

    We train:
      - small encoder-decoder Transformer,
      - on a synthetic dataset of independent samples,
      - for a few epochs with MSE loss.

    This checks:
      - gradient flow through multi-head attention and FFN,
      - correct handling of readout in encoder-decoder mode,
      - that the network can learn a simple, low-complexity mapping.
    """
    torch.manual_seed(1)
    np.random.seed(1)

    B = 32           # number of training samples
    W = 6            # window size
    F = 2            # feature dimension
    H = 3            # forecast steps

    # 1. Synthetic dataset:
    #    src: (B, W, F), tgt_y: (B, H, F) – copies last encoder step
    src = torch.randn(B, W, F)
    last_step = src[:, -1:, :]                       # (B, 1, F)
    tgt_y = last_step.repeat(1, H, 1)                # (B, H, F)

    # 2. Small encoder–decoder Transformer
    model = TransformerModel(
        encoder_input_size=F,
        decoder_input_size=F,        # we will feed true y as decoder input during training
        num_features=F,
        forecast_steps=H,
        window_size=W,
        hidden_size=16,
        num_heads=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_ff_multiplier=2.0,
        dropout=0.0,
        attention_type="full",
        readout="last",
        architecture="encoder-decoder",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # 3. Decoder input during training: teacher forcing with true y
    #    To keep things simple, we pass tgt_y as decoder input.
    tgt_in = tgt_y.clone().detach()

    # 4. Initial loss
    model.train()
    with torch.no_grad():
        initial_out = model(src, tgt_in)
        initial_loss = loss_fn(initial_out, tgt_y).item()

    # 5. Train for several epochs
    n_epochs = 40
    for _ in range(n_epochs):
        optimizer.zero_grad()
        out = model(src, tgt_in)
        loss = loss_fn(out, tgt_y)
        loss.backward()
        optimizer.step()

    # 6. Final loss
    model.eval()
    with torch.no_grad():
        final_out = model(src, tgt_in)
        final_loss = loss_fn(final_out, tgt_y).item()

    # 7. Assertions
    assert np.isfinite(final_loss), "Transformer final loss is not finite."
    assert final_loss < initial_loss, (
        f"Transformer did not improve the loss: initial={initial_loss:.6f}, final={final_loss:.6f}"
    )
    # Identity copy is an easy task; we expect a strong reduction
    assert final_loss < 0.3 * initial_loss, (
        f"Transformer did not learn enough: initial={initial_loss:.6f}, final={final_loss:.6f}"
    )
