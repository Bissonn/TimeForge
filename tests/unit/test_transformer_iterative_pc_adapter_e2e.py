import torch
import numpy as np
import pandas as pd

from models.transformer import TransformerForecaster
from utils.dataset import TimeSeriesDataset


def test_transformer_iterative_pc_adapter_produces_finite_predictions(base_context):
    """
    E2E contract test:
    Iterative Transformer with PC-mode must produce
    finite, non-NaN predictions when P + C variables are present.
    """

    torch.manual_seed(0)
    np.random.seed(0)

    # -------------------------------------------------
    # Config
    # -------------------------------------------------
    W = 24
    H = 12

    target_cols = ["y"]
    enc_cols = ["p1", "c1"]     # P + C
    dec_cols = ["c1"]           # C only (shared)

    all_cols = target_cols + enc_cols

    # -------------------------------------------------
    # Fake data
    # -------------------------------------------------
    total_len = W + 2 * H + 10
    df = pd.DataFrame(
        np.random.randn(total_len, len(all_cols)),
        columns=all_cols,
        index=pd.date_range("2023-01-01", periods=total_len, freq="D")
    )

    # New API: past_covariates (encoder-only), future_covariates (shared)
    past_cols = ["p1"]     # P only (encoder-only)
    future_cols = ["c1"]   # C only (shared, both encoder and decoder)

    dataset = TimeSeriesDataset(
        "test_transformer_pc",
        {},
        num_features=1,
        data=df,
        columns=target_cols,
        past_covariates=past_cols,
        future_covariates=future_cols,
        freq="D",
    )

    train_len = W + H + 5
    train_df = df.iloc[:train_len]
    future_df = df.iloc[train_len: train_len + H][future_cols]

    # -------------------------------------------------
    # Model - Using encoder-decoder for covariate support
    # -------------------------------------------------
    model = TransformerForecaster(
        model_params={
            "architecture": "encoder-decoder",  # Required for covariates in iterative mode
            "d_model": 16,
            "n_heads": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dropout": 0.0,
            "strategy": "iterative",
            "epochs": 1,
            "batch_size": 2,
        },
        num_features=1,
        forecast_steps=H,
        window_size=W,
        dataset=dataset,
        run_context=base_context,
    )

    loss, _ = model.fit(train_df, dataset=dataset, is_final_fit=True)
    assert np.isfinite(loss)

    # -------------------------------------------------
    # Predict
    # -------------------------------------------------
    input_df = train_df.iloc[-W:]
    preds = model.predict(input_df, future_exog=future_df)

    # -------------------------------------------------
    # CONTRACT ASSERTIONS
    # -------------------------------------------------
    assert preds.shape == (H, 1)
    assert not preds.isna().any().any()
    assert np.isfinite(preds.values).all()
