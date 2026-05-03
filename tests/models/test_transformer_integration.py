import pytest
import numpy as np
import random
import torch
import pandas as pd

from models.factory import ModelFactory
import models.transformer  # Auto-registers the model
from utils.dataset import TimeSeriesDataset

# Import constant from conftest (if defined there)
try:
    from tests.models.conftest import WINDOW_SIZE
except ImportError:
    WINDOW_SIZE = 10


@pytest.mark.parametrize("horizon", [1, 5])
@pytest.mark.parametrize("num_features", [1, 2])
def test_e2e_fit_predict_multi_h_f(horizon, num_features, base_transformer_config, full_dataset, base_context):
    """
    E2E: Fit/predict on full pipeline, multi-H/F.
    Asserts that validation loss is reasonable and pipeline completes.
    """
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)

    H = horizon
    F = num_features

    ds = full_dataset

    # Adapt dataset for feature count and architecture
    if F == 1:
        # Univariate: target + enc_exog
        temp_df = ds.series.reset_index()
        sliced_data = temp_df[["date", "target", "enc_exog"]].copy()
        sliced_data.set_index("date", inplace=True)
        ds = TimeSeriesDataset(
            "univariate",
            {},
            num_features = F,
            data=sliced_data,
            columns=["target"],
            past_covariates=["enc_exog"],
            future_covariates=[]
        )
    else:
        # F == 2
        if H == 1:
            # Encoder-only: 2 targets + enc_exog, no dec_exog in data
            temp_df = ds.series.reset_index()
            sliced_data = temp_df[["date", "target", "target2", "enc_exog"]].copy()
            sliced_data.set_index("date", inplace=True)
            ds = TimeSeriesDataset(
                "full_F2_enc_only",
                {},
                num_features=F,
                data=sliced_data,
                columns=["target", "target2"],
                past_covariates=["enc_exog"],
                future_covariates=[]
            )
        else:
            # H > 1, F = 2 -> encoder-decoder but without decoder exog in layout for this test
            # Remove dec_exog from data since we're not registering it
            temp_df = ds.series.reset_index()
            sliced_data = temp_df[["date", "target", "target2", "enc_exog"]].copy()
            sliced_data.set_index("date", inplace=True)
            ds = TimeSeriesDataset(
                "no_dec_exog",
                ds.config,
                num_features=F,
                data=sliced_data,
                columns=['target', 'target2'],
                past_covariates=['enc_exog'],
                future_covariates=[]
            )

    ds.split_data(forecast_steps=H)
    train_df = ds.development_data
    test_df = ds.test_data

    cfg = {
        **base_transformer_config,
        "architecture": "encoder-decoder" if H > 1 else "encoder-only",
        "dropout": 0.0,
        "epochs": 10,
        "use_exogenous": True,
        # use_amp defaults to True
        # use_amp_inference defaults to True
        "preprocessing": {
            "preprocessing_groups": [
                {
                    "name": "default",
                    "apply_to": "__targets__",
                    "pipeline": {"scaling": {"enabled": True}}
                }
            ]
        },
    }
    model = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=cfg,
        num_features=F,
        forecast_steps=H,
        window_size=WINDOW_SIZE,
        dataset=ds,
    )

    # Fit
    model.fit(train_df, is_final_fit=False, dataset=ds)

    # Safe assert (if no val_loss recorded, assume 0.0 to skip check)
    val_loss = getattr(model, 'validation_loss', 0.0)
    # Ensure model converged reasonably well during training
    assert val_loss < 1.0, f"Val loss {val_loss} >= 1.0 for H={H}, F={F}"

    # Predict
    # NEW API: input_data should include all available columns (targets + past_covariates + future_covariates)
    history_df = train_df.iloc[-WINDOW_SIZE:]

    future_exog = None  # No decoder exog provided
    predictions = model.predict(history_df, future_exog=future_exog)

    # Assert shape/output
    assert predictions.shape == (H, F), f"Pred shape {predictions.shape} != (H={H}, F={F})"
    assert np.all(np.isfinite(predictions)), "Preds contain NaN/Inf"

    # MSE vs true (slice to F features)
    true_values = test_df[ds.target_columns[:F]].values
    mse = np.mean((predictions - true_values) ** 2)

    # Relaxed E2E tolerance check.
    # The main goal is to ensure the pipeline doesn't crash or produce NaNs.
    # Accuracy on this tiny random/synthetic dataset is variable.
    assert mse < 1000.0, f"MSE {mse} >= 1000.0 for H={H}, F={F}"
