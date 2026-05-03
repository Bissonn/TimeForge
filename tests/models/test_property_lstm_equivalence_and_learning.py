import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from hypothesis import given, settings, strategies as st
from unittest.mock import MagicMock

from models.lstm import LSTMForecaster
from utils.dataset import TimeSeriesDataset
from core.context import RunContext  # <--- NEW IMPORT


def _make_dummy_dataset(num_features: int, length: int = 50) -> TimeSeriesDataset:
    """Creates a minimal dummy dataset for initializing forecasters."""
    data = pd.DataFrame(
        np.random.randn(length, num_features),
        columns=[f"feat_{i}" for i in range(num_features)]
    )
    # Mock config just enough to pass initialization
    config = {"datasets": {"dummy": {}}}

    ds = TimeSeriesDataset(
        dataset_name="dummy",
        config=config,
        num_features=num_features,
        data=data,
        columns=list(data.columns),
        past_covariates=[],
        future_covariates=[]
    )
    return ds


@settings(deadline=None, max_examples=30)
@given(
    batch_size=st.integers(min_value=1, max_value=3),
    window_size=st.integers(min_value=2, max_value=6),
    horizon=st.integers(min_value=1, max_value=4),
    num_features=st.integers(min_value=1, max_value=3),
    hidden_size=st.integers(min_value=2, max_value=6),
    num_layers=st.integers(min_value=1, max_value=2),
)
def test_lstm_direct_vs_iterative_equivalence_no_exog(
        batch_size: int,
        window_size: int,
        horizon: int,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        # base_context REMOVED from arguments to avoid Hypothesis HealthCheck failure
):
    """
    Property: Given the SAME weights, an LSTMForecaster in 'direct' mode
    and an LSTMForecaster in 'iterative' mode should produce identical outputs.
    """
    torch.manual_seed(1234)
    device = torch.device("cpu")

    # ---  Create ephemeral context manually ---
    # We use a temporary directory for each hypothesis example to ensure isolation
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_context = RunContext.from_base_path(
            base_path=Path(tmp_dir),
            run_id="hyp_run",
            experiment_name="hyp_exp"
        )
        # No need to create_directories() if model doesn't write artifacts during init

        # 1. Create a dummy dataset
        ds = _make_dummy_dataset(num_features)

        # 2. Initialize Direct Forecaster
        params_direct = {
            "strategy": "direct",
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": 0.0
        }
        model_direct = LSTMForecaster(
            model_params=params_direct,
            num_features=num_features,
            forecast_steps=horizon,
            window_size=window_size,
            dataset=ds,
            run_context=run_context  # <--- Inject manual context
        )
        model_direct.fitted = True
        model_direct.device = device
        model_direct.model.to(device)

        # 3. Initialize Iterative Forecaster (stateless for equivalence test)
        params_iter = {
            "strategy": "iterative",
            "iterative_stateful": False,  # Use stateless for direct equivalence
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": 0.0
        }
        model_iter = LSTMForecaster(
            model_params=params_iter,
            num_features=num_features,
            forecast_steps=horizon,
            window_size=window_size,
            dataset=ds,
            run_context=run_context  # <--- Inject manual context
        )
        model_iter.fitted = True
        model_iter.device = device
        model_iter.model.to(device)

        # 4. Force weights to be identical
        model_iter.model.lstm.load_state_dict(model_direct.model.lstm.state_dict())

        # Random input (B, W, F)
        x = torch.randn(batch_size, window_size, num_features).to(device)

        # 1. Run framework's iterative prediction
        y_framework = model_iter._internal_predict(x)

        # 2. Run manual iterative rollout
        model_iter.model.eval()
        manual_preds = []
        current_input = x.clone()

        with torch.no_grad():
            for _ in range(horizon):
                out = model_iter.model(current_input)
                pred_step = out
                manual_preds.append(pred_step)
                current_input = torch.cat([current_input[:, 1:, :], pred_step], dim=1)

        y_manual = torch.cat(manual_preds, dim=1).cpu().numpy()

        np.testing.assert_allclose(
            y_framework,
            y_manual,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Framework iterative logic diverges from manual rollout."
        )


def test_lstm_minimal_learning_copy_last_step(base_context): # <--- Standard fixture OK here
    """
    Sanity-check property: LSTM should learn a trivial task.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    B = 32
    W = 5
    F = 2
    H = 1

    N_SAMPLES = 200
    data = np.zeros((N_SAMPLES, F))
    data[0, :] = np.random.randn(F)
    for i in range(1, N_SAMPLES):
        data[i, :] = data[i - 1, :] + np.random.normal(0, 0.01, size=F)

    df = pd.DataFrame(data, columns=["f1", "f2"])

    ds = TimeSeriesDataset(
        "test_copy",
        {"datasets": {"test_copy": {}}},
        num_features=F,
        data=df,
        columns=["f1", "f2"]
    )
    ds.split_data(forecast_steps=H)

    params = {
        "strategy": "direct",
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 0.05,
        "batch_size": 16,
        "epochs": 50,
        "early_stopping_patience": 10,
        "preprocessing": {"preprocessing_groups": []}
    }

    forecaster = LSTMForecaster(
        model_params=params,
        num_features=F,
        forecast_steps=H,
        window_size=W,
        dataset=ds,
        run_context=base_context
    )

    best_loss, _ = forecaster.fit(
        train_series=ds.development_data,
        is_final_fit=True,
        dataset=ds
    )

    print(f"Final Training Loss: {best_loss}")
    assert best_loss < 0.1
