import pytest
import torch
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from hypothesis import given, settings, strategies as st, HealthCheck

from models.factory import ModelFactory
from utils.dataset import TimeSeriesDataset
from core.context import RunContext
from utils.preprocessor import Preprocessor

# --- Helpers for Hypothesis Tests ---

def window_and_dims():
    return st.tuples(
        st.integers(2, 5),  # window_size
        st.integers(1, 3),  # n_targets
        st.integers(4, 8),  # hidden_size
    )

def _make_context_and_dataset(n_targets, n_enc=0, n_dec=0):
    """Helper to create context and dataset manually for hypothesis tests."""
    # 1. Ephemeral Context
    tmp_dir = tempfile.mkdtemp()
    ctx = RunContext.from_base_path(
        base_path=Path(tmp_dir),
        run_id="hyp_run",
        experiment_name="hyp_exp"
    )

    # 2. Dataset
    # Ensure distinct prefixes to avoid accidental overlaps
    targets = [f"tgt_{i}" for i in range(n_targets)]
    enc_exog = [f"enc_{i}" for i in range(n_enc)]
    dec_exog = [f"dec_{i}" for i in range(n_dec)]
    all_cols = targets + enc_exog + dec_exog

    data = pd.DataFrame(
        np.random.randn(50, len(all_cols)),
        columns=all_cols,
        index=pd.date_range("2020-01-01", periods=50, freq="D")
    )

    # Pass 'targets' list explicitly as the 4th argument.
    # Passing 'all_cols' caused the dataset to treat exog cols as targets -> Overlap Error.
    ds = TimeSeriesDataset(
        "dummy",
        {},
        num_features=n_targets,
        data=data,
        columns=targets,
        past_covariates=enc_exog,
        future_covariates=dec_exog
    )
    ds.development_data = data
    ds.test_data = data

    return ctx, ds

def _manual_fit_setup(model, dataset):
    """Helper to setup model state as if it was fitted, including preprocessor."""
    model.fitted = True
    # Crucial: Sync device to CPU for tests to avoid runtime errors (input on cuda, weights on cpu)
    model.device = torch.device("cpu")
    model.model.to("cpu")

    # Initialize preprocessor manually
    preprocessing_config = model.model_params.get("preprocessing", {})
    model.preprocessor = Preprocessor(
        preprocessing_config,
        target_columns=dataset.target_columns,
        exog_columns=dataset.past_covariates + dataset.future_covariates
    )
    # Use fit_transform() as fit() might not exist or be empty in base Preprocessor
    model.preprocessor.fit_transform(dataset.series)

# --- Tests ---

@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(window_and_dims())
def test_lstm_direct_equals_iterative_H1(dims):
    # Unpack hypothesis args manually
    window_size, n_targets, hidden_size = dims

    # Setup infrastructure
    ctx, ds = _make_context_and_dataset(n_targets)

    # Common Params
    params = {
        "hidden_size": hidden_size,
        "num_layers": 1,
        "dropout": 0.0,
        "forecast_steps": 1, # H=1 guarantees equivalence
        "batch_size": 4
    }

    # 1. Direct Model
    direct_params = {**params, "strategy": "direct"}
    model_direct = ModelFactory.create(
        "lstm", "lstm_direct",
        run_context=ctx,
        model_params=direct_params,
        num_features=n_targets,
        forecast_steps=1,
        window_size=window_size,
        dataset=ds
    )
    _manual_fit_setup(model_direct, ds)

    # 2. Iterative Model (stateless for H=1 equivalence)
    iter_params = {**params, "strategy": "iterative", "iterative_stateful": False}
    model_iter = ModelFactory.create(
        "lstm", "lstm_iter",
        run_context=ctx,
        model_params=iter_params,
        num_features=n_targets,
        forecast_steps=1,
        window_size=window_size,
        dataset=ds
    )
    _manual_fit_setup(model_iter, ds)

    # Force weights equality
    # Sync FULL model state (including linear projection layer), not just LSTM cells
    model_iter.model.load_state_dict(model_direct.model.state_dict())

    # Predict
    input_data = ds.series.iloc[:window_size+5] # Small chunk

    pred_direct = model_direct.predict(input_data)
    pred_iter = model_iter.predict(input_data)

    pd.testing.assert_frame_equal(pred_direct, pred_iter, atol=1e-5)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(window_and_dims())
def test_transformer_direct_equals_iterative_H1(dims):
    window_size, n_targets, hidden_size = dims
    # Ensure hidden_size is divisible by num_heads (default 4) for Transformer
    hidden_size = (hidden_size // 4 + 1) * 4

    ctx, ds = _make_context_and_dataset(n_targets)

    params = {
        "hidden_size": hidden_size,
        "num_heads": 4,
        "num_encoder_layers": 1,
        "dropout": 0.0,
        "forecast_steps": 1,
        "batch_size": 4
    }

    # Direct
    model_direct = ModelFactory.create(
        "transformer", "tf_direct", run_context=ctx,
        model_params={**params, "strategy": "direct", "architecture": "encoder-only"},
        num_features=n_targets, forecast_steps=1, window_size=window_size, dataset=ds
    )
    _manual_fit_setup(model_direct, ds)

    # Iterative
    model_iter = ModelFactory.create(
        "transformer", "tf_iter", run_context=ctx,
        model_params={**params, "strategy": "iterative", "architecture": "encoder-only"},
        num_features=n_targets, forecast_steps=1, window_size=window_size, dataset=ds
    )
    _manual_fit_setup(model_iter, ds)

    model_iter.model.load_state_dict(model_direct.model.state_dict())

    input_data = ds.series.iloc[:window_size+5]

    pred_direct = model_direct.predict(input_data)
    pred_iter = model_iter.predict(input_data)

    pd.testing.assert_frame_equal(pred_direct, pred_iter, atol=1e-5)


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(window_and_dims())
def test_transformer_encoder_only_iterative_uses_horizon0(dims):
    """Verify that encoder-only iterative model sets internal horizon to 1."""
    window_size, n_targets, hidden_size = dims
    hidden_size = (hidden_size // 4 + 1) * 4

    ctx, ds = _make_context_and_dataset(n_targets)

    model = ModelFactory.create(
        "transformer", "tf_iter_chk", run_context=ctx,
        model_params={
            "hidden_size": hidden_size, "num_heads": 4, "num_encoder_layers": 1,
            "strategy": "iterative", "architecture": "encoder-only", "dropout": 0.0
        },
        num_features=n_targets, forecast_steps=10, window_size=window_size, dataset=ds
    )

    # Internal model should have horizon 1 (autoregressive step)
    assert model.model.forecast_steps == 1
    # But the wrapper knows it needs to forecast 10 steps
    assert model.forecast_steps == 10


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.integers(1, 3))
def test_encoder_decoder_direct_respects_future_exog(n_dec):
    """
    Verify that encoder-decoder architecture correctly wires decoder exogenous variables.
    """
    ctx, ds = _make_context_and_dataset(1, n_enc=0, n_dec=n_dec)

    model = ModelFactory.create(
        "transformer", "tf_enc_dec", run_context=ctx,
        model_params={
            "hidden_size": 32, "num_heads": 4, "num_encoder_layers": 1,
            "num_decoder_layers": 1, "architecture": "encoder-decoder",
            "strategy": "direct", "dropout": 0.0
        },
        num_features=1, forecast_steps=5, window_size=10, dataset=ds
    )

    _manual_fit_setup(model, ds)

    # Create input and future exog
    input_data = ds.series.iloc[:15]

    # Future exog must cover forecast horizon
    future_exog = pd.DataFrame(
        np.random.randn(5, n_dec),
        columns=ds.future_covariates,
        index=pd.date_range(input_data.index[-1] + pd.Timedelta(days=1), periods=5)
    )

    # This should pass without error
    try:
        model.predict(input_data, future_exog=future_exog)
    except Exception as e:
        pytest.fail(f"Prediction with future exog failed: {e}")


def test_encoder_only_direct_ignores_future_exog():
    """
    Verifies Encoder-Only Direct mode works correctly:
    1. Dataset config must strictly match model capabilities (n_dec=0 for encoder-only direct).
    2. If extraneous future_exog is passed to predict(), it should be handled/ignored gracefully.
    """
    # Create dataset with NO decoder columns (to pass validation in __init__)
    ctx, ds = _make_context_and_dataset(1, n_dec=0)

    model = ModelFactory.create(
        "transformer", "tf_enc_only", run_context=ctx,
        model_params={
            "hidden_size": 32, "num_heads": 4, "num_encoder_layers": 1,
            "architecture": "encoder-only", "strategy": "direct"
        },
        num_features=1, forecast_steps=5, window_size=10, dataset=ds
    )
    _manual_fit_setup(model, ds)

    input_data = ds.series.iloc[:15]
    # Passing random exog, should be ignored or handled gracefully
    fut_exog = pd.DataFrame(
        np.random.randn(5, 1),
        index=pd.date_range(input_data.index[-1] + pd.Timedelta(days=1), periods=5),
        columns=["ignored_col"]
    )

    preds = model.predict(input_data, future_exog=fut_exog)
    assert not preds.empty
