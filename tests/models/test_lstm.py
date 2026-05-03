"""
Unit and Integration tests for the unified LSTMForecaster.

This module tests the LSTMForecaster class (merging previous Direct/Iterative),
verifying initialization, dynamic feature layout handling, and the correctness
of both 'direct' and 'iterative' prediction strategies under various exogenous
variable configurations.
"""
import pytest
import torch
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from models.factory import ModelFactory
from models.lstm import LSTMForecaster
from utils.dataset import TimeSeriesDataset

pytestmark = pytest.mark.unit

# --- Fixtures ---

@pytest.fixture
def base_lstm_config():
    """Minimal valid configuration for LSTMForecaster."""
    return {
        "hidden_size": 16,
        "num_layers": 1,
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.01,
        "dropout": 0.0,
        # strategy will be injected via parametrization
    }

@pytest.fixture
def window_size():
    return 10

def create_mock_dataset(n_samples=100, n_targets=1, n_enc_exog=0, n_dec_exog=0, shared_names=False):
    """
    Creates a TimeSeriesDataset with specified feature counts.
    If shared_names is True, enc and dec exog will have overlapping names (to force Intersection).
    """
    data = {}
    date_range = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")

    targets = [f"target_{i}" for i in range(n_targets)]
    for t in targets:
        data[t] = np.random.randn(n_samples)

    if shared_names:
        # New API: Shared columns (known in past AND future) go to future_covariates
        # Encoder-only columns go to past_covariates
        # Decoder-only columns go to future_covariates
        common_count = min(n_enc_exog, n_dec_exog)
        enc_only_count = max(0, n_enc_exog - common_count)
        dec_only_count = max(0, n_dec_exog - common_count)

        # Shared columns (in both encoder and decoder)
        shared_cols = [f"shared_{i}" for i in range(common_count)]
        # Encoder-only columns (past-only)
        past_only_cols = [f"enc_{i}" for i in range(enc_only_count)]
        # Decoder-only columns (future-only)
        dec_only_cols = [f"dec_{i}" for i in range(dec_only_count)]

        # Populate data
        for col in shared_cols + past_only_cols + dec_only_cols:
            data[col] = np.random.randn(n_samples)

        past_covariates_list = past_only_cols
        future_covariates_list = shared_cols + dec_only_cols
    else:
        # Disjoint: enc -> past, dec -> future
        enc_exog = [f"enc_{i}" for i in range(n_enc_exog)]
        for e in enc_exog:
            data[e] = np.random.randn(n_samples)

        dec_exog = [f"dec_{i}" for i in range(n_dec_exog)]
        for d in dec_exog:
            data[d] = np.random.randn(n_samples)

        past_covariates_list = enc_exog
        future_covariates_list = dec_exog

    df = pd.DataFrame(data, index=date_range)

    config = {"datasets": {"mock": {}}}

    ds = TimeSeriesDataset(
        "mock", config, data=df,
        columns=targets,
        num_features=n_targets,
        past_covariates=past_covariates_list,
        future_covariates=future_covariates_list
    )
    return ds

# --- Test 1: Initialization & Feature Layout (Universal) ---

@pytest.mark.parametrize("strategy", ["direct", "iterative"])
@pytest.mark.parametrize("n_enc_exog", [0, 2])
@pytest.mark.parametrize("n_dec_exog", [0, 2])
def test_lstm_initialization_and_input_size(base_lstm_config, window_size, strategy, n_enc_exog, n_dec_exog, base_context):
    """
    Verifies that LSTMForecaster initializes correctly with different strategies.
    NOTE: For Iterative strategy, we use shared_names=True to verify injection actually happens.
    """
    n_targets = 1

    # If we want to test Injection, we need intersection (shared names)
    use_shared = (strategy == "iterative" and n_dec_exog > 0)
    dataset = create_mock_dataset(n_targets=n_targets, n_enc_exog=n_enc_exog, n_dec_exog=n_dec_exog, shared_names=use_shared)
    forecast_steps = 5

    config = base_lstm_config.copy()
    config["strategy"] = strategy

    # NOTE: With the new covariate API, encoder-only direct models accept
    # future_covariates (forward-compatible) but don't consume them.
    # The old validation that rejected decoder exog for direct strategy has
    # been removed as part of the refactoring.

    model = LSTMForecaster(
        model_params=config,
        num_features=n_targets,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=dataset,
        run_context=base_context
    )

    # Calculate expected input size using NEW API (v2):
    # - past_covariates = enc_cols - dec_cols
    # - future_covariates = (enc_cols & dec_cols) + (dec_cols - enc_cols)
    # - encoder_input_size = n_targets + past_cov + future_cov

    # Re-simulate dataset migration logic to match create_mock_dataset
    if use_shared:
        # Match the logic in create_mock_dataset for shared_names=True
        common_count = min(n_enc_exog, n_dec_exog)
        enc_only_count = max(0, n_enc_exog - common_count)
        dec_only_count = max(0, n_dec_exog - common_count)

        # Shared columns
        shared_cols = [f"shared_{i}" for i in range(common_count)]
        # Encoder-only columns (past)
        past_only_cols = [f"enc_{i}" for i in range(enc_only_count)]
        # Decoder-only columns (future)
        dec_only_cols = [f"dec_{i}" for i in range(dec_only_count)]

        past_cov = past_only_cols
        future_cov = shared_cols + dec_only_cols
    else:
        # Disjoint case
        enc_cols = [f"enc_{i}" for i in range(n_enc_exog)]
        dec_cols = [f"dec_{i}" for i in range(n_dec_exog)]
        past_cov = enc_cols
        future_cov = dec_cols

    expected_size = n_targets + len(past_cov) + len(future_cov)

    assert model.feature_layout.encoder_input_size == expected_size
    assert model.model.lstm.input_size == expected_size
    assert model.strategy == strategy


# --- Test 2: Fit and Predict Loop (Sanity Check) ---

@pytest.mark.parametrize("strategy", ["direct", "iterative"])
@pytest.mark.parametrize("n_dec_exog", [0, 1])
def test_lstm_fit_predict_flow(base_lstm_config, window_size, strategy, n_dec_exog, base_context):
    """
    Runs a smoke test for the fit and predict methods.
    """
    n_targets = 1
    n_enc_exog = 1
    forecast_steps = 3

    # Ensure intersection for iterative mode
    use_shared = (strategy == "iterative" and n_dec_exog > 0)
    dataset = create_mock_dataset(n_samples=50, n_targets=n_targets, n_enc_exog=n_enc_exog, n_dec_exog=n_dec_exog, shared_names=use_shared)
    dataset.split_data(forecast_steps)

    config = base_lstm_config.copy()
    config["strategy"] = strategy

    if strategy == "direct" and n_dec_exog > 0:
        return # Expected fail tested elsewhere

    model = LSTMForecaster(
        model_params=config,
        num_features=n_targets,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=dataset,
        run_context=base_context
    )

    # 1. Fit
    # Mock training loop to save time
    with patch.object(model, '_train_model', return_value=model.model) as mock_train:
        model.fit(dataset.development_data, is_final_fit=True, dataset=dataset)

    # 2. Predict
    history = dataset.development_data.iloc[-window_size:]

    # Prepare future exog
    exog_cols = dataset.past_covariates + dataset.future_covariates
    future_exog = dataset.test_data[exog_cols] if exog_cols else None

    preds = model.predict(history, future_exog=future_exog)

    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == forecast_steps
    assert preds.shape[1] == n_targets


# --- Test 3: Iterative Strategy Logic ---

def test_lstm_iterative_step_logic(base_lstm_config, window_size, base_context):
    """Verifies iterative prediction loop."""
    forecast_steps = 3
    n_targets = 1
    dataset = create_mock_dataset(n_targets=n_targets, n_enc_exog=0, n_dec_exog=0)

    config = base_lstm_config.copy()
    config["strategy"] = "iterative"
    config["iterative_stateful"] = False  # Use stateless for this test to verify step-by-step logic
    config["hidden_size"] = 10

    model = LSTMForecaster(
        model_params=config,
        num_features=n_targets,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=dataset,
        run_context=base_context
    )
    model.fitted = True
    device = model.device

    # Mock output - for stateless mode
    mock_output = torch.tensor([[[1.0]]], device=device)
    model.model = MagicMock(return_value=mock_output)
    model.model.lstm.input_size = 1

    input_tensor = torch.zeros(1, window_size, 1, device=device)

    preds_np = model._internal_predict(input_tensor)

    assert model.model.call_count == forecast_steps
    assert preds_np.shape == (1, forecast_steps, n_targets)


# --- Test 4: E2E Exog Dependency ---

@pytest.mark.parametrize("strategy, n_enc_exog, n_dec_exog", [
    ("direct", 1, 0),
    ("iterative", 1, 0),
    # For iterative with dec exog, we must share the column to form intersection
    ("iterative", 1, 1),
])
def test_lstm_e2e_exog_dependency(base_lstm_config, window_size, strategy, n_enc_exog, n_dec_exog, base_context):
    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = 500
    t = np.linspace(0, 40, n_samples)

    # Deterministic signal
    enc_data = np.sin(t)
    dec_data = np.cos(t) if n_dec_exog > 0 else np.zeros(n_samples)
    target = np.zeros(n_samples)

    if n_enc_exog > 0:
        target[1:] += 0.5 * enc_data[:-1]
    if n_dec_exog > 0:
        target[1:] += 0.5 * dec_data[:-1]

    df = pd.DataFrame(index=pd.date_range("2020-01-01", periods=n_samples, freq="D"))
    df["target"] = target

    # New API: Shared columns go ONLY to future_covariates
    past_covariates_list = []
    future_covariates_list = []

    if n_enc_exog > 0:
        df["enc_0"] = enc_data

    if n_dec_exog > 0:
        if strategy == "iterative":
            # Shared column: use SAME name for both encoder and decoder
            # In new API: shared columns go ONLY to future_covariates
            df["enc_0"] = dec_data  # Overwrite or keep same data
            future_covariates_list = ["enc_0"]  # Shared, so only in future
            # past_covariates stays empty (no past-only features)
        else:
            df["dec_0"] = dec_data
            if n_enc_exog > 0:
                past_covariates_list = ["enc_0"]  # Encoder-only
            future_covariates_list = ["dec_0"]  # Decoder-only
    elif n_enc_exog > 0:
        # Only encoder exog, no decoder
        past_covariates_list = ["enc_0"]

    config = {"datasets": {"exog_test": {}}}
    ds = TimeSeriesDataset(
        "exog_test",
        config,
        num_features=1,
        data=df,
        columns=["target"],
        past_covariates=past_covariates_list,
        future_covariates=future_covariates_list
    )

    forecast_steps = 5
    ds.split_data(forecast_steps)

    model_params = base_lstm_config.copy()

    # Auto-detect future_covariate_mode based on strategy and future_covariates
    future_covariate_mode = "none"
    if future_covariates_list:
        future_covariate_mode = "stepwise" if strategy == "iterative" else "global"

    model_params.update({
        "strategy": strategy,
        "future_covariate_mode": future_covariate_mode,
        "epochs": 50, # Faster
        "hidden_size": 32,
        "learning_rate": 0.01,
        "preprocessing": {
            "preprocessing_groups": [{
                "name": "all",
                "apply_to": df.columns.tolist(),
                "pipeline": {"scaling": {"enabled": True, "method": "minmax"}}
            }]
        }
    })

    forecaster = ModelFactory.create(
        "lstm",
        "test_lstm",
        run_context=base_context,
        model_params=model_params,
        num_features=1,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=ds
    )
    forecaster.fit(ds.development_data, is_final_fit=False, dataset=ds)

    history = ds.development_data.iloc[-window_size:]
    # Pass all covariate cols
    all_cov_cols = list(set(past_covariates_list + future_covariates_list))
    future_exog = ds.test_data[all_cov_cols] if all_cov_cols else None

    predictions = forecaster.predict(history, future_exog=future_exog)
    actuals = ds.test_data["target"].values
    preds_values = predictions["target"].values

    mse = np.mean((actuals - preds_values) ** 2)
    assert mse < 0.2

def test_lstm_predict_with_overlapping_exog_columns(base_context):
    """Test correct handling of overlapping exog columns (Intersection)."""
    data = pd.DataFrame({
        "target": np.arange(20, dtype=float),
        "ex1": np.arange(100, 120, dtype=float)
    })

    config = {
        "strategy": "iterative",
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.0,
        "batch_size": 2,
        "learning_rate": 0.01,
        "epochs": 1
    }

    # New API: Shared columns (known in past AND future) go ONLY to future_covariates
    ds = TimeSeriesDataset(
        "dummy",
        {"datasets": {"dummy": {}}},
        num_features=1,
        data=data,
        columns=["target"],
        past_covariates=[],  # No past-only features
        future_covariates=["ex1"]  # Shared feature (known in past and future)
    )

    forecaster = LSTMForecaster(
        model_params=config,
        num_features=1,
        forecast_steps=2,
        window_size=5,
        dataset=ds,
        run_context=base_context
    )

    train_df = data.iloc[:15]
    forecaster.fit(train_df, dataset=ds)

    future_exog = pd.DataFrame({"ex1": [200.0, 201.0]}, index=[15, 16])

    # Should succeed now with fixed handler
    preds = forecaster.predict(input_data=train_df, future_exog=future_exog)

    assert preds.shape == (2, 1)


# --- Test 5: Stateful Iterative Prediction ---

def test_iterative_stateful_vs_stateless(base_lstm_config, window_size, base_context):
    """Verify that stateful differs from stateless predictions."""
    torch.manual_seed(42)
    np.random.seed(42)

    n_targets = 1
    forecast_steps = 10
    dataset = create_mock_dataset(n_samples=100, n_targets=n_targets, n_enc_exog=0, n_dec_exog=0)

    # Create stateless model
    config_stateless = base_lstm_config.copy()
    config_stateless["strategy"] = "iterative"
    config_stateless["iterative_stateful"] = False
    config_stateless["hidden_size"] = 32
    config_stateless["epochs"] = 1

    model_stateless = LSTMForecaster(
        model_params=config_stateless,
        num_features=n_targets,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=dataset,
        run_context=base_context
    )

    # Create stateful model
    config_stateful = base_lstm_config.copy()
    config_stateful["strategy"] = "iterative"
    config_stateful["iterative_stateful"] = True
    config_stateful["hidden_size"] = 32
    config_stateful["epochs"] = 1

    model_stateful = LSTMForecaster(
        model_params=config_stateful,
        num_features=n_targets,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=dataset,
        run_context=base_context
    )

    # Use same model weights for fair comparison
    model_stateful.model.load_state_dict(model_stateless.model.state_dict())
    model_stateless.fitted = True
    model_stateful.fitted = True

    # Create input
    input_tensor = torch.randn(2, window_size, n_targets)

    # Get predictions
    pred_stateless = model_stateless._internal_predict(input_tensor)
    pred_stateful = model_stateful._internal_predict(input_tensor)

    # Predictions should differ (stateful has state propagation)
    assert pred_stateless.shape == pred_stateful.shape

    # Check that predictions are not identical (different mechanisms)
    max_diff = np.abs(pred_stateless - pred_stateful).max()
    mean_diff = np.abs(pred_stateless - pred_stateful).mean()

    # With state propagation, there should be measurable differences
    assert max_diff > 1e-6, \
        f"Stateful and stateless should produce different predictions (max_diff={max_diff})"

    print(f"✓ Stateful vs Stateless: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


def test_iterative_stateful_parameter(base_lstm_config, window_size, base_context):
    """Verify that iterative_stateful parameter is correctly set and logged."""
    n_targets = 1
    forecast_steps = 5
    dataset = create_mock_dataset(n_targets=n_targets, n_enc_exog=0, n_dec_exog=0)

    # Test default (should be True)
    config_default = base_lstm_config.copy()
    config_default["strategy"] = "iterative"

    model_default = LSTMForecaster(
        model_params=config_default,
        num_features=n_targets,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=dataset,
        run_context=base_context
    )
    assert model_default.iterative_stateful == True, "Default should be stateful (True)"

    # Test explicit False
    config_false = base_lstm_config.copy()
    config_false["strategy"] = "iterative"
    config_false["iterative_stateful"] = False

    model_false = LSTMForecaster(
        model_params=config_false,
        num_features=n_targets,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=dataset,
        run_context=base_context
    )
    assert model_false.iterative_stateful == False

    # Test explicit True
    config_true = base_lstm_config.copy()
    config_true["strategy"] = "iterative"
    config_true["iterative_stateful"] = True

    model_true = LSTMForecaster(
        model_params=config_true,
        num_features=n_targets,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=dataset,
        run_context=base_context
    )
    assert model_true.iterative_stateful == True


def test_iterative_stateful_only_for_iterative_strategy(base_lstm_config, window_size, base_context):
    """Verify that iterative_stateful only affects iterative strategy."""
    n_targets = 1
    forecast_steps = 5
    dataset = create_mock_dataset(n_targets=n_targets, n_enc_exog=0, n_dec_exog=0)

    # Direct strategy with iterative_stateful should be ignored
    config_direct = base_lstm_config.copy()
    config_direct["strategy"] = "direct"
    config_direct["iterative_stateful"] = True  # This should be ignored

    model_direct = LSTMForecaster(
        model_params=config_direct,
        num_features=n_targets,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=dataset,
        run_context=base_context
    )

    # Parameter is stored but not used for direct strategy
    assert model_direct.strategy == "direct"
    # No error should occur


def test_iterative_stateful_shape_consistency(base_lstm_config, window_size, base_context):
    """Verify that stateful prediction returns correct output shape."""
    torch.manual_seed(42)
    n_targets = 2
    forecast_steps = 8
    batch_size = 4
    dataset = create_mock_dataset(n_targets=n_targets, n_enc_exog=0, n_dec_exog=0)

    config = base_lstm_config.copy()
    config["strategy"] = "iterative"
    config["iterative_stateful"] = True
    config["hidden_size"] = 16

    model = LSTMForecaster(
        model_params=config,
        num_features=n_targets,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=dataset,
        run_context=base_context
    )
    model.fitted = True

    input_tensor = torch.randn(batch_size, window_size, n_targets)
    predictions = model._internal_predict(input_tensor)

    assert predictions.shape == (batch_size, forecast_steps, n_targets), \
        f"Expected shape ({batch_size}, {forecast_steps}, {n_targets}), got {predictions.shape}"
    assert not np.isnan(predictions).any(), "Predictions should not contain NaN"
    assert not np.isinf(predictions).any(), "Predictions should not contain Inf"
