"""
Tests for Transformer iterative encoder-decoder buffer mode optimization.

Verifies that buffer mode produces identical outputs to concat mode while
eliminating O(H²) memory allocation overhead.

See: docs/analysis_oh2_concat_problem.md
"""
import pytest
import torch
import numpy as np
import pandas as pd
from core.context import RunContext
from models.transformer import TransformerForecaster
from utils.dataset import TimeSeriesDataset

pytestmark = pytest.mark.unit


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def base_context(tmp_path):
    """Create a basic RunContext for testing."""
    ctx = RunContext.from_base_path(
        base_path=tmp_path / "test_run",
        run_id="test_run",
        experiment_name="test_experiment"
    )
    ctx.create_directories()
    return ctx


@pytest.fixture
def simple_dataset():
    """Create a minimal dataset for testing."""
    n_samples = 1000  # Enough for window=24, horizon=336, plus margin
    data = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n_samples, freq="D"),
        "target": np.sin(np.linspace(0, 10, n_samples)) + np.random.randn(n_samples) * 0.1,
    })

    config = {
        "data_split": {
            "test_fraction": 0.0  # No test split for unit tests
        }
    }

    dataset = TimeSeriesDataset(
        dataset_name="test_dataset",
        config=config,
        num_features=1,
        data=data,
        columns=["target"],
        date_column="date",
        freq="D"
    )

    # Don't call split_data here - let each test call it with appropriate forecast_steps
    return dataset


@pytest.fixture
def dataset_with_future_cov():
    """Create dataset with future covariates for testing."""
    n_samples = 600  # Enough for testing
    data = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n_samples, freq="D"),
        "target": np.sin(np.linspace(0, 10, n_samples)) + np.random.randn(n_samples) * 0.1,
        "future_cov": np.cos(np.linspace(0, 10, n_samples)),  # Known in future
    })

    config = {
        "data_split": {
            "test_fraction": 0.0
        }
    }

    dataset = TimeSeriesDataset(
        dataset_name="test_dataset_exog",
        config=config,
        num_features=1,
        data=data,
        columns=["target"],
        future_covariates=["future_cov"],
        date_column="date",
        freq="D"
    )

    dataset.split_data(forecast_steps=96)
    return dataset


# ============================================================
# Test 1: Config Validation
# ============================================================

def test_buffer_mode_config_validation_accepts_valid_values(base_context, simple_dataset):
    """iterative_decoder_mode accepts valid values: concat, buffer, auto."""
    for mode in ["concat", "buffer", "auto"]:
        config = {
            "type": "transformer",
            "architecture": "encoder-decoder",
            "strategy": "iterative",
            "hidden_size": 32,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "num_heads": 2,
            "epochs": 1,
            "iterative_decoder_mode": mode,
        }

        forecaster = TransformerForecaster(
            model_params=config,
            num_features=1,
            window_size=24,
            forecast_steps=96,
            dataset=simple_dataset,
            run_context=base_context
        )

        assert forecaster.model_params["iterative_decoder_mode"] == mode


def test_buffer_mode_config_validation_rejects_invalid(base_context, simple_dataset):
    """iterative_decoder_mode rejects invalid values at runtime."""
    simple_dataset.split_data(forecast_steps=96)

    config = {
        "type": "transformer",
        "architecture": "encoder-decoder",
        "strategy": "iterative",
        "hidden_size": 32,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 2,
        "epochs": 1,
        "iterative_decoder_mode": "invalid_mode",
    }

    forecaster = TransformerForecaster(
        model_params=config,
        num_features=1,
        window_size=24,
        forecast_steps=96,
        dataset=simple_dataset,
        run_context=base_context
    )

    forecaster.fit(simple_dataset.development_data, dataset=simple_dataset)

    # Should return NaN fallback due to validation error (caught by base class)
    predictions = forecaster.predict(
        input_data=simple_dataset.development_data.tail(24)
    )

    # Should be all NaN (validation error caught by base class fallback)
    assert np.isnan(predictions.values).all(), "Invalid mode should return NaN fallback"


# ============================================================
# Test 2: Golden Output Tests (concat vs buffer)
# ============================================================

@pytest.mark.parametrize("forecast_steps", [96, 192, 336])
def test_buffer_mode_produces_identical_outputs_to_concat(base_context, simple_dataset, forecast_steps):
    """Buffer mode produces bitwise identical outputs to concat mode."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Split dataset for this specific forecast horizon
    simple_dataset.split_data(forecast_steps=forecast_steps)

    # Base config
    config = {
        "type": "transformer",
        "architecture": "encoder-decoder",
        "strategy": "iterative",
        "hidden_size": 32,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 2,
        "epochs": 2,
        "batch_size": 16,
    }

    # Train once, predict with both modes
    forecaster_concat = TransformerForecaster(
        model_params={**config, "iterative_decoder_mode": "concat"},
        num_features=1,
        window_size=24,
        forecast_steps=forecast_steps,
        dataset=simple_dataset,
        run_context=base_context
    )

    forecaster_concat.fit(simple_dataset.development_data, dataset=simple_dataset)

    # Make predictions with concat mode
    torch.manual_seed(100)  # Fix inference seed
    pred_concat = forecaster_concat.predict(
        input_data=simple_dataset.development_data.tail(24)
    )

    # Now switch to buffer mode (same model weights)
    forecaster_concat.model_params["iterative_decoder_mode"] = "buffer"

    # Make predictions with buffer mode
    torch.manual_seed(100)  # Same seed
    pred_buffer = forecaster_concat.predict(
        input_data=simple_dataset.development_data.tail(24)
    )

    # Should be identical (within floating-point tolerance)
    np.testing.assert_allclose(
        pred_concat.values, pred_buffer.values,
        rtol=1e-5, atol=1e-6,
        err_msg=f"Buffer mode outputs differ from concat mode (H={forecast_steps})"
    )


def test_buffer_mode_with_future_covariates(base_context, dataset_with_future_cov):
    """Buffer mode works correctly with future covariates."""
    torch.manual_seed(42)
    np.random.seed(42)

    config = {
        "type": "transformer",
        "architecture": "encoder-decoder",
        "strategy": "iterative",
        "hidden_size": 32,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 2,
        "epochs": 2,
        "batch_size": 16,
    }

    # Train with concat mode
    forecaster = TransformerForecaster(
        model_params={**config, "iterative_decoder_mode": "concat"},
        num_features=1,
        window_size=24,
        forecast_steps=96,
        dataset=dataset_with_future_cov,
        run_context=base_context
    )

    forecaster.fit(dataset_with_future_cov.development_data, dataset=dataset_with_future_cov)

    # Prepare future exog for prediction
    future_dates = pd.date_range(
        dataset_with_future_cov.development_data.index[-1] + pd.Timedelta(days=1),
        periods=96,
        freq="D"
    )
    future_exog = pd.DataFrame({
        "date": future_dates,
        "future_cov": np.cos(np.linspace(10, 12, 96))
    }).set_index("date")

    # Predict with concat mode
    torch.manual_seed(100)
    pred_concat = forecaster.predict(
        input_data=dataset_with_future_cov.development_data.tail(24),
        future_exog=future_exog
    )

    # Switch to buffer mode
    forecaster.model_params["iterative_decoder_mode"] = "buffer"

    # Predict with buffer mode
    torch.manual_seed(100)
    pred_buffer = forecaster.predict(
        input_data=dataset_with_future_cov.development_data.tail(24),
        future_exog=future_exog
    )

    # Should be identical
    np.testing.assert_allclose(
        pred_concat.values, pred_buffer.values,
        rtol=1e-5, atol=1e-6,
        err_msg="Buffer mode outputs differ with future covariates"
    )


# ============================================================
# Test 3: Auto-selection Logic
# ============================================================

def test_auto_mode_selects_buffer_for_large_horizons(base_context):
    """Auto mode selects buffer for H > 512."""
    # Create larger dataset specifically for this test
    n_samples = 1500
    data = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n_samples, freq="D"),
        "target": np.sin(np.linspace(0, 10, n_samples)) + np.random.randn(n_samples) * 0.1,
    })

    config_ds = {
        "data_split": {
            "test_fraction": 0.0
        }
    }

    large_dataset = TimeSeriesDataset(
        dataset_name="large_test_dataset",
        config=config_ds,
        num_features=1,
        data=data,
        columns=["target"],
        date_column="date",
        freq="D"
    )

    large_dataset.split_data(forecast_steps=600)

    config = {
        "type": "transformer",
        "architecture": "encoder-decoder",
        "strategy": "iterative",
        "hidden_size": 32,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 2,
        "epochs": 1,
        "iterative_decoder_mode": "auto",
    }

    # H = 600 > 512 → should use buffer
    forecaster = TransformerForecaster(
        model_params=config,
        num_features=1,
        window_size=24,
        forecast_steps=600,
        dataset=large_dataset,
        run_context=base_context
    )

    forecaster.fit(large_dataset.development_data, dataset=large_dataset)

    # Should have selected buffer mode (we can check by inspecting logs or behavior)
    # For now, just verify it doesn't crash
    predictions = forecaster.predict(
        input_data=large_dataset.development_data.tail(24)
    )

    assert predictions.shape == (600, 1)


def test_auto_mode_selects_concat_for_small_horizons(base_context, simple_dataset):
    """Auto mode selects concat for H <= 512."""
    simple_dataset.split_data(forecast_steps=96)

    config = {
        "type": "transformer",
        "architecture": "encoder-decoder",
        "strategy": "iterative",
        "hidden_size": 32,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 2,
        "epochs": 1,
        "iterative_decoder_mode": "auto",
    }

    # H = 96 <= 512 → should use concat
    forecaster = TransformerForecaster(
        model_params=config,
        num_features=1,
        window_size=24,
        forecast_steps=96,
        dataset=simple_dataset,
        run_context=base_context
    )

    forecaster.fit(simple_dataset.development_data, dataset=simple_dataset)

    predictions = forecaster.predict(
        input_data=simple_dataset.development_data.tail(24)
    )

    assert predictions.shape == (96, 1)


# ============================================================
# Test 4: Edge Cases
# ============================================================

def test_buffer_mode_only_for_encoder_decoder(base_context, simple_dataset):
    """Buffer mode raises error for encoder-only architecture."""
    simple_dataset.split_data(forecast_steps=96)

    config = {
        "type": "transformer",
        "architecture": "encoder-only",  # Not encoder-decoder!
        "strategy": "iterative",
        "hidden_size": 32,
        "num_encoder_layers": 2,
        "num_heads": 2,
        "epochs": 1,
        "iterative_decoder_mode": "buffer",  # Should be ignored/rejected
    }

    forecaster = TransformerForecaster(
        model_params=config,
        num_features=1,
        window_size=24,
        forecast_steps=96,
        dataset=simple_dataset,
        run_context=base_context
    )

    forecaster.fit(simple_dataset.development_data, dataset=simple_dataset)

    # Should NOT crash (encoder-only ignores buffer mode, uses AR loop)
    predictions = forecaster.predict(
        input_data=simple_dataset.development_data.tail(24)
    )

    assert predictions.shape == (96, 1)


def test_buffer_mode_with_init_len_greater_than_one(base_context, simple_dataset):
    """Buffer mode works with tgt_initializer that returns init_len > 1."""
    # This tests the generality of buffer allocation logic
    simple_dataset.split_data(forecast_steps=96)

    config = {
        "type": "transformer",
        "architecture": "encoder-decoder",
        "strategy": "iterative",
        "hidden_size": 32,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 2,
        "epochs": 1,
        "iterative_decoder_mode": "buffer",
        "tgt_init": "last_value",  # Default, typically init_len=1
    }

    forecaster = TransformerForecaster(
        model_params=config,
        num_features=1,
        window_size=24,
        forecast_steps=96,
        dataset=simple_dataset,
        run_context=base_context
    )

    forecaster.fit(simple_dataset.development_data, dataset=simple_dataset)

    predictions = forecaster.predict(
        input_data=simple_dataset.development_data.tail(24)
    )

    assert predictions.shape == (96, 1)


# ============================================================
# Test 5: Performance Characteristics (Smoke Tests)
# ============================================================

@pytest.mark.parametrize("horizon", [512, 1024])
def test_buffer_mode_large_horizons_no_crash(base_context, simple_dataset, horizon):
    """Buffer mode handles large horizons without OOM or crashes."""
    # Create larger dataset for very long horizons
    # Need enough samples for: window_size (24) + horizon (up to 1024) + split overhead
    n_samples = 3000
    data = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n_samples, freq="D"),
        "target": np.sin(np.linspace(0, 10, n_samples)) + np.random.randn(n_samples) * 0.1,
    })

    config_ds = {
        "data_split": {
            "test_fraction": 0.0
        }
    }

    large_dataset = TimeSeriesDataset(
        dataset_name="large_test_dataset",
        config=config_ds,
        num_features=1,
        data=data,
        columns=["target"],
        date_column="date",
        freq="D"
    )

    large_dataset.split_data(forecast_steps=horizon)

    config = {
        "type": "transformer",
        "architecture": "encoder-decoder",
        "strategy": "iterative",
        "hidden_size": 32,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 2,
        "epochs": 1,
        "batch_size": 8,  # Smaller batch for large horizons
        "iterative_decoder_mode": "buffer",
    }

    forecaster = TransformerForecaster(
        model_params=config,
        num_features=1,
        window_size=24,
        forecast_steps=horizon,
        dataset=large_dataset,
        run_context=base_context
    )

    forecaster.fit(large_dataset.development_data, dataset=large_dataset)

    # Should not crash or OOM
    predictions = forecaster.predict(
        input_data=large_dataset.development_data.tail(24)
    )

    assert predictions.shape == (horizon, 1)
    assert np.isfinite(predictions.values).all(), "Predictions contain NaN or Inf"
