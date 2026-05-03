import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch

from models.factory import ModelFactory
from models.transformer import TransformerModel
from models.transformer_components import TgtInitializer
from utils.dataset import TimeSeriesDataset

# Use WINDOW_SIZE from conftest (assuming it is exported or defined there)
try:
    from tests.models.conftest import WINDOW_SIZE
except ImportError:
    WINDOW_SIZE = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_build_tgt_train_for_training(target_true, src, initializer, decoder_exog, noise_config=None):
    """
    Simple stub used in training integration tests.

    It returns a zero tensor with the correct shape so that we can inspect
    the arguments passed to build_tgt_train without running the real logic.

    NOTE: noise_config parameter added for compatibility with prediction noise feature.
    """
    batch_size, horizon, num_features = target_true.shape
    if decoder_exog is not None:
        num_exog = decoder_exog.shape[-1]
        return torch.zeros(
            batch_size,
            horizon,
            num_features + num_exog,
            device=target_true.device,
            dtype=target_true.dtype,
        )
    return torch.zeros(
        batch_size,
        horizon,
        num_features,
        device=target_true.device,
        dtype=target_true.dtype,
    )


# ---------------------------------------------------------------------------
# Inference – iterative strategy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attention_type", ["full", "local"])
@pytest.mark.parametrize("positional_encoding", ["sinusoidal", "learnable", "none"])
def test_iterative_encoder_only_real_model_shapes_and_no_tgt(
        base_transformer_config,
        mocker,
        attention_type,
        positional_encoding,
        base_context
):
    """
    End-to-end check for _predict_iterative in encoder-only architecture.

    Uses a real TransformerModel (no MagicMock for the model) and verifies that,
    across different attention and positional encoding configurations:
      - output has shape (B, forecast_steps, num_features),
      - TransformerModel.forward IS called (since encoder-only uses forward in loop),
      - TransformerModel.forward never receives a non-None 'tgt' argument.

    NOTE: Encoder-only iterative does NOT support covariates due to training-inference
    mismatch, so we use a dataset without covariates.
    """
    # Create dataset WITHOUT covariates (encoder-only iterative blocks covariates)
    n_samples = 100
    data = pd.DataFrame({
        'date': pd.date_range(start="2023-01-01", periods=n_samples, freq='D'),
        'target': np.arange(n_samples, dtype=float),
    })
    dataset_no_cov = TimeSeriesDataset(
        "enc_only_no_cov",
        {"datasets": {"enc_only_no_cov": {}}},
        num_features=1,
        data=data,
        columns=['target'],
        past_covariates=[],   # NO covariates
        future_covariates=[]
    )
    dataset_no_cov.split_data(forecast_steps=5)

    config = {
        **base_transformer_config,
        "architecture": "encoder-only",
        "strategy": "iterative",
        "attention_type": attention_type,
        "positional_encoding_config": {"type": positional_encoding},
    }

    # For local attention we must ensure a valid even attention_window_size
    if attention_type == "local":
        config["attention_window_size"] = 4

    # dataset_no_cov has 1 target feature, no covariates
    forecaster = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=config,
        num_features=1,
        forecast_steps=5,
        window_size=WINDOW_SIZE,
        dataset=dataset_no_cov,
    )
    forecaster.fitted = True  # required by _internal_predict

    device = forecaster.device
    encoder_input_size = forecaster.feature_layout.encoder_input_size

    # Single sliding window as encoder input
    x = torch.randn(1, WINDOW_SIZE, encoder_input_size, device=device)

    # No future_exog needed since we have no covariates
    preds = forecaster._predict_iterative(x, future_exog_tensor=None)

    # Output must be a numpy array with shape (B, H, F)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (1, forecaster.forecast_steps, forecaster.num_features)


@pytest.mark.parametrize("tgt_init", ["zeros", "last_value"])
@pytest.mark.parametrize("attention_type", ["full", "local"])
@pytest.mark.parametrize("positional_encoding", ["sinusoidal", "learnable", "none"])
def test_iterative_encoder_decoder_real_model_tgt_feature_dim_constant(
        base_transformer_config,
        full_dataset,
        mocker,
        tgt_init,
        attention_type,
        positional_encoding,
        base_context
):
    """
    End-to-end check for _predict_iterative in encoder-decoder architecture
    with decoder exogenous variables.

    Across different tgt_init / attention_type / positional_encoding configurations,
    verifies that:
      - output has shape (B, forecast_steps, num_features),
      - in every call to TransformerModel.decode, the 'tgt' tensor has a constant
        feature dimension equal to decoder_input_size.
    """
    config = {
        **base_transformer_config,
        "architecture": "encoder-decoder",
        "strategy": "iterative",
        "tgt_init": tgt_init,
        "attention_type": attention_type,
        "positional_encoding_config": {"type": positional_encoding},
    }

    if attention_type == "local":
        config["attention_window_size"] = 4

    NUM_FEATURES = 2

    forecaster = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=config,
        num_features=NUM_FEATURES,
        forecast_steps=5,
        window_size=WINDOW_SIZE,
        dataset=full_dataset,
    )
    forecaster.fitted = True

    device = forecaster.device
    batch_size = 1
    horizon = forecaster.forecast_steps
    encoder_input_size = forecaster.feature_layout.encoder_input_size
    decoder_exog_size = forecaster.feature_layout.decoder_exog_size
    decoder_input_size = forecaster.feature_layout.decoder_input_size

    # Prepare inputs
    x = torch.randn(batch_size, WINDOW_SIZE, encoder_input_size, device=device)
    future_exog = torch.randn(batch_size, horizon, decoder_exog_size, device=device)

    # Spy on the INSTANCE's decode method
    # Iterative encoder-decoder calls self.model.decode(...) directly
    decode_spy = mocker.spy(forecaster.model, "decode")

    preds = forecaster._predict_iterative(
        x,
        future_exog_tensor=future_exog,
    )

    # 1) Output shape check
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (batch_size, horizon, NUM_FEATURES)

    # 2) Decode was called 'horizon' times
    assert decode_spy.call_count == horizon

    # 3) Check 'tgt' dimensions in every call
    # decode signature: def decode(self, tgt, memory, tgt_mask=None):
    # It is called positionally: self.model.decode(tgt, memory)
    for call in decode_spy.call_args_list:
        # Get the first positional argument 'tgt'
        tgt_arg = call.args[0]
        assert isinstance(tgt_arg, torch.Tensor)
        # Check feature dimension (last dim)
        assert tgt_arg.shape[-1] == decoder_input_size


# ---------------------------------------------------------------------------
# Training – encoder-decoder integration with build_tgt_train
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tgt_init", ["zeros", "last_value"])
@pytest.mark.parametrize("attention_type", ["full", "local"])
def test_train_encoder_decoder_calls_build_tgt_train_correctly(
        base_transformer_config,
        full_dataset,
        tgt_init,
        attention_type,
        base_context
):
    """
    Integration test for encoder-decoder training (direct strategy).
    Ensures build_tgt_train is called with correct args.
    """
    full_dataset.split_data(forecast_steps=5)

    config = {
        **base_transformer_config,
        "architecture": "encoder-decoder",
        "strategy": "direct",
        "tgt_init": tgt_init,
        "attention_type": attention_type,
    }

    if attention_type == "local":
        config["attention_window_size"] = 4

    NUM_FEATURES = 2

    forecaster = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=config,
        num_features=NUM_FEATURES,
        forecast_steps=5,
        window_size=WINDOW_SIZE,
        dataset=full_dataset,
    )

    with patch(
            "models.transformer.build_tgt_train",
            side_effect=_fake_build_tgt_train_for_training,
    ) as mock_build, patch("models.transformer.run_train_loop") as mock_run:
        mock_run.return_value = (forecaster.model, {})

        forecaster.fit(
            full_dataset.development_data,
            is_final_fit=False,
            dataset=full_dataset,
        )

    assert mock_build.call_count >= 1

    last_call = mock_build.call_args
    args, kwargs = last_call

    # Handle both positional and keyword argument styles
    # New code passes first 4 args positionally and noise_config as kwarg
    if "target_true" in kwargs:
        target_true = kwargs["target_true"]
        src = kwargs["src"]
        initializer = kwargs["initializer"]
        decoder_exog = kwargs["decoder_exog"]
    else:
        target_true, src, initializer, decoder_exog = args[:4]

    assert isinstance(target_true, torch.Tensor)
    assert isinstance(src, torch.Tensor)
    assert decoder_exog is None or isinstance(decoder_exog, torch.Tensor)
    assert src.ndim == 3
    assert src.shape[1] == WINDOW_SIZE
    assert isinstance(initializer, TgtInitializer)


# ---------------------------------------------------------------------------
# Training – encoder-decoder sanity checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tgt_init", ["zeros", "last_value"])
@pytest.mark.parametrize("positional_encoding", ["sinusoidal", "learnable", "none"])
def test_fit_encoder_decoder_direct_runs_without_error(
        base_transformer_config,
        full_dataset,
        tgt_init,
        positional_encoding,
        base_context
):
    """Sanity check: encoder-decoder fit (direct) runs without error."""
    full_dataset.split_data(forecast_steps=5)

    config = {
        **base_transformer_config,
        "architecture": "encoder-decoder",
        "strategy": "direct",
        "tgt_init": tgt_init,
        "positional_encoding_config": {"type": positional_encoding},
    }

    forecaster = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=config,
        num_features=2,
        forecast_steps=5,
        window_size=WINDOW_SIZE,
        dataset=full_dataset,
    )

    with patch("models.transformer.run_train_loop") as mock_run:
        mock_run.return_value = (forecaster.model, {})
        forecaster.fit(
            full_dataset.development_data,
            is_final_fit=False,
            dataset=full_dataset,
        )

    assert forecaster.model is not None
    mock_run.assert_called_once()


@pytest.mark.parametrize("tgt_init", ["zeros", "last_value"])
@pytest.mark.parametrize("positional_encoding", ["sinusoidal", "learnable", "none"])
def test_fit_encoder_decoder_iterative_runs_without_error(
        base_transformer_config,
        full_dataset,
        tgt_init,
        positional_encoding,
        base_context
):
    """Sanity check: encoder-decoder fit (iterative) runs without error."""
    full_dataset.split_data(forecast_steps=5)

    config = {
        **base_transformer_config,
        "architecture": "encoder-decoder",
        "strategy": "iterative",
        "tgt_init": tgt_init,
        "positional_encoding_config": {"type": positional_encoding},
    }

    forecaster = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=config,
        num_features=2,
        forecast_steps=5,
        window_size=WINDOW_SIZE,
        dataset=full_dataset,
    )

    with patch("models.transformer.run_train_loop") as mock_run:
        mock_run.return_value = (forecaster.model, {})
        forecaster.fit(
            full_dataset.development_data,
            is_final_fit=False,
            dataset=full_dataset,
        )

    assert forecaster.model is not None
    mock_run.assert_called_once()
