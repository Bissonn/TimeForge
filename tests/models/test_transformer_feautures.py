"""
Unit and integration tests for the unified TransformerForecaster model,
including its extended educational features like selectable architectures,
readout mechanisms, and positional encodings.
"""

import pytest
import pandas as pd
import numpy as np
import torch
from torch import nn
from unittest.mock import MagicMock

# Import model modules to ensure they are registered
import models.transformer
import models.lstm

from models.factory import ModelFactory
from utils.dataset import TimeSeriesDataset

# --- Fixtures for Tests ---

@pytest.fixture
def transformer_config():
    """Provides a minimal, valid configuration for the Transformer model."""
    return {
        'window_size': 10,
        'hidden_size': 16,
        'num_heads': 2,
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,
        'dim_ff_multiplier': 4.0,
        'epochs': 1,
        'batch_size': 4,
        'learning_rate': 0.01
    }

@pytest.fixture
def sample_timeseries_dataset():
    """Creates a simple TimeSeriesDataset for testing."""
    index = pd.date_range(start='2023-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'value_1': 50 + 10 * np.sin(np.linspace(0, 4 * np.pi, 50)),
        'value_2': 20 + 5 * np.cos(np.linspace(0, 4 * np.pi, 50))
    }, index=index)

    config = {
        'datasets': {'test_data': {'path': 'dummy.csv'}},
        'experiments': [{'validation_setup': {'forecast_steps': 5}}]
    }

    dataset = TimeSeriesDataset('test_data', config, data=data)
    dataset.split_data(forecast_steps=5)
    return dataset

# --- Main Tests ---

def test_transformer_baseline_end_to_end(transformer_config, sample_timeseries_dataset):
    """
    Scenario: Verify the basic end-to-end functionality of the TransformerForecaster
    using the default 'encoder-only' architecture.
    """
    num_features = sample_timeseries_dataset.series.shape[1]
    forecast_steps = 5

    forecaster = ModelFactory.create(
        "transformer",
        model_params=transformer_config,
        num_features=num_features,
        forecast_steps=forecast_steps
    )
    assert forecaster is not None
    assert forecaster.model is not None

    forecaster.fit(sample_timeseries_dataset.development_data, is_final_fit=True)
    assert forecaster.fitted

    input_data = sample_timeseries_dataset.development_data.iloc[-transformer_config['window_size']:]
    predictions = forecaster.predict(input_data)

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (forecast_steps, num_features)
    assert not predictions.isnull().values.any()

def test_transformer_architecture_selection(transformer_config):
    """
    Scenario: Verify that the model is built correctly based on the 'architecture' parameter.
    """
    num_features, forecast_steps = 2, 5

    config_enc = {**transformer_config, 'architecture': 'encoder-only'}
    forecaster_enc = ModelFactory.create("transformer", config_enc, num_features, forecast_steps)

    assert forecaster_enc.model.architecture == 'encoder-only'
    assert forecaster_enc.model.transformer_decoder is None

    config_enc_dec = {**transformer_config, 'architecture': 'encoder-decoder'}
    forecaster_enc_dec = ModelFactory.create("transformer", config_enc_dec, num_features, forecast_steps)

    assert forecaster_enc_dec.model.architecture == 'encoder-decoder'
    assert isinstance(forecaster_enc_dec.model.transformer_decoder, nn.TransformerDecoder)

@pytest.mark.parametrize("readout_mode", ["last", "mean", "max", "cls"])
def test_readout_modes_for_encoder_only(transformer_config, readout_mode):
    """
    Scenario: Test that all readout modes work correctly for the 'encoder-only' architecture.
    """
    config = {**transformer_config, 'architecture': 'encoder-only', 'readout': readout_mode}
    forecaster = ModelFactory.create("transformer", config, num_features=2, forecast_steps=5)
    model = forecaster.model

    batch_size, window_size = config['batch_size'], config['window_size']
    src = torch.randn(batch_size, window_size, 2).to(forecaster.device)

    try:
        output = model(src)
        assert output.shape == (batch_size, 5, 2)
    except Exception as e:
        pytest.fail(f"Model forward pass failed for readout mode '{readout_mode}': {e}")

@pytest.mark.parametrize("encoding_mode", ["sinusoidal", "learned", "none"])
def test_positional_encoding_modes(transformer_config, encoding_mode):
    """
    Scenario: Test that all positional encoding modes can be initialized correctly.
    """
    config = {**transformer_config, 'positional_encoding': encoding_mode}
    try:
        model = ModelFactory.create("transformer", config, num_features=2, forecast_steps=5).model
        if encoding_mode == "learned":
            assert isinstance(model.pos_encoder, nn.Parameter)
        elif encoding_mode == "sinusoidal":
            assert hasattr(model, 'pos_encoder_buffer')
        else:
            assert model.pos_encoder is None
    except Exception as e:
        pytest.fail(f"Model initialization failed for positional encoding '{encoding_mode}': {e}")

def test_encoder_decoder_e2e_pass(transformer_config, sample_timeseries_dataset):
    """
    Scenario: Verify the end-to-end training functionality for the 'encoder-decoder' architecture.
    """
    num_features, forecast_steps = 2, 5
    config = {**transformer_config, 'architecture': 'encoder-decoder'}

    forecaster = ModelFactory.create("transformer", config, num_features, forecast_steps)
    forecaster.fit(sample_timeseries_dataset.development_data, is_final_fit=True)
    assert forecaster.fitted

def test_prediction_strategy_dispatch(transformer_config, mocker):
    """
    Scenario: Verify that the correct internal prediction method is called based on the 'strategy' config.
    """
    num_features, forecast_steps = 2, 5
    dummy_input = pd.DataFrame(np.random.rand(10, num_features), index=pd.date_range('2023-01-01', periods=10, freq='D'))

    # --- SOLUTION: Mock the correct, deeper internal methods ---
    mock_internal_predict_direct = mocker.patch('models.base.NeuralTSForecaster._internal_predict')
    mock_internal_predict_iterative = mocker.patch('models.transformer.TransformerForecaster._internal_predict_iterative')

    # --- Test 1: Direct strategy ---
    config_direct = {**transformer_config, 'strategy': 'direct'}
    forecaster_direct = ModelFactory.create("transformer", config_direct, num_features, forecast_steps)
    forecaster_direct.fitted = True
    forecaster_direct.preprocessor = MagicMock()
    # Simulate preprocessor output
    forecaster_direct.preprocessor.transform.return_value = pd.DataFrame(np.random.rand(10, num_features))

    forecaster_direct.predict(dummy_input)

    mock_internal_predict_direct.assert_called_once()
    mock_internal_predict_iterative.assert_not_called()

    # Reset mocks
    mock_internal_predict_direct.reset_mock()

    # --- Test 2: Iterative strategy ---
    config_iterative = {**transformer_config, 'strategy': 'iterative'}
    forecaster_iterative = ModelFactory.create("transformer", config_iterative, num_features, forecast_steps)
    forecaster_iterative.fitted = True
    forecaster_iterative.preprocessor = MagicMock()
    forecaster_iterative.preprocessor.transform.return_value = pd.DataFrame(np.random.rand(10, num_features))
    forecaster_iterative.preprocessor.inverse_transforms.return_value = pd.DataFrame(np.random.rand(forecast_steps, num_features))

    forecaster_iterative.predict(dummy_input)

    mock_internal_predict_direct.assert_not_called()
    mock_internal_predict_iterative.assert_called_once()