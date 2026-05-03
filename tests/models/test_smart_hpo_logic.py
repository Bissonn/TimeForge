import pytest
from unittest.mock import MagicMock
from models.transformer import TransformerForecaster
from models.lstm import LSTMForecaster
from models.arima import ARIMAForecaster
from utils.dataset import TimeSeriesDataset

@pytest.fixture
def mock_dataset():
    ds = MagicMock(spec=TimeSeriesDataset)
    ds.target_columns = ["target"]
    ds.past_covariates = []
    ds.future_covariates = []
    ds.columns = ["target"]
    # New API
    ds.past_covariates = []
    ds.future_covariates = []
    return ds

@pytest.fixture
def base_context():
    return MagicMock()

class TestSmartHPO:
    """
    Verifies that model-specific HPO heuristics (filtering, priors, validation)
    work correctly without running full training.
    """

    def test_transformer_smart_sampling_head_consistency(self, mock_dataset, base_context):
        """Verify Transformer enforces mathematical consistency (hidden_size % num_heads == 0)."""
        model = TransformerForecaster(
            model_params={"hidden_size": 64, "num_heads": 4},
            num_features=1, forecast_steps=5, window_size=10, dataset=mock_dataset, run_context=base_context
        )

        # Valid combination
        assert model.validate_param_combination({"hidden_size": 128, "num_heads": 8}) is True

        # Invalid combination (128 is not divisible by 6)
        assert model.validate_param_combination({"hidden_size": 128, "num_heads": 6}) is False

    def test_transformer_filter_space_encoder_only(self, mock_dataset, base_context):
        """Verify Transformer removes decoder params when architecture is encoder-only."""
        # Case 1: Fixed in config
        model = TransformerForecaster(
            model_params={"architecture": "encoder-only", "hidden_size": 64, "num_heads": 4},
            num_features=1, forecast_steps=5, window_size=10, dataset=mock_dataset, run_context=base_context
        )

        space = {
            "hidden_size": {"min": 32, "max": 64},
            "num_decoder_layers": [1, 2, 3],
            "decoder_input_size": 32
        }

        filtered = model.filter_search_space(space, model.model_params)

        assert "hidden_size" in filtered
        assert "num_decoder_layers" not in filtered
        assert "decoder_input_size" not in filtered

    def test_lstm_filter_dropout_single_layer(self, mock_dataset, base_context):
        """Verify LSTM removes dropout from search space if num_layers is fixed to 1."""
        model = LSTMForecaster(
            model_params={"num_layers": 1, "hidden_size": 32},
            num_features=1, forecast_steps=5, window_size=10, dataset=mock_dataset, run_context=base_context
        )

        space = {
            "hidden_size": [32, 64],
            "dropout": {"min": 0.1, "max": 0.5}
        }

        filtered = model.filter_search_space(space, model.model_params)

        assert "hidden_size" in filtered
        assert "dropout" not in filtered # Should be removed

    def test_lstm_keep_dropout_multi_layer(self, mock_dataset, base_context):
        """Verify LSTM keeps dropout if num_layers > 1."""
        model = LSTMForecaster(
            model_params={"num_layers": 2, "hidden_size": 32},
            num_features=1, forecast_steps=5, window_size=10, dataset=mock_dataset, run_context=base_context
        )

        space = {"dropout": {"min": 0.1, "max": 0.5}}
        filtered = model.filter_search_space(space, model.model_params)
        assert "dropout" in filtered

    def test_arima_filter_seasonal_params(self, mock_dataset, base_context):
        """Verify ARIMAForecaster (non-seasonal) removes seasonal params."""
        model = ARIMAForecaster(
            model_params={"p": 1, "d": 1, "q": 1},
            num_features=1, forecast_steps=5, window_size=10, dataset=mock_dataset, run_context=base_context
        )

        space = {
            "p": [1, 2],
            "P": [0, 1], # Seasonal
            "D": [0, 1], # Seasonal
            "seasonal_period": 12
        }

        filtered = model.filter_search_space(space, model.model_params)

        assert "p" in filtered
        assert "P" not in filtered
        assert "D" not in filtered
        assert "seasonal_period" not in filtered

    def test_transformer_smart_priors(self, mock_dataset, base_context):
        """Verify Transformer suggests valid priors."""
        model = TransformerForecaster(
            model_params={"architecture": "encoder-only", "hidden_size": 128, "num_heads": 4},
            num_features=1, forecast_steps=5, window_size=10, dataset=mock_dataset, run_context=base_context
        )

        space = {"hidden_size": [64, 128], "num_heads": [4, 8], "num_encoder_layers": [2, 4]}
        priors = model.suggest_smart_priors(space, model.model_params)

        assert len(priors) > 0
        # Verify prior only contains keys from space
        for p in priors:
            assert all(k in space for k in p.keys())
