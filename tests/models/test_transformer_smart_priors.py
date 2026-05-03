"""
Pytest Test Suite for Transformer Smart Priors with Dataset-Aware HPO
=======================================================================

Tests that Transformer properly uses helper methods from NeuralTSForecaster
for dataset-aware hyperparameter optimization.

This is parallel to the LSTM test suite and verifies:
1. Helper methods are inherited from base class (not duplicated)
2. Smart priors adapt based on dataset characteristics
3. Dropout adjustment for small datasets
4. Huber loss prioritization
5. Encoder/decoder size adaptation

Run:
    pytest test_transformer_smart_priors.py -v
    pytest test_transformer_smart_priors.py -v -s  # with print output
"""

import pytest
import pandas as pd
from utils.dataset import TimeSeriesDataset
from models.transformer import TransformerForecaster
from models.lstm import LSTMForecaster
from models.base import NeuralTSForecaster


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def small_hourly_dataset():
    """Create a small hourly dataset for testing (800 points)."""
    df = pd.DataFrame(
        {'value': range(800)},
        index=pd.date_range('2020-01-01', periods=800, freq='h')
    )

    config = {"datasets": {"test": {"freq": "H"}}}
    dataset = TimeSeriesDataset("test", config, num_features=1,data=df, columns=['value'], freq='H')
    dataset.split_data(forecast_steps=24)

    return dataset


@pytest.fixture
def medium_daily_dataset():
    """Create a medium daily dataset for testing (5000 points)."""
    df = pd.DataFrame(
        {'value': range(5000)},
        index=pd.date_range('2020-01-01', periods=5000, freq='D')
    )

    config = {"datasets": {"test": {"freq": "D"}}}
    dataset = TimeSeriesDataset("test", config, num_features=1, data=df, columns=['value'], freq='D')
    dataset.split_data(forecast_steps=7)

    return dataset


@pytest.fixture
def large_weekly_dataset():
    """Create a large weekly dataset for testing (520 points = ~10 years)."""
    df = pd.DataFrame(
        {'value': range(520)},
        index=pd.date_range('2020-01-01', periods=520, freq='W')
    )

    config = {"datasets": {"test": {"freq": "W"}}}
    dataset = TimeSeriesDataset("test", config, num_features=1, data=df, columns=['value'], freq='W')
    dataset.split_data(forecast_steps=4)

    return dataset


@pytest.fixture
def basic_model_params():
    """Basic Transformer model parameters for testing."""
    return {
        "hidden_size": 64,  # Changed from d_model
        "num_heads": 4,     # Changed from n_heads
        "num_encoder_layers": 2,  # Changed from n_encoder_layers
        "num_decoder_layers": 2,  # Changed from n_decoder_layers
        "d_ff": 256,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 10,
        "architecture": "encoder-decoder",  # Added!
        "strategy": "direct"  # Added!
    }


@pytest.fixture
def transformer_model(small_hourly_dataset, basic_model_params, base_context):
    """Create a basic Transformer model instance for testing."""
    model = TransformerForecaster(
        model_params=basic_model_params,
        num_features=1,
        forecast_steps=24,
        window_size=168,
        dataset=small_hourly_dataset,
        run_context=base_context
    )
    return model


@pytest.fixture
def param_space():
    """Standard hyperparameter search space for testing."""
    return {
        'hidden_size': [32, 64, 128, 256],  # Changed from d_model
        'num_heads': [2, 4, 8],  # Changed from n_heads
        'num_encoder_layers': [1, 2, 3, 4],  # Changed from n_encoder_layers
        'num_decoder_layers': [1, 2, 3, 4],  # Changed from n_decoder_layers
        'd_ff': [128, 256, 512, 1024],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [16, 32, 64]
    }


# ============================================================================
# TEST: INHERITANCE
# ============================================================================

class TestInheritance:
    """Test that Transformer properly inherits helper methods from NeuralTSForecaster."""

    def test_has_inherited_methods(self, transformer_model):
        """Transformer should have all 3 helper methods inherited from base class."""
        assert hasattr(transformer_model, '_extract_dataset_metadata'), \
            "Transformer should inherit _extract_dataset_metadata"

        assert hasattr(transformer_model, '_infer_seasonal_period'), \
            "Transformer should inherit _infer_seasonal_period"

        assert hasattr(transformer_model, '_categorize_dataset_size'), \
            "Transformer should inherit _categorize_dataset_size"

    def test_methods_are_from_base_class(self):
        """Transformer methods should be the exact same objects as base class methods."""
        assert (TransformerForecaster._infer_seasonal_period is
                NeuralTSForecaster._infer_seasonal_period), \
            "Transformer should use base class _infer_seasonal_period, not override it"

        assert (TransformerForecaster._categorize_dataset_size is
                NeuralTSForecaster._categorize_dataset_size), \
            "Transformer should use base class _categorize_dataset_size, not override it"

        assert (TransformerForecaster._extract_dataset_metadata is
                NeuralTSForecaster._extract_dataset_metadata), \
            "Transformer should use base class _extract_dataset_metadata, not override it"

    def test_consistency_with_lstm(self):
        """Transformer and LSTM should use identical inherited methods."""
        assert (TransformerForecaster._infer_seasonal_period is
                LSTMForecaster._infer_seasonal_period), \
            "Transformer and LSTM should share _infer_seasonal_period"

        assert (TransformerForecaster._categorize_dataset_size is
                LSTMForecaster._categorize_dataset_size), \
            "Transformer and LSTM should share _categorize_dataset_size"

        assert (TransformerForecaster._extract_dataset_metadata is
                LSTMForecaster._extract_dataset_metadata), \
            "Transformer and LSTM should share _extract_dataset_metadata"


# ============================================================================
# TEST: FREQUENCY INFERENCE (inherited from base)
# ============================================================================

class TestFrequencyInference:
    """Test _infer_seasonal_period helper method."""

    @pytest.mark.parametrize("freq,expected_period", [
        ('H', 24),      # Hourly (old uppercase)
        ('h', 24),      # Hourly (new lowercase)
        ('D', 7),       # Daily → weekly cycle
        ('W', 52),      # Weekly → annual cycle
        ('M', 12),      # Monthly → annual cycle
        ('Q', 4),       # Quarterly → annual cycle
        ('T', 1440),    # Minutely (old)
        ('min', 1440),  # Minutely (new)
        ('15T', 96),    # 15-minute intervals (old)
        ('15min', 96),  # 15-minute intervals (new)
        ('2H', 12),     # 2-hour intervals (old)
        ('2h', 12),     # 2-hour intervals (new)
    ])
    def test_standard_frequencies(self, freq, expected_period):
        """Test frequency to seasonal period mapping."""
        result = TransformerForecaster._infer_seasonal_period(freq)
        assert result == expected_period, \
            f"Expected {expected_period} for freq='{freq}', got {result}"

    def test_unknown_frequency(self):
        """Unknown frequencies should return None."""
        assert TransformerForecaster._infer_seasonal_period('XYZ') is None
        assert TransformerForecaster._infer_seasonal_period('') is None
        assert TransformerForecaster._infer_seasonal_period(None) is None


# ============================================================================
# TEST: SIZE CATEGORIZATION (inherited from base)
# ============================================================================

class TestSizeCategorization:
    """Test _categorize_dataset_size helper method."""

    @pytest.mark.parametrize("length,expected_category", [
        (100, 'small'),
        (500, 'small'),
        (999, 'small'),
        (1000, 'medium'),
        (5000, 'medium'),
        (9999, 'medium'),
        (10000, 'large'),
        (50000, 'large'),
        (100000, 'large'),
    ])
    def test_size_categories(self, length, expected_category):
        """Test dataset size categorization."""
        result = TransformerForecaster._categorize_dataset_size(length)
        assert result == expected_category, \
            f"Expected '{expected_category}' for length={length}, got '{result}'"


# ============================================================================
# TEST: METADATA EXTRACTION (inherited from base)
# ============================================================================

class TestMetadataExtraction:
    """Test _extract_dataset_metadata helper method."""

    def test_small_hourly_dataset(self, transformer_model, small_hourly_dataset):
        """Test metadata extraction for small hourly dataset."""
        metadata = transformer_model._extract_dataset_metadata(small_hourly_dataset)

        assert metadata['freq'] == 'H'
        assert metadata['seasonal_period'] == 24
        assert metadata['series_length'] == 800
        assert metadata['size_category'] == 'small'
        assert metadata.get('has_past_exog', False) is False
        assert metadata.get('has_future_exog', False) is False

    def test_medium_daily_dataset(self, basic_model_params, medium_daily_dataset, base_context):
        """Test metadata extraction for medium daily dataset."""
        model = TransformerForecaster(
            model_params=basic_model_params,
            num_features=1,
            forecast_steps=7,
            window_size=30,
            dataset=medium_daily_dataset,
            run_context=base_context
        )

        metadata = model._extract_dataset_metadata(medium_daily_dataset)

        assert metadata['freq'] == 'D'
        assert metadata['seasonal_period'] == 7
        assert metadata['series_length'] == 5000
        assert metadata['size_category'] == 'medium'

    def test_large_weekly_dataset(self, basic_model_params, large_weekly_dataset, base_context):
        """Test metadata extraction for large weekly dataset."""
        model = TransformerForecaster(
            model_params=basic_model_params,
            num_features=1,
            forecast_steps=4,
            window_size=52,
            dataset=large_weekly_dataset,
            run_context=base_context
        )

        metadata = model._extract_dataset_metadata(large_weekly_dataset)

        assert metadata['freq'] == 'W'
        assert metadata['seasonal_period'] == 52
        assert metadata['series_length'] == 520
        assert metadata['size_category'] == 'small'  # 520 < 1000

    def test_none_dataset(self, transformer_model):
        """Passing None should return empty dict."""
        metadata = transformer_model._extract_dataset_metadata(None)
        assert metadata == {}


# ============================================================================
# TEST: SMART PRIORS INTEGRATION
# ============================================================================

class TestSmartPriorsIntegration:
    """Test that suggest_smart_priors uses inherited methods correctly."""

    def test_smart_priors_method_exists(self, transformer_model):
        """Transformer should have suggest_smart_priors method."""
        assert hasattr(transformer_model, 'suggest_smart_priors'), \
            "Transformer should have suggest_smart_priors method"

    def test_generates_priors(self, transformer_model, param_space, small_hourly_dataset):
        """Smart priors should generate configurations based on dataset."""
        priors = transformer_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_hourly_dataset
        )

        assert priors is not None, "Should return priors"
        assert isinstance(priors, list), "Priors should be a list"
        assert len(priors) > 0, "Should return at least one prior"
        assert len(priors) <= 5, "Should return at most 5 priors (typical)"

        # Each prior should be a dict with hyperparameters
        for prior in priors:
            assert isinstance(prior, dict), "Each prior should be a dict"
            assert len(prior) > 0, "Each prior should have at least one parameter"

    def test_priors_adapt_to_dataset_size(self, basic_model_params, param_space, base_context):
        """Priors should adapt dropout based on dataset size."""
        # Small dataset
        small_df = pd.DataFrame(
            {'value': range(800)},
            index=pd.date_range('2020-01-01', periods=800, freq='h')
        )
        small_dataset = TimeSeriesDataset("small", {}, num_features=1, data=small_df, columns=['value'], freq='H')
        small_dataset.split_data(forecast_steps=24)

        small_model = TransformerForecaster(
            model_params=basic_model_params,
            num_features=1,
            forecast_steps=24,
            window_size=168,
            dataset=small_dataset,
            run_context=base_context
        )

        small_priors = small_model.suggest_smart_priors(param_space, {}, small_dataset)

        # Large dataset (12000 points)
        large_df = pd.DataFrame(
            {'value': range(12000)},
            index=pd.date_range('2020-01-01', periods=12000, freq='h')
        )
        large_dataset = TimeSeriesDataset("large", {}, num_features=1, data=large_df, columns=['value'], freq='H')
        large_dataset.split_data(forecast_steps=24)

        large_model = TransformerForecaster(
            model_params=basic_model_params,
            num_features=1,
            forecast_steps=24,
            window_size=168,
            dataset=large_dataset,
            run_context=base_context
        )

        large_priors = large_model.suggest_smart_priors(param_space, {}, large_dataset)

        # Extract dropout values
        small_dropouts = [p.get('dropout') for p in small_priors if 'dropout' in p]
        large_dropouts = [p.get('dropout') for p in large_priors if 'dropout' in p]

        # Small datasets should have higher dropout on average
        if small_dropouts and large_dropouts:
            avg_small_dropout = sum(small_dropouts) / len(small_dropouts)
            avg_large_dropout = sum(large_dropouts) / len(large_dropouts)

            assert avg_small_dropout >= avg_large_dropout, \
                f"Small dataset should have higher dropout (got {avg_small_dropout} vs {avg_large_dropout})"

    def test_priors_include_transformer_specific_params(self, transformer_model, param_space, small_hourly_dataset):
        """Priors should include Transformer-specific parameters."""
        priors = transformer_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_hourly_dataset
        )

        # Check that at least some priors have Transformer-specific params
        has_hidden_size = any('hidden_size' in p for p in priors)  # Changed from d_model
        has_num_heads = any('num_heads' in p for p in priors)  # Changed from n_heads
        has_encoder_layers = any('num_encoder_layers' in p for p in priors)  # Changed
        has_decoder_layers = any('num_decoder_layers' in p for p in priors)  # Changed

        # At least some of the Transformer-specific params should be present
        assert has_hidden_size or has_num_heads or has_encoder_layers or has_decoder_layers, \
            "Priors should include Transformer-specific parameters"

    def test_huber_loss_prioritization(self, transformer_model, param_space, small_hourly_dataset):
        """Huber loss should be prioritized in early priors."""
        # Add loss to param space
        extended_param_space = param_space.copy()
        extended_param_space['loss'] = ['mse', 'huber', 'l1']

        priors = transformer_model.suggest_smart_priors(
            param_space=extended_param_space,
            fixed_params={},
            dataset=small_hourly_dataset
        )

        # Check if Huber appears in early priors (typically prior 2)
        if len(priors) >= 2:
            # Huber should appear in one of the first 3 priors
            early_priors_losses = [p.get('loss') for p in priors[:3]]
            assert 'huber' in early_priors_losses, \
                "Huber loss should be prioritized in early priors"


# ============================================================================
# TEST: DRY PRINCIPLE VERIFICATION
# ============================================================================

class TestDRYPrinciple:
    """Verify that DRY refactoring was successful."""

    def test_no_duplicate_helper_methods(self):
        """Transformer should not have its own implementation of helper methods."""
        import inspect

        # Get Transformer source code
        transformer_source = inspect.getsource(TransformerForecaster)

        # These methods should NOT be defined in Transformer
        # (they should be inherited from NeuralTSForecaster)
        assert 'def _infer_seasonal_period' not in transformer_source, \
            "Transformer should not define _infer_seasonal_period (should be inherited)"

        assert 'def _categorize_dataset_size' not in transformer_source, \
            "Transformer should not define _categorize_dataset_size (should be inherited)"

        # _extract_dataset_metadata might appear in docstrings/comments
        # Check for actual method definition
        lines = [line.strip() for line in transformer_source.split('\n')]
        method_definitions = [line for line in lines if line.startswith('def _extract_dataset_metadata')]

        assert len(method_definitions) == 0, \
            "Transformer should not define _extract_dataset_metadata (should be inherited)"

    def test_single_source_of_truth(self):
        """All neural models should use the same helper implementations."""
        from models.base import NeuralTSForecaster

        # Get the actual function objects
        base_seasonal = NeuralTSForecaster._infer_seasonal_period
        base_size = NeuralTSForecaster._categorize_dataset_size
        base_metadata = NeuralTSForecaster._extract_dataset_metadata

        transformer_seasonal = TransformerForecaster._infer_seasonal_period
        transformer_size = TransformerForecaster._categorize_dataset_size
        transformer_metadata = TransformerForecaster._extract_dataset_metadata

        lstm_seasonal = LSTMForecaster._infer_seasonal_period
        lstm_size = LSTMForecaster._categorize_dataset_size
        lstm_metadata = LSTMForecaster._extract_dataset_metadata

        # All should be the exact same object (identity, not just equality)
        assert base_seasonal is transformer_seasonal is lstm_seasonal
        assert base_size is transformer_size is lstm_size
        assert base_metadata is transformer_metadata is lstm_metadata


# ============================================================================
# PERFORMANCE MARKERS
# ============================================================================

class TestPerformance:
    """Performance-related tests (marked as slow)."""

    def test_metadata_extraction_performance(self, transformer_model, small_hourly_dataset):
        """Metadata extraction should be fast."""
        import time

        start = time.time()
        for _ in range(1000):
            transformer_model._extract_dataset_metadata(small_hourly_dataset)
        elapsed = time.time() - start

        # Should complete 1000 extractions in less than 1 second
        assert elapsed < 1.0, \
            f"Metadata extraction too slow: {elapsed:.3f}s for 1000 calls"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])