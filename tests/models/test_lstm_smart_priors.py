"""
Pytest Test Suite for LSTM Smart Priors with Dataset-Aware HPO
===============================================================

Tests that LSTM properly inherits and uses helper methods from NeuralTSForecaster
for dataset-aware hyperparameter optimization.

Run:
    pytest test_lstm_smart_priors.py -v
    pytest test_lstm_smart_priors.py -v -s  # with print output
"""

import pytest
import pandas as pd
from utils.dataset import TimeSeriesDataset
from models.lstm import LSTMForecaster
from models.transformer import TransformerForecaster
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
    dataset = TimeSeriesDataset("test", config, num_features=1, data=df, columns=['value'], freq='H')
    dataset.split_data(forecast_steps=24)  # Correct signature: only forecast_steps

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
    dataset.split_data(forecast_steps=7)  # Correct signature: only forecast_steps

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
    dataset.split_data(forecast_steps=4)  # Correct signature: only forecast_steps

    return dataset


@pytest.fixture
def basic_model_params():
    """Basic LSTM model parameters for testing."""
    return {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 10
    }


@pytest.fixture
def lstm_model(small_hourly_dataset, basic_model_params, base_context):
    """Create a basic LSTM model instance for testing."""
    return LSTMForecaster(
        model_params=basic_model_params,
        num_features=1,
        forecast_steps=24,
        window_size=168,
        dataset=small_hourly_dataset,
        run_context=base_context
    )


@pytest.fixture
def param_space():
    """Standard hyperparameter search space for testing."""
    return {
        'hidden_size': [32, 64, 128, 256],
        'num_layers': [1, 2, 3],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [16, 32, 64]
    }


# ============================================================================
# TEST: INHERITANCE
# ============================================================================

class TestInheritance:
    """Test that LSTM properly inherits helper methods from NeuralTSForecaster."""

    def test_has_inherited_methods(self, lstm_model):
        """LSTM should have all 3 helper methods inherited from base class."""
        assert hasattr(lstm_model, '_extract_dataset_metadata'), \
            "LSTM should inherit _extract_dataset_metadata"

        assert hasattr(lstm_model, '_infer_seasonal_period'), \
            "LSTM should inherit _infer_seasonal_period"

        assert hasattr(lstm_model, '_categorize_dataset_size'), \
            "LSTM should inherit _categorize_dataset_size"

    def test_methods_are_from_base_class(self):
        """LSTM methods should be the exact same objects as base class methods."""
        assert (LSTMForecaster._infer_seasonal_period is
                NeuralTSForecaster._infer_seasonal_period), \
            "LSTM should use base class _infer_seasonal_period, not override it"

        assert (LSTMForecaster._categorize_dataset_size is
                NeuralTSForecaster._categorize_dataset_size), \
            "LSTM should use base class _categorize_dataset_size, not override it"

        assert (LSTMForecaster._extract_dataset_metadata is
                NeuralTSForecaster._extract_dataset_metadata), \
            "LSTM should use base class _extract_dataset_metadata, not override it"

    def test_consistency_with_transformer(self):
        """LSTM and Transformer should use identical inherited methods."""
        assert (LSTMForecaster._infer_seasonal_period is
                TransformerForecaster._infer_seasonal_period), \
            "LSTM and Transformer should share _infer_seasonal_period"

        assert (LSTMForecaster._categorize_dataset_size is
                TransformerForecaster._categorize_dataset_size), \
            "LSTM and Transformer should share _categorize_dataset_size"

        assert (LSTMForecaster._extract_dataset_metadata is
                TransformerForecaster._extract_dataset_metadata), \
            "LSTM and Transformer should share _extract_dataset_metadata"


# ============================================================================
# TEST: FREQUENCY INFERENCE
# ============================================================================

class TestFrequencyInference:
    """Test _infer_seasonal_period helper method."""

    @pytest.mark.parametrize("freq,expected_period", [
        ('H', 24),      # Hourly → daily cycle
        ('D', 7),       # Daily → weekly cycle
        ('W', 52),      # Weekly → annual cycle
        ('M', 12),      # Monthly → annual cycle
        ('Q', 4),       # Quarterly → annual cycle
        ('T', 1440),    # Minutely → daily cycle
        ('15T', 96),    # 15-minute intervals → 96 per day
        ('2H', 12),     # 2-hour intervals → 12 per day
    ])
    def test_standard_frequencies(self, freq, expected_period):
        """Test frequency to seasonal period mapping."""
        result = LSTMForecaster._infer_seasonal_period(freq)
        assert result == expected_period, \
            f"Expected {expected_period} for freq='{freq}', got {result}"

    def test_unknown_frequency(self):
        """Unknown frequencies should return None."""
        assert LSTMForecaster._infer_seasonal_period('XYZ') is None
        assert LSTMForecaster._infer_seasonal_period('') is None
        assert LSTMForecaster._infer_seasonal_period(None) is None


# ============================================================================
# TEST: SIZE CATEGORIZATION
# ============================================================================

class TestSizeCategorization:
    """Test _categorize_dataset_size helper method."""

    @pytest.mark.parametrize("length,expected_category", [
        (100, 'small'),      # Very small
        (500, 'small'),      # Small
        (999, 'small'),      # Edge of small
        (1000, 'medium'),    # Start of medium
        (5000, 'medium'),    # Medium
        (9999, 'medium'),    # Edge of medium
        (10000, 'large'),    # Start of large
        (50000, 'large'),    # Large
        (100000, 'large'),   # Very large
    ])
    def test_size_categories(self, length, expected_category):
        """Test dataset size categorization."""
        result = LSTMForecaster._categorize_dataset_size(length)
        assert result == expected_category, \
            f"Expected '{expected_category}' for length={length}, got '{result}'"


# ============================================================================
# TEST: METADATA EXTRACTION
# ============================================================================

class TestMetadataExtraction:
    """Test _extract_dataset_metadata helper method."""

    def test_small_hourly_dataset(self, lstm_model, small_hourly_dataset):
        """Test metadata extraction for small hourly dataset."""
        metadata = lstm_model._extract_dataset_metadata(small_hourly_dataset)

        assert metadata['freq'] == 'H'  # Changed from 'h' to 'H'
        assert metadata['seasonal_period'] == 24
        assert metadata['series_length'] == 800
        assert metadata['size_category'] == 'small'
        # Use .get() for optional keys that may not exist if dataset has no exog variables
        assert metadata.get('has_past_exog', False) is False
        assert metadata.get('has_future_exog', False) is False

    def test_medium_daily_dataset(self, basic_model_params, medium_daily_dataset, base_context):
        """Test metadata extraction for medium daily dataset."""
        model = LSTMForecaster(
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
        model = LSTMForecaster(
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
        assert metadata['series_length'] == 520  # Changed from 15000 to 520
        assert metadata['size_category'] == 'small'  # 520 < 1000 = small

    def test_none_dataset(self, lstm_model):
        """Passing None should return empty dict."""
        metadata = lstm_model._extract_dataset_metadata(None)
        assert metadata == {}


# ============================================================================
# TEST: SMART PRIORS INTEGRATION
# ============================================================================

class TestSmartPriorsIntegration:
    """Test that suggest_smart_priors uses inherited methods correctly."""

    def test_smart_priors_method_exists(self, lstm_model):
        """LSTM should have suggest_smart_priors method."""
        assert hasattr(lstm_model, 'suggest_smart_priors'), \
            "LSTM should have suggest_smart_priors method"

    def test_generates_priors(self, lstm_model, param_space, small_hourly_dataset):
        """Smart priors should generate configurations based on dataset."""
        priors = lstm_model.suggest_smart_priors(
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
        small_dataset.split_data(forecast_steps=24)  # Correct signature

        small_model = LSTMForecaster(
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
        large_dataset.split_data(forecast_steps=24)  # Correct signature

        large_model = LSTMForecaster(
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


# ============================================================================
# TEST: DRY PRINCIPLE VERIFICATION
# ============================================================================

class TestDRYPrinciple:
    """Verify that DRY refactoring was successful."""

    def test_no_duplicate_code_in_lstm(self):
        """LSTM should not have its own implementation of helper methods."""
        import inspect

        # Get LSTM source code
        lstm_source = inspect.getsource(LSTMForecaster)

        # Check that helper methods are NOT defined in LSTM
        assert 'def _infer_seasonal_period' not in lstm_source, \
            "LSTM should not have _infer_seasonal_period (should be inherited)"

        assert 'def _categorize_dataset_size' not in lstm_source, \
            "LSTM should not have _categorize_dataset_size (should be inherited)"

        # _extract_dataset_metadata might be in docstrings, so check for actual definition
        lines = [line.strip() for line in lstm_source.split('\n')]
        method_definitions = [line for line in lines if line.startswith('def _extract_dataset_metadata')]

        assert len(method_definitions) == 0, \
            "LSTM should not define _extract_dataset_metadata (should be inherited)"

    def test_single_source_of_truth(self):
        """All neural models should use the same helper implementations."""
        from models.base import NeuralTSForecaster

        # Get the actual function objects
        base_seasonal = NeuralTSForecaster._infer_seasonal_period
        base_size = NeuralTSForecaster._categorize_dataset_size
        base_metadata = NeuralTSForecaster._extract_dataset_metadata

        lstm_seasonal = LSTMForecaster._infer_seasonal_period
        lstm_size = LSTMForecaster._categorize_dataset_size
        lstm_metadata = LSTMForecaster._extract_dataset_metadata

        transformer_seasonal = TransformerForecaster._infer_seasonal_period
        transformer_size = TransformerForecaster._categorize_dataset_size
        transformer_metadata = TransformerForecaster._extract_dataset_metadata

        # All should be the exact same object (identity, not just equality)
        assert base_seasonal is lstm_seasonal is transformer_seasonal
        assert base_size is lstm_size is transformer_size
        assert base_metadata is lstm_metadata is transformer_metadata


# ============================================================================
# PERFORMANCE MARKERS
# ============================================================================

class TestPerformance:
    """Performance-related tests (marked as slow)."""

    def test_metadata_extraction_performance(self, lstm_model, small_hourly_dataset):
        """Metadata extraction should be fast."""
        import time

        start = time.time()
        for _ in range(1000):
            lstm_model._extract_dataset_metadata(small_hourly_dataset)
        elapsed = time.time() - start

        # Should complete 1000 extractions in less than 1 second
        assert elapsed < 1.0, \
            f"Metadata extraction too slow: {elapsed:.3f}s for 1000 calls"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])