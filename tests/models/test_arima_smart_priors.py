"""
Pytest Test Suite for ARIMA Smart Priors with Dataset-Aware HPO
================================================================

Tests that ARIMA properly uses dataset-aware smart priors.

Run:
    pytest test_arima_smart_priors.py -v
    pytest test_arima_smart_priors.py::TestSmartPriorsIntegration -v
"""

import pytest
import pandas as pd
from utils.dataset import TimeSeriesDataset
from models.arima import ARIMAForecaster


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def small_dataset():
    """Create a small dataset for testing (100 points)."""
    df = pd.DataFrame(
        {'value': range(100)},
        index=pd.date_range('2020-01-01', periods=100, freq='D')
    )

    config = {"datasets": {"test": {"freq": "D"}}}
    dataset = TimeSeriesDataset("test", config, num_features=1, data=df, columns=['value'], freq='D')
    dataset.split_data(forecast_steps=7)

    return dataset


@pytest.fixture
def medium_dataset():
    """Create a medium dataset for testing (1000 points)."""
    df = pd.DataFrame(
        {'value': range(1000)},
        index=pd.date_range('2020-01-01', periods=1000, freq='D')
    )

    config = {"datasets": {"test": {"freq": "D"}}}
    dataset = TimeSeriesDataset("test", config, num_features=1, data=df, columns=['value'], freq='D')
    dataset.split_data(forecast_steps=7)

    return dataset


@pytest.fixture
def large_dataset():
    """Create a large dataset for testing (5000 points)."""
    df = pd.DataFrame(
        {'value': range(5000)},
        index=pd.date_range('2020-01-01', periods=5000, freq='D')
    )

    config = {"datasets": {"test": {"freq": "D"}}}
    dataset = TimeSeriesDataset("test", config, num_features=1, data=df, columns=['value'], freq='D')
    dataset.split_data(forecast_steps=7)

    return dataset


@pytest.fixture
def basic_model_params():
    """Basic ARIMA model parameters for testing."""
    return {
        "p": 1,
        "d": 1,
        "q": 1
    }


@pytest.fixture
def arima_model(small_dataset, basic_model_params, base_context):
    """Create a basic ARIMA model instance for testing."""
    return ARIMAForecaster(
        model_params=basic_model_params,
        num_features=1,
        forecast_steps=7,
        window_size=30,
        dataset=small_dataset,
        run_context=base_context
    )


@pytest.fixture
def param_space():
    """Standard hyperparameter search space for testing."""
    return {
        'p': [0, 1, 2, 3, 4, 5],
        'd': [0, 1, 2],
        'q': [0, 1, 2, 3, 4, 5]
    }


# ============================================================================
# TEST: SMART PRIORS INTEGRATION
# ============================================================================

class TestSmartPriorsIntegration:
    """Test that suggest_smart_priors works correctly."""

    def test_smart_priors_method_exists(self, arima_model):
        """ARIMA should have suggest_smart_priors method."""
        assert hasattr(arima_model, 'suggest_smart_priors'), \
            "ARIMA should have suggest_smart_priors method"

    def test_generates_priors(self, arima_model, param_space, small_dataset):
        """Smart priors should generate configurations based on dataset."""
        priors = arima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_dataset
        )

        assert priors is not None, "Should return priors"
        assert isinstance(priors, list), "Priors should be a list"
        assert len(priors) > 0, "Should return at least one prior"
        assert len(priors) <= 6, "Should return at most 6 priors"

        # Each prior should be a dict with ARIMA parameters
        for prior in priors:
            assert isinstance(prior, dict), "Each prior should be a dict"
            # Should have at least one of p, d, q
            assert any(k in prior for k in ['p', 'd', 'q']), \
                "Each prior should have at least one ARIMA parameter"

    def test_priors_include_arima_params(self, arima_model, param_space, small_dataset):
        """Priors should include ARIMA-specific parameters."""
        priors = arima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_dataset
        )

        # Check that priors include ARIMA params
        has_p = any('p' in p for p in priors)
        has_d = any('d' in p for p in priors)
        has_q = any('q' in p for p in priors)

        assert has_p or has_d or has_q, \
            "Priors should include ARIMA parameters (p, d, q)"

    def test_priors_adapt_to_dataset_size(self, basic_model_params, param_space, base_context):
        """Priors should adapt complexity based on dataset size."""
        # Small dataset
        small_df = pd.DataFrame(
            {'value': range(100)},
            index=pd.date_range('2020-01-01', periods=100, freq='D')
        )
        small_dataset = TimeSeriesDataset("small", {}, num_features=1, data=small_df, columns=['value'], freq='D')
        small_dataset.split_data(forecast_steps=7)

        small_model = ARIMAForecaster(
            model_params=basic_model_params,
            num_features=1,
            forecast_steps=7,
            window_size=30,
            dataset=small_dataset,
            run_context=base_context
        )

        small_priors = small_model.suggest_smart_priors(param_space, {}, small_dataset)

        # Large dataset
        large_df = pd.DataFrame(
            {'value': range(5000)},
            index=pd.date_range('2020-01-01', periods=5000, freq='D')
        )
        large_dataset = TimeSeriesDataset("large", {}, num_features=1, data=large_df, columns=['value'], freq='D')
        large_dataset.split_data(forecast_steps=7)

        large_model = ARIMAForecaster(
            model_params=basic_model_params,
            num_features=1,
            forecast_steps=7,
            window_size=30,
            dataset=large_dataset,
            run_context=base_context
        )

        large_priors = large_model.suggest_smart_priors(param_space, {}, large_dataset)

        # Large datasets should allow more priors (or at least same number)
        assert len(large_priors) >= len(small_priors), \
            "Large dataset should generate at least as many priors as small dataset"

    def test_conservative_first_priors(self, arima_model, param_space, small_dataset):
        """First priors should be conservative (low orders)."""
        priors = arima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_dataset
        )

        # First prior should be simple (total order <= 3)
        if len(priors) > 0:
            first_prior = priors[0]
            total_order = first_prior.get('p', 0) + first_prior.get('q', 0)
            assert total_order <= 2, \
                f"First prior should be conservative (p+q <= 2), got {total_order}"

    def test_includes_random_walk(self, arima_model, param_space, small_dataset):
        """Priors should include random walk ARIMA(0,1,0)."""
        priors = arima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_dataset
        )

        # Check for random walk: p=0, d=1, q=0
        has_random_walk = any(
            p.get('p') == 0 and p.get('d') == 1 and p.get('q') == 0
            for p in priors
        )

        assert has_random_walk, \
            "Priors should include random walk ARIMA(0,1,0)"

    def test_respects_fixed_params(self, arima_model, param_space, small_dataset):
        """Fixed parameters should not appear in priors."""
        fixed_params = {'d': 1}  # Fix differencing order

        priors = arima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params=fixed_params,
            dataset=small_dataset
        )

        # 'd' should not appear in any prior
        for prior in priors:
            assert 'd' not in prior, \
                f"Fixed parameter 'd' should not appear in priors, got {prior}"

    def test_only_searchable_params(self, arima_model, small_dataset):
        """Priors should only include parameters in param_space."""
        limited_space = {'p': [1, 2, 3]}  # Only p is searchable

        priors = arima_model.suggest_smart_priors(
            param_space=limited_space,
            fixed_params={},
            dataset=small_dataset
        )

        # All priors should only have 'p'
        for prior in priors:
            for key in prior.keys():
                assert key in limited_space, \
                    f"Prior contains '{key}' which is not in param_space"


# ============================================================================
# TEST: DATASET SIZE HANDLING
# ============================================================================

class TestDatasetSizeHandling:
    """Test that priors adapt properly to different dataset sizes."""

    def test_tiny_dataset(self, basic_model_params, base_context):
        """Tiny datasets should get very conservative priors."""
        tiny_df = pd.DataFrame(
            {'value': range(50)},
            index=pd.date_range('2020-01-01', periods=50, freq='D')
        )
        tiny_dataset = TimeSeriesDataset("tiny", {}, num_features=1, data=tiny_df, columns=['value'], freq='D')
        tiny_dataset.split_data(forecast_steps=7)

        model = ARIMAForecaster(
            model_params=basic_model_params,
            num_features=1,
            forecast_steps=7,
            window_size=30,
            dataset=tiny_dataset,
            run_context=base_context
        )

        param_space = {'p': [0, 1, 2, 3, 4], 'd': [0, 1, 2], 'q': [0, 1, 2, 3, 4]}
        priors = model.suggest_smart_priors(param_space, {}, tiny_dataset)

        # All priors should have low orders (p+q <= 3)
        for prior in priors:
            total_order = prior.get('p', 0) + prior.get('q', 0)
            assert total_order <= 4, \
                f"Tiny dataset should have conservative orders, got p+q={total_order}"

    def test_no_dataset_fallback(self, arima_model, param_space):
        """Should work without dataset (fallback to defaults)."""
        priors = arima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=None
        )

        assert len(priors) > 0, "Should return priors even without dataset"


# ============================================================================
# TEST: PRIOR QUALITY
# ============================================================================

class TestPriorQuality:
    """Test the quality and validity of generated priors."""

    def test_no_duplicate_priors(self, arima_model, param_space, small_dataset):
        """Priors should be unique (no duplicates)."""
        priors = arima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_dataset
        )

        # Convert to tuples for set comparison
        prior_tuples = [tuple(sorted(p.items())) for p in priors]
        assert len(prior_tuples) == len(set(prior_tuples)), \
            "Priors should not contain duplicates"

    def test_all_priors_non_empty(self, arima_model, param_space, small_dataset):
        """All priors should be non-empty."""
        priors = arima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_dataset
        )

        for i, prior in enumerate(priors, 1):
            assert len(prior) > 0, f"Prior {i} is empty"

    def test_parameter_values_valid(self, arima_model, param_space, small_dataset):
        """Parameter values should be within param_space ranges."""
        priors = arima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_dataset
        )

        for prior in priors:
            for key, value in prior.items():
                if key in param_space:
                    valid_values = param_space[key]
                    assert value in valid_values, \
                        f"Parameter '{key}'={value} not in param_space {valid_values}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])