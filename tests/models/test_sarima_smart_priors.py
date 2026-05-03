"""
Pytest Test Suite for SARIMA Smart Priors with Seasonal & Dataset Awareness
============================================================================

Tests that SARIMA properly uses seasonal and dataset-aware smart priors.

Run:
    pytest test_sarima_smart_priors.py -v
    pytest test_sarima_smart_priors.py::TestSmartPriorsIntegration -v
"""

import pytest
import pandas as pd
from utils.dataset import TimeSeriesDataset
from models.sarima import SARIMAForecaster
from models.base import TSForecaster


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def monthly_dataset():
    """Create a monthly dataset for testing (12 months × 10 years)."""
    df = pd.DataFrame(
        {'value': range(120)},
        index=pd.date_range('2020-01-01', periods=120, freq='ME')  # Month End
    )

    config = {"datasets": {"test": {"freq": "ME"}}}
    dataset = TimeSeriesDataset("test", config, num_features=1, data=df, columns=['value'], freq='ME')
    dataset.split_data(forecast_steps=12)

    return dataset


@pytest.fixture
def daily_dataset():
    """Create a daily dataset for testing (2 years)."""
    df = pd.DataFrame(
        {'value': range(730)},
        index=pd.date_range('2020-01-01', periods=730, freq='D')
    )

    config = {"datasets": {"test": {"freq": "D"}}}
    dataset = TimeSeriesDataset("test", config, num_features=1, data=df, columns=['value'], freq='D')
    dataset.split_data(forecast_steps=7)

    return dataset


@pytest.fixture
def quarterly_dataset():
    """Create a quarterly dataset for testing (10 years)."""
    df = pd.DataFrame(
        {'value': range(40)},
        index=pd.date_range('2020-01-01', periods=40, freq='QE')  # Quarter End
    )

    config = {"datasets": {"test": {"freq": "QE"}}}
    dataset = TimeSeriesDataset("test", config, num_features=1, data=df, columns=['value'], freq='QE')
    dataset.split_data(forecast_steps=4)

    return dataset


@pytest.fixture
def basic_model_params():
    """Basic SARIMA model parameters for testing."""
    return {
        "p": 1, "d": 1, "q": 1,
        "P": 1, "D": 1, "Q": 1,
        "s": 12
    }


@pytest.fixture
def sarima_model(monthly_dataset, basic_model_params, base_context):
    """Create a basic SARIMA model instance for testing."""
    return SARIMAForecaster(
        model_params=basic_model_params,
        num_features=1,
        forecast_steps=12,
        window_size=24,
        dataset=monthly_dataset,
        run_context=base_context
    )


@pytest.fixture
def param_space():
    """Standard hyperparameter search space for testing."""
    return {
        'p': [0, 1, 2, 3],
        'd': [0, 1, 2],
        'q': [0, 1, 2, 3],
        'P': [0, 1, 2],
        'D': [0, 1],
        'Q': [0, 1, 2],
        's': [4, 7, 12, 24, 52]
    }


# ============================================================================
# TEST: SEASONAL AWARENESS
# ============================================================================

class TestSeasonalAwareness:
    """Test that SARIMA infers and uses seasonal periods correctly."""

    def test_infers_monthly_seasonality(self, sarima_model, param_space, monthly_dataset):
        """Monthly data should infer s=12."""
        priors = sarima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=monthly_dataset
        )

        # At least one prior should have s=12
        has_monthly = any(p.get('s') == 12 for p in priors)
        assert has_monthly, "Monthly data should infer s=12"

    def test_infers_daily_seasonality(self, basic_model_params, param_space, daily_dataset, base_context):
        """Daily data should infer s=7."""
        model = SARIMAForecaster(
            model_params=basic_model_params,
            num_features=1,
            forecast_steps=7,
            window_size=14,
            dataset=daily_dataset,
            run_context=base_context
        )

        priors = model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=daily_dataset
        )

        # At least one prior should have s=7
        has_weekly = any(p.get('s') == 7 for p in priors)
        assert has_weekly, "Daily data should infer s=7"

    def test_infers_quarterly_seasonality(self, basic_model_params, param_space, quarterly_dataset, base_context):
        """Quarterly data should infer s=4."""
        model = SARIMAForecaster(
            model_params=basic_model_params,
            num_features=1,
            forecast_steps=4,
            window_size=8,
            dataset=quarterly_dataset,
            run_context=base_context
        )

        priors = model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=quarterly_dataset
        )

        # At least one prior should have s=4
        has_quarterly = any(p.get('s') == 4 for p in priors)
        assert has_quarterly, "Quarterly data should infer s=4"


# ============================================================================
# TEST: SMART PRIORS INTEGRATION
# ============================================================================

class TestSmartPriorsIntegration:
    """Test that suggest_smart_priors works correctly."""

    def test_smart_priors_method_exists(self, sarima_model):
        """SARIMA should have suggest_smart_priors method."""
        assert hasattr(sarima_model, 'suggest_smart_priors'), \
            "SARIMA should have suggest_smart_priors method"

    def test_generates_priors(self, sarima_model, param_space, monthly_dataset):
        """Smart priors should generate configurations."""
        priors = sarima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=monthly_dataset
        )

        assert priors is not None, "Should return priors"
        assert isinstance(priors, list), "Priors should be a list"
        assert len(priors) > 0, "Should return at least one prior"
        assert len(priors) <= 6, "Should return at most 6 priors"

    def test_priors_include_seasonal_params(self, sarima_model, param_space, monthly_dataset):
        """Priors should include seasonal SARIMA parameters."""
        priors = sarima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=monthly_dataset
        )

        # Check that priors include seasonal params
        has_P = any('P' in p for p in priors)
        has_D = any('D' in p for p in priors)
        has_Q = any('Q' in p for p in priors)
        has_s = any('s' in p for p in priors)

        assert has_P or has_D or has_Q or has_s, \
            "Priors should include seasonal parameters (P, D, Q, s)"

    def test_includes_simple_seasonal_model(self, sarima_model, param_space, monthly_dataset):
        """Priors should include simple seasonal model."""
        priors = sarima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=monthly_dataset
        )

        # Check for SARIMA(1,1,1)(1,1,1,s) or similar
        has_simple_seasonal = any(
            p.get('P', 0) >= 1 and p.get('D', 0) >= 1 and p.get('Q', 0) >= 1
            for p in priors
        )

        assert has_simple_seasonal, \
            "Priors should include simple seasonal model"

    def test_includes_non_seasonal_fallback(self, sarima_model, param_space, monthly_dataset):
        """Priors should include non-seasonal fallback."""
        priors = sarima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=monthly_dataset
        )

        # Check for model with P=0, D=0, Q=0 (no seasonal component)
        has_non_seasonal = any(
            p.get('P') == 0 and p.get('D') == 0 and p.get('Q') == 0
            for p in priors
        )

        assert has_non_seasonal, \
            "Priors should include non-seasonal fallback"

    def test_respects_fixed_params(self, sarima_model, param_space, monthly_dataset):
        """Fixed parameters should not appear in priors."""
        fixed_params = {'s': 12, 'D': 1}

        priors = sarima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params=fixed_params,
            dataset=monthly_dataset
        )

        # Fixed params should not appear in priors
        for prior in priors:
            assert 's' not in prior and 'D' not in prior, \
                f"Fixed parameters should not appear in priors, got {prior}"


# ============================================================================
# TEST: DATASET SIZE HANDLING
# ============================================================================

class TestDatasetSizeHandling:
    """Test that priors adapt to different dataset sizes."""

    def test_small_dataset_conservative_orders(self, basic_model_params, param_space, base_context):
        """Small datasets should get conservative seasonal orders."""
        small_df = pd.DataFrame(
            {'value': range(60)},
            index=pd.date_range('2020-01-01', periods=60, freq='ME')
        )
        small_dataset = TimeSeriesDataset("small", {}, num_features=1, data=small_df, columns=['value'], freq='ME')
        small_dataset.split_data(forecast_steps=12)

        model = SARIMAForecaster(
            model_params=basic_model_params,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=small_dataset,
            run_context=base_context
        )

        priors = model.suggest_smart_priors(param_space, {}, small_dataset)

        # All priors should have low seasonal orders (P+Q <= 2)
        for prior in priors:
            seasonal_order = prior.get('P', 0) + prior.get('Q', 0)
            assert seasonal_order <= 2, \
                f"Small dataset should have conservative seasonal orders, got P+Q={seasonal_order}"

    def test_no_dataset_fallback(self, sarima_model, param_space):
        """Should work without dataset (fallback)."""
        priors = sarima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=None
        )

        assert len(priors) > 0, "Should return priors even without dataset"


# ============================================================================
# TEST: PRIOR QUALITY
# ============================================================================

class TestPriorQuality:
    """Test the quality of generated priors."""

    def test_no_duplicate_priors(self, sarima_model, param_space, monthly_dataset):
        """Priors should be unique."""
        priors = sarima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=monthly_dataset
        )

        prior_tuples = [tuple(sorted(p.items())) for p in priors]
        assert len(prior_tuples) == len(set(prior_tuples)), \
            "Priors should not contain duplicates"

    def test_all_priors_non_empty(self, sarima_model, param_space, monthly_dataset):
        """All priors should be non-empty."""
        priors = sarima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=monthly_dataset
        )

        for i, prior in enumerate(priors, 1):
            assert len(prior) > 0, f"Prior {i} is empty"

    def test_parameter_values_valid(self, sarima_model, param_space, monthly_dataset):
        """Parameter values should be within param_space ranges."""
        priors = sarima_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=monthly_dataset
        )

        for prior in priors:
            for key, value in prior.items():
                if key in param_space:
                    valid_values = param_space[key]
                    assert value in valid_values, \
                        f"Parameter '{key}'={value} not in param_space {valid_values}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])