"""
Pytest Test Suite for VAR Smart Priors with Multivariate & Dataset Awareness
==============================================================================

Tests that VAR properly uses multivariate and dataset-aware smart priors.

Run:
    pytest test_var_smart_priors.py -v
    pytest test_var_smart_priors.py::TestSmartPriorsIntegration -v
"""

import pytest
import pandas as pd
import numpy as np
from utils.dataset import TimeSeriesDataset
from models.var import VARForecaster


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def small_multivariate_dataset():
    """Create a small multivariate dataset (300 points, 3 variables)."""
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(300, 3),
        columns=['var1', 'var2', 'var3'],
        index=pd.date_range('2020-01-01', periods=300, freq='D')
    )

    config = {"datasets": {"test": {"freq": "D"}}}
    dataset = TimeSeriesDataset("test", config, num_features=3, data=df, columns=['var1', 'var2', 'var3'], freq='D')
    dataset.split_data(forecast_steps=7)

    return dataset


@pytest.fixture
def large_multivariate_dataset():
    """Create a large multivariate dataset (2000 points, 5 variables)."""
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(2000, 5),
        columns=['var1', 'var2', 'var3', 'var4', 'var5'],
        index=pd.date_range('2020-01-01', periods=2000, freq='D')
    )

    config = {"datasets": {"test": {"freq": "D"}}}
    dataset = TimeSeriesDataset("test", config, num_features=5, data=df,
                                columns=['var1', 'var2', 'var3', 'var4', 'var5'], freq='D')
    dataset.split_data(forecast_steps=7)

    return dataset


@pytest.fixture
def high_dimensional_dataset():
    """Create a high-dimensional dataset (1000 points, 10 variables)."""
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(1000, 10),
        columns=[f'var{i}' for i in range(1, 11)],
        index=pd.date_range('2020-01-01', periods=1000, freq='D')
    )

    config = {"datasets": {"test": {"freq": "D"}}}
    dataset = TimeSeriesDataset("test", config, num_features=10, data=df,
                                columns=[f'var{i}' for i in range(1, 11)], freq='D')
    dataset.split_data(forecast_steps=7)

    return dataset


@pytest.fixture
def basic_model_params():
    """Basic VAR model parameters for testing."""
    return {
        "max_lags": 3,
        "ic": "aic"
    }


@pytest.fixture
def var_model(small_multivariate_dataset, basic_model_params, base_context):
    """Create a basic VAR model instance for testing."""
    return VARForecaster(
        model_params=basic_model_params,
        num_features=3,
        forecast_steps=7,
        window_size=14,
        dataset=small_multivariate_dataset,
        run_context=base_context
    )


@pytest.fixture
def param_space():
    """Standard hyperparameter search space for testing."""
    return {
        'max_lags': [1, 2, 3, 4, 5, 6, 8, 10],
        'ic': ['aic', 'bic', 'hqic']
    }


# ============================================================================
# TEST: MULTIVARIATE AWARENESS
# ============================================================================

class TestMultivariateAwareness:
    """Test that VAR adapts to number of variables."""

    def test_few_variables_higher_lags(self, basic_model_params, param_space, large_multivariate_dataset, base_context):
        """Few variables with large dataset should allow higher lag orders."""
        # 5 variables, 2000 points
        model = VARForecaster(
            model_params=basic_model_params,
            num_features=5,
            forecast_steps=7,
            window_size=14,  # Added
            dataset=large_multivariate_dataset,
            run_context=base_context
        )

        priors = model.suggest_smart_priors(param_space, {}, large_multivariate_dataset)

        # Should include priors with lag >= 3
        max_lag = max(p.get('max_lags', 0) for p in priors)
        assert max_lag >= 3, \
            f"Few variables with large dataset should allow higher lags, got max={max_lag}"

    def test_many_variables_conservative_lags(self, basic_model_params, param_space, high_dimensional_dataset, base_context):
        """Many variables should get conservative lag orders."""
        # 10 variables, 1000 points
        model = VARForecaster(
            model_params=basic_model_params,
            num_features=10,
            forecast_steps=7,
            window_size=14,  # Added
            dataset=high_dimensional_dataset,
            run_context=base_context
        )

        priors = model.suggest_smart_priors(param_space, {}, high_dimensional_dataset)

        # All lags should be conservative (max_lags <= 4)
        for prior in priors:
            lag = prior.get('max_lags', 1)
            assert lag <= 4, \
                f"Many variables should have conservative lags, got max_lags={lag}"


# ============================================================================
# TEST: SMART PRIORS INTEGRATION
# ============================================================================

class TestSmartPriorsIntegration:
    """Test that suggest_smart_priors works correctly."""

    def test_smart_priors_method_exists(self, var_model):
        """VAR should have suggest_smart_priors method."""
        assert hasattr(var_model, 'suggest_smart_priors'), \
            "VAR should have suggest_smart_priors method"

    def test_generates_priors(self, var_model, param_space, small_multivariate_dataset):
        """Smart priors should generate configurations."""
        priors = var_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_multivariate_dataset
        )

        assert priors is not None, "Should return priors"
        assert isinstance(priors, list), "Priors should be a list"
        assert len(priors) > 0, "Should return at least one prior"
        assert len(priors) <= 6, "Should return at most 6 priors"

    def test_priors_include_var_params(self, var_model, param_space, small_multivariate_dataset):
        """Priors should include VAR-specific parameters."""
        priors = var_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_multivariate_dataset
        )

        # Check that priors include VAR params
        has_max_lags = any('max_lags' in p for p in priors)
        has_ic = any('ic' in p for p in priors)

        assert has_max_lags or has_ic, \
            "Priors should include VAR parameters (max_lags, ic)"

    def test_starts_with_var1(self, var_model, param_space, small_multivariate_dataset):
        """First prior should be VAR(1)."""
        priors = var_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_multivariate_dataset
        )

        # First prior should have max_lags=1
        if len(priors) > 0:
            first_lag = priors[0].get('max_lags', None)
            assert first_lag == 1, \
                f"First prior should be VAR(1), got max_lags={first_lag}"

    def test_includes_bic_for_small_datasets(self, basic_model_params, param_space, small_multivariate_dataset, base_context):
        """Small datasets should prioritize BIC (penalizes complexity)."""
        model = VARForecaster(
            model_params=basic_model_params,
            num_features=3,
            forecast_steps=7,
            window_size=14,  # Added
            dataset=small_multivariate_dataset,
            run_context=base_context
        )

        priors = model.suggest_smart_priors(param_space, {}, small_multivariate_dataset)

        # Should include BIC in priors
        has_bic = any(p.get('ic') == 'bic' for p in priors)
        assert has_bic, "Small dataset should prioritize BIC"

    def test_includes_aic_for_large_datasets(self, basic_model_params, param_space, large_multivariate_dataset, base_context):
        """Large datasets should prioritize AIC."""
        model = VARForecaster(
            model_params=basic_model_params,
            num_features=5,
            forecast_steps=7,
            window_size=14,  # Added
            dataset=large_multivariate_dataset,
            run_context=base_context
        )

        priors = model.suggest_smart_priors(param_space, {}, large_multivariate_dataset)

        # Should include AIC in priors
        has_aic = any(p.get('ic') == 'aic' for p in priors)
        assert has_aic, "Large dataset should prioritize AIC"

    def test_respects_fixed_params(self, var_model, param_space, small_multivariate_dataset):
        """Fixed parameters should not appear in priors."""
        fixed_params = {'ic': 'bic'}

        priors = var_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params=fixed_params,
            dataset=small_multivariate_dataset
        )

        # 'ic' should not appear in priors
        for prior in priors:
            assert 'ic' not in prior, \
                f"Fixed parameter 'ic' should not appear in priors, got {prior}"


# ============================================================================
# TEST: DATASET SIZE HANDLING
# ============================================================================

class TestDatasetSizeHandling:
    """Test that priors adapt to different dataset sizes."""

    def test_tiny_dataset_only_var1(self, basic_model_params, param_space, base_context):
        """Tiny datasets should only have VAR(1) (max_safe_lag=1)."""
        np.random.seed(42)
        tiny_df = pd.DataFrame(
            np.random.randn(50, 3),
            columns=['var1', 'var2', 'var3'],
            index=pd.date_range('2020-01-01', periods=50, freq='D')
        )
        tiny_dataset = TimeSeriesDataset("tiny", {}, num_features=3, data=tiny_df,
                                         columns=['var1', 'var2', 'var3'], freq='D')
        tiny_dataset.split_data(forecast_steps=7)

        model = VARForecaster(
            model_params=basic_model_params,
            num_features=3,
            forecast_steps=7,
            window_size=10,  # Added - smaller for tiny dataset
            dataset=tiny_dataset,
            run_context=base_context
        )

        priors = model.suggest_smart_priors(param_space, {}, tiny_dataset)

        # Tiny dataset: max_safe_lag=1, so only VAR(1) should be generated
        for prior in priors:
            lag = prior.get('max_lags', 1)
            assert lag == 1, \
                f"Tiny dataset (50 points) should only have VAR(1), got max_lags={lag}"

    def test_small_dataset_allows_var2(self, basic_model_params, param_space, base_context):
        """Small datasets should allow up to VAR(2)."""
        np.random.seed(42)
        small_df = pd.DataFrame(
            np.random.randn(150, 3),
            columns=['var1', 'var2', 'var3'],
            index=pd.date_range('2020-01-01', periods=150, freq='D')
        )
        small_dataset = TimeSeriesDataset("small", {}, num_features=3, data=small_df,
                                          columns=['var1', 'var2', 'var3'], freq='D')
        small_dataset.split_data(forecast_steps=7)

        model = VARForecaster(
            model_params=basic_model_params,
            num_features=3,
            forecast_steps=7,
            window_size=14,
            dataset=small_dataset,
            run_context=base_context
        )

        priors = model.suggest_smart_priors(param_space, {}, small_dataset)

        # Small dataset: max_safe_lag=2, should include VAR(2)
        max_lag = max(p.get('max_lags', 1) for p in priors)
        assert max_lag == 2, \
            f"Small dataset should allow VAR(2), got max_lag={max_lag}"

    def test_no_dataset_fallback(self, var_model, param_space):
        """Should work without dataset (fallback)."""
        priors = var_model.suggest_smart_priors(
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

    def test_no_duplicate_priors(self, var_model, param_space, small_multivariate_dataset):
        """Priors should be unique."""
        priors = var_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_multivariate_dataset
        )

        prior_tuples = [tuple(sorted(p.items())) for p in priors]
        assert len(prior_tuples) == len(set(prior_tuples)), \
            "Priors should not contain duplicates"

    def test_all_priors_non_empty(self, var_model, param_space, small_multivariate_dataset):
        """All priors should be non-empty."""
        priors = var_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_multivariate_dataset
        )

        for i, prior in enumerate(priors, 1):
            assert len(prior) > 0, f"Prior {i} is empty"

    def test_parameter_values_valid(self, var_model, param_space, small_multivariate_dataset):
        """Parameter values should be within param_space ranges."""
        priors = var_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_multivariate_dataset
        )

        for prior in priors:
            for key, value in prior.items():
                if key in param_space:
                    valid_values = param_space[key]
                    assert value in valid_values, \
                        f"Parameter '{key}'={value} not in param_space {valid_values}"

    def test_conservative_progression(self, var_model, param_space, small_multivariate_dataset):
        """Priors should progress conservatively (low lag first)."""
        priors = var_model.suggest_smart_priors(
            param_space=param_space,
            fixed_params={},
            dataset=small_multivariate_dataset
        )

        # Extract lags
        lags = [p.get('max_lags', 1) for p in priors if 'max_lags' in p]

        if len(lags) >= 2:
            # First lags should be lower than later lags (generally)
            assert lags[0] <= lags[1], \
                f"Priors should start conservative, got lags={lags}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
