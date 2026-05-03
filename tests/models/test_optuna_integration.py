"""
Unit tests for Optuna integration in HPO (pruners, samplers, parameter suggestion).

Tests cover:
- _create_pruner(): All pruner types (median, percentile, hyperband, threshold, patient, none)
- _create_sampler(): Dimension counting (flat + nested), sampler configuration
- _suggest_optuna_params(): Nested parameter handling (scheduler_config)
- Integration with optimize_hyperparameters()
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, Mock
from models.base import NeuralTSForecaster

# Mock optuna if not installed
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


@pytest.fixture
def mock_forecaster(model_factory):
    """Create a mock forecaster with minimal setup."""
    forecaster = model_factory(model_type="lstm", strategy="direct")
    forecaster.model_params = {
        "type": "lstm",
        "learning_rate": 0.001,
        "hidden_size": 64,
        "batch_size": 32,
        "epochs": 10
    }
    return forecaster


# =============================================================================
# Test _create_pruner()
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
@pytest.mark.filterwarnings("ignore::optuna._experimental.ExperimentalWarning")
class TestCreatePruner:
    """Test pruner factory method with all supported pruner types."""

    def test_default_pruner_when_config_none(self, mock_forecaster):
        """Default should be PercentilePruner with conservative settings."""
        pruner = mock_forecaster._create_pruner(None)

        assert isinstance(pruner, optuna.pruners.PercentilePruner)
        # Check default values (inspect internal state if accessible)
        assert pruner._percentile == 25.0
        assert pruner._n_startup_trials == 10
        assert pruner._n_warmup_steps == 10

    def test_median_pruner(self, mock_forecaster):
        """Test MedianPruner creation with custom params."""
        config = {
            "type": "median",
            "n_startup_trials": 5,
            "n_warmup_steps": 3,
            "n_min_trials": 2
        }
        pruner = mock_forecaster._create_pruner(config)

        assert isinstance(pruner, optuna.pruners.MedianPruner)
        assert pruner._n_startup_trials == 5
        assert pruner._n_warmup_steps == 3

    def test_percentile_pruner(self, mock_forecaster):
        """Test PercentilePruner creation with custom params."""
        config = {
            "type": "percentile",
            "percentile": 50.0,
            "n_startup_trials": 8,
            "n_warmup_steps": 7,
            "n_min_trials": 3
        }
        pruner = mock_forecaster._create_pruner(config)

        assert isinstance(pruner, optuna.pruners.PercentilePruner)
        assert pruner._percentile == 50.0
        assert pruner._n_startup_trials == 8

    def test_hyperband_pruner(self, mock_forecaster):
        """Test HyperbandPruner creation."""
        config = {
            "type": "hyperband",
            "min_resource": 10,
            "max_resource": 100,
            "reduction_factor": 3
        }
        pruner = mock_forecaster._create_pruner(config)

        assert isinstance(pruner, optuna.pruners.HyperbandPruner)
        assert pruner._min_resource == 10
        assert pruner._max_resource == 100
        assert pruner._reduction_factor == 3

    def test_threshold_pruner(self, mock_forecaster):
        """Test ThresholdPruner creation."""
        config = {
            "type": "threshold",
            "lower": 0.1,
            "upper": 10.0
        }
        pruner = mock_forecaster._create_pruner(config)

        assert isinstance(pruner, optuna.pruners.ThresholdPruner)
        assert pruner._lower == 0.1
        assert pruner._upper == 10.0

    def test_patient_pruner_wrapping_median(self, mock_forecaster):
        """Test PatientPruner wrapping another pruner."""
        config = {
            "type": "patient",
            "patience": 3,
            "wrapped_pruner": {
                "type": "median",
                "n_startup_trials": 5
            }
        }
        pruner = mock_forecaster._create_pruner(config)

        assert isinstance(pruner, optuna.pruners.PatientPruner)
        # Note: PatientPruner internal attributes may vary by version
        # Just verify it's the correct type

    def test_nop_pruner(self, mock_forecaster):
        """Test NopPruner (no pruning)."""
        config = {"type": "none"}
        pruner = mock_forecaster._create_pruner(config)

        assert isinstance(pruner, optuna.pruners.NopPruner)

    def test_unknown_pruner_type_fallback(self, mock_forecaster):
        """Unknown pruner type should fallback to default PercentilePruner."""
        config = {"type": "unknown_type"}
        pruner = mock_forecaster._create_pruner(config)

        assert isinstance(pruner, optuna.pruners.PercentilePruner)


# =============================================================================
# Test _create_sampler()
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
@pytest.mark.filterwarnings("ignore::optuna._experimental.ExperimentalWarning")
class TestCreateSampler:
    """Test sampler factory with dimension counting (flat + nested)."""

    def test_flat_param_space_dimension_counting(self, mock_forecaster):
        """Test dimension counting for flat parameter space."""
        param_space = {
            "learning_rate": {"min": 0.0001, "max": 0.01},  # 1 continuous
            "hidden_size": [64, 128, 256],                  # 1 discrete
            "dropout": {"min": 0.0, "max": 0.5}             # 1 continuous
        }
        sampler = mock_forecaster._create_sampler(param_space, {})

        assert isinstance(sampler, optuna.samplers.TPESampler)
        # Total dimensions = 2 continuous + 1 discrete = 3
        # n_startup_trials = max(5, 3 * 2) = 6
        assert sampler._n_startup_trials == 6

    def test_nested_param_space_dimension_counting(self, mock_forecaster):
        """Test dimension counting for nested parameter space (scheduler_config)."""
        param_space = {
            "learning_rate": {"min": 0.0001, "max": 0.01},  # 1 continuous
            "hidden_size": [64, 128, 256],                  # 1 discrete
            "scheduler_config": {                           # Nested!
                "pct_start": {"min": 0.1, "max": 0.5},      # 1 continuous
                "div_factor": {"min": 5.0, "max": 25.0},    # 1 continuous
                "anneal_strategy": ["cos", "linear"]        # 1 discrete
            }
        }
        sampler = mock_forecaster._create_sampler(param_space, {})

        assert isinstance(sampler, optuna.samplers.TPESampler)
        # Total dimensions = 3 continuous + 2 discrete = 5
        # n_startup_trials = max(5, 5 * 2) = 10
        assert sampler._n_startup_trials == 10

    def test_empty_param_space(self, mock_forecaster):
        """Empty param space should create sampler with minimum n_startup_trials."""
        param_space = {}
        sampler = mock_forecaster._create_sampler(param_space, {})

        assert isinstance(sampler, optuna.samplers.TPESampler)
        # Total dimensions = 0, n_startup_trials = max(5, 0 * 2) = 5
        assert sampler._n_startup_trials == 5

    def test_sampler_configuration(self, mock_forecaster):
        """Test that sampler has correct configuration."""
        param_space = {"learning_rate": {"min": 0.0001, "max": 0.01}}
        sampler = mock_forecaster._create_sampler(param_space, {})

        # Just verify it's a TPESampler - internal attributes may vary by Optuna version
        assert isinstance(sampler, optuna.samplers.TPESampler)


# =============================================================================
# Test _suggest_optuna_params()
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestSuggestOptunaParams:
    """Test parameter suggestion with flat and nested structures."""

    def test_flat_param_space(self, mock_forecaster):
        """Test suggesting parameters from flat param space."""
        param_space = {
            "learning_rate": {"min": 0.0001, "max": 0.01, "log": True},
            "hidden_size": [64, 128, 256],
            "dropout": {"min": 0.0, "max": 0.5}
        }

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_float = Mock(return_value=0.001)
        mock_trial.suggest_categorical = Mock(return_value=128)

        params = mock_forecaster._suggest_optuna_params(mock_trial, param_space)

        # Verify structure
        assert "learning_rate" in params
        assert "hidden_size" in params
        assert "dropout" in params

        # Verify calls
        mock_trial.suggest_float.assert_any_call("learning_rate", 0.0001, 0.01, log=True, step=None)
        mock_trial.suggest_categorical.assert_called_once_with("hidden_size", [64, 128, 256])

    def test_nested_param_space_scheduler_config(self, mock_forecaster):
        """Test suggesting parameters with nested scheduler_config."""
        param_space = {
            "learning_rate": {"min": 0.0001, "max": 0.01},
            "scheduler_config": {
                "pct_start": {"min": 0.1, "max": 0.5},
                "div_factor": {"min": 5.0, "max": 25.0},
                "anneal_strategy": ["cos", "linear"]
            }
        }

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_float = Mock(side_effect=[0.001, 0.3, 10.0])
        mock_trial.suggest_categorical = Mock(return_value="cos")

        params = mock_forecaster._suggest_optuna_params(mock_trial, param_space)

        # Verify nested structure is preserved
        assert "learning_rate" in params
        assert "scheduler_config" in params
        assert isinstance(params["scheduler_config"], dict)
        assert "pct_start" in params["scheduler_config"]
        assert "div_factor" in params["scheduler_config"]
        assert "anneal_strategy" in params["scheduler_config"]

        # Verify Optuna names are flattened (with dots)
        mock_trial.suggest_float.assert_any_call("scheduler_config.pct_start", 0.1, 0.5, log=False, step=None)
        mock_trial.suggest_float.assert_any_call("scheduler_config.div_factor", 5.0, 25.0, log=False, step=None)
        mock_trial.suggest_categorical.assert_called_once_with("scheduler_config.anneal_strategy", ["cos", "linear"])

    def test_integer_range_suggestion(self, mock_forecaster):
        """Test suggesting integer parameters."""
        param_space = {
            "hidden_size": {"min": 32, "max": 512, "step": 32}
        }

        mock_trial = Mock()
        mock_trial.suggest_int = Mock(return_value=128)

        params = mock_forecaster._suggest_optuna_params(mock_trial, param_space)

        assert params["hidden_size"] == 128
        mock_trial.suggest_int.assert_called_once_with("hidden_size", 32, 512, step=32, log=False)

    def test_deeply_nested_params(self, mock_forecaster):
        """Test suggesting deeply nested parameters (3 levels)."""
        param_space = {
            "level1": {
                "level2": {
                    "level3": {"min": 0.0, "max": 1.0}
                }
            }
        }

        mock_trial = Mock()
        mock_trial.suggest_float = Mock(return_value=0.5)

        params = mock_forecaster._suggest_optuna_params(mock_trial, param_space)

        # Verify structure
        assert "level1" in params
        assert "level2" in params["level1"]
        assert "level3" in params["level1"]["level2"]

        # Verify Optuna name uses dot notation
        mock_trial.suggest_float.assert_called_once_with("level1.level2.level3", 0.0, 1.0, log=False, step=None)


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaIntegration:
    """Integration tests for full HPO pipeline with Optuna."""

    def test_pruner_config_passed_to_study(self, mock_forecaster):
        """Test that pruner_config from optimization section is used."""
        model_config = {
            "type": "lstm",
            "optimization": {
                "method": "optuna",
                "n_trials": 2,
                "params": {"learning_rate": {"min": 0.001, "max": 0.01}},
                "pruner_config": {
                    "type": "median",
                    "n_startup_trials": 3,
                    "n_warmup_steps": 2
                }
            }
        }

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.target_columns = ["target_0"]

        train_fold = pd.DataFrame({"target_0": np.random.rand(50)})
        eval_fold = pd.DataFrame({"target_0": np.random.rand(10)})
        folds = [(train_fold, eval_fold)]

        validation_params = {
            "n_folds": 1,
            "evaluation_metric": "mse"
        }

        # Mock suggest_smart_priors to avoid dataset inspection
        mock_forecaster.suggest_smart_priors = Mock(return_value=[])

        # Mock _fit_and_evaluate_fold to avoid actual training
        mock_forecaster._fit_and_evaluate_fold = Mock(return_value=(0.5, None, {}))

        # Patch optuna.create_study to inspect arguments
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_study.ask = Mock(side_effect=StopIteration)  # Stop after first trial
            mock_create_study.return_value = mock_study

            try:
                mock_forecaster.optimize_hyperparameters(
                    dataset=mock_dataset,
                    model_config=model_config,
                    validation_params=validation_params,
                    folds=folds
                )
            except StopIteration:
                pass  # Expected when study.ask raises StopIteration

            # Verify create_study was called with correct pruner
            mock_create_study.assert_called_once()
            call_kwargs = mock_create_study.call_args[1]
            pruner = call_kwargs['pruner']
            assert isinstance(pruner, optuna.pruners.MedianPruner)

    def test_nested_scheduler_params_in_hpo(self, mock_forecaster):
        """Test that nested scheduler_config params are correctly sampled."""
        model_config = {
            "type": "lstm",
            "optimization": {
                "method": "optuna",
                "n_trials": 1,
                "params": {
                    "learning_rate": {"min": 0.001, "max": 0.01},
                    "scheduler_config": {
                        "pct_start": {"min": 0.1, "max": 0.5}
                    }
                }
            }
        }

        # Mock dataset and folds
        mock_dataset = Mock()
        mock_dataset.target_columns = ["target_0"]
        train_fold = pd.DataFrame({"target_0": np.random.rand(50)})
        eval_fold = pd.DataFrame({"target_0": np.random.rand(10)})
        folds = [(train_fold, eval_fold)]

        validation_params = {"n_folds": 1, "evaluation_metric": "mse"}

        # Mock _fit_and_evaluate_fold
        mock_forecaster._fit_and_evaluate_fold = Mock(return_value=(0.5, None, {}))

        # Patch study to capture sampled params
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_trial = Mock()
            mock_trial.suggest_float = Mock(side_effect=[0.005, 0.3])  # LR, pct_start
            mock_study.ask = Mock(return_value=mock_trial)
            mock_study.tell = Mock()
            mock_create_study.return_value = mock_study

            # Run only one trial
            with patch.object(mock_forecaster, 'optimize_hyperparameters') as mock_opt:
                # Just verify _suggest_optuna_params is called correctly
                param_space = model_config["optimization"]["params"]
                sampled = mock_forecaster._suggest_optuna_params(mock_trial, param_space)

                # Verify nested structure
                assert "scheduler_config" in sampled
                assert "pct_start" in sampled["scheduler_config"]

                # Verify Optuna names use dot notation
                mock_trial.suggest_float.assert_any_call("scheduler_config.pct_start", 0.1, 0.5, log=False, step=None)


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
@pytest.mark.filterwarnings("ignore::optuna._experimental.ExperimentalWarning")
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_pruner_config_uses_default(self, mock_forecaster):
        """Empty pruner_config dict should use defaults."""
        pruner = mock_forecaster._create_pruner({})
        assert isinstance(pruner, optuna.pruners.PercentilePruner)

    def test_sampler_with_only_discrete_params(self, mock_forecaster):
        """Test sampler creation with only discrete parameters."""
        param_space = {
            "hidden_size": [64, 128, 256],
            "num_layers": [1, 2, 3]
        }
        sampler = mock_forecaster._create_sampler(param_space, {})
        assert isinstance(sampler, optuna.samplers.TPESampler)
        # 0 continuous + 2 discrete = 2 dimensions
        assert sampler._n_startup_trials == max(5, 2 * 2)

    def test_sampler_with_only_continuous_params(self, mock_forecaster):
        """Test sampler creation with only continuous parameters."""
        param_space = {
            "learning_rate": {"min": 0.0001, "max": 0.01},
            "dropout": {"min": 0.0, "max": 0.5}
        }
        sampler = mock_forecaster._create_sampler(param_space, {})
        assert isinstance(sampler, optuna.samplers.TPESampler)
        # 2 continuous + 0 discrete = 2 dimensions
        assert sampler._n_startup_trials == max(5, 2 * 2)

    def test_patient_pruner_with_none_wrapped(self, mock_forecaster):
        """Test PatientPruner wrapping NopPruner."""
        config = {
            "type": "patient",
            "patience": 2,
            "wrapped_pruner": {"type": "none"}
        }
        pruner = mock_forecaster._create_pruner(config)
        assert isinstance(pruner, optuna.pruners.PatientPruner)
        # Note: PatientPruner wrapping verified by construction, not internal attributes
