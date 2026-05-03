"""
Integration tests for AuxiliaryMultiStepLoss and prediction noise in Transformer.

Tests verify that the new training enhancements work correctly with the full
Transformer training pipeline.
"""

import pytest
import torch
import numpy as np
import pandas as pd

from models.factory import ModelFactory
from utils.dataset import TimeSeriesDataset

# Import shared fixtures and constants
from tests.models.conftest import WINDOW_SIZE


@pytest.fixture
def simple_dataset():
    """Create minimal synthetic dataset for integration testing."""
    np.random.seed(42)
    n_samples = 100

    time_values = np.linspace(0, 10, n_samples)
    target = np.sin(time_values) + np.random.normal(0, 0.1, n_samples)

    data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'target': target,
        'enc_exog': np.random.rand(n_samples)
    })

    ds = TimeSeriesDataset(
        "simple_test",
        {"datasets": {"simple_test": {}}},
        num_features=1,
        data=data,
        columns=['target'],
        past_covariates=['enc_exog'],
        future_covariates=[]
    )
    ds.split_data(forecast_steps=5)
    return ds


class TestTransformerAuxiliaryLossIntegration:
    """Integration tests for auxiliary multi-step loss."""

    def test_training_with_auxiliary_loss_enabled(
        self,
        base_transformer_config,
        simple_dataset,
        base_context
    ):
        """Test that training runs successfully with auxiliary loss enabled."""
        config = {
            **base_transformer_config,
            "architecture": "encoder-decoder",
            "strategy": "direct",
            "auxiliary_loss": {
                "enabled": True,
                "weight": 0.15,
                "position_weighting": True
            }
        }

        forecaster = ModelFactory.create(
            "transformer",
            "test_aux_loss",
            run_context=base_context,
            model_params=config,
            num_features=1,
            forecast_steps=5,
            window_size=WINDOW_SIZE,
            dataset=simple_dataset,
        )

        # Training should complete without errors
        forecaster.fit(
            simple_dataset.development_data,
            is_final_fit=False,
            dataset=simple_dataset
        )

        # Verify model was trained
        assert forecaster.fitted is True
        assert forecaster.model is not None

    def test_training_without_auxiliary_loss_backward_compatible(
        self,
        base_transformer_config,
        simple_dataset,
        base_context
    ):
        """Test backward compatibility - training without auxiliary loss still works."""
        config = {
            **base_transformer_config,
            "architecture": "encoder-decoder",
            "strategy": "direct",
            # No auxiliary_loss in config - should use default MSE
        }

        forecaster = ModelFactory.create(
            "transformer",
            "test_backward_compat",
            run_context=base_context,
            model_params=config,
            num_features=1,
            forecast_steps=5,
            window_size=WINDOW_SIZE,
            dataset=simple_dataset,
        )

        # Should work exactly as before
        forecaster.fit(
            simple_dataset.development_data,
            is_final_fit=False,
            dataset=simple_dataset
        )

        assert forecaster.fitted is True
        assert forecaster.model is not None

    def test_auxiliary_loss_with_different_weights(
        self,
        base_transformer_config,
        simple_dataset,
        base_context
    ):
        """Test auxiliary loss with different weight values."""
        for weight in [0.0, 0.1, 0.5, 1.0]:
            config = {
                **base_transformer_config,
                "architecture": "encoder-decoder",
                "strategy": "direct",
                "epochs": 1,  # Fast test
                "auxiliary_loss": {
                    "enabled": True,
                    "weight": weight,
                    "position_weighting": True
                }
            }

            forecaster = ModelFactory.create(
                "transformer",
                f"test_weight_{weight}",
                run_context=base_context,
                model_params=config,
                num_features=1,
                forecast_steps=5,
                window_size=WINDOW_SIZE,
                dataset=simple_dataset,
            )

            forecaster.fit(
                simple_dataset.development_data,
                is_final_fit=False,
                dataset=simple_dataset
            )

            assert forecaster.fitted is True
            assert forecaster.model is not None


class TestTransformerPredictionNoiseIntegration:
    """Integration tests for prediction noise injection."""

    def test_training_with_constant_noise_enabled(
        self,
        base_transformer_config,
        simple_dataset,
        base_context
    ):
        """Test that training runs with constant prediction noise."""
        config = {
            **base_transformer_config,
            "architecture": "encoder-decoder",
            "strategy": "direct",
            "prediction_noise": {
                "enabled": True,
                "std": 0.05,
                "schedule": "constant"
            }
        }

        forecaster = ModelFactory.create(
            "transformer",
            "test_const_noise",
            run_context=base_context,
            model_params=config,
            num_features=1,
            forecast_steps=5,
            window_size=WINDOW_SIZE,
            dataset=simple_dataset,
        )

        # Training should complete without errors
        forecaster.fit(
            simple_dataset.development_data,
            is_final_fit=False,
            dataset=simple_dataset
        )

        assert forecaster.fitted is True
        assert forecaster.model is not None

    def test_training_with_curriculum_noise_schedule(
        self,
        base_transformer_config,
        simple_dataset,
        base_context
    ):
        """Test training with curriculum noise schedule."""
        config = {
            **base_transformer_config,
            "architecture": "encoder-decoder",
            "strategy": "direct",
            "epochs": 2,
            "prediction_noise": {
                "enabled": True,
                "std": 0.1,
                "schedule": "curriculum"
            }
        }

        forecaster = ModelFactory.create(
            "transformer",
            "test_curriculum_noise",
            run_context=base_context,
            model_params=config,
            num_features=1,
            forecast_steps=5,
            window_size=WINDOW_SIZE,
            dataset=simple_dataset,
        )

        # Training should complete without errors
        forecaster.fit(
            simple_dataset.development_data,
            is_final_fit=False,
            dataset=simple_dataset
        )

        assert forecaster.fitted is True
        assert forecaster.model is not None


class TestTransformerCombinedFeatures:
    """Integration tests for combined auxiliary loss + prediction noise."""

    def test_training_with_both_features_enabled(
        self,
        base_transformer_config,
        simple_dataset,
        base_context
    ):
        """Test training with both auxiliary loss and prediction noise enabled."""
        config = {
            **base_transformer_config,
            "architecture": "encoder-decoder",
            "strategy": "direct",
            "auxiliary_loss": {
                "enabled": True,
                "weight": 0.15,
                "position_weighting": True
            },
            "prediction_noise": {
                "enabled": True,
                "std": 0.05,
                "schedule": "curriculum"
            }
        }

        forecaster = ModelFactory.create(
            "transformer",
            "test_combined",
            run_context=base_context,
            model_params=config,
            num_features=1,
            forecast_steps=5,
            window_size=WINDOW_SIZE,
            dataset=simple_dataset,
        )

        # Training with both features should work
        forecaster.fit(
            simple_dataset.development_data,
            is_final_fit=False,
            dataset=simple_dataset
        )

        assert forecaster.fitted is True
        assert forecaster.model is not None

    def test_combined_features_with_iterative_mode(
        self,
        base_transformer_config,
        simple_dataset,
        base_context
    ):
        """Test combined features work with iterative architecture."""
        config = {
            **base_transformer_config,
            "architecture": "encoder-decoder",
            "strategy": "iterative",  # Iterative mode
            "tgt_init": "zeros",
            "epochs": 1,
            "auxiliary_loss": {
                "enabled": True,
                "weight": 0.2,
                "position_weighting": True
            },
            "prediction_noise": {
                "enabled": True,
                "std": 0.05,
                "schedule": "constant"
            }
        }

        forecaster = ModelFactory.create(
            "transformer",
            "test_combined_iterative",
            run_context=base_context,
            model_params=config,
            num_features=1,
            forecast_steps=5,
            window_size=WINDOW_SIZE,
            dataset=simple_dataset,
        )

        forecaster.fit(
            simple_dataset.development_data,
            is_final_fit=False,
            dataset=simple_dataset
        )

        assert forecaster.fitted is True
        assert forecaster.model is not None


class TestTransformerConfigValidation:
    """Test validation of auxiliary loss and prediction noise configs."""

    def test_invalid_auxiliary_loss_weight_raises_error(
        self,
        base_transformer_config,
        simple_dataset,
        base_context
    ):
        """Test that invalid auxiliary loss weight raises ValueError."""
        config = {
            **base_transformer_config,
            "architecture": "encoder-decoder",
            "strategy": "direct",
            "auxiliary_loss": {
                "enabled": True,
                "weight": 1.5,  # Invalid: must be in [0, 1]
            }
        }

        with pytest.raises(ValueError, match="auxiliary_loss.weight must be in"):
            forecaster = ModelFactory.create(
                "transformer",
                "test_invalid_weight",
                run_context=base_context,
                model_params=config,
                num_features=1,
                forecast_steps=5,
                window_size=WINDOW_SIZE,
                dataset=simple_dataset,
            )

    def test_invalid_noise_std_raises_error(
        self,
        base_transformer_config,
        simple_dataset,
        base_context
    ):
        """Test that invalid noise std raises ValueError."""
        config = {
            **base_transformer_config,
            "architecture": "encoder-decoder",
            "strategy": "direct",
            "prediction_noise": {
                "enabled": True,
                "std": -0.1,  # Invalid: must be positive
            }
        }

        with pytest.raises(ValueError, match="prediction_noise.std must be positive"):
            forecaster = ModelFactory.create(
                "transformer",
                "test_invalid_std",
                run_context=base_context,
                model_params=config,
                num_features=1,
                forecast_steps=5,
                window_size=WINDOW_SIZE,
                dataset=simple_dataset,
            )

    def test_invalid_noise_schedule_raises_error(
        self,
        base_transformer_config,
        simple_dataset,
        base_context
    ):
        """Test that invalid noise schedule raises ValueError."""
        config = {
            **base_transformer_config,
            "architecture": "encoder-decoder",
            "strategy": "direct",
            "prediction_noise": {
                "enabled": True,
                "std": 0.05,
                "schedule": "invalid_schedule"  # Invalid
            }
        }

        with pytest.raises(ValueError, match="prediction_noise.schedule must be"):
            forecaster = ModelFactory.create(
                "transformer",
                "test_invalid_schedule",
                run_context=base_context,
                model_params=config,
                num_features=1,
                forecast_steps=5,
                window_size=WINDOW_SIZE,
                dataset=simple_dataset,
            )


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
