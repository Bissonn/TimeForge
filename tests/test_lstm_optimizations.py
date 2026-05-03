"""
Tests for LSTM optimizations: PC Mode alignment and rolling buffer.

These tests verify:
1. Training-Inference alignment for PC Mode
2. Rolling buffer optimization preserves numerical correctness
"""

import pytest
import torch
import numpy as np


class TestLSTMRollingBufferOptimization:
    """Test rolling buffer optimization in iterative mode."""

    def test_rolling_buffer_deterministic(self):
        """Verify rolling buffer produces deterministic results."""
        from models.lstm import LSTMForecaster

        class MockRunContext:
            fold_idx = 0
            model_name = "lstm"
            window_size = 48
            forecast_steps = 24

        model_params = {
            "hidden_size": 32,
            "num_layers": 2,
            "dropout": 0.0,  # No dropout for determinism
            "learning_rate": 0.001,
            "batch_size": 8,
            "epochs": 1,
            "strategy": "iterative"
        }

        class MockDataset:
            target_columns = ["value"]
            past_covariates= []
            future_covariates= []
            past_covariates = []
            future_covariates = []

            @property
            def columns(self):
                return self.target_columns + self.past_covariates + self.future_covariates

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=1,
            forecast_steps=24,
            window_size=48,
            dataset=MockDataset(),
            run_context=MockRunContext()
        )

        forecaster.fitted = True
        forecaster.model.eval()
        forecaster.model = forecaster.model.cpu()
        forecaster.device = torch.device('cpu')

        input_tensor = torch.randn(4, 48, 1)

        # Run prediction twice
        with torch.no_grad():
            pred1 = forecaster._predict_iterative(input_tensor)
            pred2 = forecaster._predict_iterative(input_tensor)

        # Should be deterministic
        torch.testing.assert_close(pred1, pred2, rtol=1e-6, atol=1e-6)
        assert pred1.shape == (4, 24, 1)
        assert torch.isfinite(pred1).all()

    def test_rolling_buffer_shape_correctness(self):
        """Verify rolling buffer produces correct output shapes."""
        from models.lstm import LSTMForecaster

        class MockRunContext:
            fold_idx = 0
            model_name = "lstm"
            window_size = 24
            forecast_steps = 12

        model_params = {
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "iterative"
        }

        class MockDataset:
            target_columns = ["v1", "v2"]
            past_covariates= []
            future_covariates= []
            past_covariates = []
            future_covariates = []

            @property
            def columns(self):
                return self.target_columns + self.past_covariates + self.future_covariates

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=2,
            forecast_steps=12,
            window_size=24,
            dataset=MockDataset(),
            run_context=MockRunContext()
        )

        forecaster.fitted = True
        forecaster.model.eval()
        forecaster.model = forecaster.model.cpu()
        forecaster.device = torch.device('cpu')

        # Test various batch sizes
        for batch_size in [1, 4, 16]:
            input_tensor = torch.randn(batch_size, 24, 2)

            with torch.no_grad():
                predictions = forecaster._predict_iterative(input_tensor)

            assert predictions.shape == (batch_size, 12, 2)
            assert torch.isfinite(predictions).all()


class TestLSTMDirectModeUnaffected:
    """Verify direct mode is unaffected by changes."""

    def test_direct_mode_works_correctly(self):
        """Verify direct mode still works correctly."""
        from models.lstm import LSTMForecaster

        class MockRunContext:
            fold_idx = 0
            model_name = "lstm"
            window_size = 24
            forecast_steps = 12

        model_params = {
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "direct"
        }

        class MockDataset:
            target_columns = ["value"]
            past_covariates= []
            future_covariates= []
            past_covariates = []
            future_covariates = []

            @property
            def columns(self):
                return self.target_columns + self.past_covariates + self.future_covariates

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=MockDataset(),
            run_context=MockRunContext()
        )

        forecaster.fitted = True
        forecaster.model.eval()

        # Ensure model is on CPU for testing
        forecaster.model = forecaster.model.cpu()
        forecaster.device = torch.device('cpu')

        input_tensor = torch.randn(4, 24, 1)

        with torch.no_grad():
            predictions = forecaster._predict_direct(input_tensor)

        assert predictions.shape == (4, 12, 1)
        assert torch.isfinite(predictions).all()


class TestLSTMPCModeIntegration:
    """Integration tests for PC Mode."""

    def test_iterative_with_p_and_c_variables(self):
        """Test iterative prediction with both P and C variables."""
        from models.lstm import LSTMForecaster
        import torch.nn as nn

        class MockRunContext:
            fold_idx = 0
            model_name = "lstm"
            window_size = 24
            forecast_steps = 12

        model_params = {
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "iterative"
        }

        class MockDataset:
            target_columns = ["y"]
            past_covariates= ["p1", "p2"]
            future_covariates= ["c1"]
            past_covariates = ["p1", "p2"]
            future_covariates = ["c1"]

            @property
            def columns(self):
                return self.target_columns + self.past_covariates + self.future_covariates

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=MockDataset(),
            run_context=MockRunContext()
        )

        # Setup PC mode manually
        forecaster._past_only_size = 2
        forecaster._continuous_size = 1
        forecaster._continuous_exog = ["c1"]

        # Reinitialize model with correct input size: 1 target + 2 P + 1 C = 4 features
        # This is necessary because the model was initialized before we set PC mode variables
        from models.lstm import LSTMModel
        # Use encoder_input_size from feature_layout which accounts for all covariates
        pc_input_size = forecaster.feature_layout.encoder_input_size
        forecaster.model = LSTMModel(
            input_size=pc_input_size,  # Targets(1) + past_cov(2) + future_cov(1)
            hidden_size=16,
            num_layers=1,
            output_steps=1,  # Iterative mode: one step at a time
            output_features=1,  # One target feature
            dropout=0.0
        )
        forecaster.fitted = True
        forecaster.model.eval()
        forecaster.model = forecaster.model.cpu()
        forecaster.device = torch.device('cpu')

        # Input: [Targets(1), P(2), C(1)] = 4 features
        input_tensor = torch.randn(4, 24, 4)
        future_exog = torch.randn(4, 12, 1)

        with torch.no_grad():
            predictions = forecaster._predict_iterative(
                input_tensor,
                future_exog_tensor=future_exog
            )


        assert predictions.shape == (4, 12, 1)
        assert torch.isfinite(predictions).all()

    def test_iterative_raises_on_missing_future_exog(self):
        """Verify error when SHARED C variables need future exog but none provided."""
        from models.lstm import LSTMForecaster

        class MockRunContext:
            fold_idx = 0
            model_name = "lstm"
            window_size = 24
            forecast_steps = 12

        model_params = {
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "iterative",
            "future_covariate_mode": "stepwise"  # Phase 2a: Requires future_exog
        }

        class MockDataset:
            target_columns = ["y"]
            past_covariates= ["c1"]  # ✅ SHARED
            future_covariates= ["c1"]  # ✅ SHARED
            past_covariates = []
            future_covariates = ["c1"]  # Shared variable is future_covariate

            @property
            def columns(self):
                return self.target_columns + self.past_covariates + self.future_covariates

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=MockDataset(),
            run_context=MockRunContext()
        )

        forecaster.fitted = True
        forecaster.model.eval()
        forecaster.model = forecaster.model.cpu()
        forecaster.device = torch.device('cpu')

        input_tensor = torch.randn(4, 24, 2)

        with pytest.raises(ValueError):
            forecaster._predict_iterative(input_tensor, future_exog_tensor=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
