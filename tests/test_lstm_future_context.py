"""
Tests for LSTM future context functionality (Phase 1: Direct mode).

These tests verify:
1. Configuration validation (mode, strategy compatibility)
2. Causality (future_exog affects predictions)
3. Backward compatibility (mode='none' behavior)
4. Training stability (no divergence)
5. Pooling strategies (mean, last, learnable)
"""

import pytest
import torch
import numpy as np
import pandas as pd


class MockRunContext:
    """Mock RunContext for testing."""
    fold_idx = 0
    model_name = "lstm"
    window_size = 24
    forecast_steps = 12
    metadata = {}


class MockDataset:
    """Mock dataset with configurable covariates."""

    def __init__(self, target_cols, past_cov=None, future_cov=None):
        self.target_columns = target_cols
        self.past_covariates = past_cov or []
        self.future_covariates = future_cov or []

    @property
    def columns(self):
        return self.target_columns + self.past_covariates + self.future_covariates


class TestFutureCovariateModeValidation:
    """Test configuration validation for future_covariate_mode."""

    def test_invalid_mode_raises_error(self):
        """Invalid mode → ValueError"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "direct",
            "future_covariate_mode": "invalid_mode",  # Invalid
        }

        with pytest.raises(ValueError, match="Invalid future_covariate_mode"):
            LSTMForecaster(
                model_params=model_params,
                num_features=1,
                forecast_steps=12,
                window_size=24,
                dataset=MockDataset(["value"]),
                run_context=MockRunContext()
            )

    def test_global_mode_requires_direct_strategy(self):
        """global mode + iterative → ValueError"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "iterative",  # Incompatible
            "future_covariate_mode": "global",
        }

        dataset = MockDataset(["value"], future_cov=["exog1"])

        with pytest.raises(ValueError, match="requires strategy='direct'"):
            LSTMForecaster(
                model_params=model_params,
                num_features=1,
                forecast_steps=12,
                window_size=24,
                dataset=dataset,
                run_context=MockRunContext()
            )

    def test_global_mode_requires_future_covariates(self):
        """global mode + no future_cov → ValueError"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "direct",
            "future_covariate_mode": "global",
        }

        # Dataset without future_covariates
        dataset = MockDataset(["value"])

        with pytest.raises(ValueError, match="requires future_covariates"):
            LSTMForecaster(
                model_params=model_params,
                num_features=1,
                forecast_steps=12,
                window_size=24,
                dataset=dataset,
                run_context=MockRunContext()
            )

    # REMOVED: test_stepwise_mode_not_implemented
    # Stepwise mode is now implemented in Phase 2a
    # See TestStepwiseMode class for comprehensive stepwise tests


class TestFutureContextCausality:
    """Test that future_exog actually affects predictions."""

    def test_future_exog_changes_prediction_global_mode(self):
        """future_exog MUST change prediction in global mode"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "direct",
            "future_covariate_mode": "global",
        }

        dataset = MockDataset(["value"], future_cov=["exog1"])

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=dataset,
            run_context=MockRunContext()
        )

        # Move model to CPU for testing
        forecaster.model = forecaster.model.cpu()
        forecaster.device = torch.device('cpu')
        forecaster.fitted = True
        forecaster.model.eval()

        # Create input tensors
        input_tensor = torch.randn(1, 24, 2)  # (B, W, features: target + future_cov)

        # Two different future contexts
        future_exog_1 = torch.zeros(1, 12, 1)  # All zeros
        future_exog_2 = torch.ones(1, 12, 1)   # All ones

        # Predictions should differ
        with torch.no_grad():
            pred1 = forecaster._predict_direct(input_tensor, future_exog_tensor=future_exog_1)
            pred2 = forecaster._predict_direct(input_tensor, future_exog_tensor=future_exog_2)

        # Convert to numpy for comparison
        pred1_np = pred1.cpu().numpy()
        pred2_np = pred2.cpu().numpy()

        # MUST differ (causality test)
        assert not np.allclose(pred1_np, pred2_np, rtol=1e-5, atol=1e-5), \
            "future_exog did not affect predictions - causality broken!"

    def test_global_mode_requires_future_exog_tensor(self):
        """global mode without future_exog_tensor → ValueError"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "direct",
            "future_covariate_mode": "global",
        }

        dataset = MockDataset(["value"], future_cov=["exog1"])

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=dataset,
            run_context=MockRunContext()
        )

        # Move to CPU
        forecaster.model = forecaster.model.cpu()
        forecaster.device = torch.device('cpu')
        forecaster.fitted = True
        forecaster.model.eval()

        input_tensor = torch.randn(1, 24, 2)

        # Missing future_exog_tensor → should raise
        with pytest.raises(ValueError, match="future_exog_tensor is required"):
            forecaster._predict_direct(input_tensor)  # No future_exog_tensor


class TestBackwardCompatibility:
    """Test that mode='none' preserves backward compatible behavior."""

    def test_mode_none_ignores_future_exog(self):
        """mode='none' → future_exog ignored, no error"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "direct",
            "future_covariate_mode": "none",  # Explicit
        }

        # Dataset has future_cov but mode is 'none'
        dataset = MockDataset(["value"], future_cov=["exog1"])

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=dataset,
            run_context=MockRunContext()
        )

        # Move to CPU
        forecaster.model = forecaster.model.cpu()
        forecaster.device = torch.device('cpu')
        forecaster.fitted = True
        forecaster.model.eval()

        input_tensor = torch.randn(1, 24, 2)
        future_exog = torch.randn(1, 12, 1)

        # Both should work and produce identical results
        with torch.no_grad():
            pred_none = forecaster._predict_direct(input_tensor)
            pred_with = forecaster._predict_direct(input_tensor, future_exog_tensor=future_exog)

        pred_none_np = pred_none.cpu().numpy()
        pred_with_np = pred_with.cpu().numpy()

        # MUST be identical (future_exog ignored)
        np.testing.assert_allclose(
            pred_none_np, pred_with_np,
            rtol=1e-6, atol=1e-6,
            err_msg="mode='none' did not ignore future_exog"
        )

    def test_default_mode_is_none(self):
        """Default mode should be 'none' for backward compatibility"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "direct",
            # No future_covariate_mode specified → defaults to 'none'
        }

        dataset = MockDataset(["value"])

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=dataset,
            run_context=MockRunContext()
        )

        assert forecaster.future_covariate_mode == "none"


class TestPoolingStrategies:
    """Test different pooling strategies for FutureContextEncoder."""

    @pytest.mark.parametrize("pooling", ["mean", "last", "learnable"])
    def test_pooling_strategies(self, pooling):
        """All pooling strategies produce valid output"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "direct",
            "future_covariate_mode": "global",
            "future_context_config": {
                "pooling": pooling,
                "dropout": 0.0,
            }
        }

        dataset = MockDataset(["value"], future_cov=["exog1"])

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=dataset,
            run_context=MockRunContext()
        )

        # Move to CPU
        forecaster.model = forecaster.model.cpu()
        forecaster.device = torch.device('cpu')
        forecaster.fitted = True
        forecaster.model.eval()

        input_tensor = torch.randn(1, 24, 2)
        future_exog = torch.randn(1, 12, 1)

        with torch.no_grad():
            pred = forecaster._predict_direct(input_tensor, future_exog_tensor=future_exog)

        pred_np = pred.cpu().numpy()

        # Validate shape
        assert pred_np.shape == (1, 12, 1), f"Unexpected shape: {pred_np.shape}"

        # Validate all values are finite
        assert np.isfinite(pred_np).all(), f"Non-finite values for pooling={pooling}"

    def test_compression_dimension(self):
        """Test optional compression layer"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "direct",
            "future_covariate_mode": "global",
            "future_context_config": {
                "pooling": "mean",
                "compression_dim": 8,  # Compress 10 features → 8
                "dropout": 0.0,
            }
        }

        # Dataset with 10 future covariates
        dataset = MockDataset(
            ["value"],
            future_cov=[f"exog{i}" for i in range(10)]
        )

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=dataset,
            run_context=MockRunContext()
        )

        # Move to CPU
        forecaster.model = forecaster.model.cpu()
        forecaster.device = torch.device('cpu')
        forecaster.fitted = True
        forecaster.model.eval()

        # Check that compression layer was created
        assert forecaster.model.future_ctx_encoder is not None
        assert forecaster.model.future_ctx_encoder.compression is not None
        assert forecaster.model.future_ctx_encoder.output_size == 8

        # Test forward pass
        input_tensor = torch.randn(1, 24, 11)  # target + 10 future_cov
        future_exog = torch.randn(1, 12, 10)

        with torch.no_grad():
            pred = forecaster._predict_direct(input_tensor, future_exog_tensor=future_exog)

        assert pred.shape == (1, 12, 1)
        assert np.isfinite(pred.cpu().numpy()).all()


class TestFutureContextEncoder:
    """Unit tests for FutureContextEncoder component."""

    def test_encoder_mean_pooling(self):
        """Test mean pooling produces correct output"""
        from models.lstm import FutureContextEncoder

        encoder = FutureContextEncoder(
            future_cov_size=3,
            forecast_steps=10,
            pooling="mean",
            dropout=0.0
        )

        input_tensor = torch.randn(4, 10, 3)  # (B=4, H=10, F=3)
        output = encoder(input_tensor)

        assert output.shape == (4, 3), f"Expected (4, 3), got {output.shape}"
        assert torch.isfinite(output).all()

        # Verify mean pooling
        expected_mean = input_tensor.mean(dim=1)
        torch.testing.assert_close(output, expected_mean, rtol=1e-5, atol=1e-5)

    def test_encoder_last_pooling(self):
        """Test last pooling produces correct output"""
        from models.lstm import FutureContextEncoder

        encoder = FutureContextEncoder(
            future_cov_size=3,
            forecast_steps=10,
            pooling="last",
            dropout=0.0
        )

        input_tensor = torch.randn(4, 10, 3)
        output = encoder(input_tensor)

        assert output.shape == (4, 3)
        assert torch.isfinite(output).all()

        # Verify last timestep
        expected_last = input_tensor[:, -1, :]
        torch.testing.assert_close(output, expected_last, rtol=1e-5, atol=1e-5)

    def test_encoder_validation(self):
        """Test encoder input validation"""
        from models.lstm import FutureContextEncoder

        encoder = FutureContextEncoder(
            future_cov_size=3,
            forecast_steps=10,
            pooling="mean"
        )

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            encoder(torch.randn(4, 3))

        # Wrong forecast steps
        with pytest.raises(ValueError, match="Expected 10 forecast steps"):
            encoder(torch.randn(4, 5, 3))  # 5 instead of 10


class TestLSTMModelWithFutureContext:
    """Unit tests for LSTMModelWithFutureContext wrapper."""

    def test_wrapper_without_encoder_behaves_like_base(self):
        """Wrapper without encoder should behave identically to base model"""
        from models.lstm import LSTMModel, LSTMModelWithFutureContext

        base_model = LSTMModel(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            output_steps=10,
            output_features=2,
            dropout=0.0
        )

        # Wrapper without encoder
        wrapper = LSTMModelWithFutureContext(
            base_model=base_model,
            future_ctx_encoder=None
        )

        input_tensor = torch.randn(4, 20, 5)  # (B=4, W=20, F=5)

        base_model.eval()
        wrapper.eval()

        with torch.no_grad():
            output_base = base_model(input_tensor)
            output_wrapper = wrapper(input_tensor, tgt=None)

        # Should be identical
        torch.testing.assert_close(output_base, output_wrapper, rtol=1e-6, atol=1e-6)

    def test_wrapper_with_encoder_produces_different_output(self):
        """Wrapper with encoder should produce different output when tgt (future covariates) provided"""
        from models.lstm import LSTMModel, LSTMModelWithFutureContext, FutureContextEncoder

        base_model = LSTMModel(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            output_steps=10,
            output_features=2,
            dropout=0.0
        )

        encoder = FutureContextEncoder(
            future_cov_size=3,
            forecast_steps=10,
            pooling="mean",
            dropout=0.0
        )

        wrapper = LSTMModelWithFutureContext(
            base_model=base_model,
            future_ctx_encoder=encoder
        )

        input_tensor = torch.randn(4, 20, 5)
        tgt_tensor = torch.randn(4, 10, 3)  # Future covariates

        wrapper.eval()

        with torch.no_grad():
            output_without = wrapper(input_tensor, tgt=None)
            output_with = wrapper(input_tensor, tgt=tgt_tensor)

        # Should be different
        assert not torch.allclose(output_without, output_with, rtol=1e-5, atol=1e-5), \
            "Future covariates (tgt) did not affect output"

        # Both should have correct shape
        assert output_without.shape == (4, 10, 2)
        assert output_with.shape == (4, 10, 2)


class TestStepwiseMode:
    """Test stepwise mode for iterative predictions (Phase 2a)."""

    def test_stepwise_mode_requires_iterative_strategy(self):
        """stepwise mode + direct → ValueError"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "direct",  # Incompatible
            "future_covariate_mode": "stepwise",
        }

        dataset = MockDataset(["value"], future_cov=["exog1"])

        with pytest.raises(ValueError, match="requires strategy='iterative'"):
            LSTMForecaster(
                model_params=model_params,
                num_features=1,
                forecast_steps=12,
                window_size=24,
                dataset=dataset,
                run_context=MockRunContext()
            )

    def test_stepwise_mode_requires_future_covariates(self):
        """stepwise mode + no future_cov → ValueError"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "iterative",
            "future_covariate_mode": "stepwise",
        }

        # Dataset without future_covariates
        dataset = MockDataset(["value"])

        with pytest.raises(ValueError, match="requires future_covariates"):
            LSTMForecaster(
                model_params=model_params,
                num_features=1,
                forecast_steps=12,
                window_size=24,
                dataset=dataset,
                run_context=MockRunContext()
            )

    def test_stepwise_requires_future_exog_tensor(self):
        """stepwise mode without future_exog_tensor → ValueError"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "iterative",
            "future_covariate_mode": "stepwise",
        }

        dataset = MockDataset(["value"], future_cov=["exog1"])

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=dataset,
            run_context=MockRunContext()
        )

        # Move to CPU
        forecaster.model = forecaster.model.cpu()
        forecaster.device = torch.device('cpu')
        forecaster.fitted = True
        forecaster.model.eval()

        input_tensor = torch.randn(1, 24, 2)

        # Missing future_exog_tensor → should raise
        with pytest.raises(ValueError, match="future_exog_tensor is required"):
            forecaster._predict_iterative(input_tensor)

    def test_stepwise_future_exog_affects_prediction(self):
        """future_exog MUST affect prediction in stepwise mode"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "iterative",
            "future_covariate_mode": "stepwise",
        }

        dataset = MockDataset(["value"], future_cov=["exog1"])

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=dataset,
            run_context=MockRunContext()
        )

        # Move to CPU
        forecaster.model = forecaster.model.cpu()
        forecaster.device = torch.device('cpu')
        forecaster.fitted = True
        forecaster.model.eval()

        input_tensor = torch.randn(1, 24, 2)  # (B, W, target + future_cov)

        # Two different future contexts
        future_exog_1 = torch.zeros(1, 12, 1)  # All zeros
        future_exog_2 = torch.ones(1, 12, 1)   # All ones

        # Predictions should differ
        with torch.no_grad():
            pred1 = forecaster._predict_iterative(input_tensor, future_exog_tensor=future_exog_1)
            pred2 = forecaster._predict_iterative(input_tensor, future_exog_tensor=future_exog_2)

        pred1_np = pred1.cpu().numpy()
        pred2_np = pred2.cpu().numpy()

        # MUST differ (causality test)
        assert not np.allclose(pred1_np, pred2_np, rtol=1e-5, atol=1e-5), \
            "future_exog did not affect predictions in stepwise mode - causality broken!"

    def test_stepwise_per_step_injection(self):
        """Verify that different future_exog values per step produce different outputs"""
        from models.lstm import LSTMForecaster

        model_params = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "iterative",
            "future_covariate_mode": "stepwise",
        }

        dataset = MockDataset(["value"], future_cov=["exog1"])

        forecaster = LSTMForecaster(
            model_params=model_params,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=dataset,
            run_context=MockRunContext()
        )

        # Move to CPU
        forecaster.model = forecaster.model.cpu()
        forecaster.device = torch.device('cpu')
        forecaster.fitted = True
        forecaster.model.eval()

        input_tensor = torch.randn(1, 24, 2)

        # Create future_exog with increasing values per step
        future_exog_increasing = torch.arange(12).reshape(1, 12, 1).float()  # [0, 1, 2, ..., 11]
        future_exog_flat = torch.ones(1, 12, 1)  # [1, 1, 1, ..., 1]

        with torch.no_grad():
            pred_increasing = forecaster._predict_iterative(input_tensor, future_exog_tensor=future_exog_increasing)
            pred_flat = forecaster._predict_iterative(input_tensor, future_exog_tensor=future_exog_flat)

        pred_increasing_np = pred_increasing.cpu().numpy()
        pred_flat_np = pred_flat.cpu().numpy()

        # MUST differ (per-step injection test)
        assert not np.allclose(pred_increasing_np, pred_flat_np, rtol=1e-5, atol=1e-5), \
            "Per-step future_exog values did not affect predictions - injection broken!"

        # Check that predictions have valid shape
        assert pred_increasing_np.shape == (1, 12, 1)
        assert np.isfinite(pred_increasing_np).all()

    def test_stepwise_vs_none_produces_different_output(self):
        """Verify that stepwise mode produces different output than none mode"""
        from models.lstm import LSTMForecaster

        # Model with stepwise
        model_params_stepwise = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "iterative",
            "future_covariate_mode": "stepwise",
        }

        # Model with none
        model_params_none = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "strategy": "iterative",
            "future_covariate_mode": "none",
        }

        dataset = MockDataset(["value"], future_cov=["exog1"])

        forecaster_stepwise = LSTMForecaster(
            model_params=model_params_stepwise,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=dataset,
            run_context=MockRunContext()
        )

        forecaster_none = LSTMForecaster(
            model_params=model_params_none,
            num_features=1,
            forecast_steps=12,
            window_size=24,
            dataset=dataset,
            run_context=MockRunContext()
        )

        # Share weights for fair comparison
        forecaster_stepwise.model = forecaster_stepwise.model.cpu()
        forecaster_none.model = forecaster_none.model.cpu()
        forecaster_stepwise.device = torch.device('cpu')
        forecaster_none.device = torch.device('cpu')

        # Copy weights
        forecaster_none.model.load_state_dict(forecaster_stepwise.model.state_dict())

        forecaster_stepwise.fitted = True
        forecaster_none.fitted = True
        forecaster_stepwise.model.eval()
        forecaster_none.model.eval()

        input_tensor = torch.randn(1, 24, 2)
        future_exog = torch.randn(1, 12, 1)

        with torch.no_grad():
            pred_stepwise = forecaster_stepwise._predict_iterative(input_tensor, future_exog_tensor=future_exog)
            pred_none = forecaster_none._predict_iterative(input_tensor)  # No future_exog

        pred_stepwise_np = pred_stepwise.cpu().numpy()
        pred_none_np = pred_none.cpu().numpy()

        # MUST differ (stepwise uses future_exog, none ignores it)
        assert not np.allclose(pred_stepwise_np, pred_none_np, rtol=1e-5, atol=1e-5), \
            "Stepwise and none modes produced identical output - stepwise not working!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
