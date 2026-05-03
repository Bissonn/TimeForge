"""
Unit tests for custom loss functions.

Tests for AuxiliaryMultiStepLoss and AdaptiveNoiseScheduler.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from utils.losses import AuxiliaryMultiStepLoss, AdaptiveNoiseScheduler


class TestAuxiliaryMultiStepLoss:
    """Test suite for AuxiliaryMultiStepLoss."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        loss_fn = AuxiliaryMultiStepLoss()

        assert loss_fn.auxiliary_weight == 0.1
        assert loss_fn.position_weighting is True
        assert loss_fn.reduction == 'mean'
        assert isinstance(loss_fn.base_loss, nn.MSELoss)

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        base_loss = nn.L1Loss()
        loss_fn = AuxiliaryMultiStepLoss(
            base_loss=base_loss,
            auxiliary_weight=0.2,
            position_weighting=False,
            reduction='sum'
        )

        assert loss_fn.auxiliary_weight == 0.2
        assert loss_fn.position_weighting is False
        assert loss_fn.reduction == 'sum'
        assert loss_fn.base_loss is base_loss

    def test_validation_auxiliary_weight(self):
        """Test validation of auxiliary_weight parameter."""
        # Valid values
        AuxiliaryMultiStepLoss(auxiliary_weight=0.0)
        AuxiliaryMultiStepLoss(auxiliary_weight=0.5)
        AuxiliaryMultiStepLoss(auxiliary_weight=1.0)

        # Invalid values
        with pytest.raises(ValueError, match="auxiliary_weight must be in"):
            AuxiliaryMultiStepLoss(auxiliary_weight=-0.1)

        with pytest.raises(ValueError, match="auxiliary_weight must be in"):
            AuxiliaryMultiStepLoss(auxiliary_weight=1.5)

    def test_validation_reduction(self):
        """Test validation of reduction parameter."""
        # Valid values
        AuxiliaryMultiStepLoss(reduction='mean')
        AuxiliaryMultiStepLoss(reduction='sum')
        AuxiliaryMultiStepLoss(reduction='none')

        # Invalid value
        with pytest.raises(ValueError, match="reduction must be"):
            AuxiliaryMultiStepLoss(reduction='invalid')

    def test_forward_shape_validation(self):
        """Test that forward validates input shapes."""
        loss_fn = AuxiliaryMultiStepLoss()

        # Valid shapes
        pred = torch.randn(8, 24, 3)  # (B, H, F)
        target = torch.randn(8, 24, 3)
        loss = loss_fn(pred, target)
        assert loss.shape == torch.Size([])  # Scalar

        # Mismatched shapes
        pred = torch.randn(8, 24, 3)
        target = torch.randn(8, 12, 3)  # Different H
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(pred, target)

        # Wrong dimension
        pred = torch.randn(8, 24)  # 2D instead of 3D
        target = torch.randn(8, 24)
        with pytest.raises(ValueError, match="Expected 3D tensors"):
            loss_fn(pred, target)

    def test_zero_auxiliary_weight_equals_base_loss(self):
        """Test that auxiliary_weight=0 produces same result as base loss."""
        B, H, F = 4, 12, 2
        pred = torch.randn(B, H, F)
        target = torch.randn(B, H, F)

        base_loss = nn.MSELoss()
        aux_loss_fn = AuxiliaryMultiStepLoss(
            base_loss=base_loss,
            auxiliary_weight=0.0
        )

        expected = base_loss(pred, target)
        actual = aux_loss_fn(pred, target)

        torch.testing.assert_close(actual, expected)

    def test_one_auxiliary_weight_only_auxiliary(self):
        """Test that auxiliary_weight=1.0 uses only auxiliary loss."""
        B, H, F = 4, 12, 2
        pred = torch.randn(B, H, F)
        target = torch.randn(B, H, F)

        aux_loss_fn = AuxiliaryMultiStepLoss(
            auxiliary_weight=1.0,
            position_weighting=False  # Uniform for simplicity
        )

        # Compute expected: mean of per-step losses
        step_losses = []
        for t in range(H):
            step_loss = torch.nn.functional.mse_loss(pred[:, t, :], target[:, t, :])
            step_losses.append(step_loss)
        expected = torch.stack(step_losses).mean()

        actual = aux_loss_fn(pred, target)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_position_weighting_effect(self):
        """Test that position_weighting changes the result."""
        B, H, F = 4, 12, 2
        pred = torch.randn(B, H, F)
        target = torch.randn(B, H, F)

        loss_weighted = AuxiliaryMultiStepLoss(
            auxiliary_weight=1.0,
            position_weighting=True
        )

        loss_uniform = AuxiliaryMultiStepLoss(
            auxiliary_weight=1.0,
            position_weighting=False
        )

        result_weighted = loss_weighted(pred, target)
        result_uniform = loss_uniform(pred, target)

        # They should be different (unless by extreme coincidence)
        assert not torch.allclose(result_weighted, result_uniform)

    def test_reduction_none_returns_per_sample_loss(self):
        """Test that reduction='none' returns per-sample losses."""
        B, H, F = 4, 12, 2
        pred = torch.randn(B, H, F)
        target = torch.randn(B, H, F)

        loss_fn = AuxiliaryMultiStepLoss(
            auxiliary_weight=0.5,
            reduction='none'
        )

        result = loss_fn(pred, target)

        # Should return (B,) shape
        assert result.shape == torch.Size([B])
        assert result.ndim == 1

    def test_perfect_predictions_zero_loss(self):
        """Test that perfect predictions give zero loss."""
        B, H, F = 4, 12, 2
        pred = torch.randn(B, H, F)
        target = pred.clone()  # Perfect match

        loss_fn = AuxiliaryMultiStepLoss(auxiliary_weight=0.5)

        result = loss_fn(pred, target)

        assert torch.allclose(result, torch.tensor(0.0), atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the loss."""
        B, H, F = 4, 12, 2
        pred = torch.randn(B, H, F, requires_grad=True)
        target = torch.randn(B, H, F)

        loss_fn = AuxiliaryMultiStepLoss(auxiliary_weight=0.5)

        loss = loss_fn(pred, target)
        loss.backward()

        # Check that gradients exist and are finite
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert (pred.grad != 0).any()  # Some gradients should be non-zero

    def test_combined_loss_weighted_average(self):
        """Test that combined loss is correctly weighted."""
        B, H, F = 4, 12, 2
        pred = torch.randn(B, H, F)
        target = torch.randn(B, H, F)

        w = 0.3
        loss_fn = AuxiliaryMultiStepLoss(
            auxiliary_weight=w,
            position_weighting=False
        )

        # Compute components manually
        primary = nn.MSELoss()(pred, target)

        step_losses = []
        for t in range(H):
            step_losses.append(
                torch.nn.functional.mse_loss(pred[:, t, :], target[:, t, :])
            )
        auxiliary = torch.stack(step_losses).mean()

        expected = (1 - w) * primary + w * auxiliary
        actual = loss_fn(pred, target)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


class TestAdaptiveNoiseScheduler:
    """Test suite for AdaptiveNoiseScheduler."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        scheduler = AdaptiveNoiseScheduler()

        assert scheduler.base_std == 0.05
        assert scheduler.multiplier == 0.5
        assert scheduler.min_std == 0.001
        assert scheduler.max_std == 0.5
        assert scheduler.current_val_error is None
        assert scheduler.history == []

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        scheduler = AdaptiveNoiseScheduler(
            base_std=0.1,
            adaptive_multiplier=0.7,
            min_std=0.01,
            max_std=0.3
        )

        assert scheduler.base_std == 0.1
        assert scheduler.multiplier == 0.7
        assert scheduler.min_std == 0.01
        assert scheduler.max_std == 0.3

    def test_validation_base_std(self):
        """Test validation of base_std parameter."""
        AdaptiveNoiseScheduler(base_std=0.01)  # Valid

        with pytest.raises(ValueError, match="base_std must be positive"):
            AdaptiveNoiseScheduler(base_std=0.0)

        with pytest.raises(ValueError, match="base_std must be positive"):
            AdaptiveNoiseScheduler(base_std=-0.1)

    def test_validation_adaptive_multiplier(self):
        """Test validation of adaptive_multiplier parameter."""
        AdaptiveNoiseScheduler(adaptive_multiplier=0.5)  # Valid
        AdaptiveNoiseScheduler(adaptive_multiplier=1.0)  # Valid

        with pytest.raises(ValueError, match="adaptive_multiplier must be in"):
            AdaptiveNoiseScheduler(adaptive_multiplier=0.0)

        with pytest.raises(ValueError, match="adaptive_multiplier must be in"):
            AdaptiveNoiseScheduler(adaptive_multiplier=1.5)

    def test_validation_min_max_std(self):
        """Test validation of min_std and max_std."""
        AdaptiveNoiseScheduler(min_std=0.01, max_std=0.5)  # Valid

        with pytest.raises(ValueError, match="min_std.*must be < max_std"):
            AdaptiveNoiseScheduler(min_std=0.5, max_std=0.5)

        with pytest.raises(ValueError, match="min_std.*must be < max_std"):
            AdaptiveNoiseScheduler(min_std=0.6, max_std=0.5)

    def test_get_noise_std_no_validation_returns_base(self):
        """Test that get_noise_std returns base_std before any validation."""
        scheduler = AdaptiveNoiseScheduler(base_std=0.08)

        noise_std = scheduler.get_noise_std()

        assert noise_std == 0.08

    def test_update_stores_validation_error(self):
        """Test that update() stores validation error."""
        scheduler = AdaptiveNoiseScheduler()

        scheduler.update(val_error=0.15)

        assert scheduler.current_val_error == 0.15
        assert len(scheduler.history) == 1
        assert scheduler.history[0] == 0.15

    def test_get_noise_std_adaptive_scaling(self):
        """Test that noise std scales with validation error."""
        scheduler = AdaptiveNoiseScheduler(
            base_std=0.05,
            adaptive_multiplier=0.5,
            min_std=0.001,
            max_std=1.0
        )

        scheduler.update(val_error=0.2)
        noise_std = scheduler.get_noise_std()

        # Expected: 0.5 * 0.2 = 0.1
        assert noise_std == pytest.approx(0.1, abs=1e-6)

    def test_get_noise_std_clipped_to_min(self):
        """Test that noise std is clipped to min_std."""
        scheduler = AdaptiveNoiseScheduler(
            adaptive_multiplier=0.5,
            min_std=0.05,
            max_std=1.0
        )

        scheduler.update(val_error=0.01)  # Would give 0.5 * 0.01 = 0.005
        noise_std = scheduler.get_noise_std()

        # Should be clipped to min_std
        assert noise_std == 0.05

    def test_get_noise_std_clipped_to_max(self):
        """Test that noise std is clipped to max_std."""
        scheduler = AdaptiveNoiseScheduler(
            adaptive_multiplier=0.5,
            min_std=0.001,
            max_std=0.1
        )

        scheduler.update(val_error=1.0)  # Would give 0.5 * 1.0 = 0.5
        noise_std = scheduler.get_noise_std()

        # Should be clipped to max_std
        assert noise_std == 0.1

    def test_history_tracking(self):
        """Test that history tracks all validation errors."""
        scheduler = AdaptiveNoiseScheduler()

        scheduler.update(0.1)
        scheduler.update(0.15)
        scheduler.update(0.12)

        assert len(scheduler.history) == 3
        assert scheduler.history == [0.1, 0.15, 0.12]
        assert scheduler.current_val_error == 0.12

    def test_state_save_and_load(self):
        """Test state save and load for checkpointing."""
        scheduler = AdaptiveNoiseScheduler(
            base_std=0.1,
            adaptive_multiplier=0.7
        )

        scheduler.update(0.15)
        scheduler.update(0.12)

        # Save state
        state = scheduler.get_state()

        # Create new scheduler and load state
        new_scheduler = AdaptiveNoiseScheduler()
        new_scheduler.load_state(state)

        assert new_scheduler.base_std == 0.1
        assert new_scheduler.multiplier == 0.7
        assert new_scheduler.current_val_error == 0.12
        assert new_scheduler.history == [0.15, 0.12]

    def test_validation_error_must_be_non_negative(self):
        """Test that update() validates non-negative errors."""
        scheduler = AdaptiveNoiseScheduler()

        scheduler.update(0.0)  # Valid
        scheduler.update(0.1)  # Valid

        with pytest.raises(ValueError, match="val_error must be non-negative"):
            scheduler.update(-0.1)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
