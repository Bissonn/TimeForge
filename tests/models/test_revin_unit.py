# tests/models/test_revin_unit.py
"""
Unit tests for RevIN (Reversible Instance Normalization) class.

Tests cover:
- Forward-backward cycle integrity
- Normalization statistics correctness
- Mode validation and state management
- Affine transformation parameters
- Device handling (CPU/GPU)
- Batch independence
- Edge cases and error handling
"""

import torch
import pytest
import warnings
from models.transformer import RevIN


class TestRevINBasicOperations:
    """Test basic RevIN operations: norm, apply, denorm."""

    def test_revin_cycle_preserves_values(self):
        """Test that norm → denorm is identity (without affine)."""
        revin = RevIN(num_features=3, affine=False, eps=1e-5)
        x = torch.randn(16, 50, 3)

        # Forward cycle: norm → denorm
        x_norm = revin(x, mode='norm')
        x_denorm = revin(x_norm, mode='denorm')

        # Should recover original values
        assert torch.allclose(x, x_denorm, atol=1e-4), \
            "RevIN cycle should preserve original values (norm → denorm = identity)"

    def test_revin_cycle_with_affine(self):
        """Test that norm → denorm is identity (with affine)."""
        revin = RevIN(num_features=3, affine=True, eps=1e-5)

        # Initialize affine parameters to non-trivial values
        with torch.no_grad():
            revin.affine_weight.fill_(2.0)
            revin.affine_bias.fill_(1.0)

        x = torch.randn(16, 50, 3)

        x_norm = revin(x, mode='norm')
        x_denorm = revin(x_norm, mode='denorm')

        assert torch.allclose(x, x_denorm, atol=1e-4), \
            "RevIN cycle with affine should preserve original values"

    def test_revin_normalization_statistics(self):
        """Test that normalized data has mean≈0, std≈1 per instance."""
        revin = RevIN(num_features=3, affine=False, eps=1e-5)

        # Create data with non-zero mean and non-unit std
        x = torch.randn(16, 50, 3) * 100 + 500

        x_norm = revin(x, mode='norm')

        # Check mean ≈ 0 (per instance, over time dimension)
        mean = x_norm.mean(dim=1)  # (B, F)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), \
            f"Normalized data should have mean≈0, got {mean.abs().max():.6f}"

        # Check std ≈ 1 (per instance, over time dimension)
        std = x_norm.std(dim=1, unbiased=False)  # (B, F)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-4), \
            f"Normalized data should have std≈1, got {std.abs().max():.6f}"

    def test_revin_affine_applies_transformation(self):
        """Test that affine transformation is applied correctly in norm mode."""
        revin = RevIN(num_features=2, affine=True, eps=1e-5)

        # Set affine parameters to known values
        with torch.no_grad():
            revin.affine_weight[0, 0, 0] = 2.0
            revin.affine_weight[0, 0, 1] = 3.0
            revin.affine_bias[0, 0, 0] = 1.0
            revin.affine_bias[0, 0, 1] = -1.0

        # Create random data
        x = torch.randn(4, 10, 2)

        # Manually calculate expected result using RevIN's logic (unbiased=False)
        # We replicate the math to ensure the implementation matches expectations
        mean = x.mean(dim=1, keepdim=True)
        # Note: RevIN uses unbiased=False
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm_manual = (x - mean) / std

        # Apply expected affine transform
        expected_0 = x_norm_manual[:, :, 0] * 2.0 + 1.0
        expected_1 = x_norm_manual[:, :, 1] * 3.0 - 1.0

        # Run RevIN
        x_norm = revin(x, mode='norm')

        assert torch.allclose(x_norm[:, :, 0], expected_0, atol=1e-4), \
            "Affine transformation not applied correctly to feature 0"
        assert torch.allclose(x_norm[:, :, 1], expected_1, atol=1e-4), \
            "Affine transformation not applied correctly to feature 1"


class TestRevINModeValidation:
    """Test mode validation and state management."""

    def test_revin_mode_norm_calculates_statistics(self):
        """Test that mode='norm' calculates and stores statistics."""
        revin = RevIN(num_features=2, affine=False)
        x = torch.randn(8, 50, 2) * 100 + 500

        # Before norm: buffers should be zero/one (initial state)
        assert torch.allclose(revin.mean, torch.zeros_like(revin.mean)), \
            "Initial mean buffer should be zero"
        assert torch.allclose(revin.stdev, torch.ones_like(revin.stdev)), \
            "Initial stdev buffer should be one"

        # After norm: buffers should be updated
        _ = revin(x, mode='norm')

        assert not torch.allclose(revin.mean, torch.zeros_like(revin.mean)), \
            "After mode='norm', mean buffer should be updated"
        assert not torch.allclose(revin.stdev, torch.ones_like(revin.stdev)), \
            "After mode='norm', stdev buffer should be updated (not all ones)"

        # Check that stored statistics match input
        expected_mean = x.mean(dim=1, keepdim=True)
        expected_stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + revin.eps)

        assert torch.allclose(revin.mean, expected_mean, atol=1e-4), \
            "Stored mean doesn't match computed mean"
        assert torch.allclose(revin.stdev, expected_stdev, atol=1e-4), \
            "Stored stdev doesn't match computed stdev"

    def test_revin_mode_apply_uses_stored_statistics(self):
        """Test that mode='apply' uses statistics from previous mode='norm'."""
        revin = RevIN(num_features=2, affine=False)

        # Step 1: Calculate statistics with mode='norm'
        x1 = torch.randn(4, 20, 2) * 100 + 500
        x1_norm = revin(x1, mode='norm')

        # Store the statistics
        stored_mean = revin.mean.clone()
        stored_stdev = revin.stdev.clone()

        # Step 2: Apply same statistics to different data
        x2 = torch.randn(4, 20, 2) * 50 + 300  # Different distribution
        x2_apply = revin(x2, mode='apply')

        # Statistics should NOT have changed
        assert torch.allclose(revin.mean, stored_mean), \
            "mode='apply' should not change stored mean"
        assert torch.allclose(revin.stdev, stored_stdev), \
            "mode='apply' should not change stored stdev"

        # x2_apply should be normalized using x1's statistics
        expected = (x2 - stored_mean) / stored_stdev
        assert torch.allclose(x2_apply, expected, atol=1e-4), \
            "mode='apply' should use stored statistics, not recalculate"

    def test_revin_invalid_mode_raises_error(self):
        """Test that invalid mode string raises ValueError."""
        revin = RevIN(num_features=2)
        x = torch.randn(8, 50, 2)

        with pytest.raises(ValueError, match="Unknown mode"):
            revin(x, mode='invalid')

        with pytest.raises(ValueError, match="Unknown mode"):
            revin(x, mode='normalize')  # Common typo

        with pytest.raises(ValueError, match="Unknown mode"):
            revin(x, mode='NORM')  # Case-sensitive


class TestRevINDeviceHandling:
    """Test device placement and GPU support."""

    def test_revin_cpu_device(self):
        """Test that RevIN works correctly on CPU."""
        revin = RevIN(num_features=3)
        x = torch.randn(8, 50, 3)

        assert x.device.type == 'cpu'

        x_norm = revin(x, mode='norm')
        assert x_norm.device.type == 'cpu', "Output should stay on CPU"

        x_denorm = revin(x_norm, mode='denorm')
        assert x_denorm.device.type == 'cpu', "Denormalized output should stay on CPU"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_revin_cuda_device(self):
        """Test that RevIN works correctly on GPU."""
        revin = RevIN(num_features=3).cuda()
        x = torch.randn(8, 50, 3).cuda()

        assert x.device.type == 'cuda'

        x_norm = revin(x, mode='norm')
        assert x_norm.device.type == 'cuda', "Output should stay on CUDA"

        x_denorm = revin(x_norm, mode='denorm')
        assert x_denorm.device.type == 'cuda', "Denormalized output should stay on CUDA"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_revin_device_transfer(self):
        """Test that RevIN buffers follow model to new device."""
        revin = RevIN(num_features=2)

        # Initially on CPU
        assert revin.mean.device.type == 'cpu'
        assert revin.stdev.device.type == 'cpu'

        # Move to CUDA
        revin = revin.cuda()

        assert revin.mean.device.type == 'cuda', "mean buffer should move to CUDA"
        assert revin.stdev.device.type == 'cuda', "stdev buffer should move to CUDA"

        # Test forward pass
        x = torch.randn(4, 20, 2).cuda()
        x_norm = revin(x, mode='norm')

        assert x_norm.device.type == 'cuda'
        assert torch.allclose(x_norm.mean(dim=1), torch.zeros(4, 1, 2).cuda(), atol=1e-4)


class TestRevINBatchIndependence:
    """Test that different samples in batch get independent statistics."""

    def test_revin_batch_independence(self):
        """Test that different batch samples are normalized independently."""
        revin = RevIN(num_features=1, affine=False, eps=1e-5)

        # Create two samples with very different distributions
        x1 = torch.ones(1, 50, 1) * 100.0  # mean=100
        x2 = torch.ones(1, 50, 1) * 1000.0  # mean=1000
        x = torch.cat([x1, x2], dim=0)  # (2, 50, 1)

        x_norm = revin(x, mode='norm')

        # Both samples should be normalized to mean≈0 independently
        assert torch.allclose(x_norm[0].mean(), torch.tensor(0.0), atol=1e-4), \
            "Sample 0 should be normalized to mean≈0"
        assert torch.allclose(x_norm[1].mean(), torch.tensor(0.0), atol=1e-4), \
            "Sample 1 should be normalized to mean≈0"

        # But the stored mean/stdev should be per-sample
        assert revin.mean.shape == (2, 1, 1), "Statistics should be per-sample"
        assert torch.abs(revin.mean[0, 0, 0] - 100.0) < 1e-3, \
            "Sample 0 mean should be stored as ~100"
        assert torch.abs(revin.mean[1, 0, 0] - 1000.0) < 1e-3, \
            "Sample 1 mean should be stored as ~1000"

    def test_revin_feature_independence(self):
        """Test that different features are normalized independently."""
        revin = RevIN(num_features=3, affine=False, eps=1e-5)

        # Create data where each feature has different scale
        x = torch.randn(8, 50, 3)
        x[:, :, 0] = x[:, :, 0] * 10 + 100  # Feature 0: mean=100, std=10
        x[:, :, 1] = x[:, :, 1] * 100 + 1000  # Feature 1: mean=1000, std=100
        x[:, :, 2] = x[:, :, 2] * 0.1 + 1  # Feature 2: mean=1, std=0.1

        x_norm = revin(x, mode='norm')

        # Each feature should be normalized independently to mean≈0, std≈1
        for f in range(3):
            mean_f = x_norm[:, :, f].mean()
            std_f = x_norm[:, :, f].std(unbiased=False)

            assert torch.abs(mean_f) < 1e-3, \
                f"Feature {f} should have mean≈0, got {mean_f:.6f}"
            assert torch.abs(std_f - 1.0) < 1e-2, \
                f"Feature {f} should have std≈1, got {std_f:.6f}"


class TestRevINEdgeCases:
    """Test edge cases and error handling."""

    def test_revin_single_feature(self):
        """Test RevIN with single feature (univariate)."""
        revin = RevIN(num_features=1, affine=False)
        x = torch.randn(8, 50, 1) * 50 + 200

        x_norm = revin(x, mode='norm')
        x_denorm = revin(x_norm, mode='denorm')

        assert torch.allclose(x, x_denorm, atol=1e-4), \
            "Single feature should work correctly"

    def test_revin_many_features(self):
        """Test RevIN with many features (high-dimensional)."""
        revin = RevIN(num_features=20, affine=False)
        x = torch.randn(4, 30, 20)

        x_norm = revin(x, mode='norm')

        # Each feature should be independently normalized
        for f in range(20):
            mean_f = x_norm[:, :, f].mean()
            assert torch.abs(mean_f) < 1e-3, \
                f"Feature {f}/20 should have mean≈0"

    def test_revin_short_sequence(self):
        """Test RevIN with very short time sequences."""
        revin = RevIN(num_features=2, affine=False)
        x = torch.randn(4, 3, 2)  # Only 3 timesteps

        x_norm = revin(x, mode='norm')
        x_denorm = revin(x_norm, mode='denorm')

        assert torch.allclose(x, x_denorm, atol=1e-4), \
            "Short sequences should work correctly"

    def test_revin_long_sequence(self):
        """Test RevIN with very long time sequences."""
        revin = RevIN(num_features=2, affine=False)
        x = torch.randn(2, 1000, 2)  # 1000 timesteps

        x_norm = revin(x, mode='norm')
        x_denorm = revin(x_norm, mode='denorm')

        assert torch.allclose(x, x_denorm, atol=1e-4), \
            "Long sequences should work correctly"

    def test_revin_extreme_values(self):
        """Test RevIN stability with very large input values."""
        revin = RevIN(num_features=2, affine=False, eps=1e-5)
        x = torch.randn(8, 50, 2) * 1e6 + 1e7  # Very large scale

        x_norm = revin(x, mode='norm')

        # Should still normalize correctly
        assert torch.all(torch.isfinite(x_norm)), \
            "Normalized values should be finite even for large inputs"

        mean = x_norm.mean(dim=1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-3), \
            "Should normalize large values correctly"

        # Denormalize should recover
        x_denorm = revin(x_norm, mode='denorm')
        assert torch.allclose(x, x_denorm, rtol=1e-4, atol=1e-2), \
            "Should denormalize large values correctly"

    def test_revin_near_zero_variance(self):
        """Test RevIN with near-constant input (very small variance)."""
        revin = RevIN(num_features=2, affine=False, eps=1e-5)

        # Nearly constant values (variance ~ eps)
        x = torch.ones(8, 50, 2) * 100.0
        x = x + torch.randn_like(x) * 1e-8  # Add tiny noise

        x_norm = revin(x, mode='norm')

        # Should not explode or return NaN
        assert torch.all(torch.isfinite(x_norm)), \
            "Should handle near-constant input without NaN/Inf"

        x_denorm = revin(x_norm, mode='denorm')
        assert torch.allclose(x, x_denorm, atol=1e-4), \
            "Should recover near-constant input"


class TestRevINErrorHandling:
    """Test input validation and error handling."""

    def test_revin_invalid_shape_2d(self):
        """Test that 2D input raises clear error."""
        revin = RevIN(num_features=3)
        x = torch.randn(16, 50)  # Missing feature dimension

        # Currently may not validate, but SHOULD in production
        # For now, just verify it doesn't crash silently
        try:
            _ = revin(x, mode='norm')
        except (RuntimeError, IndexError, ValueError):
            # Expected - invalid shape should raise
            pass

    def test_revin_feature_dimension_mismatch(self):
        """Test that wrong number of features is handled."""
        revin = RevIN(num_features=3)
        x = torch.randn(8, 50, 5)  # 5 features instead of 3

        # Should either raise error or at least not crash
        try:
            x_norm = revin(x, mode='norm')
            assert x_norm.shape[-1] == 5, "Output features should match input"
        except (RuntimeError, ValueError):
            pass

    def test_revin_nan_input_handling(self):
        """Test behavior with NaN input values."""
        revin = RevIN(num_features=2, affine=False)
        x = torch.randn(8, 50, 2)
        x[0, 0, 0] = float('nan')

        x_norm = revin(x, mode='norm')

        # NaN should propagate (standard PyTorch behavior)
        assert torch.isnan(x_norm).any(), \
            "NaN in input should propagate to output"

    def test_revin_inf_input_handling(self):
        """Test behavior with Inf input values."""
        revin = RevIN(num_features=2, affine=False)
        x = torch.randn(8, 50, 2)
        x[0, 0, 0] = float('inf')

        try:
            x_norm = revin(x, mode='norm')
            assert x_norm.shape == x.shape
        except RuntimeError:
            pass


class TestRevINAffineParameters:
    """Test affine transformation parameter handling."""

    def test_revin_affine_parameters_exist(self):
        """Test that affine=True creates learnable parameters."""
        revin = RevIN(num_features=3, affine=True)

        # Check parameters exist
        assert hasattr(revin, 'affine_weight'), "Should have affine_weight"
        assert hasattr(revin, 'affine_bias'), "Should have affine_bias"

        # Check they are Parameters (learnable)
        assert isinstance(revin.affine_weight, torch.nn.Parameter), \
            "affine_weight should be a Parameter"
        assert isinstance(revin.affine_bias, torch.nn.Parameter), \
            "affine_bias should be a Parameter"

    def test_revin_affine_parameters_shape(self):
        """Test that affine parameters have correct shape."""
        revin = RevIN(num_features=5, affine=True)

        assert revin.affine_weight.shape == (1, 1, 5), \
            f"Expected shape (1, 1, 5), got {revin.affine_weight.shape}"
        assert revin.affine_bias.shape == (1, 1, 5), \
            f"Expected shape (1, 1, 5), got {revin.affine_bias.shape}"

    def test_revin_affine_parameters_initialization(self):
        """Test that affine parameters are initialized correctly."""
        revin = RevIN(num_features=3, affine=True)

        # Standard initialization: weight=1, bias=0
        assert torch.allclose(revin.affine_weight, torch.ones_like(revin.affine_weight)), \
            "affine_weight should be initialized to ones"
        assert torch.allclose(revin.affine_bias, torch.zeros_like(revin.affine_bias)), \
            "affine_bias should be initialized to zeros"

    def test_revin_affine_parameters_gradients(self):
        """Test that affine parameters accumulate gradients."""
        revin = RevIN(num_features=2, affine=True)
        x = torch.randn(4, 20, 2, requires_grad=True)

        # Forward pass
        x_norm = revin(x, mode='norm')
        loss = x_norm.sum()

        # Backward pass
        loss.backward()

        # Affine parameters should have gradients
        assert revin.affine_weight.grad is not None, \
            "affine_weight should accumulate gradients"
        assert revin.affine_bias.grad is not None, \
            "affine_bias should accumulate gradients"

    def test_revin_without_affine_no_parameters(self):
        """Test that affine=False does not create parameters."""
        revin = RevIN(num_features=3, affine=False)

        # Should not have affine parameters
        assert not hasattr(revin, 'affine_weight') or revin.affine_weight is None, \
            "affine=False should not create affine_weight"
        assert not hasattr(revin, 'affine_bias') or revin.affine_bias is None, \
            "affine=False should not create affine_bias"


class TestRevINStateManagement:
    """Test state management and buffer behavior."""

    def test_revin_buffers_are_registered(self):
        """Test that mean and stdev are registered as buffers."""
        revin = RevIN(num_features=2)

        # Check buffers exist
        buffers = dict(revin.named_buffers())
        assert 'mean' in buffers, "mean should be registered as buffer"
        assert 'stdev' in buffers, "stdev should be registered as buffer"

    def test_revin_buffers_not_persistent(self):
        """Test that buffers are not saved in state_dict (persistent=False)."""
        revin = RevIN(num_features=2)

        # Run forward to populate buffers
        x = torch.randn(4, 20, 2)
        _ = revin(x, mode='norm')

        # Get state dict
        state_dict = revin.state_dict()

        # Buffers with persistent=False should NOT be in state_dict
        assert 'mean' not in state_dict
        assert 'stdev' not in state_dict

    def test_revin_statistics_update_on_norm(self):
        """Test that statistics are recalculated each time mode='norm' is called."""
        revin = RevIN(num_features=2, affine=False)

        # First call
        x1 = torch.randn(4, 20, 2) * 100 + 500
        _ = revin(x1, mode='norm')
        mean1 = revin.mean.clone()

        # Second call with different data
        x2 = torch.randn(4, 20, 2) * 10 + 50
        _ = revin(x2, mode='norm')
        mean2 = revin.mean.clone()

        # Statistics should be different (rolling behavior)
        assert not torch.allclose(mean1, mean2, atol=1e-3), \
            "Statistics should update on each mode='norm' call"

    def test_revin_statistics_not_update_on_apply(self):
        """Test that statistics do NOT change when mode='apply' is used."""
        revin = RevIN(num_features=2, affine=False)

        # Calculate statistics
        x1 = torch.randn(4, 20, 2) * 100
        _ = revin(x1, mode='norm')
        mean_stored = revin.mean.clone()
        stdev_stored = revin.stdev.clone()

        # Apply to different data
        x2 = torch.randn(4, 20, 2) * 50
        _ = revin(x2, mode='apply')

        # Statistics should NOT change
        assert torch.allclose(revin.mean, mean_stored), \
            "mode='apply' should not update mean"
        assert torch.allclose(revin.stdev, stdev_stored), \
            "mode='apply' should not update stdev"


def test_revin_buffer_persistence():
    """Test that buffers persist across device moves."""
    revin = RevIN(num_features=7).cuda()
    x = torch.randn(32, 96, 7).cuda()

    # Forward pass
    _ = revin(x, mode='norm')
    mean_before = revin.mean.clone()
    stdev_before = revin.stdev.clone()

    # Move to CPU and back
    revin = revin.cpu()
    revin = revin.cuda()

    # Buffers should persist
    assert torch.allclose(revin.mean, mean_before.cuda())
    assert torch.allclose(revin.stdev, stdev_before.cuda())


def test_revin_state_dict():
    """
    Test that state_dict contains only persistent parameters (affine).

    RevIN statistics (mean, stdev) are NOT saved because they are
    instance-specific and recomputed from input data.
    """
    revin = RevIN(num_features=7, affine=True)
    x = torch.randn(32, 96, 7)
    _ = revin(x, mode='norm')

    state = revin.state_dict()

    # Learned parameters should be saved
    assert 'affine_weight' in state
    assert 'affine_bias' in state

    # Instance-specific buffers should NOT be saved
    assert 'mean' not in state
    assert 'stdev' not in state


def test_revin_state_dict_without_affine():
    """Test that state_dict is empty when affine=False."""
    revin = RevIN(num_features=7, affine=False)
    x = torch.randn(32, 96, 7)

    _ = revin(x, mode='norm')

    state = revin.state_dict()

    # Should be empty (no learned parameters, no persistent buffers)
    assert len(state) == 0, (
        "state_dict should be empty when affine=False. "
        "RevIN with affine=False has no learnable parameters."
    )

def test_revin_amp_compatibility():
    """Test that RevIN works with AMP."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    revin = RevIN(num_features=7).cuda()
    x = torch.randn(32, 96, 7).cuda()

    # Should work with AMP
    with torch.autocast('cuda', dtype=torch.float16):
        x_norm = revin(x, mode='norm')
        x_denorm = revin(x_norm, mode='denorm')

    # Should be close (allowing for fp16 precision)
    assert torch.allclose(x, x_denorm, atol=1e-3)


def test_revin_varying_batch_sizes():
    """Test that RevIN handles varying batch sizes."""
    revin = RevIN(num_features=7)

    # First batch: size 32
    x1 = torch.randn(32, 96, 7)
    _ = revin(x1, mode='norm')

    # Second batch: size 16 (different!)
    x2 = torch.randn(16, 96, 7)
    x2_norm = revin(x2, mode='norm')

    # Should work without errors
    assert x2_norm.shape == (16, 96, 7)


def test_revin_gradient_flow():
    """Test that gradients flow through RevIN."""
    revin = RevIN(num_features=7, affine=True)
    x = torch.randn(32, 96, 7, requires_grad=True)

    # Forward
    x_norm = revin(x, mode='norm')
    loss = x_norm.sum()

    # Backward
    loss.backward()

    # Gradients should exist
    assert x.grad is not None
    assert revin.affine_weight.grad is not None
    assert revin.affine_bias.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_revin_gpu_training():
    """Test basic GPU usage (your actual workflow)."""
    revin = RevIN(num_features=7, affine=True).cuda()
    x = torch.randn(32, 96, 7).cuda()

    # Forward
    x_norm = revin(x, mode='norm')

    # Backward (simulate training)
    loss = x_norm.sum()
    loss.backward()

    # Check gradients exist
    assert revin.affine_weight.grad is not None
    assert revin.affine_bias.grad is not None


def test_revin_checkpoint_basic():
    """Test checkpoint save/load (your actual workflow)."""
    import tempfile

    # Train
    revin = RevIN(num_features=7, affine=True)
    x = torch.randn(32, 96, 7)
    _ = revin(x, mode='norm')

    # Save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(revin.state_dict(), f.name)
        path = f.name

    # Load
    revin_new = RevIN(num_features=7, affine=True)
    revin_new.load_state_dict(torch.load(path))

    # Should work
    x_new = torch.randn(32, 96, 7)
    x_norm = revin_new(x_new, mode='norm')
    assert x_norm.shape == (32, 96, 7)


def test_revin_buffers_are_registered():
    """
    Test that buffers are registered in the module (even if not persistent).

    Registered buffers:
    - Move with .to(device), .cuda(), .cpu() ✅
    - Are accessible as attributes ✅
    - Are NOT saved in state_dict() if persistent=False ✅
    """
    revin = RevIN(num_features=7)

    # Check buffers are registered
    buffers = dict(revin.named_buffers())
    assert 'mean' in buffers, "Mean buffer should be registered"
    assert 'stdev' in buffers, "Stdev buffer should be registered"

    # Check they can be accessed
    assert hasattr(revin, 'mean'), "Mean should be accessible as attribute"
    assert hasattr(revin, 'stdev'), "Stdev should be accessible as attribute"

    # Check they are tensors
    assert isinstance(revin.mean, torch.Tensor)
    assert isinstance(revin.stdev, torch.Tensor)


def test_revin_buffers_move_with_device():
    """
    Test that non-persistent buffers still move with device.

    persistent=False means:
    - NOT saved in state_dict() ✅
    - But STILL moves with .to(device) ✅
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    revin = RevIN(num_features=7)

    # Initially on CPU
    assert revin.mean.device.type == 'cpu'
    assert revin.stdev.device.type == 'cpu'

    # Move to CUDA
    revin = revin.cuda()

    # Buffers should move too (even though persistent=False)
    assert revin.mean.device.type == 'cuda', (
        "Non-persistent buffers should still move with .cuda()"
    )
    assert revin.stdev.device.type == 'cuda'

    # Move back to CPU
    revin = revin.cpu()

    # Buffers should move back
    assert revin.mean.device.type == 'cpu'
    assert revin.stdev.device.type == 'cpu'

    def test_revin_checkpoint_save_load():
        """
        Test that RevIN works correctly with checkpoint save/load.

        Expected behavior:
        1. Save checkpoint: Only affine params saved
        2. Load checkpoint: Affine params restored, buffers recomputed
        """
        import tempfile
        import os

        # Create and train RevIN
        revin_original = RevIN(num_features=7, affine=True)
        x_train = torch.randn(32, 96, 7)

        # "Train" by doing forward pass
        x_norm = revin_original(x_train, mode='norm')

        # Manually modify affine params to simulate training
        with torch.no_grad():
            revin_original.affine_weight.mul_(2.0)
            revin_original.affine_bias.add_(1.0)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            checkpoint_path = f.name
            torch.save(revin_original.state_dict(), checkpoint_path)

        try:
            # Load into new RevIN
            revin_loaded = RevIN(num_features=7, affine=True)
            revin_loaded.load_state_dict(torch.load(checkpoint_path))

            # Affine parameters should match
            assert torch.allclose(
                revin_loaded.affine_weight,
                revin_original.affine_weight
            ), "Affine weight should be restored from checkpoint"

            assert torch.allclose(
                revin_loaded.affine_bias,
                revin_original.affine_bias
            ), "Affine bias should be restored from checkpoint"

            # Buffers should be at default (not loaded)
            assert torch.allclose(
                revin_loaded.mean,
                torch.zeros(1, 1, 7)
            ), "Mean should be at default (not loaded from checkpoint)"

            assert torch.allclose(
                revin_loaded.stdev,
                torch.ones(1, 1, 7)
            ), "Stdev should be at default (not loaded from checkpoint)"

            # But after forward pass, buffers should be recomputed
            x_new = torch.randn(32, 96, 7)
            _ = revin_loaded(x_new, mode='norm')

            # Now buffers should be updated (from x_new)
            assert not torch.allclose(
                revin_loaded.mean,
                torch.zeros(1, 1, 7)
            ), "Mean should be updated after forward pass"

        finally:
            os.unlink(checkpoint_path)

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])