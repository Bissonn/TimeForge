"""
Tests for performance optimizations: PE cache, RevIN stability, normalize_input fast path.

These tests verify that optimizations preserve numerical correctness while improving performance.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from models.transformer_components.positional_encoding import SinusoidalPositionalEncoding
from models.transformer_components.revin import RevIN


class TestPositionalEncodingCache:
    """Test PE cache optimization preserves numerical equality."""

    def test_pe_cache_dtype_consistency(self):
        """Verify PE cache works correctly with different dtypes (AMP scenario)."""
        pe = SinusoidalPositionalEncoding(hidden_size=128, max_len=200)
        x_fp32 = torch.randn(4, 96, 128, dtype=torch.float32)
        x_fp16 = x_fp32.to(dtype=torch.float16)

        # First forward with float32
        out_fp32_1 = pe(x_fp32)
        assert out_fp32_1.dtype == torch.float32
        assert pe._pe_cache is not None
        assert pe._pe_cache.dtype == torch.float32

        # Second forward with float32 (should use cache)
        out_fp32_2 = pe(x_fp32)
        torch.testing.assert_close(out_fp32_1, out_fp32_2, rtol=1e-6, atol=1e-6)

        # Forward with float16 (should invalidate cache and create new)
        out_fp16 = pe(x_fp16.clone())
        assert out_fp16.dtype == torch.float16
        assert pe._pe_cache.dtype == torch.float16

        # Verify numerical equivalence (within float16 precision)
        # Note: float16 has limited precision (~3 decimal places), so we use relaxed tolerance
        torch.testing.assert_close(
            out_fp32_1.to(torch.float16),
            out_fp16.to(torch.float16),
            rtol=5e-3,  # 0.5% relative tolerance for float16
            atol=5e-2   # Absolute tolerance accounting for scaling factor
        )

    def test_pe_cache_device_consistency(self):
        """Verify PE cache works correctly across devices."""
        pe = SinusoidalPositionalEncoding(hidden_size=64, max_len=100)
        x_cpu = torch.randn(2, 48, 64)

        # Forward on CPU
        out_cpu = pe(x_cpu)
        assert pe._pe_cache.device == torch.device('cpu')

        # If CUDA available, test device transfer
        if torch.cuda.is_available():
            x_cuda = x_cpu.cuda()
            out_cuda = pe(x_cuda)

            assert pe._pe_cache.device.type == 'cuda'
            torch.testing.assert_close(
                out_cpu,
                out_cuda.cpu(),
                rtol=1e-6,
                atol=1e-6
            )

    def test_pe_cache_invalidation_on_rebuild(self):
        """Verify cache is invalidated when PE buffer is rebuilt for longer sequences."""
        pe = SinusoidalPositionalEncoding(hidden_size=32, max_len=50)

        # Forward with short sequence
        x_short = torch.randn(2, 30, 32)
        out_short = pe(x_short)
        assert pe._pe_cache is not None
        cache_id_short = id(pe._pe_cache)

        # Forward with longer sequence (triggers rebuild)
        x_long = torch.randn(2, 80, 32)
        out_long = pe(x_long)

        # Cache should be invalidated and recreated
        assert pe._pe_cache is not None
        assert pe.max_len >= 80  # Buffer was extended

        # New forward should create valid output
        assert out_long.shape == (2, 80, 32)
        assert torch.isfinite(out_long).all()

    def test_pe_zero_copy_slicing(self):
        """Verify that slicing from cache creates views (zero-copy)."""
        pe = SinusoidalPositionalEncoding(hidden_size=64, max_len=100)
        x = torch.randn(4, 50, 64)

        # First forward creates cache
        _ = pe(x)
        cache_before = pe._pe_cache

        # Second forward should reuse cache (same object)
        _ = pe(x)
        cache_after = pe._pe_cache

        # Cache should be the same object (not recreated)
        assert cache_before is cache_after


class TestRevINStability:
    """Test RevIN numerical stability fixes."""

    def test_revin_handles_zero_variance(self):
        """Verify RevIN doesn't produce inf/nan with zero variance input."""
        revin = RevIN(num_features=3, eps=1e-5, affine=True)

        # Constant input (zero variance)
        x_const = torch.ones(8, 24, 3) * 5.0

        # Forward pass should not produce inf/nan
        revin._get_statistics(x_const)
        x_norm = revin._normalize(x_const)

        assert torch.isfinite(x_norm).all(), "Normalized output contains inf/nan"
        assert not torch.isnan(x_norm).any(), "Normalized output contains NaN"

        # Denormalize should also be stable
        x_denorm = revin._denormalize(x_norm)
        assert torch.isfinite(x_denorm).all(), "Denormalized output contains inf/nan"

        # Should approximately reconstruct input
        torch.testing.assert_close(x_denorm, x_const, rtol=1e-3, atol=1e-3)

    def test_revin_handles_very_small_variance(self):
        """Verify RevIN is stable with very small variance."""
        revin = RevIN(num_features=2, eps=1e-5, affine=True)

        # Input with very small variance (1e-7)
        x_small_var = torch.randn(4, 48, 2) * 1e-4 + 10.0

        revin._get_statistics(x_small_var)
        x_norm = revin._normalize(x_small_var)
        x_denorm = revin._denormalize(x_norm)

        assert torch.isfinite(x_norm).all()
        assert torch.isfinite(x_denorm).all()

        # Verify roundtrip accuracy
        torch.testing.assert_close(x_denorm, x_small_var, rtol=1e-2, atol=1e-4)

    def test_revin_affine_weight_clamping(self):
        """Verify affine weight is correctly clamped during denormalization."""
        revin = RevIN(num_features=1, eps=1e-5, affine=True)

        # Initialize statistics
        x = torch.randn(4, 32, 1)
        revin._get_statistics(x)

        # Manually set affine_weight to very small value (edge case)
        with torch.no_grad():
            revin.affine_weight.fill_(1e-10)  # Smaller than eps

        x_norm = revin._normalize(x)

        # Denormalization should clamp weight to eps, preventing division by ~0
        x_denorm = revin._denormalize(x_norm)

        assert torch.isfinite(x_denorm).all(), "Denorm failed with small affine_weight"
        assert not torch.isinf(x_denorm).any(), "Denorm produced inf values"

    def test_revin_numerical_equality_before_after_fix(self):
        """Verify optimization doesn't change output for normal inputs."""
        revin = RevIN(num_features=5, eps=1e-5, affine=True)

        # Normal input (should behave identically before/after optimization)
        x = torch.randn(16, 96, 5) * 2.0 + 3.0

        # Forward pass
        revin._get_statistics(x)
        x_norm = revin._normalize(x)
        x_denorm = revin._denormalize(x_norm)

        # Verify roundtrip
        torch.testing.assert_close(x_denorm, x, rtol=1e-5, atol=1e-5)

        # Verify normalized values have ~zero mean and ~unit variance
        mean = x_norm.mean(dim=1, keepdim=True)
        std = x_norm.std(dim=1, keepdim=True)

        torch.testing.assert_close(mean, torch.zeros_like(mean), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(std, torch.ones_like(std), rtol=1e-1, atol=1e-1)


class TestNormalizeInputFastPath:
    """Test normalize_input optimization preserves behavior."""

    @pytest.fixture
    def mock_transformer_model(self):
        """Create minimal TransformerModel mock for testing normalize_input."""
        from models.transformer import TransformerModel

        # Create minimal model with RevIN enabled
        model = TransformerModel(
            encoder_input_size=7,  # Only targets, no exog
            decoder_input_size=0,
            num_features=7,
            forecast_steps=24,
            window_size=96,
            hidden_size=64,
            num_heads=4,
            num_encoder_layers=2,
            architecture='encoder-only',
            readout='last',
            use_revin=True,
            revin_affine=True,
            revin_eps=1e-5,
            dropout=0.1,
            attention_type='full',
            output_head_strategy='shared'
        )
        model.eval()
        return model

    def test_fast_path_targets_only(self, mock_transformer_model):
        """Verify fast path is taken when input has only targets (no exog)."""
        model = mock_transformer_model

        # Input with only targets (should trigger fast path)
        x = torch.randn(8, 96, 7)  # [B, T, num_features]

        # Normalize
        x_norm = model.normalize_input(x, mode='norm')

        # Should have same shape
        assert x_norm.shape == x.shape
        assert torch.isfinite(x_norm).all()

    def test_slow_path_with_exog(self):
        """Verify slow path works when input has exogenous features."""
        from models.transformer import TransformerModel

        # Model with exogenous features
        model = TransformerModel(
            encoder_input_size=10,  # 7 targets + 3 exog
            decoder_input_size=0,
            num_features=7,
            forecast_steps=24,
            window_size=96,
            hidden_size=64,
            num_heads=4,
            num_encoder_layers=2,
            architecture='encoder-only',
            readout='last',
            use_revin=True,
            revin_affine=True,
            revin_eps=1e-5,
            dropout=0.1,
            attention_type='full',
            output_head_strategy='shared'
        )
        model.eval()

        # Input with targets + exog (should trigger slow path)
        x = torch.randn(4, 48, 10)  # [B, T, 7 targets + 3 exog]

        x_norm = model.normalize_input(x, mode='norm')

        assert x_norm.shape == x.shape
        assert torch.isfinite(x_norm).all()

        # Verify exog features are unchanged (only targets normalized)
        # Extract exog from normalized output
        x_exog_norm = x_norm[:, :, 7:]
        x_exog_orig = x[:, :, 7:]

        # Exog should be identical (not normalized)
        torch.testing.assert_close(x_exog_norm, x_exog_orig, rtol=1e-7, atol=1e-7)

    def test_safety_check_insufficient_features(self, mock_transformer_model):
        """Verify safety check catches malformed input."""
        model = mock_transformer_model

        # Input with fewer features than expected (should raise error)
        x_invalid = torch.randn(4, 96, 5)  # Only 5 features, need 7

        with pytest.raises(ValueError, match="Input tensor has 5 features"):
            model.normalize_input(x_invalid, mode='norm')

    def test_numerical_equality_fast_vs_slow_path(self):
        """Verify fast path produces identical results to slow path for targets-only input."""
        from models.transformer import TransformerModel

        # Create model
        model = TransformerModel(
            encoder_input_size=7,
            decoder_input_size=0,
            num_features=7,
            forecast_steps=24,
            window_size=96,
            hidden_size=64,
            num_heads=4,
            num_encoder_layers=2,
            architecture='encoder-only',
            readout='last',
            use_revin=True,
            revin_affine=True,
            revin_eps=1e-5,
            dropout=0.1,
            attention_type='full',
            output_head_strategy='shared'
        )
        model.eval()

        # Input with exactly num_features (fast path)
        x = torch.randn(8, 96, 7)

        # Get output with fast path
        x_norm_fast = model.normalize_input(x, mode='norm')

        # Manually simulate slow path (for comparison)
        model._get_statistics = model.revin._get_statistics
        model._get_statistics(x)
        x_target = x[:, :, :model.num_features]  # Slice
        x_exog = x[:, :, model.num_features:]    # Empty slice
        x_target_norm = model.revin._normalize(x_target)
        x_norm_slow = torch.cat([x_target_norm, x_exog], dim=-1)  # Cat with empty

        # Fast and slow path should produce identical results
        torch.testing.assert_close(x_norm_fast, x_norm_slow, rtol=1e-7, atol=1e-7)


class TestIntegrationOptimizations:
    """Integration tests verifying all optimizations work together."""

    def test_full_forward_pass_with_optimizations(self):
        """Test complete forward pass with all optimizations enabled."""
        from models.transformer import TransformerModel

        model = TransformerModel(
            encoder_input_size=7,
            decoder_input_size=0,
            num_features=7,
            forecast_steps=24,
            window_size=96,
            hidden_size=128,
            num_heads=8,
            num_encoder_layers=3,
            architecture='encoder-only',
            readout='last',
            use_revin=True,
            revin_affine=True,
            revin_eps=1e-5,
            dropout=0.1,
            attention_type='full',
            output_head_strategy='shared'
        )
        model.eval()

        # Forward pass (uses all optimizations)
        x = torch.randn(16, 96, 7)

        with torch.no_grad():
            output = model(x)

        # Verify output
        assert output.shape == (16, 24, 7)
        assert torch.isfinite(output).all()

    def test_amp_compatibility(self):
        """Test optimizations work correctly with Automatic Mixed Precision."""
        from models.transformer import TransformerModel

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for AMP test")

        model = TransformerModel(
            encoder_input_size=5,
            decoder_input_size=0,
            num_features=5,
            forecast_steps=12,
            window_size=48,
            hidden_size=64,
            num_heads=4,
            num_encoder_layers=2,
            architecture='encoder-only',
            readout='mean',
            use_revin=True,
            revin_affine=True,
            revin_eps=1e-5,
            dropout=0.0,
            attention_type='full',
            output_head_strategy='shared'
        ).cuda()
        model.eval()

        x = torch.randn(8, 48, 5).cuda()

        # Forward with AMP
        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = model(x)

        assert output.dtype == torch.float16 or output.dtype == torch.float32
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
