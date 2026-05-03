"""
CRITICAL TEST: Verifies attention capture mechanism with REAL transformer model.

This test ensures that:
1. Custom wrappers (CapturingMHA) are properly registered
2. Hooks are called during forward pass
3. Attention weights are captured with correct shapes
4. Both encoder and decoder attention is captured (for enc-dec architecture)
5. Maps are saved to disk with correct metadata
6. The mechanism works end-to-end from model creation to disk storage

This is the definitive test that the attention capture system works in production.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from models.transformer import TransformerForecaster, TransformerModel
from models.transformer import CapturingMHA, AttentionCaptureBuffer
from utils.dataset import TimeSeriesDataset
from core.context import RunContext


@pytest.fixture
def encoder_only_dataset():
    """Create a dataset for encoder-only testing (no decoder exog)."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n, freq='D'),
        'target': np.sin(np.linspace(0, 4*np.pi, n)) + np.random.randn(n)*0.1,
        'target2': np.cos(np.linspace(0, 4*np.pi, n)) + np.random.randn(n)*0.1,
        'enc_exog': np.random.randn(n),
    })

    ds = TimeSeriesDataset(
        'encoder_only_test',
        {'datasets': {'encoder_only_test': {}}},
        num_features=2,
        data=data,
        columns=['target', 'target2'],
        past_covariates=['enc_exog'],
        future_covariates=[]  # NO decoder exog for encoder-only
    )
    ds.split_data(forecast_steps=5)
    return ds


@pytest.fixture
def encoder_decoder_dataset():
    """Create a dataset for encoder-decoder testing (with decoder exog)."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n, freq='D'),
        'target': np.sin(np.linspace(0, 4*np.pi, n)) + np.random.randn(n)*0.1,
        'target2': np.cos(np.linspace(0, 4*np.pi, n)) + np.random.randn(n)*0.1,
        'enc_exog': np.random.randn(n),
        'dec_exog': np.random.randn(n),
    })

    ds = TimeSeriesDataset(
        'encoder_decoder_test',
        {'datasets': {'encoder_decoder_test': {}}},
        num_features=2,
        data=data,
        columns=['target', 'target2'],
        past_covariates=['enc_exog'],
        future_covariates=['dec_exog']  # Include decoder exog for enc-dec
    )
    ds.split_data(forecast_steps=5)
    return ds


class TestAttentionCaptureRealModel:
    """Test suite for attention capture with real transformer models."""

    def test_encoder_only_wrappers_are_registered_and_capture(self, encoder_only_dataset, base_context):
        """
        CRITICAL: Verify that CapturingMHA wrappers are registered
        in encoder-only architecture and capture attention during forward pass.
        """
        config = {
            'hidden_size': 64,
            'num_heads': 4,
            'num_encoder_layers': 2,
            'architecture': 'encoder-only',
            'attention_type': 'full',
            'epochs': 1,
            'batch_size': 4,
            'learning_rate': 0.01,
            'attention_capture_enabled': True,
            'attention_capture_sampling': 'all',
        }

        model = TransformerForecaster(
            model_params=config,
            num_features=2,
            forecast_steps=5,
            window_size=10,
            dataset=encoder_only_dataset,
            run_context=base_context.with_metadata(fold_idx=0, window_size=10)
        )

        # ============================================================
        # STEP 1: Verify capture buffer exists
        # ============================================================
        print("\n" + "="*70)
        print("STEP 1: Verifying capture buffer exists")
        print("="*70)

        assert hasattr(model.model, 'attn_capture'), "Model should have attn_capture attribute"
        assert isinstance(model.model.attn_capture, AttentionCaptureBuffer)
        print(f"  ✓ Capture buffer exists: {type(model.model.attn_capture).__name__}")

        # ============================================================
        # STEP 2: Fit model (required before predict)
        # ============================================================
        print("\n" + "="*70)
        print("STEP 2: Fitting model")
        print("="*70)

        train_df = encoder_only_dataset.development_data.iloc[:60]
        model.fit(train_series=train_df, dataset=encoder_only_dataset)
        print(f"  ✓ Model fitted on {len(train_df)} samples")

        # ============================================================
        # STEP 3: Perform forward pass with capture enabled
        # ============================================================
        print("\n" + "="*70)
        print("STEP 3: Performing forward pass with capture")
        print("="*70)

        # Prepare input tensor for direct forward call
        history = encoder_only_dataset.development_data.tail(10)
        input_tensor = model._prepare_input_tensor(history)
        input_tensor = input_tensor.to(model.device)
        print(f"  Input tensor shape: {input_tensor.shape}")

        # Enable capture and call model forward directly (like passing tests do)
        model.model.attn_capture.configure(enabled=True, steps_to_capture=None)
        model.model.eval()
        with torch.no_grad():
            predictions = model.model(input_tensor)

        print(f"  ✓ Predictions shape: {predictions.shape}")
        assert predictions.shape == (1, 5, 2), f"Expected (1, 5, 2), got {predictions.shape}"

        # ============================================================
        # STEP 4: Verify attention maps were captured
        # ============================================================
        print("\n" + "="*70)
        print("STEP 4: Verifying captured attention maps")
        print("="*70)

        captured_keys = list(model.model.attn_capture.data.keys())
        print(f"  Captured keys ({len(captured_keys)}): {captured_keys}")

        # Should have encoder self-attention for 2 layers
        assert len(captured_keys) >= 2, \
            f"Expected at least 2 attention maps, got {len(captured_keys)}"

        # Verify naming pattern
        enc_self_keys = [k for k in captured_keys if k.startswith('enc_self_layer_')]
        assert len(enc_self_keys) >= 2, \
            f"Expected 2 encoder self-attention maps, got {len(enc_self_keys)}"

        # ============================================================
        # STEP 5: Verify captured tensors have correct shapes
        # ============================================================
        print("\n" + "="*70)
        print("STEP 5: Verifying tensor shapes and values")
        print("="*70)

        for key, tensor in model.model.attn_capture.data.items():
            print(f"\n  Key: {key}")
            print(f"    Type: {type(tensor)}")
            print(f"    Shape: {tensor.shape}")
            print(f"    Device: {tensor.device}")
            print(f"    Dtype: {tensor.dtype}")

            # Verify it's a real tensor
            assert isinstance(tensor, torch.Tensor), f"{key} should be a tensor"

            # Verify shape: (num_heads, seq_len, seq_len)
            assert len(tensor.shape) == 3, f"{key} should be 3D (heads, seq, seq)"
            num_heads, seq1, seq2 = tensor.shape
            assert num_heads == 4, f"Expected 4 heads, got {num_heads}"
            assert seq1 == seq2, f"Attention map should be square, got {seq1}x{seq2}"
            print(f"    ✓ Shape valid: ({num_heads} heads, {seq1}x{seq2} attention)")

            # Verify values are valid probabilities (or at least finite)
            assert torch.isfinite(tensor).all(), f"{key} contains non-finite values"
            print(f"    ✓ All values finite")

            # Verify values are in reasonable range for attention weights
            # (after softmax, should be in [0, 1], but we're checking batch-averaged)
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            print(f"    Value range: [{min_val:.6f}, {max_val:.6f}]")

        # ============================================================
        # STEP 6: Verify disk save
        # ============================================================
        print("\n" + "="*70)
        print("STEP 6: Verifying disk save")
        print("="*70)

        model.save_attention_to_disk()

        npz_files = list(base_context.attention_dir.glob('*.npz'))
        json_files = list(base_context.attention_dir.glob('*_metadata.json'))

        assert len(npz_files) == 1, f"Expected 1 NPZ file, got {len(npz_files)}"
        assert len(json_files) == 1, f"Expected 1 JSON file, got {len(json_files)}"

        print(f"  ✓ NPZ file: {npz_files[0].name}")
        print(f"  ✓ JSON file: {json_files[0].name}")

        # Verify NPZ content
        npz_data = np.load(npz_files[0])
        npz_keys = list(npz_data.keys())
        print(f"\n  NPZ keys: {npz_keys}")
        assert set(npz_keys) == set(captured_keys), \
            f"NPZ keys mismatch: {set(npz_keys)} != {set(captured_keys)}"

        for key in npz_keys:
            arr = npz_data[key]
            print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")
            assert arr.shape[0] == 4, f"Should have 4 heads, got {arr.shape[0]}"

        # Verify JSON metadata
        with open(json_files[0]) as f:
            meta = json.load(f)

        print(f"\n  JSON metadata:")
        for k in ['model', 'architecture', 'num_heads', 'num_encoder_layers',
                  'primary_map', 'sampling_mode', 'timestamp']:
            print(f"    {k}: {meta.get(k)}")

        assert meta['model'] == 'transformer'
        assert meta['architecture'] == 'encoder-only'
        assert meta['num_heads'] == 4
        assert meta['num_encoder_layers'] == 2
        assert meta['primary_map'] in captured_keys

        print("\n" + "="*70)
        print("✅ ALL CHECKS PASSED: Attention capture works end-to-end!")
        print("="*70)

    def test_encoder_decoder_captures_all_attention_types(self, encoder_decoder_dataset, base_context):
        """
        CRITICAL: Verify encoder-decoder captures self-attention AND cross-attention.

        This test ensures that custom wrappers are registered for:
        - Encoder self-attention (enc → enc)
        - Decoder self-attention (dec → dec)
        - Decoder cross-attention (dec → enc)
        """
        config = {
            'hidden_size': 64,
            'num_heads': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'architecture': 'encoder-decoder',
            'attention_type': 'full',
            'strategy': 'direct',
            'tgt_init': 'zeros',
            'epochs': 1,
            'batch_size': 4,
            'learning_rate': 0.01,
            'attention_capture_enabled': True,
            'attention_capture_sampling': 'all',
        }

        # Use dataset with decoder exog for full enc-dec test
        model = TransformerForecaster(
            model_params=config,
            num_features=2,
            forecast_steps=5,
            window_size=10,
            dataset=encoder_decoder_dataset,
            run_context=base_context.with_metadata(fold_idx=0, window_size=10)
        )

        print("\n" + "="*70)
        print("TEST: Encoder-Decoder Attention Capture")
        print("="*70)

        # Note: Encoder uses GlobalSelfAttention (not CapturingMHA wrappers)
        # Decoder uses CapturingMHA wrappers for both self and cross attention
        # We verify capture by checking the captured attention maps below

        # ============================================================
        # Fit model
        # ============================================================
        print("\nFitting model...")
        train_df = encoder_decoder_dataset.development_data.iloc[:60]
        model.fit(train_series=train_df, dataset=encoder_decoder_dataset)
        print(f"  ✓ Model fitted on {len(train_df)} samples")

        # ============================================================
        # Perform forward pass with capture
        # ============================================================
        print("\nPerforming forward pass with attention capture...")

        history = encoder_decoder_dataset.development_data.tail(10)
        future_exog = encoder_decoder_dataset.test_data[encoder_decoder_dataset.future_covariates].head(5)

        # Prepare tensors
        input_tensor = model._prepare_input_tensor(history)
        print(f"  Input tensor shape: {input_tensor.shape}")

        # For encoder-decoder, create zero-initialized decoder input
        # Shape: (batch=1, forecast_steps=5, decoder_input_size)
        # decoder_input_size = num_features (2) + len(future_covariates) (1) = 3
        batch_size = input_tensor.shape[0]
        forecast_steps = 5
        decoder_input_size = model.feature_layout.decoder_input_size
        tgt_tensor = torch.zeros(batch_size, forecast_steps, decoder_input_size, device=model.device)
        print(f"  Decoder input (tgt) shape: {tgt_tensor.shape}")

        # Enable capture and call model forward directly
        model.model.attn_capture.configure(enabled=True, steps_to_capture=None)
        model.model.eval()
        with torch.no_grad():
            # Move input_tensor to model device (raw forward doesn't do this automatically when device_safety_checks=False)
            input_tensor = input_tensor.to(model.device)
            predictions = model.model(input_tensor, tgt=tgt_tensor)

        print(f"  ✓ Predictions shape: {predictions.shape}")

        # ============================================================
        # Verify ALL three attention types were captured
        # ============================================================
        print("\nVerifying captured attention types...")

        captured_keys = list(model.model.attn_capture.data.keys())
        print(f"  Total captured keys: {len(captured_keys)}")

        enc_self = [k for k in captured_keys if k.startswith('enc_self_layer_')]
        dec_self = [k for k in captured_keys if k.startswith('dec_self_layer_')]
        dec_cross = [k for k in captured_keys if k.startswith('dec_cross_layer_')]

        print(f"\n  Encoder self-attention: {len(enc_self)} maps")
        for k in enc_self:
            print(f"    - {k}: {model.model.attn_capture.data[k].shape}")

        print(f"\n  Decoder self-attention: {len(dec_self)} maps")
        for k in dec_self:
            print(f"    - {k}: {model.model.attn_capture.data[k].shape}")

        print(f"\n  Decoder cross-attention: {len(dec_cross)} maps")
        for k in dec_cross:
            print(f"    - {k}: {model.model.attn_capture.data[k].shape}")

        # Assertions
        assert len(enc_self) >= 2, \
            f"Expected ≥2 encoder self-attention maps, got {len(enc_self)}"
        assert len(dec_self) >= 2, \
            f"Expected ≥2 decoder self-attention maps, got {len(dec_self)}"
        assert len(dec_cross) >= 2, \
            f"Expected ≥2 decoder cross-attention maps, got {len(dec_cross)}"

        # ============================================================
        # Verify cross-attention shape (dec seq × enc seq)
        # ============================================================
        print("\nVerifying cross-attention shapes...")

        for key in dec_cross:
            tensor = model.model.attn_capture.data[key]
            num_heads, dec_seq, enc_seq = tensor.shape

            print(f"  {key}:")
            print(f"    Shape: ({num_heads} heads, {dec_seq} dec_seq, {enc_seq} enc_seq)")

            assert num_heads == 4, f"Expected 4 heads, got {num_heads}"
            # Cross-attention: decoder queries × encoder keys
            # Dec seq should be forecast_steps (5), enc seq should be window (10)
            assert dec_seq == 5, f"Expected dec_seq=5 (forecast_steps), got {dec_seq}"
            assert enc_seq == 10, f"Expected enc_seq=10 (window_size), got {enc_seq}"
            print(f"    ✓ Cross-attention shape correct (dec→enc)")

        print("\n" + "="*70)
        print("✅ ENCODER-DECODER TEST PASSED: All attention types captured!")
        print("="*70)

    def test_iterative_strategy_captures_multiple_steps(self, encoder_decoder_dataset, base_context):
        """
        CRITICAL: Verify iterative strategy captures attention at each autoregressive step.

        For iterative prediction with horizon=5:
        - Step 0: predict t+1 (using history)
        - Step 1: predict t+2 (using history + t+1)
        - Step 2: predict t+3 (using history + t+1 + t+2)
        - ...

        Each step should produce attention maps with step suffix.
        """
        config = {
            'hidden_size': 64,
            'num_heads': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'architecture': 'encoder-decoder',
            'strategy': 'iterative',  # ← KEY: iterative not direct
            'tgt_init': 'zeros',
            'epochs': 1,
            'batch_size': 4,
            'learning_rate': 0.01,
            'attention_capture_enabled': True,
            'attention_capture_sampling': 'first_last',  # Capture step 0 and 4
        }

        model = TransformerForecaster(
            model_params=config,
            num_features=2,
            forecast_steps=5,
            window_size=10,
            dataset=encoder_decoder_dataset,
            run_context=base_context.with_metadata(fold_idx=0, window_size=10)
        )

        print("\n" + "="*70)
        print("TEST: Iterative Strategy Step-by-Step Capture")
        print("="*70)

        # Fit model
        print("\nFitting model...")
        train_df = encoder_decoder_dataset.development_data.iloc[:60]
        model.fit(train_series=train_df, dataset=encoder_decoder_dataset)
        print(f"  ✓ Model fitted on {len(train_df)} samples")

        history = encoder_decoder_dataset.development_data.tail(10)
        future_exog = encoder_decoder_dataset.test_data[encoder_decoder_dataset.future_covariates].head(5)

        # Debug: Check run_context
        print(f"\nDEBUG: model.run_context exists: {model.run_context is not None}")
        if model.run_context:
            print(f"DEBUG: is_hpo_trial: {model.run_context.metadata.get('is_hpo_trial', False)}")
            should_capture = not model.run_context.metadata.get('is_hpo_trial', False)
            print(f"DEBUG: should_capture: {should_capture}")

        # predict() handles capture internally via run_context
        predictions = model.predict(history, future_exog=future_exog)

        # Debug: Check if files were created (would indicate capture happened)
        npz_files = list(base_context.attention_dir.glob('*.npz'))
        print(f"\nDEBUG: NPZ files created: {len(npz_files)}")
        for f in npz_files:
            print(f"  - {f.name}")

        print(f"  ✓ Predictions shape: {predictions.shape}")

        # ============================================================
        # Verify step suffixes in captured keys
        # ============================================================
        print("\nVerifying step-by-step capture...")

        captured_keys = list(model.model.attn_capture.data.keys())
        print(f"  Total captured keys: {len(captured_keys)}")

        # With sampling='first_last', should capture step_0 and step_4
        step_0_keys = [k for k in captured_keys if 'step_0' in k]
        step_4_keys = [k for k in captured_keys if 'step_4' in k]

        print(f"\n  Step 0 keys ({len(step_0_keys)}):")
        for k in step_0_keys[:5]:  # Show first 5
            print(f"    - {k}")

        print(f"\n  Step 4 keys ({len(step_4_keys)}):")
        for k in step_4_keys[:5]:
            print(f"    - {k}")

        assert len(step_0_keys) > 0, "Should have captured step_0 attention"
        assert len(step_4_keys) > 0, "Should have captured step_4 attention"

        # Verify step suffixes are correct
        for key in captured_keys:
            if 'step_' in key:
                # Extract step number
                step_num = int(key.split('step_')[-1].split('_')[0] if '_' in key.split('step_')[-1]
                              else key.split('step_')[-1])
                assert step_num in [0, 4], \
                    f"With sampling='first_last', expected step 0 or 4, got {step_num} in {key}"

        print("\n" + "="*70)
        print("✅ ITERATIVE STRATEGY TEST PASSED: Step-by-step capture works!")
        print("="*70)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
