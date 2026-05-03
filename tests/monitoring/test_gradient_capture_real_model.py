"""
CRITICAL TEST: Verifies gradient capture mechanism with REAL models (LSTM & Transformer).

This test ensures that:
1. GradientMonitor is properly initialized during training
2. Gradients are captured during real backward passes
3. CSV files are created with correct structure
4. Component-wise gradients (encoder vs head) are calculated correctly
5. global_step is continuous across epochs
6. All gradient values are finite

This is the definitive test that the gradient capture system works in production.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from models.lstm import LSTMForecaster
from models.transformer import TransformerForecaster
from utils.dataset import TimeSeriesDataset
from core.context import RunContext


@pytest.fixture
def gradient_dataset():
    """Create a real time series dataset for gradient testing."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n, freq='D'),
        'target': np.sin(np.linspace(0, 4*np.pi, n)) + np.random.randn(n)*0.1,
        'exog': np.random.randn(n),
    })

    ds = TimeSeriesDataset(
        'gradient_test',
        {'datasets': {'gradient_test': {}}},
        num_features=1,
        data=data,
        columns=['target'],
        past_covariates=['exog'],
        future_covariates=[]
    )
    ds.split_data(forecast_steps=5)
    return ds


class TestGradientCaptureRealModel:
    """Test suite for gradient capture with real LSTM and Transformer models."""

    def test_lstm_gradient_capture_end_to_end(self, gradient_dataset, base_context):
        """
        CRITICAL: Verify gradient capture works end-to-end for LSTM model.

        Verifies:
        - GradientMonitor initialized during training
        - CSV file created with correct structure
        - Component-wise gradients (encoder vs head) calculated
        - global_step continuous across epochs
        - All values finite
        """
        print("\n" + "="*70)
        print("TEST: LSTM Gradient Capture End-to-End")
        print("="*70)

        config = {
            'hidden_size': 32,
            'num_layers': 2,
            'dropout': 0.0,
            'strategy': 'direct',
            'learning_rate': 0.01,
            'epochs': 2,  # 2 epochs to verify global_step continuity
            'early_stopping_patience': 999,
            'batch_size': 8,
            'gradient_monitor': {
                'enabled': True,
                'log_interval': 1  # Log every batch
            }
        }

        model = LSTMForecaster(
            model_name='test_lstm',
            run_context=base_context.with_metadata(
                model_name='test_lstm',
                model_type='lstm',
                fold_idx=0,
                window_size=10
            ),
            model_params=config,
            num_features=1,
            forecast_steps=5,
            window_size=10,
            dataset=gradient_dataset
        )

        # ============================================================
        # STEP 1: Train model (gradient capture happens automatically)
        # ============================================================
        print("\nSTEP 1: Training LSTM model...")
        train_df = gradient_dataset.development_data.iloc[:60]
        model.fit(train_series=train_df, dataset=gradient_dataset)
        print("  ✓ Model trained")

        # ============================================================
        # STEP 2: Verify CSV file was created
        # ============================================================
        print("\nSTEP 2: Verifying CSV file creation...")

        csv_files = list(base_context.gradients_dir.glob('*.csv'))
        print(f"  Found {len(csv_files)} CSV files:")
        for f in csv_files:
            print(f"    - {f.name}")

        assert len(csv_files) >= 1, "Expected at least 1 gradient CSV file"

        csv_path = csv_files[0]
        print(f"  ✓ CSV file: {csv_path.name}")

        # ============================================================
        # STEP 3: Verify CSV structure and content
        # ============================================================
        print("\nSTEP 3: Verifying CSV structure and content...")

        df = pd.read_csv(csv_path)
        print(f"  CSV shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        # Expected columns
        expected_cols = ['epoch', 'step', 'global_step', 'batch_loss',
                        'total_grad_norm', 'encoder_grad_norm', 'head_grad_norm']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
        print("  ✓ All expected columns present")

        # ============================================================
        # STEP 4: Verify gradient values
        # ============================================================
        print("\nSTEP 4: Verifying gradient values...")

        print(f"  Total rows: {len(df)}")
        print(f"  Epochs: {df['epoch'].min()} to {df['epoch'].max()}")
        print(f"  Global steps: {df['global_step'].min()} to {df['global_step'].max()}")

        # Check that we have data from both epochs
        assert df['epoch'].max() == 2, f"Expected 2 epochs, got {df['epoch'].max()}"
        print("  ✓ Both epochs recorded")

        # Check global_step is monotonic increasing
        assert df['global_step'].is_monotonic_increasing, "global_step should be monotonic increasing"
        print("  ✓ global_step is monotonic increasing")

        # Check global_step continuity across epochs
        epoch_1_max = df[df['epoch'] == 1]['global_step'].max()
        epoch_2_min = df[df['epoch'] == 2]['global_step'].min()
        assert epoch_2_min == epoch_1_max + 1, \
            f"global_step should be continuous: epoch1_max={epoch_1_max}, epoch2_min={epoch_2_min}"
        print("  ✓ global_step continuous across epochs")

        # Check all values are finite
        assert df['batch_loss'].notna().all(), "batch_loss should have no NaN"
        assert np.isfinite(df['batch_loss']).all(), "batch_loss should be finite"
        print("  ✓ batch_loss values are finite")

        assert df['total_grad_norm'].notna().all(), "total_grad_norm should have no NaN"
        assert np.isfinite(df['total_grad_norm']).all(), "total_grad_norm should be finite"
        assert (df['total_grad_norm'] >= 0).all(), "total_grad_norm should be non-negative"
        print("  ✓ total_grad_norm values are finite and non-negative")

        assert df['encoder_grad_norm'].notna().all(), "encoder_grad_norm should have no NaN"
        assert np.isfinite(df['encoder_grad_norm']).all(), "encoder_grad_norm should be finite"
        assert (df['encoder_grad_norm'] >= 0).all(), "encoder_grad_norm should be non-negative"
        print("  ✓ encoder_grad_norm values are finite and non-negative")

        assert df['head_grad_norm'].notna().all(), "head_grad_norm should have no NaN"
        assert np.isfinite(df['head_grad_norm']).all(), "head_grad_norm should be finite"
        assert (df['head_grad_norm'] >= 0).all(), "head_grad_norm should be non-negative"
        print("  ✓ head_grad_norm values are finite and non-negative")

        # Check that gradients were actually captured (not all zeros)
        assert df['encoder_grad_norm'].mean() > 0, "encoder gradients should be non-zero"
        assert df['head_grad_norm'].mean() > 0, "head gradients should be non-zero"
        print("  ✓ Gradients are non-zero (actual training happened)")

        # ============================================================
        # STEP 5: Verify component-wise gradient relationship
        # ============================================================
        print("\nSTEP 5: Verifying component-wise gradient relationship...")

        # total_grad_norm should be >= max(encoder, head) but not necessarily equal to sum
        # (it's L2 norm of all parameters, not sum of component norms)
        max_component = df[['encoder_grad_norm', 'head_grad_norm']].max(axis=1)
        assert (df['total_grad_norm'] >= max_component * 0.99).all(), \
            "total_grad_norm should be >= max component norm"
        print("  ✓ total_grad_norm >= max(encoder, head)")

        print("\nStatistics:")
        print(f"  Mean batch_loss: {df['batch_loss'].mean():.6f}")
        print(f"  Mean total_grad_norm: {df['total_grad_norm'].mean():.6f}")
        print(f"  Mean encoder_grad_norm: {df['encoder_grad_norm'].mean():.6f}")
        print(f"  Mean head_grad_norm: {df['head_grad_norm'].mean():.6f}")

        print("\n" + "="*70)
        print("✅ LSTM GRADIENT CAPTURE TEST PASSED!")
        print("="*70)

    def test_transformer_gradient_capture_end_to_end(self, gradient_dataset, base_context):
        """
        CRITICAL: Verify gradient capture works end-to-end for Transformer model.

        Verifies:
        - GradientMonitor initialized during training
        - CSV file created with correct structure
        - Component-wise gradients (encoder vs head) calculated
        - global_step continuous across epochs
        - All values finite
        """
        print("\n" + "="*70)
        print("TEST: Transformer Gradient Capture End-to-End")
        print("="*70)

        config = {
            'hidden_size': 32,
            'num_heads': 2,
            'num_encoder_layers': 2,
            'architecture': 'encoder-only',
            'attention_type': 'full',
            'learning_rate': 0.01,
            'epochs': 2,  # 2 epochs to verify global_step continuity
            'early_stopping_patience': 999,
            'batch_size': 8,
            'gradient_monitor': {
                'enabled': True,
                'log_interval': 1  # Log every batch
            }
        }

        model = TransformerForecaster(
            model_params=config,
            num_features=1,
            forecast_steps=5,
            window_size=10,
            dataset=gradient_dataset,
            run_context=base_context.with_metadata(
                model_name='test_transformer',
                model_type='transformer',
                fold_idx=0,
                window_size=10
            )
        )

        # ============================================================
        # STEP 1: Train model (gradient capture happens automatically)
        # ============================================================
        print("\nSTEP 1: Training Transformer model...")
        train_df = gradient_dataset.development_data.iloc[:60]
        model.fit(train_series=train_df, dataset=gradient_dataset)
        print("  ✓ Model trained")

        # ============================================================
        # STEP 2: Verify CSV file was created
        # ============================================================
        print("\nSTEP 2: Verifying CSV file creation...")

        csv_files = list(base_context.gradients_dir.glob('*.csv'))
        print(f"  Found {len(csv_files)} CSV files:")
        for f in csv_files:
            print(f"    - {f.name}")

        assert len(csv_files) >= 1, "Expected at least 1 gradient CSV file"

        # Find transformer CSV (not LSTM from previous test if run together)
        transformer_csv = None
        for f in csv_files:
            if 'transformer' in f.name.lower():
                transformer_csv = f
                break
        if transformer_csv is None:
            transformer_csv = csv_files[-1]  # Take last one

        print(f"  ✓ CSV file: {transformer_csv.name}")

        # ============================================================
        # STEP 3: Verify CSV structure and content
        # ============================================================
        print("\nSTEP 3: Verifying CSV structure and content...")

        df = pd.read_csv(transformer_csv)
        print(f"  CSV shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        # Expected columns
        expected_cols = ['epoch', 'step', 'global_step', 'batch_loss',
                        'total_grad_norm', 'encoder_grad_norm', 'head_grad_norm']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
        print("  ✓ All expected columns present")

        # ============================================================
        # STEP 4: Verify gradient values
        # ============================================================
        print("\nSTEP 4: Verifying gradient values...")

        print(f"  Total rows: {len(df)}")
        print(f"  Epochs: {df['epoch'].min()} to {df['epoch'].max()}")
        print(f"  Global steps: {df['global_step'].min()} to {df['global_step'].max()}")

        # Check that we have data from both epochs
        assert df['epoch'].max() == 2, f"Expected 2 epochs, got {df['epoch'].max()}"
        print("  ✓ Both epochs recorded")

        # Check global_step is monotonic increasing
        assert df['global_step'].is_monotonic_increasing, "global_step should be monotonic increasing"
        print("  ✓ global_step is monotonic increasing")

        # Check global_step continuity across epochs
        epoch_1_max = df[df['epoch'] == 1]['global_step'].max()
        epoch_2_min = df[df['epoch'] == 2]['global_step'].min()
        assert epoch_2_min == epoch_1_max + 1, \
            f"global_step should be continuous: epoch1_max={epoch_1_max}, epoch2_min={epoch_2_min}"
        print("  ✓ global_step continuous across epochs")

        # Check all values are finite
        assert df['batch_loss'].notna().all(), "batch_loss should have no NaN"
        assert np.isfinite(df['batch_loss']).all(), "batch_loss should be finite"
        print("  ✓ batch_loss values are finite")

        assert df['total_grad_norm'].notna().all(), "total_grad_norm should have no NaN"
        assert np.isfinite(df['total_grad_norm']).all(), "total_grad_norm should be finite"
        assert (df['total_grad_norm'] >= 0).all(), "total_grad_norm should be non-negative"
        print("  ✓ total_grad_norm values are finite and non-negative")

        assert df['encoder_grad_norm'].notna().all(), "encoder_grad_norm should have no NaN"
        assert np.isfinite(df['encoder_grad_norm']).all(), "encoder_grad_norm should be finite"
        assert (df['encoder_grad_norm'] >= 0).all(), "encoder_grad_norm should be non-negative"
        print("  ✓ encoder_grad_norm values are finite and non-negative")

        assert df['head_grad_norm'].notna().all(), "head_grad_norm should have no NaN"
        assert np.isfinite(df['head_grad_norm']).all(), "head_grad_norm should be finite"
        assert (df['head_grad_norm'] >= 0).all(), "head_grad_norm should be non-negative"
        print("  ✓ head_grad_norm values are finite and non-negative")

        # Check that gradients were actually captured (not all zeros)
        assert df['encoder_grad_norm'].mean() > 0, "encoder gradients should be non-zero"
        assert df['head_grad_norm'].mean() > 0, "head gradients should be non-zero"
        print("  ✓ Gradients are non-zero (actual training happened)")

        # ============================================================
        # STEP 5: Verify component-wise gradient relationship
        # ============================================================
        print("\nSTEP 5: Verifying component-wise gradient relationship...")

        # total_grad_norm should be >= max(encoder, head)
        max_component = df[['encoder_grad_norm', 'head_grad_norm']].max(axis=1)
        assert (df['total_grad_norm'] >= max_component * 0.99).all(), \
            "total_grad_norm should be >= max component norm"
        print("  ✓ total_grad_norm >= max(encoder, head)")

        print("\nStatistics:")
        print(f"  Mean batch_loss: {df['batch_loss'].mean():.6f}")
        print(f"  Mean total_grad_norm: {df['total_grad_norm'].mean():.6f}")
        print(f"  Mean encoder_grad_norm: {df['encoder_grad_norm'].mean():.6f}")
        print(f"  Mean head_grad_norm: {df['head_grad_norm'].mean():.6f}")

        print("\n" + "="*70)
        print("✅ TRANSFORMER GRADIENT CAPTURE TEST PASSED!")
        print("="*70)

    def test_gradient_monitor_disabled_no_csv_created(self, gradient_dataset, base_context):
        """
        Verify that when gradient_monitor is disabled, no CSV is created.
        """
        print("\n" + "="*70)
        print("TEST: Gradient Monitor Disabled - No CSV Created")
        print("="*70)

        # Clear any existing CSV files first
        for f in base_context.gradients_dir.glob('*.csv'):
            f.unlink()

        config = {
            'hidden_size': 16,
            'num_layers': 1,
            'epochs': 1,
            'batch_size': 8,
            'gradient_monitor': {
                'enabled': False  # ← Disabled
            }
        }

        model = LSTMForecaster(
            model_name='test_lstm_disabled',
            run_context=base_context.with_metadata(
                model_name='test_lstm_disabled',
                model_type='lstm',
                fold_idx=0,
                window_size=10
            ),
            model_params=config,
            num_features=1,
            forecast_steps=5,
            window_size=10,
            dataset=gradient_dataset
        )

        print("\nTraining with gradient_monitor.enabled=False...")
        train_df = gradient_dataset.development_data.iloc[:40]
        model.fit(train_series=train_df, dataset=gradient_dataset)
        print("  ✓ Model trained")

        # Verify no CSV files were created
        csv_files = list(base_context.gradients_dir.glob('*.csv'))
        assert len(csv_files) == 0, f"Expected 0 CSV files, found {len(csv_files)}"
        print("  ✓ No CSV files created (as expected)")

        print("\n" + "="*70)
        print("✅ GRADIENT MONITOR DISABLED TEST PASSED!")
        print("="*70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
