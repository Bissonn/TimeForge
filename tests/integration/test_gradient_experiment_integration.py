"""
Integration test: Full training loop with gradient monitoring.
Verifies that global_step is tracked and CSVs are produced correctly.
"""
import pytest
import pandas as pd
from pathlib import Path
from models.lstm import LSTMForecaster
from utils.dataset import TimeSeriesDataset
from core.context import RunContext

@pytest.fixture
def sample_dataset():
    """Minimal time series dataset."""
    df = pd.DataFrame({
        'value': [i + 0.1 * i for i in range(200)],
        'date': pd.date_range('2020-01-01', periods=200, freq='D')
    })
    df.set_index('date', inplace=True)

    dummy_config = {
        "datasets": {
            "integration_test": {
                "freq": "D",
                "columns": ["value"],
                "time_features": []
            }
        }
    }

    return TimeSeriesDataset(
        dataset_name="integration_test",
        config=dummy_config,
        num_features=1,
        data=df,
        columns=['value'],
        past_covariates=[],
        future_covariates=[]
    )

@pytest.fixture
def run_context(tmp_path):
    """RunContext for testing."""
    ctx = RunContext.from_base_path(
        base_path=tmp_path / 'results',
        run_id='test_integration',
        experiment_name='test_exp'
    )
    ctx.create_directories()
    return ctx

def test_lstm_training_with_gradient_monitoring(sample_dataset, run_context):
    """Test full LSTM training with gradient monitoring enabled."""

    # Setup model with monitoring enabled
    model = LSTMForecaster(
        model_name='test_lstm',
        run_context=run_context.with_metadata(
            model_name='test_lstm',
            model_type='lstm',
            fold_idx=0,
            window_size=20
        ),
        model_params={
            'hidden_size': 32,
            'num_layers': 1,
            'dropout': 0.0,
            'strategy': 'direct',
            'learning_rate': 0.001,
            'epochs': 2,  # Run 2 epochs to check global_step continuity
            'early_stopping_patience': 999,
            'batch_size': 16,
            'gradient_monitor': {
                'enabled': True,
                'log_interval': 1
            }
        },
        num_features=1,
        forecast_steps=5,
        window_size=20,
        dataset=sample_dataset
    )

    # Split data manually for fit (simulate training set)
    # Use .series instead of .data (already fixed previously)
    train_df = sample_dataset.series.iloc[:150]

    # Use correct argument 'train_series' and remove 'val_data'
    # The fit method in NeuralTSForecaster splits train_series internally for validation
    model.fit(
        train_series=train_df,
        dataset=sample_dataset
    )

    # Verify CSV created
    gradient_csv = run_context.gradients_dir / 'test_lstm_fold_0_w20_gradients.csv'
    assert gradient_csv.exists(), f"Gradient CSV should exist at {gradient_csv}"

    # Verify content
    df = pd.read_csv(gradient_csv)

    print(f"\nCSV Content Head:\n{df.head()}")

    # 1. Check Columns
    expected_cols = ['epoch', 'step', 'global_step', 'batch_loss',
                     'total_grad_norm', 'encoder_grad_norm', 'head_grad_norm']
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    # 2. Check Logic
    assert len(df) > 0
    assert df['epoch'].max() == 2
    assert df['global_step'].is_monotonic_increasing

    # Check that gradients were actually recorded (not all zeros)
    assert df['encoder_grad_norm'].mean() > 0
    # total_grad_norm might be close to encoder_grad_norm + head_grad_norm
    assert df['total_grad_norm'].mean() > 0

    # 3. Check continuity of global_step across epochs
    epoch_1_max = df[df['epoch'] == 1]['global_step'].max()
    epoch_2_min = df[df['epoch'] == 2]['global_step'].min()

    assert epoch_2_min == epoch_1_max + 1, "global_step should be continuous across epochs"

    print("✅ Integration Test Passed: CSV structure and logic valid.")
