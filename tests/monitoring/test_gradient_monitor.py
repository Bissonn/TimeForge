"""
Unit tests for CSV-based GradientMonitor.

CRITICAL: Tests use ACTUAL framework models (LSTMForecaster, TransformerForecaster),
not toy models, to ensure classifier works in production.
"""
import pytest
import csv
import math
from pathlib import Path
import torch
import pandas as pd

from monitoring.gradient_monitor import GradientMonitor
from models.lstm import LSTMForecaster
from models.transformer import TransformerForecaster
from utils.dataset import TimeSeriesDataset
from core.context import RunContext


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory."""
    grad_dir = tmp_path / "gradients"
    grad_dir.mkdir()
    return grad_dir


@pytest.fixture
def sample_dataset():
    """Minimal dataset for model initialization."""
    df = pd.DataFrame({
        'value': range(200),
        'date': pd.date_range('2020-01-01', periods=200, freq='D')
    })
    df.set_index('date', inplace=True)

    # Minimal dummy config required by TimeSeriesDataset
    dummy_config = {
        "datasets": {
            "test_dataset": {
                "freq": "D",
                "columns": ["value"],
                "time_features": []
            }
        }
    }

    # CORRECTED INSTANTIATION
    return TimeSeriesDataset(
        dataset_name="test_dataset",  # Required arg
        config=dummy_config,  # Required arg\
        num_features=1,
        data=df,
        columns=['value'],  # Renamed from target_columns
        past_covariates=[],
        future_covariates=[]
    )


@pytest.fixture
def run_context(tmp_path):
    """Mock RunContext."""
    return RunContext.from_base_path(
        base_path=tmp_path / 'results',
        run_id='test',
        experiment_name='test'
    )


@pytest.fixture
def lstm_model(sample_dataset, run_context):
    """ACTUAL LSTMForecaster from framework."""
    return LSTMForecaster(
        model_name='test_lstm',
        run_context=run_context,
        model_params={
            'hidden_size': 32,
            'num_layers': 1,
            'dropout': 0.0,
            'strategy': 'direct'
        },
        num_features=1,
        forecast_steps=5,
        window_size=20,
        dataset=sample_dataset
    )


@pytest.fixture
def transformer_model(sample_dataset, run_context):
    """ACTUAL TransformerForecaster from framework."""
    return TransformerForecaster(
        model_name='test_transformer',
        run_context=run_context,
        model_params={
            'hidden_size': 32,
            'num_heads': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'architecture': 'encoder-decoder',
            'dropout': 0.0
        },
        num_features=1,
        forecast_steps=5,
        window_size=20,
        dataset=sample_dataset
    )


def test_initialization_creates_csv(temp_dir, lstm_model):
    """Test CSV creation with header."""
    monitor = GradientMonitor(
        model=lstm_model.model,  # Pass actual nn.Module
        save_dir=temp_dir,
        model_name='test_lstm',
        fold_idx=0,
        window_size=96,
        model_type='lstm',
        enabled=True,
        log_interval=1
    )

    csv_path = temp_dir / 'test_lstm_fold_0_w96_gradients.csv'
    assert csv_path.exists()

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            'epoch', 'step', 'global_step', 'batch_loss',
            'total_grad_norm', 'encoder_grad_norm', 'head_grad_norm'
        ]

    monitor.close()


def test_classify_parameters_lstm(lstm_model):
    """Test parameter classification for ACTUAL LSTMForecaster."""
    # Create dummy gradients
    for param in lstm_model.model.parameters():
        param.grad = torch.randn_like(param)

    encoder_params, head_params = GradientMonitor.classify_parameters(
        lstm_model.model, 'lstm'
    )

    assert len(encoder_params) > 0, "LSTM should have encoder parameters"
    assert len(head_params) > 0, "LSTM should have head parameters"

    # All params should be classified
    total_params = len(list(lstm_model.model.parameters()))
    assert len(encoder_params) + len(head_params) == total_params

    # Encoder should dominate (>80% typically)
    total_numel = sum(p.numel() for p in lstm_model.model.parameters())
    encoder_numel = sum(p.numel() for p in encoder_params)
    encoder_pct = 100 * encoder_numel / total_numel

    assert encoder_pct > 80, f"Encoder should be >80% of params, got {encoder_pct:.1f}%"


def test_classify_parameters_transformer(transformer_model):
    """Test parameter classification for ACTUAL TransformerForecaster."""
    # Create dummy gradients
    for param in transformer_model.model.parameters():
        param.grad = torch.randn_like(param)

    encoder_params, head_params = GradientMonitor.classify_parameters(
        transformer_model.model, 'transformer'
    )

    assert len(encoder_params) > 0, "Transformer should have encoder parameters"
    assert len(head_params) > 0, "Transformer should have head parameters"

    # All params should be classified
    total_params = len(list(transformer_model.model.parameters()))
    assert len(encoder_params) + len(head_params) == total_params

    # Encoder should dominate
    total_numel = sum(p.numel() for p in transformer_model.model.parameters())
    encoder_numel = sum(p.numel() for p in encoder_params)
    encoder_pct = 100 * encoder_numel / total_numel

    assert encoder_pct > 80, f"Encoder should be >80% of params, got {encoder_pct:.1f}%"


def test_compute_component_norms(lstm_model):
    """Test component norm computation with real model."""
    # Create dummy gradients
    for param in lstm_model.model.parameters():
        param.grad = torch.randn_like(param) * 0.1  # Small gradients

    encoder_params, head_params = GradientMonitor.classify_parameters(
        lstm_model.model, 'lstm'
    )

    enc_norm, head_norm = GradientMonitor.compute_component_norms(
        encoder_params, head_params
    )

    assert enc_norm > 0, "Encoder norm should be positive"
    assert head_norm > 0, "Head norm should be positive"
    assert isinstance(enc_norm, float)
    assert isinstance(head_norm, float)


def test_logging_with_global_step(temp_dir, lstm_model):
    """Test that global_step is logged correctly."""
    monitor = GradientMonitor(
        model=lstm_model.model,
        save_dir=temp_dir,
        model_name='test',
        fold_idx=0,
        window_size=96,
        model_type='lstm',
        enabled=True,
        log_interval=1
    )

    # Simulate 2 epochs with 3 batches each
    global_step = 0
    for epoch in range(2):
        for step in range(3):
            global_step += 1
            monitor.log_gradients(
                epoch=epoch + 1,
                step=step,
                global_step=global_step,
                batch_loss=0.5,
                total_grad_norm=1.0,
                encoder_grad_norm=0.8,
                head_grad_norm=0.2
            )

    monitor.close()

    # Verify
    csv_path = temp_dir / 'test_fold_0_w96_gradients.csv'
    df = pd.read_csv(csv_path)

    assert len(df) == 6, "Should have 6 rows (2 epochs × 3 batches)"
    assert df['global_step'].tolist() == [1, 2, 3, 4, 5, 6], "global_step should be monotonic"
    assert df['epoch'].tolist() == [1, 1, 1, 2, 2, 2], "Epochs should reset"
    assert df['step'].tolist() == [0, 1, 2, 0, 1, 2], "Steps should reset per epoch"


def test_auto_detect_model_type(temp_dir, lstm_model, transformer_model):
    """Test auto-detection works with framework models."""
    monitor_lstm = GradientMonitor(
        model=lstm_model.model,
        save_dir=temp_dir,
        model_name='auto_lstm',
        fold_idx=0,
        window_size=96,
        model_type='auto',
        enabled=False
    )
    assert monitor_lstm.model_type == 'lstm'

    monitor_trans = GradientMonitor(
        model=transformer_model.model,
        save_dir=temp_dir,
        model_name='auto_trans',
        fold_idx=0,
        window_size=96,
        model_type='auto',
        enabled=False
    )
    assert monitor_trans.model_type == 'transformer'