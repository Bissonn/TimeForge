"""
Test updated GradientAnalyzer with CSV support.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from analysis.gradient_analysis import GradientAnalyzer


@pytest.fixture
def temp_gradient_dir(tmp_path):
    """Temporary gradients directory with CSV files."""
    grad_dir = tmp_path / "gradients"
    grad_dir.mkdir()

    # Create sample CSV content
    # epoch,step,global_step,batch_loss,total_grad_norm,encoder_grad_norm,head_grad_norm
    csv_data = """epoch,step,global_step,batch_loss,total_grad_norm,encoder_grad_norm,head_grad_norm
1,0,1,0.5,1.0,0.8,0.2
1,1,2,0.4,0.9,0.7,0.2
2,0,3,0.3,0.8,0.6,0.2"""

    csv_file = grad_dir / "test_lstm_fold_0_w96_gradients.csv"
    csv_file.write_text(csv_data)

    return grad_dir


def test_load_gradient_logs_csv(temp_gradient_dir):
    """Test loading CSV gradient logs."""
    analyzer = GradientAnalyzer(gradients_dir=temp_gradient_dir)

    df = analyzer.load_gradient_logs('test_lstm', 96, 0)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert 'global_step' in df.columns
    assert 'encoder_grad_norm' in df.columns
    assert df['global_step'].iloc[-1] == 3


def test_discover_artifacts_csv(temp_gradient_dir):
    """Test discovering CSV artifacts."""
    analyzer = GradientAnalyzer(gradients_dir=temp_gradient_dir)

    artifacts = analyzer.discover_all_artifacts()

    assert len(artifacts) == 1
    assert artifacts[0].name.endswith('.csv')


def test_compute_gradient_decay(temp_gradient_dir):
    """Test decay computation from DataFrame."""
    analyzer = GradientAnalyzer(gradients_dir=temp_gradient_dir)
    df = analyzer.load_gradient_logs('test_lstm', 96, 0)

    decay = analyzer.compute_gradient_decay(df)

    assert isinstance(decay, float)
    # Encoder norm decreasing: 0.8 -> 0.7 -> 0.6, so negative decay
    assert decay < 0


def test_plot_generation(temp_gradient_dir, tmp_path):
    """Test that plotting generates a file without error."""
    analyzer = GradientAnalyzer(gradients_dir=temp_gradient_dir)
    output_plot = tmp_path / "plot.png"

    analyzer.plot_gradient_flow(
        model_names=['test_lstm'],
        window_sizes=[96],
        fold_idx=0,
        output_path=output_plot
    )

    assert output_plot.exists()