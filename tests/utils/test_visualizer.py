"""Module for logging training-related events in the forecasting framework.

This module provides functions to log the start and completion of model training,
hyperparameter optimization results, and trial failures, ensuring consistent logging
across models like ARIMA, VAR, LSTM, and Transformer.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from utils.visualizer import Visualizer

pytestmark = pytest.mark.unit

@pytest.fixture
def sample_data():
    """
    Pytest fixture to provide sample data for visualization tests.
    Returns a dictionary containing test data, predictions, and metadata for plotting.
    """
    # Create a sample DataFrame with two features and datetime index
    test_df = pd.DataFrame({
        'feature_1': np.arange(10, dtype=float),
        'feature_2': np.arange(100, 110, dtype=float)
    }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=10)))

    # Create predictions array with shape (10, 2) for two features
    predictions_arr = np.array([
        np.arange(0.5, 10.5, dtype=float),
        np.arange(100.5, 110.5, dtype=float)
    ]).T  # Transpose to shape (10, 2)

    return {
        "dataset_name": "my_test_dataset",
        "model_name": "test_model",
        "test_data": test_df,
        "predictions": predictions_arr,
        "columns": ['feature_1', 'feature_2'],
        "forecast_steps": 10
    }


def test_plot_predictions_calls_dependencies_correctly(sample_data, mocker, base_context):
    """
    Tests that plot_predictions calls matplotlib functions with correct arguments.
    Verifies that figures are generated, plots are drawn, and files are saved.
    """
    # utils.visualizer uses pathlib, not os.makedirs.
    # We rely on base_context to provide a valid temp dir, so we don't mock mkdir.

    # Mock matplotlib functions to isolate plotting behavior
    mock_figure = mocker.patch('utils.visualizer.plt.figure')
    mock_plot = mocker.patch('utils.visualizer.plt.plot')
    mock_savefig = mocker.patch('utils.visualizer.plt.savefig')
    mock_close = mocker.patch('utils.visualizer.plt.close')
    # Mock additional matplotlib functions to prevent side effects
    mocker.patch('utils.visualizer.plt.title')
    mocker.patch('utils.visualizer.plt.xlabel')
    mocker.patch('utils.visualizer.plt.ylabel')
    mocker.patch('utils.visualizer.plt.legend')
    mocker.patch('utils.visualizer.plt.grid')

    # Pass save_dir from base_context (renamed from run_context)
    save_dir = base_context.plots_dir

    # Call the plot_predictions method with sample data
    Visualizer.plot_predictions(**sample_data, save_dir=save_dir)

    # Verify that two figures are created (one for each feature)
    assert mock_figure.call_count == 2
    # Verify that four plots are drawn (two per feature: actual and predicted)
    assert mock_plot.call_count == 4

    # Verify that savefig is called for each feature's plot
    # We check if the path ends with the expected filename
    assert mock_savefig.call_count == 2
    args0, _ = mock_savefig.call_args_list[0]
    args1, _ = mock_savefig.call_args_list[1]
    assert "feature_1_test_model_predictions.png" in str(args0[0])
    assert "feature_2_test_model_predictions.png" in str(args1[0])

    # Verify that figures are closed to free memory
    assert mock_close.call_count == 2


def test_plot_error_accumulation_calls_dependencies_correctly(sample_data, mocker, base_context):
    """
    Tests that plot_error_accumulation calls matplotlib functions correctly.
    Verifies figure generation, plotting, and file saving for error accumulation.
    """
    # Mock matplotlib functions to isolate plotting behavior
    mock_figure = mocker.patch('utils.visualizer.plt.figure')
    mock_plot = mocker.patch('utils.visualizer.plt.plot')
    mock_savefig = mocker.patch('utils.visualizer.plt.savefig')
    mock_close = mocker.patch('utils.visualizer.plt.close')
    # Mock additional matplotlib functions to prevent side effects
    mocker.patch('utils.visualizer.plt.title')
    mocker.patch('utils.visualizer.plt.xlabel')
    mocker.patch('utils.visualizer.plt.ylabel')
    mocker.patch('utils.visualizer.plt.legend')
    mocker.patch('utils.visualizer.plt.grid')

    # Pass save_dir from base_context
    save_dir = base_context.plots_dir

    # Call the plot_error_accumulation method with sample data
    Visualizer.plot_error_accumulation(**sample_data, save_dir=save_dir)

    # Verify that two figures are created (one for each feature)
    assert mock_figure.call_count == 2
    # Verify that two plots are drawn (one per feature for error accumulation)
    assert mock_plot.call_count == 2

    # Verify that savefig is called
    assert mock_savefig.call_count == 2
    args0, _ = mock_savefig.call_args_list[0]
    assert "feature_1_test_model_error_accumulation.png" in str(args0[0])

    # Verify that figures are closed to free memory
    assert mock_close.call_count == 2


@pytest.mark.parametrize("invalid_arg, error_msg", [
    ({"test_data": pd.DataFrame()}, "test_data cannot be empty."),
    ({"forecast_steps": 0}, "forecast_steps must be positive."),
    ({"forecast_steps": 11}, "test_data has 10 rows, but forecast_steps is 11."),
    ({"predictions": np.random.rand(5, 2)}, "predictions has 5 rows, but forecast_steps is 10."),
    ({"columns": ["one_col"]}, "Number of columns must match test_data and predictions feature dimensions."),
    # Edge case: Single-row data with mismatched forecast_steps
    ({"test_data": pd.DataFrame({'feature_1': [1.0], 'feature_2': [100.0]}, index=pd.to_datetime(['2023-01-01']))},
     "test_data has 1 rows, but forecast_steps is 10."),
    # Edge case: Empty predictions array
    ({"predictions": np.array([]).reshape(0, 2)}, "predictions has 0 rows, but forecast_steps is 10."),
])
def test_plot_functions_raise_value_error_for_invalid_inputs(sample_data, invalid_arg, error_msg):
    """
    Tests that plot_predictions and plot_error_accumulation raise ValueError for invalid inputs.
    """
    # Update sample data with invalid arguments
    valid_args = sample_data.copy()
    valid_args.update(invalid_arg)

    # Verify that plot_predictions raises the expected error
    with pytest.raises(ValueError, match=error_msg):
        Visualizer.plot_predictions(**valid_args)

    # Verify that plot_error_accumulation raises the expected error
    with pytest.raises(ValueError, match=error_msg):
        Visualizer.plot_error_accumulation(**valid_args)


def test_plot_functions_handle_univariate_predictions(sample_data, mocker, base_context):
    """
    Tests that plot_predictions and plot_error_accumulation handle univariate predictions correctly.
    """
    # Mock matplotlib functions to isolate plotting behavior
    mocker.patch('utils.visualizer.plt.figure')
    mock_plot = mocker.patch('utils.visualizer.plt.plot')
    mocker.patch('utils.visualizer.plt.savefig')
    mocker.patch('utils.visualizer.plt.close')
    mocker.patch('utils.visualizer.plt.title')
    mocker.patch('utils.visualizer.plt.xlabel')
    mocker.patch('utils.visualizer.plt.ylabel')
    mocker.patch('utils.visualizer.plt.legend')
    mocker.patch('utils.visualizer.plt.grid')

    # Modify sample data to include only one feature
    sample_data['test_data'] = sample_data['test_data'][['feature_1']]
    sample_data['predictions'] = sample_data['predictions'][:, 0]
    sample_data['columns'] = ['feature_1']

    # Pass save_dir
    save_dir = base_context.plots_dir

    # Test plot_predictions with univariate data
    Visualizer.plot_predictions(**sample_data, save_dir=save_dir)
    # Verify that two plots are drawn (actual and predicted for one feature)
    assert mock_plot.call_count == 2

    # Test plot_error_accumulation with univariate data
    Visualizer.plot_error_accumulation(**sample_data, save_dir=save_dir)
    # Verify that one additional plot is drawn for error accumulation
    assert mock_plot.call_count == 3


def test_plot_functions_handle_edge_cases(sample_data, mocker, base_context):
    """
    Tests edge cases for plot_predictions and plot_error_accumulation.
    """
    # Mock matplotlib functions
    mock_figure = mocker.patch('utils.visualizer.plt.figure')
    mock_plot = mocker.patch('utils.visualizer.plt.plot')
    mock_savefig = mocker.patch('utils.visualizer.plt.savefig')
    mock_close = mocker.patch('utils.visualizer.plt.close')
    mocker.patch('utils.visualizer.plt.title')
    mocker.patch('utils.visualizer.plt.xlabel')
    mocker.patch('utils.visualizer.plt.ylabel')
    mocker.patch('utils.visualizer.plt.legend')
    mocker.patch('utils.visualizer.plt.grid')

    # Pass save_dir
    save_dir = base_context.plots_dir

    # Edge Case: Single-row data with matching forecast_steps
    single_row_data = sample_data.copy()
    single_row_data['test_data'] = pd.DataFrame({
        'feature_1': [1.0],
        'feature_2': [100.0]
    }, index=pd.to_datetime(['2023-01-01']))
    single_row_data['predictions'] = np.array([[1.5, 100.5]])
    single_row_data['forecast_steps'] = 1

    Visualizer.plot_predictions(**single_row_data, save_dir=save_dir)

    # Verify calls
    assert mock_figure.call_count == 2
    assert mock_plot.call_count == 4
    assert mock_savefig.call_count == 2
    assert mock_close.call_count == 2

    # Reset mock counters
    mock_figure.reset_mock()
    mock_plot.reset_mock()
    mock_savefig.reset_mock()
    mock_close.reset_mock()

    Visualizer.plot_error_accumulation(**single_row_data, save_dir=save_dir)

    assert mock_figure.call_count == 2
    assert mock_plot.call_count == 2
    assert mock_savefig.call_count == 2
    assert mock_close.call_count == 2
