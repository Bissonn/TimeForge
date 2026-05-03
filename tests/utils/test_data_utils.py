"""
Unit tests for the sliding window data utility.

This module provides a comprehensive suite of tests for the `create_sliding_window`
function from `utils.data_utils`. It ensures the function correctly handles
univariate, multivariate, and various combinations of exogenous data, while also
validating all input parameters and edge cases.
"""

import pytest
import numpy as np
from utils.data_utils import create_sliding_window

pytestmark = pytest.mark.unit

# --- Test Data Fixtures ---

# A simple multivariate dataset with 10 time steps and 2 features.
SAMPLE_DATA_MULTIVARIATE = np.arange(20, dtype=float).reshape(10, 2)
# A simple univariate dataset with 10 time steps.
SAMPLE_DATA_UNIVARIATE = np.arange(10, dtype=float)
# A dataset with 2 target features, 1 encoder-only exogenous feature,
# and 1 decoder-ready exogenous feature. Shape (10, 4).
SAMPLE_DATA_FULL_EXOG = np.arange(40, dtype=float).reshape(10, 4)


# --- Success Cases ---

def test_standard_multivariate_case():
    """
    Verifies sliding window creation for a standard multivariate case where
    all features are treated as targets.
    """
    X, y_targets, y_decoder_exog = create_sliding_window(
        data=SAMPLE_DATA_MULTIVARIATE,
        window_size=3,
        forecast_steps=2,
        target_indices=[0, 1] # All columns are targets
    )

    # Expected shapes: (n_samples, window_size, features) and (n_samples, forecast_steps, targets)
    # n_samples = 10 - 3 - 2 + 1 = 6
    assert X.shape == (6, 3, 2)
    assert y_targets.shape == (6, 2, 2)
    assert y_decoder_exog is None # No decoder exog indices were provided

    # Verify the content of the first and last windows
    np.testing.assert_array_equal(X[0], np.array([[0, 1], [2, 3], [4, 5]]))
    np.testing.assert_array_equal(y_targets[0], np.array([[6, 7], [8, 9]]))

    np.testing.assert_array_equal(X[-1], np.array([[10, 11], [12, 13], [14, 15]]))
    np.testing.assert_array_equal(y_targets[-1], np.array([[16, 17], [18, 19]]))

def test_standard_univariate_case():
    """
    Verifies sliding window creation for a standard univariate case, ensuring
    the output arrays have the correct feature dimension (1).
    """
    X, y_targets, y_decoder_exog = create_sliding_window(
        data=SAMPLE_DATA_UNIVARIATE,
        window_size=4,
        forecast_steps=1,
        target_indices=[0]
    )

    # n_samples = 10 - 4 - 1 + 1 = 6
    assert X.shape == (6, 4, 1)
    assert y_targets.shape == (6, 1, 1)
    assert y_decoder_exog is None

    # Verify content of the first window
    np.testing.assert_array_equal(X[0], np.array([[0], [1], [2], [3]]))
    np.testing.assert_array_equal(y_targets[0], np.array([[4]]))

def test_with_target_and_decoder_exog_indices():
    """
    Verifies that `create_sliding_window` correctly separates target values,
    decoder-specific exogenous values, while keeping all features in the input window X.
    This is the primary use case for the Transformer encoder-decoder architecture.
    """
    # Data columns: [target_1, target_2, encoder_exog, decoder_exog]
    target_indices = [0, 1]
    decoder_exog_indices = [3]

    X, y_targets, y_decoder_exog = create_sliding_window(
        data=SAMPLE_DATA_FULL_EXOG,
        window_size=3,
        forecast_steps=2,
        target_indices=target_indices,
        decoder_exog_indices=decoder_exog_indices
    )

    # n_samples = 10 - 3 - 2 + 1 = 6
    assert X.shape == (6, 3, 4)          # X should contain all 4 features
    assert y_targets.shape == (6, 2, 2)  # y_targets should contain the 2 target features
    assert y_decoder_exog.shape == (6, 2, 1) # y_decoder_exog should contain the 1 decoder exog feature

    # Verify the content of the first window
    # X[0] should contain all columns from the first 3 time steps
    np.testing.assert_array_equal(X[0], np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11]
    ]))
    # y_targets[0] should contain only target columns from the forecast horizon
    np.testing.assert_array_equal(y_targets[0], np.array([
        [12, 13], # targets at step t+1
        [16, 17]  # targets at step t+2
    ]))
    # y_decoder_exog[0] should contain only decoder exog columns from the forecast horizon
    np.testing.assert_array_equal(y_decoder_exog[0], np.array([
        [15], # decoder_exog at step t+1
        [19]  # decoder_exog at step t+2
    ]))

def test_different_step_size():
    """
    Verifies that the function correctly handles a `step` size greater than 1,
    producing fewer but correctly spaced samples.
    """
    X, y_targets, _ = create_sliding_window(
        data=SAMPLE_DATA_MULTIVARIATE,
        window_size=3,
        forecast_steps=2,
        step=3,
        target_indices=[0, 1]
    )

    # Expected number of samples: floor((10 - 3 - 2) / 3) + 1 = 2
    assert X.shape == (2, 3, 2)
    assert y_targets.shape == (2, 2, 2)

    # First window starts at index 0
    np.testing.assert_array_equal(X[0], np.array([[0, 1], [2, 3], [4, 5]]))
    # Second window should start at index 3 (0 + step)
    np.testing.assert_array_equal(X[1], np.array([[6, 7], [8, 9], [10, 11]]))

# --- Failure Cases and Input Validation ---

def test_raises_error_on_insufficient_data_length():
    """
    Verifies that a ValueError is raised if the data is too short to create
    even a single window and forecast pair.
    """
    with pytest.raises(ValueError, match="Data length .* is insufficient"):
        create_sliding_window(SAMPLE_DATA_UNIVARIATE, window_size=8, forecast_steps=3)

def test_raises_error_on_empty_data():
    """Verifies that a ValueError is raised for an empty input array."""
    with pytest.raises(ValueError, match="data cannot be empty"):
        create_sliding_window(np.array([]), window_size=1, forecast_steps=1)

def test_raises_error_on_nan_values():
    """Verifies that a ValueError is raised if the input data contains NaN."""
    data_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError, match="data cannot contain NaN or infinite values"):
        create_sliding_window(data_with_nan, window_size=1, forecast_steps=1)

@pytest.mark.parametrize("args, error_msg", [
    ({'window_size': 0}, "window_size must be a positive integer"),
    ({'window_size': -1}, "window_size must be a positive integer"),
    ({'forecast_steps': 0}, "forecast_steps must be a positive integer"),
    ({'forecast_steps': -1}, "forecast_steps must be a positive integer"),
    ({'step': 0}, "step must be a positive integer"),
    ({'step': -1}, "step must be a positive integer"),
])
def test_raises_error_on_invalid_parameters(args, error_msg):
    """
    Verifies that ValueErrors are raised for various invalid integer parameters
    using a parameterized test.
    """
    valid_args = {'window_size': 3, 'forecast_steps': 2, 'step': 1}
    # Override with the invalid argument for this test case
    valid_args.update(args)

    with pytest.raises(ValueError, match=error_msg):
        create_sliding_window(SAMPLE_DATA_UNIVARIATE, **valid_args)

def test_raises_error_on_non_numpy_input():
    """Verifies that a TypeError is raised if the input is not a NumPy array."""
    with pytest.raises(TypeError, match="data must be a NumPy array"):
        create_sliding_window([1, 2, 3, 4, 5], window_size=2, forecast_steps=1)