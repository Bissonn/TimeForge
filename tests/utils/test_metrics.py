import pytest
import numpy as np
from utils.metrics import calculate_metrics

pytestmark = pytest.mark.unit

# -- Test Cases --

# 1. Standard test case with simple, predictable values
def test_calculate_metrics_standard_case():
    """Tests the basic metric calculations on typical data."""
    actual = np.array([1, 2, 3, 4, 5])
    predicted = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    # Manual calculations for verification:
    # Errors: |-0.5|, |-0.5|, |-0.5|, |-0.5|, |-0.5| -> all are 0.5
    # MAE = 0.5
    # RMSE = sqrt(5 * 0.5^2 / 5) = 0.5
    # Naive Error = mean(|2-1|, |3-2|, |4-3|, |5-4|) = 1.0
    # MASE = MAE / Naive Error = 0.5 / 1.0 = 0.5

    metrics = calculate_metrics(actual, predicted)

    assert isinstance(metrics, dict)
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'smape' in metrics
    assert 'mase' in metrics

    assert metrics['mae'] == pytest.approx(0.5)
    assert metrics['rmse'] == pytest.approx(0.5)
    assert metrics['mase'] == pytest.approx(0.5)
    # SMAPE calculation is more complex, so we just check if it's within a sensible range
    assert 0 < metrics['smape'] < 100

# 2. Case where predictions are perfect
def test_calculate_metrics_perfect_prediction():
    """Tests the case where prediction errors are zero."""
    actual = np.array([10, 20, 30, 40, 50])
    predicted = np.array([10, 20, 30, 40, 50])

    metrics = calculate_metrics(actual, predicted)

    assert metrics['mae'] == 0.0
    assert metrics['rmse'] == 0.0
    assert metrics['smape'] == 0.0
    assert metrics['mase'] == 0.0

# 3. Case where all input values are zeros
def test_calculate_metrics_all_zeros():
    """Tests behavior when all input data points are zero."""
    actual = np.array([0, 0, 0, 0])
    predicted = np.array([0, 0, 0, 0])

    metrics = calculate_metrics(actual, predicted)

    # All metrics should be zero, without division-by-zero errors (thanks to epsilon)
    assert metrics['mae'] == 0.0
    assert metrics['rmse'] == 0.0
    assert metrics['smape'] == 0.0
    assert metrics['mase'] == 0.0

# 4. Case with negative values
def test_calculate_metrics_with_negative_values():
    """Tests that metrics correctly handle negative values."""
    actual = np.array([-1, -2, -3])
    predicted = np.array([-1.5, -2.5, -3.5])

    metrics = calculate_metrics(actual, predicted)

    assert metrics['mae'] == pytest.approx(0.5)
    assert metrics['rmse'] == pytest.approx(0.5)
    # Naive Error = mean(|-2 - -1|, |-3 - -2|) = 1.0
    # MASE = 0.5 / 1.0 = 0.5
    assert metrics['mase'] == pytest.approx(0.5)

# 5. Edge case: input arrays with a single element
def test_calculate_metrics_single_element_array_raises_error():
    """
    Tests that the function correctly raises a ValueError for single-element arrays,
    as MASE cannot be calculated.
    """
    actual = np.array([100])
    predicted = np.array([110])

    with pytest.raises(ValueError, match="MASE requires at least two actual values"):
        calculate_metrics(actual, predicted)

# 6. Test to ensure the function accepts standard Python lists as input
def test_calculate_metrics_with_lists():
    """Checks if the function works correctly when inputs are lists instead of numpy arrays."""
    actual = [1, 2, 3]
    predicted = [1.5, 2.5, 3.5]

    metrics = calculate_metrics(actual, predicted)

    # Expect the same results as for numpy arrays
    assert metrics['mae'] == pytest.approx(0.5)
    assert metrics['rmse'] == pytest.approx(0.5)
