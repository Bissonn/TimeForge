import pytest
import pandas as pd
import numpy as np
from utils.dataset import TimeSeriesDataset

pytestmark = pytest.mark.unit


def test_dataset_manual_diff_robustness():
    """
    SAFETY TEST: Verifies that manual differencing is robust to outliers
    and does not crash during initialization (unlike auto-diff).
    """
    # Training data
    train_vals = list(range(20))  # [0, 1, ..., 19]

    # Scenario: Test data contains a massive outlier
    # In 'auto' mode, this crashed pmdarima. In 'manual' mode, this should be fine.
    test_vals = [100000] * 5
    data = pd.DataFrame({'y': train_vals + test_vals})

    config_manual = {
        'datasets': {
            'leak_test': {
                'differencing': {
                    'enabled': True,
                    'auto': 'none',  # Explicitly manual
                    'order': 1,  # Force d=1
                    'seasonal_order': 0
                }
            }
        }
    }

    # Initialize dataset
    try:
        ds = TimeSeriesDataset("leak_test", config_manual, num_features=1, data=data, columns=['y'])
    except ValueError as e:
        pytest.fail(f"Dataset initialization failed with manual differencing: {e}")

    # Check if differencing was applied
    state = ds._diff_state
    print(f"\nDifferencing state: {state}")

    assert state['d'] == 1
    assert state['D'] == 0

    # Check values: The outlier should simply result in a large difference, not a crash
    # original: ..., 19, 100000, 100000 ...
    # diff:     ...,  1, 99981,       0 ...

    # We want to verify the jump 19 -> 100000.
    # The jump is at the index where 100000 first appears.
    # Since we dropped the first NaN, the indices shifted.

    # Let's verify the MAX value in the differenced series, which corresponds to the outlier jump.
    max_diff = ds.series['y'].max()
    assert max_diff == 100000 - 19