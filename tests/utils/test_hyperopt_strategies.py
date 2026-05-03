import pytest
from utils.hyperopt.grid_params import generate_grid_params
from utils.hyperopt.random_params import generate_random_params


def test_grid_search_generates_all_combinations():
    """
    Verify that grid search generates the Cartesian product of parameters.
    """
    param_space = {
        "a": [1, 2],
        "b": ["x", "y"],
        "c": [True]
    }
    # Expected: 2 * 2 * 1 = 4 combinations
    candidates = generate_grid_params(param_space, n_trials=100)  # n_trials ignored for full grid unless capped?
    # Usually grid generates all. Let's check implementation assumption.

    assert len(candidates) == 4

    # Check content
    expected = [
        {"a": 1, "b": "x", "c": True},
        {"a": 1, "b": "y", "c": True},
        {"a": 2, "b": "x", "c": True},
        {"a": 2, "b": "y", "c": True},
    ]
    # Sort by keys to compare ignoring order
    candidates_sorted = sorted([str(sorted(d.items())) for d in candidates])
    expected_sorted = sorted([str(sorted(d.items())) for d in expected])

    assert candidates_sorted == expected_sorted


def test_random_search_respects_bounds_and_count():
    """
    Verify that random search generates correct number of trials and respects boundaries.
    """
    param_space = {
        "int_param": {"min": 10, "max": 20},
        "float_param": {"min": 0.0, "max": 1.0},
        "categorical": ["A", "B", "C"]
    }
    n_trials = 50

    candidates = generate_random_params(param_space, n_trials=n_trials)

    assert len(candidates) == n_trials

    for c in candidates:
        # Check Int
        assert 10 <= c["int_param"] <= 20
        assert isinstance(c["int_param"], int)

        # Check Float
        assert 0.0 <= c["float_param"] <= 1.0
        assert isinstance(c["float_param"], float)

        # Check Categorical
        assert c["categorical"] in ["A", "B", "C"]


def test_random_search_handles_log_scale():
    """
    Verify log scale logic (smoke test).
    """
    param_space = {
        "lr": {"min": 1e-5, "max": 1e-2, "log": True}
    }
    candidates = generate_random_params(param_space, n_trials=10)
    for c in candidates:
        assert 1e-5 <= c["lr"] <= 1e-2


def test_random_search_handles_step():
    """
    Verify step logic for integer/float ranges.
    """
    param_space = {
        "stepped_int": {"min": 0, "max": 10, "step": 2}  # 0, 2, 4, 6, 8, 10
    }
    candidates = generate_random_params(param_space, n_trials=20)
    for c in candidates:
        val = c["stepped_int"]
        assert val % 2 == 0
        assert 0 <= val <= 10