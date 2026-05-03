
import pytest
import numpy as np
from unittest.mock import patch
from utils.hyperopt.grid_params import generate_grid_params

pytestmark = pytest.mark.unit

def test_generate_grid_params_with_valid_list():
    param_space = {
        "param1": [1, 2],
        "param2": [0.1, 0.2]
    }
    result = generate_grid_params(param_space, n_trials=10)
    assert isinstance(result, list)
    assert all(isinstance(r, dict) for r in result)
    assert len(result) == 4  # 2 * 2 = 4 combinations

def test_generate_grid_params_with_dicts():
    param_space = {
        "param1": {"min": 1, "max": 3, "grid_steps": 3},
        "param2": {"min": 0.1, "max": 1.0, "grid_steps": 3}
    }
    result = generate_grid_params(param_space, n_trials=10)
    assert len(result) <= 9  # 3 * 3 = 9 combinations max

def test_generate_grid_params_with_log():
    param_space = {
        "param1": {"min": 1, "max": 100, "grid_steps": 3, "log": True}
    }
    result = generate_grid_params(param_space, n_trials=10)
    assert len(result) == 3
    assert all(isinstance(r["param1"], float) for r in result)

def test_generate_grid_params_exceeds_n_trials():
    param_space = {
        "param1": [1, 2, 3, 4, 5],
        "param2": [10, 20, 30, 40, 50]
    }
    result = generate_grid_params(param_space, n_trials=10)
    assert len(result) == 10  # capped at n_trials

def test_generate_grid_params_invalid_hidden_size_and_heads():
    param_space = {
        "hidden_size": [7],
        "num_heads": [3]
    }
    result = generate_grid_params(param_space, n_trials=10)
    assert result == []  # 7 % 3 != 0, should be excluded

def test_generate_grid_params_empty_space():
    # Empty param_space is valid and returns single trial
    result = generate_grid_params({}, n_trials=5)
    assert result == [{}]

def test_generate_grid_params_invalid_n_trials_type():
    with pytest.raises(ValueError):
        generate_grid_params({"param1": [1, 2]}, n_trials=0)

def test_generate_grid_params_invalid_grid_steps():
    param_space = {
        "param1": {"min": 1, "max": 10, "grid_steps": 1}
    }
    with pytest.raises(ValueError):
        generate_grid_params(param_space, n_trials=5)
