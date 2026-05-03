import pytest
import numpy as np
from utils.hyperopt.random_params import generate_random_params

pytestmark = pytest.mark.unit

def test_generate_random_params():
    param_space = {
        'learning_rate': {'min': 0.001, 'max': 0.1, 'log': True},
        'n_estimators': [100, 200, 300]
    }
    n_trials = 2
    combinations = generate_random_params(param_space, n_trials)

    assert len(combinations) <= n_trials
    assert all(isinstance(combo, dict) for combo in combinations)
    assert all('learning_rate' in combo and 'n_estimators' in combo for combo in combinations)
    assert all(0.001 <= combo['learning_rate'] <= 0.1 for combo in combinations)
    assert all(combo['n_estimators'] in [100, 200, 300] for combo in combinations)

def test_invalid_param_space():
    # Empty param_space should now be valid (returns single empty dict for fixed config)
    result = generate_random_params({}, n_trials=5)
    assert result == [{}]

    # Only structurally invalid params (like empty lists) should raise ValueError
    with pytest.raises(ValueError):
        generate_random_params({'param': []}, n_trials=5)

def test_invalid_n_trials():
    param_space = {'param': [1, 2, 3]}
    with pytest.raises(ValueError):
        generate_random_params(param_space, n_trials=0)
    with pytest.raises(TypeError):
        generate_random_params(param_space, n_trials=1.5)
