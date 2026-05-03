import pytest
from utils.hyperopt.params_utils import _validate_param_space

pytestmark = pytest.mark.unit

def test_valid_continuous_range():
    param_space = {"x": {"min": 0.1, "max": 1.0}}
    result = _validate_param_space(param_space)
    assert result == [("x", {"min": 0.1, "max": 1.0})]

def test_valid_discrete_step_range():
    param_space = {"x": {"min": 1, "max": 5, "step": 1}}
    result = _validate_param_space(param_space)
    assert result == [("x", [1, 2, 3, 4, 5])]

def test_valid_log_range():
    param_space = {"x": {"min": 0.001, "max": 0.1, "log": True}}
    result = _validate_param_space(param_space)
    assert result == [("x", {"min": 0.001, "max": 0.1, "log": True})]

def test_valid_categorical_list():
    param_space = {"x": [0.1, 0.2, 0.3]}
    result = _validate_param_space(param_space)
    assert result == [("x", [0.1, 0.2, 0.3])]

def test_valid_categorical_strings():
    param_space = {"activation": ["relu", "tanh", "sigmoid"]}
    result = _validate_param_space(param_space)
    assert result == [("activation", ["relu", "tanh", "sigmoid"])]

@pytest.mark.parametrize("bad_range", [
    {"min": 10, "max": 5},  # min > max
    {"min": "a", "max": 10},  # non-numeric min
    {"min": 0.1, "max": 0.5, "step": -0.1},  # negative step
])
def test_invalid_range_raises(bad_range):
    with pytest.raises((ValueError, TypeError)):
        _validate_param_space({"x": bad_range})

def test_invalid_log_with_step():
    param_space = {"x": {"min": 0.001, "max": 0.1, "log": True, "step": 0.01}}
    with pytest.raises(ValueError, match=r".*step.*log.*"):
        _validate_param_space(param_space)

def test_invalid_log_with_discrete_range():
    param_space = {"x": {"min": 1, "max": 5, "step": 1, "log": True}}
    with pytest.raises(ValueError, match=r"cannot use 'step' with 'log'"):
        _validate_param_space(param_space)

def test_empty_list_param_raises():
    with pytest.raises(ValueError, match=r"empty list"):
        _validate_param_space({"x": []})

def test_too_large_value_list_raises():
    param_space = {"x": {"min": 0, "max": 1_000_000, "step": 1}}
    with pytest.raises(ValueError, match="Too many values"):
        _validate_param_space(param_space)

