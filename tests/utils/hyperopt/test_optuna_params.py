import pytest
from optuna.exceptions import OptunaError
from utils.hyperopt.optuna_params import generate_optuna_params

pytestmark = pytest.mark.unit

def test_valid_categorical_and_int_params():
    param_space = {
        "batch_size": [16, 32, 64],
        "hidden_size": {"min": 64, "max": 128, "step": 32},
        "n_trials": 10
    }
    results = generate_optuna_params(param_space, n_trials=5)
    assert len(results) > 0
    for combo in results:
        assert combo["batch_size"] in param_space["batch_size"]
        assert param_space["hidden_size"]["min"] <= combo["hidden_size"] <= param_space["hidden_size"]["max"]


def test_valid_float_log_params():
    param_space = {
        "learning_rate": {"min": 1e-4, "max": 1e-2, "log": True},
    }
    results = generate_optuna_params(param_space, n_trials=5)
    assert len(results) > 0
    for combo in results:
        assert 1e-4 <= combo["learning_rate"] <= 1e-2


def test_invalid_empty_param_space():
    with pytest.raises(ValueError, match="param_space cannot be empty"):
        generate_optuna_params({})


def test_invalid_n_trials():
    with pytest.raises(ValueError, match="n_trials must be a positive integer"):
        generate_optuna_params({"lr": [0.1]}, n_trials=0)


def test_invalid_direction():
    with pytest.raises(ValueError, match="direction must be 'minimize' or 'maximize'"):
        generate_optuna_params({"lr": [0.1]}, direction="invalid")


def test_invalid_param_format():
    # Not a list or dict
    with pytest.raises(ValueError, match="Invalid format for lr"):
        generate_optuna_params({"lr": "not_valid"})


def test_invalid_range_types():
    # min/max not numeric
    param_space = {"dropout": {"min": "low", "max": "high"}}
    with pytest.raises(ValueError, match="must be numeric"):
        generate_optuna_params(param_space)


def test_invalid_range_logic():
    # min > max
    param_space = {"dropout": {"min": 0.5, "max": 0.1}}
    with pytest.raises(ValueError, match="min.*must be <= max"):
        generate_optuna_params(param_space)


def test_invalid_step_with_float_log():
    param_space = {"dropout": {"min": 0.1, "max": 0.5, "log": True, "step": 0.1}}
    with pytest.raises(ValueError, match="Cannot use both 'log=True' and 'step'"):
        generate_optuna_params(param_space)


def test_invalid_step_value():
    param_space = {"units": {"min": 10, "max": 20, "step": 0}}
    with pytest.raises(ValueError, match="step.*must be positive"):
        generate_optuna_params(param_space)


def test_log_with_discrete_range():
    param_space = {"units": {"min": 10, "max": 20, "step": 2, "log": True}}
    with pytest.raises(ValueError, match="Cannot use both 'log=True' and 'step'"):
        generate_optuna_params(param_space)


def test_no_valid_keys():
    param_space = {"n_trials": 20}
    with pytest.raises(ValueError, match="No valid parameters found"):
        generate_optuna_params(param_space)


def test_transformer_constraint_filtering():
    param_space = {
        "hidden_size": {"min": 65, "max": 65, "step": 1},  # not divisible by 8
        "num_heads": {"min": 8, "max": 8, "step": 1}
    }
    with pytest.raises(ValueError, match="No valid parameter combinations after validation"):
        generate_optuna_params(param_space, n_trials=1)


def test_valid_transformer_divisibility():
    param_space = {
        "hidden_size": {"min": 64, "max": 64, "step": 1},  # divisible by 8
        "num_heads": {"min": 8, "max": 8, "step": 1}
    }
    results = generate_optuna_params(param_space, n_trials=1)
    assert len(results) == 1
    assert results[0]["hidden_size"] % results[0]["num_heads"] == 0
