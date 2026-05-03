import pytest
from schema import SchemaError
import os
import yaml
import re

# Import the public API `load_config` and other necessary components
from utils.config_utils import load_config, get_model_config, ConfigValidationError, MODEL_TYPES, ALLOWED_TIME_FEATURES

pytestmark = pytest.mark.unit


@pytest.fixture
def valid_config_dict(dummy_data_path):
    """
    Pytest fixture to provide a base valid configuration dictionary using a
    temporary data file.
    """
    return {
        'paths': {
            'model_save_path_template': "results/{model_name}.pkl"
        },
        'datasets': {
            'my_data': {
                'path': dummy_data_path,
                'columns': ['value'],
                'past_covariates': ['enc_col'],
                'future_covariates': ['dec_col'],
                'time_features': ['hour', 'day_of_week'],
                'freq': 'D',
                'differencing': {
                    'enabled': True,
                    'order': 1,
                    'seasonal_order': 0
                }
            }
        },
        'models': {
            'arima_base': {  # Config name
                'type': 'arima',  # Required type
                'p': 1, 'd': 1, 'q': 1,
            },
            'transformer_v1': {  # Config name
                'type': 'transformer',  # Required type
                "hidden_size": 16, "num_heads": 2,
                "num_encoder_layers": 2, "dim_ff_multiplier": 4.0,
                "preprocessing": {
                    "preprocessing_groups": [{
                        "name": "default", "apply_to": "__targets__",
                        "pipeline": {"scaling": {"enabled": True}}
                    }]
                }
            }
        },
        'experiments': [{
            'name': 'test_experiment',
            'description': 'A test experiment with advanced configuration.',
            'dataset': 'my_data',
            'models': [
                {
                    "name": "transformer_v1",  # Reference by config name
                    "use_exogenous": True,
                    "past_covariates": ["enc_col"],
                    "future_covariates": ["dec_col", "hour"],
                    "use_raw_data_source": False
                },
                {
                    "name": "arima_base",  # Reference by config name
                    "use_raw_data_source": True
                }
            ],
            'validation_setup': {
                'forecast_steps': 10,
                'n_folds': 3,
                'window_size': 50
            }
        }],
    }


@pytest.fixture
def dummy_data_path(tmp_path):
    """Creates a dummy data file for config validation and returns its path."""
    data_file = tmp_path / "data.csv"
    data_file.write_text("date,value,enc_col,dec_col\n2023-01-01,10,1,2")
    return str(data_file)


def write_config_to_file(config_dict, path):
    """Writes a config dictionary to a temporary YAML file for testing."""
    config_file = path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f, sort_keys=False)
    return str(config_file)


def test_config_validation_succeeds_with_valid_config(valid_config_dict, tmp_path):
    """Verifies that a correctly structured config, including new keys, passes validation."""
    config_path = write_config_to_file(valid_config_dict, tmp_path)
    try:
        validated = load_config(config_path)
        assert isinstance(validated, dict)
        assert validated['models']['arima_base']['type'] == 'arima'
        assert validated['models']['transformer_v1']['type'] == 'transformer'
    except Exception as e:
        pytest.fail(f"load_config() raised an unexpected exception: {e}")


def test_config_fails_missing_type_field(valid_config_dict, tmp_path):
    """Tests that validation fails if a model config is missing the 'type' field."""
    invalid_config = valid_config_dict.copy()
    # Remove 'type' from arima_base
    del invalid_config['models']['arima_base']['type']

    config_path = write_config_to_file(invalid_config, tmp_path)

    with pytest.raises(ConfigValidationError, match=r"missing required field 'type'"):
        load_config(config_path)


def test_config_fails_invalid_type_value(valid_config_dict, tmp_path):
    """Tests that validation fails if 'type' is not a supported model type."""
    invalid_config = valid_config_dict.copy()
    # Set invalid type
    invalid_config['models']['arima_base']['type'] = 'super_model_3000'

    config_path = write_config_to_file(invalid_config, tmp_path)

    with pytest.raises(ConfigValidationError, match=r"Invalid model type 'super_model_3000'"):
        load_config(config_path)


# ... (rest of the tests: test_config_fails_if_differencing_in_preprocessor, etc. - make sure they use valid_config_dict structure)

def test_config_fails_if_differencing_in_preprocessor(valid_config_dict, tmp_path):
    """Tests that schema validation FAILS if 'differencing' is in 'pipeline'."""
    invalid_config = valid_config_dict.copy()
    # Manually inject 'differencing' where it should not be
    invalid_config['models']['transformer_v1']['preprocessing']['preprocessing_groups'][0]['pipeline']['differencing'] = {
        'enabled': True
    }
    config_path = write_config_to_file(invalid_config, tmp_path)

    # We expect an error because 'differencing' is no longer a valid key in 'pipeline'
    # The path in error message depends on how schema traversal reports it
    with pytest.raises(ConfigValidationError,
                       match=r"Wrong key 'differencing'"): # ZAKTUALIZOWANO MATCH
        load_config(config_path)


def test_config_validation_fails_with_missing_experiment_model(valid_config_dict, tmp_path):
    """Validation must fail if an experiment references an undefined model config."""
    invalid_config = valid_config_dict.copy()
    # "lstm_config" is not defined in the 'models' block of 'valid_config_dict'
    invalid_config["experiments"][0]["models"] = [{"name": "lstm_config"}]
    config_path = write_config_to_file(invalid_config, tmp_path)
    with pytest.raises(ConfigValidationError, match="Model configuration 'lstm_config' .* is not defined"):
        load_config(config_path)


def test_get_model_config_retrieves_correctly(valid_config_dict, mocker):
    """Test retrieval of model config using a mocked load_config."""
    mocker.patch('utils.config_utils.load_config', return_value=valid_config_dict)
    # We request 'arima_base' which is the key in 'models'
    model_config = get_model_config('arima_base', config_path="dummy_path.yaml")
    expected = {
        'type': 'arima',
        'p': 1, 'd': 1, 'q': 1,
        'optimization': {'method': 'grid', 'params': {}}
    }
    assert model_config == expected


# ═══════════════════════════════════════════════════════════════════════════
# Tests for NEW ARIMA/SARIMA Configuration Parameters
# ═══════════════════════════════════════════════════════════════════════════

def test_arima_accepts_new_advanced_parameters(valid_config_dict, tmp_path):
    """Test that ARIMA schema accepts all new advanced configuration parameters."""
    config = valid_config_dict.copy()
    config['models']['arima_advanced'] = {
        'type': 'arima',
        'p': 1, 'd': 1, 'q': 1,
        # New advanced parameters
        'enforce_stationarity': False,
        'enforce_invertibility': True,
        'method': 'css-mle',
        'maxiter': 30,
        'remove_data_after_fit': True
    }

    config_path = write_config_to_file(config, tmp_path)
    loaded_config = load_config(config_path)

    # Verify all parameters were accepted
    arima_config = loaded_config['models']['arima_advanced']
    assert arima_config['enforce_stationarity'] is False
    assert arima_config['enforce_invertibility'] is True
    assert arima_config['method'] == 'css-mle'
    assert arima_config['maxiter'] == 30
    assert arima_config['remove_data_after_fit'] is True


def test_sarima_accepts_new_advanced_parameters(valid_config_dict, tmp_path):
    """Test that SARIMA schema accepts all new advanced configuration parameters."""
    config = valid_config_dict.copy()
    config['models']['sarima_advanced'] = {
        'type': 'sarima',
        'p': 1, 'd': 1, 'q': 1,
        'P': 1, 'D': 1, 'Q': 1,
        'seasonal_period': 12,
        # New advanced parameters
        'enforce_stationarity': False,
        'enforce_invertibility': False,
        'method': 'bfgs',
        'maxiter': 100,
        'remove_data_after_fit': True
    }

    config_path = write_config_to_file(config, tmp_path)
    loaded_config = load_config(config_path)

    # Verify all parameters were accepted
    sarima_config = loaded_config['models']['sarima_advanced']
    assert sarima_config['enforce_stationarity'] is False
    assert sarima_config['enforce_invertibility'] is False
    assert sarima_config['method'] == 'bfgs'
    assert sarima_config['maxiter'] == 100
    assert sarima_config['remove_data_after_fit'] is True


def test_arima_rejects_invalid_method(valid_config_dict, tmp_path):
    """Test that ARIMA schema rejects invalid solver methods."""
    config = valid_config_dict.copy()
    config['models']['arima_base']['method'] = 'invalid_solver'

    config_path = write_config_to_file(config, tmp_path)

    with pytest.raises(ConfigValidationError, match=r"invalid_solver"):
        load_config(config_path)


def test_arima_rejects_negative_maxiter(valid_config_dict, tmp_path):
    """Test that ARIMA schema rejects negative maxiter values."""
    config = valid_config_dict.copy()
    config['models']['arima_base']['maxiter'] = -10

    config_path = write_config_to_file(config, tmp_path)

    with pytest.raises(ConfigValidationError):
        load_config(config_path)


def test_arima_rejects_zero_maxiter(valid_config_dict, tmp_path):
    """Test that ARIMA schema rejects zero maxiter values."""
    config = valid_config_dict.copy()
    config['models']['arima_base']['maxiter'] = 0

    config_path = write_config_to_file(config, tmp_path)

    with pytest.raises(ConfigValidationError):
        load_config(config_path)


def test_arima_accepts_all_valid_methods(valid_config_dict, tmp_path):
    """Test that ARIMA accepts all statsmodels-supported solver methods."""
    valid_methods = ["lbfgs", "css-mle", "bfgs", "newton", "nm", "powell"]

    for method in valid_methods:
        config = valid_config_dict.copy()
        config['models'][f'arima_{method}'] = {
            'type': 'arima',
            'p': 1, 'd': 1, 'q': 1,
            'method': method
        }

        config_path = write_config_to_file(config, tmp_path)
        loaded_config = load_config(config_path)

        # Verify method was accepted
        assert loaded_config['models'][f'arima_{method}']['method'] == method


def test_sarima_accepts_all_valid_methods(valid_config_dict, tmp_path):
    """Test that SARIMA accepts all statsmodels-supported solver methods."""
    valid_methods = ["lbfgs", "css-mle", "bfgs", "newton", "nm", "powell"]

    for method in valid_methods:
        config = valid_config_dict.copy()
        config['models'][f'sarima_{method}'] = {
            'type': 'sarima',
            'p': 1, 'd': 1, 'q': 1,
            'P': 1, 'D': 1, 'Q': 1,
            'seasonal_period': 12,
            'method': method
        }

        config_path = write_config_to_file(config, tmp_path)
        loaded_config = load_config(config_path)

        # Verify method was accepted
        assert loaded_config['models'][f'sarima_{method}']['method'] == method


def test_arima_boolean_parameters_enforce_type(valid_config_dict, tmp_path):
    """Test that boolean parameters enforce correct type (no strings like 'true')."""
    config = valid_config_dict.copy()
    config['models']['arima_base']['enforce_stationarity'] = "true"  # String, not bool

    config_path = write_config_to_file(config, tmp_path)

    with pytest.raises(ConfigValidationError):
        load_config(config_path)


# ═══════════════════════════════════════════════════════════════════════════
# Tests for NEW VAR Configuration Parameters
# ═══════════════════════════════════════════════════════════════════════════

def test_var_accepts_new_advanced_parameters(valid_config_dict, tmp_path):
    """Test that VAR schema accepts all new advanced configuration parameters."""
    config = valid_config_dict.copy()
    config['models']['var_advanced'] = {
        'type': 'var',
        'max_lags': 2,
        # New advanced parameters
        'enforce_stationarity': False,
        'enforce_invertibility': True,
        'method': 'bfgs',
        'maxiter': 150,
        'remove_data_after_fit': True
    }

    config_path = write_config_to_file(config, tmp_path)
    loaded_config = load_config(config_path)

    # Verify all parameters were accepted
    var_config = loaded_config['models']['var_advanced']
    assert var_config['enforce_stationarity'] is False
    assert var_config['enforce_invertibility'] is True
    assert var_config['method'] == 'bfgs'
    assert var_config['maxiter'] == 150
    assert var_config['remove_data_after_fit'] is True


def test_var_rejects_invalid_method(valid_config_dict, tmp_path):
    """Test that VAR schema rejects invalid solver methods."""
    config = valid_config_dict.copy()
    config['models']['var_base'] = {
        'type': 'var',
        'max_lags': 1,
        'method': 'invalid_solver'
    }

    config_path = write_config_to_file(config, tmp_path)

    with pytest.raises(ConfigValidationError, match=r"invalid_solver"):
        load_config(config_path)


def test_var_rejects_negative_maxiter(valid_config_dict, tmp_path):
    """Test that VAR schema rejects negative maxiter values."""
    config = valid_config_dict.copy()
    config['models']['var_base'] = {
        'type': 'var',
        'max_lags': 1,
        'maxiter': -10
    }

    config_path = write_config_to_file(config, tmp_path)

    with pytest.raises(ConfigValidationError):
        load_config(config_path)


def test_var_rejects_zero_maxiter(valid_config_dict, tmp_path):
    """Test that VAR schema rejects zero maxiter values."""
    config = valid_config_dict.copy()
    config['models']['var_base'] = {
        'type': 'var',
        'max_lags': 1,
        'maxiter': 0
    }

    config_path = write_config_to_file(config, tmp_path)

    with pytest.raises(ConfigValidationError):
        load_config(config_path)


def test_var_accepts_all_valid_methods(valid_config_dict, tmp_path):
    """Test that VAR accepts all statsmodels-supported solver methods."""
    valid_methods = ["lbfgs", "bfgs", "newton", "nm", "powell"]

    for method in valid_methods:
        config = valid_config_dict.copy()
        config['models'][f'var_{method}'] = {
            'type': 'var',
            'max_lags': 1,
            'method': method
        }

        config_path = write_config_to_file(config, tmp_path)
        loaded_config = load_config(config_path)

        # Verify method was accepted
        assert loaded_config['models'][f'var_{method}']['method'] == method


def test_var_boolean_parameters_enforce_type(valid_config_dict, tmp_path):
    """Test that VAR boolean parameters enforce correct type (no strings like 'true')."""
    config = valid_config_dict.copy()
    config['models']['var_base'] = {
        'type': 'var',
        'max_lags': 1,
        'enforce_stationarity': "true"  # String, not bool
    }

    config_path = write_config_to_file(config, tmp_path)

    with pytest.raises(ConfigValidationError):
        load_config(config_path)

def test_config_accepts_new_covariate_api(valid_config_dict, tmp_path):
    """Test that new past_covariates/future_covariates API is accepted."""
    config = valid_config_dict.copy()

    # Update dataset to use new API
    config['datasets']['my_data_new_api'] = {
        'path': config['datasets']['my_data']['path'],
        'columns': ['value'],
        'past_covariates': ['past_only_feature'],  # New API
        'future_covariates': ['shared_feature'],    # New API
        'freq': 'D'
    }

    # Test that experiment can also use new API
    config['experiments'] = [{
        'name': 'test_exp',
        'dataset': 'my_data_new_api',
        'models': [{
            'name': 'transformer_v1',
            'past_covariates': ['past_only'],
            'future_covariates': ['future_known']
        }],
        'validation_setup': {
            'forecast_steps': 5,
            'n_folds': 2,
            'window_size': 10
        }
    }]

    config_path = write_config_to_file(config, tmp_path)

    # Should load without errors
    loaded_config = load_config(config_path)

    # Verify new API fields are present
    assert 'past_covariates' in loaded_config['datasets']['my_data_new_api']
    assert 'future_covariates' in loaded_config['datasets']['my_data_new_api']
    assert loaded_config['datasets']['my_data_new_api']['past_covariates'] == ['past_only_feature']
    assert loaded_config['datasets']['my_data_new_api']['future_covariates'] == ['shared_feature']
