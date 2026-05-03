"""Unit tests for the dependency checking utility.

This module provides a comprehensive suite of tests for the `check_dependencies`
function located in `utils/dependencies.py`. The tests cover various scenarios,
including successful validation, missing model-specific dependencies, missing
optional dependencies, and invalid function arguments. All external dependencies
are mocked to ensure the tests are isolated and run deterministically.
"""

import pytest
from unittest.mock import MagicMock
from utils import dependencies
import logging

pytestmark = pytest.mark.unit

# --- Fixtures and Mocks Setup ---

@pytest.fixture
def mock_importer(mocker):
    """
    Fixture to mock `importlib.util.find_spec`.
    By default, it simulates that all libraries are installed by returning a mock spec object.
    """
    return mocker.patch('utils.dependencies.importlib.util.find_spec', return_value=MagicMock())

@pytest.fixture
def mock_model_registry(mocker):
    """Fixture to mock the model registry."""
    return mocker.patch(
        'utils.dependencies.list_registered_models',
        return_value=["arima", "sarima", "var", "lstm"]
    )

# --- Test Cases ---

def test_check_dependencies_all_installed(mock_importer, mock_model_registry, caplog):
    """
    Scenario: All required and optional dependencies are installed.
    Assumptions: The function should execute without raising any exceptions and
    should log a success message.
    """
    with caplog.at_level(logging.INFO):
        dependencies.check_dependencies()
    assert "All required libraries for models" in caplog.text
    assert "Missing required libraries" not in caplog.text

def test_check_dependencies_missing_model_specific_library(mock_importer, mock_model_registry):
    """
    Scenario: A required dependency for a specific model is missing.
    Assumptions: The function should raise an ImportError with a message that
    correctly identifies the missing library, its purpose, and the installation command.
    """
    mock_importer.side_effect = lambda module_name: None if module_name == 'statsmodels' else MagicMock()

    with pytest.raises(ImportError) as excinfo:
        dependencies.check_dependencies(model_names=['arima'])

    error_message = str(excinfo.value)
    assert "Missing required libraries" in error_message
    assert "statsmodels" in error_message
    assert "ARIMA implementation from statsmodels" in error_message
    assert "pip install statsmodels" in error_message

def test_check_dependencies_missing_optional_library(mock_importer, mock_model_registry):
    """
    Scenario: An optional dependency is missing when checking all models.
    Assumptions: The function should raise an ImportError with the correct details.
    """
    mock_importer.side_effect = lambda module_name: None if module_name == 'optuna' else MagicMock()

    with pytest.raises(ImportError) as excinfo:
        dependencies.check_dependencies(model_names=None)

    error_message = str(excinfo.value)
    assert "Optuna" in error_message
    assert "hyperparameter optimization" in error_message
    assert "pip install optuna" in error_message

def test_check_dependencies_multiple_missing(mock_importer, mock_model_registry):
    """
    Scenario: Multiple required dependencies are missing.
    Assumptions: The ImportError message should list all missing libraries.
    """
    mock_importer.side_effect = lambda module_name: None if module_name in ['torch', 'statsmodels'] else MagicMock()

    with pytest.raises(ImportError) as excinfo:
        dependencies.check_dependencies(model_names=['arima', 'lstm'])

    error_message = str(excinfo.value)
    assert "statsmodels" in error_message
    assert "PyTorch" in error_message

def test_check_dependencies_invalid_model_name(mock_model_registry):
    """
    Scenario: The function is called with a model name that is not registered.
    Assumptions: A ValueError should be raised, indicating which model name is invalid.
    """
    with pytest.raises(ValueError, match=r"Invalid model names: \['nonexistent_model'\]"):
        dependencies.check_dependencies(model_names=['nonexistent_model'])

def test_check_dependencies_invalid_package_manager():
    """
    Scenario: The function is called with an unsupported package manager.
    Assumptions: A ValueError should be raised.
    """
    with pytest.raises(ValueError, match="package_manager must be 'pip' or 'conda'"):
        dependencies.check_dependencies(package_manager='npm')

def test_check_dependencies_generates_conda_install_command(mock_importer, mock_model_registry):
    """
    Scenario: A dependency is missing, and the package manager is set to 'conda'.
    Assumptions: The installation command in the error message should use 'conda install'.
    """
    # Simulate that 'statsmodels' is missing
    mock_importer.side_effect = lambda module_name: None if module_name == 'statsmodels' else MagicMock()

    with pytest.raises(ImportError) as excinfo:
        # Check a model that requires statsmodels, with conda as the package manager
        dependencies.check_dependencies(model_names=['var'], package_manager='conda')

    # Verify the error message contains the correct conda command
    assert "conda install statsmodels" in str(excinfo.value)

def test_check_dependencies_handles_no_specific_dependencies(mock_importer, mock_model_registry, caplog):
    """
    Scenario: A model with only common dependencies is checked.
    Assumptions: The function should run successfully without raising an error.
    """
    # Simulate that only numpy and pandas are installed, which are enough for VAR
    mock_importer.side_effect = lambda module_name: module_name in ['numpy', 'pandas', 'statsmodels']

    with caplog.at_level(logging.INFO):
        dependencies.check_dependencies(model_names=['var'])

    assert "All required libraries for models ['var'] are installed." in caplog.text
