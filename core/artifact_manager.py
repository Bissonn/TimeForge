"""
Module for managing experiment artifacts, such as saved models, parameters,
and metrics.
"""
import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Base path for all experiment results, can be patched in tests
BASE_RESULTS_PATH = "results"


def sanitize_name(name: str, name_type: str = "name") -> str:
    r"""
    Validate and sanitize experiment/model names to prevent path traversal attacks.

    Args:
        name: The name to validate
        name_type: Description of the name type for error messages

    Returns:
        str: The validated name

    Raises:
        ValueError: If name contains invalid characters or is too long

    Security:
        - Only allows alphanumeric characters, underscores, and hyphens
        - Prevents directory traversal attacks (e.g., "../../../etc/passwd")
        - Prevents special path characters (/, \, ., etc.)
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Invalid {name_type}: must be a non-empty string")

    # Security: Allow only alphanumeric, underscore, hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValueError(
            f"Invalid {name_type}: '{name}'. "
            f"Only alphanumeric characters, underscores, and hyphens are allowed."
        )

    # Reasonable length check
    if len(name) > 100:
        raise ValueError(f"{name_type} too long (max 100 characters): '{name}'")

    return name


def validate_path_within_base(path: str) -> str:
    """
    Validate that a constructed path is within BASE_RESULTS_PATH.

    Args:
        path: The path to validate

    Returns:
        str: The validated path

    Raises:
        ValueError: If path escapes BASE_RESULTS_PATH

    Security:
        - Ensures path doesn't escape the results directory via path traversal
        - Uses resolve() to handle symbolic links and relative paths
    """
    base = Path(BASE_RESULTS_PATH).resolve()
    target = Path(path).resolve()

    try:
        target.relative_to(base)
    except ValueError:
        raise ValueError(
            f"Security: Path '{path}' is outside the allowed results directory. "
            f"This may indicate a path traversal attack."
        )

    return path

def get_run_id() -> str:
    """Generates a unique, sortable timestamp-based run ID."""
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]


def get_run_path(experiment_name: str, model_name: str, run_id: str, create: bool = True) -> str:
    """
     Constructs the full path to a specific run directory.

    Args:
        experiment_name: Name of the experiment.
        model_name: Name of the model.
        run_id: The unique ID of the run.
        create: If True, ensure the directory exists. Use False when *reading* artifacts.

    Returns:
        The absolute path to the run directory.

    Raises:
        ValueError: If experiment_name, model_name, or run_id contain invalid characters
    """
    # Security: Validate all name components to prevent path traversal
    experiment_name = sanitize_name(experiment_name, "experiment_name")
    model_name = sanitize_name(model_name, "model_name")
    run_id = sanitize_name(run_id, "run_id")

    base_path = os.path.join(BASE_RESULTS_PATH, experiment_name)
    path = os.path.join(base_path, f"{model_name}_{run_id}")

    # Security: Validate final path is within allowed directory
    path = validate_path_within_base(path)

    if create:
        os.makedirs(path, exist_ok=True)
    return path


def find_latest_run_id(experiment_name: str, model_name: str) -> Optional[str]:
    """
    Finds the latest run ID for a given experiment and model by sorting run directories.

    Args:
        experiment_name: Name of the experiment
        model_name: Name of the model

    Returns:
        The latest run ID string, or None if no runs are found.

    Raises:
        ValueError: If experiment_name or model_name contain invalid characters
    """
    # Security: Validate names to prevent path traversal
    experiment_name = sanitize_name(experiment_name, "experiment_name")
    model_name = sanitize_name(model_name, "model_name")

    base_path = os.path.join(BASE_RESULTS_PATH, experiment_name)

    # Security: Validate path is within allowed directory
    base_path = validate_path_within_base(base_path)

    if not os.path.isdir(base_path):
        return None

    run_dirs = [d for d in os.listdir(base_path) if d.startswith(f"{model_name}_")]
    if not run_dirs:
        return None

    latest_run_dir = sorted(run_dirs)[-1]
    # Extract the timestamp part from "model_name_YYYYMMDD_HHMMSS_ms"
    run_id = latest_run_dir.replace(f"{model_name}_", "", 1)
    return run_id

def find_latest_hpo_run_id(experiment_name: str, model_name: str) -> Optional[str]:
    """
    Finds the latest HPO run ID for a given experiment and model that contains
    the 'best_params.json' artifact.

    Args:
        experiment_name: Name of the experiment
        model_name: Name of the model

    Returns:
        The latest valid HPO run ID string, or None if no such run is found.

    Raises:
        ValueError: If experiment_name or model_name contain invalid characters
    """
    # Security: Validate names to prevent path traversal
    experiment_name = sanitize_name(experiment_name, "experiment_name")
    model_name = sanitize_name(model_name, "model_name")

    base_path = os.path.join(BASE_RESULTS_PATH, experiment_name)

    # Security: Validate path is within allowed directory
    base_path = validate_path_within_base(base_path)

    if not os.path.isdir(base_path):
        return None

    candidates = [d for d in os.listdir(base_path) if d.startswith(f"{model_name}_")]
    # Sort descending to check the latest first
    for d in sorted(candidates, reverse=True):
        if os.path.exists(os.path.join(base_path, d, "best_params.json")):
            return d.replace(f"{model_name}_", "", 1) # Return the ID part
    return None

def save_json(data: Dict, path: str, filename: str):
    """Saves a dictionary to a JSON file."""
    filepath = os.path.join(path, filename)
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Successfully saved {filename} to {filepath}")
    except (IOError, TypeError) as e:
        logger.error(f"Failed to save {filename} to {filepath}: {e}")
        raise


def load_json(path: str, filename: str) -> Dict:
    """Loads a dictionary from a JSON file."""
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filename} not found at {filepath}")
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load {filename} from {filepath}: {e}")
        raise
