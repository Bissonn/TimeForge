import random
import numpy as np
import logging
from typing import Dict, List, Union, Tuple
from utils.hyperopt.params_utils import _validate_param_space

# Configure logging
logger = logging.getLogger(__name__)

def generate_random_params(param_space: Dict[str, Union[List, Dict]], n_trials: int = 10) -> List[Dict[str, float]]:
    """
    Generate random unique parameter combinations for hyperparameter tuning.

    This function creates a specified number of unique parameter combinations from the given
    parameter space, supporting both discrete (lists) and continuous (ranges with min/max)
    parameters. Continuous parameters can use logarithmic sampling if specified.

    Args:
        param_space: Dictionary where keys are parameter names and values are either:
            - A list of discrete values (e.g., [1, 2, 3]).
            - A dictionary with 'min', 'max', and optional 'step' or 'log' keys for ranges.
            - Empty dict {} for fixed parameters (returns single trial with no param changes).
        n_trials: Number of unique parameter combinations to generate.

    Returns:
        List[Dict[str, float]]: List of dictionaries, each containing a parameter combination.
        Empty list [{}] if param_space is empty (allows single trial with fixed parameters).

    Raises:
        ValueError: If param_space is invalid or n_trials is not positive.
        TypeError: If param_space or n_trials has an invalid type.

    Example:
        >>> param_space = {
        ...     'learning_rate': {'min': 0.001, 'max': 0.1, 'log': True},
        ...     'n_estimators': [100, 200, 300]
        ... }
        >>> params = generate_random_params(param_space, n_trials=2)
        >>> print(params)
        [{'learning_rate': 0.023456, 'n_estimators': 200}, {'learning_rate': 0.067891, 'n_estimators': 100}]
    """
    # Validate inputs
    if not isinstance(param_space, dict):
        raise TypeError("param_space must be a dictionary")

    # Handle empty param_space (fixed parameters - single trial)
    if not param_space:
        logger.info("Empty param_space - returning single trial with no parameter changes (fixed config)")
        return [{}]

    if not isinstance(n_trials, int):
        raise TypeError("n_trials must be an integer")
    if n_trials <= 0:
        raise ValueError("n_trials must be a positive integer")

    # Process param_space for random search
    param_pairs = _validate_param_space(param_space, allow_log=True)
    keys = [key for key, _ in param_pairs]
    value_lists = [values for _, values in param_pairs]

    # Generate unique random combinations
    combinations = []
    unique_combinations = set()
    max_attempts = n_trials * 10  # Configurable limit to prevent infinite loops

    for _ in range(max_attempts):
        if len(combinations) >= n_trials:
            break

        param_dict = {}
        param_values = []
        for key, values in zip(keys, value_lists):
            if isinstance(values, dict):
                min_val, max_val = values['min'], values['max']
                if values.get('log', False):
                    # Use logarithmic sampling for specified parameters
                    value = np.exp(np.random.uniform(np.log(min_val), np.log(max_val)))
                    value = round(value, 6)
                else:
                    # Use uniform sampling for continuous or discrete ranges
                    step = values.get('step', 1)
                    possible_values = np.arange(min_val, max_val + step, step)
                    value = random.choice(possible_values.tolist())
            else:
                # Select random value from discrete list
                value = random.choice(values)
            param_dict[key] = value
            param_values.append(value)

        # Ensure uniqueness of combinations
        param_tuple = tuple(param_values)
        if param_tuple not in unique_combinations:
            unique_combinations.add(param_tuple)
            combinations.append(param_dict)

    # Log warning if fewer combinations than requested
    if len(combinations) < n_trials:
        logger.warning(
            "Generated only %d unique combinations out of %d requested",
            len(combinations), n_trials
        )

    return combinations
