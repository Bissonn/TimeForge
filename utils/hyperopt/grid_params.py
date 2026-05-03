"""Module for generating parameter combinations for grid search in the forecasting framework.

This module provides utilities to create all possible parameter combinations from a given
parameter space, used for hyperparameter optimization in models like ARIMA, VAR, LSTM, and Transformer.
"""

import logging
import random
from typing import Dict, List, Union

import numpy as np
import itertools

from utils.hyperopt.params_utils import _validate_param_space

logger = logging.getLogger(__name__)

def generate_grid_params(
    param_space: Dict[str, Union[List, Dict]], n_trials: int = 10
) -> List[Dict[str, Union[int, float]]]:
    """
    Generate all parameter combinations for grid search, capped at n_trials.

    Args:
        param_space: Dictionary with parameter names as keys and either:
            - Lists of values (e.g., [1, 2, 3] for discrete values).
            - Dictionaries with 'min', 'max', and optional 'grid_steps' (int >= 2) and 'log' (bool).
            - Empty dict {} for fixed parameters (returns single trial with no param changes).
        n_trials: Maximum number of combinations to return.

    Returns:
        List of dictionaries, each containing a parameter combination with parameter names
        as keys and values as integers or floats. Empty list [{}] if param_space is empty
        (allows single trial with fixed parameters from model config).

    Raises:
        ValueError: If n_trials is invalid or parameter ranges are invalid.
    """
    # Handle empty param_space (fixed parameters - single trial)
    if not param_space:
        logger.info("Empty param_space - returning single trial with no parameter changes (fixed config)")
        return [{}]

    if not isinstance(n_trials, int) or n_trials < 1:
        logger.error(f"Invalid n_trials: {n_trials}. Must be a positive integer.")
        raise ValueError("n_trials must be a positive integer")

    logger.debug(f"Generating grid parameters for param_space: {param_space}, n_trials: {n_trials}")

    # Process param_space using _validate_param_space
    try:
        param_pairs = _validate_param_space(param_space, allow_log=True)
    except Exception as e:
        logger.error(f"Failed to validate param_space: {str(e)}")
        raise ValueError(f"Invalid param_space: {str(e)}")

    keys = [key for key, _ in param_pairs]
    value_lists = []
    for key, values in param_pairs:
        if isinstance(values, dict):
            min_val, max_val = values["min"], values["max"]
            grid_steps = values.get("grid_steps", 5)
            if not isinstance(grid_steps, int) or grid_steps < 2:
                logger.error(f"Invalid grid_steps for {key}: {grid_steps}. Must be integer >= 2")
                raise ValueError(f"Invalid grid_steps for {key}: must be integer >= 2")
            if values.get("log", False):
                grid_values = np.logspace(np.log10(min_val), np.log10(max_val), grid_steps)
            else:
                grid_values = np.linspace(min_val, max_val, grid_steps)
            value_lists.append([float(np.format_float_positional(v, precision=8, unique=False)) for v in grid_values])
        else:
            value_lists.append(values)

    if not all(value_lists):
        logger.error("No valid parameter values after processing param_space")
        raise ValueError("No valid parameter values in param_space")

    # Generate all combinations
    all_combinations = list(itertools.product(*value_lists))
    logger.info(f"Generated {len(all_combinations)} parameter combinations")

    # Randomly sample if exceeding n_trials
    if len(all_combinations) > n_trials:
        logger.warning(f"Truncating {len(all_combinations)} combinations to {n_trials} using random sampling")
        all_combinations = random.sample(all_combinations, n_trials)

    combinations = [dict(zip(keys, combo)) for combo in all_combinations]

    # Validate transformer-specific constraints
    for combo in combinations:
        if "hidden_size" in combo and "num_heads" in combo:
            if combo["hidden_size"] % combo["num_heads"] != 0:
                logger.warning(
                    f"Invalid combination: hidden_size ({combo['hidden_size']}) "
                    f"not divisible by num_heads ({combo['num_heads']})"
                )
                combinations.remove(combo)

    logger.debug(f"Returning {len(combinations)} parameter combinations")
    return combinations
