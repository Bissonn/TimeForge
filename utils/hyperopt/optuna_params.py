"""Module for generating parameter combinations for Optuna optimization in the forecasting framework.

This module provides utilities to simulate Optuna trials for hyperparameter optimization,
generating parameter combinations for models like ARIMA, VAR, LSTM, and Transformer.
"""

import logging
from typing import Dict, List, Union

import numpy as np
import optuna
from optuna.exceptions import OptunaError

logger = logging.getLogger(__name__)


def generate_optuna_params(
    param_space: Dict[str, Union[List, Dict]],
    n_trials: int = 10,
    direction: str = "minimize",
    seed: int = 42
) -> List[Dict[str, Union[int, float]]]:
    """
    Generate parameter combinations for Optuna optimization by simulating trials.

    Args:
        param_space: Dictionary with parameter names as keys and either:
            - Lists for categorical values (e.g., [10, 20, 50]).
            - Dictionaries for ranges with 'min', 'max' and optional 'step' (for integers),
              'log' (bool, for floats), and 'grid_steps' (for compatibility with grid search).
            - The 'n_trials' key is ignored.
        n_trials: Number of trials to simulate (combinations to generate).
        direction: Optimization direction for Optuna study ("minimize" or "maximize"). Defaults to "minimize".
        seed: Random seed for Optuna's TPESampler. Defaults to 42 for reproducibility.

    Returns:
        List of dictionaries, each containing a parameter combination with parameter names
        as keys and values as integers or floats.

    Raises:
        ValueError: If param_space is empty, n_trials is invalid, or parameter ranges are invalid.
        OptunaError: If Optuna trial generation fails.
    """
    if not param_space:
        logger.error("param_space cannot be empty")
        raise ValueError("param_space cannot be empty")
    if not isinstance(n_trials, int) or n_trials < 1:
        logger.error(f"Invalid n_trials: {n_trials}. Must be a positive integer")
        raise ValueError("n_trials must be a positive integer")
    if direction not in ["minimize", "maximize"]:
        logger.error(f"Invalid direction: {direction}. Must be 'minimize' or 'maximize'")
        raise ValueError("direction must be 'minimize' or 'maximize'")

    logger.debug(f"Generating Optuna parameters for param_space: {param_space}, n_trials: {n_trials}")

    # Prepare keys and parameter definitions
    keys = []
    param_defs = []
    for key, value in param_space.items():
        if key == "n_trials":
            continue
        if isinstance(value, list) and value:
            keys.append(key)
            param_defs.append(("categorical", value))
        elif isinstance(value, dict) and "min" in value and "max" in value:
            min_val, max_val = value["min"], value["max"]
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                logger.error(f"Invalid min/max for {key}: must be numeric")
                raise ValueError(f"Invalid min/max for {key}: must be numeric")
            if min_val > max_val:
                logger.error(f"Invalid range for {key}: min ({min_val}) must be <= max ({max_val})")
                raise ValueError(f"Invalid range for {key}: min ({min_val}) must be <= max ({max_val})")

            log = value.get("log", False)
            has_step = "step" in value

            if log and has_step:
                logger.error(
                    f"Cannot use both 'log=True' and 'step' for '{key}': "
                    f"conflict between continuous and discrete sampling"
                )
                raise ValueError(f"Cannot use both 'log=True' and 'step' for '{key}'")

            if log or key in ["dropout", "learning_rate"]:
                keys.append(key)
                param_defs.append(("float", min_val, max_val, log))
            else:
                step = value.get("step", 1)
                if not isinstance(step, (int, float)) or step <= 0:
                    logger.error(f"Invalid step for {key}: must be positive")
                    raise ValueError(f"Invalid step for {key}: must be positive")
                keys.append(key)
                param_defs.append(("int", min_val, max_val, step))
        else:
            logger.error(f"Invalid format for {key}: must be a non-empty list or range dict")
            raise ValueError(f"Invalid format for {key}: must be a non-empty list or range dict")

    if not keys:
        logger.error("No valid parameters found in param_space")
        raise ValueError("No valid parameters found in param_space")

    # Simulate trials
    combinations = []
    try:
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))
        for _ in range(n_trials):
            trial = study.ask()
            param_dict = {}
            for key, (dist_type, *args) in zip(keys, param_defs):
                if dist_type == "categorical":
                    value = trial.suggest_categorical(key, args[0])
                elif dist_type == "float":
                    min_val, max_val, log = args
                    value = trial.suggest_float(key, min_val, max_val, log=log)
                    value = float(np.format_float_positional(value, precision=8, unique=False))
                elif dist_type == "int":
                    min_val, max_val, step = args
                    value = trial.suggest_int(key, min_val, max_val, step=step)
                param_dict[key] = value
            combinations.append(param_dict)
        logger.info(f"Generated {len(combinations)} Optuna parameter combinations")
    except OptunaError as e:
        logger.error(f"Optuna trial generation failed: {str(e)}")
        raise

    # Validate transformer-specific constraints
    filtered_combinations = []
    for combo in combinations:
        if "hidden_size" in combo and "num_heads" in combo:
            if combo["hidden_size"] % combo["num_heads"] != 0:
                logger.warning(
                    f"Skipping invalid combination: hidden_size ({combo['hidden_size']}) "
                    f"not divisible by num_heads ({combo['num_heads']})"
                )
                continue
        filtered_combinations.append(combo)

    if not filtered_combinations:
        logger.error("No valid parameter combinations after validation")
        raise ValueError("No valid parameter combinations after validation")

    logger.debug(f"Returning {len(filtered_combinations)} valid parameter combinations")
    return filtered_combinations
