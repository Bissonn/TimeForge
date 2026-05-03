import logging
from typing import Any, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)

def _validate_param_space(
    param_space: Dict[str, Union[List, Dict]], allow_log: bool = True
) -> List[Tuple[str, Union[List, Dict]]]:
    """
    Validate and preprocess a parameter space for hyperparameter tuning.

    This function checks the validity of a parameter space, ensuring that each parameter is
    either a non-empty list of discrete values or a dictionary specifying a range with 'min'
    and 'max' keys. Continuous ranges may include a 'step' for discrete sampling or 'log' for
    logarithmic sampling.

    Args:
        param_space: Dictionary where keys are parameter names and values are either:
            - A non-empty list of discrete values (e.g., [1, 2, 3]).
            - A dictionary with 'min', 'max', and optional 'step' or 'log' keys.
        allow_log: Whether to allow logarithmic sampling for continuous ranges.

    Returns:
        List[Tuple[str, Union[List, Dict]]]: List of tuples containing parameter names and
            their values (lists for discrete parameters, dictionaries for continuous ranges).

    Raises:
        ValueError: If param_space is invalid (e.g., empty lists, invalid ranges, or incorrect types).
        TypeError: If min, max, or step values are not numeric.

    Example:
        >>> param_space = {
        ...     'learning_rate': {'min': 0.001, 'max': 0.1, 'log': True},
        ...     'n_estimators': [100, 200, 300],
        ...     'lags': {'min': 1, 'max': 5, 'step': 1}
        ... }
        >>> _validate_param_space(param_space)
        [('learning_rate', {'min': 0.001, 'max': 0.1, 'log': True}),
         ('n_estimators', [100, 200, 300]),
         ('lags', [1, 2, 3, 4, 5])]
    """
    if not param_space:
        raise ValueError("param_space cannot be empty")

    validated_params = []
    for key, value in param_space.items():
        # Handle discrete lists
        if isinstance(value, list):
            if not value:
                raise ValueError(f"Parameter '{key}' has an empty list of values")
            validated_params.append((key, value))
            continue

        # Handle range dictionaries
        if isinstance(value, dict):
            if 'min' not in value or 'max' not in value:
                raise ValueError(f"Parameter '{key}' range must include 'min' and 'max' keys")

            min_value, max_value = value['min'], value['max']
            if not isinstance(min_value, (int, float)) or not isinstance(max_value, (int, float)):
                raise TypeError(f"Parameter '{key}' min/max must be numeric")
            if min_value > max_value:
                raise ValueError(f"Parameter '{key}' has invalid range: min ({min_value}) > max ({max_value})")

            step = value.get('step')
            log_scale = value.get('log', False)

            if log_scale and not allow_log:
                raise ValueError(f"Logarithmic sampling not allowed for parameter '{key}'")

            if step is not None:
                if not isinstance(step, (int, float)) or step <= 0:
                    raise ValueError(f"Parameter '{key}' has invalid step: must be positive number")
                if log_scale:
                    raise ValueError(f"Parameter '{key}' cannot use 'step' with 'log' sampling")
                values = list(_frange(min_value, max_value, step))
                if len(values) > 1_000_000:
                    raise ValueError(f"Too many values generated for '{key}'")
                validated_params.append((key, values))
            else:
                validated_params.append((key, value))
        else:
            raise ValueError(f"Invalid format for parameter '{key}': must be list or dict")

    if not validated_params:
        raise ValueError("No valid parameters found in param_space")

    return validated_params

def _frange(start: float, stop: float, step: float) -> List[float]:
    """Generate a range of float values."""
    vals = []
    while start <= stop:
        vals.append(round(start, 10))
        start += step
    return vals
