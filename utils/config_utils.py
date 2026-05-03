"""Module for loading and validating configuration files in the forecasting framework.

This module provides utilities to load YAML configuration files and validate their structure
for datasets, models, and preprocessing steps, ensuring compatibility with models like ARIMA,
VAR, LSTM, and Transformer.
"""

import logging
import os
from typing import Dict, Optional, List

from ruamel.yaml import YAML
from ruamel.yaml.error import MarkedYAMLError
import yaml as pyyaml_loader
import re
from schema import Schema, And, Or, Use, Optional as SchemaOptional, SchemaError
from pandas.tseries.frequencies import to_offset
import textwrap

logger = logging.getLogger(__name__)

# Model names for validation, aligned with model_registry.py
# NOTE: These are now Model TYPES, not necessarily configuration keys.
MODEL_TYPES = {
    "arima", "sarima", "var", "lstm", "transformer", "simple_seasonal"
}

# A list of valid time feature strings that can be generated from a DatetimeIndex.
# This corresponds to attributes available under `pandas.Series.dt`.
ALLOWED_TIME_FEATURES = [
    "year", "month", "day", "hour", "minute", "second",
    "day_of_week", "day_of_year", "week", "quarter",
    "is_month_start", "is_month_end", "is_quarter_start",
    "is_quarter_end", "is_year_start", "is_year_end", "is_leap_year"
]

class ConfigValidationError(Exception):
    """Compact, human-readable configuration validation error."""
    pass

def _find_line_number(config_data: object, error_path: List[str]) -> Optional[int]:
    """
    Traverse the ruamel.yaml loaded data to find the line number of an error.
    If the final key is missing, fall back to the closest existing parent node.
    """
    # Try progressively shorter prefixes of the path; first success wins
    for end in range(len(error_path), -1, -1):
        try:
            node = config_data
            ok = True
            for key in error_path[:end]:
                if isinstance(key, str) and hasattr(node, 'get'):
                    if key in node:
                        node = node.get(key)
                    else:
                        ok = False
                        break
                elif isinstance(key, int) and isinstance(node, list) and len(node) > key:
                    node = node[key]
                else:
                    ok = False
                    break
            if ok and hasattr(node, 'lc'):
                return node.lc.line + 1  # ruamel uses 0-based line numbers
        except (KeyError, IndexError, AttributeError):
            continue
    return None

def _sanitize_path_tokens(tokens: List[object]) -> List[str]:
    """
    Clean up schema-reported path tokens so we only keep meaningful YAML keys.
    Accepts mixed types and returns a flat list of str/int tokens representing keys/indices.
    - Converts strings like "Key 'foo' error:" -> "foo"
    - Converts strings like "Missing key: 'bar'" -> "bar"
    - Drops None/empty/verbose non-key strings
    """
    cleaned: List[str] = []
    for t in tokens or []:
        if isinstance(t, list):
            cleaned.extend(_sanitize_path_tokens(t))
            continue
        if isinstance(t, int):
            cleaned.append(str(t))
            continue
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s or s.lower() == "none":
            continue
        m = re.match(r"^Key '([^']+)' error:\s*$", s)
        if m:
            cleaned.append(m.group(1)); continue
        m = re.match(r"^Missing key:\s*'([^']+)'\s*$", s)
        if m:
            cleaned.append(m.group(1)); continue
        if re.match(r"^[A-Za-z0-9_\-\.]+$", s):
            cleaned.append(s)
    return cleaned

def _squash_adjacent_dupes(tokens: List[str]) -> List[str]:
    """Collapse immediate adjacent duplicates in the path tokens."""
    out: List[str] = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out

def _collapse_repeated_suffix(tokens: List[str]) -> List[str]:
    """
    If tokens look like an exact concatenation of the same block A + A,
    return only A (e.g., ['models','preprocessing'] * 2 -> ['models','preprocessing']).
    """
    n = len(tokens)
    if n % 2 == 0 and n > 0:
        mid = n // 2
        if tokens[:mid] == tokens[mid:]:
            return tokens[:mid]
    return tokens

def _format_schema_error(err: SchemaError, config_data: object, config_path: str) -> str:
    """
    Format a SchemaError to be more user-friendly, including line numbers and path context.
    This version robustly handles errors from any part of the config.
    """
    # 1. Get the full error message, as it often contains the most complete key hierarchy.
    msg = str(err.code or err)

    # 2. Try to reconstruct the path by parsing the "Key '...' error:" chain from the message.
    key_chain_path = _sanitize_path_tokens(re.findall(r"Key '([^']+)' error:", msg))

    # 3. Get the path stored in the error object's attributes. This is the primary source
    #    if the error was enriched (e.g., in `validate_each_model_against_its_schema`).
    attr_path = _sanitize_path_tokens(getattr(err, 'path', None) or getattr(err, 'autos', None) or [])

    # 4. Choose the most detailed path available. An enriched path from attributes will
    #    typically be longer and more specific than one parsed from a generic message.
    if len(attr_path) > len(key_chain_path):
        final_path = attr_path
    else:
        final_path = key_chain_path

    # 5. Ensure the final missing key is appended to the path, as it might not be in the chain.
    missing_key_match = re.search(r"Missing key:\s*'([^']+)'", msg)
    if missing_key_match:
        missing_key = missing_key_match.group(1)
        if not final_path or final_path[-1] != missing_key:
            final_path.append(missing_key)

    # 6. Final cleanup for display.
    final_path = _squash_adjacent_dupes(final_path)

    # --- Rest of the formatting logic (unchanged) ---
    line_number = _find_line_number(config_data, final_path) if final_path else None
    error_message = re.sub(r' in <Schema\(.*\)>', '', str(err.code or err)).strip()
    last_line = [line.strip() for line in error_message.split('\n') if line.strip()]
    detail = last_line[-1] if last_line else error_message

    path_str = " -> ".join(final_path) if final_path else ""
    if line_number and path_str:
        message = f"Configuration error at path '{path_str}' near line {line_number}: {detail}"
    elif path_str:
        message = f"Configuration error at path '{path_str}': {detail}"
    elif line_number:
        message = f"Configuration error near line {line_number}: {detail}"
    else:
        message = f"Configuration error: {detail}"

    # Best-effort context snippet
    if line_number:
        try:
            with open(config_path, 'r', encoding='utf-8') as fh:
                lines = fh.readlines()
            start = max(0, line_number - 3)
            end = min(len(lines), line_number + 2)
            snippet = ''.join(lines[start:end]).rstrip()
            if snippet:
                message += "\n\nContext:\n" + textwrap.indent(snippet, "  ")
        except Exception:
            pass

    return message

def path_exists(path):
    """Custom validator for schema to check if a file path exists."""
    if not os.path.exists(path):
        raise SchemaError(f"Dataset file path does not exist: '{path}'")
    return True

def is_valid_freq(x: str) -> bool:
    try:
        return to_offset(x) is not None
    except ValueError:
        return False

def _define_log_transform_schema() -> Schema:
    """
    Defines the schema for the log_transform preprocessing step.

    Validates the configuration for applying a logarithmic transformation to time series data.
    This transformation can help stabilize variance and make the data more suitable for modeling.

    Parameters:
        enabled (bool): Whether the logarithmic transformation is enabled.
        method (str, optional): The logarithm method to use. Must be one of ['log', 'log1p'].
                              'log' applies natural logarithm, 'log1p' applies log(1+x). Defaults to 'log1p'.
        epsilon (float, optional): A small positive value added to avoid log(0). Defaults to 1e-6.

    Returns:
        Schema: A schema object for validating the log_transform configuration.

    Example:
        ```yaml
        log_transform:
          enabled: true
          method: "log1p"
          epsilon: 1e-6
        ```
    Raises:
        SchemaError: If the configuration does not meet the schema requirements (e.g., invalid method).
    """
    def validate_method(method):
        if method not in ["log", "log1p"]:
            raise SchemaError(f"Invalid log_transform method: {method}. Must be one of ['log', 'log1p'].")
        return method

    return Schema({
        "enabled": bool,
        SchemaOptional("method"): And(str, validate_method),
        SchemaOptional("epsilon"): And(float, lambda x: x > 0),
    })

def _define_winsorize_schema() -> Schema:
    """
    Defines the schema for the winsorize preprocessing step.

    Validates the configuration for winsorizing time series data, which limits extreme values
    to reduce the impact of outliers by clipping values at specified percentiles.

    Parameters:
        enabled (bool): Whether the winsorization transformation is enabled.
        limits (list[float], optional): A list of two floats specifying the lower and upper
                                       percentile limits for clipping (e.g., [0.05, 0.95]).
                                       Must satisfy 0 <= lower < upper <= 1.

    Returns:
        Schema: A schema object for validating the winsorize configuration.

    Example:
        ```yaml
        winsorize:
          enabled: true
          limits: [0.05, 0.95]
        ```
    Raises:
        SchemaError: If the configuration does not meet the schema requirements (e.g., invalid limits).
    """
    return Schema({
        "enabled": bool,
        SchemaOptional("limits"): And([float], lambda l: len(l) == 2 and 0 <= l[0] < l[1] <= 1),
    })

def _define_scaling_schema() -> Schema:
    """
    Defines the schema for the scaling preprocessing step.

    Validates the configuration for scaling time series data to a specified range or standardization,
    which can improve model convergence and performance.

    Parameters:
        enabled (bool): Whether the scaling transformation is enabled.
        method (str, optional): The scaling method to use. Must be one of ['minmax', 'standard', 'robust'].
                                'minmax' scales to a specified range, 'standard' standardizes to zero mean
                                and unit variance, 'robust' uses robust scaling (median/IQR). Defaults to 'minmax'.
        range (list[float|int], optional): A list of two numbers specifying the target range for
                                          'minmax' scaling (e.g., [0, 1]). Required if method is 'minmax'.

    Returns:
        Schema: A schema object for validating the scaling configuration.

    Example:
        ```yaml
        scaling:
          enabled: true
          method: "minmax"
          range: [0, 1]
        ```
    Raises:
        SchemaError: If the configuration does not meet the schema requirements (e.g., invalid method).
    """
    def validate_method(method):
        if method not in ["minmax", "standard", "robust"]:
            raise SchemaError(f"Invalid scaling method: {method}. Must be one of ['minmax', 'standard', 'robust']")
        return method

    return Schema({
        "enabled": bool,
        SchemaOptional("method"): And(str, validate_method),
        SchemaOptional("range"): And([Or(float, int)], lambda l: len(l) == 2 and l[0] < l[1]),
    })

def _define_differencing_schema() -> Schema:
    """
    Defines the schema for the differencing preprocessing step.

    Validates the configuration for applying differencing to time series data to make it stationary,
    which is often required for models like ARIMA or to improve model performance.

    Parameters:
        enabled (bool): Whether the differencing transformation is enabled.
        order (int, optional): Order of non-seasonal differencing. Must be >= 0. Defaults to 0.
        seasonal_order (int, optional): Order of seasonal differencing. Must be >= 0. Defaults to 0.
        seasonal_period (int, optional): Period of seasonality for seasonal differencing. Must be > 0.

    Returns:
        Schema: A schema object for validating the differencing configuration.

    Example:
        ```yaml
        differencing:
          enabled: true
          order: 1
          seasonal_order: 1
          seasonal_period: 12
        ```
    Raises:
        SchemaError: If the configuration does not meet the schema requirements (e.g., negative order).
    """
    return Schema({
        "enabled": bool,
        SchemaOptional("order"): And(int, lambda x: x >= 0),
        SchemaOptional("seasonal_order"): And(int, lambda x: x >= 0),
        SchemaOptional("seasonal_period"): And(int, lambda x: x > 0),
    })

def _define_pipeline_schema() -> Schema:
    """
    Dynamically defines the schema for a preprocessing pipeline by combining individual transformation schemas.

    This schema validates the preprocessing pipeline configuration, which consists of multiple transformations
    applied to time series data. Each transformation (e.g., log_transform, scaling) is optional and validated
    independently using its own schema.

    Returns:
        Schema: A schema object for validating the entire preprocessing pipeline configuration.

    Example:
        ```yaml
        preprocessing:
          log_transform:
            enabled: true
            method: "log1p"
            epsilon: 1e-6
          scaling:
            enabled: true
            method: "minmax"
            range: [0, 1]
          differencing:
            enabled: true
            order: 1
        ```
    Raises:
        SchemaError: If any transformation configuration does not meet its schema requirements.
    """
    transformation_schemas = {
        "log_transform": _define_log_transform_schema(),
        "winsorize": _define_winsorize_schema(),
        "scaling": _define_scaling_schema(),
    }
    return Schema({SchemaOptional(key): schema for key, schema in transformation_schemas.items()})

def _define_preprocessing_groups_schema() -> Schema:
    """Defines the schema for the new preprocessing_groups structure."""
    return Schema({
        SchemaOptional("time_features"): [And(str, lambda s: s in ALLOWED_TIME_FEATURES)],
        "preprocessing_groups": [{
            "name": str,
            "apply_to": Or("__targets__", [str]),
            "pipeline": _define_pipeline_schema()
        }]
    })

def validate_config(config: Dict, config_with_line_info: object, config_path: str) -> Dict:
    """
    Validate the configuration for datasets, models, and experiments.

    Args:
        config: Configuration dictionary loaded from YAML.
        config_with_line_info: line info for config data
        config_path: directory path to configuration file
    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If required configuration sections or parameters are missing or invalid.
        SchemaError: If the configuration does not match the schema.
    """
    # Shared schemas
    integer_range = Schema(And(
        {"min": int, "max": int, SchemaOptional("step"): And(int, lambda x: x > 0)},
        lambda d: d["min"] <= d["max"]
    ))
    float_range = Schema(And(
        {"min": float, "max": float, SchemaOptional("step"): And(float, lambda x: x > 0),
         SchemaOptional("log"): bool},
        lambda d: d["min"] <= d["max"]
    ))

    # Define shared schemas for HPO constraints and LR scaling
    hpo_constraints_schema = Schema({
        SchemaOptional("max_complexity_small"): And(int, lambda x: x > 0),
        SchemaOptional("max_complexity_medium"): And(int, lambda x: x > 0),
        SchemaOptional("max_complexity_large"): And(int, lambda x: x > 0),
        SchemaOptional("max_complexity_very_large"): And(int, lambda x: x > 0),
    })

    optimizer_schema = {
        SchemaOptional("optimizer"): And(str, lambda s: s.lower() in ["adam", "adamw"]),
        SchemaOptional("optimizer_config"): {
            SchemaOptional("eps"): And(float, lambda x: x > 0),
        }
    }

    hpo_lr_scaling_schema = Schema({
        SchemaOptional("mode"): And(str, lambda s: s in ["sqrt", "linear"]),
        SchemaOptional("ref_batch"): And(int, lambda x: x > 0),
        SchemaOptional("lr0"): And(float, lambda x: x > 0),
    })

    def _define_pe_config_schema() -> Schema:
        """Defines scheme for transformer positional encoding configuration."""
        return Schema({
            SchemaOptional("type"): And(str, lambda s: s in ["sinusoidal", "learnable", "none"]),
            SchemaOptional("max_len"): And(int, lambda x: x > 0),
            SchemaOptional("pe_dropout"): And(float, lambda x: 0.0 <= x <= 1.0),
            SchemaOptional("scale_with_sqrt_hidden_size"): bool
        })

    arima_opt_param_schema = Schema({
        SchemaOptional("p"): Or([And(int, lambda x: x >= 0)], integer_range),
        SchemaOptional("d"): Or([And(int, lambda x: x >= 0)], integer_range),
        SchemaOptional("q"): Or([And(int, lambda x: x >= 0)], integer_range),
        SchemaOptional("P"): Or([And(int, lambda x: x >= 0)], integer_range),
        SchemaOptional("D"): Or([And(int, lambda x: x >= 0)], integer_range),
        SchemaOptional("Q"): Or([And(int, lambda x: x >= 0)], integer_range),
        SchemaOptional("seasonal_period"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("trend"): [And(str, lambda s: s in ["n", "c", "t", "ct"])],
    })

    simple_seasonal_opt_param_schema = Schema({
        SchemaOptional("seasonal_period"): Or([And(int, lambda x: x >= 0)], integer_range),
    })

    var_opt_param_schema = Schema({
        SchemaOptional("max_lags"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("maxiter"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("ic"): [And(str, lambda s: s in ["aic", "bic", "hqic"])],
        SchemaOptional("trend"): [And(str, lambda s: s in ["n", "c", "t", "ct"])],
        SchemaOptional("error_cov_type"): [And(str, lambda s: s in ["unstructured", "diagonal", "scalar"])],
    })

    # Reusable schema for LR scheduler optimization (OneCycleLR, Cosine, Step, Plateau, etc.)
    scheduler_opt_param_schema = Schema({
        # OneCycleLR params
        SchemaOptional("pct_start"): Or([And(float, lambda x: 0 < x < 1)], float_range),
        SchemaOptional("div_factor"): Or([And(float, lambda x: x > 1)], float_range),
        SchemaOptional("final_div_factor"): Or([And(float, lambda x: x > 1)], float_range),
        SchemaOptional("anneal_strategy"): [And(str, lambda s: s in ["cos", "linear"])],

        # CosineAnnealing params
        SchemaOptional("eta_min"): Or([And(float, lambda x: x >= 0)], float_range),
        SchemaOptional("T_max"): Or([And(int, lambda x: x > 0)], integer_range),

        # Step / Exponential params
        SchemaOptional("gamma"): Or([And(float, lambda x: 0 < x < 1)], float_range),
        SchemaOptional("step_size"): Or([And(int, lambda x: x > 0)], integer_range),

        # Plateau params
        SchemaOptional("factor"): Or([And(float, lambda x: 0 < x < 1)], float_range),
        SchemaOptional("patience"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("threshold"): Or([And(float, lambda x: x > 0)], float_range),
        SchemaOptional("min_lr"): Or([And(float, lambda x: x >= 0)], float_range),
    })

    # Reusable schema for Optuna pruner configuration
    pruner_config_schema = Schema({
        SchemaOptional("type"): And(
            str, lambda s: s in ["median", "percentile", "hyperband", "threshold", "patient", "none"]
        ),

        # MedianPruner params
        SchemaOptional("n_startup_trials"): And(int, lambda x: x >= 0),
        SchemaOptional("n_warmup_steps"): And(int, lambda x: x >= 0),
        SchemaOptional("interval_steps"): And(int, lambda x: x > 0),
        SchemaOptional("n_min_trials"): And(int, lambda x: x >= 1),

        # PercentilePruner params
        SchemaOptional("percentile"): And(float, lambda x: 0 < x < 100),

        # HyperbandPruner params
        SchemaOptional("min_resource"): And(int, lambda x: x > 0),
        SchemaOptional("max_resource"): And(int, lambda x: x > 0),
        SchemaOptional("reduction_factor"): And(int, lambda x: x >= 2),

        # ThresholdPruner params
        SchemaOptional("lower"): float,
        SchemaOptional("upper"): float,

        # PatientPruner params (wraps another pruner)
        SchemaOptional("wrapped_pruner"): dict,  # Recursive pruner config
        SchemaOptional("patience"): And(int, lambda x: x >= 0),
    })

    lstm_opt_param_schema = Schema({
        SchemaOptional("strategy"): [And(str, lambda s: s in ["direct", "iterative"])],
        SchemaOptional("hidden_size"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("num_layers"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("dropout"): Or([And(float, lambda x: 0.0 <= x < 1.0)], float_range),
        SchemaOptional("batch_size"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("learning_rate"): Or([And(float, lambda x: x > 0)], float_range),
        SchemaOptional("max_grad_norm"): Or([And(float, lambda x: x >= 0)], float_range),
        SchemaOptional("epochs"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("early_stopping_patience"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("weight_decay"): Or([And(float, lambda x: x >= 0)], float_range),
        SchemaOptional("optimizer"): [And(str, lambda s: s.lower() in ["adam", "adamw"])],
        SchemaOptional("loss"): [And(str, lambda s: s in ["mse", "mae", "huber", "rmse", "smape"])],
        SchemaOptional("loss_params"): dict,
        SchemaOptional("iterative_stateful"): [bool],
        SchemaOptional("scheduler_config"): scheduler_opt_param_schema,
    })

    transformer_opt_param_schema = Schema({
        SchemaOptional("hidden_size"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("num_heads"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("num_encoder_layers"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("num_decoder_layers"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("architecture"): [And(str, lambda s: s in ['encoder-only', 'encoder-decoder'])],
        SchemaOptional("strategy"): [And(str, lambda s: s in ['direct', 'iterative'])],
        SchemaOptional("attention_type"): [And(str, lambda s: s in ['full', 'local'])],
        SchemaOptional("readout"): [And(str, lambda s: s in ["last", "mean", "max", "cls"])],
        SchemaOptional("attention_window_size"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("dim_ff_multiplier"): Or([And(float, lambda x: x > 0)], float_range),
        SchemaOptional("dropout"): Or([And(float, lambda x: 0 <= x < 1)], float_range),
        SchemaOptional("batch_size"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("learning_rate"): Or([And(float, lambda x: x > 0)], float_range),
        SchemaOptional("max_grad_norm"): Or([And(float, lambda x: x >= 0)], float_range),
        SchemaOptional("epochs"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("early_stopping_patience"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("weight_decay"): Or([And(float, lambda x: x >= 0)], float_range),
        SchemaOptional("optimizer"): [And(str, lambda s: s.lower() in ["adam", "adamw"])],
        SchemaOptional("loss"): [And(str, lambda s: s in ["mse", "mae", "l1", "huber"])],
        SchemaOptional("tgt_init"): [And(
            str, lambda s: s in ["seasonal", "trend", "last_value", "mean", "median", "zeros"]
        )],
        SchemaOptional("seasonal_period"): Or([And(int, lambda x: x > 0)], integer_range),
        SchemaOptional("use_revin"): Or(bool, [bool]),
        SchemaOptional("revin_affine"): Or(bool, [bool]),
        SchemaOptional("revin_eps"): Or([And(float, lambda x: x >= 0)], float_range),
        SchemaOptional("revin_robust"): Or(bool, [bool]),
        # Memory optimization for iterative encoder-decoder (see docs/analysis_oh2_concat_problem.md)
        SchemaOptional("iterative_decoder_mode"): [And(str, lambda s: s in ["concat", "buffer", "auto"])],
        # Performance & debugging flags (see docs/analysis_transformer_performance_bottlenecks.md)
        SchemaOptional("nan_guard_enabled"): Or(bool, [bool]),
        SchemaOptional("device_safety_checks"): Or(bool, [bool]),
        SchemaOptional("scheduler_config"): scheduler_opt_param_schema,
    })

    advanced_training_schema = {
        SchemaOptional("use_amp"): bool,
        SchemaOptional("max_grad_norm"): And(float, lambda x: x >= 0),
        SchemaOptional("save_horizon_csv"): bool,
        SchemaOptional("auto_tune_horizon"): bool,
        SchemaOptional("degradation_threshold"): And(float, lambda x: x > 1.0),
        SchemaOptional("save_scheduler_plot"): bool,
        SchemaOptional("save_scheduler_csv"): bool,
    }

    # Scheme for Gradient Monitor (used in LSTM and Transformer)
    gradient_monitor_schema = {
        SchemaOptional("gradient_monitor"): {
            "enabled": bool,
            SchemaOptional("log_dir"): str,
            SchemaOptional("log_interval"): And(int, lambda x: x > 0),
        }
    }

    # Scheme for Attention Capture (used in Transformer)
    attention_capture_schema = {
        SchemaOptional("attention_capture"): {
            "enabled": bool,
            SchemaOptional("log_dir"): str,
        }
    }

    # Schema for loss configuration
    loss_schema = {
        SchemaOptional("loss"): And(str, lambda s: s in ["mse", "mae", "l1", "huber"]),
        SchemaOptional("loss_params"): {SchemaOptional("delta"): And(Or(int, float), lambda x: x > 0)},
    }

    # ─────────────────────────────────────────────────────────────────────────────
    # SCHEDULER CONFIG SCHEMA (reusable for LSTM and Transformer)
    # ─────────────────────────────────────────────────────────────────────────────
    # Validates learning rate scheduler configuration for neural models

    def _define_scheduler_config_schema():
        '''
        Schema for scheduler_config parameter in neural models.

        Supports multiple scheduler types with their specific parameters:
        - onecycle: OneCycleLR with warmup and annealing
        - cosine: CosineAnnealingLR for smooth decay
        - step: StepLR for periodic reductions
        - exponential: ExponentialLR for exponential decay
        - plateau: ReduceLROnPlateau for adaptive reduction
        '''
        return Schema({
            # Required: scheduler type
            SchemaOptional("type"): And(
                str,
                lambda s: s in ["onecycle", "cosine", "step", "exponential", "plateau"],
                error="scheduler type must be one of: onecycle, cosine, step, exponential, plateau"
            ),

            # OneCycleLR parameters
            SchemaOptional("max_lr"): And(
                Use(float),
                lambda x: x > 0,
                error="max_lr must be positive"
            ),
            SchemaOptional("pct_start"): And(
                Use(float),
                lambda x: 0 < x < 1,
                error="pct_start must be between 0 and 1"
            ),
            SchemaOptional("div_factor"): And(
                Use(float),
                lambda x: x > 1,
                error="div_factor must be > 1"
            ),
            SchemaOptional("final_div_factor"): And(
                Use(float),
                lambda x: x > 1,
                error="final_div_factor must be > 1"
            ),
            SchemaOptional("anneal_strategy"): And(
                str,
                lambda s: s in ["cos", "linear"],
                error="anneal_strategy must be 'cos' or 'linear'"
            ),

            # CosineAnnealingLR parameters
            SchemaOptional("T_max"): And(
                Use(int),
                lambda x: x > 0,
                error="T_max must be positive"
            ),
            SchemaOptional("eta_min"): And(
                Use(float),
                lambda x: x >= 0,
                error="eta_min must be non-negative"
            ),

            # StepLR / ExponentialLR parameters
            SchemaOptional("step_size"): And(
                Use(int),
                lambda x: x > 0,
                error="step_size must be positive"
            ),
            SchemaOptional("gamma"): And(
                Use(float),
                lambda x: 0 < x < 1,
                error="gamma must be between 0 and 1"
            ),

            # ReduceLROnPlateau parameters
            SchemaOptional("mode"): And(
                str,
                lambda s: s in ["min", "max"],
                error="mode must be 'min' or 'max'"
            ),
            SchemaOptional("factor"): And(
                Use(float),
                lambda x: 0 < x < 1,
                error="factor must be between 0 and 1"
            ),
            SchemaOptional("patience"): And(
                Use(int),
                lambda x: x > 0,
                error="patience must be positive"
            ),
            SchemaOptional("threshold"): And(
                Use(float),
                lambda x: x > 0,
                error="threshold must be positive"
            ),
            SchemaOptional("threshold_mode"): And(
                str,
                lambda s: s in ["rel", "abs"],
                error="threshold_mode must be 'rel' or 'abs'"
            ),
            SchemaOptional("cooldown"): And(
                Use(int),
                lambda x: x >= 0,
                error="cooldown must be non-negative"
            ),
            SchemaOptional("min_lr"): And(
                Use(float),
                lambda x: x >= 0,
                error="min_lr must be non-negative"
            ),
            SchemaOptional("eps"): And(
                Use(float),
                lambda x: x > 0,
                error="eps must be positive"
            ),
        })

    MODEL_SCHEMAS = {
        "arima": Schema({
            "type": And(str, lambda s: s == "arima"),
            "p": And(int, lambda x: x >= 0),
            "d": And(int, lambda x: x >= 0),
            "q": And(int, lambda x: x >= 0),
            SchemaOptional("trend"): And(str, lambda s: s in ["n", "c", "t", "ct"]),
            SchemaOptional("use_exogenous"): bool,
            # ═══════════════════════════════════════════════════════════════════
            # NEW: Advanced ARIMA Configuration Parameters
            # ═══════════════════════════════════════════════════════════════════
            SchemaOptional("enforce_stationarity"): bool,
            SchemaOptional("enforce_invertibility"): bool,
            SchemaOptional("method"): And(str, lambda s: s in ["lbfgs", "css-mle", "bfgs", "newton", "nm", "powell"]),
            SchemaOptional("maxiter"): And(int, lambda x: x > 0),
            SchemaOptional("remove_data_after_fit"): bool,
            # ═══════════════════════════════════════════════════════════════════
            SchemaOptional("optimize"): bool,
            SchemaOptional("optimization"): {
                "method": And(str, lambda s: s in ["grid", "random", "optuna"]),
                SchemaOptional("n_trials"): And(int, lambda x: x > 0),
                SchemaOptional("params"): arima_opt_param_schema,
                SchemaOptional("pruner_config"): pruner_config_schema,
            },
            SchemaOptional("preprocessing"): _define_preprocessing_groups_schema(),
        }),
        "simple_seasonal": Schema({
            "type": And(str, lambda s: s == "simple_seasonal"),
            SchemaOptional("seasonal_period"): And(int, lambda x: x > 0),
            SchemaOptional("optimize"): bool,
            SchemaOptional("optimization"): {
                "method": And(str, lambda s: s in ["grid", "random", "optuna"]),
                SchemaOptional("n_trials"): And(int, lambda x: x > 0),
                SchemaOptional("params"): simple_seasonal_opt_param_schema,
            },
            SchemaOptional("preprocessing"): _define_preprocessing_groups_schema(),
        }),
        "sarima": Schema({
            "type": And(str, lambda s: s == "sarima"),
            "p": And(int, lambda x: x >= 0),
            "d": And(int, lambda x: x >= 0),
            "q": And(int, lambda x: x >= 0),
            "P": And(int, lambda x: x >= 0),
            "D": And(int, lambda x: x >= 0),
            "Q": And(int, lambda x: x >= 0),
            SchemaOptional("trend"): And(str, lambda s: s in ["n", "c", "t", "ct"]),
            SchemaOptional("seasonal_period"): And(int, lambda x: x > 0),
            SchemaOptional("use_exogenous"): bool,
            # ═══════════════════════════════════════════════════════════════════
            # NEW: Advanced SARIMA Configuration Parameters
            # ═══════════════════════════════════════════════════════════════════
            SchemaOptional("enforce_stationarity"): bool,
            SchemaOptional("enforce_invertibility"): bool,
            SchemaOptional("method"): And(str, lambda s: s in ["lbfgs", "css-mle", "bfgs", "newton", "nm", "powell"]),
            SchemaOptional("maxiter"): And(int, lambda x: x > 0),
            SchemaOptional("remove_data_after_fit"): bool,
            # ═══════════════════════════════════════════════════════════════════
            SchemaOptional("optimize"): bool,
            SchemaOptional("optimization"): {
                "method": And(str, lambda s: s in ["grid", "random", "optuna"]),
                SchemaOptional("n_trials"): And(int, lambda x: x > 0),
                SchemaOptional("params"): arima_opt_param_schema,
                SchemaOptional("pruner_config"): pruner_config_schema,
            },
            SchemaOptional("preprocessing"): _define_preprocessing_groups_schema(),
        }),
        "var": Schema({
            "type": And(str, lambda s: s == "var"),
            "max_lags": And(int, lambda x: x > 0),
            SchemaOptional("ic"): And(str, lambda s: s in ["aic", "bic", "hqic"]),
            SchemaOptional("trend"): And(str, lambda s: s in ["n", "c", "t", "ct"]),
            SchemaOptional("error_cov_type"): And(str, lambda s: s in ["unstructured", "diagonal", "scalar"]),
            SchemaOptional("maxiter"): And(int, lambda x: x > 0),
            SchemaOptional("use_exogenous"): bool,
            # ═══════════════════════════════════════════════════════════════════
            # NEW: Advanced VAR Configuration Parameters
            # ═══════════════════════════════════════════════════════════════════
            SchemaOptional("enforce_stationarity"): bool,
            SchemaOptional("enforce_invertibility"): bool,
            SchemaOptional("method"): And(str, lambda s: s in ["lbfgs", "bfgs", "newton", "nm", "powell"]),
            SchemaOptional("remove_data_after_fit"): bool,
            # ═══════════════════════════════════════════════════════════════════
            SchemaOptional("optimize"): bool,
            SchemaOptional("optimization"): {
                "method": And(str, lambda s: s in ["grid", "random", "optuna"]),
                SchemaOptional("n_trials"): And(int, lambda x: x > 0),
                SchemaOptional("params"): var_opt_param_schema,
                SchemaOptional("pruner_config"): pruner_config_schema,
            },
            SchemaOptional("preprocessing"): _define_preprocessing_groups_schema(),
        }),
        "lstm": Schema({
            "type": And(str, lambda s: s == "lstm"),
            "strategy": And(str, lambda s: s in ["direct", "iterative"]),
            "hidden_size": And(int, lambda x: x > 0),
            "num_layers": And(int, lambda x: x > 0),
            SchemaOptional("dropout"): And(float, lambda x: 0.0 <= x < 1.0),
            SchemaOptional("batch_size"): And(int, lambda x: x > 0),
            SchemaOptional("learning_rate"): And(float, lambda x: x > 0),
            SchemaOptional("epochs"): And(int, lambda x: x > 0),
            SchemaOptional("early_stopping_patience"): And(int, lambda x: x > 0),
            SchemaOptional("min_epochs"): And(int, lambda x: x > 0),  # Minimum epochs before early stopping can trigger
            SchemaOptional("weight_decay"): And(float, lambda x: x >= 0),
            # DataLoader workers: 0=main process, >0=multiprocessing
            SchemaOptional("num_workers"): And(int, lambda x: x >= 0),
            SchemaOptional("use_exogenous"): bool,
            # Past covariate handling in iterative mode
            SchemaOptional("past_covariate_policy"): And(
                str, lambda s: s in ["frozen", "last_window", "zero", "custom"]
            ),
            # Phase 1 & 2a: Future covariate support
            SchemaOptional("future_covariate_mode"): And(str, lambda s: s in ["none", "global", "stepwise"]),
            SchemaOptional("future_context_config"): {
                SchemaOptional("pooling"): And(str, lambda s: s in ["mean", "last", "learnable"]),
                SchemaOptional("compression_dim"): And(int, lambda x: x > 0),
                SchemaOptional("dropout"): And(float, lambda x: 0.0 <= x < 1.0),
            },
            # Phase 2a: Stateful iterative prediction (LSTM state propagation)
            SchemaOptional("iterative_stateful"): bool,
            # LSTM Train/Inference Mismatch Mitigation (analogous to Transformer Phase 2)
            SchemaOptional("fc_dropout"): And(float, lambda x: 0.0 <= x <= 1.0),
            SchemaOptional("input_noise_injection"): {
                "enabled": bool,
                SchemaOptional("std"): And(float, lambda x: x > 0),
                SchemaOptional("probability"): And(float, lambda x: 0.0 <= x <= 1.0),
            },
            **loss_schema,
            **optimizer_schema,
            SchemaOptional("optimize"): bool,
            SchemaOptional("optimization"): {
                "method": And(str, lambda s: s in ["grid", "random", "optuna"]),
                SchemaOptional("n_trials"): And(int, lambda x: x > 0),
                SchemaOptional("params"): lstm_opt_param_schema,
                SchemaOptional("pruner_config"): pruner_config_schema,
            },
            SchemaOptional("scheduler_config"): _define_scheduler_config_schema(),
            SchemaOptional("preprocessing"): _define_preprocessing_groups_schema(),
            **advanced_training_schema,
            ** gradient_monitor_schema,
            SchemaOptional("hpo_constraints"): hpo_constraints_schema,
            SchemaOptional("hpo_lr_scaling"): hpo_lr_scaling_schema,
        }),
        "transformer": Schema({
            "type": And(str, lambda s: s == "transformer"),
            SchemaOptional("hidden_size"): And(int, lambda x: x > 0),
            SchemaOptional("num_heads"): And(int, lambda x: x > 0),
            SchemaOptional("num_encoder_layers"): And(int, lambda x: x > 0),
            SchemaOptional("num_decoder_layers"): And(int, lambda x: x > 0),
            SchemaOptional("architecture"): And(str, lambda s: s in ['encoder-only', 'encoder-decoder']),
            SchemaOptional("activation"): And(str, lambda s: s in ["relu", "gelu"]),
            SchemaOptional("norm_first"): bool,
            SchemaOptional("positional_encoding_config"): _define_pe_config_schema(),
            SchemaOptional("readout"): And(str, lambda s: s in ["last", "mean", "max", "cls"]),
            SchemaOptional("attention_type"): And(str, lambda s: s in ["full", "local"]),
            SchemaOptional("attention_window_size"): And(int, lambda x: x > 0),
            SchemaOptional("strategy"): And(str, lambda s: s in ['direct', 'iterative']),
            SchemaOptional("dim_ff_multiplier"): And(float, lambda x: x > 0),
            SchemaOptional("tgt_init"): And(
                str, lambda s: s in ["seasonal", "trend", "last_value", "mean", "median", "zeros"]
            ),
            SchemaOptional("seasonal_period"): And(int, lambda x: x > 0),
            SchemaOptional("use_revin"): bool,
            SchemaOptional("revin_affine"): bool,
            SchemaOptional("revin_eps"): And(float, lambda x: x > 0),
            SchemaOptional("revin_robust"): bool,
            SchemaOptional("dropout"): And(float, lambda x: 0 <= x < 1),
            SchemaOptional("batch_size"): And(int, lambda x: x > 0),
            SchemaOptional("learning_rate"): And(float, lambda x: x > 0),
            SchemaOptional("epochs"): And(int, lambda x: x > 0),
            SchemaOptional("early_stopping_patience"): And(int, lambda x: x > 0),
            SchemaOptional("min_epochs"): And(int, lambda x: x > 0),  # Minimum epochs before early stopping can trigger
            SchemaOptional("weight_decay"): And(float, lambda x: x >= 0),
            # DataLoader workers: 0=main process, >0=multiprocessing
            SchemaOptional("num_workers"): And(int, lambda x: x >= 0),
            SchemaOptional("use_exogenous"): bool,
            # Phase 2: Auxiliary Multi-Step Loss
            SchemaOptional("auxiliary_loss"): {
                "enabled": bool,
                SchemaOptional("weight"): And(float, lambda x: 0.0 <= x <= 1.0),
                SchemaOptional("position_weighting"): bool,
            },
            # Phase 2: Prediction Noise Injection
            SchemaOptional("prediction_noise"): {
                "enabled": bool,
                SchemaOptional("std"): And(float, lambda x: x > 0),
                SchemaOptional("schedule"): And(str, lambda s: s in ["constant", "curriculum"]),
            },
            # Memory optimization for iterative encoder-decoder
            SchemaOptional("iterative_decoder_mode"): And(str, lambda s: s in ["concat", "buffer", "auto"]),
            # Output head configuration
            SchemaOptional("head_type"): And(str, lambda s: s in ["linear", "mlp"]),
            SchemaOptional("output_head_strategy"): And(str, lambda s: s in ["shared", "multiple"]),
            # Safety & debugging flags
            SchemaOptional("nan_guard_enabled"): bool,
            SchemaOptional("device_safety_checks"): bool,
            # AMP inference configuration
            SchemaOptional("use_amp_inference"): bool,
            SchemaOptional("amp_inference_dtype"): Or(str, type(None)),
            SchemaOptional("debug_amp_inference"): bool,
            **loss_schema,
            **optimizer_schema,
            SchemaOptional("optimize"): bool,
            SchemaOptional("optimization"): {
                "method": And(str, lambda s: s in ["grid", "random", "optuna"]),
                SchemaOptional("n_trials"): And(int, lambda x: x > 0),
                SchemaOptional("params"): transformer_opt_param_schema,
                SchemaOptional("pruner_config"): pruner_config_schema,
            },
            SchemaOptional("scheduler_config"): _define_scheduler_config_schema(),
            SchemaOptional("preprocessing"): _define_preprocessing_groups_schema(),
            **advanced_training_schema,
            **gradient_monitor_schema,
            **attention_capture_schema,
            SchemaOptional("hpo_constraints"): hpo_constraints_schema,
            SchemaOptional("hpo_lr_scaling"): hpo_lr_scaling_schema,
        }),
    }

    def validate_each_model_against_its_schema(models_dict: Dict) -> Dict:
        """
        Validates each model in the config against its corresponding schema based on 'type'.

        The key in `models_dict` is now the configuration name (e.g., 'transformer_large'),
        and the value MUST contain a 'type' field (e.g., 'type': 'transformer').
        """

        for config_name, model_params in models_dict.items():
            # 1. Check if 'type' field is present
            if "type" not in model_params:
                # Enhance error path
                error = SchemaError(f"Model configuration '{config_name}' is missing required field 'type'.")
                full_path = ['models', config_name, 'type']
                setattr(error, 'path', full_path)
                raise error

            model_type = model_params["type"]

            # 2. Check if the type is valid
            if model_type not in MODEL_SCHEMAS:
                sorted_model_types = sorted(list(MODEL_SCHEMAS.keys()))
                error = SchemaError(
                    f"Invalid model type '{model_type}' in configuration '{config_name}'. "
                    f"Must be one of {sorted_model_types}"
                )
                full_path = ['models', config_name, 'type']
                setattr(error, 'path', full_path)
                raise error

        # 3. Validate using the specific schema for this type
            try:
                MODEL_SCHEMAS[model_type].validate(model_params)
            except SchemaError as e:
                new_message = f"Configuration '{config_name}' (type: {model_type}) error:\n{e.code or e}"
                inner_path = _sanitize_path_tokens(getattr(e, 'path', None) or getattr(e, 'autos', None) or [])
                full_path = ['models', config_name] + inner_path
                enriched = SchemaError(new_message, errors=getattr(e, 'errors', None))
                setattr(enriched, 'path', full_path)
                raise enriched

        return models_dict

    def validate_experiment_model_name(name: str) -> bool:
        """
        Validates that the model name in experiment refers to a defined model configuration.
        We cannot validate existence here easily without passing the full config,
        so we check basically that it's a string. Existence is checked in validate_experiments_and_models.
        """
        return isinstance(name, str) and len(name) > 0

    experiment_schema = Schema({
        "name": str,
        SchemaOptional("description"): str,
        "dataset": str,
        "models": [
            Or(
                # Option 1: User provided only the name (string) -> convert to dict {'name': '...'}
                And(str, validate_experiment_model_name, Use(lambda s: {"name": s})),
                # Option 2: User provided full configuration (dict) -> validate as before
                {
                    "name": And(str, validate_experiment_model_name),
                    SchemaOptional("use_exogenous"): bool,
                    SchemaOptional("preprocessing"): _define_preprocessing_groups_schema(),
                    SchemaOptional("past_covariates"): [str],
                    SchemaOptional("future_covariates"): [str],
                    SchemaOptional("use_raw_data_source"): bool,
                }
            )
        ],
        "validation_setup": {
            "forecast_steps": And(int, lambda x: x > 0),
            "n_folds": And(int, lambda x: x > 0),
            "window_size": And(int, lambda x: x > 0),
            SchemaOptional("early_stopping_validation_percentage"): And(Or(int, float), lambda x: 0 < x <= 100),
            SchemaOptional("evaluation_metric"): And(str, lambda s: s in ['mse', 'rmse', 'mae', 'smape', 'mase']),
        },
    })

    def validate_models(models):
        """Validate the models section to ensure at least one model is defined."""
        if not models:
            raise SchemaError("At least one model must be defined in the 'models' section")
        return models

    def validate_experiments_and_models(config: Dict) -> Dict:
        """Validate that all models in experiments are defined in the models section."""
        if "experiments" in config:
            defined_model_configs = set(config["models"].keys())
            for exp in config["experiments"]:
                if "models" in exp and exp["models"]:
                    for model in exp["models"]:
                        config_name = model["name"]
                        if config_name not in defined_model_configs:
                            raise SchemaError(
                                f"Model configuration '{config_name}' used in experiment '{exp.get('name')}' "
                                f"is not defined in the global 'models' section."
                            )
        return config

    main_schema = Schema({
        SchemaOptional("paths"): {
            SchemaOptional("model_save_path_template"): str
        },
        SchemaOptional("logging"): {
            # Legacy support - simple level string
            SchemaOptional("level"): And(str, lambda s: s.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
            # New centralized logging configuration
            SchemaOptional("environment"): And(str, lambda s: s.lower() in ["prod", "dev", "debug"]),
            SchemaOptional("file"): str,  # Path to log file (supports {experiment_name} placeholder)
            SchemaOptional("console_level"): And(
                str, lambda s: s.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            ),
            SchemaOptional("file_level"): And(
                str, lambda s: s.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            ),
            SchemaOptional("use_context"): bool,  # Enable contextual logging (exp/fold/epoch tags)
            SchemaOptional("custom_levels"): {  # Per-module log level overrides
                str: And(str, lambda s: s.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
            },
        },
        SchemaOptional("experiments"): [experiment_schema],
        "datasets": {
            str: {
                "path": And(str, path_exists),
                "columns": [str],
                SchemaOptional("past_covariates"): [str],
                SchemaOptional("future_covariates"): [str],
                SchemaOptional("freq"): And(str, is_valid_freq),
                SchemaOptional("preprocessing"): _define_pipeline_schema(),
                SchemaOptional("time_features"): [And(str, lambda s: s in ALLOWED_TIME_FEATURES)],
                SchemaOptional("differencing"): _define_differencing_schema(),
            },
        },
        "models": And(
            {str: dict},  # Allow arbitrary string keys for model configuration names
            validate_models,
            validate_each_model_against_its_schema
        ),
    })
    main_schema = Schema(And(
        main_schema,
        validate_experiments_and_models
    ))

    try:
        validated_config = main_schema.validate(config)
        logger.info("Configuration validation passed successfully")
        return validated_config
    except SchemaError as e:
        error_message = _format_schema_error(e, config_with_line_info, config_path)
        raise ConfigValidationError(error_message)

def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load and validate a configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file. Defaults to 'config.yaml'.

    Returns:
        Validated configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
        SchemaError: If the configuration does not match the schema.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # Security: Use safe YAML loading to prevent arbitrary code execution
    # - ruamel.yaml YAML(typ='rt') is safe (round-trip mode, no Python object execution)
    #   Used ONLY for line/column info to provide better error messages
    # - pyyaml.safe_load() is the standard safe loader (only allows standard Python types)
    #   Used for actual configuration data
    yaml_parser = YAML(typ='rt') # Use round-trip loader to robustly preserve line/column info (.lc)
    try:
        with open(config_path, 'r') as file:
            config_with_line_info = yaml_parser.load(file)  # Safe: typ='rt' doesn't execute code
        with open(config_path, 'r') as file:
            plain_config = pyyaml_loader.safe_load(file)  # Safe: only loads standard types

        if not plain_config:
            raise ValueError("Configuration file is empty.")

        return validate_config(plain_config, config_with_line_info, config_path)
    except MarkedYAMLError as e:
        if hasattr(e, 'problem_mark') and e.problem_mark:
            line = e.problem_mark.line + 1
            col = e.problem_mark.column + 1
            raise ConfigValidationError(
                f"YAML syntax error in {config_path} at line {line}, column {col}: {getattr(e, 'problem', str(e))}"
            )
        raise ConfigValidationError(f"YAML syntax error: {e}")
    except pyyaml_loader.YAMLError as e:
        mark = getattr(e, 'problem_mark', None)
        if mark:
            line = mark.line + 1
            col = mark.column + 1
            raise ConfigValidationError(
                f"YAML syntax error in {config_path} at line {line}, column {col}"
            )
        raise ConfigValidationError(f"YAML syntax error: {e}")
    except SchemaError as e:
        error_message = _format_schema_error(e, config_with_line_info, config_path)
        raise ConfigValidationError(error_message)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}", exc_info=False)
        raise ConfigValidationError(f"Failed to load configuration from {config_path}: {e}")

def get_model_config(config_name: str, config_path: str = "config.yaml") -> Dict:
    """
    Load configuration for a specific model config name from a YAML file.

    Args:
        config_name: Name of the model configuration (key in 'models' section).
        config_path: Path to the YAML configuration file. Defaults to 'config.yaml'.

    Returns:
        Model configuration dictionary, with default optimization settings if not specified.

    Raises:
        ValueError: If config_name is invalid or no configuration is found.
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
    """
    config = load_config(config_path)
    model_config = config.get("models", {}).get(config_name, {})
    if not model_config:
        # If it's not found, raise error as per new structure requirement
        raise ValueError(f"Model configuration '{config_name}' not found in 'models' section of {config_path}.")

    model_config.setdefault("optimization", {"method": "grid", "params": {}})
    return model_config
