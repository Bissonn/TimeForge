"""
Module for training and evaluating time series forecasting models.

This script serves as the main entry point for the forecasting framework.
It is responsible for parsing command-line arguments, loading configurations,
and delegating the primary workflow (experiment execution) to the
ExperimentRunner class.
"""

import os
import torch
import argparse
import logging
import numpy as np
from typing import Dict, Tuple, Any, Optional
import sys

# Add project root to sys.path to ensure module resolution works regardless of execution method
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_utils import load_config, ConfigValidationError
from utils.logging_config import setup_logging as setup_centralized_logging, silence_library_loggers
# Import models to register them in the global model registry
import models.transformer
import models.lstm
import models.var
import models.arima
import models.sarima
import models.simple

from core.experiment_manager import DataProvider, ExperimentRun
from core.runner import ExperimentRunner

def setup_logging(log_dir: str = "results/logs", level: str = "INFO") -> None:
    """
    Configure logging using centralized configuration.

    This is a wrapper for backward compatibility with the old setup_logging() API.
    Uses the new centralized logging configuration from utils.logging_config.

    Args:
        log_dir (str): Directory to store log files. Defaults to 'results/logs'.
        level (str): The logging level to set. Defaults to 'INFO'.
    """
    # Map level string to environment
    level_upper = level.upper()
    if level_upper == 'DEBUG':
        environment = 'debug'
    elif level_upper == 'INFO':
        environment = 'prod'
    else:
        environment = 'prod'

    # Setup centralized logging with file output
    log_file = os.path.join(log_dir, "train.log")
    console_level = getattr(logging, level_upper, logging.INFO)

    setup_centralized_logging(
        environment=environment,
        log_file=log_file,
        console_level=console_level,
        use_context=True
    )

    # Silence noisy library loggers
    silence_library_loggers()

logger = logging.getLogger(__name__)

def validate_config_path(path_str: str) -> str:
    """
    Validate and sanitize config file path to prevent path traversal attacks.

    Args:
        path_str: Config file path from command-line

    Returns:
        str: Validated absolute path

    Raises:
        ValueError: If path is invalid or unsafe
        FileNotFoundError: If config file doesn't exist
    """
    from pathlib import Path
    import re

    # Resolve to absolute path
    path = Path(path_str).resolve()

    # Security: Must be a YAML file
    if path.suffix not in ['.yaml', '.yml']:
        raise ValueError(
            f"Config file must have .yaml or .yml extension, got: {path.suffix}"
        )

    # Security: Must exist and be a file
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Config path must be a file, not a directory: {path}")

    # Security: Prevent path traversal - config should be in project or specified configs dir
    # Allow paths in current directory, configs/, or absolute paths
    try:
        # Check if in current directory
        path.relative_to(Path.cwd())
    except ValueError:
        # If not in current directory, warn but allow (for flexibility)
        logger.warning(
            f"Config file '{path}' is outside current working directory. "
            f"Ensure this is intentional and the file is trusted."
        )

    return str(path)

def validate_experiment_name(name: str) -> str:
    """
    Validate experiment name to prevent injection attacks.

    Args:
        name: Experiment name from command-line

    Returns:
        str: Validated experiment name

    Raises:
        ValueError: If name contains invalid characters
    """
    import re

    # Security: Allow only alphanumeric, underscore, hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValueError(
            f"Invalid experiment name: '{name}'. "
            f"Only alphanumeric characters, underscores, and hyphens are allowed."
        )

    # Additional check: reasonable length
    if len(name) > 100:
        raise ValueError(f"Experiment name too long (max 100 characters): '{name}'")

    return name

def parse_arguments() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """
    Parse command-line arguments for training.

    Returns:
        argparse.Namespace: Parsed arguments.

    Raises:
        ValueError: If required arguments are invalid.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate time series forecasting models.", formatter_class=argparse.RawTextHelpFormatter)

    # --- Required Arguments ---
    parser.add_argument('--experiment', type=str, required=True, help="Name of the experiment to run from config.yaml.")
    # Allow both --config and --config-path for convenience
    parser.add_argument('--config', '--config-path', dest='config_path', required=True, help="Path to the configuration file.")

    # --- Mutually Exclusive Work Modes ---
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--optimize', action='store_true', help="Run in Hyperparameter Optimization (HPO) mode.")
    mode_group.add_argument('--evaluate', action='store_true', help="Run in evaluation (backtesting) mode. This is the default behavior.")

    # --- Optional Modifiers ---
    parser.add_argument('--epochs', type=int, help="Override the number of epochs for all models (useful for dry runs/debugging).")
    parser.add_argument('--run-id', type=str, help="Specify a run ID (e.g., a timestamp) to load artifacts from.")
    parser.add_argument('--log-level', type=str, help="Override the logging level from the config file (e.g., DEBUG, INFO).")
    parser.add_argument('--no-visualization', action='store_true', help="Disable the generation of plots.")
    parser.add_argument('--force-defaults', action='store_true',
                        help="Force evaluation using default params from config.yaml, ignoring HPO results.")

    args = parser.parse_args()

    # --- Security: Validate inputs ---
    try:
        args.config_path = validate_config_path(args.config_path)
        args.experiment = validate_experiment_name(args.experiment)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))

    # --- Logic for default behavior ---
    # If no mode is specified, default to evaluation mode.
    args.evaluate = not args.optimize

    return parser, args

def initialize_environment(config: Dict, log_level_override: Optional[str] = None, experiment_name: Optional[str] = None) -> None:
    """
    Initialize the environment by setting up logging and random seeds.

    Supports both legacy and new logging configuration:
    - Legacy: logging.level (simple string)
    - New: logging.environment, logging.file, logging.custom_levels, etc.

    Args:
        config: Configuration dictionary from YAML
        log_level_override: Command-line log level override
        experiment_name: Experiment name for log file path substitution
    """
    # Get logging configuration from YAML
    logging_config = config.get("logging", {})

    # Determine environment (prod/dev/debug)
    if "environment" in logging_config:
        # New configuration style
        environment = logging_config.get("environment", "prod")
    elif "level" in logging_config or log_level_override:
        # Legacy configuration - map level to environment
        level = log_level_override or logging_config.get("level", "INFO")
        level_upper = level.upper()
        environment = "debug" if level_upper == "DEBUG" else "prod"
    else:
        environment = "prod"

    # Determine log file path
    log_file = logging_config.get("file", None)
    if log_file and experiment_name:
        # Substitute {experiment_name} placeholder
        log_file = log_file.format(experiment_name=experiment_name)
    elif log_file is None:
        # Fallback to legacy log_dir
        log_dir = config.get('log_dir', 'results/logs')
        log_file = os.path.join(log_dir, "train.log")

    # Console and file levels
    console_level_str = logging_config.get("console_level", None)
    console_level = getattr(logging, console_level_str.upper(), None) if console_level_str else None

    file_level_str = logging_config.get("file_level", None)
    file_level = getattr(logging, file_level_str.upper(), None) if file_level_str else None

    # Custom per-module levels
    custom_levels_config = logging_config.get("custom_levels", {})
    custom_levels = {}
    for module, level_str in custom_levels_config.items():
        custom_levels[module] = getattr(logging, level_str.upper())

    # Contextual logging
    use_context = logging_config.get("use_context", True)

    # Setup centralized logging
    setup_centralized_logging(
        environment=environment,
        log_file=log_file,
        console_level=console_level,
        file_level=file_level,
        custom_levels=custom_levels if custom_levels else None,
        use_context=use_context
    )

    # Silence noisy library loggers
    silence_library_loggers()

    # Set random seeds (full reproducibility)
    import random

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    logger.info(f"Environment initialized: logging_environment={environment}, experiment={experiment_name}")

def main() -> None:
    """
    Main function to run the training pipeline.

    This function orchestrates the entire process:
    1. Parses command-line arguments.
    2. Validates user input and configuration completeness.
    3. Loads the main configuration and initializes the environment.
    4. Determines which experiments to run based on arguments.
    5. Delegates the entire experiment execution to the ExperimentRunner.
    """
    # Fail-safe basic logging before anything else
    logging.basicConfig(level=logging.INFO)
    
    parser, args = parse_arguments()

    if len(sys.argv) <= 1:
        logger.warning("No arguments provided. Please specify an experiment.")
        logger.info("-" * 60)
        parser.print_help()
        return

    try:
        config = load_config(args.config_path)
    except (FileNotFoundError, ConfigValidationError) as e:
        # Create temporary logger to log the error, logger configuration might fail in config file
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Error loading configuration: {e}")
        return

    # --- Apply Global Overrides ---
    # Epochs override (Critical for Dry Run)
    if args.epochs is not None:
        logger.warning(f"⚠️  OVERRIDE: Setting epochs={args.epochs} for all models (command line argument)")
        # Override in 'models' section
        if "models" in config:
            for model_name, model_conf in config["models"].items():
                if isinstance(model_conf, dict):
                    # Only override for Neural models that have 'epochs'
                    if "epochs" in model_conf or model_conf.get("type") in ["lstm", "transformer"]:
                        model_conf["epochs"] = args.epochs
                        # Disable Early Stopping to ensure it runs exactly N epochs if it's a dry run
                        if args.epochs == 1:
                             model_conf["early_stopping_patience"] = None

    # Find the specific experiment to run from the config file first
    # (needed for experiment_name in logging configuration)
    experiment_to_run = None
    for exp in config.get("experiments", []):
        if exp.get("name") == args.experiment:
            experiment_to_run = exp
            break

    if not experiment_to_run:
        # Setup basic logging to report the error
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Experiment '{args.experiment}' not found in the configuration file.")
        return

    # Initialize environment with experiment name for log file path substitution
    initialize_environment(config, log_level_override=args.log_level, experiment_name=args.experiment)

    # Delegate the entire process to the ExperimentRunner
    runner = ExperimentRunner(experiment_to_run, config, args)
    runner.run()

if __name__ == "__main__":
    main()
