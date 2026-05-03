"""
Centralized logging configuration for the time series forecasting framework.

This module provides:
1. Standardized log formatting across all modules
2. Environment-aware configuration (dev/prod/debug)
3. Multi-handler setup (console + file)
4. Per-module log level control
5. Context-aware logging (experiment, fold, epoch)
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import os
import warnings


# ============================================================================
# LOGGING LEVELS BY MODULE CATEGORY
# ============================================================================

LOG_LEVELS = {
    # Production defaults (clean output, essential info only)
    'prod': {
        'root': logging.WARNING,
        'models': logging.INFO,           # Model training progress
        'core': logging.INFO,             # Experiment execution
        'utils.train_loop': logging.INFO, # Training loop progress
        'utils.dataset': logging.WARNING, # Data warnings only
        'utils.preprocessor': logging.WARNING,
        'utils.hyperopt': logging.INFO,   # HPO progress
        'scripts': logging.INFO,          # Script execution
        'analysis': logging.INFO,         # Analysis results
    },
    # Development defaults (more verbose for debugging)
    'dev': {
        'root': logging.INFO,
        'models': logging.DEBUG,          # Full model diagnostics
        'core': logging.DEBUG,
        'utils.train_loop': logging.DEBUG, # Training diagnostics
        'utils.dataset': logging.DEBUG,
        'utils.preprocessor': logging.DEBUG,
        'utils.hyperopt': logging.DEBUG,
        'scripts': logging.DEBUG,
        'analysis': logging.INFO,
    },
    # Debug mode (everything at DEBUG level)
    'debug': {
        'root': logging.DEBUG,
        'models': logging.DEBUG,
        'core': logging.DEBUG,
        'utils': logging.DEBUG,
        'scripts': logging.DEBUG,
        'analysis': logging.DEBUG,
    }
}


# ============================================================================
# LOG FORMATTERS
# ============================================================================

class ContextualFormatter(logging.Formatter):
    """
    Custom formatter that includes contextual information (experiment, fold, epoch)
    when available in the log record.

    Format:
        2024-01-03 18:59:17 - INFO - [models.transformer] [exp:hpo_w336] [fold:2] Message
    """

    def format(self, record: logging.LogRecord) -> str:
        # Build context string from extra fields
        context_parts = []

        if hasattr(record, 'experiment'):
            context_parts.append(f"exp:{record.experiment}")
        if hasattr(record, 'model'):
            context_parts.append(f"model:{record.model}")
        if hasattr(record, 'fold'):
            context_parts.append(f"fold:{record.fold}")
        if hasattr(record, 'epoch'):
            context_parts.append(f"epoch:{record.epoch}")
        if hasattr(record, 'trial'):
            context_parts.append(f"trial:{record.trial}")

        # Add context to message
        if context_parts:
            context_str = " ".join(f"[{part}]" for part in context_parts)
            record.msg = f"{context_str} {record.msg}"

        return super().format(record)


# Standard console formatter
CONSOLE_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
CONSOLE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# File formatter (more detailed, includes line numbers)
FILE_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] %(message)s"
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

def setup_logging(
    environment: str = 'prod',
    log_file: Optional[str] = None,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    custom_levels: Optional[Dict[str, int]] = None,
    use_context: bool = True
) -> None:
    """
    Configure logging for the entire framework.

    Args:
        environment: 'prod', 'dev', or 'debug' (determines default levels)
        log_file: Path to log file (if None, no file logging)
        console_level: Override console log level
        file_level: Override file log level (defaults to DEBUG if file enabled)
        custom_levels: Dict of module name -> log level overrides
        use_context: Whether to use contextual formatter (adds exp/fold/epoch)

    Example:
        # Production mode (INFO to console, DEBUG to file)
        setup_logging('prod', log_file='results/experiment.log')

        # Development mode (DEBUG everywhere)
        setup_logging('dev')

        # Custom levels
        setup_logging('prod', custom_levels={
            'models.transformer': logging.DEBUG,
            'utils.train_loop': logging.DEBUG
        })
    """
    # Get base configuration for environment
    if environment not in LOG_LEVELS:
        raise ValueError(f"Invalid environment: {environment}. Must be 'prod', 'dev', or 'debug'")

    levels_config = LOG_LEVELS[environment].copy()

    # Apply custom level overrides
    if custom_levels:
        levels_config.update(custom_levels)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(levels_config.get('root', logging.INFO))
    root_logger.handlers.clear()  # Remove any existing handlers

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    if console_level is not None:
        console_handler.setLevel(console_level)
    else:
        # Set handler to DEBUG - let individual logger levels do the filtering
        # This ensures messages from child loggers aren't filtered by the handler
        console_handler.setLevel(logging.DEBUG)

    if use_context:
        console_formatter = ContextualFormatter(CONSOLE_FORMAT, CONSOLE_DATE_FORMAT)
    else:
        console_formatter = logging.Formatter(CONSOLE_FORMAT, CONSOLE_DATE_FORMAT)

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # --- File Handler (if enabled) ---
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        if file_level is not None:
            file_handler.setLevel(file_level)
        else:
            file_handler.setLevel(logging.DEBUG)  # File gets everything by default

        if use_context:
            file_formatter = ContextualFormatter(FILE_FORMAT, FILE_DATE_FORMAT)
        else:
            file_formatter = logging.Formatter(FILE_FORMAT, FILE_DATE_FORMAT)

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # --- Configure module-specific loggers ---
    for module_pattern, level in levels_config.items():
        if module_pattern == 'root':
            continue
        logger = logging.getLogger(module_pattern)
        logger.setLevel(level)

    # Log configuration summary (only to console, not to file to avoid recursion)
    if log_file:
        root_logger.info(
            f"Logging configured: environment={environment}, "
            f"console_level={console_level or levels_config.get('root')}, file={log_file}"
        )
    else:
        root_logger.info(
            f"Logging configured: environment={environment}, "
            f"console_level={console_level or levels_config.get('root')}"
        )


def get_contextual_logger(name: str, **context):
    """
    Get a logger with automatic context injection.

    Args:
        name: Logger name (usually __name__)
        **context: Context fields (experiment, fold, epoch, trial, model, etc.)

    Example:
        logger = get_contextual_logger(__name__, experiment='hpo_w336', fold=2)
        logger.info("Training started")  # Will include [exp:hpo_w336] [fold:2]

    Returns:
        LoggerAdapter that automatically adds context to all messages
    """
    logger = logging.getLogger(name)
    return logging.LoggerAdapter(logger, context)


def configure_from_env():
    """
    Auto-configure logging based on environment variables.

    Environment Variables:
        LOG_LEVEL: prod/dev/debug (default: prod)
        LOG_FILE: Path to log file (default: None)
        LOG_CONSOLE_LEVEL: Console log level (default: from environment)
        LOG_CONTEXT: Enable contextual logging (default: true)

    Example:
        export LOG_LEVEL=dev
        export LOG_FILE=results/training.log
        export LOG_CONSOLE_LEVEL=INFO
        python scripts/train.py config.yaml
    """
    environment = os.getenv('LOG_LEVEL', 'prod')
    log_file = os.getenv('LOG_FILE', None)
    console_level_str = os.getenv('LOG_CONSOLE_LEVEL', None)
    use_context = os.getenv('LOG_CONTEXT', 'true').lower() == 'true'

    console_level = None
    if console_level_str:
        console_level = getattr(logging, console_level_str.upper(), None)

    setup_logging(
        environment=environment,
        log_file=log_file,
        console_level=console_level,
        use_context=use_context
    )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def silence_library_loggers():
    """
    Silence noisy third-party library loggers.

    Call this after setup_logging() to reduce noise from libraries.
    """
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.WARNING)
    logging.getLogger('h5py').setLevel(logging.WARNING)

    # Suppress Optuna experimental warnings
    warnings.filterwarnings('ignore', category=Warning, module='optuna')
    warnings.filterwarnings('ignore', message='.*experimental feature.*')


def reset_logging():
    """Reset logging configuration to default state (useful for testing)."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)
