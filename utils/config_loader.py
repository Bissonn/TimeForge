import yaml
import os
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Safely loads a YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if not config:
                raise ValueError("Config file is empty")
            return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise