"""
Configuration utilities for the ML pipeline.
"""

import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise Exception(f"Failed to load configuration: {str(e)}")
