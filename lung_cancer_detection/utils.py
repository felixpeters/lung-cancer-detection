from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    """Load configuration from a YAML file

    Args:
        path (Path): Path to config file

    Returns:
        dict: Dictionary containing configuration
    """
    with open(path) as fp:
        config = yaml.safe_load(fp)
    return config
