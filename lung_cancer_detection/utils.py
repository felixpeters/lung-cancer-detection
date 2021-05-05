import json
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


def save_json(path: Path, data: dict):
    """Save dictionary to given file in JSON format.

    Args:
        path (Path): Path to JSON file
        data (dict): Data to save
    """
    with open(path, "w") as fp:
        json.dump(data, fp, indent=4)
    return


def load_json(path: Path) -> dict:
    """Load data from JSON file as dictionary.

    Args:
        path (Path): Path to JSON file
    Returns:
        dict: Dictionary containing data
    """
    with open(path) as fp:
        data = json.load(fp)
    return data
