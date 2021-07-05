import json
import math
from pathlib import Path

import yaml
import matplotlib.pyplot as plt


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

def preview_dataset(ds, z=None):
    imgs_per_row = 6
    nrows, ncols = math.ceil(len(ds)/imgs_per_row), imgs_per_row
    plt.figure("dataset", (ncols*2.5, nrows*3))
    for i, item in enumerate(ds, start=1):
        img = item["image"].numpy()[0]
        plt.subplot(nrows, ncols, i)
        if z:
            plt.imshow(img[:,:,z], cmap="gray")
        else:
            plt.imshow(img[:,:,int(img.shape[2]/2)], cmap="gray")
        plt.title(f"label: {int(item['label'].numpy()[0])}")
    plt.show()

