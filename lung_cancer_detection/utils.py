import json
import math
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np


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

def preview_dataset(ds: Dataset, z: int = None, preds: np.ndarray = None):
    """Displays given slice of nodule images in dataset.

    Args:
        ds (Dataset): Dataset containing nodule volumes.
        z (int): Slice to display. Defaults to middle slice.
        preds (np.ndarray): Predictions to show next to labels.
    """
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
        title = f"label: {int(item['label'].numpy()[0])}"
        if preds is not None:
            title += f"\nmalign. prob.: {preds[i-1]:.4f}"
        plt.title(title)
    plt.show()

def preview_explanations(inputs: torch.Tensor, attributions: torch.Tensor, z: int = None):
    """Displays input image and corresponding attributions.

    Args:
        inputs (torch.Tensor): Input samples, including batch dimension.
        attributions (torch.Tensor): Attribution values, same dimensions as
        inputs.
        z (int): Slice to display
    """
    for attr, img in zip(attributions, inputs):
        arr = F.relu(attr).mean(dim=0).detach()
        arr /= arr.quantile(0.98)
        arr = torch.clamp(arr, 0, 1).numpy()
        img = img.numpy()[0]
        ig_arr = arr*img
        _, (ax_img, ax_exp) = plt.subplots(1, 2, figsize=(7.5, 6))
        ax_img.imshow(img[:,:,z], cmap="gray")
        ax_img.set_title("Input")
        ax_exp.imshow(ig_arr[:,:,z], cmap="copper")
        ax_exp.set_title("Heatmap")
        plt.show()

