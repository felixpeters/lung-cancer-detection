from typing import Optional
from pathlib import Path
import os

import pytorch_lightning as pl
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    Spacingd,
    ToTensord,
    CenterSpatialCropd,
)
from monai.data import Dataset, PersistentDataset
from monai.utils import set_determinism
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .image_reader import LIDCReader


class LIDCDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: Path, cache_dir: Path, batch_size: int, val_split: float = 0.2, seed: int = 47):
        """Module that deals with preparation of the LIDC dataset for training segmentation models.

        Args:
            data_dir (Path): Folder where preprocessed data is stored. See `LIDCReader` docs for expected structure.
            cache_dir (Path): Folder where deterministic data transformations should be cached.
            batch_size (int): Number of training examples in each batch.
            val_split (float, optional): Percentage of examples to set aside for validation. Defaults to 0.2.
            seed (int, optional): Random seed used for deterministic sampling and transformations. Defaults to 47.
        """
        super().__init__()
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        reader = LIDCReader(data_dir)
        self.train_transforms = Compose([
            LoadImaged(keys=["image", "label"], reader=reader),
            AddChanneld(keys=["image", "label"]),
            # TODO: Test different spacing configurations
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            # TODO: Test different scaling methods
            ScaleIntensityd(keys=["image"]),
            # TODO: Test different cropping methods
            CenterSpatialCropd(keys=["image", "label"],
                               roi_size=[180, 180, 90]),
            # TODO: Test data augmentation methods
            ToTensord(keys=["image", "label"]),
        ])
        self.val_transforms = Compose([
            LoadImaged(keys=["image", "label"], reader=reader),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityd(keys=["image"]),
            CenterSpatialCropd(keys=["image", "label"],
                               roi_size=[180, 180, 90]),
            ToTensord(keys=["image", "label"]),
        ])
        return

    def prepare_data(self):
        """Not needed as of current version.
        """
        return

    def setup(self, stage: Optional[str] = None):
        """Set up persistent datasets for training and validation.

        Args:
            stage (Optional[str], optional): Stage in the model lifecycle, e.g., `fit` or `test`. Only needed datasets will be created. Defaults to None.
        """
        self.scans = pd.read_csv(
            self.data_dir/"meta/scans.csv", index_col="PatientID")
        set_determinism(seed=self.seed)

        if stage == "fit" or stage is None:
            train_idx, val_idx = train_test_split(
                list(self.scans.index), test_size=self.val_split, random_state=self.seed, shuffle=True)
            train_dicts = [
                {"image": f"images/{idx}.npy", "label": f"masks/{idx}.npy"} for idx in train_idx
            ]
            val_dicts = [
                {"image": f"images/{idx}.npy", "label": f"masks/{idx}.npy"} for idx in val_idx
            ]
            self.train_ds = PersistentDataset(
                train_dicts, transform=self.train_transforms, cache_dir=self.cache_dir)
            self.val_ds = PersistentDataset(
                val_dicts, transform=self.val_transforms, cache_dir=self.cache_dir)
        return

    def train_dataloader(self) -> DataLoader:
        """Create data loader for model training.

        Returns:
            DataLoader: Data loader for model training
        """
        train_loader = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())
        return train_loader

    def val_dataloader(self) -> DataLoader:
        """Create data loader for model validation.

        Returns:
            DataLoader: Data loader for model validation
        """
        val_loader = DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=os.cpu_count())
        return val_loader

    def test_dataloader(self):
        """Not needed in the current library version.
        """
        return
