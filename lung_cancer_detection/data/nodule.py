import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import pytorch_lightning as pl
from monai.data import Dataset, PersistentDataset, list_data_collate
from monai.transforms import (AddChanneld, CenterSpatialCropd, Compose,
                              LoadImaged, ScaleIntensityd, Spacingd,
                              SpatialPadd, ToTensord)
from monai.utils import set_determinism
from torch.utils.data import DataLoader

from .image_reader import LIDCReader


class ClassificationDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: Path,
                 cache_dir: Path,
                 splits: Sequence[Dict],
                 batch_size: int = 16,
                 spacing: Sequence[float] = (1.5, 1.5, 2.0),
                 roi_size: Sequence[int] = [40, 40, 30],
                 seed: int = 47):
        """Handles all things data related for classifying lung nodules from the LIDC-IDRI dataset. Adheres to the PyTorch Lightning DataModule interface.

        Args:
            data_dir (Path): Directory with preprocessed LIDC dataset, as outputted by `preprocess_data` script.
            cache_dir (Path): Directory where deterministic transformations of input samples will be cached.
            splits (Sequence[Dict]): Dictionaries containing metadata of training and validation sets. See `split_data` script for more information.
            batch_size (int, optional): Batch size for training and validation. Defaults to 16.
            spacing (Sequence[float], optional): Pixel spacing (in mm) that inputs will be transformed into. Defaults to (1.5, 1.5, 2.0).
            roi_size (Sequence[int], optional): Shape that inputs will be transformed into. Defaults to [40, 40, 30].
            seed (int, optional): Random seed for transformations etc. Defaults to 47.
        """
        super().__init__()
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.splits = splits
        self.batch_size = batch_size
        self.spacing = spacing
        self.roi_size = roi_size
        self.seed = seed
        self.hparams = {
            "batch_size": self.batch_size,
            "spacing": self.spacing,
            "roi_size": self.roi_size,
            "seed": self.seed,
        }
        reader = LIDCReader(self.data_dir, nodule_mode=True)
        self.train_transforms = Compose([
            LoadImaged(keys=["image"], reader=reader),
            AddChanneld(keys=["image"]),
            Spacingd(keys=["image"], pixdim=self.spacing, mode="bilinear"),
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(keys=["image"], spatial_size=self.roi_size,
                        mode="constant"),
            CenterSpatialCropd(keys=["image"], roi_size=self.roi_size),
            ToTensord(keys=["image", "label"]),
        ])
        self.val_transforms = Compose([
            LoadImaged(keys=["image"], reader=reader),
            AddChanneld(keys=["image"]),
            Spacingd(keys=["image"], pixdim=self.spacing, mode="bilinear"),
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(keys=["image"], spatial_size=self.roi_size,
                        mode="constant"),
            CenterSpatialCropd(keys=["image"], roi_size=self.roi_size),
            ToTensord(keys=["image", "label"]),
        ])
        return

    def prepare_data(self):
        """Not needed in current library version.
        """
        return

    def setup(self, stage: Optional[str] = None):
        """Creates persistent training and validation sets based on provided splits.

        Args:
            stage (Optional[str], optional): Stage (e.g., "fit", "eval") for more efficient setup. Defaults to None.
        """
        set_determinism(seed=self.seed)
        if stage == "fit" or stage is None:
            train_scans, val_scans = self.splits
            self.train_dicts = [
                {"image": nod["image"], "label": nod["malignancy"]} for
                scan in train_scans for nod in scan["nodules"]
            ]
            self.val_dicts = [
                {"image": nod["image"], "label": nod["malignancy"]} for
                scan in val_scans for nod in scan["nodules"]
            ]
            self.train_ds = PersistentDataset(
                self.train_dicts, transform=self.train_transforms, cache_dir=self.cache_dir)
            self.val_ds = PersistentDataset(
                self.val_dicts, transform=self.val_transforms, cache_dir=self.cache_dir)
            return

    def train_dataloader(self) -> DataLoader:
        """Creates training data loader.

        Returns:
            DataLoader: PyTorch data loader
        """
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=os.cpu_count(),
                          collate_fn=list_data_collate)

    def val_dataloader(self) -> DataLoader:
        """Creates validation data loader.

        Returns:
            DataLoader: PyTorch data loader
        """
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=os.cpu_count(),
                          collate_fn=list_data_collate)

    def test_dataloader(self):
        """Not needed in current library version.
        """
