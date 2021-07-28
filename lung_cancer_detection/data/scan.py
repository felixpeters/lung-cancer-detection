import os
from pathlib import Path
from typing import Optional, Sequence, Dict

import pandas as pd
import pytorch_lightning as pl
from monai.data import Dataset, PersistentDataset, list_data_collate
from monai.transforms import (AddChanneld, CenterSpatialCropd, Compose,
                              LoadImaged, RandCropByPosNegLabeld,
                              ScaleIntensityd, SelectItemsd, Spacingd,
                              SpatialPadd, ToTensord)
from monai.utils import set_determinism
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .reader import LIDCReader


class SegmentationDataModule(pl.LightningDataModule):

    def __init__(self, 
            data_dir: Path, 
            cache_dir: Path, 
            splits: Sequence[Sequence[Dict]],
            batch_size: int,
            spacing: Sequence[float] = (1.5, 1.5, 2.0),
            crop_size: Sequence[int] = [48, 48, 36], 
            roi_size: Sequence[int] = [192, 192, 144], 
            seed: int = 47, **kwargs):
        """Module that deals with preparation of the LIDC dataset for training segmentation models.

        Args:
            data_dir (Path): Folder where preprocessed data is stored. See `LIDCReader` docs for expected structure.
            cache_dir (Path): Folder where deterministic data transformations should be cached.
            splits (Sequence[Sequence[Dict]]): Data dictionaries for training
            and validation split.
            batch_size (int): Number of training examples in each batch.
            spacing (Sequence[float]): Pixel and slice spacing. Defaults to 1.5x1.5x2mm.
            crop_size (Sequence[int]): Size of crop that is used for training. Defaults to 48x48x36px.
            roi_size (Sequence[int]): Size of crop that is used for validation. Defaults to 192x192x144px.
            seed (int, optional): Random seed used for deterministic sampling and transformations. Defaults to 47.
        """
        super().__init__()
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.splits = splits
        self.batch_size = batch_size
        self.val_split = val_split
        self.spacing = spacing
        self.crop_size = crop_size
        self.roi_size = roi_size
        self.seed = seed
        reader = LIDCReader(data_dir)
        self.train_transforms = Compose([
            LoadImaged(keys=["image", "label"], reader=reader),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=self.spacing,
                     mode=("bilinear", "nearest")),
            ScaleIntensityd(keys=["image"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=self.crop_size,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
                ),
            ToTensord(keys=["image", "label"]),
            SelectItemsd(keys=["image", "label"]),
        ])
        self.val_transforms = Compose([
            LoadImaged(keys=["image", "label"], reader=reader),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=self.spacing,
                     mode=("bilinear", "nearest")),
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(keys=["image", "label"], spatial_size=self.roi_size,
                        mode="constant"),
            CenterSpatialCropd(keys=["image", "label"], roi_size=self.roi_size),
            ToTensord(keys=["image", "label"]),
            SelectItemsd(keys=["image", "label"]),
        ])
        self.hparams = {
            "batch_size": self.batch_size,
            "val_split": self.val_split,
            "spacing": self.spacing,
            "crop_size": self.crop_size,
            "roi_size": self.roi_size,
        }
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
            train_scans, val_scans = self.splits
            self.train_dicts = [{"image": scan["image"], "label": scan["mask"]}
                for scan in train_scans]
            self.val_dicts = [{"image": scan["image"], "label": scan["mask"]}
                for scan in val_scans]
            self.train_ds = PersistentDataset(
                self.train_dicts, transform=self.train_transforms, cache_dir=self.cache_dir)
            self.val_ds = PersistentDataset(
                self.val_dicts, transform=self.val_transforms, cache_dir=self.cache_dir)
        return

    def train_dataloader(self) -> DataLoader:
        """Create data loader for model training.

        Returns:
            DataLoader: Data loader for model training
        """
        train_loader = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, 
            num_workers=os.cpu_count(), collate_fn=list_data_collate)
        return train_loader

    def val_dataloader(self) -> DataLoader:
        """Create data loader for model validation.

        Returns:
            DataLoader: Data loader for model validation
        """
        val_loader = DataLoader(
            self.val_ds, batch_size=self.batch_size,
            num_workers=os.cpu_count(), collate_fn=list_data_collate)
        return val_loader

    def test_dataloader(self):
        """Not needed in the current library version.
        """
        return
