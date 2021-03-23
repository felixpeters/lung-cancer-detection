from typing import Optional
from pathlib import Path

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

from image_reader import LIDCReader


class LIDCDataModule(pl.LightningDataModule):
    """See examples:
            - https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d_lightning.ipynb
            - https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
            - https://github.com/felixpeters/melanoma-detection/blob/master/src/data/data_module.py
    """

    def __init__(self, data_dir: Path, cache_dir: Path, batch_size: int, val_split: float = 0.2, sample=None, seed: int = 47):
        super().__init__()
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.sample = sample
        self.seed = seed
        reader = LIDCReader(data_dir)
        self.train_transform = Compose([
            LoadImaged(keys=["image", "label"], reader=reader),
            AddChanneld(keys=["image", "label"]),
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

    def prepare_data(self):
        return

    def setup(self, stage: Optional[str] = None):
        self.scans = pd.read_csv(
            self.data_dir/"meta/scans.csv", index_col="PatientID")

        return

    def train_dataloader(self):
        return

    def val_dataloader(self):
        return

    def test_dataloader(self):
        return
