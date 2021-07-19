import os
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
from monai.data import Dataset, PersistentDataset, list_data_collate
from monai.transforms import (AddChanneld, CenterSpatialCropd, Compose,
                              LoadImaged, MapLabelValued, RandAffined, ScaleIntensityd,
                              SelectItemsd, Spacingd, SpatialPadd, ToTensord)
from monai.utils import set_determinism
from torch.utils.data import DataLoader

from .reader import LIDCReader


class ClassificationDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: Path,
                 cache_dir: Path,
                 splits: Sequence[Sequence[Dict]],
                 target: str = "malignancy",
                 min_anns: int = 3,
                 exclude_labels: Sequence[int] = [3],
                 label_mapping: Tuple[Sequence[int]] = (
                     [1, 2, 4, 5], [0, 0, 1, 1]),
                 batch_size: int = 16,
                 spacing: Sequence[float] = (1.5, 1.5, 2.0),
                 roi_size: Sequence[int] = [40, 40, 30],
                 aug_prob: float = 0.0,
                 seed: int = 47):
        """Handles all things data related for classifying lung nodules from the LIDC-IDRI dataset. Adheres to the PyTorch Lightning DataModule interface.

        Args:
            data_dir (Path): Directory with preprocessed LIDC dataset, as outputted by `preprocess_data` script.
            cache_dir (Path): Directory where deterministic transformations of input samples will be cached.
            splits (Sequence[Dict]): Dictionaries containing metadata of training and validation sets. See `split_data` script for more information.
            target (str): Target variable, as denoted in splits dictionary. Defaults to malignancy.
            min_anns (int): Minimum number of annotations required for including nodule. Defaults to 0.
            exclude_labels (Sequence[int]): Label values to exclude in dataset.
            label_mapping (Tuple[Sequence[int]]): Label mapping for discretization.
            batch_size (int, optional): Batch size for training and validation. Defaults to 16.
            spacing (Sequence[float], optional): Pixel spacing (in mm) that inputs will be transformed into. Defaults to (1.5, 1.5, 2.0).
            roi_size (Sequence[int], optional): Shape that inputs will be transformed into. Defaults to [40, 40, 30].
            aug_prob (float): Probability of applying random data augmentation. Defaults to 0.0.
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
        self.target = target
        self.min_anns = min_anns
        self.exclude_labels = exclude_labels
        self.label_mapping = label_mapping
        self.aug_prob = aug_prob
        self.hparams = {
            "batch_size": self.batch_size,
            "spacing": self.spacing,
            "roi_size": self.roi_size,
            "seed": self.seed,
            "target": self.target,
            "min_anns": self.min_anns,
            "exclude_labels": self.exclude_labels,
            "label_mapping": self.label_mapping,
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
            MapLabelValued(keys=["label"], orig_labels=self.label_mapping[0],
                           target_labels=self.label_mapping[1]),
            RandAffined(
                keys=["image"], 
                spatial_size=self.roi_size,
                prob=self.aug_prob, 
                mode="bilinear", 
                rotate_range=(np.pi/18, np.pi/18, np.pi/4),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
            ),
            ToTensord(keys=["image", "label"]),
            SelectItemsd(keys=["image", "label"]),
        ])
        self.val_transforms = Compose([
            LoadImaged(keys=["image"], reader=reader),
            AddChanneld(keys=["image"]),
            Spacingd(keys=["image"], pixdim=self.spacing, mode="bilinear"),
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(keys=["image"], spatial_size=self.roi_size,
                        mode="constant"),
            CenterSpatialCropd(keys=["image"], roi_size=self.roi_size),
            MapLabelValued(keys=["label"], orig_labels=self.label_mapping[0],
                           target_labels=self.label_mapping[1]),
            ToTensord(keys=["image", "label"]),
            SelectItemsd(keys=["image", "label"]),
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
                {"image": nod["image"], "label": nod[self.target]} for
                scan in train_scans for nod in scan["nodules"] if
                nod["annotations"] >= self.min_anns and nod[self.target] not in
                self.exclude_labels
            ]
            self.val_dicts = [
                {"image": nod["image"], "label": nod[self.target]} for
                scan in val_scans for nod in scan["nodules"] if
                nod["annotations"] >= self.min_anns and nod[self.target] not in
                self.exclude_labels
            ]
            self.train_ds = PersistentDataset(
                self.train_dicts, transform=self.train_transforms,
                cache_dir=self.cache_dir)
            self.val_ds = PersistentDataset(
                self.val_dicts, transform=self.val_transforms,
                cache_dir=self.cache_dir)
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
        return

    def query_by_label(self, split: str = "train", n: int = 20, labels: Sequence[int] =
            None, sort: bool = True) -> Dataset:
        """Returns data sample containing nodules which match the given labels.

        Args:
            split (str): Data split to query. Defaults to training set.
            n (int): Number of samples to return. Defaults to 20.
            labels (Sequence[int]): Only return samples with given labels.
            sort (bool): Whether to sort returned samples by label. Defaults to
            true.

        Returns:
            Dataset: Dataset containing samples. Transformations depend on
            which split was used.
        """
        ds = self.train_ds if split == "train" else self.val_ds
        if labels:
            ds = [item for item in ds if int(item["label"]) in labels]
        if n:
            ds = ds[:n]
        return ds
        
    def query_by_case(self, patient_id: str) -> Dataset:
        """Return nodule volumes for one specific case.

        Args:
            patient_id (str): Patient ID of desired case.

        Returns:
            Dataset: Dataset containing case nodules.
        """
        train_cases, valid_cases = self.splits
        train_pids = [case["pid"] for case in train_cases]
        valid_pids = [case["pid"] for case in valid_cases]
        if patient_id in train_pids:
            data_dict = [
                {"image": nod["image"], "label": nod[self.target]} for
                case in train_cases if case["pid"] == patient_id for nod in case["nodules"]
            ]
        elif patient_id in valid_pids:
            data_dict = [
                {"image": nod["image"], "label": nod[self.target]} for
                case in valid_cases if case["pid"] == patient_id for nod in case["nodules"]
            ]
        else:
            raise ValueError("Case with given ID could not be found.")

        return Dataset(data_dict, transform=self.val_transforms)


