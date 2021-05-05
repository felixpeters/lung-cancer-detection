import json
from pathlib import Path

import numpy as np
import pytest

from .nodule import ClassificationDataModule
from .test_scan import data_dir


@pytest.fixture(scope="session")
def splits(data_dir):
    with open(data_dir/"splits/train.json") as fp:
        train_data = json.load(fp)
    with open(data_dir/"splits/valid.json") as fp:
        valid_data = json.load(fp)
    return (train_data, valid_data)


@pytest.fixture(scope="session")
def class_dm(data_dir, splits):
    dm = ClassificationDataModule(
        data_dir, data_dir/"cache", splits, roi_size=[50, 50, 30], batch_size=4)
    dm.setup()
    return dm


@pytest.fixture(scope="session")
def class_tl(class_dm):
    loader = class_dm.train_dataloader()
    return loader


@pytest.fixture(scope="session")
def class_vl(class_dm):
    loader = class_dm.val_dataloader()
    return loader


def test_init(class_dm):
    assert class_dm.batch_size == 4
    assert class_dm.data_dir.exists() == True


def test_setup(class_dm):
    assert len(class_dm.train_ds) == 20
    assert len(class_dm.val_ds) == 5
    item = class_dm.train_ds[0]
    img = item["image"].numpy()
    label = item["label"].numpy()
    affine = item["image_meta_dict"]["affine"]
    assert len(img.shape) == 4
    #assert list(img.shape) == [1, 50, 50, 30]
    assert label.shape == (1,)
    assert np.all(img >= 0.0) == True
    assert np.all(img <= 1.0) == True
    assert affine[0, 0] == 1.5
    assert affine[1, 1] == 1.5
    assert affine[2, 2] == 2.0


def test_train_dataloader(class_tl):
    assert len(class_tl) == 5


def test_val_dataloader(class_vl):
    assert len(class_vl) == 2
