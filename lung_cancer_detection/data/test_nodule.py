from pathlib import Path

import numpy as np
import pytest

from .nodule import ClassificationDataModule
from .test_scan import data_dir, splits


@pytest.fixture(scope="session")
def class_dm(data_dir, splits):
    dm = ClassificationDataModule(
        data_dir, data_dir/"cache", splits, batch_size=4)
    dm.setup()
    return dm


@ pytest.fixture(scope="session")
def class_tl(class_dm):
    loader = class_dm.train_dataloader()
    return loader


@ pytest.fixture(scope="session")
def class_vl(class_dm):
    loader = class_dm.val_dataloader()
    return loader


def test_should_create_training_set(class_dm):
    assert len(class_dm.train_ds) == 9
    item = class_dm.train_ds[0]
    img = item["image"].numpy()
    label = item["label"].numpy()
    assert len(img.shape) == 4
    assert list(img.shape) == [1, 40, 40, 30]
    assert np.all(img >= 0.0) == True
    assert np.all(img <= 1.0) == True
    assert label.shape == (1,)
    assert np.all(label >= 0.0) == True
    assert np.all(label <= 1.0) == True


def test_should_create_validation_set(class_dm):
    assert len(class_dm.val_ds) == 1
    item = class_dm.val_ds[0]
    img = item["image"].numpy()
    label = item["label"].numpy()
    assert len(img.shape) == 4
    assert list(img.shape) == [1, 40, 40, 30]
    assert np.all(img >= 0.0) == True
    assert np.all(img <= 1.0) == True
    assert label.shape == (1,)
    assert np.all(label >= 0.0) == True
    assert np.all(label <= 1.0) == True


def test_should_yield_training_batches(class_tl):
    assert len(class_tl) == 3
    for i, batch in enumerate(class_tl):
        if i < 2:
            assert batch["image"].numpy().shape == (4, 1, 40, 40, 30)
        else:
            assert batch["image"].numpy().shape == (1, 1, 40, 40, 30)


def test_should_yield_validation_batches(class_vl):
    assert len(class_vl) == 1
    for batch in class_vl:
        assert batch["image"].numpy().shape == (1, 1, 40, 40, 30)
