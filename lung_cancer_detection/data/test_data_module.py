from pathlib import Path

import numpy as np
import pytest

from .data_module import SegmentationDataModule


@pytest.fixture(scope="session")
def data_dir():
    data_dir = Path(__file__).absolute().parents[2] / "data/test"
    return data_dir


@pytest.fixture(scope="session")
def data_module(data_dir):
    data_module = SegmentationDataModule(
        data_dir, data_dir/"cache", batch_size=2)
    data_module.setup()
    return data_module


@pytest.fixture(scope="session")
def train_loader(data_module):
    loader = data_module.train_dataloader()
    return loader


@pytest.fixture(scope="session")
def val_loader(data_module):
    loader = data_module.val_dataloader()
    return loader


def test_init(data_module):
    assert data_module.batch_size == 2
    assert data_module.data_dir.exists() == True


def test_setup(data_module):
    assert len(data_module.train_ds) == 8
    assert len(data_module.val_ds) == 2
    item = data_module.train_ds[0]
    img = item["image"].numpy()
    label = item["label"].numpy()
    affine = item["image_meta_dict"]["affine"]
    assert list(img.shape) == [1, 180, 180, 90]
    assert list(label.shape) == [1, 180, 180, 90]
    assert np.all(img >= 0.0) == True
    assert np.all(img <= 1.0) == True
    assert affine[0, 0] == 1.5
    assert affine[1, 1] == 1.5
    assert affine[2, 2] == 2.0


def test_train_dataloader(train_loader):
    assert len(train_loader) == 4
    for batch in train_loader:
        assert batch["image"].numpy().shape == (2, 1, 180, 180, 90)
        assert batch["label"].numpy().shape == (2, 1, 180, 180, 90)


def test_val_dataloader(val_loader):
    assert len(val_loader) == 1
    for batch in val_loader:
        assert batch["image"].numpy().shape == (2, 1, 180, 180, 90)
        assert batch["label"].numpy().shape == (2, 1, 180, 180, 90)
