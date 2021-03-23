from pathlib import Path

import pytest
import numpy as np

from .data_module import LIDCDataModule


@pytest.fixture(scope="session")
def data_dir():
    data_dir = Path(__file__).absolute().parents[2] / "data/test"
    return data_dir


@pytest.fixture(scope="session")
def data_module(data_dir):
    data_module = LIDCDataModule(data_dir, data_dir/"cache", batch_size=2)
    return data_module


def test_init(data_module):
    assert data_module.batch_size == 2
    assert data_module.data_dir.exists() == True


def test_setup(data_module):
    data_module.setup()
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


def test_train_dataloader(data_module):
    data_module.setup()
    loader = data_module.train_dataloader()
    assert len(loader) == 4
    for batch in loader:
        assert batch["image"].numpy().shape == (2, 1, 180, 180, 90)
        assert batch["label"].numpy().shape == (2, 1, 180, 180, 90)


def test_val_dataloader(data_module):
    data_module.setup()
    loader = data_module.val_dataloader()
    assert len(loader) == 1
    for batch in loader:
        assert batch["image"].numpy().shape == (2, 1, 180, 180, 90)
        assert batch["label"].numpy().shape == (2, 1, 180, 180, 90)
