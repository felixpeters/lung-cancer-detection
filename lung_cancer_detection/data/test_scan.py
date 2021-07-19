import json
from pathlib import Path

import numpy as np
import pytest

from .scan import SegmentationDataModule


@pytest.fixture(scope="session")
def data_dir():
    data_dir = Path(__file__).absolute().parents[2] / "data/test"
    return data_dir


@pytest.fixture(scope="session")
def splits(data_dir):
    with open(data_dir/"splits/train.json") as fp:
        train_data = json.load(fp)
    with open(data_dir/"splits/valid.json") as fp:
        valid_data = json.load(fp)
    return (train_data, valid_data)

@pytest.fixture(scope="session")
def data_module(data_dir, splits):
    data_module = SegmentationDataModule(
        data_dir, data_dir/"cache", splits, batch_size=2)
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

def test_train_dataloader(train_loader):
    assert len(train_loader) == 4
    for batch in train_loader:
        assert batch["image"].numpy().shape == (4, 1, 48, 48, 36)
        assert batch["label"].numpy().shape == (4, 1, 48, 48, 36)


def test_val_dataloader(val_loader):
    assert len(val_loader) == 1
    for batch in val_loader:
        assert batch["image"].numpy().shape == (2, 1, 192, 192, 144)
        assert batch["label"].numpy().shape == (2, 1, 192, 192, 144)
