from pathlib import Path

import pytest

from data_module import LIDCDataModule


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
    assert len(data_module.train_ds) == 4
    assert len(data_module.val_ds) == 1
