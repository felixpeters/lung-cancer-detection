from pathlib import Path

import pytest

from ..data.scan import SegmentationDataModule
from ..data.test_scan import data_dir, data_module, train_loader, val_loader
from .segmentation import NoduleSegmentationUNet


@pytest.fixture(scope="session")
def train_batch(train_loader):
    batch = next(iter(train_loader))
    return batch


@pytest.fixture(scope="session")
def val_batch(val_loader):
    batch = next(iter(val_loader))
    return batch


@pytest.fixture(scope="session")
def seg_model():
    model = NoduleSegmentationUNet(features=(4, 4, 8, 16, 32, 4))
    return model


def test_init(seg_model):
    assert seg_model.lr == 1e-4


def test_forward(seg_model, train_batch):
    x, y = train_batch["image"], train_batch["label"]
    output = seg_model.forward(x)
    assert output.detach().numpy().shape == (2, 2, 180, 180, 90)


def test_training_step(seg_model, train_batch):
    loss = seg_model.training_step(train_batch, 0)
    assert loss.item() > 0.0


def test_validation_step(seg_model, val_batch):
    loss = seg_model.validation_step(val_batch, 0)
    assert loss.item() > 0.0
