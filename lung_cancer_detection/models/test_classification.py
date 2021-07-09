import pytest

import numpy as np
from monai.networks.nets import DenseNet    

from ..data.nodule import ClassificationDataModule
from ..data.test_scan import data_dir
from ..data.test_nodule import class_tl, class_vl, class_dm, splits
from .classification import NoduleClassificationModule

@pytest.fixture(scope="session")
def class_train_batch(class_tl):
    batch = next(iter(class_tl))
    return batch

@pytest.fixture(scope="session")
def class_val_batch(class_vl):
    batch = next(iter(class_vl))
    return batch

@pytest.fixture(scope="session")
def class_model():
    net = DenseNet(spatial_dims=3, in_channels=1, out_channels=2, init_features=4, growth_rate=2,
            block_config=(2, 2, 2, 2), bn_size=2)
    model = NoduleClassificationModule(net, num_classes=2)
    return model

def test_should_produce_output(class_model, class_train_batch):
    x, _ = class_train_batch["image"], class_train_batch["label"]
    output = class_model(x).detach().numpy()
    assert output.shape == (4, 2)

def test_should_output_training_loss(class_model, class_train_batch):
    loss = class_model.training_step(class_train_batch, 0)
    assert loss["loss"].item() >= 0.0

def test_should_output_validation_loss(class_model, class_train_batch):
    loss = class_model.validation_step(class_train_batch, 0)
    assert loss["val_loss"].item() >= 0.0
