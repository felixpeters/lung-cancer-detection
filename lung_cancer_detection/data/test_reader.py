from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .reader import LIDCReader


@pytest.fixture(scope="session")
def data_dir():
    data_dir = Path(__file__).absolute().parents[2] / "data/test"
    return data_dir


@pytest.fixture(scope="session")
def reader(data_dir):
    reader = LIDCReader(data_dir)
    return reader


def test_init(data_dir):
    reader = LIDCReader(data_dir)
    assert len(reader.meta_df) == 10


def test_verify_suffix(reader):
    assert reader.verify_suffix("images/LIDC-IDRI-0001.npy") == True
    assert reader.verify_suffix("masks/LIDC-IDRI-0001.npy") == True
    assert reader.verify_suffix("example.txt") == False


def test_read(reader):
    img, meta = reader.read("images/LIDC-IDRI-0001.npy")
    assert img.shape == (512, 512, 133)
    assert meta.name == "LIDC-IDRI-0001"
    with pytest.raises(ValueError):
        reader.read(["images/LIDC-IDRI-0001.npy", "images/LIDC-IDRI-0002.npy"])
    with pytest.raises(ValueError):
        reader.read("images/LIDC-IDRI-0001.png")
    with pytest.raises(FileNotFoundError):
        reader.read("images/example.npy")


def test_get_data(reader):
    raw_data = reader.read("images/LIDC-IDRI-0001.npy")
    img, meta = reader.get_data(raw_data)
    assert img.shape == (512, 512, 133)
    assert "affine" in meta
    assert "original_affine" in meta
    assert "spatial_shape" in meta
    assert meta["affine"].shape == (4, 4)
    assert np.equal(meta["affine"], meta["original_affine"]).all()
    assert np.equal(meta["spatial_shape"], np.asarray(img.shape)).all()


def test_nodule_mode(data_dir):
    reader = LIDCReader(data_dir, nodule_mode=True)
    assert reader.nodule_mode == True
    raw_data = reader.read("nodules/LIDC-IDRI-0001_0.npy")
    img, meta = reader.get_data(raw_data)
    assert img.shape == (100, 100, 60)
    assert "affine" in meta
    assert "original_affine" in meta
    assert "spatial_shape" in meta
    assert meta["affine"].shape == (4, 4)
    assert np.equal(meta["affine"], meta["original_affine"]).all()
    assert np.equal(meta["spatial_shape"], np.asarray(img.shape)).all()
