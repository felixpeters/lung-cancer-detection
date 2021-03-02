from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from image_reader import LIDCReader


@pytest.fixture(scope="session")
def data_dir():
    data_dir = Path(__file__).absolute().parents[2] / "data/test"
    return data_dir


def test_init(data_dir):
    reader = LIDCReader(data_dir)
    assert len(reader.meta_df) == 1


def test_verify_suffix(data_dir):
    reader = LIDCReader(data_dir)
    assert reader.verify_suffix(
        str(data_dir/"images/LIDC-IDRI-0001.npy")) == True
    assert reader.verify_suffix(
        str(data_dir/"masks/LIDC-IDRI-0001.npy")) == True
    assert reader.verify_suffix("example.txt") == False
