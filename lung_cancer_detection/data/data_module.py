from typing import Optional

import pytorch_lightning as pl

from image_reader import LIDCReader


class LIDCDataModule(pl.LightningDataModule):
    """See examples:
            - https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d_lightning.ipynb
            - https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
            - https://github.com/felixpeters/melanoma-detection/blob/master/src/data/data_module.py
    """

    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        return

    def setup(self, stage: Optional[str] = None):
        return

    def train_dataloader(self):
        return

    def val_dataloader(self):
        return

    def test_dataloader(self):
        return
