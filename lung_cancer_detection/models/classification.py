from typing import Dict

import pytorch_lightning as pl
import torch


class NoduleClassificationDenseNet(pl.LightningModule):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        pass

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        pass
