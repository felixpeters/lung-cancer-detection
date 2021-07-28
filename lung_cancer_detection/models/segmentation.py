import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from monai.losses import DiceLoss
from typing import Sequence, Union


class NoduleSegmentationModel(pl.LightningModule):

    def __init__(self, model: nn.Module, lr: float = 1e-4):
        """Creates a simple UNet for segmenting nodules in chest CTs. Defines its training and validation logic.

        Args:
            model: (nn.Module): Model to use for training.
            lr (float, optional): Initial learning rate. Defaults to 1e-4.
        """
        super().__init__()
        self.model = model
        self.loss = DiceLoss(to_onehot_y=True, softmax=True)
        self.lr = lr
        self.save_hyperparameters()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss(output, labels)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss(output, labels)
        self.log("val_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = Adam(self.model.parameters(), self.lr)
        return optimizer
