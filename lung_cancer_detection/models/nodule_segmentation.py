import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss


class NoduleSegmentationModel(pl.LightningModule):

    def __init__(self, model: nn.Module = BasicUNet(), loss: _Loss = DiceLoss(to_onehot_y=True, softmax=True), lr: float = 1e-4):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        return

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss(output, labels)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss(output, labels)
        self.log("val_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), self.lr)
        return optimizer
