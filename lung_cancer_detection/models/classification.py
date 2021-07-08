from typing import Dict

import pytorch_lightning as pl
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.nn import Softmax
import torch.nn.functional as F
import torchmetrics


class NoduleClassificationModule(pl.LightningModule):

    def __init__(self, model: nn.Module, num_classes: int = 2, lr: float = 1e-4, **kwargs):
        super().__init__()
        self.model = model
        self.lr = lr
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.train_auc = torchmetrics.AUROC(num_classes=num_classes)
        self.val_auc = torchmetrics.AUROC(num_classes=num_classes)
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return F.softmax(output, dim=1)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        x, y = batch["image"], batch["label"]
        y = y.squeeze().type(torch.LongTensor)
        output = self(x)
        loss = F.cross_entropy(output, y)
        logits = F.softmax(output, dim=1)
        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc)
        self.train_auc(logits, y)
        self.log("train_auc", self.train_auc)
        return {"loss": loss}

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        x, y = batch["image"], batch["label"]
        y = y.squeeze().type(torch.LongTensor)
        output = self(x)
        logits = F.softmax(output, dim=1)
        loss = F.cross_entropy(output, y)
        self.log("val_loss", loss)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc)
        self.val_auc(logits, y)
        self.log("val_auc", self.val_auc)
        return {"val_loss": loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = Adam(self.model.parameters(), self.lr)
        return optimizer
        
