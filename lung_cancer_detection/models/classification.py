from typing import Dict

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.nn import Softmax
from monai.networks.nets import DenseNet    
import torch.nn.functional as F
import torchmetrics


class NoduleClassificationDenseNet(pl.LightningModule):

    def __init__(self, spatial_dims: int = 3, in_channels: int = 1,
            out_channels: int = 2, lr: float = 1e-4, **kwargs):
        super().__init__()
        self.model = DenseNet(spatial_dims=spatial_dims,
                in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.lr = lr
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.train_auc = torchmetrics.AUROC()
        self.val_auc = torchmetrics.AUROC()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        x, y = batch["image"], batch["label"]
        y = y.squeeze().type(torch.LongTensor)
        output = self(x)
        loss = F.cross_entropy(output, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        x, y = batch["image"], batch["label"]
        y = y.squeeze().type(torch.LongTensor)
        output = self(x)
        loss = F.cross_entropy(output, y)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = Adam(self.model.parameters(), self.lr)
        return optimizer
        
