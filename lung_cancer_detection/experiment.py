from typing import Sequence

import wandb
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers import WandbLogger


class Experiment:

    def __init__(self, model: LightningModule, data: LightningDataModule,
                 logger: WandbLogger, input_artifact: dict = None,
                 callbacks: Sequence[Callback] = [], random_seed: int = 47, **kwargs):
        self.model = model
        self.data = data
        self.logger = logger
        seed_everything(random_seed)
        self.model.hparams.update(self.data.hparams)
        self.logger.log_hyperparams(self.model.hparams)
        self.trainer = Trainer(
            logger=self.logger, callbacks=callbacks, **kwargs)
        self.logger.experiment.use_artifact(input_artifact["name"] + ":" +
                                            input_artifact["version"], type=input_artifact["type"])

    def find_lr(self):
        self.trainer.tune(self.model, datamodule=self.data)
        self.model.hparams.lr = self.model.lr
        self.logger.log_hyperparams(self.model.hparams)

    def run(self):
        self.trainer.fit(self.model, self.data)

    def finish(self, output_artifact):
        model_artifact = wandb.Artifact(output_artifact["name"],
                                        type=output_artifact["type"],
                                        description=output_artifact["description"])
        model_artifact.add_file(output_artifact["path"])
        self.logger.experiment.log_artifact(model_artifact)
