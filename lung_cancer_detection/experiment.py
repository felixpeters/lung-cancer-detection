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
        """Runs an experiment with the given model on the provided dataset.

        Args:
            model (LightningModule): Model to be trained
            data (LightningDataModule): Dataset to be used
            logger (WandbLogger): Weights & Biases logger for experiment tracking
            input_artifact (dict, optional): Configuration of W&B data artifact. Defaults to None.
            callbacks (Sequence[Callback], optional): Callbacks for model training. Defaults to [].
            random_seed (int, optional): Random seed for reproducibility. Defaults to 47.
        """
        self.model = model
        self.data = data
        self.logger = logger
        seed_everything(random_seed)
        self.model.hparams.update(self.data.hparams)
        self.logger.log_hyperparams(self.model.hparams)
        self.trainer = Trainer(
            logger=self.logger, callbacks=callbacks, **kwargs)
        if input_artifact:
            self.logger.experiment.use_artifact(input_artifact["name"] + ":" +
                                                input_artifact["version"], type=input_artifact["type"])

    def find_params(self):
        """Finds the optimal learning rate and batch size using the PyTorch Lightning tuner.
        """
        self.trainer.tune(self.model, datamodule=self.data)
        self.model.hparams.lr = self.model.lr
        self.logger.log_hyperparams(self.model.hparams)

    def run(self):
        """Runs the entire training process.
        """
        self.trainer.fit(self.model, self.data)

    def finish(self, output_artifact: dict):
        """[summary]

        Args:
            output_artifact (dict): Configuration of W&B model artifact
        """
        model_artifact = wandb.Artifact(output_artifact["name"],
                                        type=output_artifact["type"],
                                        description=output_artifact["description"])
        model_artifact.add_file(output_artifact["path"])
        self.logger.experiment.log_artifact(model_artifact)
