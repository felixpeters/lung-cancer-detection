from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers.base import LightningLoggerBase


class Experiment:

    def __init__(self, model: LightningModule, data: LightningDataModule, logger: LightningLoggerBase, random_seed: int = 47, **kwargs):
        self.model = model
        self.data = data
        self.logger = logger
        seed_everything(random_seed)
        self.logger.log_hyperparams(
            self.model.hparams.update(self.data.hparams))
        self.trainer = Trainer(logger=self.logger, **kwargs)

    def find_lr(self):
        self.trainer.tune(self.model, datamodule=self.data)
        return

    def run(self):
        self.trainer.fit(self.model, self.data)
        return
