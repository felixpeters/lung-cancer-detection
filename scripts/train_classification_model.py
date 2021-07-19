#!/usr/bin/env python
import argparse
import warnings
from pathlib import Path

import numpy as np
import wandb
import torch
from monai.networks.nets import DenseNet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lung_cancer_detection.data.nodule import ClassificationDataModule
from lung_cancer_detection.models.classification import NoduleClassificationModule
from lung_cancer_detection.utils import load_config, load_json, preview_dataset

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Trains nodule classification model")
    parser.add_argument(
        '--config',
        type=lambda p: Path(p).absolute(),
        default=Path(__file__).absolute().parents[1] / "configs/cloud.yaml",
        help="Path to configuration file")
    parser.set_defaults(version=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print(f"Run configuration: {config}")
    
    splits = [
        load_json(Path(config["data"]["split_dir"])/"train.json"), 
        load_json(Path(config["data"]["split_dir"])/"valid.json")
    ]
    label_mapping = ([1,2,3,4,5], [0,0,0,1,1])
    
    dm = ClassificationDataModule(
        data_dir=Path(config["data"]["data_dir"]),
        cache_dir=Path(config["data"]["cache_dir"]),
        splits=splits,
        min_anns=config["data"]["min_anns"],
        exclude_labels=[],
        label_mapping=label_mapping,
        aug_prob=config["data"]["aug_prob"],
        batch_size=config["data"]["batch_size"]
    )
    
    net = DenseNet(
        spatial_dims=config["class_model"]["spatial_dims"],
        in_channels=config["class_model"]["in_channels"],
        out_channels=config["class_model"]["out_channels"],
        dropout_prob=config["class_model"]["dropout"],
    )
    model = NoduleClassificationModule(
        net, 
        num_classes=config["class_model"]["num_classes"], 
        lr=config["class_model"]["lr"]
    )
    
    wandb.login()
    
    logger = WandbLogger(project=config["wandb"]["project"], job_type="training")
    
    es = EarlyStopping(monitor="val_loss", verbose=True)
    mc = ModelCheckpoint(
        monitor="val_loss", 
        filename="{epoch}-{step}-{val_loss:.4f}-{val_acc:.4f}", 
        verbose=True, 
        save_top_k=1
    )
    callbacks = [es, mc]
    
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        **config["experiment"]
    )
    
    trainer.logger.experiment.use_artifact(config["artifacts"]["train"]["name"] + ":" + config["artifacts"]["train"]["version"])
    trainer.logger.experiment.use_artifact(config["artifacts"]["valid"]["name"] + ":" + config["artifacts"]["valid"]["version"])
    
    trainer.fit(model, datamodule=dm)
    
    model_artifact = wandb.Artifact(
        config["artifacts"]["class_model"]["name"],
        type=config["artifacts"]["class_model"]["type"],
        description=config["artifacts"]["class_model"]["description"],
    )
    model_artifact.add_file(mc.best_model_path)
    trainer.logger.experiment.log_artifact(model_artifact)
    
    wandb.finish()
