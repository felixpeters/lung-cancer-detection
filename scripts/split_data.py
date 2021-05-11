import argparse
import warnings
from pathlib import Path

import wandb
import yaml
from lung_cancer_detection.data.preprocessing import split_lidc
from lung_cancer_detection.utils import save_json

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits raw LIDC-IDRI dataset into training and validation data")
    parser.add_argument(
        '--config',
        type=lambda p: Path(p).absolute(),
        default=Path(__file__).absolute().parents[1] / "configs/baseline.yaml",
        help="Path to configuration file")
    parser.add_argument(
        '--no-version',
        dest="version",
        action="store_false",
        help="Skip versioning data artifacts in Weights & Biases",
    )
    parser.set_defaults(version=True)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    data_dir = Path(config["data"]["data_dir"]).absolute()
    split_dir = Path(config["data"]["split_dir"]).absolute()
    split_dir.mkdir(parents=True, exist_ok=True)
    print("CONFIGURATION:")
    print(f"Data directory: {data_dir}")
    print(f"Split directory: {split_dir}")

    if args.version:
        run = wandb.init(project=config["wandb"]["project"],
                         job_type="split", tags=[
            "nodule-segmentation",
            "nodule-classification",
        ])
        run.use_artifact(config["artifacts"]["data"]["name"] +
                         ":" + config["artifacts"]["data"]["version"])

    print("Splitting raw LIDC-IDRI data into training and validation sets...")
    train, valid = split_lidc(
        data_dir / "meta", val_split=config["data"]["val_split"], seed=config["random_seed"])
    print("Saving training and validation in JSON format...")
    train_file = split_dir / "train.json"
    save_json(train_file, train)
    val_file = split_dir / "valid.json"
    save_json(val_file, valid)

    if args.version:
        train_artifact = wandb.Artifact(config["artifacts"]["train"]["name"],
                                        type=config["artifacts"]["train"]["type"],
                                        description=config["artifacts"]["train"]["description"])
        val_artifact = wandb.Artifact(config["artifacts"]["valid"]["name"],
                                      type=config["artifacts"]["valid"]["type"],
                                      description=config["artifacts"]["valid"]["description"])
        train_artifact.add_file(str(train_file))
        val_artifact.add_file(str(val_file))
        run.log_artifact(train_artifact)
        run.log_artifact(val_artifact)
