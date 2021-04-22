import argparse
import shutil
import warnings
from pathlib import Path

import wandb
import yaml
from lung_cancer_detection.data.preprocessing import preprocess_lidc

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts volumes, masks and metadata from LIDC-IDRI dataset")
    parser.add_argument(
        '--config',
        type=lambda p: Path(p).absolute(),
        default=Path(__file__).absolute().parents[1] / "configs/baseline.yaml",
        help="Path to configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    src_dir = Path(config["data"]["raw_dir"]).absolute()
    dest_dir = Path(config["data"]["data_dir"]).absolute()
    zip_dir = Path(config["data"]["zip_dir"]).absolute()
    print("CONFIGURATION:")
    print(f"Source directory: {src_dir}")
    print(f"Destination directory: {dest_dir}")
    print(f"Archive directory: {zip_dir}")

    preprocess_lidc(src_dir, dest_dir,
                    sample_size=config["data"]["sample_size"])

    shutil.make_archive(zip_dir/"segmentation", "zip", dest_dir)

    run = wandb.init(project=config["wandb"]["project"],
                     job_type="data", tags=config["wandb"]["tags"])
    artifact = wandb.Artifact(config["wandb"]["seg_data_artifact"]["name"], type="dataset",
                              description=config["wandb"]["seg_data_artifact"]["description"])
    artifact.add_reference("file://" + str(zip_dir))
    run.log_artifact(artifact)
