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

    src_dir = Path(config["data"]["raw_dir"]).absolute()
    dest_dir = Path(config["data"]["data_dir"]).absolute()
    zip_dir = Path(config["data"]["zip_dir"]).absolute()
    sample = config["data"]["sample"]
    print("CONFIGURATION:")
    print(f"Source directory: {src_dir}")
    print(f"Destination directory: {dest_dir}")
    print(f"Archive directory: {zip_dir}")
    print(f"Sample: {sample}")

    print("Start preprocessing of raw DICOM files...")
    preprocess_lidc(src_dir, dest_dir, sample=sample)

    print("Start compressing processed data files...")
    shutil.make_archive(zip_dir/"processed", "zip", dest_dir)

    if args.version:
        run = wandb.init(project=config["wandb"]["project"],
                         job_type="preprocess", tags=["nodule-segmentation",
                                                      "nodule-classification"])
        artifact = wandb.Artifact(config["artifacts"]["data"]["name"],
                                  type=config["artifacts"]["data"]["type"],
                                  description=config["artifacts"]["data"]["description"])
        artifact.add_reference("file://" + str(zip_dir))
        run.log_artifact(artifact)
