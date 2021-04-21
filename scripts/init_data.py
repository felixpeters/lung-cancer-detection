import wandb
import yaml
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Versions raw dataset in Weights & Biases")
    parser.add_argument(
        '--config',
        type=lambda p: Path(p).absolute(),
        default=Path(__file__).absolute().parents[1] / "configs/test.yaml",
        help="Path to configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    data_dir = Path(config["data"]["data_dir"]).absolute()
    print("CONFIGURATION:")
    print(f"Source directory: {data_dir}")
    ref_link = "file://" + str(data_dir)
    print(f"Reference link: {ref_link}")

    run = wandb.init(project=config["wandb"]["project"],
                     job_type="data", tags=config["wandb"]["tags"])
    artifact = wandb.Artifact("lidc-idri-cts", type="dataset",
                              description="Sample of ten chest CTs from LIDC-IDRI dataset, converted to npy files, including metadata")
    artifact.add_reference(ref_link)
    run.log_artifact(artifact)
